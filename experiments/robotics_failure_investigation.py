"""Thirty fixed-eight discovery attempts; conditional same-state recovery pilot.

This is diagnostic infrastructure, not an adaptive controller. Run from the
repository root with .venv-robotics. Existing six recordings remain untouched.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import gymnasium as gym
import numpy as np
import torch

from robotics_baseline_check import (
    load_policy, pair_contact_observer, physical_state, prime_history,
    replay_prefix, run_episode, write_json,
    states_match,
)
from robotics_feasibility import MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256
from robotics_feasibility import observation_batch
from robotics_trajectory_analysis import RULES, analyze_collection


SEEDS = list(range(30))
ORIGINAL = Path("results/robotics-baseline-check")
OUTPUT = Path("results/robotics-failure-patterns")
RECOVERY_SEEDS = [2000, 2001, 2002]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def original_evidence():
    summary = read_json(ORIGINAL / "summary.json")
    protocol = read_json(ORIGINAL / "protocol.json")
    assert summary["complete"] and protocol["seeds"] == list(range(6))
    assert protocol["weights_sha256"] == WEIGHTS_SHA256
    files = ["summary.json", "protocol.json"] + [row["trace"] for row in summary["episodes"]]
    hashes = {name: digest(ORIGINAL / name) for name in files}
    return summary, protocol, hashes


def experiment_protocol(config, hashes):
    return {
        "purpose": "failure-pattern discovery and conditional recovery opportunity, not method evaluation",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256, "config": config,
        "seeds": SEEDS, "reused_seeds": list(range(6)), "new_seeds": list(range(6, 30)),
        "max_episode_steps": 300, "disturbances": "none", "pattern_rules": RULES,
        "original_directory": ORIGINAL.as_posix(), "original_hashes": hashes,
        "diagnostic_state_not_policy_input": True,
        "recovery_selection": "first three failed seeds with a near-target cached regression and >=16 steps left; first successful seed as control, if eligible",
        "recovery_diffusion_seeds": RECOVERY_SEEDS,
        "recovery_arms": ["continue_cached", "replan_once"],
        "recovery_scope": "remaining episode under original total 300-action budget; fixed eight after initial branch",
        "recovery_limitations": "oracle-selected states; fresh random continuations, not exact original future RNG; discovery only",
        "packages": {name: importlib.metadata.version(name) for name in
                     ["lerobot", "gym-pusht", "pymunk", "torch", "numpy", "diffusers"]},
    }


def collect(output, model_dir, resume):
    original, old_protocol, hashes = original_evidence()
    config = read_json(model_dir / "config.json")
    assert config == old_protocol["config"], "Baseline configuration changed."
    protocol = experiment_protocol(config, hashes)
    if (output / "protocol.json").exists():
        assert resume, "Existing results require --resume; never overwrite completed recordings."
        assert read_json(output / "protocol.json") == protocol, "Protocol or original evidence changed."
        summary = read_json(output / "summary.json")
    else:
        assert not output.exists() or not any(output.iterdir()), "Use a separate empty output directory."
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "protocol.json", protocol)
        summary = {"baseline_complete": False, "episodes": []}
        for episode in original["episodes"]:
            summary["episodes"].append(dict(episode, source_directory=ORIGINAL.as_posix(), reused=True))
        write_json(output / "summary.json", summary)
    recorded = [row["seed"] for row in summary["episodes"]]
    assert recorded == SEEDS[:len(recorded)], "Results must remain chronological and unique."
    if recorded == SEEDS:
        print("All 30 baseline recordings already complete; no GPU reruns.", flush=True)
        return
    episode_dir = output / "episodes"
    episode_dir.mkdir(exist_ok=True)
    policy, _ = load_policy(model_dir)
    for seed in SEEDS[len(recorded):]:
        saved = episode_dir / f"seed-{seed}.json"
        if saved.exists():
            # An interruption after per-episode saving must not force a GPU rerun.
            episode = read_json(saved)
            assert episode["seed"] == seed and (episode_dir / episode["trace"]).exists()
            print(f"Recovering already-saved baseline seed {seed}.", flush=True)
        else:
            assert not (episode_dir / f"seed-{seed}.trace.json.gz").exists(), "Partial recording needs inspection."
            episode = run_episode(policy, seed, episode_dir, Path("tmp/robotics-failure-pattern-videos"))
        summary["episodes"].append(dict(episode, source_directory=episode_dir.as_posix(), reused=False))
        summary["baseline_complete"] = len(summary["episodes"]) == 30
        write_json(output / "summary.json", summary)
        print(f"Discovery progress: {len(summary['episodes'])}/30 attempts saved.", flush=True)
    assert original_evidence()[2] == hashes, "Original evidence was modified."


def load_records(output):
    summary = read_json(output / "summary.json")
    records = []
    for episode in summary["episodes"]:
        with gzip.open(Path(episode["source_directory"]) / episode["trace"], "rt", encoding="utf-8") as stream:
            trace = json.load(stream)
        assert len(trace["steps"]) == episode["steps"]
        records.append({"episode": episode, "rows": trace["steps"]})
    return records


def analyze(output):
    assert read_json(output / "summary.json")["baseline_complete"], "Complete the predetermined seed list first."
    analysis = analyze_collection(load_records(output))
    write_json(output / "analysis.json", analysis)
    print(json.dumps(analysis["groups"], indent=2), flush=True)


def select_recovery_cases(analysis):
    """Chronological eligible branch; seed order, never largest drop or best restart."""
    selected, success_control = [], None
    for episode in sorted(analysis["episodes"], key=lambda row: row["seed"]):
        eligible = [event for event in episode["regression_events"] if 300 - event["branch_step"] >= 16]
        if not eligible:
            continue
        event = min(eligible, key=lambda row: row["branch_step"])
        case = {"seed": episode["seed"], "baseline_success": episode["success"], "event": event}
        if not episode["success"] and len(selected) < 3:
            selected.append(case)
        elif episode["success"] and success_control is None:
            success_control = case
    # A one-off observation is not yet a recurring pattern to target.
    if len(selected) < 2:
        return []
    return selected + ([success_control] if success_control is not None else [])


def cached_remainder(rows, branch):
    """Unnormalized actions still pending in this chunk, without a new prediction."""
    pending = rows[branch - 1]["pending_actions"]
    remainder = rows[branch:branch + pending]
    assert all(not row["new_prediction"] and row["chunk"] == rows[branch - 1]["chunk"]
               for row in remainder), "Cached remainder crosses a prediction boundary."
    # A successful recorded episode may end before its action queue is exhausted.
    assert len(remainder) == pending or branch + len(remainder) == len(rows)
    return remainder


def continuation(policy, record, case, diffusion_seed, arm, output):
    episode, rows = record["episode"], record["rows"]
    branch = case["event"]["branch_step"]
    env, observation, history = replay_prefix(episode["seed"], rows, branch)
    initial = physical_state(env, observation)
    prefix_best = max(rows[0]["before"]["coverage"],
                      max(row["after"]["coverage"] for row in rows[:branch]))
    prefix_queries = sum(row["new_prediction"] for row in rows[:branch])
    try:
        prime_history(policy, history)
        if arm == "continue_cached":
            policy._queues["action"].extend(torch.tensor(row["action"], device="cuda", dtype=torch.float32)
                                           .unsqueeze(0) for row in cached_remainder(rows, branch))
        else:
            assert arm == "replan_once"
        np.random.seed(diffusion_seed)
        torch.manual_seed(diffusion_seed)
        counter = pair_contact_observer(env)
        suffix, queries = [], 0
        original_cached_prefix = arm == "continue_cached"
        best = initial["coverage"]
        tick = time.perf_counter()
        for absolute_step in range(branch + 1, 301):
            new_prediction = not policy._queues["action"]
            if new_prediction:
                original_cached_prefix = False
            queries += bool(new_prediction)
            counter["points"] = 0
            with torch.inference_mode():
                action = policy.select_action(observation_batch(observation, "cuda"))
            action_np = action.squeeze(0).cpu().numpy()
            assert np.isfinite(action_np).all()
            observation, reward, terminated, truncated, _ = env.step(action_np)
            after = physical_state(env, observation)
            if original_cached_prefix:
                assert states_match(after, rows[absolute_step - 1]["after"]), "Restored cached actions differ from original continuation."
            best = max(best, after["coverage"])
            suffix.append({"step": absolute_step, "action": action_np.tolist(),
                           "new_prediction": bool(new_prediction), "reward": float(reward),
                           "pair_contact_points": counter["points"], "after": after})
            if len(suffix) % 40 == 0:
                print(f"Recovery scene {episode['seed']}, sample {diffusion_seed}, {arm}: "
                      f"step {absolute_step}, coverage={after['coverage']:.3f}", flush=True)
            if terminated or truncated:
                break
        torch.cuda.synchronize()
        name = f"seed-{episode['seed']}-sample-{diffusion_seed}-{arm}"
        result = {
            "seed": episode["seed"], "branch_step": branch, "diffusion_seed": diffusion_seed,
            "arm": arm, "identical_branch_state": True, "success": bool(terminated),
            "cached_prefix_check": "verified" if arm == "continue_cached" else "not applicable",
            "truncated": bool(truncated), "remaining_actions_executed": len(suffix),
            "total_actions": branch + len(suffix), "remaining_model_queries": queries,
            "total_model_queries": prefix_queries + queries,
            "final_coverage": suffix[-1]["after"]["coverage"], "post_branch_max_coverage": best,
            "full_episode_max_coverage": max(prefix_best, best),
            "wall_seconds": time.perf_counter() - tick, "trace": name + ".trace.json.gz",
        }
        with gzip.open(output / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"branch_state": initial, "steps": suffix}, stream, allow_nan=False)
        write_json(output / (name + ".json"), result)
        print(f"Recovery complete: scene {episode['seed']}, sample {diffusion_seed}, "
              f"{arm}, success={terminated}, queries={queries}.", flush=True)
        return result
    finally:
        env.close()


def recover(output, model_dir):
    protocol = read_json(output / "protocol.json")
    assert original_evidence()[2] == protocol["original_hashes"]
    assert read_json(model_dir / "config.json") == protocol["config"]
    records = load_records(output)
    analysis = read_json(output / "analysis.json")
    assert [row["episode"]["seed"] for row in records] == SEEDS
    assert analysis == analyze_collection(records), "Analysis differs from saved traces."
    cases = select_recovery_cases(analysis)
    result_path = output / "recovery.json"
    if result_path.exists():
        result = read_json(result_path)
        assert result["cases"] == cases, "Do not change recovery selection on resume."
        assert result["fresh_diffusion_seeds"] == RECOVERY_SEEDS
        expected = {(case["seed"], sample, arm) for case in cases for sample in RECOVERY_SEEDS
                    for arm in ["continue_cached", "replan_once"]}
        saved = [(row["seed"], row["diffusion_seed"], row["arm"]) for row in result["trials"]]
        assert len(saved) == len(set(saved)) and set(saved) <= expected
        for trial in result["trials"]:
            case = next(row for row in cases if row["seed"] == trial["seed"])
            assert trial["branch_step"] == case["event"]["branch_step"]
            assert (output / "recovery" / trial["trace"]).exists()
        if result["complete"]:
            assert set(saved) == expected, "Completed pilot lacks expected trials."
            print("Recovery pilot already complete; no GPU reruns.", flush=True)
            return
    else:
        result = {
            "complete": False, "cases": cases, "trials": [],
            "minimum_recurrent_failed_scenes": 2,
            "fresh_diffusion_seeds": RECOVERY_SEEDS,
            "randomness": "same fresh seed per arm, aligned by query order, not by control time",
            "recorded_suffix": "reference only; continue_cached uses fresh future randomness",
            "status": "running" if cases else "no recurring eligible regression; recovery not justified",
        }
        write_json(result_path, result)
    if not cases:
        result["complete"] = True
        write_json(result_path, result)
        return
    policy, _ = load_policy(model_dir)
    recovery_dir = output / "recovery"
    recovery_dir.mkdir(exist_ok=True)
    for case in cases:
        record = next(row for row in records if row["episode"]["seed"] == case["seed"])
        for diffusion_seed in RECOVERY_SEEDS:
            for arm in ["continue_cached", "replan_once"]:
                key = (case["seed"], diffusion_seed, arm)
                if any((row["seed"], row["diffusion_seed"], row["arm"]) == key for row in result["trials"]):
                    continue
                path = recovery_dir / f"seed-{case['seed']}-sample-{diffusion_seed}-{arm}.json"
                trial = read_json(path) if path.exists() else continuation(policy, record, case, diffusion_seed, arm, recovery_dir)
                assert (trial["seed"], trial["diffusion_seed"], trial["arm"]) == key
                assert trial["branch_step"] == case["event"]["branch_step"]
                assert (recovery_dir / trial["trace"]).exists()
                result["trials"].append(trial)
                write_json(result_path, result)
    result["complete"], result["status"] = True, "completed diagnostic pilot, not held-out controller evaluation"
    write_json(result_path, result)


def review(output):
    """CPU-only visual replay of prespecified branch scenes and every pilot sample."""
    result = read_json(output / "recovery.json")
    assert result["complete"]
    records = load_records(output)
    visual_dir = output / "figures"
    visual_dir.mkdir(exist_ok=True)
    for case in result["cases"]:
        record = next(row for row in records if row["episode"]["seed"] == case["seed"])
        rows, branch = record["rows"], case["event"]["branch_step"]
        peak = case["event"]["peak_step"]
        selected = {peak, branch, min(branch + 8, len(rows)), len(rows)}
        tiles = []

        def tile(env, label, coverage):
            frame = cv2.resize(env.render(), (256, 256))
            cv2.rectangle(frame, (0, 0), (256, 43), (255, 255, 255), -1)
            cv2.putText(frame, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"coverage {coverage:.2%}", (4, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            return frame

        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
        observation, _ = env.reset(seed=case["seed"])
        try:
            if 0 in selected:
                tiles.append(tile(env, "recorded step 0", rows[0]["before"]["coverage"]))
            for row in rows:
                observation, *_ = env.step(np.asarray(row["action"], dtype=np.float32))
                if row["step"] in selected:
                    assert states_match(physical_state(env, observation), row["after"])
                    tiles.append(tile(env, f"recorded step {row['step']}", row["after"]["coverage"]))
        finally:
            env.close()
        for sample in RECOVERY_SEEDS:
            for arm in ["continue_cached", "replan_once"]:
                trial = next(row for row in result["trials"] if row["seed"] == case["seed"]
                             and row["diffusion_seed"] == sample and row["arm"] == arm)
                with gzip.open(output / "recovery" / trial["trace"], "rt", encoding="utf-8") as stream:
                    suffix = json.load(stream)["steps"]
                env, observation, _ = replay_prefix(case["seed"], rows, branch)
                try:
                    for row in suffix:
                        observation, *_ = env.step(np.asarray(row["action"], dtype=np.float32))
                    assert states_match(physical_state(env, observation), suffix[-1]["after"])
                    short_arm = "continue" if arm == "continue_cached" else "replan"
                    tiles.append(tile(env, f"{short_arm} {sample}: step {trial['total_actions']}", trial["final_coverage"]))
                finally:
                    env.close()
        while len(tiles) % 4:
            tiles.append(np.full((256, 256, 3), 255, dtype=np.uint8))
        sheet = np.vstack([np.hstack(tiles[start:start + 4]) for start in range(0, len(tiles), 4)])
        assert cv2.imwrite(str(visual_dir / f"seed-{case['seed']}-recovery-frames.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("Saved replay-verified frame sheets; no GPU inference.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["collect", "analyze", "recover", "review"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=Path("data/raw/robotics/diffusion_pusht"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "collect":
        assert torch.cuda.is_available(), "Use the validated GPU environment; no silent CPU substitution."
        collect(args.output_dir, args.model_dir, args.resume)
    elif args.stage == "analyze":
        analyze(args.output_dir)
    elif args.stage == "recover":
        assert torch.cuda.is_available()
        recover(args.output_dir, args.model_dir)
    else:
        review(args.output_dir)


if __name__ == "__main__":
    main()
