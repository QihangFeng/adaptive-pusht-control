"""Bounded development screen of eight/twelve-action reuse and visual interruption.

The completed benefit pilot is immutable. This separate, non-learned controller
uses camera-derived signals and its own predicted commands, never dynamic truth.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

import robotics_benefit_pilot as base
from robotics_baseline_check import load_policy, physical_state, states_match
from robotics_feasibility import MemorySampler, observation_batch
from robotics_visual_monitor import RollingVisualFeatures, VisualAlignmentEstimator


OUTPUT = Path("results/robotics-adaptive-length")
OLD = Path("results/robotics-benefit-pilot")
SEEDS = list(range(8))
METHODS = ["fixed8", "fixed12", "adaptive12"]
RULES = {
    "base_length": 8, "maximum_length": 12, "minimum_history": 8,
    "extension_max_coverage": .75, "extension_max_drawdown": .015,
    "extension_min_recovery": -.01, "minimum_command_step": 2.,
    "minimum_direction_cosine": .5, "interrupt_warnings": 2,
    "interrupt_drawdown": .03, "interrupt_peak": .80,
    "interrupt_recovery": -.005, "confirmation_steps": 2,
    "minimum_plan_age": 4, "minimum_pending": 2,
    "maximum_interruptions": 2, "cooldown_actions": 12,
}
PACKAGES = ["torch", "lerobot", "gym-pusht", "numpy", "scipy", "shapely", "diffusers"]


def extension_candidate(features):
    """Low-warning, far-from-goal eligibility is NOT a safety/failure label."""
    return bool(features["quality"]
                and features["history_steps"] >= RULES["minimum_history"]
                and features["coverage"] < RULES["extension_max_coverage"]
                and features["warning_count"] == 0
                and features["drawdown"] <= RULES["extension_max_drawdown"]
                and features["recovery"] >= RULES["extension_min_recovery"])


def _smooth(points):
    differences = np.diff(np.asarray(points, dtype=float), axis=0)
    speeds = np.linalg.norm(differences, axis=1)
    if len(speeds) < 2 or not np.isfinite(differences).all() or np.any(speeds < RULES["minimum_command_step"]):
        return False
    cosine = np.sum(differences[:-1] * differences[1:], axis=1) / (speeds[:-1] * speeds[1:])
    return bool(np.all(cosine >= RULES["minimum_direction_cosine"]))


def plan_continuity(commands):
    commands = np.asarray(commands, dtype=float)
    if commands.shape != (12, 2) or not np.isfinite(commands).all():
        return False
    # Last four prefix commands plus the proposed suffix, including their join.
    return _smooth(commands[4:])


class AdaptiveLengthController:
    def __init__(self):
        self.interruptions = 0
        self.last_interrupt = -RULES["cooldown_actions"]
        self.confirmations = 0
        self.armed = True

    def update(self, features, age, pending, step, commands, done=False):
        if done:
            return None
        valid = features["quality"]
        regression = bool(valid and features["warning_count"] >= RULES["interrupt_warnings"]
                          and features["drawdown"] >= RULES["interrupt_drawdown"]
                          and features["recent_peak"] >= RULES["interrupt_peak"]
                          and features["recovery"] < RULES["interrupt_recovery"])
        self.confirmations = self.confirmations + 1 if regression else 0
        if valid and not regression:
            self.armed = True
        if (regression and self.armed and self.confirmations >= RULES["confirmation_steps"]
                and age >= RULES["minimum_plan_age"] and pending >= RULES["minimum_pending"]
                and self.interruptions < RULES["maximum_interruptions"]
                and step - self.last_interrupt >= RULES["cooldown_actions"]):
            self.interruptions += 1
            self.last_interrupt, self.armed = step, False
            return {"kind": "interrupt", "reason": "confirmed_regression", "discarded": pending}
        if age == RULES["base_length"] and pending == RULES["maximum_length"] - RULES["base_length"]:
            retain = extension_candidate(features) and plan_continuity(commands)
            return {"kind": "extend" if retain else "trim",
                    "reason": "camera_and_plan_eligible" if retain else "retain_eight_action_interval",
                    "discarded": 0 if retain else pending}
        return None


def fingerprints():
    result = {"experiments/robotics_adaptive_length.py": base.sha(Path(__file__)),
              (OLD / "protocol.json").as_posix(): base.sha(OLD / "protocol.json"),
              (OLD / "visual-features.json.gz").as_posix(): base.sha(OLD / "visual-features.json.gz"),
              (OLD / "near-target-validation.json").as_posix(): base.sha(OLD / "near-target-validation.json")}
    result.update(base.source_fingerprints())
    result.update({("experiments/" + name if name != "requirements-robotics.txt" else name): value
                   for name, value in base.code_fingerprints().items()})
    return result


def analyze(output):
    base.verify_protocol(OLD)
    with gzip.open(OLD / "visual-features.json.gz", "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    records = {r["episode"]["seed"]: r for r in base.load_records(base.DISCOVERY)}
    details = []
    for episode in episodes:
        rows = records[episode["seed"]]["rows"]
        candidates = []
        for i, features in enumerate(episode["features"]):
            if i + 4 >= len(rows) or features["pending_actions"] != 0 or not extension_candidate(features):
                continue
            # These five commands are already executed: a proxy, not the unseen suffix.
            prefix_smooth = _smooth([r["action"] for r in rows[i - 4:i + 1]])
            following_drop = features["coverage"] - episode["features"][i + 4]["coverage"]
            candidates.append({"step": features["step"], "coverage": features["coverage"],
                               "prefix_smooth_proxy": prefix_smooth,
                               "fresh_next_four_drop": following_drop})
        details.append({"seed": episode["seed"], "candidates": candidates,
                        "candidate_count": len(candidates),
                        "prefix_smooth_count": sum(c["prefix_smooth_proxy"] for c in candidates),
                        "fresh_next_four_drops_over_three_points": sum(c["fresh_next_four_drop"] > .03 for c in candidates)})
    document = {"scope": "discovery scenes 0-29 only; no simulator or inference",
                "warning": "The recorded next four actions are fresh predictions, NOT the discarded suffix.",
                "rules": RULES, "episodes": details,
                "total_candidates": sum(r["candidate_count"] for r in details),
                "total_prefix_smooth": sum(r["prefix_smooth_count"] for r in details)}
    base.save_json(output / "discovery-analysis.json", document)
    return document


def prepare(output):
    base.verify_protocol(OLD)
    if not (output / "discovery-analysis.json").exists():
        assert not (output / "protocol.json").exists(), "Frozen discovery analysis is missing."
        analyze(output)
    protocol = {"scope": "24 development attempts, not held-out validation or publication proof",
                "seeds": SEEDS, "methods": METHODS, "rules": RULES,
                "max_total_actions": 300, "success": "coverage strictly >.95; simulator termination",
                "rng": "reset numpy/torch to scene seed; aligned by prediction order, not time",
                "checkpoint": {"model_id": base.MODEL_ID, "revision": base.MODEL_REVISION,
                               "weights_sha256": base.WEIGHTS_SHA256},
                "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
                "fingerprints": fingerprints(),
                "analysis_sha256": base.sha(output / "discovery-analysis.json")}
    path = output / "protocol.json"
    if path.exists():
        assert base.read_json(path) == protocol, "Frozen protocol, sources or installed packages changed."
    else:
        assert not (output / "episodes").exists(), "Do not bind existing episodes to a new protocol."
        base.save_json(path, protocol)
        return protocol
    return protocol


def verify_protocol(output):
    base.verify_protocol(OLD)
    protocol = base.read_json(output / "protocol.json")
    assert protocol["fingerprints"] == fingerprints(), "Frozen code or evidence changed."
    assert protocol["packages"] == {name: importlib.metadata.version(name) for name in PACKAGES}
    assert protocol["analysis_sha256"] == base.sha(output / "discovery-analysis.json")
    assert protocol["rules"] == RULES and protocol["seeds"] == SEEDS and protocol["methods"] == METHODS
    return protocol


def smoke(output):
    verify_protocol(output)
    destination = output / "prefix-smoke.json"
    if destination.exists():
        assert base.read_json(destination)["passed"]
        return
    policy, _ = load_policy(base.MODEL)
    traces = {}
    for method in ["fixed8", "fixed12", "forced_trim12"]:
        base.configure_horizon(policy, 8 if method == "fixed8" else 12)
        np.random.seed(1234)
        torch.manual_seed(1234)
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
        observation, _ = env.reset(seed=0)
        rows = []
        try:
            for step in range(1, 17):
                with torch.inference_mode():
                    command = policy.select_action(observation_batch(observation, "cuda")).squeeze(0).cpu().numpy()
                observation, _, _, _, _ = env.step(command)
                rows.append({"action": command.tolist(), "after": physical_state(env, observation),
                             "pending": len(policy._queues["action"])})
                if method == "forced_trim12" and step % 8 == 0:
                    policy._queues["action"].clear()
        finally:
            env.close()
        traces[method] = rows
    for method in ["fixed12", "forced_trim12"]:
        for a, b in zip(traces["fixed8"][:8], traces[method][:8]):
            assert a["action"] == b["action"] and states_match(a["after"], b["after"])
    for a, b in zip(traces["fixed8"], traces["forced_trim12"]):
        assert a["action"] == b["action"] and states_match(a["after"], b["after"])
    assert traces["fixed12"][7]["pending"] == 4
    assert traces["fixed12"][11]["pending"] == 0
    base.save_json(destination, {"passed": True, "seed": 0, "diffusion_seed": 1234,
                               "first_eight_exact": True, "forced_trim_sixteen_exact": True,
                               "fixed12_executed_cached_suffix": True, "traces": traces})


def evaluate_episode(policy, seed, method, output):
    base.configure_horizon(policy, 8 if method == "fixed8" else 12)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    observation, _ = env.reset(seed=seed)
    controller, estimator, rolling = AdaptiveLengthController(), VisualAlignmentEstimator(), RollingVisualFeatures()
    initial = physical_state(env, observation)
    rows, units, events = [], [], []
    queries, age, monitor_seconds = 0, 0, 0.
    started = time.perf_counter()
    if method == "adaptive12":
        tick = time.perf_counter()
        rolling.update(estimator.estimate(observation["pixels"]), 0)
        monitor_seconds += time.perf_counter() - tick
    try:
        for step in range(1, 301):
            fresh = not policy._queues["action"]
            if fresh:
                if age:
                    units.append(age)
                age, queries = 0, queries + 1
            with torch.inference_mode():
                command = policy.select_action(observation_batch(observation, "cuda")).squeeze(0).cpu().numpy()
            if fresh:
                commands = [command.tolist()] + [a.squeeze(0).detach().cpu().numpy().tolist()
                                                for a in policy._queues["action"]]
            age += 1
            observation, _, terminated, truncated, _ = env.step(command)
            pending = len(policy._queues["action"])
            row = {"step": step, "action": command.tolist(), "new_prediction": bool(fresh),
                   "plan_age": age, "pending_before": pending, "after": physical_state(env, observation)}
            if method == "adaptive12" and not (terminated or truncated):
                tick = time.perf_counter()
                features = rolling.update(estimator.estimate(observation["pixels"]), pending)
                event = controller.update(features, age, pending, step, commands)
                if event:
                    events.append({"step": step, "plan_age": age, **event, "features": features})
                    if event["discarded"]:
                        policy._queues["action"].clear()
                monitor_seconds += time.perf_counter() - tick
                row["features"] = features
            row["pending_after"] = len(policy._queues["action"])
            rows.append(row)
            if step % 80 == 0:
                print(f"Running {method}, seed {seed}: action {step}, calls {queries}", flush=True)
            if terminated or truncated:
                break
        torch.cuda.synchronize()
        units.append(age)
        result = {"seed": seed, "method": method, "success": bool(terminated), "truncated": bool(truncated),
                  "steps": len(rows), "model_queries": queries, "initial": initial,
                  "final_coverage": rows[-1]["after"]["coverage"],
                  "wall_seconds": time.perf_counter() - started, "monitor_seconds": monitor_seconds,
                  "unit_lengths": units, "events": events, "trace": f"seed-{seed}-{method}.trace.json.gz"}
        with gzip.open(output / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"initial": initial, "steps": rows}, stream, allow_nan=False)
        result["trace_sha256"] = base.sha(output / result["trace"])
        base.save_json(output / f"seed-{seed}-{method}.json", result)
        return result
    finally:
        env.close()


def verify_trial(trial, directory):
    assert trial["seed"] in SEEDS and trial["method"] in METHODS
    assert base.sha(directory / trial["trace"]) == trial["trace_sha256"]
    with gzip.open(directory / trial["trace"], "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["steps"]
    assert len(rows) == trial["steps"] <= 300
    assert trial["model_queries"] == sum(r["new_prediction"] for r in rows) == len(trial["unit_lengths"])
    assert sum(trial["unit_lengths"]) == len(rows)
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and len(rows) == 300)


def run(output):
    verify_protocol(output)
    assert torch.cuda.is_available(), "Do not silently fall back to CPU policy inference."
    smoke(output)
    directory = output / "episodes"
    directory.mkdir(exist_ok=True)
    results, policy = [], None
    for index, seed in enumerate(SEEDS):
        for method in METHODS[index % 3:] + METHODS[:index % 3]:
            path = directory / f"seed-{seed}-{method}.json"
            if path.exists():
                result = base.read_json(path)
            else:
                assert not (directory / f"seed-{seed}-{method}.trace.json.gz").exists(), "Inspect partial trace before resuming."
                if policy is None:
                    policy, _ = load_policy(base.MODEL)
                print(f"Starting {method}, seed {seed}", flush=True)
                result = evaluate_episode(policy, seed, method, directory)
            assert (result["seed"], result["method"]) == (seed, method)
            verify_trial(result, directory)
            results.append(result)
            base.save_json(output / "episodes.json", {"complete": len(results) == 24, "episodes": results})
            print(f"Progress {len(results)}/24: {method} seed {seed}, success={result['success']}, "
                  f"calls={result['model_queries']}, time={result['wall_seconds']:.1f}s", flush=True)
    verify_protocol(output)
    summarize(output)


def summarize(output):
    verify_protocol(output)
    document = base.read_json(output / "episodes.json")
    assert document["complete"] and len(document["episodes"]) == 24
    assert len({(r["seed"], r["method"]) for r in document["episodes"]}) == 24
    table = {}
    for method in METHODS:
        rows = [r for r in document["episodes"] if r["method"] == method]
        for row in rows:
            verify_trial(row, output / "episodes")
        table[method] = {"attempts": len(rows), "successes": sum(r["success"] for r in rows),
                         **{f"mean_{name}": float(np.mean([r[name] for r in rows]))
                            for name in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                         "extensions": sum(e["kind"] == "extend" for r in rows for e in r["events"]),
                         "interruptions": sum(e["kind"] == "interrupt" for r in rows for e in r["events"])}
    base.save_json(output / "summary.json", {"complete": True, "methods": table,
                                            "scope": "Eight reused development scenes, one sample per method; not held-out evidence."})
    print(json.dumps(table, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["analyze", "prepare", "smoke", "run", "summarize"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    stage = globals()[args.stage]
    if args.stage == "run":
        prepare(args.output_dir)
    tick = time.perf_counter()
    if args.stage in ["smoke", "run"]:
        assert torch.cuda.is_available()
        torch.cuda.reset_peak_memory_stats()
    with MemorySampler() as sampler:
        stage(args.output_dir)
    resources = {"wall_seconds": time.perf_counter() - tick, "peak_process_rss_bytes": sampler.peak_rss,
                 "minimum_system_available_bytes": sampler.minimum_available}
    if args.stage in ["smoke", "run"]:
        resources.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    base.save_json(args.output_dir / f"{args.stage}-resources.json", resources)


if __name__ == "__main__":
    main()
