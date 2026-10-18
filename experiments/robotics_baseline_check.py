"""Passive fixed-eight PushT screen and bounded, oracle-selected local probes.

The policy still receives only pixels and pusher position. Simulator geometry and
pair-specific contacts are diagnostics, NOT inputs to a proposed visual monitor.
"""

from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from robotics_feasibility import (
    MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256, observation_batch, summarize,
)
import cv2
import draccus
import gymnasium as gym
import numpy as np
import torch
from lerobot.constants import OBS_IMAGES
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import populate_queues


RULES = {
    "window_steps": 3,
    "near_gap_pixels": 10.0,
    "max_block_vertex_motion_pixels": 2.0,
    "min_command_distance_pixels": 5.0,
    "minimum_pending_actions": 2,
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_policy(model_dir):
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    with (model_dir / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == WEIGHTS_SHA256, "Unexpected checkpoint weights."
    assert (raw["n_obs_steps"], raw["horizon"], raw["n_action_steps"]) == (2, 16, 8)
    config = draccus.decode(DiffusionConfig, {k: v for k, v in raw.items() if k != "type"})
    config.device = "cuda"
    policy = DiffusionPolicy.from_pretrained(model_dir, config=config, strict=True).to("cuda").eval()
    assert all(torch.isfinite(buffer).all() for _, buffer in policy.named_buffers())
    return policy, raw


def pair_contact_observer(env):
    """Keep the simulator's callback, counting only agent--block contact separately."""
    base = env.unwrapped
    counter = {"points": 0}
    original = base.collision_handeler.post_solve

    def observe(arbiter, space, data):
        original(arbiter, space, data)
        bodies = [shape.body for shape in arbiter.shapes]
        if ((bodies[0] is base.agent and bodies[1] is base.block)
                or (bodies[1] is base.agent and bodies[0] is base.block)):
            counter["points"] += len(arbiter.contact_point_set.points)

    base.collision_handeler.post_solve = observe
    return counter


def physical_state(env, observation):
    base = env.unwrapped
    radius = next(iter(base.agent.shapes)).radius
    gap = min(shape.point_query(base.agent.position).distance - radius for shape in base.block.shapes)
    shapes = sorted(base.block.shapes, key=lambda shape: tuple(tuple(v) for v in shape.get_vertices()))
    vertices = [list(shape.body.local_to_world(v))
                for shape in shapes for v in shape.get_vertices()]
    return {
        "pusher_pos": list(base.agent.position), "pusher_velocity": list(base.agent.velocity),
        "block_pos": list(base.block.position), "block_angle": float(base.block.angle),
        "block_velocity": list(base.block.velocity),
        "block_angular_velocity": float(base.block.angular_velocity),
        "block_vertices": vertices, "pusher_block_gap": float(gap),
        "coverage": float(base._get_coverage()),
        "image_sha256": hashlib.sha256(observation["pixels"].tobytes()).hexdigest(),
    }


def vertex_motion(a, b):
    delta = np.asarray(b["block_vertices"]) - np.asarray(a["block_vertices"])
    return float(np.max(np.linalg.norm(delta, axis=1)))


def states_match(a, b):
    """Geometry union order may change coverage's last bit; physics/pixels stay exact."""
    left, right = dict(a), dict(b)
    left_coverage, right_coverage = left.pop("coverage"), right.pop("coverage")
    return left == right and abs(left_coverage - right_coverage) <= 1e-12


def feasibility_matches(episodes):
    previous = json.loads(Path("results/robotics-feasibility/summary.json").read_text(encoding="utf-8"))
    matches = []
    for episode in episodes:
        original = next((row for row in previous["episodes"] if row["seed"] == episode["seed"]), None)
        if original:
            exact_keys = ["steps", "success", "truncated", "model_queries"]
            differences = {key: abs(episode[key] - original[key]) for key in ["max_reward", "final_reward"]}
            matches.append({"seed": episode["seed"],
                            "all_match": all(episode[k] == original[k] for k in exact_keys)
                            and max(differences.values()) <= 1e-12,
                            "reward_absolute_differences": differences, "absolute_tolerance": 1e-12})
    assert all(row["all_match"] for row in matches), "Baseline differs beyond numeric rounding tolerance."
    return matches


def screen_candidates(rows, rules=RULES):
    """Conservative flags for human review, never automatic failure labels."""
    candidates = []
    width = rules["window_steps"]
    for end in range(width - 1, len(rows)):
        window = rows[end - width + 1:end + 1]
        if len({row["chunk"] for row in window}) != 1:
            continue
        if window[-1]["pending_actions"] < rules["minimum_pending_actions"]:
            continue
        start_state = window[0]["before"]
        motion = max(vertex_motion(start_state, row["after"]) for row in window)
        if motion > rules["max_block_vertex_motion_pixels"]:
            continue
        command = float(np.mean([row["command_distance"] for row in window]))
        if command < rules["min_command_distance_pixels"]:
            continue
        label = None
        if all(row["pair_contact_points"] > 0 for row in window):
            label = "contact_stall_candidate"
        elif all(row["pair_contact_points"] == 0 for row in window):
            prior = [row for row in rows[:end - width + 1] if row["chunk"] == window[0]["chunk"]]
            prior_contact = any(row["pair_contact_points"] > 0 for row in prior)
            near = all(min(row["before"]["pusher_block_gap"], row["after"]["pusher_block_gap"])
                       <= rules["near_gap_pixels"] for row in window)
            if prior_contact and near:
                label = "contact_loss_candidate"
        if label:
            # Keep the first non-overlapping flag of this type in a chunk.
            if any(c["chunk"] == window[0]["chunk"] and c["label"] == label for c in candidates):
                continue
            candidates.append({
                "label": label, "start_step": window[0]["step"], "end_step": window[-1]["step"],
                "chunk": window[0]["chunk"], "pending_actions": window[-1]["pending_actions"],
                "max_block_vertex_motion": motion, "mean_command_distance": command,
                "coverage_change": window[-1]["after"]["coverage"] - start_state["coverage"],
            })
    return candidates


def passive_simulator_test():
    """Observation-only contact callback and rendering must not affect physics."""
    left = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    right = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    left.reset(seed=2)
    right.reset(seed=2)
    counter = pair_contact_observer(right)
    for step in range(100):
        action = np.array([256 + 100 * np.sin(step / 9), 256 + 100 * np.cos(step / 7)], dtype=np.float32)
        lo, lr, lt, lx, _ = left.step(action)
        counter["points"] = 0
        ro, rr, rt, rx, _ = right.step(action)
        right.render()
        assert states_match(physical_state(left, lo), physical_state(right, ro))
        assert (lr, lt, lx) == (rr, rt, rx)
    left.close()
    right.close()
    return {"identical_steps": 100, "observation_only": True}


def run_episode(policy, seed, output, video_dir):
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    observation, _ = env.reset(seed=seed)
    counter = pair_contact_observer(env)
    rows, frames, query_times = [], [env.render()], []
    initial = physical_state(env, observation)
    chunk = 0
    start = time.perf_counter()
    try:
        for index in range(300):
            before = physical_state(env, observation)
            new_prediction = not policy._queues["action"]
            if new_prediction:
                chunk += 1
            batch = observation_batch(observation, "cuda")
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            torch.cuda.synchronize()
            prediction_seconds = time.perf_counter() - tick
            if new_prediction:
                query_times.append(prediction_seconds)
            action_np = action.squeeze(0).cpu().numpy()
            assert np.isfinite(action_np).all()
            counter["points"] = 0
            observation, reward, terminated, truncated, info = env.step(action_np)
            after = physical_state(env, observation)
            rows.append({
                "step": index + 1, "chunk": chunk, "new_prediction": bool(new_prediction),
                "action": action_np.tolist(), "pending_actions": len(policy._queues["action"]),
                "before": before, "after": after, "pair_contact_points": counter["points"],
                "generic_contact_points": int(env.unwrapped.n_contact_points),
                "reported_n_contacts": int(info["n_contacts"]), "reward": float(reward),
                "command_distance": float(np.linalg.norm(action_np - np.asarray(before["pusher_pos"]))),
                "pusher_motion": float(np.linalg.norm(np.asarray(after["pusher_pos"]) - before["pusher_pos"])),
                "block_vertex_motion": vertex_motion(before, after),
                "prediction_seconds": prediction_seconds if new_prediction else None,
            })
            frames.append(env.render())
            if (index + 1) % 40 == 0:
                print(f"Baseline seed {seed}: step {index + 1}, coverage={after['coverage']:.3f}", flush=True)
            if terminated or truncated:
                break
        candidates = screen_candidates(rows)
        result = {
            "seed": seed, "steps": len(rows), "success": bool(terminated), "truncated": bool(truncated),
            "initial_coverage": initial["coverage"], "final_coverage": rows[-1]["after"]["coverage"],
            "max_coverage": max(row["after"]["coverage"] for row in rows),
            "max_reward": max(row["reward"] for row in rows), "final_reward": rows[-1]["reward"],
            "model_queries": len(query_times), "query_seconds": summarize(query_times),
            "wall_seconds": time.perf_counter() - start, "candidates": candidates,
            "trace": f"seed-{seed}.trace.json.gz",
        }
        with gzip.open(output / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"initial": initial, "steps": rows}, stream, allow_nan=False)
        write_json(output / f"seed-{seed}.json", result)
        if video_dir:
            video_dir.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(video_dir / f"seed-{seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10, (frames[0].shape[1], frames[0].shape[0]))
            assert writer.isOpened(), "Video writer unavailable."
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        # Small, lossless visual evidence: start/end and candidate windows.
        selected = sorted({0, len(rows), *[max(0, c["start_step"] - 1) for c in candidates],
                           *[c["end_step"] for c in candidates]})
        selected = selected[:14] + ([len(rows)] if len(rows) not in selected[:14] else [])
        tiles = []
        for step in selected:
            tile = cv2.resize(frames[step], (256, 256))
            cv2.putText(tile, f"seed {seed}, step {step}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.full((256, 256, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        assert cv2.imwrite(str(output / f"seed-{seed}-frames.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"Baseline seed {seed} complete: success={terminated}, flags={len(candidates)}", flush=True)
        return result
    finally:
        env.close()


def replay_prefix(seed, rows, end_step):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    observation, _ = env.reset(seed=seed)
    last_observations = deque(maxlen=2)
    for row in rows[:end_step]:
        last_observations.append(observation)
        observation, _, _, _, _ = env.step(np.asarray(row["action"], dtype=np.float32))
    expected = rows[end_step - 1]["after"]
    assert states_match(physical_state(env, observation), expected), "Action-prefix replay differs from recorded state."
    return env, observation, last_observations


def prime_history(policy, observations):
    """Restore recorded observation history, not a cold-start/double-current-frame reset."""
    for key, queue in policy._queues.items():
        queue.clear()
    for observation in observations:
        batch = policy.normalize_inputs(observation_batch(observation, "cuda"))
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
        policy._queues = populate_queues(policy._queues, batch)


def local_probes(policy, episodes, output):
    """First flagged window per seed; all three fixed diffusion seeds are retained.

    Compare only the remainder of the CURRENT chunk. This is an oracle-selected
    local opportunity check, not a deployable monitor or full-episode recovery test.
    """
    probes = []
    for episode in episodes:
        if not episode["candidates"]:
            continue
        candidate = episode["candidates"][0]
        branch = candidate["end_step"]
        count = candidate["pending_actions"]
        with gzip.open(output / episode["trace"], "rt", encoding="utf-8") as stream:
            rows = json.load(stream)["steps"]
        if branch >= len(rows):
            probes.append({"seed": episode["seed"], "candidate": candidate,
                           "skipped": True, "reason": "episode ended at flagged window"})
            continue
        assert all(not row["new_prediction"] for row in rows[branch:branch + count])
        if len(rows[branch:branch + count]) < count:
            count = len(rows[branch:branch + count])
        baseline_end = rows[branch + count - 1]["after"]
        trials = []
        for diffusion_seed in [1000, 1001, 1002]:
            env, observation, history = replay_prefix(episode["seed"], rows, branch)
            try:
                prime_history(policy, history)
                torch.manual_seed(diffusion_seed)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                for _ in range(count):
                    with torch.inference_mode():
                        action = policy.select_action(observation_batch(observation, "cuda"))
                    observation, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
                    if terminated or truncated:
                        break
                torch.cuda.synchronize()
                end = physical_state(env, observation)
                trials.append({
                    "diffusion_seed": diffusion_seed, "end_coverage": end["coverage"],
                    "coverage_difference": end["coverage"] - baseline_end["coverage"],
                    "block_vertex_motion": vertex_motion(rows[branch - 1]["after"], end),
                    "reached_target": bool(terminated), "extra_model_queries": 1,
                    "wall_seconds": time.perf_counter() - tick,
                })
            finally:
                env.close()
        probes.append({"seed": episode["seed"], "candidate": candidate, "branch_step": branch,
                       "remaining_steps": count, "identical_branch_state": True,
                       "baseline_end_coverage": baseline_end["coverage"],
                       "baseline_block_vertex_motion": vertex_motion(rows[branch - 1]["after"], baseline_end),
                       "trials": trials})
        print(f"Local probe seed {episode['seed']} at step {branch}: "
              f"coverage differences={[round(t['coverage_difference'], 4) for t in trials]}", flush=True)
    return probes


def make_review_plot(episodes, output):
    """Saved coverage/contact traces, not a new experiment or detector."""
    canvas = np.full((210 * len(episodes) + 70, 1050, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, episode in enumerate(episodes):
        with gzip.open(output / episode["trace"], "rt", encoding="utf-8") as stream:
            rows = json.load(stream)["steps"]
        steps = [0] + [row["step"] for row in rows]
        coverage = [episode["initial_coverage"]] + [row["after"]["coverage"] for row in rows]
        top, bottom = panel * 210 + 35, panel * 210 + 185
        x = lambda step: 80 + int(step / 300 * 930)
        y = lambda value: bottom - int(value * 150)
        for value in [0, 0.5, 1]:
            cv2.line(canvas, (80, y(value)), (1010, y(value)), (220, 220, 220), 1)
            cv2.putText(canvas, f"{value:.1f}", (30, y(value) + 5), font, 0.42, (60, 60, 60), 1)
        for candidate in episode["candidates"]:
            cv2.rectangle(canvas, (x(candidate["start_step"]), top),
                          (x(candidate["end_step"]), bottom), (210, 235, 255), -1)
        for left in range(80, 1010, 15):
            cv2.line(canvas, (left, y(0.95)), (min(left + 7, 1010), y(0.95)), (40, 40, 40), 1)
        points = np.array([[x(step), y(value)] for step, value in zip(steps, coverage)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, (200, 90, 30), 2, cv2.LINE_AA)
        for row in rows:
            if row["pair_contact_points"] > 0:
                cv2.line(canvas, (x(row["step"]), bottom + 4), (x(row["step"]), bottom + 8), (40, 140, 40), 1)
        for step in [0, 100, 200, 300]:
            cv2.putText(canvas, str(step), (x(step) - 10, bottom + 22), font, 0.4, (60, 60, 60), 1)
        cv2.putText(canvas, f"Seed {episode['seed']}: {'success' if episode['success'] else 'timeout'}",
                    (80, top - 12), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    offset = 210 * len(episodes)
    cv2.putText(canvas, "Blue: target coverage | dashed: >95% success | green: pusher-block contact | orange: candidate",
                (30, offset + 30), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Horizontal axis: control steps (10 steps = 1 simulated second)",
                (30, offset + 55), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(output / "coverage-contact-review.png"), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/raw/robotics/diffusion_pusht"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/robotics-baseline-check"))
    parser.add_argument("--video-dir", type=Path, default=Path("tmp/robotics-baseline-videos"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if not args.probes_only and not args.review_only and output.exists() and any(output.iterdir()):
        parser.error("Output is nonempty; use a separate *-repeat directory or --probes-only.")
    output.mkdir(parents=True, exist_ok=True)
    if args.review_only:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        make_review_plot(summary["episodes"], output)
        return
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA required; do not silently substitute CPU."
    policy, config = load_policy(args.model_dir)
    if args.probes_only:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        assert [episode["seed"] for episode in summary["episodes"]] == protocol["seeds"]
        assert protocol["config"] == config, "Do not mix another model configuration with saved episodes."
        summary["feasibility_summary_matches"] = feasibility_matches(summary["episodes"])
        summary["local_probes"] = local_probes(policy, summary["episodes"], output)
        summary["complete"] = True
        write_json(output / "summary.json", summary)
        return
    protocol = {
        "purpose": "exploratory unchanged-baseline diagnosis, not controller validation",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
        "config": config, "seeds": args.seeds, "max_episode_steps": 300, "disturbances": "none",
        "candidate_rules": RULES, "diagnostic_state_not_policy_input": True,
        "selection": "first flagged window per seed, chronological; all local trials retained",
        "probe_diffusion_seeds": [1000, 1001, 1002],
        "probe_scope": "current-chunk remainder only; additional one model query",
        "packages": {name: importlib.metadata.version(name) for name in
                     ["lerobot", "gym-pusht", "pymunk", "torch", "numpy", "diffusers"]},
        "sources": {"policy": "https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py",
                    "simulator": "https://github.com/huggingface/gym-pusht",
                    "paper": "https://arxiv.org/abs/2303.04137"},
    }
    write_json(output / "protocol.json", protocol)
    summary = {"complete": False, "passive_simulator_check": passive_simulator_test(), "episodes": []}
    write_json(output / "summary.json", summary)
    for seed in args.seeds:
        summary["episodes"].append(run_episode(policy, seed, output, args.video_dir))
        write_json(output / "summary.json", summary)
    summary["feasibility_summary_matches"] = feasibility_matches(summary["episodes"])
    summary["local_probes"] = local_probes(policy, summary["episodes"], output)
    summary["complete"] = True
    write_json(output / "summary.json", summary)
    print(f"Saved complete baseline screen to {output}", flush=True)


if __name__ == "__main__":
    main()
