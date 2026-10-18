"""Bounded pixel-only pose precision gate before any fine-pushing controller.

Replays recorded CPU actions to obtain saved RGB frames, not alternate futures. No policy load, new AI
predictions, corrective actions, estimator tuning or controller changes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import cv2
import gymnasium as gym
import gym_pusht
import numpy as np
import psutil

import robotics_visual_monitor as visual


OUTPUT = Path("results/robotics-pose-gate")
SOURCE = Path("results/robotics-failure-patterns/summary.json")
VALIDATION = Path("results/robotics-benefit-pilot/visual-validation.json")
SEEDS = list(range(30))
METHODS = ["cold", "previous_frame"]
GOAL = [256., 256., math.pi / 4]
DEADBANDS = [1., 1., math.radians(.5)]
LIMITS = {"minimum_valid_fraction": .98, "position_error_p95": 1.,
          "max_vertex_error_p95": 1., "angle_error_degrees_p95": .5,
          "minimum_direction_agreement": .95}
PACKAGES = ["numpy", "scipy", "shapely", "opencv-python", "gymnasium", "gym-pusht", "pymunk", "pygame", "psutil"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def signed_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def image_to_world_pose(pose):
    values = np.asarray(pose, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected three finite image-pose values.")
    return [float((values[0] - visual.PIXEL_CENTER_OFFSET) / visual.SCALE),
            float((values[1] - visual.PIXEL_CENTER_OFFSET) / visual.SCALE), float(values[2])]


def goal_residual(pose):
    return [GOAL[0] - pose[0], GOAL[1] - pose[1], signed_angle(GOAL[2] - pose[2])]


def world_vertices(pose):
    vertices = np.concatenate(visual.RECTANGLES) / visual.SCALE
    return vertices @ visual._rotation(pose[2]).T + np.asarray(pose[:2])


def score_pose_frame(estimate, state):
    truth = [*state["block_pos"], state["block_angle"]]
    true_residual = goal_residual(truth)
    eligible = [abs(value) >= deadband for value, deadband in zip(true_residual, DEADBANDS)]
    result = {"valid": bool(estimate["quality"]), "truth_pose": truth,
              "truth_residual": true_residual, "direction_eligible": eligible,
              "estimated_pose": None, "estimated_residual": None,
              "position_error": None, "angle_error_degrees": None,
              "max_vertex_error": None, "direction_agreement": [None] * 3}
    if not result["valid"]:
        return result
    predicted = image_to_world_pose(estimate["pose_image"])
    residual = goal_residual(predicted)
    result.update(estimated_pose=predicted, estimated_residual=residual,
                  position_error=math.dist(predicted[:2], truth[:2]),
                  angle_error_degrees=math.degrees(abs(signed_angle(predicted[2] - truth[2]))),
                  max_vertex_error=float(np.linalg.norm(world_vertices(predicted) - world_vertices(truth), axis=1).max()),
                  direction_agreement=[bool(a * b > 0) if use else None
                                       for a, b, use in zip(true_residual, residual, eligible)])
    return result


def select_steps(rows, success):
    eligible = [r["step"] for r in rows if .90 <= r["after"]["coverage"] <= .95]
    selected = sorted({eligible[index] for index in [0, len(eligible) // 2, len(eligible) - 1]}) if eligible else []
    result = [{"step": step, "stratum": "fine_alignment"} for step in selected]
    if success:
        assert rows[-1]["after"]["coverage"] > .95
        result.append({"step": rows[-1]["step"], "stratum": "completed_control"})
    return result


def checked_pixels(pixels, state):
    assert pixels.shape == (96, 96, 3) and pixels.dtype == np.uint8
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == state["image_sha256"], "Saved RGB mismatch; do not score altered pixels."
    return pixels.copy()


def checked_pose(env, state):
    assert list(env.unwrapped.block.position) == state["block_pos"]
    assert env.unwrapped.block.angle == state["block_angle"]
    assert list(env.unwrapped.agent.position) == state["pusher_pos"]


def replay_frames(seed, rows, steps):
    """Replay only recorded commands; no oracle-selected actions or model calls."""
    assert steps and all(isinstance(step, int) and 1 <= step <= len(rows) for step in steps)
    wanted = set(steps) | {step - 1 for step in steps}
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    frames = {}
    try:
        observation, _ = env.reset(seed=seed)
        initial_pixels = checked_pixels(observation["pixels"], rows[0]["before"])
        checked_pose(env, rows[0]["before"])
        if 0 in wanted:
            frames[0] = initial_pixels
        for row in rows[:max(steps)]:
            observation, _, _, _, _ = env.step(np.asarray(row["action"], dtype=np.float32))
            pixels = checked_pixels(observation["pixels"], row["after"])
            checked_pose(env, row["after"])
            if row["step"] in wanted:
                frames[row["step"]] = pixels
    finally:
        env.close()
    assert set(frames) == wanted
    return frames, max(steps)


def sample_pair(pixels, previous_pixels):
    cold = visual.VisualAlignmentEstimator()
    warm = visual.VisualAlignmentEstimator()
    cold_estimate = cold.estimate(pixels)
    warm.estimate(previous_pixels)
    return {"cold": cold_estimate, "previous_frame": warm.estimate(pixels)}


def discovery():
    source, validation = read(SOURCE), read(VALIDATION)
    assert source["baseline_complete"] and validation["passed"]
    episodes = source["episodes"]
    assert len(episodes) == 30 and [e["seed"] for e in episodes] == SEEDS
    loaded, bindings, cases = {}, {}, []
    for episode in episodes:
        path = Path(episode["source_directory"]) / episode["trace"]
        assert sha(path) == validation["source_trace_hashes"][path.as_posix()]
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)["steps"]
        assert len(rows) == episode["steps"] and [r["step"] for r in rows] == list(range(1, len(rows) + 1))
        assert rows[-1]["after"]["coverage"] == episode["final_coverage"]
        assert episode["success"] == (rows[-1]["after"]["coverage"] > .95)
        loaded[episode["seed"]] = rows
        bindings[path.as_posix()] = sha(path)
        for item in select_steps(rows, episode["success"]):
            index = item["step"] - 1
            previous_state = rows[index - 1]["after"] if index else rows[0]["before"]
            cases.append({"seed": episode["seed"], "scene_success": episode["success"], **item,
                          "source_trace": path.as_posix(), "image_sha256": rows[index]["after"]["image_sha256"],
                          "previous_image_sha256": previous_state["image_sha256"]})
    for path in [SOURCE, VALIDATION, Path("experiments/robotics_visual_monitor.py"), Path(__file__).relative_to(Path.cwd())]:
        bindings[path.as_posix()] = sha(path)
    return loaded, cases, bindings


def runtime_hashes():
    root = Path(gym_pusht.__file__).parent / "envs"
    return {name: sha(root / name) for name in ["pusht.py", "pymunk_override.py"]}


def protocol_document():
    _, cases, bindings = discovery()
    return {"scope": "Pose-precision feasibility screen; not a new controller or improvement evaluation.",
            "seeds": SEEDS, "methods": METHODS, "cases": cases, "fingerprints": bindings,
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            "renderer_fingerprints": runtime_hashes(), "goal": GOAL, "limits": LIMITS,
            "direction_deadbands": DEADBANDS, "image_size": 96,
            "selection": "Up to three90-95% frames per scene:first/middle/last temporal eligible indices; successful terminal>95% controls.",
            "primary": "cold stateless fit; previous-frame image warm start is sensitivity only, not full chronological reproduction.",
            "accuracy_target": "Engineering screen for an intended three-coordinate-unit pose correction with <=one-unit pose/vertex uncertainty; not a bound on actual pushing success.",
            "rgb_recovery": "CPU replay of recorded actions with exact pixel and physical-coordinate checks; no model inference or alternate future.",
            "runtime_input_contract": "Estimator receives RGB pixels only; known static silhouette/goal permitted. Saved dynamic truth is reconstruction/scoring only.",
            "ai_calls": 0, "policy_training": False, "corrective_actions": 0}


def check_output(output):
    path = Path(output).resolve()
    if path not in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}:
        raise ValueError("Use the dedicated pose-gate output or its -repeat directory.")


def prepare(output):
    check_output(output)
    document = protocol_document()
    path = output / "protocol.json"
    if path.exists():
        assert read(path) == document, "Frozen pose gate changed; do not retune or overwrite."
    else:
        assert not any((output / name).exists() for name in ["samples.json", "summary.json"])
        write(path, document)
    return document


def method_metrics(rows, method):
    scores = [r["scores"][method] for r in rows]
    valid = [s for s in scores if s["valid"]]
    directions = [value for s in valid for value in s["direction_agreement"] if value is not None]
    metrics = {"frames": len(rows), "valid_frames": len(valid),
               "valid_fraction": len(valid) / len(rows) if rows else 0.,
               "eligible_valid_direction_components": len(directions),
               "direction_agreement": sum(directions) / len(directions) if directions else None,
               "opposed_direction_components": sum(not value for value in directions),
               "per_scene_mean_errors": {}, "scene_equal_mean_errors": {},
               "direction_by_component": {},
               "frames_from_successful_scenes": sum(r["scene_success"] for r in rows),
               "frames_from_failed_scenes": sum(not r["scene_success"] for r in rows)}
    for index, name in enumerate(["x", "y", "rotation"]):
        eligible = sum(s["direction_eligible"][index] for s in scores)
        available = [s["direction_agreement"][index] for s in valid
                     if s["direction_agreement"][index] is not None]
        metrics["direction_by_component"][name] = {
            "eligible_frames_including_invalid": eligible, "available_frames": len(available),
            "agreement": sum(available) / len(available) if available else None,
            "opposed_frames": sum(not value for value in available), "unavailable_frames": eligible - len(available)}
    for name in ["position_error", "angle_error_degrees", "max_vertex_error"]:
        values = [s[name] for s in valid]
        metrics[f"{name}_mean"] = float(np.mean(values)) if values else None
        metrics[f"{name}_p95"] = float(np.quantile(values, .95)) if values else None
        metrics[f"{name}_max"] = max(values) if values else None
    for seed in sorted({r["seed"] for r in rows}):
        current = [r["scores"][method] for r in rows if r["seed"] == seed and r["scores"][method]["valid"]]
        metrics["per_scene_mean_errors"][str(seed)] = {name: float(np.mean([s[name] for s in current])) if current else None
                                                      for name in ["position_error", "angle_error_degrees", "max_vertex_error"]}
    for name in ["position_error", "angle_error_degrees", "max_vertex_error"]:
        means = [values[name] for values in metrics["per_scene_mean_errors"].values() if values[name] is not None]
        metrics["scene_equal_mean_errors"][name] = float(np.mean(means)) if means else None
    metrics["passed"] = bool(valid and metrics["valid_fraction"] >= LIMITS["minimum_valid_fraction"]
                             and metrics["position_error_p95"] <= LIMITS["position_error_p95"]
                             and metrics["max_vertex_error_p95"] <= LIMITS["max_vertex_error_p95"]
                             and metrics["angle_error_degrees_p95"] <= LIMITS["angle_error_degrees_p95"]
                             and metrics["direction_agreement"] is not None
                             and metrics["direction_agreement"] >= LIMITS["minimum_direction_agreement"]
                             and all(values["agreement"] is not None and values["agreement"] >= LIMITS["minimum_direction_agreement"]
                                     for values in metrics["direction_by_component"].values()))
    return metrics


def summarize(samples):
    strata = {name: {method: method_metrics([r for r in samples if r["stratum"] == name], method)
                      for method in METHODS} for name in ["fine_alignment", "completed_control"]}
    return {"complete": True, "samples": len(samples), "by_stratum": strata,
            "passed": strata["fine_alignment"]["cold"]["passed"], "limits": LIMITS,
            "decision": "A passing camera gate permits only a bounded correction pilot, not a claim of controller benefit.",
            "ai_calls": 0, "corrective_actions": 0,
            "limits_of_evidence": ["Sampled already-inspected discovery frames, not independent scenes or real-camera validation.",
                                   "Primary estimator is stateless; two-image warm-start sensitivity is not the original full history.",
                                   "Truth selects/scorers offline strata; a future controller cannot use it at runtime.",
                                   "Recorded-action RGB replay does not restore model observation/RNG history for a matched alternative future.",
                                   "Pose accuracy does not identify contact dynamics or prove correction efficacy."]}


def verify_sample(sample, case, rows):
    assert {k: sample[k] for k in case} == case
    state = rows[case["step"] - 1]["after"]
    assert set(sample["estimates"]) == set(sample["scores"]) == set(METHODS)
    for method in METHODS:
        assert sample["scores"][method] == score_pose_frame(sample["estimates"][method], state)
    assert math.isfinite(sample["fit_seconds"]) and sample["fit_seconds"] >= 0


def run(output):
    protocol = prepare(output)
    loaded, cases, _ = discovery()
    samples_path, summary_path = output / "samples.json", output / "summary.json"
    existing = read(samples_path) if samples_path.exists() else {"complete": False, "samples": []}
    samples = existing["samples"]
    completed_before = len(samples)
    assert len(samples) <= len(cases)
    for sample, case in zip(samples, cases):
        verify_sample(sample, case, loaded[case["seed"]])
    if summary_path.exists():
        assert existing["complete"] and len(samples) == len(cases) and read(summary_path) == summarize(samples)
        print("Completed pose gate verified; no images or fits repeated.")
        return read(summary_path)
    assert not existing["complete"] or len(samples) == len(cases)
    resource_path = output / f"resources-from-{completed_before:03d}.json"
    assert not resource_path.exists(), "Preserve earlier invocation resource evidence."
    cv2.setNumThreads(1)
    started, peak_rss = time.perf_counter(), psutil.Process().memory_info().rss
    frames, active_seed, replayed_movements = {}, None, 0
    for case in cases[len(samples):]:
        if active_seed != case["seed"]:
            steps = [item["step"] for item in cases if item["seed"] == case["seed"]]
            frames, replayed = replay_frames(case["seed"], loaded[case["seed"]], steps)
            replayed_movements += replayed
            active_seed = case["seed"]
        rows, index = loaded[case["seed"]], case["step"] - 1
        previous_pixels, pixels = frames[case["step"] - 1], frames[case["step"]]
        fit_started = time.perf_counter()
        estimates = sample_pair(pixels, previous_pixels)
        sample = {**case, "estimates": estimates,
                  "scores": {method: score_pose_frame(estimates[method], rows[index]["after"]) for method in METHODS},
                  "fit_seconds": time.perf_counter() - fit_started}
        verify_sample(sample, case, rows)
        samples.append(sample)
        write(samples_path, {"complete": len(samples) == len(cases), "samples": samples})
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        print(f"Pose gate{len(samples)}/{len(cases)}:scene{case['seed']},movement{case['step']},{case['stratum']}", flush=True)
    assert prepare(output) == protocol
    summary = summarize(samples)
    write(summary_path, summary)
    write(resource_path, {"wall_seconds": time.perf_counter() - started, "peak_observed_rss_bytes": peak_rss,
                          "frames_before": completed_before, "frames_complete": len(samples),
                          "replayed_movements": replayed_movements,
                          "ai_calls": 0, "corrective_actions": 0})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "run"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output_dir) if args.stage == "prepare" else run(args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ["cases", "fingerprints", "packages"]}, indent=2))


if __name__ == "__main__":
    main()
