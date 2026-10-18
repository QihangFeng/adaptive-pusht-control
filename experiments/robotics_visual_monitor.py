"""Small CPU image monitor for the fixed PushT task, not a failure detector.

Runtime inputs are the 96x96 RGB policy image and pending-action count only.
The two rectangles and fixed target are task specification copied from
gym-pusht 0.1.5 ``PushTEnv.add_tee`` / ``_setup`` (huggingface/gym-pusht).
No dynamic simulator block pose, contact or coverage enters the estimator.
Simulator coverage is accessed ONLY by the offline replay validation routine.
"""

from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from shapely.geometry import Polygon
from shapely.ops import unary_union


IMAGE_SIZE = 96
SCALE = IMAGE_SIZE / 512.0
# cv2.resize maps destination pixel centers to (x + .5) / SCALE - .5.
# Therefore a world coordinate maps to SCALE * world - (1 - SCALE) / 2.
PIXEL_CENTER_OFFSET = -(1 - SCALE) / 2
GOAL_POSE = (256.0 * SCALE + PIXEL_CENTER_OFFSET,
             256.0 * SCALE + PIXEL_CENTER_OFFSET, np.pi / 4)
RECTANGLES = (
    np.array([[-60, 0], [60, 0], [60, 30], [-60, 30]], dtype=float) * SCALE,
    np.array([[-15, 30], [15, 30], [15, 120], [-15, 120]], dtype=float) * SCALE,
)
T_CENTROID = np.array([0.0, (3600 * 15 + 2700 * 75) / 6300]) * SCALE
FEATURE_NAMES = (
    "coverage", "warning_count", "drawdown", "recovery", "stagnation",
    "position_error", "orientation_error", "pending_actions",
)
VISUAL_RULES = {
    "warning_window": 5, "drawdown_window": 8, "progress_window": 24,
    "warning_drop": 0.015, "maximum_fit_rms_pixels": 0.75,
    "minimum_object_pixels": 120,
}
MONITOR_RULES = VISUAL_RULES


def _rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def _geometry(pose):
    xy, angle = np.asarray(pose[:2]), pose[2]
    return unary_union([Polygon(vertices @ _rotation(angle).T + xy)
                        for vertices in RECTANGLES])


GOAL_GEOMETRY = _geometry(GOAL_POSE)


def _rectangle_distance(points, lower, upper):
    midpoint = (np.asarray(lower) + np.asarray(upper)) / 2
    halfsize = (np.asarray(upper) - np.asarray(lower)) / 2
    delta = np.abs(points - midpoint) - halfsize
    return np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)


def _tee_distance(points, pose):
    local = (points - pose[:2]) @ _rotation(pose[2])
    return np.minimum(_rectangle_distance(local, (-60 * SCALE, 0), (60 * SCALE, 30 * SCALE)),
                      _rectangle_distance(local, (-15 * SCALE, 30 * SCALE), (15 * SCALE, 120 * SCALE)))


class VisualAlignmentEstimator:
    """Fit the known rigid silhouette to color-segmented camera pixels.

    The baseline renderer has an approximately 0.375-pixel outline after resize.
    Boundary fitting allows that border; the coverage calculation uses the
    unexpanded mathematical task shape, avoiding an inflated gray-mask overlap.
    Failed/ambiguous visual fits return quality=False; callers must continue.
    """

    def __init__(self):
        self.previous_pose = None

    def estimate(self, pixels):
        image = np.asarray(pixels)
        if image.shape != (96, 96, 3) or image.dtype != np.uint8:
            raise ValueError("Expected the policy's 96x96 uint8 RGB image.")
        rgb = image.astype(np.int16)
        red, green, blue = rgb.transpose(2, 0, 1)
        gray = ((green - red > 7) & (blue - green > 7)
                & (blue - green < 45) & (red > 65) & (red < 220)).astype(np.uint8)
        occluded = ((blue - green > 50) & (blue - red > 70)).astype(np.uint8)
        count = int(gray.sum())
        if count < VISUAL_RULES["minimum_object_pixels"]:
            return self._missing(count)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        near_blue = cv2.dilate(occluded, np.ones((3, 3), dtype=np.uint8))
        contour = contour[near_blue[contour[:, 1].astype(int), contour[:, 0].astype(int)] == 0]
        if len(contour) < 20:
            return self._missing(count)
        y, x = np.nonzero(gray)
        points = np.column_stack((x, y)).astype(float)
        center = points.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov(points.T))
        direction = vectors[:, -1]
        angle = float(np.arctan2(direction[1], direction[0]) - np.pi / 2)
        guesses = [angle + k * np.pi / 2 for k in range(4)]
        if self.previous_pose is not None:
            guesses.insert(0, self.previous_pose[2])
        candidates = []
        for candidate in guesses:
            origin = center - _rotation(candidate) @ T_CENTROID
            initial = np.array([*origin, candidate])
            # Quantized pixel centers lie inside the 0.375-pixel renderer border.
            fit = least_squares(lambda pose: _tee_distance(contour, pose) - 0.20,
                                initial, max_nfev=30, diff_step=1e-4,
                                bounds=([origin[0] - 5, origin[1] - 5, candidate - 0.65],
                                        [origin[0] + 5, origin[1] + 5, candidate + 0.65]))
            rms = float(np.sqrt(np.mean(fit.fun ** 2)))
            candidates.append((rms, fit.x))
        rms, pose = min(candidates, key=lambda item: item[0])
        quality = rms <= VISUAL_RULES["maximum_fit_rms_pixels"]
        if not quality:
            return self._missing(count, rms)
        self.previous_pose = pose.copy()
        coverage = float(_geometry(pose).intersection(GOAL_GEOMETRY).area / GOAL_GEOMETRY.area)
        angle_error = abs((float(pose[2]) - GOAL_POSE[2] + np.pi) % (2 * np.pi) - np.pi)
        return {"quality": True, "coverage": coverage,
                "position_error": float(np.linalg.norm(pose[:2] - np.asarray(GOAL_POSE[:2])) / 96),
                "orientation_error": angle_error / np.pi,
                "pose_image": [float(v) for v in pose], "fit_rms_pixels": rms,
                "object_pixels": count}

    @staticmethod
    def _missing(count, rms=None):
        return {"quality": False, "coverage": None, "position_error": None,
                "orientation_error": None, "pose_image": None,
                "fit_rms_pixels": rms, "object_pixels": count}


class RollingVisualFeatures:
    """Past-only windows; warning flags are descriptive, not failure labels."""

    def __init__(self):
        self.history = deque(maxlen=VISUAL_RULES["progress_window"] + 1)

    def update(self, estimate, pending_actions):
        if not isinstance(pending_actions, (int, np.integer)) or pending_actions < 0:
            raise ValueError("pending_actions must be a nonnegative integer.")
        if not estimate["quality"]:
            self.history.clear()
            return {"quality": False, **{name: None for name in FEATURE_NAMES},
                    "pending_actions": int(pending_actions), "history_steps": 0,
                    "recent_best_gain": None, "recent_peak": None}
        coverage = float(estimate["coverage"])
        self.history.append(coverage)
        values = np.asarray(self.history)
        changes = np.diff(values)
        recent = values[-VISUAL_RULES["drawdown_window"]:]
        recent_best_gain = float(max(values) - values[0])
        # These are current signals only; recovery/stagnation is not a label.
        return {"quality": True, "coverage": coverage,
                "warning_count": int(np.sum(changes[-VISUAL_RULES["warning_window"]:] < -VISUAL_RULES["warning_drop"])),
                "drawdown": float(max(recent) - coverage),
                "recovery": float(changes[-3:].sum()) if len(changes) else 0.0,
                "stagnation": recent_best_gain,
                "position_error": float(estimate["position_error"]),
                "orientation_error": float(estimate["orientation_error"]),
                "pending_actions": int(pending_actions),
                "history_steps": max(0, len(values) - 1),
                "recent_best_gain": recent_best_gain, "recent_peak": float(max(recent))}


def validate_saved_collection(output=Path("results/robotics-benefit-pilot"), limit=None):
    """Replay saved actions with CPU physics only; truth is validation-only."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    output = Path(output)
    destination = output / "visual-validation.json"
    feature_destination = output / "visual-features.json.gz"
    if destination.exists() or feature_destination.exists():
        raise FileExistsError("Do not overwrite completed visual validation.")
    source = Path("results/robotics-failure-patterns/summary.json")
    episodes = json.loads(source.read_text(encoding="utf-8"))["episodes"]
    if limit is not None:
        episodes = episodes[:limit]
    all_errors, all_change_errors, timings, records = [], [], [], []
    hashes = {source.as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()}
    frames, exact = 0, 0
    for episode in episodes:
        trace_path = Path(episode["source_directory"]) / episode["trace"]
        hashes[trace_path.as_posix()] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)["steps"]
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
        observation, _ = env.reset(seed=episode["seed"])
        estimator, rolling = VisualAlignmentEstimator(), RollingVisualFeatures()
        initial = estimator.estimate(observation["pixels"])
        rolling.update(initial, 0)
        previous_estimated = initial["coverage"]
        previous_truth = float(rows[0]["before"]["coverage"])
        features, errors = [], []
        invalid = 0
        for row in rows:
            observation, _, _, _, info = env.step(np.asarray(row["action"], dtype=np.float32))
            digest = hashlib.sha256(observation["pixels"].tobytes()).hexdigest()
            exact += digest == row["after"]["image_sha256"]
            frames += 1
            start = time.perf_counter()
            estimate = estimator.estimate(observation["pixels"])
            item = rolling.update(estimate, int(row["pending_actions"]))
            timings.append(time.perf_counter() - start)
            truth = float(info["coverage"])
            if estimate["quality"]:
                error = abs(estimate["coverage"] - truth)
                errors.append(error)
                all_errors.append(error)
                if previous_estimated is not None:
                    all_change_errors.append(abs((estimate["coverage"] - previous_estimated)
                                                - (truth - previous_truth)))
            else:
                invalid += 1
            features.append({"step": row["step"], **item})
            previous_estimated, previous_truth = estimate["coverage"], truth
        env.close()
        records.append({"seed": episode["seed"], "steps": len(rows),
                        "invalid_frames": invalid, "mean_absolute_error": float(np.mean(errors)) if errors else None,
                        "features": features})
        print(f"Visual replay seed {episode['seed']}: {len(rows)} steps, invalid={invalid}, MAE={np.mean(errors):.4f}", flush=True)
    summary = {"episodes": len(records), "frames": frames, "exact_image_hash_matches": exact,
               "input_contract": "96x96 RGB image only; known static task silhouette and goal; pending-action count",
               "runtime_simulator_truth_inputs": False, "ai_calls": 0,
               "rules": VISUAL_RULES, "feature_names": list(FEATURE_NAMES), "source_trace_hashes": hashes,
               "coverage_mean_absolute_error": float(np.mean(all_errors)),
               "coverage_error_p95": float(np.quantile(all_errors, .95)),
               "coverage_error_max": float(max(all_errors)),
               "coverage_change_error_p95": float(np.quantile(all_change_errors, .95)),
               "coverage_change_error_max": float(max(all_change_errors)),
               "monitor_seconds_mean": float(np.mean(timings)),
               "monitor_seconds_p95": float(np.quantile(timings, .95)),
               "invalid_frame_fraction": sum(r["invalid_frames"] for r in records) / frames,
               "validation_limits": {"coverage_mean_absolute_error": .015,
                                     "coverage_error_p95": .03, "change_error_p95": .015,
                                     "invalid_frame_fraction": .02},
               "per_episode": [{k: v for k, v in r.items() if k != "features"} for r in records]}
    summary["passed"] = (exact == frames and summary["coverage_mean_absolute_error"] <= .015
                         and summary["coverage_error_p95"] <= .03
                         and summary["coverage_change_error_p95"] <= .015
                         and summary["invalid_frame_fraction"] <= .02)
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Source evidence changed."
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(feature_destination, "wt", encoding="utf-8") as stream:
        json.dump({"feature_names": list(FEATURE_NAMES), "rules": MONITOR_RULES,
                   "episodes": [{"seed": r["seed"], "steps": r["steps"], "features": r["features"]}
                                for r in records]}, stream, allow_nan=False)
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def render_visual_example(output=Path("results/robotics-benefit-pilot")):
    """One illustrative saved frame, CPU action replay only; no inference."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    with gzip.open("results/robotics-baseline-check/seed-0.trace.json.gz", "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["steps"]
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    observation, _ = env.reset(seed=0)
    for row in rows:
        observation, _, _, _, info = env.step(np.asarray(row["action"], dtype=np.float32))
    image = observation["pixels"]
    assert hashlib.sha256(image.tobytes()).hexdigest() == rows[-1]["after"]["image_sha256"]
    estimate = VisualAlignmentEstimator().estimate(image)
    env.close()
    canvas = np.full((430, 800, 3), 255, np.uint8)
    raw = cv2.resize(image, (360, 360), interpolation=cv2.INTER_NEAREST)
    overlay = raw.copy()
    for pose, color in [(GOAL_POSE, (40, 160, 40)), (estimate["pose_image"], (220, 40, 40))]:
        if pose is None:
            continue
        for rectangle in RECTANGLES:
            vertices = rectangle @ _rotation(pose[2]).T + np.asarray(pose[:2])
            cv2.polylines(overlay, [np.rint(vertices * 360 / 96).astype(np.int32)], True, color, 2)
    canvas[35:395, 20:380] = raw
    canvas[35:395, 420:780] = overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Policy image: seed 0, step {rows[-1]['step']}", (20, 24), font, .55, (0, 0, 0), 1)
    cv2.putText(canvas, "Red: fitted T; green: fixed goal", (420, 24), font, .55, (0, 0, 0), 1)
    cv2.putText(canvas, f"Diagnostic truth: {info['coverage']:.3f} | image estimate: {estimate['coverage']:.3f}",
                (20, 418), font, .55, (0, 0, 0), 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "visual-example.png"
    if destination.exists():
        raise FileExistsError("Do not overwrite an existing example.")
    assert cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return destination


def _near_target_errors(rows, features, threshold=.75):
    """Truth stratification is offline-only; skip changes across invalid frames."""
    if len(rows) != len(features):
        raise ValueError("Saved visual features must match the original trace length.")
    errors, changes = [], []
    eligible = 0
    previous = None
    for row, feature in zip(rows, features):
        if row["step"] != feature["step"]:
            raise ValueError("Feature step indices differ from original recordings.")
        truth = float(row["after"]["coverage"])
        eligible += truth >= threshold
        if not feature["quality"]:
            previous = None
            continue
        estimated = float(feature["coverage"])
        if truth >= threshold:
            errors.append(abs(estimated - truth))
            if previous is not None and previous[1] >= threshold:
                changes.append(abs((estimated - previous[0]) - (truth - previous[1])))
        previous = estimated, truth
    return errors, changes, eligible


def validate_near_target(output=Path("results/robotics-benefit-pilot")):
    """Audit existing camera estimates; no rendering, simulation or inference."""
    output = Path(output)
    destination = output / "near-target-validation.json"
    if destination.exists():
        raise FileExistsError("Do not overwrite completed near-target validation.")
    feature_path = output / "visual-features.json.gz"
    validation_path = output / "visual-validation.json"
    source_path = Path("results/robotics-failure-patterns/summary.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["passed"], "Run the full visual gate first."
    fingerprints = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (feature_path, validation_path, source_path)}
    with gzip.open(feature_path, "rt", encoding="utf-8") as stream:
        feature_records = json.load(stream)["episodes"]
    episodes = json.loads(source_path.read_text(encoding="utf-8"))["episodes"]
    if [e["seed"] for e in episodes] != [e["seed"] for e in feature_records]:
        raise ValueError("Visual audit must cover the same discovery seeds.")
    errors, changes, eligible, per_episode = [], [], 0, []
    for episode, features in zip(episodes, feature_records):
        trace_path = Path(episode["source_directory"]) / episode["trace"]
        digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        assert digest == validation["source_trace_hashes"][trace_path.as_posix()], "Original trace changed."
        fingerprints[trace_path.as_posix()] = digest
        with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)["steps"]
        current_errors, current_changes, current_eligible = _near_target_errors(rows, features["features"])
        errors.extend(current_errors)
        changes.extend(current_changes)
        eligible += current_eligible
        per_episode.append({"seed": episode["seed"], "eligible_frames": current_eligible,
                            "valid_frames": len(current_errors), "valid_change_pairs": len(current_changes)})
    summary = {
        "truth_coverage_threshold": .75,
        "purpose": "offline near-target camera-accuracy audit, not controller input or evaluation",
        "rendered_frames": 0, "ai_calls": 0, "episodes": len(episodes),
        "eligible_frames": eligible, "valid_frames": len(errors), "valid_change_pairs": len(changes),
        "coverage_mean_absolute_error": float(np.mean(errors)) if errors else None,
        "coverage_error_p95": float(np.quantile(errors, .95)) if errors else None,
        "coverage_error_max": float(max(errors)) if errors else None,
        "coverage_change_error_p95": float(np.quantile(changes, .95)) if changes else None,
        "coverage_change_error_max": float(max(changes)) if changes else None,
        "validation_limits": {"coverage_mean_absolute_error": .015,
                              "coverage_error_p95": .03, "coverage_change_error_p95": .015},
        "source_fingerprints": fingerprints, "per_episode": per_episode,
    }
    summary["passed"] = bool(errors and changes
                             and summary["coverage_mean_absolute_error"] <= .015
                             and summary["coverage_error_p95"] <= .03
                             and summary["coverage_change_error_p95"] <= .015)
    for path, expected in fingerprints.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, "Audit input changed."
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/robotics-benefit-pilot"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--near-target", action="store_true")
    args = parser.parse_args()
    if args.near_target:
        result = validate_near_target(args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in {"per_episode", "source_fingerprints"}}, indent=2))
    elif args.example:
        print(render_visual_example(args.output))
    else:
        result = validate_saved_collection(args.output, args.limit)
        print(json.dumps({k: v for k, v in result.items() if k not in {"per_episode", "source_trace_hashes"}}, indent=2))
