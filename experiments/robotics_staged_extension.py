"""Locked staged 8/12/15 execution comparison; frozen H16 generation throughout.

The milestone is unchanged. Twelve actions are scheduled initially; three cached
actions are appended only after the new twelve-action checkpoint approves them.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

import robotics_generation_length as gen


adaptive, base = gen.adaptive, gen.base
OUTPUT = Path("results/robotics-staged-extension")
REUSED = Path("results/robotics-stateful-budget")
DEVELOPMENT_SCENES = list(range(90, 110))
FRESH_SCENES = list(range(110, 150))
SCENES = DEVELOPMENT_SCENES + FRESH_SCENES
METHODS = ["fixed8", "adaptive12", "staged15"]
STATISTICS = {"bootstrap_samples": 10000, "bootstrap_seed": 20261015,
              "unit": "paired initial scene; one prediction sample per method",
              "primary": "fresh forty scenes; inspected twenty and pooled sixty secondary",
              "timing": "fresh cohort primary; mixed-session development/pooled timing descriptive"}


def draw_for(scene):
    assert scene in SCENES
    return 30000 if scene in DEVELOPMENT_SCENES else 40000


def schedule():
    return [{"scene": scene, "draw": draw_for(scene), "method": method,
             "reused": scene in DEVELOPMENT_SCENES and method != "staged15"}
            for index, scene in enumerate(SCENES)
            for method in METHODS[index % 3:] + METHODS[:index % 3]]


def extension_after_twelve(features, commands):
    points = np.asarray(commands, dtype=float)
    return bool(points.shape == (15, 2) and np.isfinite(points).all()
                and adaptive.extension_candidate(features)
                and adaptive._smooth(points[8:15]))


class StagedExtensionController:
    def __init__(self, late_mode=None):
        assert late_mode is None or type(late_mode) is bool
        self.late_mode = late_mode  # Forced/disabled mechanics only; None when scored.
        self.guard = adaptive.AdaptiveLengthController()

    def update(self, features, age, pending, step, commands, done=False):
        if done:
            return None
        assert 1 <= age <= 15
        assert pending == (12 - age if age <= 12 else 15 - age)
        # Original pending counts, command window and state transitions through12.
        original = self.guard.update(features, age, pending, step, commands[:12])
        if original:
            return original
        if age == 12:
            retain = (extension_after_twelve(features, commands) if self.late_mode is None
                      else self.late_mode)
            return {"kind": "extend15" if retain else "replan12",
                    "reason": ("camera_and_suffix_eligible" if retain else "retain_twelve_action_interval")
                    if self.late_mode is None else "mechanics_override",
                    "discarded": 0, "appended": 3 if retain else 0,
                    "cached_discarded": 0 if retain else 3}
        return None


def reserve_suffix(policy):
    queue = policy._queues["action"]
    assert queue.maxlen == 15 and len(queue) == 14
    items = list(queue)
    queue.clear()
    queue.extend(items[:11])
    return items[11:]


def append_tail(policy, tail):
    queue = policy._queues["action"]
    assert queue.maxlen == 15 and not queue and len(tail) == 3
    queue.extend(tail)  # Original tensors, dtype and device; no new prediction.


def check_output(output):
    assert Path(output).resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}


def protocol_document():
    inherited = gen.protocol_document()
    old = base.read_json(REUSED / "protocol.json")
    for key in ["checkpoint", "packages", "model_sources"]:
        assert old[key] == inherited[key], "Reused controls have incompatible settings."
    assert old["reused_milestone_rules"] == adaptive.RULES
    fingerprints = {**inherited["fingerprints"],
                    Path(__file__).resolve().relative_to(Path.cwd()).as_posix(): base.sha(Path(__file__)),
                    (REUSED / "protocol.json").as_posix(): base.sha(REUSED / "protocol.json")}
    references = []
    for item in schedule():
        if not item["reused"]:
            continue
        path = REUSED / "episodes" / (gen.stem(item["scene"], item["draw"], item["method"]) + ".json")
        trial = base.read_json(path)
        trace = path.parent / trial["trace"]
        assert base.sha(trace) == trial["trace_sha256"]
        for target in [path, trace]:
            fingerprints[target.as_posix()] = base.sha(target)
        references.append({**item, "sidecar": path.as_posix(), "trace": trace.as_posix()})
    return {"scope": "180 compared attempts, 140 new and 40 reused; sixty scenes, one sample each",
            "milestone": gen.MILESTONE, "methods": METHODS, "schedule": schedule(),
            "development_scenes": DEVELOPMENT_SCENES, "fresh_scenes": FRESH_SCENES,
            "draw_offsets": {"development": 30000, "fresh": 40000},
            "generation": {"horizon": 16, "usable_future_commands": 15, "observations": 2,
                           "denoising_steps": 100, "training": False},
            "candidate": {"initial_scheduled": 12, "reserved": 3, "checkpoints": [8, 12],
                          "late_path_window": [8, 15], "persistent_budget": False,
                          "original_guard_and_eight_check_unchanged": True},
            "rules": adaptive.RULES, "max_movements": 300,
            "success": "true coverage strictly greater than .95", "runtime_truth_inputs": False,
            "statistics": STATISTICS, "reused_controls": references,
            "checkpoint": inherited["checkpoint"], "packages": inherited["packages"],
            "model_sources": inherited["model_sources"], "fingerprints": fingerprints,
            "smoke": {"scene": 0, "draw": 1234, "movements": 32,
                      "disabled_late_extension": "exact milestone reproduction across predictions",
                      "forced_late_extension": "cached commands13-15 execute without another prediction"},
            "timing": inherited["timing"]}


def verify_protocol(output):
    check_output(output)
    document = base.read_json(output / "protocol.json")
    assert document == protocol_document(), "Frozen inputs, rules, controls or settings changed."
    return document


def prepare(output):
    check_output(output)
    if (output / "protocol.json").exists():
        return verify_protocol(output)
    assert not any((output / name).exists() for name in ["smoke.json", "episodes", "episodes.json", "summary.json"])
    document = protocol_document()
    base.save_json(output / "protocol.json", document)
    return document


def configure_candidate(policy):
    assert policy.config.horizon == policy.diffusion.config.horizon == 16
    assert policy.config.n_obs_steps == policy.diffusion.config.n_obs_steps == 2
    for config in [policy.config, policy.diffusion.config]:
        config.n_action_steps = 15
    policy.reset()  # Attempt boundary only. Never reset between predictions.
    assert policy._queues["action"].maxlen == 15 and not policy._queues["action"]


def evaluate_episode(policy, scene, draw, method, directory, max_steps=300, late_mode=None):
    if method != "staged15":
        assert late_mode is None
        return gen.evaluate_episode(policy, scene, draw, method, directory, max_steps)
    configure_candidate(policy)
    assert policy.diffusion.num_inference_steps == 100
    np.random.seed(scene + draw)
    torch.manual_seed(scene + draw)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=max_steps)
    observation, _ = env.reset(seed=scene)
    initial = adaptive.physical_state(env, observation)
    controller = StagedExtensionController(late_mode)
    estimator, rolling = adaptive.VisualAlignmentEstimator(), adaptive.RollingVisualFeatures()
    versions = tuple(parameter._version for parameter in policy.parameters())
    rows, units, events, prediction_seconds = [], [], [], []
    age, monitor_seconds = 0, 0.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        tick = time.perf_counter()
        initial_features = rolling.update(estimator.estimate(observation["pixels"]), 0)
        monitor_seconds += time.perf_counter() - tick
        with gen.capture_generation(policy) as generated_shapes:
            for step in range(1, max_steps + 1):
                fresh = not policy._queues["action"]
                if fresh:
                    if age:
                        units.append(age)
                    age = 0
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                with torch.inference_mode():
                    command = policy.select_action(adaptive.observation_batch(observation, "cuda")).squeeze(0).cpu().numpy()
                if fresh:
                    torch.cuda.synchronize()
                    prediction_seconds.append(time.perf_counter() - tick)
                    commands = [command.tolist()] + [item.squeeze(0).detach().cpu().numpy().tolist()
                                                     for item in policy._queues["action"]]
                    assert np.asarray(commands).shape == (15, 2) and np.isfinite(commands).all()
                    tail = reserve_suffix(policy)
                age += 1
                observation, _, terminated, truncated, _ = env.step(command)
                pending = len(policy._queues["action"])
                row = {"step": step, "action": command.tolist(), "new_prediction": bool(fresh),
                       "plan_age": age, "pending_before": pending,
                       "after": adaptive.physical_state(env, observation)}
                if fresh:
                    row.update(predicted_commands=commands, prediction_seconds=prediction_seconds[-1],
                               reserved_suffix_count=3)
                if not (terminated or truncated):
                    tick = time.perf_counter()
                    features = rolling.update(estimator.estimate(observation["pixels"]), pending)
                    event = controller.update(features, age, pending, step, commands)
                    if event:
                        events.append({"step": step, "plan_age": age, **event, "features": features})
                        if event["discarded"]:
                            policy._queues["action"].clear()
                        if event.get("appended", 0):
                            append_tail(policy, tail)
                            tail = []
                    monitor_seconds += time.perf_counter() - tick
                    row["features"] = features
                row["pending_after"] = len(policy._queues["action"])
                rows.append(row)
                if step % 80 == 0:
                    print(f"Running {method}, scene {scene}: movement {step}, calls {len(prediction_seconds)}", flush=True)
                if terminated or truncated:
                    break
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        units.append(age)
        assert versions == tuple(parameter._version for parameter in policy.parameters())
        result = {"scene": scene, "draw": draw, "prediction_seed": scene + draw, "method": method,
                  "horizon": 16, "execution": 15, "late_mode": late_mode,
                  "success": bool(terminated), "truncated": bool(truncated), "steps": len(rows),
                  "model_queries": len(prediction_seconds), "initial": initial, "initial_features": initial_features,
                  "final_coverage": rows[-1]["after"]["coverage"], "wall_seconds": wall_seconds,
                  "monitor_seconds": monitor_seconds, "prediction_seconds": prediction_seconds,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "unit_lengths": units, "events": events, "generated_shapes": generated_shapes,
                  "trace": gen.stem(scene, draw, method) + ".trace.json.gz"}
        directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(directory / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"scene": scene, "draw": draw, "method": method, "initial": initial, "steps": rows}, stream, allow_nan=False)
        result["trace_sha256"] = base.sha(directory / result["trace"])
        base.save_json(directory / (gen.stem(scene, draw, method) + ".json"), result)
        return result
    finally:
        env.close()
        gen.configure(policy, "fixed8")


def verify_trial(trial, directory, allowed_scenes=SCENES, allowed_draws=(30000, 40000), max_steps=300, allow_force=False):
    scene, draw, method = trial["scene"], trial["draw"], trial["method"]
    assert scene in allowed_scenes and draw in allowed_draws and method in METHODS
    if method != "staged15":
        return gen.verify_trial(trial, directory, allowed_scenes, allowed_draws, max_steps)
    mode = trial["late_mode"]
    assert mode is None or (allow_force and type(mode) is bool)
    assert trial["prediction_seed"] == scene + draw and (trial["horizon"], trial["execution"]) == (16, 15)
    trace = gen.read_trace(trial, directory)
    assert (trace["scene"], trace["draw"], trace["method"]) == (scene, draw, method)
    assert trace["initial"] == trial["initial"]
    rows = trace["steps"]
    assert 1 <= len(rows) == trial["steps"] <= max_steps
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and len(rows) == max_steps)
    assert all(row["after"]["coverage"] <= .95 for row in rows[:-1])
    assert trial["wall_seconds"] >= trial["monitor_seconds"] >= 0 and trial["peak_gpu_allocated_bytes"] > 0
    assert all(math.isfinite(trial[key]) for key in ["wall_seconds", "monitor_seconds", "final_coverage"])
    assert trial["initial_features"] is not None
    controller = StagedExtensionController(mode)
    pending, age, units, seconds, events = 0, 0, [], [], []
    for index, row in enumerate(rows):
        assert row["step"] == index + 1 and row["new_prediction"] == (pending == 0)
        if pending == 0:
            if age:
                units.append(age)
            age, pending = 0, 12
            commands = row["predicted_commands"]
            assert np.asarray(commands).shape == (15, 2) and np.isfinite(commands).all()
            assert row["reserved_suffix_count"] == 3
            assert math.isfinite(row["prediction_seconds"]) and row["prediction_seconds"] > 0
            seconds.append(row["prediction_seconds"])
        else:
            assert "predicted_commands" not in row and "prediction_seconds" not in row and "reserved_suffix_count" not in row
        age, pending = age + 1, pending - 1
        assert row["plan_age"] == age and row["pending_before"] == pending
        assert row["action"] == commands[age - 1]
        if index < len(rows) - 1:
            features = row["features"]
            assert features["pending_actions"] == pending
            event = controller.update(features, age, pending, index + 1, commands)
            if event:
                events.append({"step": index + 1, "plan_age": age, **event, "features": features})
                pending = pending - event["discarded"] + event.get("appended", 0)
        else:
            assert "features" not in row
        assert row["pending_after"] == pending
    units.append(age)
    assert trial["events"] == events and trial["unit_lengths"] == units
    assert trial["model_queries"] == len(units) == len(seconds) and sum(units) == len(rows)
    assert trial["prediction_seconds"] == seconds and sum(seconds) <= trial["wall_seconds"]
    assert trial["generated_shapes"] == [[1, 16, 2]] * len(seconds)
    return trace


def collect(policy, scene, draw, method, directory, max_steps=300, late_mode=None):
    path = directory / (gen.stem(scene, draw, method) + ".json")
    if path.exists():
        trial = base.read_json(path)
    else:
        assert not path.with_suffix(".trace.json.gz").exists() and not path.with_suffix(".json.tmp").exists(), "Inspect partial/orphan output."
        trial = evaluate_episode(policy, scene, draw, method, directory, max_steps, late_mode)
    assert (trial["scene"], trial["draw"], trial["method"]) == (scene, draw, method)
    if method == "staged15":
        assert trial["late_mode"] is late_mode
    verify_trial(trial, directory, [scene], [draw], max_steps, late_mode is not None)
    return trial


def common_prefix(candidate, reference, candidate_dir, reference_dir):
    assert adaptive.states_match(candidate["initial"], reference["initial"])
    actual = gen.read_trace(candidate, candidate_dir)["steps"]
    expected = gen.read_trace(reference, reference_dir)["steps"]
    late = [event["step"] for event in candidate["events"] if event["kind"] == "extend15"]
    limit = late[0] if late else len(actual)
    assert len(expected) >= limit
    for left, right in zip(actual[:limit], expected[:limit]):
        assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
        for key in ["step", "new_prediction", "plan_age", "pending_before"]:
            assert left[key] == right[key]
        if "features" in right:
            assert left["features"] == right["features"]
    if not late:
        assert len(actual) == len(expected) and candidate["success"] == reference["success"]
    return {"scene": candidate["scene"], "matched_through_movement": limit,
            "first_late_extension": late[0] if late else None}


def verify_smoke(output):
    saved = base.read_json(output / "smoke.json")
    assert saved["passed"] and len(saved["trials"]) == 3
    for trial in saved["trials"]:
        assert base.read_json(output / "smoke" / (gen.stem(0, 1234, trial["method"]) + ".json")) == trial
        verify_trial(trial, output / "smoke", [0], [1234], 32)
    milestone = next(row for row in saved["trials"] if row["method"] == "adaptive12")
    normal = next(row for row in saved["trials"] if row["method"] == "staged15")
    assert saved["normal_prefix"] == common_prefix(normal, milestone, output / "smoke", output / "smoke")
    for label, mode in [("disabled", False), ("forced", True)]:
        trial = saved[label]
        assert base.read_json(output / label / (gen.stem(0, 1234, "staged15") + ".json")) == trial
        assert trial["late_mode"] is mode
        verify_trial(trial, output / label, [0], [1234], 32, True)
    disabled = saved["disabled"]
    assert common_prefix(disabled, milestone, output / "disabled", output / "smoke")["matched_through_movement"] == 32
    assert disabled["unit_lengths"] == milestone["unit_lengths"]
    assert [{k: v for k, v in e.items()} for e in disabled["events"] if e["kind"] != "replan12"] == milestone["events"]
    forced = saved["forced"]
    assert forced["unit_lengths"][0] == 15
    rows = gen.read_trace(forced, output / "forced")["steps"]
    assert rows[11]["pending_before"] == 0 and rows[11]["pending_after"] == 3
    assert all(not row["new_prediction"] for row in rows[1:15]) and rows[15]["new_prediction"]
    assert [row["action"] for row in rows[:15]] == rows[0]["predicted_commands"]
    return saved


def smoke(output):
    verify_protocol(output)
    if (output / "smoke.json").exists():
        return verify_smoke(output)
    assert torch.cuda.is_available(), "GPU required; no silent CPU fallback."
    policy, _ = adaptive.load_policy(base.MODEL)
    trials = [collect(policy, 0, 1234, method, output / "smoke", 32) for method in METHODS]
    modes = {label: collect(policy, 0, 1234, "staged15", output / label, 32, mode)
             for label, mode in [("disabled", False), ("forced", True)]}
    normal = next(row for row in trials if row["method"] == "staged15")
    milestone = next(row for row in trials if row["method"] == "adaptive12")
    base.save_json(output / "smoke.json", {"passed": True, "trials": trials, **modes,
                   "normal_prefix": common_prefix(normal, milestone, output / "smoke", output / "smoke")})
    return verify_smoke(output)


def metric_table(trials, scenes):
    expected = {(scene, draw_for(scene), method) for scene in scenes for method in METHODS}
    assert len(trials) == len(expected) and {(r["scene"], r["draw"], r["method"]) for r in trials} == expected
    return {method: {"attempts": len(rows), "successes": sum(r["success"] for r in rows),
                     **{f"mean_{key}": float(np.mean([r[key] for r in rows]))
                        for key in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]}}
            for method in METHODS for rows in [[r for r in trials if r["method"] == method]]}


def paired(trials, reference, scenes):
    lookup = {(r["scene"], r["method"]): r for r in trials}
    assert len(lookup) == len(trials) and reference in ["fixed8", "adaptive12"]
    rng = np.random.default_rng(STATISTICS["bootstrap_seed"])
    indices = rng.integers(0, len(scenes), size=(STATISTICS["bootstrap_samples"], len(scenes)))
    differences = {}
    for key in ["success", "final_coverage", "model_queries", "wall_seconds"]:
        values = np.asarray([float(lookup[s, "staged15"][key]) - float(lookup[s, reference][key]) for s in scenes])
        differences[key] = {"mean_difference": float(values.mean()),
                            "scene_pair_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    gained = [s for s in scenes if lookup[s, "staged15"]["success"] and not lookup[s, reference]["success"]]
    lost = [s for s in scenes if lookup[s, reference]["success"] and not lookup[s, "staged15"]["success"]]
    n = len(gained) + len(lost)
    probability = min(1., 2 * sum(math.comb(n, k) for k in range(min(len(gained), len(lost)) + 1)) / 2 ** n) if n else 1.
    return {"reference": reference, "method": "staged15", "differences_method_minus_reference": differences,
            "gained_successes": gained, "lost_successes": lost, "exact_discordant_two_sided_p": probability}


def summarize(output):
    verify_protocol(output)
    verify_smoke(output)
    document = base.read_json(output / "episodes.json")
    assert document["complete"] and len(document["records"]) == 180
    records = document["records"]
    trials = []
    lookup = {}
    kinds, units, prefixes = Counter(), Counter(), []
    for item, identity in zip(records, schedule()):
        trial, directory = item["trial"], Path(item["directory"])
        assert {key: trial[key] for key in ["scene", "draw", "method"]} == {key: identity[key] for key in ["scene", "draw", "method"]}
        assert item["reused"] == identity["reused"]
        expected_dir = REUSED / "episodes" if identity["reused"] else output / "episodes"
        assert directory == expected_dir
        assert base.read_json(directory / (gen.stem(trial["scene"], trial["draw"], trial["method"]) + ".json")) == trial
        verify_trial(trial, directory)
        trials.append(trial)
        lookup[trial["scene"], trial["method"]] = (trial, directory)
        if trial["method"] == "staged15":
            kinds.update(event["kind"] for event in trial["events"])
            units.update(trial["unit_lengths"])
    for scene in SCENES:
        candidate, candidate_dir = lookup[scene, "staged15"]
        reference, reference_dir = lookup[scene, "adaptive12"]
        prefixes.append(common_prefix(candidate, reference, candidate_dir, reference_dir))
        baseline, baseline_dir = lookup[scene, "fixed8"]
        actual = gen.read_trace(candidate, candidate_dir)["steps"]
        expected = gen.read_trace(baseline, baseline_dir)["steps"]
        limit = min(8, candidate["unit_lengths"][0], reference["unit_lengths"][0])
        assert actual[0]["predicted_commands"][:8] == expected[0]["predicted_commands"][:8]
        for a, b in zip(actual[:limit], expected[:limit]):
            assert a["action"] == b["action"] and adaptive.states_match(a["after"], b["after"])
    cohorts = {}
    for label, scenes in [("fresh_primary", FRESH_SCENES), ("development", DEVELOPMENT_SCENES), ("pooled_secondary", SCENES)]:
        selected = [trial for trial in trials if trial["scene"] in scenes]
        cohorts[label] = {"scenes": scenes, "methods": metric_table(selected, scenes),
                          "comparisons": [paired(selected, ref, scenes) for ref in ["fixed8", "adaptive12"]]}
    result = {"complete": True, "new_attempts": 140, "reused_attempts": 40, "cohorts": cohorts,
              "candidate_event_counts": dict(kinds), "candidate_realized_unit_lengths": {str(k): v for k, v in units.items()},
              "milestone_equivalent_prefixes": prefixes,
              "limits": "One sample per scene; inspected cohort and mixed-session timing secondary; no deployment guarantee"}
    path = output / "summary.json"
    if path.exists():
        assert base.read_json(path) == result, "Completed results changed."
    else:
        base.save_json(path, result)
    for label in cohorts:
        print(json.dumps({label: cohorts[label]["methods"]}, indent=2), flush=True)
    return result


def run(output):
    verify_protocol(output)
    if (output / "summary.json").exists():
        return summarize(output)
    smoke(output)
    records, policy = [], None
    for identity in schedule():
        scene, draw, method = identity["scene"], identity["draw"], identity["method"]
        directory = REUSED / "episodes" if identity["reused"] else output / "episodes"
        path = directory / (gen.stem(scene, draw, method) + ".json")
        if not identity["reused"] and not path.exists() and policy is None:
            policy, _ = adaptive.load_policy(base.MODEL)
        if identity["reused"]:
            trial = base.read_json(path)
            verify_trial(trial, directory)
        else:
            trial = collect(policy, scene, draw, method, directory)
        records.append({"reused": identity["reused"], "directory": directory.as_posix(), "trial": trial})
        base.save_json(output / "episodes.json", {"complete": len(records) == 180, "records": records})
        new_done = sum(not record["reused"] for record in records)
        print(f"Progress {len(records)}/180, new {new_done}/140: {method} scene {scene}, "
              f"success={trial['success']}, calls={trial['model_queries']}, time={trial['wall_seconds']:.1f}s", flush=True)
    return summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "run", "summarize"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    check_output(args.output_dir)
    if args.stage in ["smoke", "run"]:
        prepare(args.output_dir)
    tick = time.perf_counter()
    with adaptive.MemorySampler() as sampler:
        globals()[args.stage](args.output_dir)
    resource = args.output_dir / f"{args.stage}-resources.json"
    if not resource.exists():
        base.save_json(resource, {"wall_seconds": time.perf_counter() - tick,
                                 "peak_process_rss_bytes": sampler.peak_rss,
                                 "minimum_system_available_bytes": sampler.minimum_available})


if __name__ == "__main__":
    main()
