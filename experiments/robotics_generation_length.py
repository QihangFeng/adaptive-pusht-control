"""Zero-shot generated-length screen, separate from adaptive execution length.

Eight inspected scenes, two new prediction seeds each, four conditions. No
training or changed milestone rule; only the new arm's generated horizon changes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from lerobot.policies.diffusion import configuration_diffusion, modeling_diffusion

import robotics_adaptive_length as adaptive


base = adaptive.base
OUTPUT = Path("results/robotics-generation-length")
SCENES = list(range(8))
DRAWS = [10000, 20000]
CONDITIONS = {
    "h8_e4": {"horizon": 8, "execution": 4},
    "h16_e4": {"horizon": 16, "execution": 4},
    "fixed8": {"horizon": 16, "execution": 8},
    "adaptive12": {"horizon": 16, "execution": 12},
}
MILESTONE = "0f4044966646df4cc38b0373fa52c0b733506e15"
STATISTICS = {"unit": "initial scene, averaging both prediction samples first",
              "bootstrap_samples": 10000, "bootstrap_seed": 20261014,
              "interpretation": "Development-only exploratory intervals; eight independent scenes, not 16."}


def spec(method):
    return CONDITIONS[method]


def prediction_seed(scene, draw):
    return scene + draw


def stem(scene, draw, method):
    return f"scene-{scene}-draw-{draw}-{method}"


def schedule():
    methods = list(CONDITIONS)
    return [{"scene": scene, "draw": draw, "method": method}
            for scene in SCENES for draw in DRAWS
            for method in methods[(2 * scene + DRAWS.index(draw)) % 4:]
            + methods[:(2 * scene + DRAWS.index(draw)) % 4]]


def check_output(output):
    assert Path(output).resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}, \
        "Use the dedicated generated-length directory or its -repeat directory."


def configure(policy, method):
    candidate = spec(method)
    horizon, execution = candidate["horizon"], candidate["execution"]
    assert policy.config.n_obs_steps == policy.diffusion.config.n_obs_steps == 2
    assert horizon % (2 ** len(policy.config.down_dims)) == 0
    assert 1 <= execution <= horizon - policy.config.n_obs_steps + 1
    for config in [policy.config, policy.diffusion.config]:
        config.horizon, config.n_action_steps = horizon, execution
    policy.reset()  # Once at the start of an attempt, never at each prediction.
    assert policy._queues["action"].maxlen == execution
    assert not policy._queues["action"]


@contextmanager
def capture_generation(policy):
    original = policy.diffusion.conditional_sample
    shapes = []

    def captured(*args, **kwargs):
        value = original(*args, **kwargs)
        shapes.append(list(value.shape))
        assert list(value.shape) == [1, policy.diffusion.config.horizon, 2]
        assert bool(torch.isfinite(value).all())
        return value

    with patch.object(policy.diffusion, "conditional_sample", captured):
        yield shapes


def fingerprints():
    values = adaptive.fingerprints()
    for path in [Path(__file__), adaptive.OUTPUT / "prefix-smoke.json"]:
        values[path.resolve().relative_to(Path.cwd()).as_posix()] = base.sha(path)
    return values


def model_sources():
    return {module.__name__: base.sha(Path(inspect.getfile(module)))
            for module in [configuration_diffusion, modeling_diffusion]}


def protocol_document():
    packages = {name: importlib.metadata.version(name) for name in
                [*adaptive.PACKAGES, "opencv-python", "gymnasium", "pymunk", "pygame", "psutil"]}
    return {"scope": "64 new development attempts on eight inspected scenes; not generalization proof",
            "milestone": MILESTONE, "scenes": SCENES, "draw_offsets": DRAWS,
            "conditions": CONDITIONS, "schedule": schedule(), "new_attempts": 64,
            "prediction_seed": "scene plus draw offset; same seeds do not imply identical samples across horizons",
            "matched_contrast": "h8_e4 minus h16_e4; other contrasts also change feedback interval",
            "max_total_actions": 300, "success": "true coverage strictly greater than .95",
            "rules": adaptive.RULES, "statistics": STATISTICS,
            "checkpoint": {"model_id": base.MODEL_ID, "revision": base.MODEL_REVISION,
                           "weights_sha256": base.WEIGHTS_SHA256,
                           "config_sha256": base.sha(base.MODEL / "config.json")},
            "training": False, "n_obs_steps": 2, "denoising_steps": 100,
            "smoke": {"scene": 0, "prediction_seed": 1234, "movements": 8,
                      "warmups_per_condition": 2, "measured_calls_per_condition": 3},
            "timing": "CUDA-synchronized fresh calls; episode includes monitoring/bookkeeping, excludes loading and trace writing",
            "packages": packages, "fingerprints": fingerprints(), "model_sources": model_sources()}


def verify_protocol(output):
    check_output(output)
    document = base.read_json(output / "protocol.json")
    assert document == protocol_document(), "Frozen code, conditions, sources or packages changed."
    return document


def prepare(output):
    check_output(output)
    if (output / "protocol.json").exists():
        return verify_protocol(output)
    assert not any((output / name).exists() for name in ["episodes", "episodes.json", "summary.json", "smoke.json"])
    document = protocol_document()
    base.save_json(output / "protocol.json", document)
    return document


def evaluate_episode(policy, scene, draw, method, directory, max_steps=300):
    configure(policy, method)
    assert policy.diffusion.num_inference_steps == 100
    np.random.seed(prediction_seed(scene, draw))
    torch.manual_seed(prediction_seed(scene, draw))
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=max_steps)
    observation, _ = env.reset(seed=scene)
    controller = adaptive.AdaptiveLengthController()
    estimator, rolling = adaptive.VisualAlignmentEstimator(), adaptive.RollingVisualFeatures()
    initial = adaptive.physical_state(env, observation)
    rows, units, events, prediction_seconds = [], [], [], []
    age, monitor_seconds, initial_features = 0, 0., None
    versions = tuple(parameter._version for parameter in policy.parameters())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        if method == "adaptive12":
            tick = time.perf_counter()
            initial_features = rolling.update(estimator.estimate(observation["pixels"]), 0)
            monitor_seconds += time.perf_counter() - tick
        with capture_generation(policy) as generated_shapes:
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
                    assert len(commands) == spec(method)["execution"]
                age += 1
                observation, _, terminated, truncated, _ = env.step(command)
                pending = len(policy._queues["action"])
                row = {"step": step, "action": command.tolist(), "new_prediction": bool(fresh),
                       "plan_age": age, "pending_before": pending,
                       "after": adaptive.physical_state(env, observation)}
                if fresh:
                    row.update(predicted_commands=commands, prediction_seconds=prediction_seconds[-1])
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
                    print(f"Running {method}, scene {scene}, draw {draw}: movement {step}, calls {len(prediction_seconds)}", flush=True)
                if terminated or truncated:
                    break
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        units.append(age)
        assert versions == tuple(parameter._version for parameter in policy.parameters()), "Weights mutated."
        result = {"scene": scene, "draw": draw, "prediction_seed": prediction_seed(scene, draw), "method": method,
                  **spec(method), "success": bool(terminated), "truncated": bool(truncated), "steps": len(rows),
                  "model_queries": len(prediction_seconds), "initial": initial, "initial_features": initial_features,
                  "final_coverage": rows[-1]["after"]["coverage"], "wall_seconds": wall_seconds,
                  "monitor_seconds": monitor_seconds, "prediction_seconds": prediction_seconds,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "unit_lengths": units, "events": events, "generated_shapes": generated_shapes,
                  "trace": stem(scene, draw, method) + ".trace.json.gz"}
        directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(directory / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"scene": scene, "draw": draw, "method": method, "initial": initial, "steps": rows}, stream, allow_nan=False)
        result["trace_sha256"] = base.sha(directory / result["trace"])
        base.save_json(directory / (stem(scene, draw, method) + ".json"), result)
        return result
    finally:
        env.close()
        configure(policy, "fixed8")


def read_trace(trial, directory):
    expected = stem(trial["scene"], trial["draw"], trial["method"]) + ".trace.json.gz"
    assert trial["trace"] == expected
    assert base.sha(directory / expected) == trial["trace_sha256"]
    with gzip.open(directory / expected, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def verify_trial(trial, directory, allowed_scenes=SCENES, allowed_draws=DRAWS, max_steps=300):
    scene, draw, method = trial["scene"], trial["draw"], trial["method"]
    assert scene in allowed_scenes and draw in allowed_draws and method in CONDITIONS
    assert trial["prediction_seed"] == prediction_seed(scene, draw)
    assert {key: trial[key] for key in ["horizon", "execution"]} == spec(method)
    trace = read_trace(trial, directory)
    assert (trace["scene"], trace["draw"], trace["method"]) == (scene, draw, method)
    assert trace["initial"] == trial["initial"]
    rows = trace["steps"]
    assert 1 <= len(rows) == trial["steps"] <= max_steps
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and len(rows) == max_steps)
    assert all(row["after"]["coverage"] <= .95 for row in rows[:-1])
    assert trial["wall_seconds"] >= trial["monitor_seconds"] >= 0
    assert all(math.isfinite(trial[key]) for key in ["wall_seconds", "monitor_seconds", "final_coverage"])
    assert trial["peak_gpu_allocated_bytes"] > 0
    controller, pending, age, units, seconds, events = adaptive.AdaptiveLengthController(), 0, 0, [], [], []
    if method != "adaptive12":
        assert trial["events"] == [] and trial["monitor_seconds"] == 0 and trial["initial_features"] is None
    else:
        assert trial["initial_features"] is not None
    for index, row in enumerate(rows):
        assert row["step"] == index + 1
        assert row["new_prediction"] == (pending == 0)
        if pending == 0:
            if age:
                units.append(age)
            age, pending = 0, spec(method)["execution"]
            commands = row["predicted_commands"]
            assert len(commands) == pending and np.asarray(commands).shape == (pending, 2)
            assert np.isfinite(commands).all()
            assert row["prediction_seconds"] > 0 and math.isfinite(row["prediction_seconds"])
            seconds.append(row["prediction_seconds"])
        else:
            assert "predicted_commands" not in row and "prediction_seconds" not in row
        age, pending = age + 1, pending - 1
        assert row["plan_age"] == age and row["pending_before"] == pending
        assert row["action"] == commands[age - 1]
        if method == "adaptive12" and index < len(rows) - 1:
            features = row["features"]
            event = controller.update(features, age, pending, index + 1, commands)
            if event:
                events.append({"step": index + 1, "plan_age": age, **event, "features": features})
                pending -= event["discarded"]
        else:
            assert "features" not in row
        assert row["pending_after"] == pending
    units.append(age)
    assert trial["events"] == events and trial["unit_lengths"] == units
    assert sum(units) == len(rows) and trial["model_queries"] == len(units) == len(seconds)
    assert trial["prediction_seconds"] == seconds
    assert sum(seconds) <= trial["wall_seconds"]
    assert trial["generated_shapes"] == [[1, spec(method)["horizon"], 2]] * len(seconds)
    return trace


def collect(policy, scene, draw, method, directory, max_steps=300):
    sidecar = directory / (stem(scene, draw, method) + ".json")
    if sidecar.exists():
        result = base.read_json(sidecar)
    else:
        assert not (directory / (stem(scene, draw, method) + ".trace.json.gz")).exists(), "Inspect orphan trace; do not overwrite."
        assert not sidecar.with_suffix(".json.tmp").exists(), "Inspect partial sidecar; do not overwrite."
        result = evaluate_episode(policy, scene, draw, method, directory, max_steps)
    assert (result["scene"], result["draw"], result["method"]) == (scene, draw, method)
    verify_trial(result, directory, [scene], [draw], max_steps)
    return result


def verify_prefixes(records, directory):
    for scene in sorted({row["scene"] for row in records}):
        for draw in sorted({row["draw"] for row in records if row["scene"] == scene}):
            trials = [row for row in records if (row["scene"], row["draw"]) == (scene, draw)]
            assert len(trials) == len(CONDITIONS)
            assert len({row["method"] for row in trials}) == len(CONDITIONS)
            reference = next(row for row in trials if row["method"] == "fixed8")
            expected = read_trace(reference, directory)["steps"]
            for trial in trials:
                assert adaptive.states_match(trial["initial"], reference["initial"])
                if trial["horizon"] != 16:
                    continue  # Different horizons need not generate the same random trajectory.
                actual = read_trace(trial, directory)["steps"]
                for left, right in zip(actual[:4], expected[:4]):
                    assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])


def verify_smoke(saved, output):
    assert saved["passed"] and saved["h16_first_four_exact"]
    assert saved["baseline_milestone_prefix_matches_saved_smoke"] and saved["parameter_versions_unchanged"]
    assert saved["warmups_per_condition"] == 2 and saved["measured_calls_per_condition"] == 3
    assert saved["model_weights_sha256"] == base.WEIGHTS_SHA256
    assert len(saved["prefix_trials"]) == 4
    assert {(r["scene"], r["draw"], r["method"]) for r in saved["prefix_trials"]} == {(0, 1234, m) for m in CONDITIONS}
    for trial in saved["prefix_trials"]:
        assert base.read_json(output / "smoke-episodes" / (stem(0, 1234, trial["method"]) + ".json")) == trial
        verify_trial(trial, output / "smoke-episodes", [0], [1234], 8)
    verify_prefixes(saved["prefix_trials"], output / "smoke-episodes")
    reference = base.read_json(adaptive.OUTPUT / "prefix-smoke.json")
    for trial in saved["prefix_trials"]:
        if trial["method"] in ["fixed8", "adaptive12"]:
            actual = read_trace(trial, output / "smoke-episodes")["steps"]
            assert len(actual) == 8
            for row, old in zip(actual, reference["traces"]["fixed8"][:8]):
                assert row["action"] == old["action"] and adaptive.states_match(row["after"], old["after"])
    for key in ["prediction_seconds", "generated_shapes", "peak_gpu_allocated_bytes"]:
        assert set(saved[key]) == set(CONDITIONS)
    for method in CONDITIONS:
        assert len(saved["prediction_seconds"][method]) == 3
        assert all(math.isfinite(value) and value > 0 for value in saved["prediction_seconds"][method])
        assert saved["generated_shapes"][method] == [[1, spec(method)["horizon"], 2]] * 3
        assert len(saved["peak_gpu_allocated_bytes"][method]) == 3
        assert all(isinstance(value, int) and value > 0 for value in saved["peak_gpu_allocated_bytes"][method])
    return saved


def smoke(output):
    verify_protocol(output)
    destination = output / "smoke.json"
    if destination.exists():
        return verify_smoke(base.read_json(destination), output)
    assert torch.cuda.is_available(), "GPU inference required; no silent CPU fallback."
    policy, raw = adaptive.load_policy(base.MODEL)
    assert raw["horizon"] == 16 and policy.diffusion.num_inference_steps == 100
    versions = tuple(parameter._version for parameter in policy.parameters())
    trials = [collect(policy, 0, 1234, method, output / "smoke-episodes", 8) for method in CONDITIONS]
    verify_prefixes(trials, output / "smoke-episodes")
    reference = base.read_json(adaptive.OUTPUT / "prefix-smoke.json")
    for trial in trials:
        if trial["method"] in ["fixed8", "adaptive12"]:
            actual = read_trace(trial, output / "smoke-episodes")["steps"]
            for row, old in zip(actual, reference["traces"]["fixed8"][:8]):
                assert row["action"] == old["action"] and adaptive.states_match(row["after"], old["after"])
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    timings = {method: [] for method in CONDITIONS}
    shapes, peaks = {method: [] for method in CONDITIONS}, {method: [] for method in CONDITIONS}
    try:
        observation, _ = env.reset(seed=0)
        methods = list(CONDITIONS)
        for repeat in range(5):
            for method in methods[repeat % 4:] + methods[:repeat % 4]:
                configure(policy, method)
                np.random.seed(4242 + repeat)
                torch.manual_seed(4242 + repeat)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                with capture_generation(policy) as generated:
                    tick = time.perf_counter()
                    with torch.inference_mode():
                        command = policy.select_action(adaptive.observation_batch(observation, "cuda"))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                assert tuple(command.shape) == (1, 2) and bool(torch.isfinite(command).all())
                assert len(policy._queues["action"]) == spec(method)["execution"] - 1
                if repeat >= 2:
                    timings[method].append(elapsed)
                    shapes[method] += generated
                    peaks[method].append(torch.cuda.max_memory_allocated())
    finally:
        env.close()
        configure(policy, "fixed8")
    assert versions == tuple(parameter._version for parameter in policy.parameters())
    document = {"passed": True, "prefix_trials": trials, "h16_first_four_exact": True,
                "baseline_milestone_prefix_matches_saved_smoke": True, "parameter_versions_unchanged": True,
                "warmups_per_condition": 2, "measured_calls_per_condition": 3,
                "prediction_seconds": timings, "generated_shapes": shapes,
                "peak_gpu_allocated_bytes": peaks, "model_weights_sha256": base.WEIGHTS_SHA256}
    base.save_json(destination, document)
    return verify_smoke(document, output)


def metric_table(episodes):
    expected = {(scene, draw, method) for scene in SCENES for draw in DRAWS for method in CONDITIONS}
    assert len(episodes) == len(expected) and {(r["scene"], r["draw"], r["method"]) for r in episodes} == expected
    return {method: {"attempts": len(rows), "successes": sum(r["success"] for r in rows),
                     **{f"mean_{key}": float(np.mean([r[key] for r in rows]))
                        for key in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                     "mean_prediction_seconds": float(np.mean([np.mean(r["prediction_seconds"]) for r in rows])),
                     "mean_total_prediction_seconds": float(np.mean([sum(r["prediction_seconds"]) for r in rows])),
                     "mean_peak_gpu_allocated_bytes": float(np.mean([r["peak_gpu_allocated_bytes"] for r in rows])),
                     "per_draw_successes": {str(draw): sum(r["success"] for r in rows if r["draw"] == draw)
                                            for draw in DRAWS}}
            for method in CONDITIONS for rows in [[r for r in episodes if r["method"] == method]]}


def paired(episodes, reference, method):
    lookup = {(r["scene"], r["draw"], r["method"]): r for r in episodes}
    assert len(lookup) == len(episodes)
    rng = np.random.default_rng(STATISTICS["bootstrap_seed"])
    indices = rng.integers(0, len(SCENES), size=(STATISTICS["bootstrap_samples"], len(SCENES)))
    differences = {}
    for key in ["success", "final_coverage", "model_queries", "wall_seconds"]:
        values = np.asarray([np.mean([float(lookup[s, d, method][key]) - float(lookup[s, d, reference][key])
                                     for d in DRAWS]) for s in SCENES])
        differences[key] = {"mean_difference": float(values.mean()),
                            "scene_cluster_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    return {"reference": reference, "method": method, "differences_method_minus_reference": differences,
            "gained_successes": [[s, d] for s in SCENES for d in DRAWS
                                 if lookup[s, d, method]["success"] and not lookup[s, d, reference]["success"]],
            "lost_successes": [[s, d] for s in SCENES for d in DRAWS
                               if lookup[s, d, reference]["success"] and not lookup[s, d, method]["success"]]}


def summarize(output):
    verify_protocol(output)
    verified_smoke = verify_smoke(base.read_json(output / "smoke.json"), output)
    document = base.read_json(output / "episodes.json")
    assert document["complete"]
    episodes = document["episodes"]
    table = metric_table(episodes)
    for trial in episodes:
        assert base.read_json(output / "episodes" / (stem(trial["scene"], trial["draw"], trial["method"]) + ".json")) == trial
        verify_trial(trial, output / "episodes")
    verify_prefixes(episodes, output / "episodes")
    result = {"complete": True, "scope": "Eight inspected development scenes, two new samples; not a variable-horizon controller or generalization proof",
              "methods": table, "matched_generation_contrast": paired(episodes, "h16_e4", "h8_e4"),
              "project_comparisons": [paired(episodes, method, "h8_e4") for method in ["fixed8", "adaptive12"]],
              "smoke": verified_smoke}
    destination = output / "summary.json"
    if destination.exists():
        assert base.read_json(destination) == result, "Completed results changed."
    else:
        base.save_json(destination, result)
    print(json.dumps(table, indent=2), flush=True)
    return result


def run(output):
    verify_protocol(output)
    if (output / "summary.json").exists():
        return summarize(output)  # No reloading model, GPU trials or timing repeats.
    smoke(output)
    records, policy = [], None
    directory = output / "episodes"
    for identity in schedule():
        scene, draw, method = identity["scene"], identity["draw"], identity["method"]
        sidecar = directory / (stem(scene, draw, method) + ".json")
        if not sidecar.exists() and policy is None:
            policy, _ = adaptive.load_policy(base.MODEL)
        result = collect(policy, scene, draw, method, directory)
        records.append(result)
        base.save_json(output / "episodes.json", {"complete": len(records) == 64, "episodes": records})
        print(f"Progress {len(records)}/64: {method} scene {scene}, draw {draw}, success={result['success']}, "
              f"calls={result['model_queries']}, time={result['wall_seconds']:.1f}s", flush=True)
    verify_protocol(output)
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
    path = args.output_dir / f"{args.stage}-resources.json"
    if not path.exists():
        base.save_json(path, {"wall_seconds": time.perf_counter() - tick, "peak_process_rss_bytes": sampler.peak_rss,
                             "minimum_system_available_bytes": sampler.minimum_available})


if __name__ == "__main__":
    main()
