"""Development-only gate selection, then frozen full-episode PushT comparison.

Earlier collectors and results are immutable. Only confirmed interruption is
gated; cached-action extension, model weights and visual estimation are reused.
"""

from __future__ import annotations

import argparse
from collections import deque
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import robotics_decline_screen as screen
import robotics_weighted_gate as gate


diagnosis, adaptive, base = screen.diagnosis, screen.adaptive, screen.base
OUTPUT = Path("results/robotics-weighted-gate")
SEEDS = list(range(70, 90))
METHODS = ["adaptive12", "threshold", "weighted"]
SMOKE_SCENE = 62
DEVELOPMENT_SMOKE_SCENE = 0
METRICS = ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]
STATISTICS = {"unit": "paired initial scene", "bootstrap_samples": 10000,
              "bootstrap_seed": 20261013, "confidence": .95,
              "interpretation": "exploratory intervals; no multiplicity adjustment"}


def check_output(output):
    screen.check_output(output)
    target, old = Path(output).resolve(), screen.OUTPUT.resolve()
    if target == old or old in target.parents or target in old.parents:
        raise ValueError("New gate evaluation must not overlap the completed decline screen.")


def development_document():
    protocol = screen.verify_protocol(screen.OUTPUT)
    assert base.read_json(screen.OUTPUT / "independent-audit.json")["passed"]
    cases = base.read_json(screen.OUTPUT / "cases.json")["cases"]
    document = base.read_json(screen.OUTPUT / "comparisons.json")
    assert document["complete"] and len(document["records"]) == 36
    for record, spec in zip(document["records"], protocol["comparison_specs"]):
        screen.verify_record(record, spec, screen.OUTPUT)
    histories = {c["scene_seed"]: gate.extract_case_history(c) for c in cases}
    return gate.development_search(cases, document["records"], histories)


def protocol_document(output):
    development = development_document()
    assert base.read_json(output / "development.json") == development
    fingerprints = dict(base.read_json(screen.OUTPUT / "protocol.json")["fingerprints"])
    paths = [Path("experiments/robotics_weighted_gate.py"), Path(__file__).relative_to(Path.cwd()),
             *[screen.OUTPUT / name for name in ["protocol.json", "cases.json", "comparisons.json",
                                               "summary.json", "independent-audit.json"]]]
    for record in base.read_json(screen.OUTPUT / "comparisons.json")["records"]:
        paths += [Path(record["source_sidecar"]), Path(record["trace_directory"]) / record["trace"]]
    fingerprints.update({p.as_posix(): base.sha(p) for p in paths})
    seen = {r["seed"] for r in base.read_json(screen.fresh.OUTPUT / "episodes.json")["episodes"]}
    assert not set(SEEDS) & (seen | set(range(50))), "Evaluation scenes were already used."
    return {"scope": "20 unused scenes, three full-episode controllers, 60 attempts",
            "milestone": diagnosis.MILESTONE, "seeds": SEEDS, "methods": METHODS,
            "prediction_seed": "equal to scene seed, shared across arms; paired by prediction order",
            "method_order": "rotate three methods by scene index", "max_total_actions": 300,
            "rules": dict(adaptive.RULES), "normalization": gate.NORMALIZATION,
            "selected": development["selected"], "development_sha256": base.sha(output / "development.json"),
            "gate_timing": "Re-evaluate each movement; veto retains confirmation history, interruption budget and cooldown. Normal eight-action trimming still applies.",
            "selection_limit": "One-shot selected-state outcomes are a development proxy, not a full-controller performance estimate or optimal-weight evidence.",
            "preflight": {"exact_unchanged_scene": SMOKE_SCENE,
                          "two_gate_development_scene": DEVELOPMENT_SMOKE_SCENE,
                          "retuning_after_smoke": False},
            "statistics": STATISTICS,
            "checkpoint": {"model_id": base.MODEL_ID, "revision": base.MODEL_REVISION,
                           "weights_sha256": base.WEIGHTS_SHA256},
            "packages": {name: importlib.metadata.version(name) for name in adaptive.PACKAGES},
            "fingerprints": fingerprints}


def prepare(output):
    check_output(output)
    path = output / "protocol.json"
    if path.exists():
        return verify_protocol(output)
    assert not any((output / name).exists() for name in ["episodes", "episodes.json", "summary.json", "smoke"])
    document = development_document()
    if (output / "development.json").exists():
        assert base.read_json(output / "development.json") == document
    else:
        base.save_json(output / "development.json", document)
    protocol = protocol_document(output)
    base.save_json(path, protocol)
    return protocol


def verify_protocol(output):
    check_output(output)
    protocol = base.read_json(output / "protocol.json")
    assert protocol == protocol_document(output), "Frozen sources, development evidence, settings or packages changed."
    return protocol


def decision(method, config, features, history):
    if method == "adaptive12":
        return {"allowed": True, "signals": None, "score": None}
    assert method == config["family"]
    return gate.gate_decision(config, features, history)


def evaluate_episode(policy, seed, method, directory, config=None):
    """Same image/model/8–12 action loop for all arms; no runtime truth features."""
    base.configure_horizon(policy, 12)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = adaptive.gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    observation, _ = env.reset(seed=seed)
    controller = adaptive.AdaptiveLengthController()
    estimator, rolling = adaptive.VisualAlignmentEstimator(), adaptive.RollingVisualFeatures()
    initial = adaptive.physical_state(env, observation)
    rows, units, events = [], [], []
    queries, age, monitor_seconds = 0, 0, 0.
    started = time.perf_counter()
    tick = time.perf_counter()
    initial_estimate = estimator.estimate(observation["pixels"])
    initial_features = rolling.update(initial_estimate, 0)
    monitor_seconds += time.perf_counter() - tick
    try:
        for step in range(1, 301):
            fresh = not policy._queues["action"]
            if fresh:
                if age:
                    units.append(age)
                age, queries = 0, queries + 1
            with torch.inference_mode():
                command = policy.select_action(adaptive.observation_batch(observation, "cuda")).squeeze(0).cpu().numpy()
            if fresh:
                commands = [command.tolist()] + [a.squeeze(0).detach().cpu().numpy().tolist()
                                                for a in policy._queues["action"]]
                assert np.asarray(commands).shape == (12, 2)
            age += 1
            observation, _, terminated, truncated, _ = env.step(command)
            pending = len(policy._queues["action"])
            row = {"step": step, "action": command.tolist(), "new_prediction": bool(fresh),
                   "plan_age": age, "pending_before": pending,
                   "after": adaptive.physical_state(env, observation)}
            if fresh:
                row["predicted_commands"] = commands
            if not (terminated or truncated):
                tick = time.perf_counter()
                features = rolling.update(estimator.estimate(observation["pixels"]), pending)
                gate_result = decision(method, config, features, list(rolling.history))
                row["controller_before"] = dict(vars(controller))
                event, blocked = diagnosis.decide(controller, features, age, pending, step, commands,
                                                   suppress_interrupt=not gate_result["allowed"])
                if event:
                    events.append({"step": step, "plan_age": age, **event, "features": features})
                    if event["discarded"]:
                        policy._queues["action"].clear()
                row.update(features=features, gate=gate_result, blocked=blocked,
                           controller_after=dict(vars(controller)))
                monitor_seconds += time.perf_counter() - tick
            row["pending_after"] = len(policy._queues["action"])
            rows.append(row)
            if step % 80 == 0:
                print(f"Running {method}, scene {seed}: movement {step}, calls {queries}", flush=True)
            if terminated or truncated:
                break
        torch.cuda.synchronize()
        units.append(age)
        result = {"seed": seed, "prediction_seed": seed, "method": method,
                  "success": bool(terminated), "truncated": bool(truncated),
                  "steps": len(rows), "model_queries": queries, "initial": initial,
                  "final_coverage": rows[-1]["after"]["coverage"],
                  "wall_seconds": time.perf_counter() - started, "monitor_seconds": monitor_seconds,
                  "unit_lengths": units, "events": events,
                  "blocked_candidates": sum(r.get("blocked", False) for r in rows),
                  "trace": f"seed-{seed}-{method}.trace.json.gz"}
        with gzip.open(directory / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"initial": initial, "initial_estimate": initial_estimate,
                       "initial_features": initial_features, "steps": rows}, stream, allow_nan=False)
        result["trace_sha256"] = base.sha(directory / result["trace"])
        base.save_json(directory / f"seed-{seed}-{method}.json", result)
        return result
    finally:
        env.close()


def verify_trial(trial, directory, config=None, allowed_seeds=SEEDS):
    assert trial["seed"] in allowed_seeds and trial["method"] in METHODS
    assert trial["prediction_seed"] == trial["seed"]
    expected = f"seed-{trial['seed']}-{trial['method']}.trace.json.gz"
    assert trial["trace"] == expected and base.sha(directory / expected) == trial["trace_sha256"]
    with gzip.open(directory / expected, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows = trace["steps"]
    assert trace["initial"] == trial["initial"] and 1 <= len(rows) == trial["steps"] <= 300
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and len(rows) == 300)
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert all(math.isfinite(trial[name]) for name in METRICS)
    assert 0 <= trial["monitor_seconds"] <= trial["wall_seconds"]
    assert trial["model_queries"] == sum(r["new_prediction"] for r in rows)
    queue, commands, history, events, lengths = deque(), None, deque(maxlen=25), [], []
    controller, age = adaptive.AdaptiveLengthController(), 0
    visual = adaptive.RollingVisualFeatures()
    initial = visual.update(trace["initial_estimate"], 0)
    assert initial == trace["initial_features"]
    if initial["quality"]:
        history.append(initial["coverage"])
    for index, row in enumerate(rows):
        assert row["new_prediction"] == (not queue)
        if row["new_prediction"]:
            if age:
                lengths.append(age)
            age = 0
            commands = row["predicted_commands"]
            assert np.asarray(commands).shape == (12, 2) and np.isfinite(commands).all()
            queue.extend(commands)
        else:
            assert "predicted_commands" not in row
        age += 1
        assert row["action"] == queue.popleft() and row["plan_age"] == age
        assert row["pending_before"] == len(queue)
        if index < len(rows) - 1:
            features = row["features"]
            assert visual.update(features, len(queue)) == features, "Camera window arithmetic changed."
            if features["quality"]:
                history.append(features["coverage"])
            else:
                history.clear()
            expected_gate = decision(trial["method"], config, features, list(history))
            assert row["gate"] == expected_gate and row["controller_before"] == vars(controller)
            event, blocked = diagnosis.decide(controller, features, age, len(queue), row["step"], commands,
                                               suppress_interrupt=not expected_gate["allowed"])
            assert row["blocked"] == blocked and row["controller_after"] == vars(controller)
            if event:
                events.append({"step": row["step"], "plan_age": age, **event, "features": features})
                if event["discarded"]:
                    queue.clear()
        else:
            assert not any(k in row for k in ["features", "gate", "blocked", "controller_before", "controller_after"])
        assert row["pending_after"] == len(queue)
    lengths.append(age)
    assert trial["events"] == events and trial["unit_lengths"] == lengths
    assert trial["model_queries"] == len(lengths) and sum(lengths) == len(rows)
    assert trial["blocked_candidates"] == sum(r.get("blocked", False) for r in rows)
    return trace


def checked_collect(policy, seed, method, directory, config, allowed_seeds):
    path = directory / f"seed-{seed}-{method}.json"
    if path.exists():
        result = base.read_json(path)
    else:
        assert not (directory / f"seed-{seed}-{method}.trace.json.gz").exists(), "Inspect orphan trace before resuming."
        result = evaluate_episode(policy, seed, method, directory, config)
    assert (result["seed"], result["method"]) == (seed, method)
    verify_trial(result, directory, config, allowed_seeds)
    return result


def smoke(output):
    protocol = verify_protocol(output)
    destination = output / "smoke.json"
    if destination.exists():
        return verify_smoke(output)
    directory = output / "smoke"
    directory.mkdir(exist_ok=True)
    policy, _ = adaptive.load_policy(base.MODEL)
    original = checked_collect(policy, SMOKE_SCENE, "adaptive12", directory, None, [SMOKE_SCENE])
    trace = verify_trial(original, directory, None, [SMOKE_SCENE])
    source, saved = diagnosis.read_source(SMOKE_SCENE)
    assert all(original[k] == source[k] for k in ["success", "steps", "model_queries", "events", "unit_lengths"])
    assert adaptive.states_match(original["initial"], source["initial"])
    for actual, old in zip(trace["steps"], saved["steps"]):
        for key, value in old.items():
            if key == "after":
                assert all(actual[key][name] == item for name, item in value.items() if name != "coverage")
                assert abs(actual[key]["coverage"] - value["coverage"]) <= diagnosis.RESTORE_COVERAGE_ATOL
            else:
                assert actual[key] == value, (actual["step"], key)
    development = [checked_collect(policy, DEVELOPMENT_SMOKE_SCENE, method, directory,
                                  protocol["selected"][method], [DEVELOPMENT_SMOKE_SCENE])
                   for method in METHODS[1:]]
    records = [original, *development]
    bindings = {p.as_posix(): base.sha(p) for r in records
                for p in [directory / f"seed-{r['seed']}-{r['method']}.json", directory / r["trace"]]}
    document = {"passed": True, "exact_reference_scene": SMOKE_SCENE,
                "exact_reference_movements": original["steps"], "new_gate_development_scene": DEVELOPMENT_SMOKE_SCENE,
                "records": records, "fingerprints": bindings,
                "retuning": False, "scope": "Functional checks, not additional parameter selection or fresh efficacy evidence"}
    base.save_json(destination, document)
    return document


def verify_smoke(output):
    document = base.read_json(output / "smoke.json")
    assert document["passed"] and document["exact_reference_scene"] == SMOKE_SCENE
    assert not document["retuning"]
    for relative, expected in document["fingerprints"].items():
        assert base.sha(Path(relative)) == expected
    return document


def evaluation_binding(output):
    verify_protocol(output)
    verify_smoke(output)
    document = {"protocol_sha256": base.sha(output / "protocol.json"),
                "smoke_sha256": base.sha(output / "smoke.json"), "seeds": SEEDS, "methods": METHODS}
    path = output / "evaluation-protocol.json"
    if path.exists():
        assert base.read_json(path) == document
    else:
        assert not (output / "episodes").exists(), "Do not bind collected episodes to a new evaluation."
        base.save_json(path, document)
    return document


def paired_comparison(episodes, reference, method):
    left = {r["seed"]: r for r in episodes if r["method"] == reference}
    right = {r["seed"]: r for r in episodes if r["method"] == method}
    assert set(left) == set(right) == set(SEEDS)
    gained = [s for s in SEEDS if right[s]["success"] and not left[s]["success"]]
    lost = [s for s in SEEDS if left[s]["success"] and not right[s]["success"]]
    discordant = len(gained) + len(lost)
    exact_p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(len(gained), len(lost)) + 1)) / 2 ** discordant) if discordant else 1.
    rng = np.random.default_rng(STATISTICS["bootstrap_seed"])
    indices = rng.integers(0, len(SEEDS), size=(STATISTICS["bootstrap_samples"], len(SEEDS)))
    differences = {}
    for metric in ["success", *METRICS[:3]]:
        values = np.asarray([float(right[s][metric]) - float(left[s][metric]) for s in SEEDS])
        differences[metric] = {"mean_difference": float(values.mean()),
                               "paired_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    return {"reference": reference, "method": method, "gained_success_seeds": gained,
            "lost_success_seeds": lost, "exact_two_sided_discordant_success_p": exact_p,
            "differences_method_minus_reference": differences}


def summarize(output):
    protocol = verify_protocol(output)
    evaluation_binding(output)
    document = base.read_json(output / "episodes.json")
    episodes = document["episodes"]
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    assert document["complete"] and len(episodes) == len(expected)
    assert {(r["seed"], r["method"]) for r in episodes} == expected
    for trial in episodes:
        verify_trial(trial, output / "episodes", protocol["selected"].get(trial["method"]))
    for seed in SEEDS:
        peers = [r for r in episodes if r["seed"] == seed]
        assert all(adaptive.states_match(peers[0]["initial"], r["initial"]) for r in peers[1:])
    table = {}
    for method in METHODS:
        rows = [r for r in episodes if r["method"] == method]
        table[method] = {"attempts": len(rows), "successes": sum(r["success"] for r in rows),
                         **{f"mean_{name}": float(np.mean([r[name] for r in rows])) for name in METRICS},
                         "extensions": sum(e["kind"] == "extend" for r in rows for e in r["events"]),
                         "interruptions": sum(e["kind"] == "interrupt" for r in rows for e in r["events"]),
                         "blocked_candidates": sum(r["blocked_candidates"] for r in rows)}
    comparisons = {f"{method}_vs_{reference}": paired_comparison(episodes, reference, method)
                   for reference, method in [("adaptive12", "threshold"), ("adaptive12", "weighted"), ("threshold", "weighted")]}
    result = {"complete": True, "methods": table, "paired": comparisons,
              "scope": "20 new scene seeds, one paired prediction sample per method; not deployment proof",
              "statistics": STATISTICS, "selected": protocol["selected"]}
    base.save_json(output / "summary.json", result)
    print(json.dumps(table, indent=2), flush=True)
    return result


def run(output):
    protocol = verify_protocol(output)
    assert not (output / "summary.json").exists(), "Completed experiment: inspect summary instead of rerunning."
    assert torch.cuda.is_available(), "Do not silently substitute CPU inference."
    smoke(output)
    evaluation_binding(output)
    directory = output / "episodes"
    directory.mkdir(exist_ok=True)
    results, policy = [], None
    for index, seed in enumerate(SEEDS):
        for method in METHODS[index % 3:] + METHODS[:index % 3]:
            path = directory / f"seed-{seed}-{method}.json"
            if not path.exists() and policy is None:
                policy, _ = adaptive.load_policy(base.MODEL)
            result = checked_collect(policy, seed, method, directory, protocol["selected"].get(method), SEEDS)
            peers = [r for r in results if r["seed"] == seed]
            assert all(adaptive.states_match(r["initial"], result["initial"]) for r in peers)
            results.append(result)
            base.save_json(output / "episodes.json", {"complete": len(results) == 60, "episodes": results})
            print(f"Progress {len(results)}/60: {method}, scene {seed}, success={result['success']}, "
                  f"calls={result['model_queries']}, time={result['wall_seconds']:.1f}s", flush=True)
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "run", "summarize"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "prepare":
        prepare(args.output_dir)
        return
    if args.stage == "summarize":
        summarize(args.output_dir)
        return
    verify_protocol(args.output_dir)
    if args.stage == "smoke" and (args.output_dir / "smoke.json").exists():
        verify_smoke(args.output_dir)
        print("Completed smoke verified; its resource measurements are preserved.")
        return
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with adaptive.MemorySampler() as sampler:
        globals()[args.stage](args.output_dir)
    base.save_json(args.output_dir / f"{args.stage}-resources.json",
                   {"wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": sampler.peak_rss,
                    "minimum_system_available_bytes": sampler.minimum_available,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()})


if __name__ == "__main__":
    main()
