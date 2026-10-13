"""Fresh PushT initializations; unchanged frozen adaptive action-reuse controller."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import robotics_adaptive_length as adaptive


OUTPUT = Path("results/robotics-fresh-evaluation")
DEVELOPMENT = adaptive.OUTPUT
SEEDS = list(range(50, 70))
METHODS = ["fixed8", "fixed12", "adaptive12"]
STATISTICS = {"unit": "paired initial scene", "bootstrap_samples": 10000,
              "bootstrap_seed": 20261012, "confidence": .95,
              "interpretation": "exploratory intervals; no multiplicity adjustment"}
METRICS = ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]


def check_output(output):
    destination = Path(output).resolve()
    for name in ["robotics-adaptive-length", "robotics-benefit-pilot", "robotics-feasibility",
                 "robotics-baseline-check", "robotics-failure-patterns"]:
        original = Path("results", name).resolve()
        if destination == original or original in destination.parents or destination in original.parents:
            raise ValueError("Fresh evaluation must not overlap an earlier evidence directory.")


def protocol_document():
    adaptive.verify_protocol(DEVELOPMENT)
    assert not set(SEEDS) & set(range(50)), "Fresh initializations must exclude earlier scenes."
    assert adaptive.base.read_json(DEVELOPMENT / "prefix-smoke.json")["passed"]
    fingerprints = adaptive.fingerprints()
    for name in ["protocol.json", "summary.json", "prefix-smoke.json"]:
        path = DEVELOPMENT / name
        fingerprints[path.as_posix()] = adaptive.base.sha(path)
    fingerprints["experiments/robotics_fresh_evaluation.py"] = adaptive.base.sha(Path(__file__))
    return {"scope": "20 fresh initial scenes, three methods, 60 attempts; unchanged controller",
            "seeds": SEEDS, "methods": METHODS, "rules": dict(adaptive.RULES),
            "prediction_repeats_per_scene": 1, "max_total_actions": 300,
            "success": "coverage strictly >.95; simulator termination",
            "rng": "numpy/torch reset to scene seed; matched by prediction order, not physical time",
            "method_order": "rotate fixed8/fixed12/adaptive12 by scene index",
            "statistics": STATISTICS,
            "checkpoint": {"model_id": adaptive.base.MODEL_ID,
                           "revision": adaptive.base.MODEL_REVISION,
                           "weights_sha256": adaptive.base.WEIGHTS_SHA256},
            "packages": {name: importlib.metadata.version(name) for name in adaptive.PACKAGES},
            "fingerprints": fingerprints}


def prepare(output):
    check_output(output)
    protocol = protocol_document()
    path = output / "protocol.json"
    if path.exists():
        assert adaptive.base.read_json(path) == protocol, "Frozen fresh-evaluation protocol changed."
    else:
        assert not any((output / name).exists() for name in ["episodes", "episodes.json", "summary.json"]), \
            "Do not bind pre-existing trials to a new protocol."
        adaptive.base.save_json(path, protocol)
    return protocol


def verify_protocol(output):
    check_output(output)
    protocol = adaptive.base.read_json(output / "protocol.json")
    assert protocol == protocol_document(), "Frozen code, evidence, settings or packages changed."
    return protocol


def verify_trial(trial, directory):
    assert trial["seed"] in SEEDS and trial["method"] in METHODS
    expected = f"seed-{trial['seed']}-{trial['method']}.trace.json.gz"
    assert trial["trace"] == expected
    assert adaptive.base.sha(directory / expected) == trial["trace_sha256"]
    with gzip.open(directory / expected, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows = trace["steps"]
    assert 1 <= len(rows) == trial["steps"] <= 300
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert trace["initial"] == trial["initial"]
    assert trial["model_queries"] == sum(r["new_prediction"] for r in rows) == len(trial["unit_lengths"])
    assert all(isinstance(length, int) and length > 0 for length in trial["unit_lengths"])
    assert sum(trial["unit_lengths"]) == len(rows)
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and len(rows) == 300)
    assert all(math.isfinite(trial[name]) for name in METRICS)
    assert 0 <= trial["monitor_seconds"] <= trial["wall_seconds"]
    return trial


def paired_comparison(episodes, reference, method):
    left = {r["seed"]: r for r in episodes if r["method"] == reference}
    right = {r["seed"]: r for r in episodes if r["method"] == method}
    assert set(left) == set(right) == set(SEEDS)
    gained = [seed for seed in SEEDS if right[seed]["success"] and not left[seed]["success"]]
    lost = [seed for seed in SEEDS if left[seed]["success"] and not right[seed]["success"]]
    discordant = len(gained) + len(lost)
    exact_p = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(len(gained), len(lost)) + 1))
                   / 2 ** discordant) if discordant else 1.)
    # Resample whole scene pairs; use the same frozen indices for every contrast.
    rng = np.random.default_rng(STATISTICS["bootstrap_seed"])
    indices = rng.integers(0, len(SEEDS), size=(STATISTICS["bootstrap_samples"], len(SEEDS)))
    differences = {}
    for metric in ["success", "final_coverage", "model_queries", "wall_seconds"]:
        values = np.asarray([float(right[s][metric]) - float(left[s][metric]) for s in SEEDS])
        means = values[indices].mean(axis=1)
        differences[metric] = {"mean_difference": float(values.mean()),
                               "paired_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist()}
    return {"reference": reference, "method": method,
            "gained_success_seeds": gained, "lost_success_seeds": lost,
            "both_success": sum(left[s]["success"] and right[s]["success"] for s in SEEDS),
            "both_failure": sum(not left[s]["success"] and not right[s]["success"] for s in SEEDS),
            "exact_two_sided_discordant_success_p": exact_p,
            "differences_method_minus_reference": differences}


def summarize(output):
    verify_protocol(output)
    document = adaptive.base.read_json(output / "episodes.json")
    episodes = document["episodes"]
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    assert document["complete"] and len(episodes) == len(expected)
    assert {(r["seed"], r["method"]) for r in episodes} == expected
    for row in episodes:
        verify_trial(row, output / "episodes")
    for seed in SEEDS:
        trio = [r for r in episodes if r["seed"] == seed]
        assert all(adaptive.states_match(trio[0]["initial"], r["initial"]) for r in trio[1:]), \
            "Methods must start from the same physical scene and image."
    table = {}
    for method in METHODS:
        rows = [r for r in episodes if r["method"] == method]
        table[method] = {"attempts": len(rows), "successes": sum(r["success"] for r in rows),
                         **{f"mean_{name}": float(np.mean([r[name] for r in rows])) for name in METRICS},
                         "extensions": sum(e["kind"] == "extend" for r in rows for e in r["events"]),
                         "interruptions": sum(e["kind"] == "interrupt" for r in rows for e in r["events"])}
    comparisons = {f"{method}_vs_{reference}": paired_comparison(episodes, reference, method)
                   for reference, method in [("fixed8", "fixed12"), ("fixed8", "adaptive12"),
                                             ("fixed12", "adaptive12")]}
    result = {"complete": True, "methods": table, "paired": comparisons,
              "scope": "20 previously unused PushT initializations; one prediction seed per scene/method",
              "statistics": STATISTICS}
    adaptive.base.save_json(output / "summary.json", result)
    print(json.dumps(table, indent=2), flush=True)
    return result


def run(output):
    verify_protocol(output)
    assert torch.cuda.is_available(), "Do not silently substitute CPU policy inference."
    directory = output / "episodes"
    directory.mkdir(exist_ok=True)
    results, policy = [], None
    total = len(SEEDS) * len(METHODS)
    for index, seed in enumerate(SEEDS):
        order = METHODS[index % 3:] + METHODS[:index % 3]
        for method in order:
            path = directory / f"seed-{seed}-{method}.json"
            if path.exists():
                result = adaptive.base.read_json(path)
            else:
                assert not (directory / f"seed-{seed}-{method}.trace.json.gz").exists(), \
                    "Inspect partial trace before resuming."
                if policy is None:
                    policy, _ = adaptive.load_policy(adaptive.base.MODEL)
                print(f"Starting {method}, fresh scene {seed}", flush=True)
                result = adaptive.evaluate_episode(policy, seed, method, directory)
            assert (result["seed"], result["method"]) == (seed, method)
            verify_trial(result, directory)
            peers = [r for r in results if r["seed"] == seed]
            assert all(adaptive.states_match(r["initial"], result["initial"]) for r in peers)
            results.append(result)
            adaptive.base.save_json(output / "episodes.json", {"complete": len(results) == total,
                                                               "episodes": results})
            print(f"Progress {len(results)}/{total}: {method} scene {seed}, success={result['success']}, "
                  f"calls={result['model_queries']}, time={result['wall_seconds']:.1f}s", flush=True)
    verify_protocol(output)
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "run", "summarize"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "prepare":
        prepare(args.output_dir)
        return
    if args.stage == "summarize":
        summarize(args.output_dir)
        return
    prepare(args.output_dir)
    assert not (args.output_dir / "summary.json").exists(), "Completed run: inspect its saved summary, not run again."
    assert torch.cuda.is_available(), "Do not silently substitute CPU policy inference."
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with adaptive.MemorySampler() as sampler:
        run(args.output_dir)
    resources = {"wall_seconds": time.perf_counter() - started,
                 "peak_process_rss_bytes": sampler.peak_rss,
                 "minimum_system_available_bytes": sampler.minimum_available,
                 "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
    adaptive.base.save_json(args.output_dir / "run-resources.json", resources)


if __name__ == "__main__":
    main()
