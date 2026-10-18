"""Complete two baselines for an inspected cohort; keep adaptive control frozen.

Collects only fixed-eight/fixed-twelve on 70--89. Adaptive and original 50--69
records are reused by hash-bound references, never rerun or overwritten.
"""

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
import robotics_fresh_evaluation as fresh
import robotics_gate_evaluation as saved


base = adaptive.base
OUTPUT = Path("results/robotics-cohort-completion")
REUSED = saved.OUTPUT
SEEDS = list(range(70, 90))
BASELINES = ["fixed8", "fixed12"]
METHODS = [*BASELINES, "adaptive12"]
MILESTONE = "0f4044966646df4cc38b0373fa52c0b733506e15"
STATISTICS = {"unit": "paired initial scene", "bootstrap_samples": 10000,
              "bootstrap_seed": 20261013, "confidence": .95,
              "interpretation": "Exploratory intervals without multiplicity correction; runtime descriptive only."}


def check_output(output):
    destination = Path(output).resolve()
    if destination not in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}:
        raise ValueError("Use the dedicated cohort-completion directory or its -repeat directory.")


def read_trace(trial, directory):
    expected = f"seed-{trial['seed']}-{trial['method']}.trace.json.gz"
    assert trial["trace"] == expected
    assert base.sha(directory / expected) == trial["trace_sha256"]
    with gzip.open(directory / expected, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def verify_baseline(trial, directory, allowed_seeds=SEEDS):
    assert trial["seed"] in allowed_seeds and trial["method"] in BASELINES
    assert trial["events"] == [] and trial["monitor_seconds"] == 0
    trace = read_trace(trial, directory)
    rows = trace["steps"]
    assert trace["initial"] == trial["initial"] and 1 <= len(rows) == trial["steps"] <= 300
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and len(rows) == 300)
    assert all(r["after"]["coverage"] <= .95 for r in rows[:-1])
    assert math.isfinite(trial["wall_seconds"]) and trial["wall_seconds"] >= 0
    horizon = 8 if trial["method"] == "fixed8" else 12
    lengths = []
    for index, row in enumerate(rows):
        age = index % horizon + 1
        assert row["new_prediction"] == (age == 1)
        assert row["plan_age"] == age
        assert row["pending_before"] == row["pending_after"] == horizon - age
        assert len(row["action"]) == 2 and all(math.isfinite(x) for x in row["action"])
        assert "features" not in row and "predicted_commands" not in row
        if age == 1:
            lengths.append(min(horizon, len(rows) - index))
    assert trial["unit_lengths"] == lengths and sum(lengths) == len(rows)
    assert trial["model_queries"] == sum(r["new_prediction"] for r in rows) == len(lengths)
    return trace


def common_prefix(trial, trace, adaptive_trial, adaptive_trace):
    assert trial["seed"] == adaptive_trial["seed"]
    assert adaptive.states_match(trial["initial"], adaptive_trial["initial"])
    assert len(trace["steps"]) >= 8 and len(adaptive_trace["steps"]) >= 8
    for row, reference in zip(trace["steps"][:8], adaptive_trace["steps"][:8]):
        assert row["step"] == reference["step"] and row["action"] == reference["action"]
        assert adaptive.states_match(row["after"], reference["after"]), row["step"]


def entry(trial, sidecar):
    return {**trial, "source_sidecar": sidecar.as_posix(), "trace_directory": sidecar.parent.as_posix()}


def reused_adaptive():
    source = base.read_json(REUSED / "episodes.json")
    assert source["complete"] and len(source["episodes"]) == 60
    assert base.read_json(REUSED / "independent-audit.json")["passed"]
    entries = []
    for seed in SEEDS:
        path = REUSED / "episodes" / f"seed-{seed}-adaptive12.json"
        trial = base.read_json(path)
        indexed = [r for r in source["episodes"] if (r["seed"], r["method"]) == (seed, "adaptive12")]
        assert len(indexed) == 1 and indexed[0] == trial
        trace = saved.verify_trial(trial, path.parent)
        assert trial["unit_lengths"][0] == 12 and not any(r.get("blocked", False) for r in trace["steps"])
        entries.append(entry(trial, path))
    return entries


def original_cohort():
    document = base.read_json(fresh.OUTPUT / "episodes.json")
    assert document["complete"] and len(document["episodes"]) == 60
    assert {(r["seed"], r["method"]) for r in document["episodes"]} == {(s, m) for s in range(50, 70) for m in METHODS}
    assert base.read_json(fresh.OUTPUT / "independent-audit.json")["passed"]
    entries = []
    for trial in document["episodes"]:
        path = fresh.OUTPUT / "episodes" / f"seed-{trial['seed']}-{trial['method']}.json"
        assert base.read_json(path) == trial
        fresh.verify_trial(trial, path.parent)
        entries.append(entry(trial, path))
    return entries


def protocol_document():
    new_reused, old = reused_adaptive(), original_cohort()
    assert base.read_json(adaptive.OUTPUT / "prefix-smoke.json")["passed"]
    source_protocol = base.read_json(REUSED / "protocol.json")
    old_protocol = base.read_json(fresh.OUTPUT / "protocol.json")
    assert source_protocol["rules"] == adaptive.RULES
    checkpoint = {"model_id": base.MODEL_ID, "revision": base.MODEL_REVISION, "weights_sha256": base.WEIGHTS_SHA256}
    assert source_protocol["checkpoint"] == old_protocol["checkpoint"] == checkpoint
    bindings = dict(source_protocol["fingerprints"])
    bindings.update(base.read_json(fresh.OUTPUT / "protocol.json")["fingerprints"])
    for source in [REUSED, fresh.OUTPUT]:
        for name in ["protocol.json", "episodes.json", "summary.json", "independent-audit.json"]:
            path = source / name
            bindings[path.as_posix()] = base.sha(path)
    for record in [*new_reused, *old]:
        for path in [Path(record["source_sidecar"]), Path(record["trace_directory"]) / record["trace"]]:
            bindings[path.as_posix()] = base.sha(path)
    source = Path(__file__).relative_to(Path.cwd())
    bindings[source.as_posix()] = base.sha(source)
    for relative, expected in bindings.items():
        assert base.sha(Path(relative)) == expected, relative
    packages = {name: importlib.metadata.version(name) for name in adaptive.PACKAGES}
    assert packages == source_protocol["packages"] == old_protocol["packages"]
    return {"milestone": MILESTONE, "scope": "Complete an already-inspected cohort, not newly blind validation.",
            "seeds": SEEDS, "new_methods": BASELINES, "comparison_methods": METHODS,
            "new_scored_attempts": 40, "reused_adaptive_attempts": 20,
            "method_order": "alternate fixed8/fixed12 order by scene index", "prediction_seed": "scene seed per method",
            "max_total_actions": 300, "success": "true coverage strictly greater than .95",
            "rules": dict(adaptive.RULES), "collector": "unchanged adaptive.evaluate_episode for baselines only",
            "integration_gate": "Exact initial states and first eight actions/physics/RGB per scene; only existing truth-coverage tolerance1e-12.",
            "preflight": "Reuse existing horizon-prefix smoke; first scored runs are integration checks, no extra GPU trials.",
            "statistics": STATISTICS, "runtime_limit": "Adaptive prior session and additional logging; runtime differences descriptive, not clean causal/inferential effects.",
            "checkpoint": source_protocol["checkpoint"], "packages": packages,
            "fingerprints": bindings, "reused_sidecars": [r["source_sidecar"] for r in new_reused],
            "pooled_seeds": list(range(50, 90)), "tuning": False, "rejected_gate_reruns": False}


def verify_protocol(output):
    check_output(output)
    protocol = base.read_json(output / "protocol.json")
    assert protocol["seeds"] == SEEDS and protocol["new_methods"] == BASELINES
    assert protocol["comparison_methods"] == METHODS and protocol["new_scored_attempts"] == 40
    assert protocol["reused_adaptive_attempts"] == 20 and protocol["rules"] == adaptive.RULES
    assert protocol["max_total_actions"] == 300 and protocol["milestone"] == MILESTONE
    assert protocol["statistics"] == STATISTICS and not protocol["tuning"] and not protocol["rejected_gate_reruns"]
    assert protocol["packages"] == {name: importlib.metadata.version(name) for name in adaptive.PACKAGES}
    assert protocol["checkpoint"] == {"model_id": base.MODEL_ID, "revision": base.MODEL_REVISION,
                                      "weights_sha256": base.WEIGHTS_SHA256}
    for relative, expected in protocol["fingerprints"].items():
        assert base.sha(Path(relative)) == expected, relative
    return protocol


def prepare(output):
    check_output(output)
    if (output / "protocol.json").exists():
        return verify_protocol(output)
    assert not any((output / name).exists() for name in ["episodes", "episodes.json", "summary.json"])
    document = protocol_document()
    base.save_json(output / "protocol.json", document)
    return document


def checked_collect(policy, seed, method, output, reference):
    assert seed in SEEDS and method in BASELINES, "Do not rerun adaptive or rejected filters."
    directory = output / "episodes"
    path = directory / f"seed-{seed}-{method}.json"
    if path.exists():
        trial = base.read_json(path)
    else:
        assert not (directory / f"seed-{seed}-{method}.trace.json.gz").exists(), "Inspect orphan trace before resuming."
        trial = adaptive.evaluate_episode(policy, seed, method, directory)
    assert (trial["seed"], trial["method"]) == (seed, method)
    trace = verify_baseline(trial, directory)
    adaptive_trace = read_trace(reference, Path(reference["trace_directory"]))
    common_prefix(trial, trace, reference, adaptive_trace)
    return entry(trial, path)


def paired(episodes, reference, method, seeds):
    left = {r["seed"]: r for r in episodes if r["method"] == reference}
    right = {r["seed"]: r for r in episodes if r["method"] == method}
    assert sum(r["method"] in [reference, method] for r in episodes) == 2 * len(seeds)
    assert set(left) == set(right) == set(seeds)
    gained = [s for s in seeds if right[s]["success"] and not left[s]["success"]]
    lost = [s for s in seeds if left[s]["success"] and not right[s]["success"]]
    n = len(gained) + len(lost)
    exact = min(1., 2 * sum(math.comb(n, k) for k in range(min(len(gained), len(lost)) + 1)) / 2**n) if n else 1.
    rng = np.random.default_rng(STATISTICS["bootstrap_seed"])
    indices = rng.integers(0, len(seeds), size=(STATISTICS["bootstrap_samples"], len(seeds)))
    differences = {}
    for metric in ["success", "final_coverage", "model_queries"]:
        values = np.asarray([float(right[s][metric]) - float(left[s][metric]) for s in seeds])
        differences[metric] = {"mean_difference": float(values.mean()),
                               "paired_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    return {"reference": reference, "method": method, "gained_success_seeds": gained, "lost_success_seeds": lost,
            "exact_two_sided_discordant_success_p": exact, "differences_method_minus_reference": differences,
            "descriptive_runtime_difference": float(np.mean([right[s]["wall_seconds"] - left[s]["wall_seconds"] for s in seeds]))}


def metric_table(episodes, seeds):
    assert len(episodes) == 3 * len(seeds)
    assert {(r["seed"], r["method"]) for r in episodes} == {(s, m) for s in seeds for m in METHODS}
    table = {}
    for method in METHODS:
        trials = [r for r in episodes if r["method"] == method]
        table[method] = {"attempts": len(trials), "successes": sum(r["success"] for r in trials),
                         **{f"mean_{metric}": float(np.mean([r[metric] for r in trials]))
                            for metric in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]}}
    return {"methods": table, "paired": {f"{m}_vs_{r}": paired(episodes, r, m, seeds)
            for r, m in [("fixed8", "fixed12"), ("fixed8", "adaptive12"), ("fixed12", "adaptive12")]}}


def summarize(output):
    verify_protocol(output)
    document = base.read_json(output / "episodes.json")
    assert document["complete"] and len(document["episodes"]) == 60
    expected_adaptive = reused_adaptive()
    adaptive_entries = [r for r in document["episodes"] if r["method"] == "adaptive12"]
    assert adaptive_entries == expected_adaptive
    for record in document["episodes"]:
        if record["method"] == "adaptive12":
            continue
        path = output / "episodes" / f"seed-{record['seed']}-{record['method']}.json"
        trial = base.read_json(path)
        assert record == entry(trial, path)
        trace = verify_baseline(trial, path.parent)
        reference = next(r for r in expected_adaptive if r["seed"] == trial["seed"])
        common_prefix(trial, trace, reference, read_trace(reference, Path(reference["trace_directory"])))
    episodes = document["episodes"]
    result = {"complete": True, "scope": "Inspected cohort completion and retrospective pooled synthesis; unchanged method.",
              "cohort_70_89": metric_table(episodes, SEEDS),
              "pooled_50_89": metric_table([*original_cohort(), *episodes], list(range(50, 90))),
              "statistics": STATISTICS, "runtime_limit": "Different sessions/logging: descriptive only.",
              "new_scored_attempts": 40, "reused_adaptive_attempts": 20, "reused_original_attempts": 60}
    path = output / "summary.json"
    if path.exists():
        assert base.read_json(path) == result, "Do not overwrite a changed completed summary."
    else:
        base.save_json(path, result)
    print(json.dumps({name: result[name]["methods"] for name in ["cohort_70_89", "pooled_50_89"]}, indent=2), flush=True)
    return result


def run(output):
    verify_protocol(output)
    assert not (output / "summary.json").exists(), "Completed comparison: inspect its summary, do not rerun."
    assert torch.cuda.is_available(), "Do not silently replace CUDA inference with CPU."
    reused = reused_adaptive()
    directory = output / "episodes"
    directory.mkdir(exist_ok=True)
    episodes, policy, count = list(reused), None, 0
    for index, seed in enumerate(SEEDS):
        order = BASELINES if index % 2 == 0 else list(reversed(BASELINES))
        reference = next(r for r in reused if r["seed"] == seed)
        for method in order:
            verify_protocol(output)
            path = directory / f"seed-{seed}-{method}.json"
            if not path.exists() and policy is None:
                policy, _ = adaptive.load_policy(base.MODEL)
            record = checked_collect(policy, seed, method, output, reference)
            episodes.append(record)
            count += 1
            base.save_json(output / "episodes.json", {"complete": count == 40, "new_baselines_complete": count,
                                                       "reused_adaptive": 20, "episodes": episodes})
            print(f"Progress {count}/40 baselines: {method}, scene{seed}, success={record['success']}, "
                  f"calls={record['model_queries']}, time={record['wall_seconds']:.1f}s", flush=True)
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "run", "summarize"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage != "run":
        globals()[args.stage](args.output_dir)
        return
    verify_protocol(args.output_dir)
    assert not (args.output_dir / "summary.json").exists()
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    existing = len(list((args.output_dir / "episodes").glob("seed-*.json")))
    resources_path = args.output_dir / f"run-resources-from-{existing:02d}.json"
    assert not resources_path.exists(), "Preserve prior invocation resource measurements."
    started = time.perf_counter()
    with adaptive.MemorySampler() as sampler:
        run(args.output_dir)
    base.save_json(resources_path, {"wall_seconds": time.perf_counter() - started,
                                   "completed_before_invocation": existing,
                                   "completed_after_invocation": len(list((args.output_dir / "episodes").glob("seed-*.json"))),
                                   "peak_process_rss_bytes": sampler.peak_rss,
                                   "minimum_system_available_bytes": sampler.minimum_available,
                                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()})


if __name__ == "__main__":
    main()
