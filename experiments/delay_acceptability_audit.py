"""Audit saved delay experiments without OpenG2G, NumPy, or new simulations.

Separates absolute experimental budgets from relative predictor improvements.
The original traces, settings, selection and results are read-only inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {
    "nominal_vmin": 0.95,
    "nominal_vmax": 1.05,
    "integrated_violation_pu_s_max": 0.10,
    "experimental_vmin": 0.94,
    "experimental_vmax": 1.06,
    "itl_deadline_s": 0.08,
    "itl_violation_fraction_max": 0.01,
    "throughput_preservation_ratio": 0.98,
}
METRICS = (
    "integrated_violation_pu_s", "violation_time_s", "worst_vmin", "worst_vmax",
    "throughput_tps", "itl_violation_fraction", "batch_switches",
    "batch_log2_total_variation",
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def classify(metrics):
    """Budgets inherited from the old screen; not operational safety limits."""
    require(all(math.isfinite(metrics[k]) for k in METRICS), "Nonfinite metric")
    failures = []
    for key, threshold, lower_bound in (
        ("integrated_violation_pu_s", BUDGETS["integrated_violation_pu_s_max"], False),
        ("worst_vmin", BUDGETS["experimental_vmin"], True),
        ("worst_vmax", BUDGETS["experimental_vmax"], False),
        ("itl_violation_fraction", BUDGETS["itl_violation_fraction_max"], False),
    ):
        if (metrics[key] < threshold) if lower_bound else (metrics[key] > threshold):
            failures.append(key)
    return {
        "experimental_budget_pass": not failures,
        "experimental_budget_failures": failures,
        "strict_nominal_voltage_pass": (
            metrics["worst_vmin"] >= BUDGETS["nominal_vmin"]
            and metrics["worst_vmax"] <= BUDGETS["nominal_vmax"]
        ),
        "modeled_itl_budget_pass": metrics["itl_violation_fraction"] <= 0.01,
        # No request arrival process, queue or demand floor was specified.
        "service_capacity_adequacy": "not_established",
    }


def recompute(rows):
    require(len(rows) == 3000, "Expected 300 seconds at 0.1-second intervals")
    for i, row in enumerate(rows):
        require(len(row) == 9 and all(math.isfinite(x) for x in row), "Invalid trace row")
        t, batch, power, itl, replicas, vmin, vmax, violation, throughput = row
        require(math.isclose(t, i * 0.1, abs_tol=1e-9), "Incorrect sampling time")
        require(batch in (8, 16, 32, 64, 128, 256, 512), "Unsupported batch")
        require(power > 0 and itl > 0 and replicas > 0, "Invalid plant measurement")
        require(0 < vmin <= vmax and violation >= 0, "Invalid voltage measurement")
        require(math.isclose(throughput, batch / itl * replicas, rel_tol=1e-12),
                "Throughput disagrees with trace inputs")
        require((violation > 0) == (vmin < 0.95 or vmax > 1.05),
                "Voltage extrema disagree with violation indicator")
    return {
        "integrated_violation_pu_s": math.fsum(r[7] for r in rows) * 0.1,
        "violation_time_s": sum(r[7] > 0 for r in rows) * 0.1,
        "worst_vmin": min(r[5] for r in rows),
        "worst_vmax": max(r[6] for r in rows),
        "throughput_tps": math.fsum(r[8] for r in rows) / len(rows),
        "itl_violation_fraction": sum(r[3] > 0.08 for r in rows) / len(rows),
        "batch_switches": sum(a[1] != b[1] for a, b in zip(rows, rows[1:])),
        "batch_log2_total_variation": math.fsum(
            abs(math.log2(a[1]) - math.log2(b[1])) for a, b in zip(rows, rows[1:])
        ),
    }


def audit_trace(directory, result):
    trace_path = directory / f"{result['id']}.trace.json.gz"
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows = trace["rows"]
    metrics = recompute(rows)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<9d", *row))
    require(digest.hexdigest() == result["trajectory_sha256"], "Trajectory hash mismatch")
    for key in METRICS:
        require(math.isclose(metrics[key], result[key], rel_tol=1e-10, abs_tol=1e-9),
                f"Saved metric mismatch: {result['id']}/{key}")
    require(result["all_solves_converged"] is True, "Saved run reports nonconvergence")
    # Check the actual controller information, not just exported aggregate metrics.
    previous = -1
    for observation in trace["observations"]:
        t = observation["time_s"]
        if not observation["available"]:
            require(t < result["delay_s"], "Unexpected unavailable observation")
            continue
        measured, arrived = observation["measured_s"], observation["arrival_s"]
        require(previous <= measured <= arrived <= t + 1e-10, "Noncausal packet")
        previous = measured
        require(math.isclose(t - measured, result["delay_s"], abs_tol=1e-9),
                "Unexpected delivered measurement age")
        for timestamp, key in ((measured, "packet_power_kw"), (t, "current_power_kw")):
            require(math.isclose(sum(observation[key]) * 1000,
                                 rows[round(timestamp / 0.1)][2], rel_tol=1e-12),
                    "Packet/current power disagrees with historical trace")
        sampled = rows[round(measured / 0.1)]
        # Upstream's initial sensitivity re-solve follows the t=0 saved snapshot.
        tolerance = 5e-5 if measured == 0 else 1e-8
        require(abs(min(observation["packet_voltage"]) - sampled[5]) <= tolerance
                and abs(max(observation["packet_voltage"]) - sampled[6]) <= tolerance,
                "Packet voltage disagrees with historical trace")
        expected = observation["packet_voltage"].copy()
        if result["method"] == "predict":
            delta = [a-b for a, b in zip(observation["current_power_kw"],
                                        observation["packet_power_kw"])]
            expected = [v + sum(h*p for h, p in zip(sensitivity, delta))
                        for v, sensitivity in zip(expected, trace["sensitivity"])]
        require(len(expected) == len(observation["voltage_used"])
                and all(math.isclose(a, b, abs_tol=1e-12, rel_tol=0)
                        for a, b in zip(expected, observation["voltage_used"])),
                "Controller observation disagrees with declared correction")
    return metrics, sha256(trace_path)


def paired_comparison(baseline, corrected):
    for key in ("scenario", "seed", "setting", "delay_s"):
        require(baseline[key] == corrected[key], f"Unmatched pair: {key}")
    require(baseline["method"] == "hold" and corrected["method"] == "predict",
            "Expected stale-versus-corrected pair")
    baseline_ok = baseline["experimental_budget_pass"]
    corrected_ok = corrected["experimental_budget_pass"]
    outcome = (
        "both_pass" if baseline_ok and corrected_ok else
        "only_correction_passes" if corrected_ok else
        "only_baseline_passes" if baseline_ok else "neither_passes"
    )
    throughput_ratio = corrected["throughput_tps"] / baseline["throughput_tps"]
    return {
        "baseline_id": baseline["id"], "corrected_id": corrected["id"],
        "scenario": baseline["scenario"], "seed": baseline["seed"],
        "absolute_budget_outcome": outcome,
        "voltage_integral_reduction_percent": 100 * (
            1 - corrected["integrated_violation_pu_s"] / baseline["integrated_violation_pu_s"]
        ) if baseline["integrated_violation_pu_s"] else None,
        "throughput_change_percent": 100 * (throughput_ratio - 1),
        "retains_98_percent_throughput": throughput_ratio >= 0.98,
        "batch_switch_difference": corrected["batch_switches"] - baseline["batch_switches"],
    }


def audit(root=ROOT):
    runs, sources = [], {}
    for name in ("delay-screen", "delay-prototype"):
        directory = root / "results" / name
        for filename in ("summary.json", "protocol.json"):
            path = directory / filename
            sources[path.relative_to(root).as_posix()] = sha256(path)
        summary = load(directory / "summary.json")
        require(summary["prepared_inputs_unchanged"], "Prior run reports changed inputs")
        for result in summary["runs"]:
            require(result == load(directory / f"{result['id']}.json"),
                    "Summary disagrees with per-run result")
            job = load(directory / f"{result['id']}.job.json")
            require(all(result[k] == value for k, value in job.items()),
                    "Job disagrees with saved result")
            metrics, trace_hash = audit_trace(directory, result)
            runs.append({
                "source_directory": name,
                **{k: result[k] for k in ("id", "scenario", "seed", "setting", "method", "delay_s")},
                "stage": "evaluation" if result["id"].startswith("eval-") else "development",
                **metrics, **classify(metrics), "trace_file_sha256": trace_hash,
            })
    require(len(runs) == 48, "Expected the completed 30 + 18 saved runs")
    selection_path = root / "results/delay-screen/selection.json"
    sources[selection_path.relative_to(root).as_posix()] = sha256(selection_path)
    selection = load(selection_path)
    development_selection_check = []
    for item in selection["development_aggregate"]:
        cases = [r for r in runs if r["source_directory"] == "delay-screen"
                 and r["setting"] == item["setting"] and r["delay_s"] > 0]
        require(len(cases) == 4, "Incomplete development comparator")
        passes = all(r["experimental_budget_pass"] for r in cases)
        require(passes == item["eligible"], "Development eligibility mismatch")
        development_selection_check.append({
            "setting": item["setting"], "passes_all_four": passes,
            "minimum_throughput_tps": min(r["throughput_tps"] for r in cases),
            "maximum_integrated_violation_pu_s": max(r["integrated_violation_pu_s"] for r in cases),
            "worst_vmin": min(r["worst_vmin"] for r in cases),
        })
    evaluation = [r for r in runs if r["stage"] == "evaluation"]
    pairs = []
    for corrected in evaluation:
        if corrected["method"] != "predict":
            continue
        baseline = next(r for r in evaluation if r["scenario"] == corrected["scenario"]
                        and r["seed"] == corrected["seed"] and r["method"] == "hold"
                        and r["delay_s"] == corrected["delay_s"])
        pairs.append(paired_comparison(baseline, corrected))
    counts = Counter((r["stage"], r["method"], r["delay_s"],
                      r["experimental_budget_pass"], r["strict_nominal_voltage_pass"])
                     for r in runs)
    return {
        "scope": "Retrospective audit of 48 saved runs; no new simulation or field evidence",
        "budgets": BUDGETS,
        "budget_interpretation": "Inherited selection budgets, not safety certification; no absolute service-demand floor",
        "source_sha256": sources,
        "auditor_sha256": sha256(Path(__file__)),
        "verified_run_count": len(runs),
        "strict_nominal_voltage_pass_count": sum(r["strict_nominal_voltage_pass"] for r in runs),
        "classification_counts": [
            dict(stage=k[0], method=k[1], delay_s=k[2], experimental_budget_pass=k[3],
                 strict_nominal_voltage_pass=k[4], count=v) for k, v in sorted(counts.items())
        ],
        "development_selection_check": development_selection_check,
        "evaluation_pairs": pairs,
        "runs": runs,
        "unanswered": [
            "Slower-update and other simple controllers on evaluation schedules",
            "Variable measurement age and independent external grid changes",
            "Request-level/tail latency and sufficient service capacity",
            "Whether fresh local-voltage control removes the need for remote feedback",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Audited {result['verified_run_count']} saved runs; no simulations rerun.")
    print(f"Strict nominal-voltage passes: {result['strict_nominal_voltage_pass_count']}/48")
    for pair in result["evaluation_pairs"]:
        print(f"{pair['scenario']} seed {pair['seed']}: {pair['absolute_budget_outcome']}; "
              f"throughput {pair['throughput_change_percent']:+.2f}%")


if __name__ == "__main__":
    main()
