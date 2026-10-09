"""Six-run recovery check and missing slower-controller comparison.

Reuses delay_screen.run_job without changing the plant, controller or sensor.
Two exact-trajectory recovery checks must pass before four new comparisons run.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys

try:
    from . import delay_acceptability_audit as audit
except ImportError:
    import delay_acceptability_audit as audit


ROOT = Path(__file__).resolve().parents[1]
JOB_KEYS = ("id", "scenario", "setting", "delay_s", "seed", "method")
EVALUATION_SCENARIOS = ("abrupt", "heldout_ramps")
EVALUATION_SEEDS = (1, 2)
THROUGHPUT_GAIN_RATIO = 1.05


def numerical_input_gate(current, original):
    """Only generated manifest bytes may differ from the original inputs.

    Repreparation writes a new timestamp to _manifest.json. Its semantic check
    belongs to recovery; it remains included in this suite's before/after check.
    """
    audit.require(set(current) == set(original), "Prepared artifact names changed")
    numerical = {key: value for key, value in current.items()
                 if Path(key).name != "_manifest.json"}
    expected = {key: value for key, value in original.items()
                if Path(key).name != "_manifest.json"}
    audit.require(bool(numerical), "No numerical inputs found")
    audit.require(numerical == expected, "Numerical prepared-input hashes changed")
    return numerical


def compare(slow, corrected):
    """Absolute acceptability comes before relative improvement."""
    for key in ("scenario", "seed", "delay_s"):
        audit.require(slow[key] == corrected[key], f"Unmatched comparator: {key}")
    audit.require(slow["method"] == "hold" and slow["setting"] == "slow_update",
                  "Expected frozen slow_update baseline")
    audit.require(corrected["method"] == "predict"
                  and corrected["setting"] == "half_primal_dual",
                  "Expected previously frozen correction")
    audit.require(slow["throughput_tps"] > 0, "Nonpositive baseline throughput")
    slow_status, corrected_status = audit.classify(slow), audit.classify(corrected)
    slow_ok = slow_status["experimental_budget_pass"]
    corrected_ok = corrected_status["experimental_budget_pass"]
    outcome = ("both_pass" if slow_ok and corrected_ok else
               "only_correction_passes" if corrected_ok else
               "only_slower_controller_passes" if slow_ok else "neither_passes")
    ratio = corrected["throughput_tps"] / slow["throughput_tps"]
    switch_cap = max(slow["batch_switches"] + 2, 1.1 * slow["batch_switches"])
    retains_throughput = ratio >= audit.BUDGETS["throughput_preservation_ratio"]
    switching_ok = corrected["batch_switches"] <= switch_cap
    gain = ratio >= THROUGHPUT_GAIN_RATIO
    return {
        "scenario": slow["scenario"], "seed": slow["seed"],
        "slower_id": slow["id"], "corrected_id": corrected["id"],
        "slower_status": slow_status, "corrected_status": corrected_status,
        "absolute_budget_outcome": outcome,
        "corrected_throughput_change_percent": 100 * (ratio - 1),
        "corrected_voltage_integral_reduction_percent": 100 * (
            1 - corrected["integrated_violation_pu_s"] / slow["integrated_violation_pu_s"]
        ) if slow["integrated_violation_pu_s"] else None,
        "corrected_batch_switch_difference": corrected["batch_switches"] - slow["batch_switches"],
        "retains_98_percent_throughput": retains_throughput,
        "at_least_5_percent_throughput_gain": gain,
        "switching_cap": switch_cap, "within_switching_cap": switching_ok,
        "additional_value_screen_pass": bool(
            corrected_ok and retains_throughput and switching_ok and (not slow_ok or gain)),
        "interpretation": "Course screening budgets and cost thresholds, not operational certification",
    }


def source_gate(root, revision):
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    actual = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(git + ["status", "--porcelain"], text=True)
    audit.require(actual == revision, "OpenG2G source revision changed")
    audit.require(not status, "OpenG2G source is not clean")
    return {"revision": actual, "clean": True}


def checked_result(directory, result):
    metrics, trace_hash = audit.audit_trace(directory, result)
    return {**result, **metrics, **audit.classify(metrics), "trace_file_sha256": trace_hash}


def suite(args):
    # Only simulation requires NumPy/psutil; comparison tests use stdlib alone.
    try:
        from . import delay_screen as screen
    except ImportError:
        import delay_screen as screen
    pilot = screen.pilot
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit.require(not any(out.iterdir()), "Use a new/empty output directory")
    args.output = str(out)
    args.openg2g_root = str(Path(args.openg2g_root).resolve())
    args.prepared_data = str(Path(args.prepared_data).resolve())
    source = Path(args.openg2g_root)
    source_before = source_gate(source, pilot.COMMIT)
    before = pilot.fingerprint(args.prepared_data)
    prior_manifest = ROOT / "results/power-model-pilot-verified/manifest.json"
    original = audit.load(prior_manifest)
    audit.require(original["openg2g_commit"] == pilot.COMMIT
                  and original["dataset_revision"] == pilot.DATA_REVISION,
                  "Pinned revisions disagree with original evidence")
    numerical = numerical_input_gate(before, original["prepared_data_sha256"])
    saved_dir = ROOT / "results/delay-prototype"
    saved_summary = audit.load(saved_dir / "summary.json")
    saved = {}
    for row in saved_summary["runs"]:
        if row["id"].startswith("eval-"):
            audit.require(row == audit.load(saved_dir / f"{row['id']}.json"),
                          "Saved summary disagrees with result")
            saved[row["id"]] = checked_result(saved_dir, row)
    audit.require(len(saved) == 12, "Expected 12 saved evaluation comparators")
    selection = audit.load(ROOT / "results/delay-screen/selection.json")
    audit.require(selection["selected_setting"] == "half_primal_dual", "Prior selection changed")
    compatibility_jobs = []
    for delay in (0, 2):
        prior_id = f"eval-heldout_ramps-s1-hold-d{delay}"
        job = {key: saved[prior_id][key] for key in JOB_KEYS}
        job["id"] = f"compat-heldout_ramps-s1-hold-d{delay}"
        compatibility_jobs.append((job, prior_id))
    evaluation_jobs = [dict(id=f"eval-{scenario}-s{seed}-slow_update-d2",
                            scenario=scenario, seed=seed, setting="slow_update",
                            delay_s=2.0, method="hold")
                       for scenario in EVALUATION_SCENARIOS for seed in EVALUATION_SEEDS]
    protocol = {
        "scope": "Two compatibility checks then four frozen slower-controller comparisons",
        "openg2g_commit": pilot.COMMIT, "dataset_revision": pilot.DATA_REVISION,
        "source_before": source_before, "prepared_data_sha256_before": before,
        "numerical_inputs_match_original": True, "numerical_input_sha256": numerical,
        "manifest_note": "Regenerated _manifest.json may differ only as separately validated during recovery; all current inputs are checked unchanged after runs",
        "original_manifest_sha256": audit.sha256(prior_manifest),
        "source_sha256": {p.relative_to(ROOT).as_posix(): audit.sha256(p) for p in (
            Path(__file__), Path(screen.__file__), Path(pilot.__file__), Path(audit.__file__),
            saved_dir / "summary.json", saved_dir / "protocol.json",
            ROOT / "results/delay-screen/selection.json")},
        "python": sys.version,
        "package_versions": {dist.metadata["Name"]: dist.version
                             for dist in importlib.metadata.distributions()},
        "max_runs": 6, "per_run_time_limit_s": 180, "per_run_memory_limit_gib": 3.5,
        "settings": {key: screen.SETTINGS[key] for key in ("half_primal_dual", "slow_update")},
        "duration_s": pilot.DURATION, "sampling_period_s": pilot.DT,
        "compatibility_rule": "Exact trajectory_sha256 for both checks, otherwise stop",
        "compatibility_jobs": [{**job, "prior_id": prior_id}
                               for job, prior_id in compatibility_jobs],
        "evaluation_jobs": evaluation_jobs,
        "budgets": audit.BUDGETS,
        "additional_value_rule": "Correction must pass every experimental budget, retain >=98% of slower-controller throughput, and switches <=max(slower+2,1.1*slower). If both pass, also require >=5% throughput gain. Neither-pass never counts as a solution.",
        "threshold_status": "Inherited budgets and 98%/switching rules; 5% gain rule specified in 2026-10-08 audit before these new runs",
        "scope_limits": "Fixed two-second delay, fixed background, two seeds; modeled mean ITL is not request-level latency; no service-demand floor or field-delay distribution",
    }
    # Freeze requirements, inputs and planned jobs before the first new run.
    pilot.write_json(out / "protocol.json", protocol)
    checks, runs = [], []
    for job, prior_id in compatibility_jobs:
        result = screen.run_job(args, job)
        checked = checked_result(out, result)
        matches = result["trajectory_sha256"] == saved[prior_id]["trajectory_sha256"]
        checks.append({"id": job["id"], "prior_id": prior_id, "matches": matches,
                       "trajectory_sha256": result["trajectory_sha256"],
                       "expected_trajectory_sha256": saved[prior_id]["trajectory_sha256"]})
        pilot.write_json(out / "compatibility.json", checks)
        audit.require(matches, "Recovery trajectory differs; do not mix new and saved evidence")
        runs.append(checked)
    comparisons = []
    for job in evaluation_jobs:
        result = checked_result(out, screen.run_job(args, job))
        runs.append(result)
        corrected = saved[f"eval-{job['scenario']}-s{job['seed']}-predict-d2"]
        comparisons.append(compare(result, corrected))
    after = pilot.fingerprint(args.prepared_data)
    audit.require(after == before, "Prepared inputs changed during simulation")
    source_after = source_gate(source, pilot.COMMIT)
    summary = dict(protocol_sha256=audit.sha256(out / "protocol.json"),
                   compatibility_checks=checks, runs=runs, saved_comparators=list(saved.values()),
                   comparisons=comparisons, prepared_inputs_unchanged=True,
                   prepared_data_sha256_after=after, source_after=source_after)
    pilot.write_json(out / "summary.json", summary)
    print("Completed two matching compatibility runs and four slower-controller runs.")
    for comparison in comparisons:
        print(f"{comparison['scenario']} seed {comparison['seed']}: "
              f"{comparison['absolute_budget_outcome']}; correction throughput "
              f"{comparison['corrected_throughput_change_percent']:+.2f}%; "
              f"additional-value screen={comparison['additional_value_screen_pass']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openg2g-root", default=str(ROOT / "external/openg2g"))
    parser.add_argument("--prepared-data", default=str(ROOT / "data/prepared"))
    parser.add_argument("--output", default=str(ROOT / "results/slow-update-check"))
    suite(parser.parse_args())


if __name__ == "__main__":
    main()
