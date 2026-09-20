"""Bounded follow-up: can fixed OFO tuning tolerate limited GPU profiling?

Reuses the unchanged pilot simulator; implements no adaptive controller.
Select on the development ramp only, then freeze before evaluating seeds 2/3.
The evaluation schedules are already known, so this is exploratory evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import power_model_pilot as pilot


def robust_rows(development):
    """Worst metrics over nominal and omitted-point fits for each setting."""
    rows = []
    for tuning in pilot.TUNINGS:
        pair = [r for r in development if r["tuning"] == tuning]
        assert len(pair) == 2
        assert {r["variant"] for r in pair} == {"nominal", "omit_batch_64"}
        row = {"tuning": tuning}
        for key in ("integrated_violation_pu_s", "worst_vmax", "itl_violation_fraction"):
            row[key] = max(r[key] for r in pair)
        for key in ("worst_vmin", "throughput_tps"):
            row[key] = min(r[key] for r in pair)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openg2g-root", required=True)
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--prior-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prior = Path(args.prior_results).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new/empty output directory")
    manifest = json.loads((prior / "manifest.json").read_text())
    old = json.loads((prior / "summary.json").read_text())
    root = Path(args.openg2g_root).resolve()
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    assert subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == pilot.COMMIT
    assert not subprocess.check_output(git + ["status", "--porcelain"], text=True)
    before = pilot.fingerprint(args.prepared_data)
    assert before == manifest["prepared_data_sha256"]
    driver_hash = hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest()
    assert driver_hash == manifest["driver_sha256"]
    variants = ("nominal", "omit_batch_64")
    parameters = {v: manifest["model_variants"][v]["parameters"] for v in variants}
    protocol = {
        "purpose": "Screen static tuning before implementing adaptation",
        "prior_results": str(prior),
        "prior_manifest_sha256": hashlib.sha256((prior / "manifest.json").read_bytes()).hexdigest(),
        "driver_sha256": driver_hash,
        "followup_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "openg2g_commit": pilot.COMMIT,
        "prepared_data_sha256": before,
        "tunings": pilot.TUNINGS,
        "variants": parameters,
        "development_seed": 0,
        "evaluation_seeds": [2, 3],
        "evaluation_scenarios": ["abrupt", "two_pulses"],
        "selection": "Across both development fits: original voltage/ITL budgets must hold "
                     "for both; maximize minimum throughput. If none qualifies, minimize "
                     "maximum integrated voltage violation, then maximum ITL misses.",
        "comparators": ["reference", old["selected_tuning"], "robust-development-selected"],
        "max_new_simulations": 29,
        "limitations": "One profile, one development schedule, one omission. Evaluation schedules "
                       "are already known. Seeds 2/3 are new but are not independent workloads. "
                       "No calibration, no realistic temporal workload change, no new error magnitudes.",
    }
    # Persist the protocol before any new simulation.
    pilot.write_json(out / "protocol.json", protocol)
    development = [r for r in old["runs"] if r["stage"] == "development"]
    assert len(development) == len(pilot.TUNINGS)
    new_results = []
    for tuning in pilot.TUNINGS:
        job = dict(id=f"dev-{tuning}-omit_batch_64", stage="development", scenario="development",
                   seed=0, tuning=tuning, variant="omit_batch_64",
                   power_parameters=parameters["omit_batch_64"])
        result = pilot.bounded_run(args, job)
        development.append(result)
        new_results.append(result)
    aggregate = robust_rows(development)
    selected = pilot.select_tuning(aggregate)["tuning"]
    tunings = list(dict.fromkeys(("reference", old["selected_tuning"], selected)))
    # Selection is written before running any new evaluation cases.
    pilot.write_json(out / "selection.json", {"selected_tuning": selected,
                     "worst_case_development": aggregate, "evaluated_tunings": tunings})
    print(f"Frozen robust development selection: {selected}", flush=True)
    for seed in (2, 3):
        for scenario in ("abrupt", "two_pulses"):
            for tuning in tunings:
                for variant in variants:
                    job = dict(id=f"eval-s{seed}-{scenario}-{tuning}-{variant}", stage="evaluation",
                               scenario=scenario, seed=seed, tuning=tuning, variant=variant,
                               power_parameters=parameters[variant])
                    new_results.append(pilot.bounded_run(args, job))
    assert pilot.fingerprint(args.prepared_data) == before
    assert len(new_results) <= protocol["max_new_simulations"]
    pilot.write_json(out / "summary.json", {"selected_tuning": selected,
                     "development": development, "new_runs": new_results,
                     "prepared_inputs_unchanged": True})
    print(f"Finished {len(new_results)} new simulations; inputs unchanged.", flush=True)


if __name__ == "__main__":
    main()
