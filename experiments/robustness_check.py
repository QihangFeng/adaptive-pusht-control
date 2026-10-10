"""Frozen, paired robustness evaluation of delayed-voltage control.

No new controller or tuning is introduced. Four historical compatibility runs
precede 72 evaluations on new seeds, variable transport delay, and independent
background demand. The four comparative metrics do not imply certification.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import struct
import sys

try:
    from . import delay_acceptability_audit as audit
    from . import slow_update_check as slow
except ImportError:
    import delay_acceptability_audit as audit
    import slow_update_check as slow


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("abrupt", "heldout_ramps")
SEEDS = (3, 4, 5)
POLICIES = {
    "tuned": ("hold", "half_primal_dual"),
    "slower": ("hold", "slow_update"),
    "corrected": ("predict", "half_primal_dual"),
}
METRICS = (
    "voltage_violation_fraction_percent", "throughput_tps",
    "mean_modeled_itl_ms", "integrated_violation_pu_s",
)
DELAY_SEED_OFFSET = 20261008
DT = 0.1
DURATION = 300
BLOCK_SECONDS = 12
EXTERNAL_BUS = "675"


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_profile(profile):
    blocks = profile["delay_blocks_s"]
    audit.require(len(blocks) == 25 and all(
        isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in blocks),
        "Expected 25 finite, nonnegative twelve-second delays")
    audit.require(profile["delay_mode"] in ("constant", "variable"), "Unknown delay mode")
    audit.require(profile["external_mode"] in ("none", "independent"), "Unknown external mode")
    audit.require(isinstance(profile["external_enabled"], bool)
                  and profile["external_enabled"] == (profile["external_mode"] == "independent"),
                  "External mode disagrees with enable flag")
    audit.require(profile["external_bus"] == EXTERNAL_BUS, "External bus changed")
    audit.require(blocks[0] == 2, "Startup transport delay must be two seconds")
    if profile["delay_mode"] == "constant":
        audit.require(blocks == [2.0] * 25, "Constant delay profile changed")
    else:
        audit.require(sorted(blocks[1:]) == [0.5] * 8 + [2.0] * 8 + [3.5] * 8,
                      "Variable delay distribution changed")
    audit.require(profile["offered_delay_sha256"] == content_hash(blocks),
                  "Delay profile hash mismatch")
    external = [external_power_kw(i * DT, profile) for i in range(3000)]
    audit.require(profile["external_profile_sha256"] == content_hash(external),
                  "External profile hash mismatch")
    return profile


def transport_delay(t, profile):
    """Offered delay of a packet measured at t, not its age at a later decision."""
    audit.require(math.isfinite(t) and 0 <= t < DURATION, "Time outside simulation")
    return float(profile["delay_blocks_s"][min(int(t // BLOCK_SECONDS), 24)])


def external_power_kw(t, profile):
    """Independent balanced demand in kW PER PHASE, zero outside two events."""
    audit.require(math.isfinite(t) and 0 <= t < DURATION, "Time outside simulation")
    if not profile["external_enabled"]:
        return 0.0
    if 75 <= t < 95:
        return 50.0
    if 150 <= t < 155:
        return 20.0 * (t - 150)
    if 155 <= t < 165:
        return 100.0
    if 165 <= t < 170:
        return 20.0 * (170 - t)
    return 0.0


def make_profile(seed, delay_mode, external_mode):
    if delay_mode == "variable":
        # Independent RNG stream: never consume or alter the plant's seed stream.
        import numpy as np
        shuffled = np.array([0.5] * 8 + [2.0] * 8 + [3.5] * 8)
        np.random.default_rng(seed + DELAY_SEED_OFFSET).shuffle(shuffled)
        blocks = [2.0, *shuffled.tolist()]
    else:
        audit.require(delay_mode == "constant", "Unknown delay mode")
        blocks = [2.0] * 25
    profile = dict(delay_mode=delay_mode, external_mode=external_mode,
                   delay_blocks_s=blocks, external_enabled=external_mode == "independent",
                   external_bus=EXTERNAL_BUS, offered_delay_sha256=content_hash(blocks))
    profile["external_profile_sha256"] = content_hash(
        [external_power_kw(i * DT, profile) for i in range(3000)])
    return validate_profile(profile)


def policy_name(result):
    return next((name for name, pair in POLICIES.items()
                 if pair == (result["method"], result["setting"])), None)


def build_jobs():
    """Matched scenario/seed/perturbation groups; no evaluation-driven selection."""
    jobs = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for delay_mode in ("constant", "variable"):
                for external_mode in ("none", "independent"):
                    profile = make_profile(seed, delay_mode, external_mode)
                    for policy, (method, setting) in POLICIES.items():
                        jobs.append(dict(
                            id=f"eval-{scenario}-s{seed}-{delay_mode}-{external_mode}-{policy}",
                            scenario=scenario, seed=seed, method=method, setting=setting,
                            delay_s=2.0, robustness=json.loads(json.dumps(profile))))
    audit.require(len(jobs) == 72 and len({j["id"] for j in jobs}) == 72,
                  "Evaluation job count/identifiers changed")
    return jobs


def recompute_metrics(rows):
    metrics = audit.recompute(rows)
    metrics["mean_modeled_itl_ms"] = math.fsum(r[3] for r in rows) / len(rows) * 1000
    metrics["voltage_violation_fraction_percent"] = metrics["violation_time_s"] / DURATION * 100
    return metrics


def replay_latest_packets(observations, delay_s, profile=None):
    """Independent offered-arrival replay: stale/out-of-order data cannot win.

    The replay constructs every sensor sample, including non-decision ticks.
    It does not call the implementation's channel or infer availability merely
    from the packet chosen by that implementation.
    """
    arrivals = sorted((i * DT + (transport_delay(i * DT, profile) if profile else delay_s),
                       i * DT) for i in range(3000))
    cursor, latest = 0, None
    expected = []
    for observation in observations:
        now = observation["time_s"]
        while cursor < len(arrivals) and arrivals[cursor][0] <= now + 1e-10:
            arrival, measured = arrivals[cursor]
            if latest is None or measured > latest[0]:
                latest = (measured, arrival)
            cursor += 1
        expected.append(latest)
    return expected


def validate_trace(trace, result):
    rows = trace["rows"]
    metrics = recompute_metrics(rows)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<9d", *row))
    audit.require(digest.hexdigest() == result["trajectory_sha256"], "Trajectory hash mismatch")
    for key in (*audit.METRICS, *METRICS):
        # Historical runs predate the two newly explicit mean/fraction fields.
        if key in result:
            audit.require(math.isclose(metrics[key], result[key], rel_tol=1e-10, abs_tol=1e-9),
                          f"Saved metric mismatch: {result['id']}/{key}")
    audit.require(result["all_solves_converged"] is True, "Power flow did not converge")
    profile = result.get("robustness")
    if profile:
        validate_profile(profile)
        logged = trace["robustness"]
        audit.require(all(logged.get(k) == v for k, v in profile.items()),
                      "Trace perturbation metadata differs from job")
        expected_external = [external_power_kw(r[0], profile) for r in rows]
        audit.require(logged["external_power_kw_per_phase"] == expected_external,
                      "External kW/phase trace differs from frozen profile")
        applied = logged["external_applied_kw_by_phase"]
        audit.require(len(applied) == len(rows)
                      and all(len(phases) == 3 and all(math.isclose(p, expected,
                          abs_tol=1e-10, rel_tol=0) for p in phases)
                          for phases, expected in zip(applied, expected_external)),
                      "Actual external load readback differs across phases/profile")
        audit.require(result["external_load_readback_matches_profile"] is True,
                      "Missing actual-load application check")
    else:
        audit.require("robustness" not in trace, "Historical adapter has unexpected perturbation")
    observations = trace["observations"]
    period = result["control_period_s"]
    audit.require(period in (1, 2), "Unexpected controller update interval")
    audit.require(len(observations) == DURATION // period, "Missing controller observations")
    audit.require(all(math.isclose(o["time_s"], i * period, abs_tol=1e-9)
                      for i, o in enumerate(observations)), "Controller observation clock changed")
    packets = replay_latest_packets(observations, result["delay_s"], profile)
    ages = []
    sensitivity = trace["sensitivity"]
    audit.require(bool(sensitivity) and all(len(row) == 3 and all(math.isfinite(v) for v in row)
                                          for row in sensitivity), "Invalid frozen sensitivity")
    for observation, expected_packet in zip(observations, packets):
        audit.require(observation["available"] == (expected_packet is not None),
                      "Packet availability disagrees with independent arrival replay")
        if expected_packet is None:
            continue
        now = observation["time_s"]
        measured, arrived = observation["measured_s"], observation["arrival_s"]
        audit.require(0 <= measured <= arrived <= now + 1e-10, "Noncausal packet")
        audit.require(math.isclose(measured, expected_packet[0], abs_tol=1e-9)
                      and math.isclose(arrived, expected_packet[1], abs_tol=1e-9),
                      "Controller did not use latest measured packet among arrived samples")
        age = now - measured
        audit.require(math.isclose(observation["age_s"], age, abs_tol=1e-9), "Incorrect packet age")
        ages.append(age)
        for timestamp, key in ((measured, "packet_power_kw"), (now, "current_power_kw")):
            audit.require(len(observation[key]) == 3
                          and math.isclose(math.fsum(observation[key]) * 1000,
                                           rows[round(timestamp / DT)][2], rel_tol=1e-12),
                          "Local packet/current power disagrees with historical trace")
        sampled = rows[round(measured / DT)]
        tolerance = 5e-5 if measured == 0 else 1e-8
        packet_voltage = observation["packet_voltage"]
        audit.require(len(packet_voltage) == len(sensitivity)
                      and all(math.isfinite(v) for v in packet_voltage)
                      and abs(min(packet_voltage) - sampled[5]) <= tolerance
                      and abs(max(packet_voltage) - sampled[6]) <= tolerance,
                      "Packet voltage disagrees with historical trace")
        expected_voltage = packet_voltage.copy()
        if result["method"] == "predict":
            delta = [a - b for a, b in zip(observation["current_power_kw"],
                                           observation["packet_power_kw"])]
            expected_voltage = [v + math.fsum(h * p for h, p in zip(hrow, delta))
                                for v, hrow in zip(expected_voltage, sensitivity)]
        else:
            audit.require(result["method"] == "hold", "Unexpected controller method")
        audit.require(len(expected_voltage) == len(observation["voltage_used"])
                      and all(math.isclose(a, b, abs_tol=1e-12, rel_tol=0) for a, b in
                              zip(expected_voltage, observation["voltage_used"])),
                      "Controller input differs from declared frozen correction")
    audit.require(bool(ages), "No delivered controller observation")
    metrics.update(mean_age_s=statistics.mean(ages), max_age_s=max(ages))
    for key in ("mean_age_s", "max_age_s"):
        if key in result:
            audit.require(math.isclose(metrics[key], result[key], abs_tol=1e-9),
                          f"Saved age mismatch: {key}")
    return metrics


def checked_result(directory, result):
    path = Path(directory) / f"{result['id']}.trace.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    metrics = validate_trace(trace, result)
    job = audit.load(Path(directory) / f"{result['id']}.job.json")
    audit.require(all(result.get(k) == v for k, v in job.items()), "Result disagrees with frozen job")
    return {**result, **metrics, "trace_file_sha256": audit.sha256(path)}


def paired_comparison(baseline, corrected):
    for key in ("scenario", "seed", "delay_s"):
        audit.require(baseline[key] == corrected[key], f"Unmatched pair: {key}")
    for key in ("delay_mode", "external_mode", "delay_blocks_s", "external_enabled",
                "external_bus", "offered_delay_sha256", "external_profile_sha256"):
        audit.require(baseline["robustness"][key] == corrected["robustness"][key],
                      f"Unmatched perturbations: {key}")
    baseline_policy = policy_name(baseline)
    audit.require(baseline_policy in ("tuned", "slower") and policy_name(corrected) == "corrected",
                  "Expected old-reading baseline versus frozen correction")
    changes = {}
    for key in METRICS:
        base, new = baseline[key], corrected[key]
        audit.require(math.isfinite(base) and math.isfinite(new), "Nonfinite paired metric")
        changes[key] = dict(baseline=base, corrected=new, change=new - base,
                            change_percent=100 * (new / base - 1) if base else None)
    return dict(scenario=baseline["scenario"], seed=baseline["seed"],
                delay_mode=baseline["robustness"]["delay_mode"],
                external_mode=baseline["robustness"]["external_mode"],
                baseline_policy=baseline_policy, baseline_id=baseline["id"],
                corrected_id=corrected["id"], metrics=changes,
                comparator_note="Same tuning isolates correction" if baseline_policy == "tuned"
                else "Alternative policy: different gains AND two-second update interval")


def aggregate_runs(runs):
    groups = defaultdict(list)
    for result in runs:
        profile = result["robustness"]
        groups[(result["scenario"], profile["delay_mode"], profile["external_mode"],
                policy_name(result))].append(result)
    aggregated = []
    for (scenario, delay, external, policy), cases in sorted(groups.items()):
        audit.require(sorted(r["seed"] for r in cases) == list(SEEDS), "Incomplete seed group")
        stats = {}
        for key in (*METRICS, "mean_age_s", "max_age_s"):
            values = [r[key] for r in cases]
            stats[key] = dict(mean=statistics.mean(values), min=min(values), max=max(values),
                              sample_sd=statistics.stdev(values))
        aggregated.append(dict(scenario=scenario, delay_mode=delay, external_mode=external,
                               policy=policy, seeds=list(SEEDS), metrics=stats))
    return aggregated


def build_comparisons(runs):
    indexed = {}
    for result in runs:
        profile = result["robustness"]
        key = (result["scenario"], result["seed"], profile["delay_mode"], profile["external_mode"],
               policy_name(result))
        audit.require(key not in indexed, "Duplicate evaluation pair key")
        indexed[key] = result
    comparisons = []
    for key, corrected in sorted(indexed.items()):
        if key[-1] == "corrected":
            for baseline_policy in ("tuned", "slower"):
                baseline = indexed.get((*key[:-1], baseline_policy))
                audit.require(baseline is not None, "Missing paired baseline")
                comparisons.append(paired_comparison(baseline, corrected))
    audit.require(len(runs) == 72 and len(comparisons) == 48, "Incomplete robustness evaluation")
    return comparisons


def suite(args):
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
    source_before = slow.source_gate(source, pilot.COMMIT)
    before = pilot.fingerprint(args.prepared_data)
    prior_manifest = ROOT / "results/power-model-pilot-verified/manifest.json"
    original = audit.load(prior_manifest)
    audit.require(original["openg2g_commit"] == pilot.COMMIT
                  and original["dataset_revision"] == pilot.DATA_REVISION,
                  "Pinned revisions disagree with original evidence")
    numerical = slow.numerical_input_gate(before, original["prepared_data_sha256"])
    selection = audit.load(ROOT / "results/delay-screen/selection.json")
    audit.require(selection["selected_setting"] == "half_primal_dual", "Frozen prior tuning changed")
    compatibility_specs = (
        ("delay-prototype", "eval-heldout_ramps-s1-hold-d0", "tuned-d0"),
        ("delay-prototype", "eval-heldout_ramps-s1-hold-d2", "tuned-d2"),
        ("delay-prototype", "eval-heldout_ramps-s1-predict-d2", "corrected-d2"),
        ("slow-update-check", "eval-heldout_ramps-s1-slow_update-d2", "slower-d2"),
    )
    compatibility_jobs = []
    for directory, prior_id, suffix in compatibility_specs:
        saved_directory = ROOT / "results" / directory
        saved = audit.load(saved_directory / f"{prior_id}.json")
        checked_result(saved_directory, saved)
        job = {k: saved[k] for k in slow.JOB_KEYS}
        job["id"] = f"compat-heldout_ramps-s1-{suffix}"
        compatibility_jobs.append((job, directory, prior_id, saved["trajectory_sha256"]))
    evaluation_jobs = build_jobs()
    source_files = (Path(__file__), Path(screen.__file__), Path(pilot.__file__),
                    Path(audit.__file__), Path(slow.__file__), prior_manifest,
                    ROOT / "results/delay-screen/selection.json")
    code_before = {p.relative_to(ROOT).as_posix(): audit.sha256(p) for p in source_files}
    protocol = dict(
        scope="Four exact historical compatibility checks then 72 frozen paired robustness runs",
        openg2g_commit=pilot.COMMIT, dataset_revision=pilot.DATA_REVISION,
        source_before=source_before, source_sha256=code_before,
        prepared_data_sha256_before=before, numerical_inputs_match_original=True,
        numerical_input_sha256=numerical, python=sys.version,
        package_versions={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        max_runs=76, evaluation_run_count=72, per_run_time_limit_s=180,
        per_run_memory_limit_gib=3.5, duration_s=DURATION, sampling_period_s=DT,
        settings={key: screen.SETTINGS[key] for key in ("half_primal_dual", "slow_update")},
        policies=POLICIES, scenarios={key: screen.SCENARIOS[key] for key in SCENARIOS},
        evaluation_seeds=SEEDS, delay_seed_offset=DELAY_SEED_OFFSET,
        delay_description="First 12 seconds fixed 2s; remaining 24 blocks independently shuffled eight each 0.5/2/3.5s; mean offered delay 2s, not mean delivered age",
        external_description="Balanced independent real demand at bus675: 50kW/phase pulse75<=t<95; trapezoid150..170 with5s ramps and100kW/phase plateau155<=t<165; fixed power factor0.96, reactive demand Q=P*tan(acos(0.96)) per phase",
        external_visibility="External profile is plant-side only; no external state or fresh voltage enters optimizer",
        frozen_sensitivity="Common initial operating point; no retuning or online re-estimation",
        metrics=METRICS, voltage_band_pu=[0.95, 1.05], include_startup=True,
        metric_interpretation="Time-weighted fleet throughput and modeled mean ITL, not request/tail/end-to-end latency; four metrics compared without invented absolute pass thresholds",
        compatibility_rule="Exact trajectory SHA256 for every historical method, otherwise stop",
        compatibility_jobs=[dict(job=job, prior_directory=directory, prior_id=prior_id,
                                 expected_trajectory_sha256=digest)
                            for job, directory, prior_id, digest in compatibility_jobs],
        evaluation_jobs=evaluation_jobs,
        scope_limits="Synthetic bounded perturbations, not a measured network-delay distribution or field validation; three independent plant seeds with paired comparisons",
    )
    # Freeze every schedule, identifier and assumption before any simulation.
    pilot.write_json(out / "protocol.json", protocol)
    checks, compatibility_runs, runs = [], [], []
    summary = dict(complete=False, protocol_sha256=audit.sha256(out / "protocol.json"),
                   compatibility_checks=checks, compatibility_runs=compatibility_runs, runs=runs)
    pilot.write_json(out / "summary.json", summary)
    try:
        for job, directory, prior_id, expected_hash in compatibility_jobs:
            result = checked_result(out, screen.run_job(args, job))
            matches = result["trajectory_sha256"] == expected_hash
            checks.append(dict(id=job["id"], prior_directory=directory, prior_id=prior_id,
                               matches=matches, trajectory_sha256=result["trajectory_sha256"],
                               expected_trajectory_sha256=expected_hash))
            compatibility_runs.append(result)
            pilot.write_json(out / "summary.json", summary)
            audit.require(matches, "Adapter trajectory changed; stop before robustness evaluation")
        for job in evaluation_jobs:
            runs.append(checked_result(out, screen.run_job(args, job)))
            pilot.write_json(out / "summary.json", summary)
        after = pilot.fingerprint(args.prepared_data)
        audit.require(after == before, "Prepared inputs changed during simulation")
        source_after = slow.source_gate(source, pilot.COMMIT)
        audit.require(code_before == {p.relative_to(ROOT).as_posix(): audit.sha256(p) for p in source_files},
                      "Driver/source evidence changed during simulation")
        summary.update(aggregates=aggregate_runs(runs), comparisons=build_comparisons(runs),
                       prepared_inputs_unchanged=True, prepared_data_sha256_after=after,
                       source_after=source_after, driver_files_unchanged=True, complete=True)
        pilot.write_json(out / "summary.json", summary)
    except BaseException as error:
        summary["complete"] = False
        summary["failure"] = f"{type(error).__name__}: {error}"
        pilot.write_json(out / "summary.json", summary)
        raise
    print("Completed four exact compatibility checks and 72 audited robustness evaluations.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openg2g-root", default=str(ROOT / "external/openg2g"))
    parser.add_argument("--prepared-data", default=str(ROOT / "data/prepared"))
    parser.add_argument("--output", default=str(ROOT / "results/robustness-check"))
    suite(parser.parse_args())


if __name__ == "__main__":
    main()
