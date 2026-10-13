"""Descriptive decline-benefit screen; no classifier or controller revision.

Reuse completed comparisons, keep fixed-eight and adaptive futures separate,
and add only missing adaptive samples plus four restoration controls. Run from
the repository root. Frozen imported collectors and earlier evidence are intact.
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

import robotics_regression_diagnosis as diagnosis


base, adaptive, fresh = diagnosis.base, diagnosis.adaptive, diagnosis.fresh
OUTPUT = Path("results/robotics-decline-screen")
OLD = diagnosis.OUTPUT
FUTURE_SEEDS = [9100, 9101, 9102]
DECISIONS = {"original": "replan_now", "defer_current_unit": "continue_unit"}
# Hypothesized direction toward more success benefit, fixed before new outcomes.
FEATURE_DIRECTIONS = {"warning_count": 1, "drawdown": 1, "accumulated_drop_5": 1,
                      "recovery": -1, "position_error_change_3": 1,
                      "orientation_error_change_3": 1, "command_consistency": -1}


def check_output(output):
    diagnosis.check_output(output)
    target, old = Path(output).resolve(), OLD.resolve()
    if target == old or old in target.parents or target in old.parents:
        raise ValueError("Decline screen must not overlap the completed diagnosis.")


def features_at(rows, series, step, age=None):
    """Use pixels-derived history and executed commands only, never after/truth.

    Missing/invalid recent camera frames yield None for derived camera inputs.
    Command direction describes commanded pusher targets, not object motion.
    """
    assert 1 <= step <= len(rows)
    by_step = {int(item["step"]): item for item in series}
    current = by_step[step]
    result = dict(current)
    result.pop("step", None)
    result["plan_age"] = int(age if age is not None else rows[step - 1].get("plan_age", 8 - current["pending_actions"]))
    result["remaining_budget"] = 300 - step
    window = [by_step.get(t) for t in range(step - 5, step + 1)]
    valid = all(item is not None and item["quality"] for item in window)
    result["accumulated_drop_5"] = (float(np.maximum(-np.diff([r["coverage"] for r in window]), 0).sum())
                                     if valid else None)
    for field in ["position_error", "orientation_error"]:
        older = by_step.get(step - 3)
        error_window = [by_step.get(t) for t in range(step - 3, step + 1)]
        errors_valid = all(item is not None and item["quality"] for item in error_window)
        result[field + "_change_3"] = (float(current[field] - older[field])
                                      if errors_valid else None)
    streak = 0
    for t in range(step, 0, -1):
        item = by_step.get(t)
        if item is None or not diagnosis.regression_condition(item):
            break
        streak += 1
    result["warning_streak"] = streak
    commands = np.asarray([r["action"] for r in rows[max(0, step - 6):step]], dtype=float)
    assert commands.ndim == 2 and commands.shape[1] == 2 and np.isfinite(commands).all()
    differences = np.diff(commands, axis=0)
    speeds = np.linalg.norm(differences, axis=1)
    nonzero = (speeds[:-1] > 1e-9) & (speeds[1:] > 1e-9)
    result["command_consistency"] = (float(np.mean(
        np.sum(differences[:-1][nonzero] * differences[1:][nonzero], axis=1)
        / (speeds[:-1][nonzero] * speeds[1:][nonzero]))) if np.any(nonzero) else None)
    result["command_goal_progress"] = (float((np.linalg.norm(commands[0] - [256., 256.])
                                                    - np.linalg.norm(commands[-1] - [256., 256.])) / 512.)
                                       if len(commands) >= 2 else None)
    return result


def selected_cases():
    """All first actual interruptions; original outcome is context, not selection."""
    document = base.read_json(fresh.OUTPUT / "episodes.json")
    assert document["complete"]
    cases = []
    for episode in document["episodes"]:
        if episode["method"] != "adaptive12":
            continue
        events = [e for e in episode["events"] if e["kind"] == "interrupt"]
        if not events:
            continue
        event = events[0]
        scene, step = episode["seed"], event["step"]
        result, trace = diagnosis.read_source(scene)
        row = trace["steps"][step - 1]
        series = [{"step": r["step"], **r["features"]} for r in trace["steps"] if "features" in r]
        cases.append({"scene_seed": scene, "checkpoint": step,
                      "source_success": result["success"], "features": features_at(trace["steps"], series, step),
                      "source_sidecar": (diagnosis.SOURCE / f"seed-{scene}-adaptive12.json").as_posix(),
                      "source_trace": (diagnosis.SOURCE / result["trace"]).as_posix()})
    return sorted(cases, key=lambda c: c["scene_seed"])


def trial_specs(cases):
    specs = []
    for index, case in enumerate(cases):
        scene, step = case["scene_seed"], case["checkpoint"]
        for sample_index, seed in enumerate(FUTURE_SEEDS):
            arms = list(DECISIONS)
            if (index + sample_index) % 2:
                arms.reverse()
            for arm in arms:
                specs.append({"id": f"decline-{scene}-{step}-{seed}-{arm}", "kind": "probe",
                              "scene_seed": scene, "checkpoint": step, "prediction_seed": seed,
                              "method": "adaptive12", "arm": arm, "decision": DECISIONS[arm]})
    return specs


def control_specs(cases):
    return [{"id": f"restore-{c['scene_seed']}-{c['checkpoint']}", "kind": "control",
             "scene_seed": c["scene_seed"], "checkpoint": c["checkpoint"], "prediction_seed": None,
             "method": "adaptive12", "arm": "original"}
            for c in cases if c["scene_seed"] not in diagnosis.CHECKPOINTS]


def reuse_path(spec):
    if spec["scene_seed"] in diagnosis.CHECKPOINTS and spec["prediction_seed"] in [9100, 9101]:
        expected = diagnosis.CHECKPOINTS[spec["scene_seed"]]["actual_interruption"]
        assert spec["checkpoint"] == expected
        return OLD / "trials" / f"probe-{spec['scene_seed']}-{expected}-{spec['prediction_seed']}-{spec['arm']}.json"
    return None


def legacy_inventory():
    protocol = base.verify_protocol(base.OUTPUT)
    labels = base.read_json(base.OUTPUT / "labels.json")
    assert labels["complete"]
    with gzip.open(base.OUTPUT / "visual-features.json.gz", "rt", encoding="utf-8") as stream:
        series = {e["seed"]: e["features"] for e in json.load(stream)["episodes"]}
    sources = {r["episode"]["seed"]: r for r in base.load_records(base.DISCOVERY)}
    contexts = []
    for case in protocol["cases"]:
        if case["reason"] != "regression":
            continue
        scene, step = case["episode_id"], case["branch_step"]
        samples = []
        for seed in protocol["label_seeds"]:
            arms = {}
            for arm, old_arm in [("continue_unit", "continue_cached"), ("replan_now", "replan_once")]:
                path = base.OUTPUT / "labels" / f"seed-{scene}-sample-{seed}-{old_arm}.json"
                trial = base.read_json(path)
                trace = base.verify_trial(trial, path.parent, step)
                assert trial["diffusion_seed"] == seed and trial["arm"] == old_arm
                arms[arm] = {"success": trial["success"], "final_coverage": trial["final_coverage"],
                             "model_queries": trial["remaining_model_queries"], "wall_seconds": trial["wall_seconds"]}
            samples.append({"prediction_seed": seed, **arms})
        contexts.append({"scene_seed": scene, "checkpoint": step,
                         "features": features_at(sources[scene]["rows"], series[scene], step),
                         "samples": samples, "future_controller": "fixed8",
                         "success_benefit": float(np.mean([int(s["replan_now"]["success"]) - int(s["continue_unit"]["success"]) for s in samples]))})
    assert len(contexts) == 5
    return {"scope": "Five reused development contexts; not new validation or adaptive-controller labels.",
            "contexts": contexts, "fixed_feature_directions": FEATURE_DIRECTIONS,
            "feature_associations": association_table(contexts)}


def fixed_direction_association(contexts, feature, sign):
    """Descriptive rank agreement, not fitted thresholds or predictive accuracy."""
    assert sign in [-1, 1]
    comparable, concordant = 0, 0
    for index, left in enumerate(contexts):
        for right in contexts[index + 1:]:
            a, b = left["features"].get(feature), right["features"].get(feature)
            if a is None or b is None or not all(math.isfinite(x) for x in
                    [a, b, left["success_benefit"], right["success_benefit"]]):
                continue
            dx, dy = sign * (a - b), left["success_benefit"] - right["success_benefit"]
            if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
                continue
            comparable += 1
            concordant += dx * dy > 0
    return {"comparable_pairs": comparable, "concordant_pairs": int(concordant),
            "agreement": concordant / comparable if comparable else None}


def association_table(contexts):
    return {feature: {"hypothesized_direction": sign,
                      **fixed_direction_association(contexts, feature, sign)}
            for feature, sign in FEATURE_DIRECTIONS.items()}


def fingerprints(cases):
    bindings = dict(base.read_json(OLD / "protocol.json")["fingerprints"])
    for path in [Path("experiments/robotics_decline_screen.py"), OLD / "protocol.json", OLD / "restore-prefix-audit.json",
                 OLD / "trials.json", base.OUTPUT / "protocol.json", base.OUTPUT / "labels.json",
                 base.OUTPUT / "visual-features.json.gz"]:
        bindings[path.as_posix()] = base.sha(path)
    for case in cases:
        for name in ["source_sidecar", "source_trace"]:
            path = Path(case[name])
            bindings[path.as_posix()] = base.sha(path)
    for spec in trial_specs(cases):
        path = reuse_path(spec)
        if path:
            trial = base.read_json(path)
            for source in [path, path.parent / trial["trace"]]:
                bindings[source.as_posix()] = base.sha(source)
    for scene, points in diagnosis.CHECKPOINTS.items():
        path = OLD / "trials" / f"control-{scene}-{points['first_condition']}.json"
        trial = base.read_json(path)
        for source in [path, path.parent / trial["trace"]]:
            bindings[source.as_posix()] = base.sha(source)
    legacy_protocol = base.read_json(base.OUTPUT / "protocol.json")
    for case in legacy_protocol["cases"]:
        if case["reason"] == "regression":
            for seed in legacy_protocol["label_seeds"]:
                for arm in ["continue_cached", "replan_once"]:
                    path = base.OUTPUT / "labels" / f"seed-{case['episode_id']}-sample-{seed}-{arm}.json"
                    trial = base.read_json(path)
                    for source in [path, path.parent / trial["trace"]]:
                        bindings[source.as_posix()] = base.sha(source)
    return bindings


def protocol_document(output):
    diagnosis.verify_protocol(OLD)
    cases = base.read_json(output / "cases.json")["cases"]
    assert cases == selected_cases(), "Past-only checkpoint selection changed."
    assert [c["scene_seed"] for c in cases] == [53, 54, 59, 62, 65, 67]
    specs, controls = trial_specs(cases), control_specs(cases)
    return {"scope": "Six selected decline contexts, three paired future samples; descriptive screening, not classifier training or controller adoption",
            "milestone": diagnosis.MILESTONE, "cases": cases, "comparison_specs": specs,
            "new_control_specs": controls, "reused_comparisons": 8, "new_comparisons": 28,
            "new_controls": 4, "reused_controls": 2, "future_seeds": FUTURE_SEEDS,
            "selection": "All first actual interruptions in the 20 frozen adaptive attempts; original outcome not used for selection",
            "decisions": {"original": "Existing interruption now; consumes original budget/cooldown",
                          "defer_current_unit": "Suppress interruption until next prediction, preserve normal trim/extend, then resume unchanged adaptive rules"},
            "features": {"camera": "Existing camera estimates and past-only windows; invalid history yields missing derived features",
                         "accumulated_drop_5": "sum(max(0,c[t-1]-c[t])) over last five changes",
                         "error_change_3": "current camera error minus error three actions earlier",
                         "warning_streak": "consecutive complete regression conditions ending at current action",
                         "command_consistency": "mean cosine of adjacent nonzero differences among last six executed target commands",
                         "command_goal_progress": "(first minus last commanded target distance to static goal)/512; NOT object motion"},
            "fixed_feature_directions": FEATURE_DIRECTIONS, "max_total_actions": 300,
            "success": "coverage strictly >.95; simulator termination",
            "statistics": "Six contexts, not 18 independent scenes. No fitted classifier, threshold or validation accuracy; rank associations exploratory, no multiplicity correction.",
            "timing": "Remaining runtime includes monitoring, excludes prefix replay/setup. Eight reused comparison times come from an earlier session; pooled timing is secondary descriptive evidence.",
            "restore_gate": "Regenerate every prefix, validate old 54/59 snapshot audits and reused original-RNG suffix controls, pass four new original-RNG suffix controls before collecting missing comparisons. Only truth coverage uses 1e-12 tolerance.",
            "fingerprints": fingerprints(cases), "cases_sha256": base.sha(output / "cases.json"),
            "legacy_inventory_sha256": base.sha(output / "legacy-inventory.json"),
            "packages": {name: importlib.metadata.version(name) for name in adaptive.PACKAGES}}


def prepare(output):
    check_output(output)
    if not (output / "protocol.json").exists():
        assert not any((output / name).exists() for name in ["trials", "comparisons.json", "summary.json"])
        base.save_json(output / "cases.json", {"cases": selected_cases()})
        base.save_json(output / "legacy-inventory.json", legacy_inventory())
        base.save_json(output / "protocol.json", protocol_document(output))
    return verify_protocol(output)


def verify_protocol(output):
    check_output(output)
    protocol = base.read_json(output / "protocol.json")
    assert protocol == protocol_document(output), "Frozen screen sources/settings/evidence changed."
    return protocol


def normalize_record(raw, spec, path, reused):
    return {**raw, **spec, "source_sidecar": path.as_posix(), "sidecar_sha256": base.sha(path),
            "trace_directory": path.parent.as_posix(), "reused": reused}


def verify_record(record, spec, output):
    assert all(record.get(key) == value for key, value in spec.items())
    path = Path(record["source_sidecar"])
    expected = reuse_path(spec) if spec["kind"] == "probe" else None
    assert path == (expected or output / "trials" / (spec["id"] + ".json"))
    assert record["reused"] == bool(expected)
    assert base.sha(path) == record["sidecar_sha256"]
    raw = base.read_json(path)
    for key, value in raw.items():
        if key not in spec:
            assert record[key] == value
    if expected:
        assert raw["id"] == path.stem
        assert all(raw.get(key) == value for key, value in spec.items() if key not in ["id", "decision"])
        diagnosis.verify_trial(raw, OLD)
    else:
        assert all(raw.get(k) == v for k, v in spec.items())
    assert Path(record["trace_directory"]) == path.parent
    assert base.sha(path.parent / raw["trace"]) == raw["trace_sha256"]
    with gzip.open(path.parent / raw["trace"], "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows, prefix = trace["steps"], spec["checkpoint"]
    assert record["prefix_step"] == prefix and trace["initial"] == record["initial"]
    assert 1 <= len(rows) == record["steps"] <= 300 - prefix
    assert [r["step"] for r in rows] == list(range(prefix + 1, prefix + len(rows) + 1))
    assert record["final_step"] == prefix + len(rows)
    assert record["model_queries"] == sum(r["new_prediction"] for r in rows)
    assert record["total_model_queries"] == record["model_queries"] + record["prefix_queries"]
    assert sum(record["unit_lengths"]) == len(rows)
    assert all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in record["unit_lengths"])
    assert record["final_coverage"] == rows[-1]["after"]["coverage"]
    assert record["success"] == (record["final_coverage"] > .95)
    assert record["success"] or (record["truncated"] and record["final_step"] == 300)
    assert all(math.isfinite(record[k]) for k in ["final_coverage", "wall_seconds", "monitor_seconds"])
    assert 0 <= record["monitor_seconds"] <= record["wall_seconds"]
    if spec["kind"] == "control":
        source_episode, source = diagnosis.read_source(spec["scene_seed"])
        diagnosis.same_rows(rows, source["steps"][prefix:])
        assert record["exact_suffix_match"]
        assert record["events"] == [e for e in source_episode["events"] if e["step"] >= prefix]
        assert record["total_model_queries"] == source_episode["model_queries"]
    return trace


def cached_recovery(record, trace, case):
    """Score camera recovery before first subsequent prediction, not a runtime input."""
    rows = []
    for row in trace["steps"]:
        if row["new_prediction"]:
            break
        rows.append(row)
    estimates = [r["features"]["coverage"] for r in rows if r.get("features", {}).get("quality")]
    witnessed = bool(estimates and max(estimates) >= case["features"]["recent_peak"])
    complete = len(estimates) == len(rows)
    return {"cached_movements": len(rows),
            "camera_frames": len(estimates),
            "camera_recovery_complete": complete,
            "regained_recent_camera_peak": True if witnessed else (False if complete else None),
            "camera_gain_before_prediction": max(estimates) - case["features"]["coverage"] if estimates else None,
            "eventual_success": record["success"]}


def pair_summary(records, cases):
    contexts = []
    for case in cases:
        samples = []
        for seed in FUTURE_SEEDS:
            arms = {}
            for arm, decision in DECISIONS.items():
                matching = [r for r in records if r["scene_seed"] == case["scene_seed"]
                            and r["prediction_seed"] == seed and r["arm"] == arm]
                assert len(matching) == 1
                arms[decision] = matching[0]
            assert adaptive.states_match(arms["continue_unit"]["initial"], arms["replan_now"]["initial"])
            samples.append({"prediction_seed": seed,
                            **{key: {name: value[name] for name in ["success", "final_coverage", "model_queries", "wall_seconds"]}
                               for key, value in arms.items()},
                            "success_difference": int(arms["replan_now"]["success"]) - int(arms["continue_unit"]["success"]),
                            "coverage_difference": arms["replan_now"]["final_coverage"] - arms["continue_unit"]["final_coverage"],
                            "calls_difference": arms["replan_now"]["model_queries"] - arms["continue_unit"]["model_queries"]})
        contexts.append({**case, "samples": samples,
                         "continue_successes": sum(s["continue_unit"]["success"] for s in samples),
                         "replan_successes": sum(s["replan_now"]["success"] for s in samples),
                         "gains": sum(s["success_difference"] == 1 for s in samples),
                         "losses": sum(s["success_difference"] == -1 for s in samples),
                         "success_benefit": float(np.mean([s["success_difference"] for s in samples])),
                         "mean_coverage_difference": float(np.mean([s["coverage_difference"] for s in samples])),
                         "mean_calls_difference": float(np.mean([s["calls_difference"] for s in samples]))})
    return contexts


def verify_controls(output, protocol):
    document = base.read_json(output / "controls.json")
    assert document["complete"] and len(document["controls"]) == 6
    controls = {c["scene_seed"]: c for c in document["controls"]}
    assert len(controls) == 6 and set(controls) == {c["scene_seed"] for c in protocol["cases"]}
    for spec in protocol["new_control_specs"]:
        verify_record(controls[spec["scene_seed"]], spec, output)
    audits = base.read_json(output / "restoration-audit.json")
    old_audits = base.read_json(OLD / "restore-prefix-audit.json")
    for scene, points in diagnosis.CHECKPOINTS.items():
        record = controls[scene]
        path = OLD / "trials" / f"control-{scene}-{points['first_condition']}.json"
        assert record["reused"] and record["exact_suffix_match"]
        assert record["source_sidecar"] == path.as_posix() and record["sidecar_sha256"] == base.sha(path)
        assert record["covers_checkpoint"] == points["actual_interruption"]
        diagnosis.verify_trial(base.read_json(path), OLD)
        assert audits[str(scene)] == old_audits[str(scene)]["checkpoints"][str(record["covers_checkpoint"])]
    return controls


def summarize(output):
    protocol = verify_protocol(output)
    verify_controls(output, protocol)
    document = base.read_json(output / "comparisons.json")
    records, specs = document["records"], protocol["comparison_specs"]
    assert document["complete"] and len(records) == len(specs) == 36
    assert len({r["id"] for r in records}) == len(records)
    by_id = {s["id"]: s for s in specs}
    cache = []
    by_scene = {c["scene_seed"]: c for c in protocol["cases"]}
    for record in records:
        trace = verify_record(record, by_id[record["id"]], output)
        if record["arm"] == "defer_current_unit":
            cache.append({"scene_seed": record["scene_seed"], "prediction_seed": record["prediction_seed"],
                          **cached_recovery(record, trace, by_scene[record["scene_seed"]])})
    contexts = pair_summary(records, protocol["cases"])
    summary = {"complete": True, "contexts": contexts, "context_count": 6, "paired_futures": 18,
               "reused_records": sum(r["reused"] for r in records), "new_records": sum(not r["reused"] for r in records),
               "continue_successes": sum(c["continue_successes"] for c in contexts),
               "replan_successes": sum(c["replan_successes"] for c in contexts),
               "gains": sum(c["gains"] for c in contexts), "losses": sum(c["losses"] for c in contexts),
               "feature_associations": association_table(contexts), "cached_recovery": cache,
               "scope": protocol["scope"], "classifier_trained": False, "controller_changed": False}
    base.save_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["contexts", "cached_recovery"]}, indent=2), flush=True)
    return summary


def saved_or_collect(policy, spec, output, source, snapshot):
    path = reuse_path(spec) if spec["kind"] == "probe" else None
    reused = bool(path)
    if not path:
        path = output / "trials" / (spec["id"] + ".json")
    if path.exists():
        raw = base.read_json(path)
    else:
        assert not (path.parent / (spec["id"] + ".trace.json.gz")).exists(), "Inspect orphan trace before inference."
        print(f"Starting {spec['id']}", flush=True)
        raw, trace, _ = diagnosis.collect(policy, spec, source=source, snapshot=snapshot)
        diagnosis.save_trial(raw, trace, output)
    record = normalize_record(raw, spec, path, reused)
    verify_record(record, spec, output)
    return record


def run(output):
    protocol = verify_protocol(output)
    assert not (output / "summary.json").exists(), "Completed screen: inspect saved results, do not rerun."
    assert torch.cuda.is_available(), "No silent CPU policy fallback."
    torch.cuda.reset_peak_memory_stats()
    policy, _ = adaptive.load_policy(base.MODEL)
    started = time.perf_counter()
    with adaptive.MemorySampler() as memory:
        snapshots, sources, audits = {}, {}, {}
        for case in protocol["cases"]:
            scene, step = case["scene_seed"], case["checkpoint"]
            _, source = diagnosis.read_source(scene)
            sources[scene] = source
            spec = {"id": f"prefix-{scene}", "kind": "prefix", "scene_seed": scene,
                    "prediction_seed": scene, "method": "adaptive12"}
            print(f"Restore prefix: scene {scene}, action {step}", flush=True)
            _, _, captured = diagnosis.collect(policy, spec, source=source, capture_steps=[step], stop=step)
            snapshots[scene] = captured[step]
            audits[str(scene)] = diagnosis.snapshot_audit(captured[step])
            if scene in diagnosis.CHECKPOINTS:
                old = base.read_json(OLD / "restore-prefix-audit.json")[str(scene)]["checkpoints"][str(step)]
                assert audits[str(scene)] == old, "Reused checkpoint hidden-state audit changed."
        path = output / "restoration-audit.json"
        if path.exists():
            assert base.read_json(path) == audits, "Prefix restoration differs on resume."
        else:
            base.save_json(path, audits)
        controls = []
        for scene, points in diagnosis.CHECKPOINTS.items():
            path = OLD / "trials" / f"control-{scene}-{points['first_condition']}.json"
            raw = base.read_json(path)
            diagnosis.verify_trial(raw, OLD)
            controls.append({"scene_seed": scene, "reused": True, "source_sidecar": path.as_posix(),
                             "sidecar_sha256": base.sha(path), "exact_suffix_match": True,
                             "covers_checkpoint": points["actual_interruption"]})
        for spec in protocol["new_control_specs"]:
            controls.append(saved_or_collect(policy, spec, output, sources[spec["scene_seed"]], snapshots[spec["scene_seed"]]))
        base.save_json(output / "controls.json", {"complete": True, "controls": controls})
        records = []
        for spec in protocol["comparison_specs"]:
            record = saved_or_collect(policy, spec, output, sources[spec["scene_seed"]], snapshots[spec["scene_seed"]])
            records.append(record)
            base.save_json(output / "comparisons.json", {"complete": len(records) == 36, "records": records})
            print(f"Progress {len(records)}/36: {record['id']}, reused={record['reused']}, success={record['success']}, calls={record['model_queries']}, time={record['wall_seconds']:.1f}s", flush=True)
        summarize(output)
    base.save_json(output / "resources.json", {"wall_seconds": time.perf_counter() - started,
                   "new_comparisons": 28, "new_controls": 4, "reused_comparisons": 8, "reused_controls": 2,
                   "peak_process_rss_bytes": memory.peak_rss,
                   "minimum_system_available_bytes": memory.minimum_available,
                   "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=["prepare", "run", "summarize"], required=True)
    args = parser.parse_args()
    check_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    {"prepare": prepare, "run": run, "summarize": summarize}[args.stage](args.output)
