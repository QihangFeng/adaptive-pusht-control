"""Read-only provenance audit of existing discovery recovery action labels.

Standard library only: no policy, rendering, physics, image fitting or training.
RGB hashes are references, not RGB bytes or evidence of a new exact replay.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


OUTPUT = Path("results/robotics-recovery-feasibility")
DISCOVERY = Path("results/robotics-failure-patterns")
PILOT = Path("results/robotics-benefit-pilot")
SCENES = list(range(30))
HORIZON = 16
OBSERVATIONS = 2


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def states_match(left, right):
    a, b = dict(left), dict(right)
    ac, bc = a.pop("coverage"), b.pop("coverage")
    return a == b and abs(ac - bc) <= 1e-12


def check_state(state):
    digest = state["image_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert len(state["pusher_pos"]) == 2 and all(math.isfinite(x) for x in state["pusher_pos"])
    assert math.isfinite(state["coverage"]) and 0 <= state["coverage"] <= 1


def check_rows(rows, first_step=1):
    assert rows, "Empty executed trajectory cannot supply action labels."
    for index, row in enumerate(rows, start=first_step):
        assert row["step"] == index, "Missing, duplicate or reordered executed movement."
        assert len(row["action"]) == 2 and all(math.isfinite(x) for x in row["action"])
        assert isinstance(row["new_prediction"], bool)
        check_state(row["after"])


def join_recovery_rows(prefix_rows, branch_rows, branch_state, branch_after_step):
    """Join ORIGINAL executed prefix and NEW suffix without a reset at branch.

    Indices are one-based movement numbers; branch_after_step is completed.
    This proves recorded chronology, not a new simulator/RGB/RNG restoration.
    """
    assert isinstance(branch_after_step, int) and branch_after_step >= 1
    assert len(prefix_rows) >= branch_after_step
    check_rows(prefix_rows)
    check_rows(branch_rows, branch_after_step + 1)
    assert states_match(prefix_rows[branch_after_step - 1]["after"], branch_state), \
        "Saved branch state does not match the original executed prefix."
    joined = list(prefix_rows[:branch_after_step]) + list(branch_rows)
    check_rows(joined)
    return joined


def observation_at(initial, rows, after_step):
    assert 0 <= after_step <= len(rows)
    return initial if after_step == 0 else rows[after_step - 1]["after"]


def training_windows(initial, rows, minimum_anchor=1):
    """Native H16/n_obs2 alignment, with no start/end padding.

    At completed movement t, observations are S[t-1], S[t]. Labels are
    executed movements t..t+15: one past command and fifteen future ones.
    The usable future starts at model output index one. No predicted plan
    is substituted for the trajectory's actually executed commands.
    """
    assert isinstance(minimum_anchor, int) and minimum_anchor >= 1
    for anchor in range(minimum_anchor, len(rows) - 14):
        observations = [observation_at(initial, rows, t) for t in (anchor - 1, anchor)]
        labels = rows[anchor - 1:anchor + 15]
        assert len(labels) == HORIZON
        sample = {"rgb_sha256": [s["image_sha256"] for s in observations],
                  "pusher_positions": [s["pusher_pos"] for s in observations],
                  "executed_actions": [r["action"] for r in labels]}
        yield {"anchor_after_step": anchor, "observation_after_steps": [anchor - 1, anchor],
               "executed_label_steps": [anchor, anchor + 15], "sample_sha256": fingerprint(sample)}


def verify_baseline(episode, trace):
    assert episode["seed"] in SCENES
    rows, initial = trace["steps"], trace["initial"]
    check_state(initial)
    check_rows(rows)
    assert len(rows) == episode["steps"] and len(rows) <= 300
    previous = initial
    for row in rows:
        assert states_match(previous, row["before"]), "Baseline state chronology changed."
        previous = row["after"]
    assert episode["success"] == (rows[-1]["after"]["coverage"] > .95)
    assert episode["final_coverage"] == rows[-1]["after"]["coverage"]
    assert all(r["after"]["coverage"] <= .95 for r in rows[:-1])
    assert sum(r["new_prediction"] for r in rows) == episode["model_queries"]
    return rows


def verify_branch(trial, trace, original):
    episode, base_trace = original
    assert trial["seed"] == episode["seed"] in SCENES
    assert trial["identical_branch_state"] is True
    branch = trial["branch_step"]
    suffix = trace["steps"]
    rows = join_recovery_rows(base_trace["steps"], suffix, trace["branch_state"], branch)
    assert len(suffix) == trial["remaining_actions_executed"]
    assert len(rows) == trial["total_actions"] <= 300
    assert trial["success"] == (suffix[-1]["after"]["coverage"] > .95)
    assert trial["final_coverage"] == suffix[-1]["after"]["coverage"]
    assert all(r["after"]["coverage"] <= .95 for r in rows[:-1])
    assert sum(r["new_prediction"] for r in suffix) == trial["remaining_model_queries"]
    assert sum(r["new_prediction"] for r in rows) == trial["total_model_queries"]
    assert trial["arm"] in ["continue_cached", "replan_once"]
    if trial["arm"] == "continue_cached":
        assert trial["cached_prefix_check"] == "verified"
        pending = base_trace["steps"][branch - 1]["pending_actions"]
        for offset, row in enumerate(suffix[:pending]):
            expected = base_trace["steps"][branch + offset]
            assert not row["new_prediction"] and row["action"] == expected["action"]
            assert states_match(row["after"], expected["after"])
    else:
        assert suffix[0]["new_prediction"] is True
    return rows


def record_document(identifier, source, trial, initial, rows, category, branch=0):
    windows = list(training_windows(initial, rows, max(1, branch)))
    return {"id": identifier, "source": source, "scene": trial["seed"],
            "success": trial["success"], "category": category,
            "branch_after_step": branch, "total_executed_movements": len(rows),
            "post_branch_executed_movements": len(rows) - branch,
            "native_h16_complete_windows": len(windows),
            "first_complete_window": windows[0] if windows else None,
            "last_complete_window": windows[-1] if windows else None,
            "trajectory_sha256": fingerprint({"initial_rgb": initial["image_sha256"],
                                              "rows": [[r["action"], r["after"]["image_sha256"],
                                                        r["after"]["pusher_pos"]] for r in rows]}),
            "window_hashes": windows}


def aggregate(records):
    groups = defaultdict(list)
    eligible = [r for r in records if r["category"] == "successful_continuation_of_failed_original"]
    for record in eligible:
        for window in record["window_hashes"]:
            groups[window["sample_sha256"]].append(
                {"record_id": record["id"], "anchor_after_step": window["anchor_after_step"]})
    return {"record_categories": dict(sorted(Counter(r["category"] for r in records).items())),
            "recovery_candidate_records": len(eligible),
            "recovery_candidate_scenes": sorted({r["scene"] for r in eligible}),
            "recovery_candidate_contexts": [list(v) for v in sorted({(r["scene"], r["branch_after_step"])
                                                                    for r in eligible})],
            "candidate_native_windows_before_deduplication": sum(len(v) for v in groups.values()),
            "candidate_native_windows_after_exact_content_deduplication": len(groups),
            "exact_duplicate_window_groups": [{"sample_sha256": key, "occurrences": value}
                                              for key, value in sorted(groups.items()) if len(value) > 1],
            "successful_ordinary_baseline_windows": sum(r["native_h16_complete_windows"] for r in records
                                                        if r["category"] == "ordinary_successful_baseline"),
            "unique_successful_branch_trajectories": len({r["trajectory_sha256"] for r in eligible})}


def build_audit(root=Path(".")):
    root, bindings = Path(root), {}

    def load(relative, compressed=False, expected=None):
        path = root / relative
        value = sha(path)
        if expected is not None:
            assert value == expected, f"Historical source hash changed: {relative}"
        bindings[Path(relative).as_posix()] = value
        opener = gzip.open if compressed else open
        with opener(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)

    summary = load(DISCOVERY / "summary.json")
    discovery_protocol = load(DISCOVERY / "protocol.json")
    recovery = load(DISCOVERY / "recovery.json")
    pilot = load(PILOT / "protocol.json")
    labels = load(PILOT / "labels.json")
    assert summary["baseline_complete"] and recovery["complete"] and labels["complete"]
    assert discovery_protocol["seeds"] == SCENES
    assert (pilot["config"]["n_obs_steps"], pilot["config"]["horizon"]) == (OBSERVATIONS, HORIZON)
    recovery_keys = [(t["seed"], t["diffusion_seed"], t["arm"]) for t in recovery["trials"]]
    expected_recovery = {(c["seed"], sample, arm) for c in recovery["cases"]
                         for sample in recovery["fresh_diffusion_seeds"]
                         for arm in ["continue_cached", "replan_once"]}
    assert len(recovery_keys) == len(set(recovery_keys)) and set(recovery_keys) == expected_recovery
    label_keys = [(t["checkpoint_id"], t["sample_seed"], t["arm"]) for t in labels["records"]]
    expected_labels = {(c["checkpoint_id"], sample, arm) for c in pilot["cases"]
                       for sample in pilot["label_seeds"] for arm in ["continue", "replan"]}
    assert len(label_keys) == len(set(label_keys)) and set(label_keys) == expected_labels
    for filename in ["robotics_recovery_data_audit.py", "robotics_failure_investigation.py",
                     "robotics_baseline_check.py", "robotics_benefit_pilot.py"]:
        path = Path("experiments") / filename
        bindings[path.as_posix()] = sha(root / path)
    originals, records = {}, []
    for episode in summary["episodes"]:
        relative = Path(episode["source_directory"]) / episode["trace"]
        trace = load(relative, True, pilot["source_fingerprints"][relative.as_posix()])
        rows = verify_baseline(episode, trace)
        assert episode["seed"] not in originals
        originals[episode["seed"]] = episode, trace
        records.append(record_document(f"baseline-{episode['seed']}", relative.as_posix(), episode,
                                       trace["initial"], rows,
                                       "ordinary_successful_baseline" if episode["success"] else "failed_baseline"))
    assert sorted(originals) == SCENES

    def add_branch(trial, directory, source_name, expected_branch, label_record=None):
        sidecar_path = directory / trial["trace"].replace(".trace.json.gz", ".json")
        sidecar = load(sidecar_path)
        assert all(sidecar[key] == value for key, value in trial.items())
        assert sidecar["branch_step"] == expected_branch
        if label_record is not None:
            assert label_record["episode_id"] == sidecar["seed"]
            assert label_record["sample_seed"] == sidecar["diffusion_seed"]
            assert label_record["success"] == sidecar["success"]
            assert label_record["final_coverage"] == sidecar["final_coverage"]
            assert label_record["model_queries"] == sidecar["remaining_model_queries"]
            assert label_record["arm"] == ("continue" if sidecar["arm"] == "continue_cached" else "replan")
        relative = directory / sidecar["trace"]
        trace = load(relative, True, sidecar.get("trace_sha256"))
        original = originals[sidecar["seed"]]
        rows = verify_branch(sidecar, trace, original)
        category = "unsuccessful_selected_continuation"
        if sidecar["success"]:
            category = ("successful_difficult_context_control" if original[0]["success"] else
                        "successful_continuation_of_failed_original")
        record = record_document(f"{source_name}:{relative.stem}", relative.as_posix(), sidecar,
                                 original[1]["initial"], rows, category, sidecar["branch_step"])
        record.update({"original_baseline_success": original[0]["success"], "arm": sidecar["arm"],
                       "prediction_seed": sidecar["diffusion_seed"], "selection_source": source_name,
                       "restoration_evidence": {"recorded_identical_branch_state": True,
                                                "recorded_cached_prefix_check": sidecar["cached_prefix_check"],
                                                "prefix_state_and_cached_rows_verified_by_audit": True,
                                                "new_simulator_or_rgb_replay": False},
                       "prefix_source": (Path(original[0]["source_directory"]) / original[0]["trace"]).as_posix()})
        records.append(record)

    recovery_cases = {c["seed"]: c for c in recovery["cases"]}
    for trial in recovery["trials"]:
        case = recovery_cases[trial["seed"]]
        assert case["baseline_success"] == originals[trial["seed"]][0]["success"]
        add_branch(trial, DISCOVERY / "recovery", "failure-patterns", case["event"]["branch_step"])
        records[-1]["context_reason"] = case["event"]["label"]
    pilot_cases = {c["checkpoint_id"]: c for c in pilot["cases"]}
    for label in labels["records"]:
        case = pilot_cases[label["checkpoint_id"]]
        sidecar = load(PILOT / "labels" / label["trace"].replace(".trace.json.gz", ".json"))
        add_branch(sidecar, PILOT / "labels", "benefit-pilot", case["branch_step"], label)
        records[-1]["context_reason"] = case["reason"]
        records[-1]["prior_pilot_partition"] = ("train" if case["episode_id"] in pilot["train_episode_ids"]
                                                 else "validation")
    paired = defaultdict(dict)
    for record in records:
        if "arm" in record:
            key = (record["selection_source"], record["scene"], record["branch_after_step"],
                   record["prediction_seed"])
            assert record["arm"] not in paired[key]
            paired[key][record["arm"]] = record
    for arms in paired.values():
        assert set(arms) == {"continue_cached", "replan_once"}
        for arm, record in arms.items():
            other = arms["continue_cached" if arm == "replan_once" else "replan_once"]
            record["same_context_draw_other_arm_success"] = other["success"]
            record["paired_outcome_not_causal_proof"] = True
    excluded = []
    for relative, key, field in [(Path("results/robotics-regression-diagnosis/protocol.json"), "trials", "scene_seed"),
                                 (Path("results/robotics-decline-screen/cases.json"), "cases", "scene_seed")]:
        document = load(relative)
        scenes = sorted({row[field] for row in document[key]})
        assert not (set(scenes) & set(SCENES)), "Additional eligible discovery context requires explicit review."
        excluded.append({"source": relative.as_posix(), "scenes": scenes,
                         "reason": "Outside fixed discovery pool 0-29; inspected evaluation, not initial training candidates."})
    totals = aggregate(records)
    for record in records:
        record.pop("window_hashes")
    return {"schema_version": 1, "scope": "Existing discovery action-label provenance only; not a training or benefit result.",
            "selection": {"allowed_scenes": SCENES, "excluded_inspected_evaluation_scenes": [30, 149],
                          "positive_semantics": "Successful complete remaining-episode continuation of an originally failed baseline, at a previously selected difficult context; no new difficulty thresholds.",
                          "ordinary_successes": "Separate potential rehearsal controls, not recovery demonstrations.",
                          "successful_original_controls": "Separate difficult-context success controls, not failure rescues.",
                          "post_branch_only_anchors": True, "prior_partition_not_a_new_holdout": True},
            "sample_alignment": {"horizon": HORIZON, "n_obs_steps": OBSERVATIONS,
                                 "observations_at_anchor_t": ["S[t-1]", "S[t]"],
                                 "executed_action_labels": "movements t..t+15 inclusive; past action at index0, fifteen usable future commands at indices1..15",
                                 "first_usable_future_movement": "t+1", "start_padding": False, "end_padding": False,
                                 "cross_branch_history": "Use original prefix, never duplicate branch image as a cold reset.",
                                 "predicted_chunks_used_as_labels": False},
            "availability": {"executed_commands_and_chronology": "verified from recorded rows",
                             "two_observation_image_hashes_and_pusher_positions": "available by prefix/suffix join",
                             "rgb_bytes_stored_in_traces": False,
                             "rgb_reconstruction": "Not performed. Requires full executed-action CPU replay from original reset and exact recorded RGB/state checks; hashes alone cannot reconstruct pixels.",
                             "latent_policy_rng_history": "Not reconstructed or required for executed-action BC labels; cannot reproduce original stochastic predictions from logs alone.",
                             "training_ready_rgb_tensors": False},
            "summary": totals, "records": records, "excluded_contexts": excluded,
            "source_bindings": dict(sorted(bindings.items())),
            "limitations": ["Outcome-conditioned selection does not prove that any action is optimal or that replanning caused recovery.",
                            "Windows overlap heavily and branches share prefixes/scenes; counts are not independent demonstrations.",
                            "Exact-content duplicate detection uses RGB hashes, observed pusher positions and executed labels, never privileged block pose as model input.",
                            "Distinct hashes do not establish meaningful diversity or statistical adequacy.",
                            "Difficult-context labels were previously inspected development evidence. No fresh generalization claim follows.",
                            "No RGB replay, model loading, gradient update, simulator step, image fit or new policy call occurred."]}


def run(output=OUTPUT, root=Path(".")):
    output, root = Path(output), Path(root)
    assert output.name == OUTPUT.name or output.name.startswith(OUTPUT.name + "-repeat"), \
        "Use the new feasibility directory or an explicitly named repeat."
    document = build_audit(root)
    path = root / output / "data-audit.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == document, "Existing audit changed; do not overwrite."
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir)["summary"], indent=2))


if __name__ == "__main__":
    main()
