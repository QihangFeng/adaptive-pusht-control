"""Standard-library, in-memory tests; no simulator, image fit or model calls."""

import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_recovery_data_audit as audit


def state(step, coverage=.2):
    return {"image_sha256": f"{step:064x}", "pusher_pos": [float(step), 10.],
            "coverage": coverage, "block_pos": [20., 30.]}


def baseline(count=32, success=True):
    initial = state(0)
    rows = []
    for step in range(1, count + 1):
        rows.append({"step": step, "action": [step + .5, 10.],
                     "new_prediction": (step - 1) % 8 == 0,
                     "pending_actions": (8 - step % 8) % 8,
                     "before": copy.deepcopy(initial if step == 1 else rows[-1]["after"]),
                     "after": state(step, .96 if success and step == count else .2)})
    episode = {"seed": 2, "steps": count, "success": success,
               "final_coverage": rows[-1]["after"]["coverage"],
               "model_queries": sum(r["new_prediction"] for r in rows)}
    return episode, {"initial": initial, "steps": rows}


def continuation(arm="replan_once", count=32, branch=8):
    original = baseline(count, False)
    rows = copy.deepcopy(original[1]["steps"][branch:])
    rows[-1]["after"]["coverage"] = .96
    trial = {"seed": 2, "branch_step": branch, "arm": arm, "success": True,
             "identical_branch_state": True,
             "cached_prefix_check": "verified" if arm == "continue_cached" else "not applicable",
             "remaining_actions_executed": len(rows), "total_actions": count,
             "remaining_model_queries": sum(r["new_prediction"] for r in rows),
             "total_model_queries": original[0]["model_queries"], "final_coverage": .96}
    trace = {"branch_state": original[1]["steps"][branch - 1]["after"], "steps": rows}
    return trial, trace, original


class NativeWindowTests(unittest.TestCase):
    def test_two_observations_and_one_past_fifteen_future_actions(self):
        _, trace = baseline()
        window = next(audit.training_windows(trace["initial"], trace["steps"], 8))
        self.assertEqual(window["observation_after_steps"], [7, 8])
        self.assertEqual(window["executed_label_steps"], [8, 23])
        self.assertEqual(window["anchor_after_step"], 8)

    def test_sixteen_executed_actions_supply_exactly_one_unpadded_window(self):
        _, trace = baseline(16)
        windows = list(audit.training_windows(trace["initial"], trace["steps"]))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["observation_after_steps"], [0, 1])
        self.assertEqual(windows[0]["executed_label_steps"], [1, 16])

    def test_fifteen_actions_do_not_silently_pad_a_label(self):
        _, trace = baseline(15)
        self.assertEqual(list(audit.training_windows(trace["initial"], trace["steps"])), [])

    def test_branch_context_uses_original_preceding_image_not_duplicate_reset(self):
        trial, branch, original = continuation(branch=9)
        joined = audit.join_recovery_rows(original[1]["steps"], branch["steps"], branch["branch_state"], 9)
        sample = next(audit.training_windows(original[1]["initial"], joined, 9))
        expected = {"rgb_sha256": [state(8)["image_sha256"], state(9)["image_sha256"]],
                    "pusher_positions": [state(8)["pusher_pos"], state(9)["pusher_pos"]],
                    "executed_actions": [r["action"] for r in joined[8:24]]}
        self.assertEqual(sample["sample_sha256"], audit.fingerprint(expected))

    def test_predictions_are_not_labels(self):
        _, trace = baseline(16)
        original = next(audit.training_windows(trace["initial"], trace["steps"]))
        for row in trace["steps"]:
            row["predicted_commands"] = [[999., 999.]] * 16
        self.assertEqual(next(audit.training_windows(trace["initial"], trace["steps"])), original)

    def test_end_anchor_includes_terminal_command_without_going_past_it(self):
        _, trace = baseline(32)
        windows = list(audit.training_windows(trace["initial"], trace["steps"]))
        self.assertEqual(len(windows), 17)
        self.assertEqual(windows[-1]["executed_label_steps"], [17, 32])

    def test_changes_in_pusher_rgb_or_executed_action_change_sample_fingerprint(self):
        for kind in ["pusher", "rgb", "action"]:
            with self.subTest(kind=kind):
                _, trace = baseline(16)
                first = next(audit.training_windows(trace["initial"], trace["steps"]))
                if kind == "pusher": trace["initial"]["pusher_pos"][0] += 1
                if kind == "rgb": trace["initial"]["image_sha256"] = "f" * 64
                if kind == "action": trace["steps"][3]["action"][0] += 1
                changed = next(audit.training_windows(trace["initial"], trace["steps"]))
                self.assertNotEqual(first["sample_sha256"], changed["sample_sha256"])

    def test_truth_is_not_part_of_runtime_sample_identity(self):
        _, trace = baseline(16)
        first = next(audit.training_windows(trace["initial"], trace["steps"]))
        trace["initial"]["block_pos"] = [999., 999.]
        trace["initial"]["coverage"] = .8
        self.assertEqual(first, next(audit.training_windows(trace["initial"], trace["steps"])))


class ProvenanceTests(unittest.TestCase):
    def test_baseline_chronology_and_termination(self):
        episode, trace = baseline()
        self.assertEqual(audit.verify_baseline(episode, trace), trace["steps"])

    def test_baseline_mismatching_before_state_is_rejected(self):
        episode, trace = baseline()
        trace["steps"][4]["before"]["pusher_pos"][0] += 1
        with self.assertRaises(AssertionError): audit.verify_baseline(episode, trace)

    def test_earlier_success_before_final_row_is_rejected(self):
        episode, trace = baseline()
        trace["steps"][3]["after"]["coverage"] = .96
        trace["steps"][4]["before"]["coverage"] = .96
        with self.assertRaises(AssertionError): audit.verify_baseline(episode, trace)

    def test_scene_thirty_is_not_in_discovery_pool(self):
        episode, trace = baseline()
        episode["seed"] = 30
        with self.assertRaises(AssertionError): audit.verify_baseline(episode, trace)

    def test_exact_coverage_threshold_is_not_success(self):
        episode, trace = baseline()
        trace["steps"][-1]["after"]["coverage"] = .95
        episode["final_coverage"] = .95
        with self.assertRaises(AssertionError): audit.verify_baseline(episode, trace)

    def test_recorded_branch_restoration_flag_required(self):
        trial, trace, original = continuation()
        trial["identical_branch_state"] = False
        with self.assertRaises(AssertionError): audit.verify_branch(trial, trace, original)

    def test_join_preserves_full_prefix_and_absolute_suffix_indices(self):
        trial, trace, original = continuation()
        joined = audit.verify_branch(trial, trace, original)
        self.assertEqual(len(joined), 32)
        self.assertEqual(joined[:8], original[1]["steps"][:8])
        self.assertEqual(joined[8:], trace["steps"])

    def test_mismatching_branch_rgb_is_rejected(self):
        trial, trace, original = continuation()
        trace["branch_state"] = copy.deepcopy(trace["branch_state"])
        trace["branch_state"]["image_sha256"] = "f" * 64
        with self.assertRaises(AssertionError): audit.verify_branch(trial, trace, original)

    def test_only_coverage_has_existing_rounding_tolerance(self):
        trial, trace, original = continuation()
        trace["branch_state"] = copy.deepcopy(trace["branch_state"])
        trace["branch_state"]["coverage"] += 5e-13
        audit.verify_branch(trial, trace, original)
        trace["branch_state"]["pusher_pos"][0] += 1e-12
        with self.assertRaises(AssertionError): audit.verify_branch(trial, trace, original)

    def test_missing_or_duplicate_suffix_steps_rejected(self):
        for value in [8, 10]:
            trial, trace, original = continuation()
            trace["steps"][0]["step"] = value
            with self.assertRaises(AssertionError): audit.verify_branch(trial, trace, original)

    def test_cached_commands_checked_against_original_prefix_future(self):
        trial, trace, original = continuation("continue_cached", branch=9)
        audit.verify_branch(trial, trace, original)
        trace["steps"][0]["action"][0] += 1
        with self.assertRaises(AssertionError): audit.verify_branch(trial, trace, original)

    def test_replan_branch_must_actually_request_new_prediction(self):
        trial, trace, original = continuation(branch=9)
        with self.assertRaises(AssertionError): audit.verify_branch(trial, trace, original)

    def test_nonfinite_or_wrong_shape_commands_rejected(self):
        for action in [[math.nan, 0.], [math.inf, 0.], [1.], [1., 2., 3.]]:
            _, trace = baseline()
            trace["steps"][0]["action"] = action
            with self.assertRaises(AssertionError): audit.check_rows(trace["steps"])


class AggregationAndOutputTests(unittest.TestCase):
    def make_record(self, identifier, category):
        episode, trace = baseline()
        return audit.record_document(identifier, "trace", episode, trace["initial"], trace["steps"], category, 8)

    def test_ordinary_success_and_successful_controls_not_called_recoveries(self):
        records = [self.make_record("recovery", "successful_continuation_of_failed_original"),
                   self.make_record("baseline", "ordinary_successful_baseline"),
                   self.make_record("control", "successful_difficult_context_control")]
        summary = audit.aggregate(records)
        self.assertEqual(summary["recovery_candidate_records"], 1)
        self.assertEqual(summary["candidate_native_windows_before_deduplication"], 10)

    def test_exact_content_duplicates_removed_not_just_same_file_hash(self):
        records = [self.make_record(x, "successful_continuation_of_failed_original") for x in ["a", "b"]]
        summary = audit.aggregate(records)
        self.assertEqual(summary["candidate_native_windows_before_deduplication"], 20)
        self.assertEqual(summary["candidate_native_windows_after_exact_content_deduplication"], 10)
        self.assertEqual(len(summary["exact_duplicate_window_groups"]), 10)
        self.assertEqual(summary["unique_successful_branch_trajectories"], 1)

    def test_failed_continuations_excluded_positive_pool(self):
        summary = audit.aggregate([self.make_record("f", "unsuccessful_selected_continuation")])
        self.assertEqual(summary["recovery_candidate_records"], 0)
        self.assertEqual(summary["candidate_native_windows_after_exact_content_deduplication"], 0)

    def test_aggregate_roundtrips_json_without_tuple_list_mismatch(self):
        records = [self.make_record("r", "successful_continuation_of_failed_original")]
        summary = audit.aggregate(records)
        self.assertEqual(summary, json.loads(json.dumps(summary)))

    def test_short_success_records_are_not_padded(self):
        episode, trace = baseline(16)
        record = audit.record_document("short", "trace", episode, trace["initial"], trace["steps"],
                                       "successful_continuation_of_failed_original", 9)
        self.assertEqual(record["native_h16_complete_windows"], 0)
        self.assertIsNone(record["first_complete_window"])

    def test_new_output_created_and_matching_output_verified(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, "build_audit", return_value={"a": 1}):
            output = audit.OUTPUT
            audit.run(output, Path(directory))
            path = Path(directory) / output / "data-audit.json"
            first = path.read_bytes()
            self.assertEqual(audit.run(output, Path(directory)), {"a": 1})
            self.assertEqual(path.read_bytes(), first)

    def test_changed_existing_audit_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(audit, "build_audit", return_value={"a": 1}):
            audit.run(audit.OUTPUT, Path(directory))
            path = Path(directory) / audit.OUTPUT / "data-audit.json"
            path.write_text('{"a": 2}', encoding="utf-8")
            with self.assertRaises(AssertionError): audit.run(audit.OUTPUT, Path(directory))
            self.assertEqual(json.loads(path.read_text()), {"a": 2})

    def test_earlier_experiment_directory_cannot_be_output(self):
        with self.assertRaises(AssertionError): audit.run(Path("results/robotics-failure-patterns"))

    def test_source_is_standard_library_only(self):
        source = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        modules = {node.module.split(".")[0] for node in ast.walk(source) if isinstance(node, ast.ImportFrom)}
        modules.update(alias.name.split(".")[0] for node in ast.walk(source)
                       if isinstance(node, ast.Import) for alias in node.names)
        self.assertTrue(modules <= {"__future__", "argparse", "collections", "gzip", "hashlib", "json", "math", "pathlib"})


if __name__ == "__main__":
    unittest.main()
