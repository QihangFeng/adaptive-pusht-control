"""CPU-only contracts for retrospective lost-success diagnostics."""

import copy
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_loss_review as review


def state(coverage=.8, block=(256., 256.), angle=math.pi / 4,
          pusher=(0., 0.)):
    return {"coverage": coverage, "block_pos": list(block), "block_angle": angle,
            "pusher_pos": list(pusher), "image_sha256": "unchanged-image"}


def row(step, coverage=.8, block=(256., 256.), angle=math.pi / 4,
        pusher=None, action=None, fresh=False, features=None):
    result = {"step": step, "new_prediction": fresh,
              "action": list(action if action is not None else (float(step), 0.)),
              "after": state(coverage, block, angle,
                             pusher if pusher is not None else (float(step), 0.))}
    if features is not None:
        result["features"] = features
    return result


def trace(values, initial=.8):
    return {"initial": state(initial),
            "steps": [row(i, value, fresh=(i - 1) % 8 == 0)
                      for i, value in enumerate(values, start=1)]}


def episode(*events):
    return {"events": list(events)}


def temp_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="loss-review-cpu-", dir=parent)


def write_fixture(directory, values=None, success=True):
    """A hash-bound synthetic record; never invokes policy, physics or camera."""
    values = values if values is not None else [.8] * 15 + [.96]
    document = trace(values)
    lengths = [min(8, len(values) - index) for index in range(0, len(values), 8)]
    record = {"seed": 70, "method": "fixed8", "success": success,
              "truncated": not success, "steps": len(values),
              "model_queries": len(lengths), "initial": copy.deepcopy(document["initial"]),
              "unit_lengths": lengths, "final_coverage": values[-1],
              "events": [], "trace": "seed-70-fixed8.trace.json.gz"}
    path = directory / "seed-70-fixed8.json"
    write_record(path, record, document)
    return path, record, document


def write_record(path, record, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.parent / record["trace"]
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        json.dump(document, stream)
    record["trace_sha256"] = review.previous.sha(target)
    path.write_text(json.dumps(record), encoding="utf-8")


class PoseDiagnosticTest(unittest.TestCase):
    def test_goal_error_uses_body_origin_and_wrapped_full_rotation(self):
        result = review.pose_error(state(block=(259., 260.),
                                         angle=review.GOAL[2] + 2 * math.pi))
        self.assertEqual(result["position_error"], 5.)
        self.assertAlmostEqual(result["orientation_error_radians"], 0.)
        self.assertAlmostEqual(result["orientation_error_degrees"], 0.)

    def test_t_shape_has_no_half_turn_symmetry(self):
        result = review.pose_error(state(angle=review.GOAL[2] + math.pi))
        self.assertAlmostEqual(result["orientation_error_radians"], math.pi)
        self.assertAlmostEqual(result["orientation_error_degrees"], 180.)
        self.assertAlmostEqual(review.angle_distance(3.1, -3.1), 2 * math.pi - 6.2)


class TerminalWindowTest(unittest.TestCase):
    def test_last_48_movements_include_the_state_before_the_first(self):
        document = trace([i / 100 for i in range(1, 61)], initial=0.)
        for r in document["steps"]:
            r["after"]["block_pos"] = [float(r["step"]), 0.]
        document["initial"]["block_pos"] = [0., 0.]
        result = review.terminal_window(document)
        self.assertEqual((result["start_before_step"], result["end_after_step"]), (12, 60))
        self.assertEqual(result["observed_movements"], 48)
        self.assertEqual(result["start_coverage"], .12)
        self.assertEqual(result["end_coverage"], .60)
        self.assertAlmostEqual(result["net_change"], .48)
        self.assertEqual(result["model_queries"], 6)
        self.assertEqual(result["block_path"], 48.)
        self.assertEqual(result["block_endpoint_displacement"], 48.)
        self.assertEqual(result["pusher_path"], 48.)
        self.assertEqual(result["command_path"], 48.)

    def test_short_success_has_only_observed_movements_not_a_full_window(self):
        document = trace([.85, .9, .96], initial=.8)
        result = review.terminal_window(document)
        self.assertEqual(result["start_before_step"], 0)
        self.assertEqual(result["end_after_step"], 3)
        self.assertEqual(result["observed_movements"], 3)
        self.assertEqual(result["minimum_coverage"], .8)
        self.assertEqual(result["maximum_coverage"], .96)
        self.assertAlmostEqual(result["best_gain"], .16)
        # There is no invented command before the first prediction.
        self.assertEqual(result["command_path"], 2.)
        self.assertEqual(result["pusher_path"], 3.)

    def test_round_trip_has_positive_path_but_zero_endpoint_displacement(self):
        document = trace([.7, .8])
        document["steps"][0]["after"]["block_pos"] = [259., 260.]
        result = review.terminal_window(document)
        self.assertEqual(result["block_path"], 10.)
        self.assertEqual(result["block_endpoint_displacement"], 0.)
        self.assertAlmostEqual(result["coverage_range"], .1)
        self.assertAlmostEqual(result["net_change"], 0.)

    def test_rotation_path_wraps_each_movement_and_differs_from_endpoint(self):
        document = trace([.8, .8])
        document["initial"]["block_angle"] = 3.1
        document["steps"][0]["after"]["block_angle"] = -3.1
        document["steps"][1]["after"]["block_angle"] = 3.1
        result = review.terminal_window(document)
        self.assertAlmostEqual(result["block_rotation_path_radians"], 2 * (2 * math.pi - 6.2))

    def test_width_one_uses_previous_coverage_command_and_physical_state(self):
        document = trace([.7, .9], initial=.8)
        result = review.terminal_window(document, 1)
        self.assertEqual(result["start_before_step"], 1)
        self.assertEqual(result["observed_movements"], 1)
        self.assertEqual(result["start_coverage"], .7)
        self.assertEqual(result["command_path"], 1.)
        self.assertEqual(result["pusher_path"], 1.)
        self.assertEqual(result["model_queries"], 0)

    def test_invalid_window_and_empty_trace_are_rejected(self):
        for width in (0, -1, 1.5):
            with self.subTest(width=width), self.assertRaises(AssertionError):
                review.terminal_window(trace([.8]), width)
        with self.assertRaises(AssertionError):
            review.terminal_window(trace([]))

    def test_diagnostics_do_not_mutate_saved_trace(self):
        document = trace([.9, .7, .91])
        original = copy.deepcopy(document)
        review.terminal_window(document)
        review.coverage_drawdown(document)
        review.stationary_tail(document)
        review.episode_diagnostics(episode(), document)
        self.assertEqual(document, original)


class CoverageDrawdownTest(unittest.TestCase):
    def test_largest_drop_tracks_its_peak_trough_and_later_recovery(self):
        result = review.coverage_drawdown(trace([.7, .9, .75, .91], initial=.8))
        self.assertEqual((result["peak_step"], result["trough_step"]), (2, 3))
        self.assertAlmostEqual(result["drawdown"], .15)
        self.assertEqual(result["first_regained_step"], 4)
        self.assertEqual(result["regained_coverage"], .91)
        # Recovering 90% overlap is not passing the >95% completion criterion.
        self.assertLess(result["regained_coverage"], .95)

    def test_initial_peak_and_equal_recovery_are_included(self):
        result = review.coverage_drawdown(trace([.7, .85, .9], initial=.9))
        self.assertEqual((result["peak_step"], result["trough_step"]), (0, 1))
        self.assertEqual(result["first_regained_step"], 3)
        self.assertEqual(result["regained_coverage"], .9)

    def test_recovery_cannot_be_a_high_value_before_the_selected_trough(self):
        result = review.coverage_drawdown(trace([.91, .4, .7], initial=.9))
        self.assertEqual((result["peak_step"], result["trough_step"]), (1, 2))
        self.assertIsNone(result["first_regained_step"])
        self.assertIsNone(result["regained_coverage"])

    def test_running_peak_not_unrelated_global_minimum_defines_largest_drop(self):
        result = review.coverage_drawdown(trace([.2, .9, .5, .92], initial=.25))
        self.assertEqual((result["peak_step"], result["trough_step"]), (2, 3))
        self.assertAlmostEqual(result["drawdown"], .4)
        self.assertEqual(result["first_regained_step"], 4)

    def test_no_drop_has_no_recovery_event(self):
        result = review.coverage_drawdown(trace([.8, .9], initial=.7))
        self.assertEqual(result["drawdown"], 0.)
        self.assertIsNone(result["first_regained_step"])
        self.assertIsNone(result["regained_coverage"])


class StationaryTailTest(unittest.TestCase):
    def test_stationary_object_can_have_moving_pusher_and_many_predictions(self):
        document = trace([.94] * 30, initial=.94)
        result = review.stationary_tail(document)
        self.assertEqual(result["movements"], 30)
        self.assertEqual(result["start_before_step"], 0)
        self.assertEqual(result["model_queries"], 4)
        self.assertEqual(result["pusher_path"], 30.)
        self.assertEqual(result["command_path"], 29.)

    def test_tail_starts_after_last_real_object_motion(self):
        document = trace([.9] * 5, initial=.9)
        for r in document["steps"][2:]:
            r["after"]["block_pos"] = [257., 256.]
        result = review.stationary_tail(document)
        self.assertEqual(result["start_before_step"], 3)
        self.assertEqual(result["movements"], 2)
        self.assertEqual(result["pusher_path"], 2.)

    def test_angle_wrap_is_stationary_but_a_real_rotation_breaks_tail(self):
        document = trace([.9] * 3, initial=.9)
        for index, r in enumerate(document["steps"]):
            r["after"]["block_angle"] += (index + 1) * 2 * math.pi
        self.assertEqual(review.stationary_tail(document)["movements"], 3)
        document["steps"][-1]["after"]["block_angle"] += .01
        result = review.stationary_tail(document)
        self.assertEqual(result["movements"], 0)
        self.assertEqual(result["pusher_path"], 0.)


class StationaryRunTest(unittest.TestCase):
    def test_nonterminal_plateau_in_successful_episode_is_not_hidden(self):
        document = trace([.94] * 30 + [.96], initial=.94)
        document["steps"][-1]["after"]["block_pos"] = [257., 256.]
        result = review.stationary_runs(document)
        self.assertEqual(result, [{"start_before_step": 0, "end_after_step": 30,
                                   "movements": 30, "coverage": .94, "model_queries": 4}])
        self.assertEqual(review.stationary_tail(document)["movements"], 0)
        self.assertEqual(review.episode_diagnostics(episode(), document)["near_stationary_runs"], result)

    def test_each_maximal_run_includes_exactly_its_stationary_movements(self):
        document = trace([.94] * 51, initial=.94)
        for r in document["steps"][24:50]:
            r["after"]["block_pos"] = [257., 256.]
        document["steps"][50]["after"]["block_pos"] = [258., 256.]
        result = review.stationary_runs(document)
        self.assertEqual([(r["start_before_step"], r["end_after_step"], r["movements"])
                          for r in result], [(0, 24, 24), (25, 50, 25)])
        self.assertEqual([r["model_queries"] for r in result], [3, 3])

    def test_minimum_length_is_inclusive_and_near_cutoff_is_inclusive(self):
        self.assertEqual(review.stationary_runs(trace([.9] * 23, initial=.9)), [])
        run = review.stationary_runs(trace([.9] * 24, initial=.9))[0]
        self.assertEqual(run["movements"], 24)
        self.assertEqual(run["coverage"], .9)
        self.assertEqual(review.stationary_runs(trace([.89] * 30, initial=.89)), [])
        for minimum in (0, -1, 1.5):
            with self.subTest(minimum=minimum), self.assertRaises(AssertionError):
                review.stationary_runs(trace([.9]), minimum)

    def test_stationary_tolerance_is_local_per_movement_not_total_displacement(self):
        document = trace([.94] * 30, initial=.94)
        for r in document["steps"]:
            r["after"]["block_pos"] = [256. + r["step"] * 5e-10, 256.]
        self.assertEqual(review.stationary_runs(document)[0]["movements"], 30)
        self.assertGreater(review.terminal_window(document)["block_endpoint_displacement"],
                           review.STATIONARY_EPSILON)


class EpisodeDiagnosticTest(unittest.TestCase):
    def test_near_duration_is_final_consecutive_movements_not_all_near_rows(self):
        document = trace([.9, .91, .89, .9, .94], initial=.85)
        result = review.episode_diagnostics(episode(), document)
        self.assertEqual(result["first_near_step"], 1)
        self.assertEqual(result["terminal_near_duration"], 2)
        self.assertEqual(result["peak_step"], 5)

    def test_initial_peak_and_near_observation_are_step_zero_not_movements(self):
        result = review.episode_diagnostics(episode(), trace([.9, .91], initial=.94))
        self.assertEqual(result["peak_step"], 0)
        self.assertEqual(result["peak_coverage"], .94)
        self.assertEqual(result["first_near_step"], 0)
        self.assertEqual(result["terminal_near_duration"], 2)
        self.assertEqual(result["peak_goal_error"]["position_error"], 0.)

    def test_camera_invalid_missing_and_nonfinite_rows_are_not_scored(self):
        document = trace([.9] * 6)
        features = [{"quality": False, "coverage": 1.},
                    {"quality": True, "coverage": float("nan")},
                    {"quality": True, "coverage": float("inf")},
                    {"quality": True, "coverage": None},
                    {"quality": True, "coverage": .96}]
        for r, feature in zip(document["steps"], features):
            r["features"] = feature
        # Sixth terminal row deliberately has no features, as in saved traces.
        camera = review.episode_diagnostics(episode(), document)["camera"]
        self.assertEqual(camera["valid_rows"], 1)
        self.assertEqual(camera["false_completion_rows"], 1)
        self.assertAlmostEqual(camera["maximum_error"], .06)
        self.assertEqual(camera["peak"], {"step": 5, "camera_coverage": .96,
                                         "true_coverage_same_step": .9})

    def test_camera_false_completion_is_strictly_over_95_truth_not_over_95(self):
        document = trace([.9, .95, .96])
        for r, value in zip(document["steps"], [.95, .96, .97]):
            r["features"] = {"quality": True, "coverage": value}
        camera = review.episode_diagnostics(episode(), document)["camera"]
        self.assertEqual(camera["valid_rows"], 3)
        self.assertEqual(camera["false_completion_rows"], 1)
        self.assertEqual(camera["peak"]["true_coverage_same_step"], .96)

    def test_no_camera_evidence_gives_none_not_zero_error(self):
        camera = review.episode_diagnostics(episode(), trace([.8, .9]))["camera"]
        self.assertEqual(camera["valid_rows"], 0)
        self.assertIsNone(camera["maximum_error"])
        self.assertIsNone(camera["peak"])

    def test_interruption_records_next_step_and_stops_at_next_prediction(self):
        document = trace([.8, .7, .71, .75, .8, .9])
        document["steps"][2]["new_prediction"] = True
        document["steps"][4]["new_prediction"] = True
        result = review.episode_diagnostics(episode({"kind": "extend", "step": 1},
                                                     {"kind": "interrupt", "step": 2}), document)
        self.assertEqual(result["extensions"], 1)
        self.assertEqual(result["last_extension_step"], 1)
        self.assertEqual(result["interruptions"], [2])
        window = result["event_windows"][0]
        self.assertEqual(window["interrupt_after_step"], 2)
        self.assertEqual(window["new_plan_starts_at"], 3)
        self.assertEqual(window["following_steps"], [3, 4])
        self.assertEqual(window["outcome"]["start_coverage"], .7)
        self.assertEqual(window["outcome"]["end_coverage"], .75)


class GroupsAndPrefixTest(unittest.TestCase):
    def test_all_four_outcome_groups_include_shared_failures(self):
        cases = {(True, True): "retained_baseline_success",
                 (True, False): "lost_baseline_success",
                 (False, True): "rescued_baseline_failure",
                 (False, False): "shared_failure"}
        for outcomes, expected in cases.items():
            with self.subTest(outcomes=outcomes):
                self.assertEqual(review.outcome_group(*outcomes), expected)

    def test_later_state_divergence_is_not_a_matched_state_alternative(self):
        left, right = trace([.8] * 12), trace([.8] * 12)
        right["steps"][8]["action"] = [-1., -1.]
        self.assertEqual(review.common_prefix(left, right), 8)
        self.assertNotEqual(left["steps"][8]["action"], right["steps"][8]["action"])

    def test_only_existing_truth_tolerance_is_permitted_in_prefix(self):
        left, right = trace([.8] * 12), trace([.8] * 12)
        right["initial"]["coverage"] += 5e-13
        right["steps"][3]["after"]["coverage"] += 5e-13
        self.assertEqual(review.common_prefix(left, right), 12)
        right["steps"][3]["after"]["coverage"] += 2e-12
        with self.assertRaises(AssertionError):
            review.common_prefix(left, right)

    def test_changed_coordinate_rgb_action_initial_or_short_prefix_fails(self):
        changes = (lambda t: t["initial"]["block_pos"].__setitem__(0, 256.000000000001),
                   lambda t: t["steps"][3]["after"].update(image_sha256="new-rgb"),
                   lambda t: t["steps"][3]["action"].__setitem__(0, 4.000000000001),
                   lambda t: t["steps"].__delitem__(slice(7, None)))
        for change in changes:
            left, right = trace([.8] * 12), trace([.8] * 12)
            change(right)
            with self.subTest(change=change), self.assertRaises(AssertionError):
                review.common_prefix(left, right)

    def test_source_route_separates_old_new_baselines_and_reused_adaptive(self):
        self.assertEqual(review.source_sidecar(54, "adaptive12"),
                         review.ORIGINAL / "episodes/seed-54-adaptive12.json")
        self.assertEqual(review.source_sidecar(78, "adaptive12"),
                         review.GATE / "episodes/seed-78-adaptive12.json")
        self.assertEqual(review.source_sidecar(78, "fixed8"),
                         review.COHORT / "episodes/seed-78-fixed8.json")


class SavedRecordContractTest(unittest.TestCase):
    def test_hash_bound_read_records_source_without_mutating_original(self):
        with temp_directory() as name:
            path, record, document = write_fixture(Path(name))
            original = copy.deepcopy(record)
            bindings = {}
            with patch.object(review, "source_sidecar", return_value=path):
                found, loaded = review.read_record(record, bindings)
            self.assertEqual(found, record)
            self.assertEqual(loaded, document)
            self.assertEqual(record, original)
            self.assertEqual(bindings[path.as_posix()], review.previous.sha(path))
            self.assertEqual(len(bindings), 2)

    def test_exact95_is_a_timeout_not_a_success(self):
        with temp_directory() as name:
            path, record, _ = write_fixture(Path(name), [.95] * 300, success=False)
            with patch.object(review, "source_sidecar", return_value=path):
                review.read_record(record, {})
                for changes in ({"success": True}, {"truncated": False}, {"steps": 299}):
                    wrong = {**record, **changes}
                    with self.subTest(changes=changes), self.assertRaises(AssertionError):
                        review.read_record(wrong, {})

    def test_first_threshold_crossing_must_be_terminal(self):
        with temp_directory() as name:
            path, record, document = write_fixture(Path(name))
            document["steps"][5]["after"]["coverage"] = .951
            write_record(path, record, document)
            with patch.object(review, "source_sidecar", return_value=path), self.assertRaises(AssertionError):
                review.read_record(record, {})

    def test_nonfinite_coverage_step_count_query_count_and_unit_count_fail(self):
        changes = (lambda r, t: t["steps"][0]["after"].update(coverage=float("nan")),
                   lambda r, t: t["steps"][0].update(step=2),
                   lambda r, t: r.update(model_queries=3),
                   lambda r, t: r.update(unit_lengths=[8, 7]),
                   lambda r, t: t["initial"].update(block_angle=0.),
                   lambda r, t: r.update(final_coverage=.97))
        for change in changes:
            with temp_directory() as name:
                path, record, document = write_fixture(Path(name))
                change(record, document)
                write_record(path, record, document)
                with patch.object(review, "source_sidecar", return_value=path), self.assertRaises(AssertionError):
                    review.read_record(record, {})

    def test_trace_path_hash_seed_method_and_reference_routing_fail(self):
        with temp_directory() as name:
            path, record, _ = write_fixture(Path(name))
            changes = ({"trace": "../unrelated.trace.json.gz"}, {"trace_sha256": "0" * 64},
                       {"seed": 90}, {"method": "weighted"},
                       {"source_sidecar": "wrong/path.json", "trace_directory": "wrong"})
            with patch.object(review, "source_sidecar", return_value=path):
                for change in changes:
                    with self.subTest(change=change), self.assertRaises(AssertionError):
                        review.read_record({**record, **change}, {})


class OutputIsolationTest(unittest.TestCase):
    def test_only_dedicated_and_exact_repeat_directories_are_allowed(self):
        review.check_output(review.OUTPUT)
        review.check_output(review.OUTPUT.with_name(review.OUTPUT.name + "-repeat"))
        forbidden = [review.COHORT, review.ORIGINAL, review.GATE, review.OUTPUT.parent,
                     review.OUTPUT / "nested", review.OUTPUT / "analysis.json",
                     review.OUTPUT.with_name(review.OUTPUT.name + "-repeat-repeat")]
        for destination in forbidden:
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                review.check_output(destination)

    def test_completed_review_verifies_without_rewriting_and_mismatch_cannot_overwrite(self):
        with temp_directory() as name:
            destination = Path(name) / "derived"
            with patch.object(review, "OUTPUT", destination):
                review.save_review({"complete": True, "count": 120}, destination)
                path = destination / "analysis.json"
                before = path.read_bytes()
                timestamp = path.stat().st_mtime_ns
                review.save_review({"complete": True, "count": 120}, destination)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(path.stat().st_mtime_ns, timestamp)
                with self.assertRaises(AssertionError):
                    review.save_review({"complete": True, "count": 119}, destination)
                self.assertEqual(path.read_bytes(), before)


class FrozenBuildContractTest(unittest.TestCase):
    def test_duplicate_or_missing_outcome_records_are_rejected_before_analysis(self):
        actual_read = review.previous.read
        for kind in ("duplicate", "missing"):
            def changed_read(path):
                result = actual_read(path)
                if Path(path) == review.COHORT / "episodes.json":
                    result = copy.deepcopy(result)
                    if kind == "duplicate":
                        result["episodes"][-1] = result["episodes"][0]
                    else:
                        result["episodes"].pop()
                return result
            with self.subTest(kind=kind), patch.object(review.previous, "read", side_effect=changed_read), \
                    self.assertRaises(AssertionError):
                review.build_review()

    def test_changed_transitive_frozen_fingerprint_is_rejected(self):
        actual_read = review.previous.read
        def changed_read(path):
            result = actual_read(path)
            if Path(path) == review.COHORT / "protocol.json":
                result = copy.deepcopy(result)
                first = next(iter(result["fingerprints"]))
                result["fingerprints"][first] = "0" * 64
            return result
        with patch.object(review.previous, "read", side_effect=changed_read), self.assertRaises(AssertionError):
            review.build_review()

    def test_any_failed_bound_source_audit_is_rejected(self):
        actual_read = review.previous.read
        def changed_read(path):
            result = actual_read(path)
            if Path(path) == review.COHORT / "independent-audit.json":
                result = {**result, "passed": False}
            return result
        with patch.object(review.previous, "read", side_effect=changed_read), self.assertRaises(AssertionError):
            review.build_review()


if __name__ == "__main__":
    unittest.main()
