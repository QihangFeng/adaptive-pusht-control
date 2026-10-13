"""Synthetic CPU checks of retrospective command windows and output isolation."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_decision_review as review


def state(coverage=.8, pusher=(0., 0.), velocity=(0., 0.), gap=0.):
    return {"coverage": coverage, "pusher_pos": list(pusher),
            "pusher_velocity": list(velocity), "pusher_block_gap": gap,
            "block_pos": [0., 0.], "block_angle": 0.}


def row(step, coverage=.8, action=None, fresh=False, age=1, **changes):
    result = {"step": step, "new_prediction": fresh, "plan_age": age,
              "pending_before": 4, "pending_after": 4,
              "action": list(action if action is not None else (float(step), 0.)),
              "features": {"quality": True, "coverage": coverage},
              "after": state(coverage)}
    result.update(changes)
    return result


def episode(*events):
    return {"events": list(events)}


class WindowStatsTest(unittest.TestCase):
    def test_net_drop_is_distinct_from_transient_dip_and_recovery(self):
        result = review.window_stats(state(.8), [row(1, .65), row(2, .83)])
        self.assertEqual(result["observed_movements"], 2)
        self.assertAlmostEqual(result["net_change"], .03)
        self.assertAlmostEqual(result["minimum_change_from_start"], -.15)
        self.assertAlmostEqual(result["maximum_drawdown"], .15)
        self.assertAlmostEqual(result["largest_single_drop"], .15)

    def test_drawdown_uses_running_peak_not_only_start_or_final_value(self):
        result = review.window_stats(state(.5), [row(1, .6), row(2, .4), row(3, .55)])
        self.assertAlmostEqual(result["net_change"], .05)
        self.assertAlmostEqual(result["minimum_change_from_start"], -.1)
        self.assertAlmostEqual(result["maximum_drawdown"], .2)
        self.assertAlmostEqual(result["largest_single_drop"], .2)

    def test_accumulated_decline_is_not_the_largest_single_drop(self):
        result = review.window_stats(state(.8), [row(1, .75), row(2, .70), row(3, .65)])
        self.assertAlmostEqual(result["net_change"], -.15)
        self.assertAlmostEqual(result["maximum_drawdown"], .15)
        self.assertAlmostEqual(result["largest_single_drop"], .05)

    def test_empty_window_preserves_start_and_has_no_decline(self):
        result = review.window_stats(state(.8), [])
        self.assertEqual(result, {"observed_movements": 0, "start_coverage": .8,
                                 "end_coverage": .8, "net_change": 0.,
                                 "minimum_change_from_start": 0.,
                                 "maximum_drawdown": 0., "largest_single_drop": 0.})

    def test_terminal_single_movement_and_inputs_are_not_modified(self):
        start, rows = state(.94), [row(1, .96)]
        original = copy.deepcopy((start, rows))
        result = review.window_stats(start, rows)
        self.assertAlmostEqual(result["net_change"], .02)
        self.assertEqual(result["maximum_drawdown"], 0.)
        self.assertEqual(result["largest_single_drop"], 0.)
        self.assertEqual((start, rows), original)


class UnitBoundariesTest(unittest.TestCase):
    def test_current_unit_stops_before_a_subsequent_fresh_prediction(self):
        rows = [row(1, fresh=True), row(2), row(3, fresh=True), row(4)]
        self.assertEqual([r["step"] for r in review.current_unit(rows, 0, 8)], [1, 2])
        self.assertEqual([r["step"] for r in review.current_unit(rows, 1, 8)], [2])

    def test_current_unit_permits_its_starting_prediction_but_obeys_length_and_terminal(self):
        rows = [row(i, fresh=i == 1) for i in range(1, 13)]
        self.assertEqual(len(review.current_unit(rows, 0, 8)), 8)
        self.assertEqual([r["step"] for r in review.current_unit(rows, 10, 8)], [11, 12])
        self.assertEqual(review.current_unit(rows, len(rows), 8), [])

    def test_extensions_observe_only_the_four_retained_movements(self):
        rows = [row(i, age=i) for i in range(1, 13)] + [row(13, fresh=True)]
        rows[7]["features"]["coverage"] = .79
        result = review.extensions(episode({"kind": "extend", "step": 8}),
                                   {"steps": rows})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["suffix_steps"], [9, 10, 11, 12])
        self.assertTrue(result[0]["full_four_executed"])
        self.assertEqual(result[0]["camera"], rows[7]["features"])
        self.assertEqual(result[0]["outcome"]["start_coverage"], rows[7]["after"]["coverage"])

    def test_extension_suffix_stops_at_early_new_prediction(self):
        rows = [row(i, age=i) for i in range(1, 11)]
        rows.extend([row(11, .1, fresh=True), row(12, .1)])
        result = review.extensions(episode({"kind": "extend", "step": 8}),
                                   {"steps": rows})[0]
        self.assertEqual(result["suffix_steps"], [9, 10])
        self.assertFalse(result["full_four_executed"])
        self.assertEqual(result["outcome"]["end_coverage"], .8)

    def test_extension_immediately_followed_by_new_prediction_has_empty_suffix(self):
        rows = [row(i, age=i) for i in range(1, 9)] + [row(9, .1, fresh=True)]
        result = review.extensions(episode({"kind": "extend", "step": 8}),
                                   {"steps": rows})[0]
        self.assertEqual(result["suffix_steps"], [])
        self.assertFalse(result["full_four_executed"])
        self.assertEqual(result["outcome"]["net_change"], 0.)

    def test_terminal_partial_extension_does_not_invent_unexecuted_movements(self):
        rows = [row(i, age=i) for i in range(1, 9)] + [row(9, .96, age=9)]
        result = review.extensions(episode({"kind": "extend", "step": 8}),
                                   {"steps": rows})[0]
        self.assertEqual(result["suffix_steps"], [9])
        self.assertFalse(result["full_four_executed"])
        self.assertEqual(result["outcome"]["observed_movements"], 1)

    def test_transient_extension_dip_is_not_labeled_a_failure(self):
        rows = [row(i, age=i) for i in range(1, 9)] + [row(9, .7), row(10, .85)]
        records = review.extensions(episode({"kind": "extend", "step": 8},
                                           {"kind": "trim", "step": 3}), {"steps": rows})
        self.assertEqual(len(records), 1)
        summary = review.summarize_records([{"success": True, "extensions": records,
                                            "transitions": []}])
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["suffix_net_drops_over_three_pp"], 0)
        self.assertEqual(summary["suffix_drawdowns_over_three_pp"], 1)
        self.assertNotIn("failure", records[0])

    def test_extension_geometry_is_endpoint_displacement_and_wrapped_rotation(self):
        rows = [row(i, age=i) for i in range(1, 10)]
        rows[7]["after"].update(block_pos=[1., 2.], block_angle=3.1)
        rows[8]["after"].update(block_pos=[4., 6.], block_angle=-3.1)
        record = review.extensions(episode({"kind": "extend", "step": 8}),
                                   {"steps": rows})[0]
        self.assertEqual(record["block_translation"], 5.)
        self.assertAlmostEqual(record["block_rotation_change"], 2 * review.math.pi - 6.2)

    def test_empty_extension_at_terminal_step_has_zero_endpoint_motion(self):
        rows = [row(i, age=i) for i in range(1, 9)]
        record = review.extensions(episode({"kind": "extend", "step": 8}),
                                   {"steps": rows})[0]
        self.assertEqual(record["suffix_steps"], [])
        self.assertEqual(record["block_translation"], 0.)
        self.assertEqual(record["block_rotation_change"], 0.)


class PlanTransitionTest(unittest.TestCase):
    def test_initial_prediction_is_not_a_plan_transition(self):
        rows = [row(1, fresh=True), row(2), row(3)]
        self.assertEqual(review.transitions(episode(), {"steps": rows}), [])

    def test_jump_is_command_target_difference_not_speed_or_physical_displacement(self):
        rows = [row(1, action=(0., 0.), fresh=True), row(2, action=(1., 0.)),
                row(3, action=(2., 0.)), row(4, action=(2., 3.), fresh=True)]
        rows[2]["after"] = state(.8, pusher=(2., 7.), velocity=(30., 40.), gap=-2.)
        rows[3]["after"] = state(.7, pusher=(102., 7.))
        record = review.transitions(episode(), {"initial": state(.1), "steps": rows})[0]
        self.assertEqual(record["command_jump"], 3.)
        self.assertEqual(record["command_direction_cosine"], 0.)
        self.assertEqual(record["command_target_distance_from_pusher"], 4.)
        self.assertEqual(record["pusher_speed_before"], 50.)
        self.assertEqual(record["first_physical_displacement"], 100.)
        self.assertEqual(record["pusher_block_signed_gap_before"], -2.)
        self.assertEqual(record["prefix_outcome"]["start_coverage"], .8)
        self.assertEqual(record["camera_before"], rows[2]["features"])

    def test_zero_direction_or_zero_boundary_jump_gives_no_cosine(self):
        for actions in (((0., 0.), (0., 0.), (1., 0.)),
                        ((0., 0.), (1., 0.), (1., 0.))):
            rows = [row(i + 1, action=a, fresh=i in (0, 2)) for i, a in enumerate(actions)]
            with self.subTest(actions=actions):
                self.assertIsNone(review.transitions(episode(), {"steps": rows})[0]
                                  ["command_direction_cosine"])

    def test_prior_trim_or_interrupt_is_associated_with_preceding_step(self):
        for event in ("trim", "interrupt"):
            rows = [row(1, fresh=True), row(2, age=2), row(3, fresh=True)]
            record = review.transitions(episode({"kind": event, "step": 2},
                                                {"kind": "extend", "step": 1}), {"steps": rows})[0]
            with self.subTest(event=event):
                self.assertEqual(record["previous_end_event"], event)
                self.assertEqual(record["previous_plan_end"], 2)
                self.assertEqual(record["previous_plan_age"], 2)

    def test_transition_prefix_stops_before_another_prediction(self):
        rows = [row(i, fresh=i in (1, 3, 6)) for i in range(1, 11)]
        records = review.transitions(episode(), {"steps": rows})
        self.assertEqual([r["new_plan_starts_at"] for r in records], [3, 6])
        self.assertEqual(records[0]["prefix_steps"], [3, 4, 5])
        self.assertFalse(records[0]["full_eight_observed"])
        self.assertEqual(records[1]["prefix_steps"], [6, 7, 8, 9, 10])

    def test_transition_prefix_is_capped_at_eight_movements(self):
        rows = [row(i, fresh=i in (1, 3)) for i in range(1, 14)]
        record = review.transitions(episode(), {"steps": rows})[0]
        self.assertEqual(record["prefix_steps"], list(range(3, 11)))
        self.assertTrue(record["full_eight_observed"])

    def test_before_uses_initial_only_at_first_step(self):
        initial = state(.1)
        rows = [row(1, .6), row(2, .9)]
        trace = {"initial": initial, "steps": rows}
        self.assertIs(review.before(trace, 0), initial)
        self.assertIs(review.before(trace, 1), rows[0]["after"])


class OutputIsolationTest(unittest.TestCase):
    def test_only_dedicated_output_and_repeat_are_accepted(self):
        dedicated = review.OUTPUT.resolve()
        review.check_output(dedicated)
        review.check_output(dedicated.with_name(dedicated.name + "-repeat"))

    def test_frozen_paths_and_allowed_ancestors_or_descendants_are_rejected(self):
        dedicated = review.OUTPUT.resolve()
        forbidden = [review.SOURCE.resolve(), review.SOURCE.resolve() / "episodes",
                     dedicated.parent, dedicated.parent.parent,
                     dedicated / "nested", dedicated / "analysis.json",
                     dedicated.with_name(dedicated.name + "-repeat") / "nested",
                     dedicated.with_name(dedicated.name + "-repeat-repeat")]
        for path in forbidden:
            with self.subTest(path=path), self.assertRaises(ValueError):
                review.check_output(path)


class RecoveryReviewTest(unittest.TestCase):
    def test_recovery_after_global_minimum_is_not_final_task_success(self):
        rows = [row(i, coverage) for i, coverage in
                enumerate([.9, .8, .91, .7, .89, .9], start=1)]
        result = review.recovery_review({"steps": rows}, 1)
        self.assertEqual(result["pre_drop"], {"step": 1, "coverage": .9})
        self.assertEqual(result["minimum_after_drop"], {"step": 4, "coverage": .7})
        self.assertEqual(result["first_regained_after_minimum"], {"step": 6, "coverage": .9})
        self.assertEqual(result["whole_attempt_peak"], {"step": 3, "coverage": .91})
        self.assertEqual(result["final_coverage"], .9)
        self.assertLess(result["final_coverage"], .95)

    def test_no_regained_overlap_is_reported_as_none(self):
        rows = [row(i, coverage) for i, coverage in enumerate([.9, .7, .8, .85], start=1)]
        result = review.recovery_review({"steps": rows}, 1)
        self.assertIsNone(result["first_regained_after_minimum"])
        self.assertEqual(result["minimum_after_drop"]["step"], 2)

    def test_camera_peak_is_valid_and_paired_with_truth_at_the_same_step(self):
        rows = [row(i, coverage) for i, coverage in enumerate([.9, .7, .85, .91], start=1)]
        rows[1]["features"].update(quality=False, coverage=1.)
        rows[2]["features"]["coverage"] = .97
        result = review.recovery_review({"steps": rows}, 1)
        self.assertEqual(result["camera_peak"], {"step": 3, "camera_coverage": .97,
                                               "true_coverage_same_step": .85})
        self.assertEqual(result["whole_attempt_peak"], {"step": 4, "coverage": .91})
        self.assertEqual(result["final_coverage"], .91)


if __name__ == "__main__":
    unittest.main()
