"""CPU-only checks of the staged 8/12/15 extension and locked comparison.

In-memory policy/environment doubles never load a checkpoint or run inference.
Old experiment sources and recorded results are read-only.
"""

import copy
from collections import deque
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_staged_extension as screen
import test_robotics_stateful_budget as doubles


def features(**changes):
    result = {"quality": True, "history_steps": 8, "coverage": .60,
              "warning_count": 0, "drawdown": .01, "recovery": 0.,
              "recent_peak": .61}
    result.update(changes)
    return result


def regression(**changes):
    return features(**{"coverage": .82, "warning_count": 2, "drawdown": .04,
                       "recovery": -.02, "recent_peak": .86, **changes})


def commands(step=3.):
    return np.column_stack((100. + np.arange(15) * step,
                            np.full(15, 200.)))


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="staged-extension-cpu-", dir=parent)


class LateEligibilityTest(unittest.TestCase):
    def test_camera_eligibility_and_smooth_tail_are_both_required(self):
        self.assertTrue(screen.extension_after_twelve(features(), commands()))
        for changed in ({"quality": False}, {"history_steps": 7},
                        {"coverage": .75}, {"warning_count": 1},
                        {"drawdown": .01501}, {"recovery": -.01001}):
            with self.subTest(changed=changed):
                self.assertFalse(screen.extension_after_twelve(features(**changed), commands()))
        self.assertFalse(screen.extension_after_twelve(features(), commands(1.999)))

    def test_original_inclusive_thresholds_are_retained(self):
        self.assertTrue(screen.extension_after_twelve(features(
            history_steps=8, drawdown=.015, recovery=-.01), commands(2.)))
        self.assertFalse(screen.extension_after_twelve({"quality": False}, commands()))

    def test_tail_junction_and_direction_change_are_checked(self):
        plan = commands()
        plan[12:] = plan[11] - np.column_stack((3. * np.arange(1, 4), np.zeros(3)))
        self.assertFalse(screen.extension_after_twelve(features(), plan))
        plan = commands()
        plan[14] = plan[13]
        self.assertFalse(screen.extension_after_twelve(features(), plan))

    def test_only_late_path_window_affects_smoothness(self):
        plan = commands()
        plan[:8] = [[0., 0.], [500., 500.]] * 4
        self.assertTrue(screen.extension_after_twelve(features(), plan))

    def test_full_fifteen_commands_must_be_finite_and_correctly_shaped(self):
        for plan in (np.zeros((14, 2)), np.zeros((16, 2)),
                     np.zeros((15, 3)), np.zeros(30)):
            with self.subTest(shape=plan.shape):
                self.assertFalse(screen.extension_after_twelve(features(), plan))
        for index in (0, 8, 14):
            for value in (np.nan, np.inf, -np.inf):
                plan = commands()
                plan[index, 0] = value
                with self.subTest(index=index, value=value):
                    self.assertFalse(screen.extension_after_twelve(features(), plan))

    def test_dynamic_truth_is_not_a_decision_input_and_inputs_are_unchanged(self):
        row = features(true_coverage=.99, success=False, contact=False)
        old_row, plan = copy.deepcopy(row), commands()
        old_plan = plan.copy()
        self.assertTrue(screen.extension_after_twelve(row, plan))
        self.assertEqual(row, old_row)
        np.testing.assert_array_equal(plan, old_plan)


class StagedControllerTest(unittest.TestCase):
    def update(self, controller, row=None, *, age=8, step=40, plan=None, done=False):
        pending = 12 - age if age <= 12 else 15 - age
        return controller.update(features() if row is None else row,
                                 age, pending, step,
                                 commands() if plan is None else plan, done=done)

    def test_default_uses_original_guard_and_no_persistent_budget(self):
        controller = screen.StagedExtensionController()
        self.assertIsNone(controller.late_mode)
        self.assertIsInstance(controller.guard, screen.adaptive.AdaptiveLengthController)
        self.assertFalse(hasattr(controller, "budget"))
        self.assertFalse(hasattr(controller, "streak"))

    def test_mechanics_modes_accept_only_none_or_actual_booleans(self):
        for mode in (None, False, True):
            self.assertIs(screen.StagedExtensionController(mode).late_mode, mode)
        for mode in (0, 1, "forced", [], {}):
            with self.subTest(mode=mode), self.assertRaises(AssertionError):
                screen.StagedExtensionController(mode)

    def test_eight_movement_decisions_are_dictionary_exact_original_events(self):
        for row, plan in ((features(), commands()),
                          (features(coverage=.8), commands()),
                          ({"quality": False}, commands()),
                          (features(), commands(1.))):
            with self.subTest(row=row):
                expected = screen.adaptive.AdaptiveLengthController().update(row, 8, 4, 40, plan[:12])
                actual = self.update(screen.StagedExtensionController(), row, plan=plan)
                self.assertEqual(actual, expected)
                self.assertNotIn("appended", actual)

    def test_reserved_tail_never_counts_as_pending_for_original_guard(self):
        controller = screen.StagedExtensionController()
        with patch.object(controller.guard, "update", return_value=None) as original:
            self.update(controller, age=7)
        args = original.call_args.args
        self.assertEqual(args[1:4], (7, 5, 40))
        np.testing.assert_array_equal(args[4], commands()[:12])

    def test_guard_state_matches_original_through_all_first_twelve_actions(self):
        controller, original = screen.StagedExtensionController(), screen.adaptive.AdaptiveLengthController()
        for age in range(1, 13):
            row = regression() if age in (2, 3) else features()
            expected = original.update(row, age, 12 - age, 30 + age, commands()[:12])
            actual = self.update(controller, row, age=age, step=30 + age)
            self.assertEqual(vars(controller.guard), vars(original))
            if age < 12:
                self.assertEqual(actual, expected)

    def test_original_confirmed_interrupt_has_precedence(self):
        controller = screen.StagedExtensionController()
        self.assertIsNone(self.update(controller, regression(), age=4, step=36))
        event = self.update(controller, regression(), age=5, step=37)
        self.assertEqual(event, {"kind": "interrupt", "reason": "confirmed_regression", "discarded": 7})
        self.assertEqual(controller.guard.interruptions, 1)

    def test_late_extension_appends_only_at_twelve_with_empty_scheduled_queue(self):
        controller = screen.StagedExtensionController()
        for age in (1, 7, 9, 11, 13, 14, 15):
            self.assertIsNone(self.update(controller, age=age))
        event = self.update(controller, age=12)
        self.assertEqual(event["kind"], "extend15")
        self.assertEqual(event["discarded"], 0)
        self.assertEqual(event["appended"], 3)
        self.assertEqual(event["cached_discarded"], 0)

    def test_ineligible_late_extension_requests_new_plan_without_live_discard(self):
        for row, plan in ((features(coverage=.8), commands()),
                          ({"quality": False}, commands()),
                          (features(), commands(1.))):
            with self.subTest(row=row):
                event = self.update(screen.StagedExtensionController(), row, age=12, plan=plan)
                self.assertEqual(event["kind"], "replan12")
                self.assertEqual((event["discarded"], event["appended"], event["cached_discarded"]), (0, 0, 3))

    def test_forced_modes_change_only_twelve_movement_extension(self):
        for mode, kind in ((False, "replan12"), (True, "extend15")):
            controller = screen.StagedExtensionController(mode)
            self.assertEqual(self.update(controller)["kind"], "extend")
            event = self.update(controller, features(coverage=.8), age=12)
            self.assertEqual(event["kind"], kind)
            self.assertEqual(event["reason"], "mechanics_override")

    def test_invalid_age_or_pending_counts_are_rejected(self):
        for age, pending in ((0, 12), (16, 0), (8, 7), (12, 3), (13, 0), (15, 1)):
            with self.subTest(age=age, pending=pending), self.assertRaises(AssertionError):
                screen.StagedExtensionController().update(features(), age, pending, 40, commands())

    def test_terminal_observation_does_not_call_or_mutate_guard(self):
        controller = screen.StagedExtensionController()
        before = copy.deepcopy(vars(controller.guard))
        with patch.object(controller.guard, "update", side_effect=AssertionError("terminal guard called")):
            self.assertIsNone(self.update(controller, regression(), age=12, done=True))
        self.assertEqual(vars(controller.guard), before)


class SuffixQueueTest(unittest.TestCase):
    def policy(self):
        actions = [object() for _ in range(14)]
        history = deque([object(), object()], maxlen=2)
        return SimpleNamespace(_queues={"action": deque(actions, maxlen=15),
                                        "observation.state": history}), actions, history

    def test_reserved_tail_preserves_exact_command_objects_and_order(self):
        policy, actions, _ = self.policy()
        tail = screen.reserve_suffix(policy)
        self.assertEqual(list(policy._queues["action"]), actions[:11])
        self.assertEqual(tail, actions[11:])
        self.assertTrue(all(a is b for a, b in zip(tail, actions[11:])))
        self.assertEqual(policy._queues["action"].maxlen, 15)

    def test_appended_tail_is_original_objects_without_history_reset(self):
        policy, actions, history = self.policy()
        old_history = list(history)
        tail = screen.reserve_suffix(policy)
        policy._queues["action"].clear()
        screen.append_tail(policy, tail)
        self.assertEqual(list(policy._queues["action"]), actions[11:])
        self.assertTrue(all(a is b for a, b in zip(policy._queues["action"], tail)))
        self.assertIs(policy._queues["observation.state"], history)
        self.assertEqual(list(history), old_history)

    def test_reserving_requires_exact_fresh_fifteen_command_queue(self):
        for count, capacity in ((0, 15), (11, 15), (15, 15), (14, 14)):
            with self.subTest(count=count, capacity=capacity):
                policy, _, _ = self.policy()
                policy._queues["action"] = deque([object() for _ in range(count)], maxlen=capacity)
                with self.assertRaises(AssertionError):
                    screen.reserve_suffix(policy)

    def test_append_rejects_nonempty_queue_or_wrong_tail_before_mutation(self):
        for count, tail_count in ((1, 3), (0, 2), (0, 4)):
            with self.subTest(count=count, tail_count=tail_count):
                policy, _, _ = self.policy()
                policy._queues["action"] = deque([object() for _ in range(count)], maxlen=15)
                original = list(policy._queues["action"])
                with self.assertRaises(AssertionError):
                    screen.append_tail(policy, [object() for _ in range(tail_count)])
                self.assertEqual(list(policy._queues["action"]), original)


def episode_patches(env, estimator):
    stack = ExitStack()
    stack.enter_context(patch.object(screen.gym, "make", return_value=env))
    stack.enter_context(patch.object(screen.adaptive, "observation_batch", side_effect=lambda obs, device: obs))
    stack.enter_context(patch.object(screen.adaptive, "physical_state", side_effect=doubles.physical_state))
    stack.enter_context(patch.object(screen.adaptive, "VisualAlignmentEstimator", return_value=estimator))
    stack.enter_context(patch.object(screen.adaptive, "RollingVisualFeatures", return_value=doubles.FakeRolling()))
    stack.enter_context(patch.object(screen.torch.cuda, "synchronize"))
    stack.enter_context(patch.object(screen.torch.cuda, "reset_peak_memory_stats"))
    stack.enter_context(patch.object(screen.torch.cuda, "max_memory_allocated", return_value=4096))
    return stack


def synthetic_episode(directory, method="staged15", mode=None, near=False,
                      finish=32, truncate=None, fail_step=None, scene=110, draw=40000):
    policy = doubles.FakePolicy()
    env = doubles.FakeEnvironment(finish, fail_step, truncate)
    estimator = doubles.FakeEstimator(near)
    with episode_patches(env, estimator):
        trial = screen.evaluate_episode(policy, scene, draw, method, directory,
                                        max_steps=32, late_mode=mode)
    return trial, screen.gen.read_trace(trial, directory), policy, env, estimator


def write_trial(directory, trial, trace):
    with gzip.open(directory / trial["trace"], "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False)
    trial["trace_sha256"] = screen.base.sha(directory / trial["trace"])
    screen.base.save_json(directory / (screen.gen.stem(trial["scene"], trial["draw"],
                                                     trial["method"]) + ".json"), trial)


class EpisodeIntegrationTest(unittest.TestCase):
    def test_synthetic_controller_runs_fifteen_cached_commands_between_predictions(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace, policy, env, estimator = synthetic_episode(directory)
            self.assertEqual(screen.verify_trial(trial, directory, max_steps=32), trace)
        self.assertEqual(trial["unit_lengths"], [15, 15, 2])
        self.assertEqual(trial["model_queries"], 3)
        self.assertEqual(trial["generated_shapes"], [[1, 16, 2]] * 3)
        self.assertEqual(trace["steps"][11]["pending_before"], 0)
        self.assertEqual(trace["steps"][11]["pending_after"], 3)
        self.assertTrue(all(not row["new_prediction"] for row in trace["steps"][1:15]))
        self.assertEqual([row["action"] for row in trace["steps"][:15]],
                         trace["steps"][0]["predicted_commands"])
        self.assertEqual(estimator.calls, 32)
        self.assertTrue(env.closed)
        self.assertEqual(policy.resets, 2)
        self.assertEqual(policy.all_seen, list(range(32)))
        self.assertEqual(policy.all_histories[-1], [30, 31])
        self.assertEqual(policy.config.horizon, 16)
        self.assertEqual(policy.config.n_action_steps, 8)

    def test_near_target_falls_back_at_original_eight_checkpoint(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace, _, _, _ = synthetic_episode(directory, near=True)
            screen.verify_trial(trial, directory, max_steps=32)
        self.assertEqual(trial["unit_lengths"], [8] * 4)
        self.assertFalse(any(e["kind"] == "extend15" for e in trial["events"]))
        self.assertEqual([e["kind"] for e in trial["events"]], ["trim"] * 3)

    def test_disabled_late_extension_exactly_reproduces_milestone_across_predictions(self):
        with test_directory() as name:
            root = Path(name)
            disabled, _, _, _, _ = synthetic_episode(root / "disabled", mode=False)
            milestone, _, _, _, _ = synthetic_episode(root / "milestone", method="adaptive12")
            screen.verify_trial(disabled, root / "disabled", max_steps=32, allow_force=True)
            comparison = screen.common_prefix(disabled, milestone,
                root / "disabled", root / "milestone")
        self.assertEqual(comparison["matched_through_movement"], 32)
        self.assertIsNone(comparison["first_late_extension"])
        self.assertEqual(disabled["unit_lengths"], milestone["unit_lengths"])
        self.assertEqual([e for e in disabled["events"] if e["kind"] != "replan12"],
                         milestone["events"])

    def test_common_prefix_stops_at_intended_first_late_extension(self):
        with test_directory() as name:
            root = Path(name)
            candidate, _, _, _, _ = synthetic_episode(root / "candidate")
            milestone, _, _, _, _ = synthetic_episode(root / "milestone", method="adaptive12")
            comparison = screen.common_prefix(candidate, milestone,
                root / "candidate", root / "milestone")
        self.assertEqual(comparison, {"scene": 110, "matched_through_movement": 12,
                                      "first_late_extension": 12})

    def test_mechanics_mode_is_rejected_from_scored_validation(self):
        for mode in (False, True):
            with self.subTest(mode=mode), test_directory() as name:
                directory = Path(name)
                trial, trace, _, _, _ = synthetic_episode(directory, mode=mode)
                with self.assertRaises(AssertionError):
                    screen.verify_trial(trial, directory, max_steps=32)
                self.assertEqual(screen.verify_trial(trial, directory, max_steps=32,
                                                    allow_force=True), trace)

    def test_reference_methods_use_unchanged_generation_runner(self):
        for method in ("fixed8", "adaptive12"):
            sentinel = object()
            with self.subTest(method=method), patch.object(screen.gen, "evaluate_episode", return_value=sentinel) as original:
                actual = screen.evaluate_episode("policy", 110, 40000, method, Path("new"), 32)
                self.assertIs(actual, sentinel)
                original.assert_called_once_with("policy", 110, 40000, method, Path("new"), 32)

    def test_environment_exception_still_restores_model_and_closes_simulator(self):
        policy, env = doubles.FakePolicy(), doubles.FakeEnvironment(fail_step=4)
        with test_directory() as name, episode_patches(env, doubles.FakeEstimator()):
            with self.assertRaisesRegex(RuntimeError, "synthetic movement error"):
                screen.evaluate_episode(policy, 110, 40000, "staged15", Path(name), 32)
            self.assertEqual(list(Path(name).iterdir()), [])
        self.assertTrue(env.closed)
        self.assertEqual(policy.resets, 2)
        self.assertEqual(policy.config.n_action_steps, 8)


class TrialIntegrityTest(unittest.TestCase):
    def test_rehashed_malformed_trace_cannot_pass_queue_and_event_validation(self):
        mutations = (
            lambda t, r: r["steps"][0].update(reserved_suffix_count=0),
            lambda t, r: r["steps"][0].update(pending_before=14),
            lambda t, r: r["steps"][11].update(pending_after=0),
            lambda t, r: r["steps"][12].update(new_prediction=True),
            lambda t, r: r["steps"][12].update(action=[-1., -1.]),
            lambda t, r: r["steps"][1].update(reserved_suffix_count=3),
            lambda t, r: r["steps"][0]["predicted_commands"].pop(),
            lambda t, r: r["steps"][11]["features"].update(pending_actions=3),
            lambda t, r: t["events"][1].update(appended=0),
            lambda t, r: t["events"][1].update(cached_discarded=3),
            lambda t, r: t.update(unit_lengths=[12, 12, 8]),
            lambda t, r: t.update(model_queries=9),
            lambda t, r: t.update(generated_shapes=[[1, 8, 2]] * 3),
            lambda t, r: r["steps"][-1].update(features=features()),
            lambda t, r: r.update(draw=30000))
        with test_directory() as name:
            directory = Path(name)
            original, trace, _, _, _ = synthetic_episode(directory)
            for mutate in mutations:
                trial, altered = copy.deepcopy(original), copy.deepcopy(trace)
                mutate(trial, altered)
                write_trial(directory, trial, altered)
                with self.subTest(mutation=mutate), self.assertRaises((AssertionError, KeyError)):
                    screen.verify_trial(trial, directory, max_steps=32)

    def test_wrong_identity_horizon_hash_or_path_is_rejected(self):
        with test_directory() as name:
            directory = Path(name)
            trial, _, _, _, _ = synthetic_episode(directory)
            for changed in ({"scene": 89}, {"draw": 20000}, {"method": "stateful15"},
                            {"prediction_seed": 0}, {"horizon": 8}, {"execution": 12},
                            {"trace_sha256": "0" * 64}, {"trace": "../old.trace.json.gz"}):
                with self.subTest(changed=changed), self.assertRaises((AssertionError, KeyError)):
                    screen.verify_trial({**trial, **changed}, directory, max_steps=32)

    def test_strict_goal_threshold_and_full_timeout_are_maintained(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace, _, _, _ = synthetic_episode(directory, finish=33, truncate=32)
            screen.verify_trial(trial, directory, max_steps=32)
            trial["final_coverage"] = trace["steps"][-1]["after"]["coverage"] = .95
            write_trial(directory, trial, trace)
            screen.verify_trial(trial, directory, max_steps=32)
            for changed in ({"success": True}, {"truncated": False}, {"steps": 31}):
                with self.subTest(changed=changed), self.assertRaises(AssertionError):
                    screen.verify_trial({**trial, **changed}, directory, max_steps=32)

    def test_nonfinite_costs_and_nonpositive_call_timings_are_rejected(self):
        with test_directory() as name:
            directory = Path(name)
            original, trace, _, _, _ = synthetic_episode(directory)
            for key, value in (("wall_seconds", np.inf), ("monitor_seconds", np.nan),
                               ("peak_gpu_allocated_bytes", 0)):
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    screen.verify_trial({**original, key: value}, directory, max_steps=32)
            for value in (0., np.nan, np.inf):
                altered = copy.deepcopy(trace)
                altered["steps"][0]["prediction_seconds"] = value
                with patch.object(screen.gen, "read_trace", return_value=altered), self.assertRaises(AssertionError):
                    screen.verify_trial(original, directory, max_steps=32)

    def test_valid_sidecar_is_reused_without_new_prediction(self):
        with test_directory() as name:
            directory = Path(name)
            original, _, _, _, _ = synthetic_episode(directory)
            with patch.object(screen, "evaluate_episode") as evaluate:
                actual = screen.collect(object(), 110, 40000, "staged15", directory, 32)
            evaluate.assert_not_called()
            self.assertEqual(actual, original)

    def test_orphan_trace_and_partial_sidecar_stop_without_overwriting(self):
        for suffix in (".trace.json.gz", ".json.tmp"):
            with self.subTest(suffix=suffix), test_directory() as name:
                directory = Path(name)
                orphan = directory / (screen.gen.stem(110, 40000, "staged15") + suffix)
                orphan.write_bytes(b"unfinished")
                with patch.object(screen, "evaluate_episode") as evaluate, self.assertRaisesRegex(AssertionError, "Inspect"):
                    screen.collect(object(), 110, 40000, "staged15", directory, 32)
                evaluate.assert_not_called()
                self.assertEqual(orphan.read_bytes(), b"unfinished")


def metric_fixture(scenes):
    trials = []
    for scene in scenes:
        for method in screen.METHODS:
            success = scene % 2 == 0
            if method == "staged15" and scene in (110, 111):
                success = not success
            trials.append({"scene": scene, "draw": screen.draw_for(scene), "method": method,
                           "success": success, "final_coverage": .81 if method == "staged15" else .80,
                           "model_queries": 20 if method == "staged15" else 30,
                           "wall_seconds": 100. if method == "staged15" else 150.,
                           "monitor_seconds": 2. if method != "fixed8" else 0.})
    return trials


class ScheduleAndStatisticsTest(unittest.TestCase):
    def test_disjoint_inspected_and_fresh_panels_have_fixed_draws(self):
        self.assertEqual(screen.DEVELOPMENT_SCENES, list(range(90, 110)))
        self.assertEqual(screen.FRESH_SCENES, list(range(110, 150)))
        self.assertEqual(screen.METHODS, ["fixed8", "adaptive12", "staged15"])
        self.assertTrue(set(screen.DEVELOPMENT_SCENES).isdisjoint(screen.FRESH_SCENES))
        self.assertEqual([screen.draw_for(s) for s in screen.DEVELOPMENT_SCENES], [30000] * 20)
        self.assertEqual([screen.draw_for(s) for s in screen.FRESH_SCENES], [40000] * 40)
        for scene in (89, 150):
            with self.assertRaises(AssertionError):
                screen.draw_for(scene)

    def test_schedule_contains_180_pairs_with_exactly_forty_reused_controls(self):
        schedule = screen.schedule()
        self.assertEqual(len(schedule), 180)
        self.assertEqual(len({(r["scene"], r["draw"], r["method"]) for r in schedule}), 180)
        self.assertEqual(sum(r["reused"] for r in schedule), 40)
        for index, scene in enumerate(screen.SCENES):
            rows = [r for r in schedule if r["scene"] == scene]
            self.assertEqual([r["method"] for r in rows], screen.METHODS[index % 3:] + screen.METHODS[:index % 3])
            self.assertEqual([r["reused"] for r in rows],
                [scene in screen.DEVELOPMENT_SCENES and r["method"] != "staged15" for r in rows])

    def test_output_guard_rejects_old_or_broad_directories(self):
        screen.check_output(screen.OUTPUT)
        screen.check_output(screen.OUTPUT.with_name(screen.OUTPUT.name + "-repeat"))
        for path in (Path("results"), screen.REUSED, screen.OUTPUT / "nested", Path("../outside")):
            with self.subTest(path=path), self.assertRaises(AssertionError):
                screen.check_output(path)

    def test_metrics_are_separate_for_each_predeclared_cohort(self):
        for scenes in (screen.DEVELOPMENT_SCENES, screen.FRESH_SCENES, screen.SCENES):
            with self.subTest(scenes=len(scenes)):
                table = screen.metric_table(metric_fixture(scenes), scenes)
                for method in screen.METHODS:
                    self.assertEqual(table[method]["attempts"], len(scenes))
                    self.assertEqual(table[method]["successes"], len(scenes) // 2)
                self.assertEqual(table["staged15"]["mean_model_queries"], 20.)
                self.assertEqual(table["staged15"]["mean_wall_seconds"], 100.)
                self.assertAlmostEqual(table["staged15"]["mean_final_coverage"], .81)

    def test_incomplete_duplicate_or_wrong_draw_panels_are_rejected(self):
        original = metric_fixture(screen.FRESH_SCENES)
        changed_draw = copy.deepcopy(original)
        changed_draw[0]["draw"] = 30000
        for trials in (original[:-1], original + [original[0]], original[:-1] + [original[0]], changed_draw):
            with self.subTest(length=len(trials)), self.assertRaises(AssertionError):
                screen.metric_table(trials, screen.FRESH_SCENES)

    def test_paired_statistics_preserve_scene_identity_and_direction(self):
        trials = metric_fixture(screen.FRESH_SCENES)
        first = screen.paired(trials, "fixed8", screen.FRESH_SCENES)
        self.assertEqual(first, screen.paired(list(reversed(trials)), "fixed8", screen.FRESH_SCENES))
        self.assertEqual(first["gained_successes"], [111])
        self.assertEqual(first["lost_successes"], [110])
        costs = first["differences_method_minus_reference"]["model_queries"]
        self.assertEqual(costs["mean_difference"], -10.)
        self.assertEqual(costs["scene_pair_bootstrap_95_interval"], [-10., -10.])
        self.assertEqual(first["exact_discordant_two_sided_p"], 1.)

    def test_bootstrap_resamples_forty_scene_pairs_not_120_rows(self):
        trials = metric_fixture(screen.FRESH_SCENES)
        actual = screen.paired(trials, "adaptive12", screen.FRESH_SCENES)
        values = np.asarray([-1. if s == 110 else 1. if s == 111 else 0. for s in screen.FRESH_SCENES])
        rng = np.random.default_rng(screen.STATISTICS["bootstrap_seed"])
        indices = rng.integers(0, 40, size=(10000, 40))
        expected = np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
        self.assertEqual(actual["differences_method_minus_reference"]["success"][
            "scene_pair_bootstrap_95_interval"], expected)

    def test_exact_discordant_test_is_one_when_no_successes_change(self):
        trials = metric_fixture(screen.DEVELOPMENT_SCENES)
        actual = screen.paired(trials, "fixed8", screen.DEVELOPMENT_SCENES)
        self.assertEqual(actual["gained_successes"], [])
        self.assertEqual(actual["lost_successes"], [])
        self.assertEqual(actual["exact_discordant_two_sided_p"], 1.)


def protocol_fixture(root):
    inherited = {"fingerprints": {}, "packages": {"fake": "1"},
                 "checkpoint": {"weights_sha256": "weights", "config_sha256": "config"},
                 "model_sources": {"fake.module": "source-hash"}, "timing": "test-only"}
    previous = root / "previous"
    screen.base.save_json(previous / "protocol.json", {
        **{key: inherited[key] for key in ("checkpoint", "packages", "model_sources")},
        "reused_milestone_rules": screen.adaptive.RULES})
    directory = previous / "episodes"
    directory.mkdir()
    for identity in screen.schedule():
        if identity["reused"]:
            stem = screen.gen.stem(identity["scene"], identity["draw"], identity["method"])
            trace = directory / (stem + ".trace.json.gz")
            trace.write_bytes(b"synthetic protocol-only trace binding")
            screen.base.save_json(directory / (stem + ".json"),
                {"trace": trace.name, "trace_sha256": screen.base.sha(trace)})
    stack = ExitStack()
    stack.enter_context(patch.object(screen, "REUSED", previous))
    stack.enter_context(patch.object(screen.gen, "protocol_document", return_value=inherited))
    stack.enter_context(patch.object(screen, "check_output"))
    return stack, previous


class ProtocolTest(unittest.TestCase):
    def test_protocol_binds_full_generation_and_exactly_forty_immutable_controls(self):
        with test_directory() as name:
            root = Path(name)
            stack, previous = protocol_fixture(root)
            with stack:
                document = screen.protocol_document()
            self.assertEqual(document["generation"], {"horizon": 16,
                "usable_future_commands": 15, "observations": 2,
                "denoising_steps": 100, "training": False})
            self.assertEqual(document["candidate"]["checkpoints"], [8, 12])
            self.assertFalse(document["candidate"]["persistent_budget"])
            self.assertTrue(document["candidate"]["original_guard_and_eight_check_unchanged"])
            self.assertFalse(document["runtime_truth_inputs"])
            self.assertEqual(document["rules"], screen.adaptive.RULES)
            self.assertEqual(len(document["reused_controls"]), 40)
            self.assertEqual(len(document["fingerprints"]), 82)
            self.assertIn("experiments/robotics_staged_extension.py", document["fingerprints"])
            self.assertIn((previous / "protocol.json").as_posix(), document["fingerprints"])
            self.assertTrue(all(row["method"] in ("fixed8", "adaptive12")
                                and row["scene"] in screen.DEVELOPMENT_SCENES
                                for row in document["reused_controls"]))

    def test_incompatible_previous_settings_or_rules_are_rejected(self):
        with test_directory() as name:
            root = Path(name)
            stack, previous = protocol_fixture(root)
            with stack:
                original = screen.base.read_json(previous / "protocol.json")
                for key in ("checkpoint", "packages", "model_sources", "reused_milestone_rules"):
                    screen.base.save_json(previous / "protocol.json", {**original, key: "changed"})
                    with self.subTest(key=key), self.assertRaises(AssertionError):
                        screen.protocol_document()

    def test_changed_reused_trace_cannot_be_rebound_under_original_hash(self):
        with test_directory() as name:
            root = Path(name)
            stack, previous = protocol_fixture(root)
            with stack:
                trace = next((previous / "episodes").glob("*.trace.json.gz"))
                trace.write_bytes(b"changed existing trace")
                with self.assertRaises(AssertionError):
                    screen.protocol_document()

    def test_prepare_is_idempotent_and_modified_protocol_cannot_resume(self):
        with test_directory() as name:
            root = Path(name)
            stack, _ = protocol_fixture(root)
            with stack:
                output = root / "new"
                original = screen.prepare(output)
                encoded = (output / "protocol.json").read_bytes()
                self.assertEqual(screen.prepare(output), original)
                self.assertEqual((output / "protocol.json").read_bytes(), encoded)
                for key in ("generation", "rules", "schedule", "fresh_scenes",
                            "reused_controls", "fingerprints", "checkpoint", "statistics"):
                    screen.base.save_json(output / "protocol.json", {**original, key: "changed"})
                    with self.subTest(key=key), self.assertRaises(AssertionError):
                        screen.prepare(output)

    def test_unbound_existing_artifacts_are_not_silently_adopted(self):
        for name in ("smoke.json", "episodes", "episodes.json", "summary.json"):
            with self.subTest(name=name), test_directory() as directory:
                output = Path(directory)
                if name == "episodes":
                    (output / name).mkdir()
                else:
                    screen.base.save_json(output / name, {})
                with patch.object(screen, "check_output"), patch.object(screen, "protocol_document") as make:
                    with self.assertRaises(AssertionError):
                        screen.prepare(output)
                make.assert_not_called()


if __name__ == "__main__":
    unittest.main()
