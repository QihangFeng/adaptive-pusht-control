"""CPU-only checks of adaptive action reuse and interruption safety."""

import copy
from collections import deque
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
import robotics_adaptive_length as adaptive
from robotics_adaptive_length import (
    AdaptiveLengthController, extension_candidate, plan_continuity,
)


def features(**changes):
    row = {"quality": True, "history_steps": 8, "coverage": .60,
           "warning_count": 0, "drawdown": .01, "recovery": 0.,
           "recent_peak": .61}
    row.update(changes)
    return row


def regression(**changes):
    row = features(coverage=.82, warning_count=2, drawdown=.04,
                   recovery=-.02, recent_peak=.86)
    row.update(changes)
    return row


def commands(step=3.):
    return np.column_stack((100. + np.arange(12) * step,
                            np.full(12, 200.)))


class ExtensionCandidateTest(unittest.TestCase):
    def test_valid_past_features_allow_extension(self):
        self.assertTrue(extension_candidate(features()))

    def test_exact_inclusive_thresholds(self):
        self.assertTrue(extension_candidate(features(
            history_steps=8, drawdown=.015, recovery=-.01)))

    def test_near_target_and_insufficient_history_are_rejected(self):
        for changed in ({"coverage": .75}, {"coverage": .95},
                        {"history_steps": 7}):
            with self.subTest(changed=changed):
                self.assertFalse(extension_candidate(features(**changed)))

    def test_warning_deterioration_or_missing_image_reject_extension(self):
        for changed in ({"quality": False}, {"warning_count": 1},
                        {"drawdown": .01501}, {"recovery": -.01001}):
            with self.subTest(changed=changed):
                self.assertFalse(extension_candidate(features(**changed)))

    def test_invalid_estimate_does_not_require_numeric_fields(self):
        self.assertFalse(extension_candidate({"quality": False}))

    def test_dynamic_truth_and_labels_are_not_decision_inputs(self):
        original = features()
        altered = {**original, "true_coverage": .99, "success": False,
                   "contact": False, "benefit_score": -100.,
                   "tree": {"value": -100.}}
        self.assertEqual(extension_candidate(original),
                         extension_candidate(altered))
        self.assertEqual(altered["tree"], {"value": -100.})


class PlanContinuityTest(unittest.TestCase):
    def test_straight_plan_accepts_numpy_and_plain_list(self):
        self.assertTrue(plan_continuity(commands()))
        self.assertTrue(plan_continuity(commands().tolist()))

    def test_exact_two_unit_movements_are_allowed(self):
        self.assertTrue(plan_continuity(commands(2.)))

    def test_small_movement_and_zero_length_suffix_are_rejected(self):
        self.assertFalse(plan_continuity(commands(1.999)))
        plan = commands()
        plan[9] = plan[8]
        self.assertFalse(plan_continuity(plan))

    def test_reversing_direction_is_rejected(self):
        plan = commands()
        plan[9:] = plan[8] - np.column_stack(
            (3. * np.arange(1, 4), np.zeros(3)))
        self.assertFalse(plan_continuity(plan))

    def test_prefix_suffix_junction_is_included(self):
        plan = commands()
        plan[8:] = plan[7] - np.column_stack(
            (3. * np.arange(1, 5), np.zeros(4)))
        self.assertFalse(plan_continuity(plan))

    def test_unexamined_early_prefix_does_not_change_suffix_test(self):
        plan = commands()
        plan[:4] = [[0, 0], [500, 500], [0, 0], [500, 500]]
        self.assertTrue(plan_continuity(plan))

    def test_wrong_shapes_and_nonfinite_values_are_rejected(self):
        for plan in (np.zeros((11, 2)), np.zeros((13, 2)),
                     np.zeros((12, 3)), np.zeros(24), np.zeros((1, 12, 2))):
            with self.subTest(shape=plan.shape):
                self.assertFalse(plan_continuity(plan))
        for value in (np.nan, np.inf, -np.inf):
            plan = commands()
            plan[0, 0] = value
            with self.subTest(value=value):
                self.assertFalse(plan_continuity(plan))

    def test_plan_is_not_modified(self):
        plan = commands()
        original = plan.copy()
        self.assertTrue(plan_continuity(plan))
        np.testing.assert_array_equal(plan, original)


class AdaptiveControllerTest(unittest.TestCase):
    def update(self, controller, row=None, *, age=8, pending=4, step=40,
               plan=None, done=False):
        return controller.update(features() if row is None else row,
                                 age, pending, step,
                                 commands() if plan is None else plan,
                                 done=done)

    def assert_event(self, event, kind, discarded):
        self.assertIsInstance(event, dict)
        self.assertEqual(event["kind"], kind)
        self.assertEqual(event["discarded"], discarded)
        self.assertTrue(event["reason"])

    def test_extends_only_at_eight_with_four_pending(self):
        controller = AdaptiveLengthController()
        for age, pending in ((7, 5), (9, 3), (8, 3), (8, 0), (12, 0)):
            with self.subTest(age=age, pending=pending):
                self.assertIsNone(self.update(controller, age=age,
                                              pending=pending))
        self.assert_event(self.update(controller), "extend", 0)

    def test_failed_feature_or_plan_guard_trims_four(self):
        for row, plan in ((features(coverage=.8), commands()),
                          (features(), commands(1.))):
            with self.subTest(row=row, plan=plan):
                self.assert_event(self.update(AdaptiveLengthController(), row,
                                              plan=plan), "trim", 4)

    def test_missing_image_keeps_no_unsafe_suffix(self):
        controller = AdaptiveLengthController()
        missing = {"quality": False}
        self.assertIsNone(self.update(controller, missing, age=6, pending=6))
        self.assert_event(self.update(controller, missing), "trim", 4)

    def test_no_event_after_terminal_observation(self):
        controller = AdaptiveLengthController()
        self.assertIsNone(self.update(controller, regression(), age=4,
                                     pending=8, step=36))
        self.assertIsNone(self.update(controller, regression(), age=5,
                                     pending=7, step=37, done=True))
        self.assertIsNone(self.update(controller, age=8, pending=4,
                                     step=40, done=True))

    def test_interruption_needs_two_confirmations_and_minimum_age(self):
        controller = AdaptiveLengthController()
        self.assertIsNone(self.update(controller, regression(), age=2,
                                     pending=10, step=34))
        self.assertIsNone(self.update(controller, regression(), age=3,
                                     pending=9, step=35))
        self.assert_event(self.update(controller, regression(), age=4,
                                      pending=8, step=36), "interrupt", 8)

    def test_single_warning_observation_is_not_interrupted(self):
        controller = AdaptiveLengthController()
        self.assertIsNone(self.update(controller, regression(), age=4,
                                     pending=8, step=36))
        self.assertIsNone(self.update(controller, features(), age=5,
                                     pending=7, step=37))
        self.assertIsNone(self.update(controller, regression(), age=6,
                                     pending=6, step=38))

    def test_invalid_observation_clears_confirmation(self):
        controller = AdaptiveLengthController()
        self.assertIsNone(self.update(controller, regression(), age=4,
                                     pending=8, step=36))
        self.assertIsNone(self.update(controller, {"quality": False}, age=5,
                                     pending=7, step=37))
        self.assertIsNone(self.update(controller, regression(), age=6,
                                     pending=6, step=38))
        self.assert_event(self.update(controller, regression(), age=7,
                                      pending=5, step=39), "interrupt", 5)

    def test_regression_requirements_and_pending_guard(self):
        for changed in ({"warning_count": 1}, {"drawdown": .029},
                        {"recent_peak": .799}, {"recovery": -.005}):
            controller = AdaptiveLengthController()
            row = regression()
            row.update(changed)
            self.assertIsNone(self.update(controller, row, age=4, pending=8,
                                         step=36))
            self.assertIsNone(self.update(controller, row, age=5, pending=7,
                                         step=37))
        controller = AdaptiveLengthController()
        for step in (36, 37):
            self.assertIsNone(self.update(controller, regression(), age=10,
                                         pending=1, step=step))

    def test_interruption_is_latched_until_valid_clear_observation(self):
        controller = AdaptiveLengthController()
        self.update(controller, regression(), age=4, pending=8, step=36)
        self.assert_event(self.update(controller, regression(), age=5,
                                      pending=7, step=37), "interrupt", 7)
        self.update(controller, {"quality": False}, age=1, pending=11, step=38)
        for step in (49, 50):
            self.assertIsNone(self.update(controller, regression(), age=4,
                                         pending=8, step=step))
        self.assertIsNone(self.update(controller, features(), age=5,
                                     pending=7, step=51))
        self.assertIsNone(self.update(controller, regression(), age=6,
                                     pending=6, step=52))
        self.assert_event(self.update(controller, regression(), age=7,
                                      pending=5, step=53), "interrupt", 5)

    def test_cooldown_prevents_immediate_repeat_after_rearming(self):
        controller = AdaptiveLengthController()
        self.update(controller, regression(), age=4, pending=8, step=36)
        self.assert_event(self.update(controller, regression(), age=5,
                                      pending=7, step=37), "interrupt", 7)
        self.update(controller, features(), age=1, pending=11, step=38)
        for step in (39, 40, 41):
            self.assertIsNone(self.update(controller, regression(), age=4,
                                         pending=8, step=step))
        self.assert_event(self.update(controller, regression(), age=4,
                                      pending=8, step=49), "interrupt", 8)

    def test_at_most_two_interruptions_per_controller(self):
        controller = AdaptiveLengthController()
        for first_step in (36, 56):
            self.update(controller, features(), age=1, pending=11,
                        step=first_step - 1)
            self.assertIsNone(self.update(controller, regression(), age=4,
                                         pending=8, step=first_step))
            self.assert_event(self.update(controller, regression(), age=5,
                                          pending=7, step=first_step + 1),
                              "interrupt", 7)
        self.update(controller, features(), age=1, pending=11, step=75)
        for step in (76, 77, 78):
            self.assertIsNone(self.update(controller, regression(), age=4,
                                         pending=8, step=step))

    def test_features_and_plan_are_not_modified(self):
        row, plan = features(), commands()
        before_row, before_plan = copy.deepcopy(row), plan.copy()
        self.assert_event(self.update(AdaptiveLengthController(), row,
                                      plan=plan), "extend", 0)
        self.assertEqual(row, before_row)
        np.testing.assert_array_equal(plan, before_plan)


class FakePolicy:
    """Mirror LeRobot's batched [1, 2] action queue without model inference."""

    def __init__(self):
        self.config = SimpleNamespace(n_action_steps=8)
        self.diffusion = SimpleNamespace(config=self.config)
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.queries = 0
        self.seen = []
        self._queues = {"action": deque(maxlen=self.config.n_action_steps),
                        "observation.state": deque(maxlen=2)}

    def select_action(self, observation):
        step = observation["step"]
        self.seen.append(step)
        self._queues["observation.state"].append(step)
        if not self._queues["action"]:
            self.queries += 1
            for index in range(self.config.n_action_steps):
                target = 100. + 100. * self.queries + step + 3. * index
                self._queues["action"].append(adaptive.torch.tensor(
                    [[target, 200.]], dtype=adaptive.torch.float64))
        return self._queues["action"].popleft()


class FakeEnvironment:
    def reset(self, seed):
        self.step_index, self.last_command, self.closed = 0, [0., 0.], False
        return {"step": 0, "pixels": None}, {}

    def step(self, command):
        self.step_index += 1
        self.last_command = np.asarray(command).tolist()
        return {"step": self.step_index, "pixels": None}, 0., self.step_index == 16, False, {}

    def close(self):
        self.closed = True


class FakeEstimator:
    def __init__(self, coverage):
        self.coverage = coverage

    def estimate(self, pixels):
        return {"quality": True, "coverage": self.coverage,
                "position_error": .1, "orientation_error": .1}


def fake_physical_state(env, observation):
    return {"coverage": .96 if env.step_index == 16 else .60,
            "position": env.last_command, "step": env.step_index}


def test_directory():
    # Keep atomic JSON writes inside the permitted, Git-ignored workspace.
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="adaptive-length-cpu-", dir=parent)


class EpisodeQueueIntegrationTest(unittest.TestCase):
    def collect(self, method, coverage):
        policy, env = FakePolicy(), FakeEnvironment()
        with test_directory() as directory:
            output = Path(directory)
            with (patch.object(adaptive.gym, "make", return_value=env),
                  patch.object(adaptive, "observation_batch", side_effect=lambda observation, device: observation),
                  patch.object(adaptive, "physical_state", side_effect=fake_physical_state),
                  patch.object(adaptive, "VisualAlignmentEstimator", return_value=FakeEstimator(coverage)),
                  patch.object(adaptive.torch.cuda, "synchronize")):
                result = adaptive.evaluate_episode(policy, 0, method, output)
            adaptive.verify_trial(result, output)
            with gzip.open(output / result["trace"], "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
        self.assertTrue(env.closed)
        self.assertEqual(policy.resets, 1)
        self.assertEqual(policy.seen, list(range(16)))
        self.assertEqual(list(policy._queues["observation.state"]), [14, 15])
        return result, trace

    def test_forced_trim_matches_fixed_eight_whole_prefix_and_history(self):
        fixed, fixed_trace = self.collect("fixed8", .80)
        trimmed, trimmed_trace = self.collect("adaptive12", .80)
        self.assertEqual(fixed["unit_lengths"], [8, 8])
        self.assertEqual(trimmed["unit_lengths"], [8, 8])
        self.assertEqual(fixed["model_queries"], trimmed["model_queries"])
        self.assertEqual([row["action"] for row in fixed_trace["steps"]],
                         [row["action"] for row in trimmed_trace["steps"]])
        self.assertEqual([row["after"] for row in fixed_trace["steps"]],
                         [row["after"] for row in trimmed_trace["steps"]])
        self.assertEqual(trimmed["events"][0]["kind"], "trim")
        self.assertEqual(trimmed_trace["steps"][7]["pending_before"], 4)
        self.assertEqual(trimmed_trace["steps"][7]["pending_after"], 0)
        self.assertTrue(trimmed_trace["steps"][8]["new_prediction"])

    def test_retained_suffix_matches_fixed_twelve_without_new_query(self):
        fixed, fixed_trace = self.collect("fixed12", .60)
        extended, extended_trace = self.collect("adaptive12", .60)
        self.assertEqual(fixed["unit_lengths"], [12, 4])
        self.assertEqual(extended["unit_lengths"], [12, 4])
        self.assertEqual(fixed["model_queries"], extended["model_queries"])
        self.assertEqual([row["action"] for row in fixed_trace["steps"]],
                         [row["action"] for row in extended_trace["steps"]])
        self.assertEqual(extended["events"][0]["kind"], "extend")
        self.assertEqual(extended_trace["steps"][7]["pending_before"], 4)
        self.assertEqual(extended_trace["steps"][7]["pending_after"], 4)
        self.assertFalse(extended_trace["steps"][8]["new_prediction"])
        self.assertTrue(extended_trace["steps"][12]["new_prediction"])

    def test_first_eight_actions_are_identical_for_all_methods(self):
        traces = [self.collect(method, .60)[1]
                  for method in ("fixed8", "fixed12", "adaptive12")]
        reference = [row["action"] for row in traces[0]["steps"][:8]]
        for trace in traces[1:]:
            self.assertEqual(reference, [row["action"] for row in trace["steps"][:8]])

    def test_saved_trace_tampering_is_rejected(self):
        with test_directory() as directory:
            output, policy, env = Path(directory), FakePolicy(), FakeEnvironment()
            with (patch.object(adaptive.gym, "make", return_value=env),
                  patch.object(adaptive, "observation_batch", side_effect=lambda observation, device: observation),
                  patch.object(adaptive, "physical_state", side_effect=fake_physical_state),
                  patch.object(adaptive.torch.cuda, "synchronize")):
                result = adaptive.evaluate_episode(policy, 0, "fixed8", output)
            corrupted = {**result, "trace_sha256": "0" * 64}
            with self.assertRaises(AssertionError):
                adaptive.verify_trial(corrupted, output)


class ProtocolResumeTest(unittest.TestCase):
    def test_prepare_is_idempotent_and_verify_rejects_changed_evidence(self):
        with test_directory() as directory:
            output = Path(directory)
            adaptive.base.save_json(output / "discovery-analysis.json", {"test": "analysis"})
            with (patch.object(adaptive.base, "verify_protocol"),
                  patch.object(adaptive, "fingerprints", return_value={"source.py": "original"}),
                  patch.object(adaptive.importlib.metadata, "version", return_value="test-version")):
                first = adaptive.prepare(output)
                encoded = (output / "protocol.json").read_bytes()
                self.assertEqual(adaptive.prepare(output), first)
                self.assertEqual((output / "protocol.json").read_bytes(), encoded)
                self.assertEqual(adaptive.verify_protocol(output), first)
                adaptive.base.save_json(output / "discovery-analysis.json", {"test": "changed"})
                with self.assertRaises(AssertionError):
                    adaptive.verify_protocol(output)

    def test_verify_rejects_source_or_package_change(self):
        with test_directory() as directory:
            output = Path(directory)
            adaptive.base.save_json(output / "discovery-analysis.json", {"test": "analysis"})
            with (patch.object(adaptive.base, "verify_protocol"),
                  patch.object(adaptive, "fingerprints", return_value={"source.py": "original"}) as fingerprint,
                  patch.object(adaptive.importlib.metadata, "version", return_value="test-version") as version):
                adaptive.prepare(output)
                fingerprint.return_value = {"source.py": "changed"}
                with self.assertRaises(AssertionError):
                    adaptive.verify_protocol(output)
                fingerprint.return_value = {"source.py": "original"}
                version.return_value = "changed-version"
                with self.assertRaises(AssertionError):
                    adaptive.verify_protocol(output)

    def test_partial_trace_prevents_collection_or_rebinding(self):
        with test_directory() as directory:
            output = Path(directory)
            episodes = output / "episodes"
            episodes.mkdir()
            (episodes / "seed-0-fixed8.trace.json.gz").write_bytes(b"partial")
            with (patch.object(adaptive, "verify_protocol"),
                  patch.object(adaptive, "smoke"),
                  patch.object(adaptive.torch.cuda, "is_available", return_value=True),
                  patch.object(adaptive, "load_policy") as load):
                with self.assertRaisesRegex(AssertionError, "partial trace"):
                    adaptive.run(output)
                load.assert_not_called()
            adaptive.base.save_json(output / "discovery-analysis.json", {"test": "analysis"})
            with (patch.object(adaptive.base, "verify_protocol"),
                  patch.object(adaptive, "fingerprints", return_value={}),
                  patch.object(adaptive.importlib.metadata, "version", return_value="test-version")):
                with self.assertRaisesRegex(AssertionError, "existing episodes"):
                    adaptive.prepare(output)


if __name__ == "__main__":
    unittest.main()
