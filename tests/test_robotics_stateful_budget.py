"""CPU-only checks for persistent eight/twelve/fifteen-action budgets.

The tests use synthetic camera features and in-memory action queues. They make
no pretrained-policy predictions and do not modify earlier frozen evidence.
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
import robotics_stateful_budget as screen


def features(**changes):
    result = {"quality": True, "history_steps": 8, "coverage": .60,
              "warning_count": 0, "drawdown": .01, "recovery": 0.,
              "recent_peak": .61}
    result.update(changes)
    return result


def regression(**changes):
    result = features(coverage=.82, warning_count=2, drawdown=.04,
                      recovery=-.02, recent_peak=.86)
    result.update(changes)
    return result


def commands(step=3.):
    return np.column_stack((100. + np.arange(15) * step,
                            np.full(15, 200.)))


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="stateful-budget-cpu-", dir=parent)


class BudgetScopeTest(unittest.TestCase):
    def test_only_approved_budgets_and_reference_methods(self):
        self.assertEqual(screen.BUDGETS, [8, 12, 15])
        self.assertEqual(screen.METHODS, ["fixed8", "adaptive12", "stateful15"])
        self.assertEqual(screen.RULES, {"promotion_units": 2,
                                       "minimum_unit_progress": 0.0})

    def test_evaluation_uses_fresh_scenes_and_one_fixed_sample(self):
        self.assertEqual(screen.SCENES, list(range(90, 110)))
        self.assertEqual(screen.DRAW, 30000)
        self.assertTrue(set(screen.SCENES).isdisjoint(range(90)))


class SmoothFutureTest(unittest.TestCase):
    def test_smooth_commands_are_eligible_at_every_budget(self):
        for budget in screen.BUDGETS:
            with self.subTest(budget=budget):
                self.assertTrue(screen.smooth_future(features(), commands(), budget))

    def test_eligibility_is_the_unchanged_milestone_rule(self):
        changes = ({"quality": False}, {"history_steps": 7},
                   {"coverage": .75}, {"warning_count": 1},
                   {"drawdown": .01501}, {"recovery": -.01001})
        for changed in changes:
            with self.subTest(changed=changed):
                self.assertFalse(screen.smooth_future(features(**changed), commands(), 12))
        self.assertTrue(screen.smooth_future(features(history_steps=8,
            drawdown=.015, recovery=-.01), commands(2.), 12))

    def test_missing_image_can_fail_without_numeric_fields(self):
        self.assertFalse(screen.smooth_future({"quality": False}, commands(), 8))

    def test_small_movements_and_direction_reversals_are_ineligible(self):
        self.assertFalse(screen.smooth_future(features(), commands(1.999), 8))
        for budget in screen.BUDGETS:
            plan = commands()
            plan[6:] = plan[5] - np.column_stack((
                3. * np.arange(1, 10), np.zeros(9)))
            with self.subTest(budget=budget):
                self.assertFalse(screen.smooth_future(features(), plan, budget))

    def test_whole_generated_plan_is_finite_even_outside_current_budget(self):
        for index in (0, 7, 14):
            for value in (np.nan, np.inf, -np.inf):
                plan = commands()
                plan[index, 0] = value
                with self.subTest(index=index, value=value):
                    self.assertFalse(screen.smooth_future(features(), plan, 8))

    def test_wrong_plan_shapes_are_ineligible(self):
        for plan in (np.zeros((14, 2)), np.zeros((16, 2)),
                     np.zeros((15, 3)), np.zeros(30), np.zeros((1, 15, 2))):
            with self.subTest(shape=plan.shape):
                self.assertFalse(screen.smooth_future(features(), plan, 12))

    def test_only_current_budget_suffix_controls_smoothness(self):
        plan = commands()
        plan[14] = plan[13]
        self.assertTrue(screen.smooth_future(features(), plan, 8))
        self.assertTrue(screen.smooth_future(features(), plan, 12))
        self.assertFalse(screen.smooth_future(features(), plan, 15))

    def test_no_dynamic_simulator_truth_is_used_or_mutated(self):
        row = features(true_coverage=.99, success=False,
                       contact=False, future_failure=True)
        old_row, plan = copy.deepcopy(row), commands()
        old_plan = plan.copy()
        self.assertTrue(screen.smooth_future(row, plan, 12))
        self.assertEqual(row, old_row)
        np.testing.assert_array_equal(plan, old_plan)


class StatefulControllerTest(unittest.TestCase):
    def update(self, controller, row=None, *, age=8, pending=0,
               step=40, plan=None, done=False):
        return controller.update(features() if row is None else row,
                                 age, pending, step,
                                 commands() if plan is None else plan,
                                 done=done)

    def unit(self, controller, *, start=.50, end=.52, step=40):
        budget = controller.begin(features(coverage=start))
        if budget > 8:
            event = self.update(controller, features(coverage=(start + end) / 2),
                                age=8, pending=budget - 8, step=step - budget + 8)
            self.assertEqual(event["kind"], "checkpoint_continue")
        return self.update(controller, features(coverage=end),
                           age=budget, pending=0, step=step)

    def test_initial_state_starts_at_eight_with_original_guard(self):
        controller = screen.StatefulBudgetController()
        self.assertEqual(controller.budget, 8)
        self.assertEqual(controller.streak, 0)
        self.assertIsInstance(controller.guard, screen.adaptive.AdaptiveLengthController)
        self.assertEqual(controller.begin(features()), 8)

    def test_two_successive_eligible_units_are_required_for_promotion(self):
        controller = screen.StatefulBudgetController()
        first = self.unit(controller)
        self.assertEqual(first["kind"], "unit_end")
        self.assertEqual(first["discarded"], 0)
        self.assertEqual((controller.budget, controller.streak), (8, 1))
        second = self.unit(controller, step=48)
        self.assertEqual((controller.budget, controller.streak), (12, 0))
        self.assertEqual(second["next_budget"], 12)
        self.assertEqual(second["streak"], 0)

    def test_promotion_is_one_rung_at_a_time_and_saturates_at_fifteen(self):
        controller = screen.StatefulBudgetController()
        budgets = []
        for index in range(8):
            self.unit(controller, step=40 + 15 * index)
            budgets.append(controller.budget)
        self.assertEqual(budgets[:4], [8, 12, 12, 15])
        self.assertTrue(all(budget == 15 for budget in budgets[4:]))

    def test_execution_budget_persists_between_predictions(self):
        controller = screen.StatefulBudgetController()
        self.unit(controller)
        self.unit(controller, step=48)
        self.assertEqual(controller.begin(features()), 12)
        self.assertEqual(controller.begin(features()), 12)
        self.assertEqual((controller.budget, controller.streak), (12, 0))

    def test_nonpositive_progress_breaks_promotion_streak(self):
        for end in (.50, .49):
            with self.subTest(end=end):
                controller = screen.StatefulBudgetController()
                self.unit(controller)
                self.unit(controller, end=end, step=48)
                self.assertEqual((controller.budget, controller.streak), (8, 0))
                self.unit(controller, step=56)
                self.assertEqual((controller.budget, controller.streak), (8, 1))

    def test_ineligible_completed_unit_resets_to_eight(self):
        for changed in ({"quality": False}, {"coverage": .80},
                        {"history_steps": 7}, {"warning_count": 1}):
            with self.subTest(changed=changed):
                controller = screen.StatefulBudgetController()
                controller.budget, controller.streak = 15, 1
                controller.begin(features(coverage=.50))
                event = self.update(controller, features(**{"coverage": .52, **changed}),
                                    age=15, pending=0)
                self.assertEqual(event["kind"], "unit_end")
                self.assertEqual((controller.budget, controller.streak), (8, 0))

    def test_unit_without_valid_start_estimate_cannot_be_promoted(self):
        controller = screen.StatefulBudgetController()
        controller.streak = 1
        controller.begin({"quality": False})
        event = self.update(controller, features(), age=8, pending=0)
        self.assertEqual(event["kind"], "unit_end")
        self.assertEqual((controller.budget, controller.streak), (8, 0))

    def test_unsmooth_completed_unit_resets_budget_and_promotion_streak(self):
        controller = screen.StatefulBudgetController()
        controller.budget, controller.streak = 12, 1
        controller.begin(features(coverage=.50))
        event = self.update(controller, features(coverage=.52),
                            age=12, pending=0, plan=commands(1.))
        self.assertEqual(event["kind"], "unit_end")
        self.assertEqual((controller.budget, controller.streak), (8, 0))

    def test_long_units_have_eight_movement_checkpoint(self):
        for budget in (12, 15):
            with self.subTest(budget=budget):
                controller = screen.StatefulBudgetController()
                controller.budget = budget
                controller.begin(features())
                event = self.update(controller, age=8, pending=budget - 8)
                self.assertEqual(event["kind"], "checkpoint_continue")
                self.assertEqual(event["discarded"], 0)
                self.assertEqual(controller.budget, budget)

    def test_checkpoint_rejection_discards_actual_pending_and_resets(self):
        for budget in (12, 15):
            with self.subTest(budget=budget):
                controller = screen.StatefulBudgetController()
                controller.budget, controller.streak = budget, 1
                controller.begin(features())
                event = self.update(controller, features(coverage=.80),
                                    age=8, pending=budget - 8)
                self.assertEqual(event["kind"], "checkpoint_replan")
                self.assertEqual(event["discarded"], budget - 8)
                self.assertEqual((controller.budget, controller.streak), (8, 0))

    def test_checkpoint_checks_smoothness_over_actual_scheduled_suffix(self):
        plan = commands()
        plan[14] = plan[13]
        for budget, kind in ((12, "checkpoint_continue"), (15, "checkpoint_replan")):
            with self.subTest(budget=budget):
                controller = screen.StatefulBudgetController()
                controller.budget = budget
                controller.begin(features())
                event = self.update(controller, age=8, pending=budget - 8, plan=plan)
                self.assertEqual(event["kind"], kind)

    def test_short_unit_does_not_trigger_milestone_suffix_extension(self):
        controller = screen.StatefulBudgetController()
        controller.begin(features(coverage=.50))
        event = self.update(controller, features(coverage=.52), age=8, pending=0)
        self.assertEqual(event["kind"], "unit_end")
        self.assertEqual(controller.budget, 8)

    def test_one_regression_observation_does_not_interrupt(self):
        controller = screen.StatefulBudgetController()
        controller.begin(features())
        self.assertIsNone(self.update(controller, regression(), age=4, pending=4, step=36))
        self.assertEqual(controller.guard.confirmations, 1)
        self.assertIsNone(self.update(controller, features(), age=5, pending=3, step=37))
        self.assertEqual(controller.guard.confirmations, 0)

    def test_original_two_confirmation_interrupt_resets_budget_and_streak(self):
        for budget in screen.BUDGETS:
            with self.subTest(budget=budget):
                controller = screen.StatefulBudgetController()
                controller.budget, controller.streak = budget, 1
                controller.begin(features())
                self.update(controller, regression(), age=4,
                            pending=budget - 4, step=36)
                event = self.update(controller, regression(), age=5,
                                    pending=budget - 5, step=37)
                self.assertEqual(event["kind"], "interrupt")
                self.assertEqual(event["discarded"], budget - 5)
                self.assertEqual((controller.budget, controller.streak), (8, 0))
                self.assertEqual(controller.guard.interruptions, 1)

    def test_discarded_unexecuted_future_commands_do_not_enable_interrupt(self):
        controller = screen.StatefulBudgetController()
        controller.begin(features())
        for age in (6, 7):
            event = self.update(controller, regression(), age=age,
                                pending=8 - age, step=36 + age)
        self.assertIsNone(event)
        self.assertEqual(controller.guard.interruptions, 0)

    def test_guard_state_matches_original_controller_for_same_logical_inputs(self):
        controller = screen.StatefulBudgetController()
        controller.budget = 15
        controller.begin(features())
        original = screen.adaptive.AdaptiveLengthController()
        records = [(features(), 1, 14, 31), (regression(), 4, 11, 34),
                   (regression(), 5, 10, 35), (features(), 1, 7, 36),
                   (regression(), 4, 4, 46), (regression(), 5, 3, 47),
                   (features(), 1, 7, 48), (regression(), 4, 4, 61),
                   (regression(), 5, 3, 62)]
        for row, age, pending, step in records:
            budget = controller.budget
            expected = original.update(row, age, pending, step, commands()[:budget])
            actual = self.update(controller, row, age=age,
                                 pending=pending, step=step)
            self.assertEqual(vars(controller.guard), vars(original))
            if expected and expected["kind"] == "interrupt":
                self.assertEqual(actual["kind"], "interrupt")
                self.assertEqual(actual["discarded"], expected["discarded"])

    def test_terminal_update_does_not_change_controller_or_guard(self):
        controller = screen.StatefulBudgetController()
        controller.budget, controller.streak = 12, 1
        controller.begin(features())
        before = copy.deepcopy(vars(controller))
        guard_before = copy.deepcopy(vars(controller.guard))
        with patch.object(controller.guard, "update", side_effect=AssertionError("terminal guard called")):
            self.assertIsNone(self.update(controller, regression(), age=8,
                                         pending=4, done=True))
        self.assertEqual(controller.budget, before["budget"])
        self.assertEqual(controller.streak, before["streak"])
        self.assertEqual(vars(controller.guard), guard_before)

    def test_truth_fields_do_not_change_stateful_decisions(self):
        original = screen.StatefulBudgetController()
        altered = screen.StatefulBudgetController()
        for controller in (original, altered):
            controller.begin(features(coverage=.50))
        expected = self.update(original, features(coverage=.52))
        actual = self.update(altered, features(coverage=.52,
            true_coverage=.99, true_contact=False, true_success=False))
        self.assertEqual(actual, expected)
        self.assertEqual((altered.budget, altered.streak),
                         (original.budget, original.streak))


class ActionQueueTest(unittest.TestCase):
    def policy(self):
        actions = [object() for _ in range(14)]
        history = deque([object(), object()], maxlen=2)
        queues = {"action": deque(actions, maxlen=15),
                  "observation.state": history,
                  "observation.images.top": deque([object(), object()], maxlen=2)}
        return SimpleNamespace(_queues=queues), actions, history

    def test_trim_preserves_exact_prefix_objects_order_and_capacity(self):
        for budget in screen.BUDGETS:
            with self.subTest(budget=budget):
                policy, actions, _ = self.policy()
                screen.select_budget(policy, budget)
                self.assertEqual(len(policy._queues["action"]), budget - 1)
                self.assertEqual(policy._queues["action"].maxlen, 15)
                self.assertTrue(all(a is b for a, b in
                                    zip(policy._queues["action"], actions[:budget - 1])))

    def test_observation_history_objects_are_not_reset_replaced_or_mutated(self):
        policy, _, history = self.policy()
        image_history = policy._queues["observation.images.top"]
        originals = {name: list(queue) for name, queue in policy._queues.items()
                     if name != "action"}
        screen.select_budget(policy, 8)
        self.assertIs(policy._queues["observation.state"], history)
        self.assertIs(policy._queues["observation.images.top"], image_history)
        for name, contents in originals.items():
            self.assertTrue(all(a is b for a, b in zip(policy._queues[name], contents)))

    def test_no_policy_reset_is_called_during_budget_selection(self):
        policy, _, _ = self.policy()
        policy.reset = lambda: self.fail("budget selection erased policy history")
        screen.select_budget(policy, 12)

    def test_invalid_budget_is_rejected_without_mutating_live_queues(self):
        for budget in (0, 4, 6, 16, 20):
            with self.subTest(budget=budget):
                policy, actions, history = self.policy()
                with self.assertRaises((AssertionError, ValueError)):
                    screen.select_budget(policy, budget)
                self.assertEqual(list(policy._queues["action"]), actions)
                self.assertIs(policy._queues["observation.state"], history)

    def test_selection_requires_exact_fifteen_action_fresh_prediction(self):
        for count in (0, 3, 7, 13, 15):
            with self.subTest(count=count):
                policy, _, _ = self.policy()
                policy._queues["action"] = deque([object() for _ in range(count)], maxlen=15)
                with self.assertRaises((AssertionError, ValueError)):
                    screen.select_budget(policy, 8)


class FakePolicy:
    """In-memory LeRobot queue/history double; no neural inference."""

    def __init__(self):
        self.config = SimpleNamespace(horizon=16, n_action_steps=8,
                                      n_obs_steps=2, down_dims=(512, 1024, 2048))
        self.diffusion = SimpleNamespace(config=copy.copy(self.config),
            num_inference_steps=100, conditional_sample=self.conditional_sample)
        self.weight = screen.torch.nn.Parameter(screen.torch.ones(1))
        self.resets, self.all_seen, self.all_histories = 0, [], []

    def parameters(self):
        return iter([self.weight])

    def conditional_sample(self):
        return screen.torch.zeros((1, self.diffusion.config.horizon, 2))

    def reset(self):
        self.resets += 1
        self.queries = 0
        self._queues = {"action": deque(maxlen=self.config.n_action_steps),
                        "observation.state": deque(maxlen=2)}

    def select_action(self, observation):
        step = observation["step"]
        self.all_seen.append(step)
        self._queues["observation.state"].append(step)
        self.all_histories.append(list(self._queues["observation.state"]))
        if not self._queues["action"]:
            self.diffusion.conditional_sample()
            self.queries += 1
            for index in range(self.config.n_action_steps):
                self._queues["action"].append(screen.torch.tensor(
                    [[100. + 100. * self.queries + 3. * index, 200.]],
                    dtype=screen.torch.float64))
        return self._queues["action"].popleft()


class FakeEnvironment:
    def __init__(self, finish=56, fail_step=None, truncate=None):
        self.finish, self.fail_step, self.truncate = finish, fail_step, truncate
        self.closed = False

    def reset(self, seed):
        self.scene_seed, self.step_index, self.last_command = seed, 0, [0., 0.]
        return {"step": 0, "pixels": 0}, {}

    def step(self, command):
        self.step_index += 1
        if self.step_index == self.fail_step:
            raise RuntimeError("synthetic movement error")
        self.last_command = np.asarray(command).tolist()
        return ({"step": self.step_index, "pixels": self.step_index}, 0.,
                self.step_index == self.finish, self.step_index == self.truncate, {})

    def close(self):
        self.closed = True


class FakeEstimator:
    def __init__(self, near=False):
        self.calls, self.near = 0, near

    def estimate(self, pixels):
        self.calls += 1
        coverage = .80 if self.near else .40 + .002 * pixels
        return {"quality": True, "coverage": coverage}


class FakeRolling:
    def update(self, estimate, pending):
        return {**features(coverage=estimate["coverage"], drawdown=0.,
                           recent_peak=estimate["coverage"]),
                "pending_actions": pending}


def physical_state(env, observation):
    return {"coverage": .96 if env.step_index == env.finish else .60,
            "block_position": [100., 200.], "block_angle": .2,
            "agent_position": env.last_command,
            "image_sha256": f"image-{env.step_index}"}


def episode_patches(env, estimator):
    stack = ExitStack()
    stack.enter_context(patch.object(screen.gym, "make", return_value=env))
    stack.enter_context(patch.object(screen.adaptive, "observation_batch", side_effect=lambda obs, device: obs))
    stack.enter_context(patch.object(screen.adaptive, "physical_state", side_effect=physical_state))
    stack.enter_context(patch.object(screen.adaptive, "VisualAlignmentEstimator", return_value=estimator))
    stack.enter_context(patch.object(screen.adaptive, "RollingVisualFeatures", return_value=FakeRolling()))
    stack.enter_context(patch.object(screen.torch.cuda, "synchronize"))
    stack.enter_context(patch.object(screen.torch.cuda, "reset_peak_memory_stats"))
    stack.enter_context(patch.object(screen.torch.cuda, "max_memory_allocated", return_value=4096))
    return stack


def write_trial(directory, trial, trace):
    with gzip.open(directory / trial["trace"], "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False)
    trial["trace_sha256"] = screen.base.sha(directory / trial["trace"])
    screen.base.save_json(directory / (screen.gen.stem(trial["scene"], trial["draw"],
                                                     trial["method"]) + ".json"), trial)


def synthetic_episode(directory, method="stateful15", near=False, force=None,
                      finish=56, truncate=None, scene=90, draw=30000):
    policy, env, estimator = FakePolicy(), FakeEnvironment(finish, truncate=truncate), FakeEstimator(near)
    with episode_patches(env, estimator):
        trial = screen.evaluate_episode(policy, scene, draw, method, directory,
                                        max_steps=56, force_budget=force)
    return trial, screen.gen.read_trace(trial, directory), policy, env, estimator


class EpisodeIntegrationTest(unittest.TestCase):
    def test_controller_reaches_all_three_budgets_without_changing_generation(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace, policy, env, estimator = synthetic_episode(directory)
            self.assertEqual(screen.verify_trial(trial, directory, max_steps=56), trace)
        self.assertTrue(env.closed)
        self.assertEqual(env.scene_seed, 90)
        self.assertEqual(trial["unit_lengths"], [8, 8, 12, 12, 15, 1])
        self.assertEqual(trial["generated_shapes"], [[1, 16, 2]] * 6)
        self.assertEqual(trial["model_queries"], 6)
        self.assertEqual([r["selected_budget"] for r in trace["steps"] if r["new_prediction"]],
                         [8, 8, 12, 12, 15, 15])
        self.assertTrue(all(len(r["predicted_commands"]) == 15
                            for r in trace["steps"] if r["new_prediction"]))
        self.assertEqual(estimator.calls, 56)  # Initial frame and 55 nonterminal frames.
        self.assertEqual(policy.resets, 2)  # Episode setup and final restoration only.
        self.assertEqual(policy.all_seen, list(range(56)))
        self.assertEqual(policy.all_histories[-1], [54, 55])
        self.assertEqual(policy.config.horizon, 16)
        self.assertEqual(policy.diffusion.config.horizon, 16)
        self.assertEqual(policy.config.n_action_steps, 8)
        self.assertEqual(policy._queues["action"].maxlen, 8)

    def test_near_target_candidate_reverts_to_eight_without_using_truth(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace, _, _, _ = synthetic_episode(directory, near=True)
            screen.verify_trial(trial, directory, max_steps=56)
        self.assertEqual(trial["unit_lengths"], [8] * 7)
        self.assertEqual(trial["final_budget"], 8)
        self.assertEqual(trial["final_streak"], 0)
        self.assertTrue(all(r["selected_budget"] == 8 for r in trace["steps"]))

    def test_reference_methods_delegate_to_unchanged_generation_runner(self):
        for method in ("fixed8", "adaptive12"):
            sentinel = object()
            with self.subTest(method=method), patch.object(screen.gen, "evaluate_episode", return_value=sentinel) as evaluate:
                result = screen.evaluate_episode("policy", 90, 30000, method, Path("new"), 56)
                self.assertIs(result, sentinel)
                evaluate.assert_called_once_with("policy", 90, 30000, method, Path("new"), 56)

    def test_fixed_eight_reference_never_uses_monitor(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace, policy, _, estimator = synthetic_episode(directory, method="fixed8")
            self.assertEqual(screen.verify_trial(trial, directory, max_steps=56), trace)
        self.assertEqual(trial["unit_lengths"], [8] * 7)
        self.assertEqual(estimator.calls, 0)
        self.assertEqual(trial["monitor_seconds"], 0.)
        self.assertEqual(trial["events"], [])
        self.assertEqual(policy.all_histories[-1], [54, 55])

    def test_forced_eight_and_fifteen_are_smoke_only_and_keep_full_generation(self):
        for force, expected in ((8, [8] * 7), (15, [15, 15, 15, 11])):
            with self.subTest(force=force), test_directory() as name:
                directory = Path(name)
                trial, trace, _, _, estimator = synthetic_episode(directory, force=force)
                with self.assertRaises(AssertionError):
                    screen.verify_trial(trial, directory, max_steps=56)
                self.assertEqual(screen.verify_trial(trial, directory, max_steps=56,
                                                    allow_force=True), trace)
                self.assertEqual(trial["unit_lengths"], expected)
                self.assertIsNone(trial["initial_features"])
                self.assertEqual(trial["events"], [])
                self.assertEqual(estimator.calls, 0)
                self.assertEqual(trial["generated_shapes"], [[1, 16, 2]] * len(expected))

    def test_environment_error_still_closes_and_restores_model_configuration(self):
        policy, env, estimator = FakePolicy(), FakeEnvironment(fail_step=4), FakeEstimator()
        with test_directory() as name, episode_patches(env, estimator):
            with self.assertRaisesRegex(RuntimeError, "synthetic movement error"):
                screen.evaluate_episode(policy, 90, 30000, "stateful15", Path(name), max_steps=56)
            self.assertEqual(list(Path(name).iterdir()), [])
        self.assertTrue(env.closed)
        self.assertEqual(policy.resets, 2)
        self.assertEqual(policy.config.n_action_steps, 8)

    def test_candidate_configuration_rejects_different_generated_horizon(self):
        policy = FakePolicy()
        policy.config.horizon = 8
        with self.assertRaises(AssertionError):
            screen.configure_candidate(policy)
        self.assertEqual(policy.resets, 0)


class TrialIntegrityTest(unittest.TestCase):
    def test_rehashed_trace_tampering_does_not_pass_state_machine_validation(self):
        mutations = (
            lambda t, r: r["steps"][0].update(selected_budget=12),
            lambda t, r: r["steps"][0].update(selection_discarded=0),
            lambda t, r: r["steps"][0].update(pending_before=14),
            lambda t, r: r["steps"][0].update(pending_after=14),
            lambda t, r: r["steps"][0].update(predicted_commands=[]),
            lambda t, r: r["steps"][1].update(action=[-1., -1.]),
            lambda t, r: r["steps"][1].update(new_prediction=True),
            lambda t, r: r["steps"][1].update(plan_age=1),
            lambda t, r: r["steps"][1].update(selection_discarded=7),
            lambda t, r: r["steps"][7]["features"].update(pending_actions=4),
            lambda t, r: t["events"][0].update(next_budget=15),
            lambda t, r: t.update(unit_lengths=[8] * 7),
            lambda t, r: t.update(final_budget=8),
            lambda t, r: t.update(final_streak=0),
            lambda t, r: t.update(model_queries=99),
            lambda t, r: t.update(generated_shapes=[[1, 8, 2]] * 6),
            lambda t, r: r["steps"][-1].update(features=features()),
            lambda t, r: r.update(scene=91))
        with test_directory() as name:
            directory = Path(name)
            original, trace, _, _, _ = synthetic_episode(directory)
            for mutate in mutations:
                trial, altered = copy.deepcopy(original), copy.deepcopy(trace)
                mutate(trial, altered)
                write_trial(directory, trial, altered)
                with self.subTest(mutation=mutate), self.assertRaises((AssertionError, KeyError)):
                    screen.verify_trial(trial, directory, max_steps=56)

    def test_wrong_identity_horizon_hash_or_path_is_rejected(self):
        with test_directory() as name:
            directory = Path(name)
            trial, _, _, _, _ = synthetic_episode(directory)
            for changed in ({"scene": 89}, {"draw": 29999}, {"method": "weighted"},
                            {"prediction_seed": 0}, {"horizon": 8}, {"execution": 12},
                            {"trace_sha256": "0" * 64}, {"trace": "../old.trace.json.gz"}):
                with self.subTest(changed=changed), self.assertRaises((AssertionError, KeyError)):
                    screen.verify_trial({**trial, **changed}, directory, max_steps=56)

    def test_strict_success_threshold_and_full_timeout_are_not_relaxed(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace, _, _, _ = synthetic_episode(directory, finish=57, truncate=56)
            screen.verify_trial(trial, directory, max_steps=56)
            trace["steps"][-1]["after"]["coverage"] = .95
            trial["final_coverage"] = .95
            write_trial(directory, trial, trace)
            screen.verify_trial(trial, directory, max_steps=56)
            for changed in ({"success": True}, {"truncated": False}, {"steps": 55}):
                with self.subTest(changed=changed), self.assertRaises(AssertionError):
                    screen.verify_trial({**trial, **changed}, directory, max_steps=56)

    def test_nonfinite_costs_and_nonpositive_timing_are_rejected(self):
        mutations = (lambda t, r: t.update(wall_seconds=float("inf")),
                     lambda t, r: t.update(monitor_seconds=float("nan")),
                     lambda t, r: t.update(peak_gpu_allocated_bytes=0),
                     lambda t, r: r["steps"][0].update(prediction_seconds=0.),
                     lambda t, r: r["steps"][0].update(prediction_seconds=float("nan")))
        with test_directory() as name:
            directory = Path(name)
            original, trace, _, _, _ = synthetic_episode(directory)
            for mutate in mutations:
                trial, altered = copy.deepcopy(original), copy.deepcopy(trace)
                mutate(trial, altered)
                with self.subTest(mutation=mutate), patch.object(screen.gen, "read_trace", return_value=altered):
                    with self.assertRaises(AssertionError):
                        screen.verify_trial(trial, directory, max_steps=56)

    def test_completed_sidecar_is_verified_without_new_predictions(self):
        with test_directory() as name:
            directory = Path(name)
            original, _, _, _, _ = synthetic_episode(directory)
            with patch.object(screen, "evaluate_episode") as evaluate:
                actual = screen.collect(object(), 90, 30000, "stateful15", directory, 56)
            evaluate.assert_not_called()
            self.assertEqual(actual, original)

    def test_orphan_trace_and_partial_sidecar_never_get_overwritten(self):
        for suffix in (".trace.json.gz", ".json.tmp"):
            with self.subTest(suffix=suffix), test_directory() as name:
                directory = Path(name)
                orphan = directory / (screen.gen.stem(90, 30000, "stateful15") + suffix)
                orphan.write_bytes(b"unfinished")
                with patch.object(screen, "evaluate_episode") as evaluate:
                    with self.assertRaisesRegex(AssertionError, "Inspect"):
                        screen.collect(object(), 90, 30000, "stateful15", directory, 56)
                evaluate.assert_not_called()
                self.assertEqual(orphan.read_bytes(), b"unfinished")


def metric_fixture():
    result = []
    for scene in screen.SCENES:
        for method in screen.METHODS:
            success = scene % 2 == 0
            if method == "stateful15" and scene in (90, 91):
                success = not success
            result.append({"scene": scene, "draw": screen.DRAW, "method": method,
                           "success": success,
                           "final_coverage": .80 + .01 * (method == "stateful15"),
                           "model_queries": 20 if method == "stateful15" else 30,
                           "wall_seconds": 100. if method == "stateful15" else 150.,
                           "monitor_seconds": 2. if method != "fixed8" else 0.,
                           "prediction_seconds": [.1, .3]})
    return result


class ScheduleAndStatisticsTest(unittest.TestCase):
    def test_schedule_contains_sixty_exact_unique_scene_method_pairs(self):
        schedule = screen.schedule()
        self.assertEqual(len(schedule), 60)
        self.assertEqual({(r["scene"], r["draw"], r["method"]) for r in schedule},
                         {(s, screen.DRAW, m) for s in screen.SCENES for m in screen.METHODS})
        for index, scene in enumerate(screen.SCENES):
            actual = [r["method"] for r in schedule if r["scene"] == scene]
            self.assertEqual(actual, screen.METHODS[index % 3:] + screen.METHODS[:index % 3])
        self.assertEqual(schedule, screen.schedule())

    def test_only_new_dedicated_result_directories_are_allowed(self):
        screen.check_output(screen.OUTPUT)
        screen.check_output(screen.OUTPUT.with_name(screen.OUTPUT.name + "-repeat"))
        for path in (Path("results"), Path("results/robotics-adaptive-length"),
                     Path("results/robotics-generation-length"), screen.OUTPUT / "nested",
                     Path("../outside")):
            with self.subTest(path=path), self.assertRaises(AssertionError):
                screen.check_output(path)

    def test_table_recomputes_all_approved_metrics(self):
        table = screen.metric_table(metric_fixture())
        for method in screen.METHODS:
            self.assertEqual(table[method]["attempts"], 20)
            self.assertEqual(table[method]["successes"], 10)
            self.assertAlmostEqual(table[method]["mean_prediction_seconds"], .2)
        self.assertEqual(table["stateful15"]["mean_model_queries"], 20)
        self.assertEqual(table["fixed8"]["mean_model_queries"], 30)
        self.assertEqual(table["stateful15"]["mean_wall_seconds"], 100.)
        self.assertAlmostEqual(table["stateful15"]["mean_final_coverage"], .81)

    def test_incomplete_or_duplicate_panels_are_rejected(self):
        original = metric_fixture()
        for rows in (original[:-1], original + [original[0]], original[:-1] + [original[0]]):
            with self.subTest(length=len(rows)), self.assertRaises(AssertionError):
                screen.metric_table(rows)
        with self.assertRaises(AssertionError):
            screen.paired(original + [original[0]], "fixed8")

    def test_paired_contrasts_preserve_sign_scene_identity_and_reproducibility(self):
        trials = metric_fixture()
        first = screen.paired(trials, "fixed8")
        self.assertEqual(first, screen.paired(list(reversed(trials)), "fixed8"))
        self.assertEqual(first["gained_successes"], [91])
        self.assertEqual(first["lost_successes"], [90])
        costs = first["differences_method_minus_reference"]["model_queries"]
        self.assertEqual(costs["mean_difference"], -10.)
        self.assertEqual(costs["scene_pair_bootstrap_95_interval"], [-10., -10.])
        self.assertEqual(first["differences_method_minus_reference"]["success"]["mean_difference"], 0.)

    def test_bootstrap_uses_twenty_scene_pairs_not_sixty_independent_trials(self):
        trials = metric_fixture()
        actual = screen.paired(trials, "adaptive12")
        values = np.asarray([-1. if s == 90 else 1. if s == 91 else 0.
                             for s in screen.SCENES])
        rng = np.random.default_rng(screen.STATISTICS["bootstrap_seed"])
        indices = rng.integers(0, 20, size=(screen.STATISTICS["bootstrap_samples"], 20))
        expected = np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
        self.assertEqual(actual["differences_method_minus_reference"]["success"][
            "scene_pair_bootstrap_95_interval"], expected)


class ProtocolTest(unittest.TestCase):
    def inherited(self):
        return {"fingerprints": {}, "packages": {"fake": "1"},
                "checkpoint": {"weights_sha256": "weights", "config_sha256": "configuration"},
                "model_sources": {"fake.module": "hash"}, "timing": "test-only"}

    def test_protocol_binds_candidate_generation_rules_sources_and_exact_panel(self):
        with test_directory() as name, patch.object(screen, "check_output"), \
                patch.object(screen.gen, "protocol_document", side_effect=self.inherited):
            output = Path(name)
            document = screen.prepare(output)
            self.assertEqual(document["budgets"], [8, 12, 15])
            self.assertEqual(document["initial_budget"], 8)
            self.assertEqual(document["generation"], {"horizon": 16,
                "usable_future_commands": 15, "observations": 2, "denoising_steps": 100,
                "training": False})
            self.assertFalse(document["runtime_truth_inputs"])
            self.assertEqual(document["schedule"], screen.schedule())
            self.assertEqual(document["reused_milestone_rules"], screen.adaptive.RULES)
            self.assertIn("experiments/robotics_stateful_budget.py", document["fingerprints"])
            encoded = (output / "protocol.json").read_bytes()
            self.assertEqual(screen.prepare(output), document)
            self.assertEqual((output / "protocol.json").read_bytes(), encoded)

    def test_modified_protocol_never_resumes_or_rebinds_existing_evidence(self):
        with test_directory() as name, patch.object(screen, "check_output"), \
                patch.object(screen.gen, "protocol_document", side_effect=self.inherited):
            output = Path(name)
            original = screen.prepare(output)
            for key in ("generation", "rules", "reused_milestone_rules", "scenes", "draw",
                        "schedule", "checkpoint", "packages", "fingerprints"):
                altered = {**original, key: "changed"}
                screen.base.save_json(output / "protocol.json", altered)
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    screen.prepare(output)

    def test_unbound_smoke_episode_or_summary_is_not_adopted(self):
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
