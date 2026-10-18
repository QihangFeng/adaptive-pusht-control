"""CPU-only frozen-protocol, selection and resume tests; never load a policy."""

from collections import deque
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_benefit_pilot as pilot


def features(step=33, pending=4, quality=True, warning_count=2, drawdown=0.05):
    result = {"step": step, "quality": quality, "coverage": 0.80,
              "warning_count": warning_count, "drawdown": drawdown,
              "recovery": -0.05, "stagnation": 0.0, "position_error": 0.03,
              "orientation_error": 0.04, "pending_actions": pending,
              "history_steps": 24, "recent_best_gain": 0.0, "recent_peak": 0.85}
    if not quality:
        result.update({name: None for name in pilot.FEATURE_NAMES})
        result["pending_actions"] = pending
    return result


def episode(seed, steps=60, candidate_step=33):
    rows = []
    for step in range(1, steps + 1):
        row = features(step, warning_count=0, drawdown=0.0)
        row["coverage"] = 0.85 if step >= candidate_step - 1 else 0.60
        if step >= candidate_step:
            row = features(step)
        rows.append(row)
    return {"seed": seed, "steps": steps, "features": rows}


class EligibilityTest(unittest.TestCase):
    def test_warning_and_net_drawdown_are_both_required(self):
        self.assertEqual(pilot.eligibility(features(), [0.85, 0.80], 33, 4), "regression")
        for change in [{"warning_count": 1}, {"drawdown": 0.029},
                       {"quality": False}]:
            item = features()
            item.update(change)
            self.assertIsNone(pilot.eligibility(item, [0.85, 0.80], 33, 4))

    def test_missing_visual_values_abstain_without_numeric_operations(self):
        self.assertIsNone(pilot.eligibility(features(quality=False), [], 33, 4))

    def test_eligibility_time_and_pending_guards(self):
        for step, pending in [(31, 4), (285, 4), (33, 1), (33, 0)]:
            self.assertIsNone(pilot.eligibility(features(), [0.85, 0.80], step, pending))
        self.assertEqual(pilot.eligibility(features(), [0.85, 0.80], 32, 2), "regression")
        self.assertEqual(pilot.eligibility(features(), [0.85, 0.80], 284, 2), "regression")

    def test_recent_peak_not_unbounded_history_peak(self):
        history = [0.99] + [0.70] * 8
        self.assertIsNone(pilot.eligibility(features(), history, 33, 4))
        self.assertFalse(pilot.hand_decision(features(), history))

    def test_stagnation_requires_complete_window_and_high_coverage(self):
        item = features(warning_count=0, drawdown=0.0)
        self.assertIsNone(pilot.eligibility(item, [0.80] * 24, 40, 4))
        self.assertEqual(pilot.eligibility(item, [0.80] * 25, 40, 4), "stagnation")
        self.assertIsNone(pilot.eligibility(item, [0.74] * 25, 40, 4))
        self.assertIsNone(pilot.eligibility(item, [0.80] + [0.83] * 24, 40, 4))

    def test_hand_rule_does_not_treat_stagnation_as_alignment_warning(self):
        item = features(warning_count=0, drawdown=0.0)
        self.assertEqual(pilot.eligibility(item, [0.80] * 25, 40, 4), "stagnation")
        self.assertFalse(pilot.hand_decision(item, [0.80] * 25))

    def test_vector_order_ignores_extra_oracle_fields(self):
        item = features()
        item.update(simulator_contact=123, actual_final_success=True)
        expected = [float(item[name]) for name in pilot.FEATURE_NAMES]
        self.assertEqual(pilot.vector(item), expected)
        self.assertEqual(len(pilot.vector(item)), 8)

    def test_vector_rejects_nonfinite_or_missing_controller_input(self):
        for value in [float("nan"), float("inf")]:
            item = features()
            item["coverage"] = value
            with self.assertRaises(ValueError):
                pilot.vector(item)
        item = features()
        del item["drawdown"]
        with self.assertRaises(KeyError):
            pilot.vector(item)


class HorizonTest(unittest.TestCase):
    def test_configs_and_capacity_change_before_reset_and_keep_generated_prefix(self):
        class FakePolicy:
            def __init__(self):
                self.config = SimpleNamespace(n_action_steps=8)
                self.diffusion = SimpleNamespace(config=SimpleNamespace(n_action_steps=8))
                self._queues = {"action": deque([99], maxlen=8)}
                self.reset_calls = 0

            def reset(self):
                assert self.config.n_action_steps == self.diffusion.config.n_action_steps
                self.reset_calls += 1
                self._queues = {"action": deque(maxlen=self.config.n_action_steps)}

        policy = FakePolicy()
        pilot.configure_horizon(policy, 4)
        self.assertEqual(policy.reset_calls, 1)
        self.assertEqual(policy.config.n_action_steps, 4)
        self.assertEqual(policy.diffusion.config.n_action_steps, 4)
        self.assertEqual(policy._queues["action"].maxlen, 4)
        generated = list(range(8))[:policy.diffusion.config.n_action_steps]
        policy._queues["action"].extend(generated)
        self.assertEqual(list(policy._queues["action"]), [0, 1, 2, 3])
        pilot.configure_horizon(policy, 8)
        self.assertEqual(policy._queues["action"].maxlen, 8)
        self.assertEqual(list(policy._queues["action"]), [])


class OneDecisionExecutionTest(unittest.TestCase):
    """Artificial observations exercise the gate lifecycle, not robot performance."""

    def run_fake_episode(self, method):
        class FakeAction:
            def squeeze(self, axis):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return pilot.np.zeros(2, dtype=pilot.np.float32)

        class FakePolicy:
            config = SimpleNamespace(n_action_steps=8)
            diffusion = SimpleNamespace(config=SimpleNamespace(n_action_steps=8))

            def reset(self):
                self._queues = {"action": deque(maxlen=self.config.n_action_steps)}

            def select_action(self, observation):
                if not self._queues["action"]:
                    self._queues["action"].extend(FakeAction() for _ in range(self.config.n_action_steps))
                return self._queues["action"].popleft()

        class FakeEnv:
            step_index = 0

            def reset(self, seed):
                return {"pixels": "unused mocked camera"}, {}

            def step(self, command):
                self.step_index += 1
                return {"pixels": "unused mocked camera"}, 0, self.step_index == 40, False, {}

            def close(self):
                pass

        class FakeMonitor:
            step_index = -1

            def update(self, estimate, pending):
                self.step_index += 1
                warned = self.step_index >= 35
                item = features(self.step_index, pending=pending,
                                warning_count=2 if warned else 0, drawdown=0.05 if warned else 0)
                item["coverage"] = 0.75 if warned else 0.80
                return item

        policy = FakePolicy()
        model = {"tree": {}, "threshold": 0.1, "minimum_margin": 0.05}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pilot.gym, "make", return_value=FakeEnv()), \
                patch.object(pilot, "VisualAlignmentEstimator") as estimator, \
                patch.object(pilot, "RollingVisualFeatures", FakeMonitor), \
                patch.object(pilot, "physical_state", side_effect=lambda env, observation:
                             {"coverage": 0.96 if env.step_index == 40 else 0.8}), \
                patch.object(pilot, "observation_batch", return_value={}), \
                patch.object(pilot.torch.cuda, "synchronize"), \
                patch.object(pilot, "predict_tree", return_value=0.5) as prediction:
            estimator.return_value.estimate.return_value = {"quality": True, "coverage": 0.80}
            result = pilot.evaluate_episode(policy, 31, method, model, Path(directory))
            calls = prediction.call_count
            image_calls = estimator.return_value.estimate.call_count
        return result, calls, image_calls

    def test_warning_rule_keeps_monitoring_after_nonwarning_stagnation(self):
        result, calls, _ = self.run_fake_episode("warning_rule")
        self.assertEqual(result["decision"]["step"], 35)
        self.assertEqual(result["decision"]["reason"], "regression")
        self.assertTrue(result["decision"]["replan"])
        self.assertEqual(calls, 0)

    def test_learned_gate_decides_once_at_first_eligible_point(self):
        result, calls, _ = self.run_fake_episode("learned_gate")
        self.assertEqual(result["decision"]["step"], 33)
        self.assertEqual(result["decision"]["reason"], "stagnation")
        self.assertTrue(result["decision"]["replan"])
        self.assertEqual(calls, 1)

    def test_fixed_baseline_never_reads_camera_monitor_or_prediction(self):
        result, calls, image_calls = self.run_fake_episode("fixed8")
        self.assertIsNone(result["decision"])
        self.assertEqual(calls, 0)
        self.assertEqual(image_calls, 0)


class TraceResumeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "trial.trace.json.gz"
        self.trace = {"steps": [
            {"step": step, "new_prediction": step != 102, "action": [0.0, 0.0],
             "after": {"coverage": 0.96 if step == 103 else 0.8}}
            for step in range(101, 104)]}
        self.trial = {"trace": self.path.name, "remaining_actions_executed": 3,
                      "remaining_model_queries": 2, "success": True,
                      "final_coverage": 0.96}
        self.write_trace()

    def write_trace(self):
        with gzip.open(self.path, "wt", encoding="utf-8") as stream:
            json.dump(self.trace, stream, allow_nan=False)

    def test_valid_continuation_checks_global_step_offset_and_query_count(self):
        self.assertEqual(pilot.verify_trial(self.trial, self.directory, branch=100), self.trace)
        self.trial["trace_sha256"] = pilot.sha(self.path)
        self.assertEqual(pilot.verify_trial(self.trial, self.directory, branch=100), self.trace)

    def test_missing_or_hash_changed_trace_rejected(self):
        trial = {**self.trial, "trace": "missing.trace.json.gz"}
        with self.assertRaisesRegex(AssertionError, "missing its trace"):
            pilot.verify_trial(trial, self.directory, branch=100)
        self.trial["trace_sha256"] = pilot.sha(self.path)
        self.trace["steps"][0]["after"]["coverage"] = 0.7
        self.write_trace()
        with self.assertRaisesRegex(AssertionError, "Saved trace changed"):
            pilot.verify_trial(self.trial, self.directory, branch=100)

    def test_wrong_count_outcome_final_coverage_or_queries_rejected(self):
        for field, value in [("remaining_actions_executed", 4), ("success", False),
                             ("final_coverage", 0.94), ("remaining_model_queries", 3)]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                pilot.verify_trial({**self.trial, field: value}, self.directory, branch=100)
        with self.assertRaises(AssertionError):
            pilot.verify_trial(self.trial, self.directory, branch=99)


class PreparedPilotTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / "pilot"
        self.output.mkdir()
        self.model = self.directory / "model"
        self.model.mkdir()
        self.discovery = self.directory / "discovery"
        self.discovery.mkdir()
        pilot.save_json(self.model / "config.json", {
            "n_obs_steps": 2, "horizon": 16, "n_action_steps": 8})
        pilot.save_json(self.output / "visual-validation.json", {"passed": True})
        pilot.save_json(self.output / "near-target-validation.json", {"passed": True})
        for name, value in [("MODEL", self.model), ("DISCOVERY", self.discovery),
                            ("source_fingerprints", lambda: {"original": "same"}),
                            ("code_fingerprints", lambda: {"runner": "same"})]:
            mocked = patch.object(pilot, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def write_episodes(self, episodes):
        with gzip.open(self.output / "visual-features.json.gz", "wt", encoding="utf-8") as stream:
            json.dump({"episodes": episodes}, stream, allow_nan=False)
        pilot.save_json(self.discovery / "summary.json", {"episodes": [
            {"seed": item["seed"], "steps": item["steps"], "success": True, "truncated": False}
            for item in episodes]})

    def prepare(self, episodes=None):
        self.write_episodes([episode(seed) for seed in range(12)] if episodes is None else episodes)
        with patch.object(pilot, "load_policy") as load:
            pilot.prepare(self.output)
        load.assert_not_called()
        return pilot.read_json(self.output / "protocol.json")

    def test_selection_is_first_eligible_point_in_seed_order_and_bounded(self):
        protocol = self.prepare([episode(seed) for seed in reversed(range(24))])
        self.assertEqual([case["episode_id"] for case in protocol["cases"]], list(range(20)))
        self.assertEqual([case["branch_step"] for case in protocol["cases"]], [33] * 20)
        self.assertEqual(len({case["checkpoint_id"] for case in protocol["cases"]}), 20)

    def test_split_is_by_episode_and_separate_from_held_out_evaluation(self):
        protocol = self.prepare()
        train, val = set(protocol["train_episode_ids"]), set(protocol["validation_episode_ids"])
        test = set(protocol["evaluation_seeds"])
        self.assertEqual(val, {2, 5, 8, 11})
        self.assertFalse(train & val)
        self.assertFalse((train | val) & test)
        self.assertEqual(train | val, set(range(12)))
        self.assertEqual(protocol["tree"], {"max_depth": 2, "min_leaf": 3})
        self.assertEqual(protocol["max_total_actions"], 300)

    def test_checkpoint_selection_ignores_success_labels_and_future_features(self):
        originals = [episode(seed) for seed in range(12)]
        first = self.prepare(originals)
        (self.output / "protocol.json").unlink()
        changed = copy.deepcopy(originals)
        for item in changed:
            item["success"] = item["seed"] % 2 == 0
            for row in item["features"][33:]:
                row["coverage"] = 0.1
                row["warning_count"] = 99
                row["drawdown"] = 0.99
        second = self.prepare(changed)
        self.assertEqual(first["cases"], second["cases"])
        self.assertEqual(first["train_episode_ids"], second["train_episode_ids"])
        self.assertEqual(first["validation_episode_ids"], second["validation_episode_ids"])

    def test_terminal_step_is_not_selected_as_an_interruptible_checkpoint(self):
        episodes = [episode(0, steps=33, candidate_step=33)]
        episodes += [episode(seed) for seed in range(1, 12)]
        protocol = self.prepare(episodes)
        self.assertNotIn(0, [case["episode_id"] for case in protocol["cases"]])

    def test_failed_visual_validation_blocks_checkpoint_selection(self):
        pilot.save_json(self.output / "visual-validation.json", {"passed": False})
        self.write_episodes([episode(seed) for seed in range(12)])
        with patch.object(pilot, "load_policy") as load:
            with self.assertRaisesRegex(AssertionError, "Visual validation failed"):
                pilot.prepare(self.output)
        load.assert_not_called()
        self.assertFalse((self.output / "protocol.json").exists())

    def test_too_few_independent_points_fail_before_label_runs(self):
        self.write_episodes([episode(seed) for seed in range(9)])
        with patch.object(pilot, "load_policy") as load:
            with self.assertRaisesRegex(AssertionError, "Too few eligible"):
                pilot.prepare(self.output)
        load.assert_not_called()

    def test_failed_near_target_validation_blocks_label_preparation(self):
        pilot.save_json(self.output / "near-target-validation.json", {"passed": False})
        self.write_episodes([episode(seed) for seed in range(12)])
        with self.assertRaisesRegex(AssertionError, "Near-target visual validation failed"):
            pilot.prepare(self.output)
        self.assertFalse((self.output / "protocol.json").exists())

    def test_resume_does_not_repeat_selection_or_modify_frozen_protocol(self):
        self.prepare()
        before = (self.output / "protocol.json").read_bytes()
        with patch.object(pilot, "eligibility") as selection, patch.object(pilot, "load_policy") as load:
            pilot.prepare(self.output)
        selection.assert_not_called()
        load.assert_not_called()
        self.assertEqual(before, (self.output / "protocol.json").read_bytes())

    def test_changed_original_evidence_rejected_before_gpu_loading(self):
        self.prepare()
        with patch.object(pilot, "source_fingerprints", return_value={"original": "changed"}), \
                patch.object(pilot, "load_policy") as load:
            with self.assertRaisesRegex(AssertionError, "Original evidence changed"):
                pilot.labels(self.output)
        load.assert_not_called()

    def test_changed_code_rejected_before_gpu_loading(self):
        self.prepare()
        with patch.object(pilot, "code_fingerprints", return_value={"runner": "changed"}), \
                patch.object(pilot, "load_policy") as load:
            with self.assertRaisesRegex(AssertionError, "Pilot source changed"):
                pilot.labels(self.output)
        load.assert_not_called()

    def test_changed_camera_evidence_or_config_rejected_on_resume(self):
        self.prepare()
        path = self.output / "visual-validation.json"
        before = path.read_bytes()
        pilot.save_json(path, {"passed": True, "changed": True})
        with self.assertRaisesRegex(AssertionError, "Visual evidence changed"):
            pilot.verify_protocol(self.output)
        path.write_bytes(before)
        pilot.save_json(self.model / "config.json", {"n_action_steps": 4})
        with self.assertRaises(AssertionError):
            pilot.verify_protocol(self.output)


if __name__ == "__main__":
    unittest.main()
