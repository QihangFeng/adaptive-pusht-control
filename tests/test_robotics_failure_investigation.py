"""CPU-only collection guards; all model loading and episode runs are mocked."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_failure_investigation as investigation


class RecoveryHelpersTest(unittest.TestCase):
    @staticmethod
    def episode(seed, success, events):
        return {"seed": seed, "success": success, "regression_events": events}

    def test_selection_uses_seed_order_and_first_event_not_largest_drop(self):
        episodes = [self.episode(seed, False, [
            {"branch_step": 180, "drop": 0.3}, {"branch_step": 40, "drop": 0.02},
        ]) for seed in [9, 8, 2, 5]]
        episodes += [self.episode(seed, True, [{"branch_step": 70}]) for seed in [6, 1]]
        selected = investigation.select_recovery_cases({"episodes": episodes})
        self.assertEqual([case["seed"] for case in selected], [2, 5, 8, 1])
        self.assertEqual([case["event"]["branch_step"] for case in selected[:3]], [40] * 3)
        self.assertTrue(selected[-1]["baseline_success"])

    def test_one_failed_scene_is_not_a_recurring_pattern_even_with_success_control(self):
        episodes = [self.episode(0, False, [{"branch_step": 100}]),
                    self.episode(1, True, [{"branch_step": 100}])]
        self.assertEqual(investigation.select_recovery_cases({"episodes": episodes}), [])

    def test_selection_requires_at_least_sixteen_actions_remaining(self):
        episodes = [self.episode(0, False, [{"branch_step": 285}, {"branch_step": 284}]),
                    self.episode(1, False, [{"branch_step": 285}]),
                    self.episode(2, False, [{"branch_step": 283}])]
        selected = investigation.select_recovery_cases({"episodes": episodes})
        self.assertEqual([case["seed"] for case in selected], [0, 2])
        self.assertEqual(selected[0]["event"]["branch_step"], 284)

    def test_cached_remainder_keeps_only_the_same_pending_chunk(self):
        rows = [
            {"chunk": 1, "pending_actions": 2, "new_prediction": True},
            {"chunk": 1, "pending_actions": 1, "new_prediction": False, "action": [1, 2]},
            {"chunk": 1, "pending_actions": 0, "new_prediction": False, "action": [3, 4]},
            {"chunk": 2, "pending_actions": 7, "new_prediction": True},
        ]
        self.assertEqual(investigation.cached_remainder(rows, 1), rows[1:3])
        for field, value in [("new_prediction", True), ("chunk", 2)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(rows)
                changed[2][field] = value
                with self.assertRaisesRegex(AssertionError, "prediction boundary"):
                    investigation.cached_remainder(changed, 1)

    def test_cached_remainder_allows_success_before_pending_queue_is_exhausted(self):
        rows = [
            {"chunk": 1, "pending_actions": 3, "new_prediction": True},
            {"chunk": 1, "pending_actions": 2, "new_prediction": False, "terminated": True},
        ]
        self.assertEqual(investigation.cached_remainder(rows, 1), [rows[1]])


class RecoveryResumeGuardsTest(unittest.TestCase):
    def assert_saved_pilot_rejected(self, trials, message=None):
        analysis = {"episodes": [
            {"seed": seed, "success": False, "regression_events": [{"branch_step": 100}]}
            for seed in range(2)
        ]}
        result = {
            "complete": True, "cases": investigation.select_recovery_cases(analysis),
            "fresh_diffusion_seeds": investigation.RECOVERY_SEEDS, "trials": trials,
        }
        records = [{"episode": {"seed": seed}} for seed in investigation.SEEDS]
        documents = {"protocol.json": {"original_hashes": {}, "config": {}},
                     "config.json": {}, "analysis.json": analysis, "recovery.json": result}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "recovery.json").touch()
            with patch.object(investigation, "original_evidence", return_value=(None, None, {})), \
                    patch.object(investigation, "load_records", return_value=records), \
                    patch.object(investigation, "analyze_collection", return_value=analysis), \
                    patch.object(investigation, "read_json", side_effect=lambda path: documents[path.name]), \
                    patch.object(investigation, "load_policy") as load, \
                    patch.object(investigation, "continuation") as continuation:
                rejection = (self.assertRaisesRegex(AssertionError, message) if message
                             else self.assertRaises(AssertionError))
                with rejection:
                    investigation.recover(output, output / "model")
            load.assert_not_called()
            continuation.assert_not_called()

    def test_complete_pilot_missing_expected_trial_is_rejected_before_model_load(self):
        self.assert_saved_pilot_rejected([], "Completed pilot lacks expected trials")

    def test_saved_pilot_with_missing_trace_is_rejected_before_model_load(self):
        self.assert_saved_pilot_rejected([{
            "seed": 0, "diffusion_seed": 2000, "arm": "continue_cached",
            "branch_step": 100, "trace": "missing.trace.json.gz",
        }])


class CollectionGuardsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.original = self.directory / "original"
        self.original.mkdir()
        self.output = self.directory / "collection"
        self.model = self.directory / "model"
        self.model.mkdir()
        self.config = {"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8}
        episodes = [{"seed": seed, "trace": f"seed-{seed}.trace.json.gz"}
                    for seed in range(6)]
        investigation.write_json(self.original / "summary.json",
                                 {"complete": True, "episodes": episodes})
        investigation.write_json(self.original / "protocol.json", {
            "seeds": list(range(6)), "config": self.config,
            "weights_sha256": investigation.WEIGHTS_SHA256,
        })
        investigation.write_json(self.model / "config.json", self.config)
        for episode in episodes:
            # The collection guard hashes bytes; it does not parse these traces.
            (self.original / episode["trace"]).write_bytes(b"existing evidence")
        original_patch = patch.object(investigation, "ORIGINAL", self.original)
        version_patch = patch.object(investigation.importlib.metadata, "version", return_value="test")
        original_patch.start()
        version_patch.start()
        self.addCleanup(original_patch.stop)
        self.addCleanup(version_patch.stop)

    def save_collection(self, protocol=None, seeds=None):
        self.output.mkdir()
        if protocol is None:
            hashes = investigation.original_evidence()[2]
            protocol = investigation.experiment_protocol(self.config, hashes)
        if seeds is None:
            seeds = investigation.SEEDS
        investigation.write_json(self.output / "protocol.json", protocol)
        investigation.write_json(self.output / "summary.json", {
            "baseline_complete": len(seeds) == 30,
            "episodes": [{"seed": seed} for seed in seeds],
        })

    def test_original_hashes_are_stable_and_detect_changed_evidence(self):
        first = investigation.original_evidence()[2]
        self.assertEqual(first, investigation.original_evidence()[2])
        (self.original / "seed-2.trace.json.gz").write_bytes(b"changed evidence")
        changed = investigation.original_evidence()[2]
        self.assertNotEqual(first["seed-2.trace.json.gz"], changed["seed-2.trace.json.gz"])
        self.assertEqual(first["protocol.json"], changed["protocol.json"])

    def test_protocol_keeps_original_baseline_and_predetermined_seeds(self):
        hashes = investigation.original_evidence()[2]
        protocol = investigation.experiment_protocol(copy.deepcopy(self.config), hashes)
        self.assertEqual(protocol["seeds"], list(range(30)))
        self.assertEqual(protocol["reused_seeds"], list(range(6)))
        self.assertEqual(protocol["new_seeds"], list(range(6, 30)))
        self.assertEqual(protocol["config"], self.config)
        self.assertEqual(protocol["max_episode_steps"], 300)
        self.assertEqual(protocol["original_hashes"], hashes)
        self.assertTrue(protocol["diagnostic_state_not_policy_input"])

    def test_resume_refuses_changed_protocol_before_model_load(self):
        hashes = investigation.original_evidence()[2]
        protocol = investigation.experiment_protocol(self.config, hashes)
        protocol["max_episode_steps"] = 301
        self.save_collection(protocol, seeds=list(range(6)))
        with patch.object(investigation, "load_policy") as load, \
                patch.object(investigation, "run_episode") as run:
            with self.assertRaisesRegex(AssertionError, "Protocol or original evidence changed"):
                investigation.collect(self.output, self.model, resume=True)
        load.assert_not_called()
        run.assert_not_called()

    def test_completed_collection_is_not_rerun_or_overwritten(self):
        self.save_collection()
        before = {name: investigation.digest(self.output / name)
                  for name in ["protocol.json", "summary.json"]}
        with patch.object(investigation, "load_policy") as load, \
                patch.object(investigation, "run_episode") as run:
            investigation.collect(self.output, self.model, resume=True)
            with self.assertRaisesRegex(AssertionError, "Existing results require --resume"):
                investigation.collect(self.output, self.model, resume=False)
        load.assert_not_called()
        run.assert_not_called()
        self.assertEqual(before, {name: investigation.digest(self.output / name) for name in before})


if __name__ == "__main__":
    unittest.main()
