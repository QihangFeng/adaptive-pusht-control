"""CPU-only protocol contract tests; no model construction or GPU operations."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PATH = Path(__file__).parents[1] / "experiments/robotics_recovery_pilot.py"
SPEC = importlib.util.spec_from_file_location("recovery_pilot_contract", PATH)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class ProtocolTests(unittest.TestCase):
    def test_budget_and_fixed_final_checkpoint(self):
        self.assertEqual(pilot.TRAINING["optimizer_steps"], 600)
        self.assertEqual(pilot.TRAINING["accumulation"], 4)
        self.assertEqual(pilot.TRAINING["selection"], "final_600_only")
        self.assertEqual(pilot.TRAINING["validation_steps"], [0, 200, 400, 600])

    def test_three_paired_arms_and_new_scenes(self):
        config = pilot.EVALUATION
        self.assertEqual(config["arms"], ["fixed8", "milestone12", "recovery12"])
        self.assertEqual(config["scenes"], list(range(150, 170)))
        self.assertEqual(len(config["arms"]) * len(config["scenes"]), 60)
        self.assertEqual(config["max_steps"], 300)

    def test_prepare_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expected = {"training": {"optimizer_steps": 600}}
            with patch.object(pilot, "check_output"), patch.object(pilot, "protocol_document", return_value=expected):
                self.assertEqual(pilot.prepare(output), expected)
                original = (output / "protocol.json").read_bytes()
                self.assertEqual(pilot.prepare(output), expected)
                self.assertEqual((output / "protocol.json").read_bytes(), original)
                (output / "protocol.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
                with self.assertRaises(AssertionError):
                    pilot.verify(output)

    def test_existing_results_block_first_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "training.json").touch()
            with patch.object(pilot, "check_output"), self.assertRaises(AssertionError):
                pilot.prepare(output)

    def test_write_once_refuses_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "value.json"
            pilot.write_once(target, {"a": 1})
            pilot.write_once(target, {"a": 1})
            with self.assertRaises(AssertionError):
                pilot.write_once(target, {"a": 2})


if __name__ == "__main__":
    unittest.main()
