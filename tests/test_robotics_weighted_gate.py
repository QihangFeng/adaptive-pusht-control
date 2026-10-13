"""Pure CPU checks of camera-only gate math and bounded development selection."""

from collections import deque
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_weighted_gate as gate


def features(**changes):
    row = {"quality": True, "coverage": .8, "drawdown": .1, "warning_count": 2}
    row.update(changes)
    return row


def weighted(**changes):
    config = {"family": "weighted", "weights": [.5, .25, .25], "threshold": .5}
    config.update(changes)
    return config


def threshold(**changes):
    config = {"family": "threshold", "min_warnings": 2, "min_drawdown": .1}
    config.update(changes)
    return config


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="weighted-gate-cpu-", dir=parent)


class CameraSignalsTest(unittest.TestCase):
    def test_frozen_normalization_and_signed_two_movement_recovery(self):
        self.assertEqual(gate.NORMALIZATION, {"D_scale": .20, "W_scale": 5, "R_scale": .10, "R_movements": 2})
        actual = gate.signals(features(), [.85, .81, .8])
        self.assertTrue(actual["quality"])
        self.assertAlmostEqual(actual["D"], .5)
        self.assertAlmostEqual(actual["W"], .4)
        self.assertAlmostEqual(actual["R"], -.5)

    def test_clipping_does_not_turn_signed_recovery_into_absolute_drop(self):
        self.assertEqual(gate.signals(features(drawdown=.30, warning_count=5), [.99, .90, .8]),
                         {"quality": True, "D": 1., "W": 1., "R": -1.})
        self.assertEqual(gate.signals(features(drawdown=0., warning_count=0), [.50, .70, .8]),
                         {"quality": True, "D": 0., "W": 0., "R": 1.})

    def test_only_final_three_values_are_used_and_middle_drop_is_not_counted_twice(self):
        self.assertEqual(gate.signals(features(), [.8, .5, .8])["R"], 0.)
        expected = gate.signals(features(), [.85, .81, .8])
        self.assertEqual(gate.signals(features(), deque([None, .0, .85, .81, .8])), expected)

    def test_missing_nonfinite_invalid_or_nonmatching_camera_history_is_rejected(self):
        histories = (None, [], [.8], [.85, .8], [.85, None, .8], [.85, float("nan"), .8],
                     [.85, float("inf"), .8], [.85, -.1, .8], [.85, 1.1, .8],
                     [.85, True, .8], [.85, .81, .800000000000001])
        for history in histories:
            with self.subTest(history=history):
                self.assertEqual(gate.signals(features(), history),
                                 {"quality": False, "D": None, "W": None, "R": None})

    def test_invalid_current_features_never_allow_either_gate(self):
        for changed in ({"quality": False}, {"coverage": float("nan")}, {"coverage": True},
                        {"drawdown": -.1}, {"drawdown": float("inf")},
                        {"warning_count": 2.}, {"warning_count": True}, {"warning_count": 6}):
            for config in (weighted(), threshold()):
                with self.subTest(changed=changed, config=config):
                    result = gate.gate_decision(config, features(**changed), [.85, .81, .8])
                    self.assertFalse(result["allowed"])
                    self.assertIsNone(result["score"])

    def test_truth_contact_old_recovery_and_future_outcomes_are_not_runtime_inputs(self):
        original = features()
        altered = {**original, "true_coverage": 0., "contact_loss": True,
                   "success": False, "recovery": 100., "accumulated_drop_5": 99.}
        for config in (weighted(), threshold()):
            self.assertEqual(gate.gate_decision(config, original, [.85, .81, .8]),
                             gate.gate_decision(config, altered, [.85, .81, .8]))


class ConfigurationAndDecisionTest(unittest.TestCase):
    def test_weighted_score_uses_minus_signed_recovery_and_inclusive_threshold(self):
        config = weighted(weights=[0., 0., 1.], threshold=.5)
        result = gate.gate_decision(config, features(), [.9, .85, .8])
        self.assertAlmostEqual(result["score"], 1.)
        self.assertTrue(result["allowed"])
        self.assertFalse(gate.gate_decision(config, features(), [.7, .75, .8])["allowed"])
        inclusive = weighted(weights=[1., 0., 0.], threshold=.5)
        self.assertTrue(gate.gate_decision(inclusive, features(), [.85, .81, .8])["allowed"])
        self.assertFalse(gate.gate_decision({**inclusive, "threshold": .500001}, features(), [.85, .81, .8])["allowed"])

    def test_threshold_gate_requires_both_inclusive_conditions(self):
        self.assertTrue(gate.gate_decision(threshold(), features(), [.85, .81, .8])["allowed"])
        self.assertFalse(gate.gate_decision(threshold(min_warnings=3), features(), [.85, .81, .8])["allowed"])
        self.assertFalse(gate.gate_decision(threshold(min_drawdown=.100001), features(), [.85, .81, .8])["allowed"])
        self.assertIsNone(gate.gate_decision(threshold(), features(), [.85, .81, .8])["score"])

    def test_validate_config_rejects_unknown_missing_boolean_and_nonfinite_values(self):
        invalid = (None, [], {"family": "learned"}, {"family": "weighted"},
                   {**weighted(), "extra": True}, weighted(weights=[.5, .5]),
                   weighted(weights=[-.1, .6, .5]), weighted(weights=[.1, .2, .3]),
                   weighted(weights=[True, 0., 0.]), weighted(weights=[float("nan"), .5, .5]),
                   weighted(threshold=True), weighted(threshold=-.1), weighted(threshold=float("inf")),
                   {**threshold(), "extra": 0}, threshold(min_warnings=2.),
                   threshold(min_warnings=True), threshold(min_warnings=6), threshold(min_drawdown=float("nan")))
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                gate.validate_config(config)

    def test_canonical_config_is_independent_and_inputs_are_not_changed(self):
        config, row, history = weighted(), features(), [.85, .81, .8]
        before = copy.deepcopy((config, row, history))
        canonical = gate.validate_config(config)
        gate.gate_decision(config, row, history)
        self.assertEqual((config, row, history), before)
        canonical["weights"][0] = 0.
        self.assertEqual(config["weights"][0], .5)

    def test_exact_candidate_grid_order_and_copy_independence(self):
        candidates = gate.candidate_configs()
        self.assertEqual(len(candidates), 18)
        self.assertEqual(sum(c["family"] == "weighted" for c in candidates), 12)
        self.assertEqual(sum(c["family"] == "threshold" for c in candidates), 6)
        self.assertEqual([c["threshold"] for c in candidates[:3]], [.50, .35, .65])
        self.assertEqual(candidates[12], threshold(min_warnings=3, min_drawdown=.15))
        self.assertEqual(candidates[-1], threshold(min_warnings=2, min_drawdown=.20))
        self.assertTrue(all(gate.validate_config(c) == c for c in candidates))
        candidates[0]["weights"][0] = 0.
        self.assertEqual(gate.candidate_configs()[0]["weights"][0], .5)


class SavedCameraHistoryTest(unittest.TestCase):
    def fixture(self, directory, rows, checkpoint=4):
        path = Path(directory) / "camera.trace.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"steps": rows}, stream, allow_nan=False)
        return {"source_trace": str(path), "checkpoint": checkpoint}

    def rows(self):
        return [{"step": step, "features": {"quality": True, "coverage": coverage},
                 "after": {"coverage": 0., "contact": False}}
                for step, coverage in enumerate([.99, .9, .85, .8, .1], start=1)]

    def test_extract_reads_only_three_camera_rows_and_not_future_or_truth(self):
        with test_directory() as directory:
            rows = self.rows()
            case = self.fixture(directory, rows)
            self.assertEqual(gate.extract_case_history(case), [.9, .85, .8])
            rows[4]["features"]["coverage"] = 1.
            for row in rows:
                row["after"]["coverage"] = 1.
            self.fixture(directory, rows)
            self.assertEqual(gate.extract_case_history(case), [.9, .85, .8])

    def test_invalid_camera_row_is_missing_and_bad_checkpoint_indices_are_rejected(self):
        with test_directory() as directory:
            rows = self.rows()
            rows[2]["features"]["quality"] = False
            case = self.fixture(directory, rows)
            self.assertEqual(gate.extract_case_history(case), [.9, None, .8])
            for checkpoint in (True, 2, 6, 4.):
                with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                    gate.extract_case_history({**case, "checkpoint": checkpoint})
            rows[2]["step"] = 99
            self.fixture(directory, rows)
            with self.assertRaises(ValueError):
                gate.extract_case_history(case)


class DevelopmentSearchTest(unittest.TestCase):
    def data(self):
        cases = [{"scene_seed": 1, "checkpoint": 10, "features": features()},
                 {"scene_seed": 2, "checkpoint": 20, "features": features(drawdown=.04, warning_count=3)}]
        histories = {1: [.85, .81, .8], 2: [.8, .8, .8]}
        records = [{"scene_seed": case["scene_seed"], "checkpoint": case["checkpoint"],
                    "prediction_seed": 9100, "arm": arm, "success": True,
                    "model_queries": 2, "final_coverage": .9}
                   for case in cases for arm in ("original", "defer_current_unit")]
        candidates = [weighted(weights=[1., 0., 0.], threshold=.5),
                      weighted(weights=[0., 1., 0.], threshold=.5),
                      threshold(min_warnings=2, min_drawdown=.1),
                      threshold(min_warnings=3, min_drawdown=0.)]
        return cases, records, histories, candidates

    def search(self, cases, records, histories, candidates):
        with patch.object(gate, "candidate_configs", return_value=candidates):
            return gate.development_search(cases, records, histories)

    def test_success_calls_alignment_and_index_lexicographic_order(self):
        for criterion, expected in (("success", 0), ("calls", 1), ("coverage", 1), ("index", 0)):
            with self.subTest(criterion=criterion):
                cases, records, histories, candidates = self.data()
                for row in records:
                    in_zero = (row["scene_seed"] == 1) == (row["arm"] == "original")
                    if criterion == "success":
                        row.update(success=in_zero, model_queries=100 if in_zero else 0,
                                   final_coverage=.5 if in_zero else 1.)
                    elif criterion == "calls":
                        row["model_queries"] = 5 if in_zero else 2
                    elif criterion == "coverage":
                        row["final_coverage"] = .7 if in_zero else .9
                result = self.search(cases, records, histories, candidates)
                self.assertEqual(result["selection_details"]["weighted"]["selected_index"], expected)
                self.assertEqual(result["selection_details"]["threshold"]["selected_index"], expected + 2)
                self.assertEqual(result["search_results"][0]["decision_mask"], [True, False])
                self.assertEqual(result["search_results"][1]["decision_mask"], [False, True])

    def test_ties_masks_and_proxy_limits_are_explicit_not_classifier_accuracy(self):
        cases, records, histories, _ = self.data()
        result = gate.development_search(cases, records, histories)
        self.assertEqual(len(result["search_results"]), 18)
        self.assertEqual(result["paired_futures"], 2)
        self.assertEqual(result["selection_details"]["weighted"]["selected_index"], 0)
        self.assertEqual(result["selection_details"]["weighted"]["tied_best_indices"], list(range(12)))
        self.assertEqual(result["selection_details"]["threshold"]["tied_best_indices"], list(range(12, 18)))
        self.assertGreater(result["selection_details"]["weighted"]["selected_mask_equivalence_count"], 0)
        self.assertTrue(result["limits"])
        self.assertNotIn("accuracy", result)
        self.assertEqual(result["normalization"], gate.NORMALIZATION)

    def test_record_order_and_original_source_outcomes_do_not_change_selection(self):
        cases, records, histories, candidates = self.data()
        original = copy.deepcopy((cases, records, histories, candidates))
        expected = self.search(cases, records, histories, candidates)
        altered_cases = [{**case, "source_success": False, "true_coverage": 0.} for case in cases]
        self.assertEqual(self.search(altered_cases, list(reversed(records)), histories, candidates), expected)
        self.assertEqual((cases, records, histories, candidates), original)

    def test_missing_duplicate_unpaired_or_invalid_records_are_rejected(self):
        cases, records, histories, candidates = self.data()
        invalid_panels = [records[:-1], records + [records[0]],
                          [{**r, "success": 1} if i == 0 else r for i, r in enumerate(records)],
                          [{**r, "checkpoint": 99} if i == 0 else r for i, r in enumerate(records)],
                          [{**r, "model_queries": True} if i == 0 else r for i, r in enumerate(records)],
                          [{**r, "final_coverage": float("nan")} if i == 0 else r for i, r in enumerate(records)]]
        for panel in invalid_panels:
            with self.assertRaises(ValueError):
                self.search(cases, panel, histories, candidates)
        with self.assertRaises(ValueError):
            self.search(cases + [cases[0]], records, histories, candidates)


if __name__ == "__main__":
    unittest.main()
