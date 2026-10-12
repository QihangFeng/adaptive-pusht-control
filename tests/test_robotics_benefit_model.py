"""The benefit model, matched labels and grouping require no GPU or simulator."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from robotics_benefit_model import (aggregate_pairs, choose_threshold, fit_tree,
                                    grouped_split, predict_tree, should_replan)


def record(case="a", episode=0, sample=100, arm="continue", success=False,
           features=None, queries=10):
    return {"checkpoint_id": case, "episode_id": episode, "sample_seed": sample,
            "arm": arm, "success": success, "model_queries": queries,
            "features": [0.0, 1.0] if features is None else features}


def case(case_id, episode, features, continue_success, replan_success,
         continue_queries=10, replan_queries=11):
    return aggregate_pairs([
        record(case_id, episode, index, arm, success, features, queries)
        for index, (old, new) in enumerate(zip(continue_success, replan_success))
        for arm, success, queries in [("continue", old, continue_queries),
                                     ("replan", new, replan_queries)]
    ], expected_repeats=len(continue_success))[0]


class BenefitTreeTest(unittest.TestCase):
    def test_piecewise_fit_and_json_round_trip(self):
        X = np.arange(12, dtype=float)[:, None]
        y = np.repeat([-1.0, -0.5, 0.5, 1.0], 3)
        tree = fit_tree(X, y)
        loaded = json.loads(json.dumps(tree, allow_nan=False))
        predictions = [predict_tree(loaded, row) for row in X]
        self.assertEqual(predictions, y.tolist())

        def check(node, depth=0):
            self.assertLessEqual(depth, 2)
            if "feature" not in node:
                self.assertGreaterEqual(node["n_samples"], 3)
            else:
                check(node["left"], depth + 1)
                check(node["right"], depth + 1)
        check(loaded["root"])

    def test_tied_features_choose_first_and_fit_is_deterministic(self):
        X = [[0, 0]] * 3 + [[1, 1]] * 3
        y = [-1] * 3 + [1] * 3
        tree = fit_tree(X, y)
        self.assertEqual(tree["root"]["feature"], 0)
        self.assertEqual(tree, fit_tree(X, y))

    def test_constant_features_and_small_dataset_use_mean(self):
        tree = fit_tree([[1], [1]], [-1, 1])
        self.assertNotIn("feature", tree["root"])
        self.assertEqual(predict_tree(tree, [1]), 0)
        shallow = fit_tree([[0], [1]], [-1, 1], max_depth=0, min_leaf=1)
        self.assertEqual(predict_tree(shallow, [0]), 0)

    def test_minimum_leaf_prevents_overfit_singleton(self):
        tree = fit_tree([[0], [0], [0], [0], [1]], [0, 0, 0, 0, 1])
        self.assertNotIn("feature", tree["root"])
        self.assertAlmostEqual(predict_tree(tree, [1]), 0.2)

    def test_invalid_matrix_label_and_hyperparameters(self):
        for X, y in [([], []), ([1, 2], [1, 2]), ([[0]], []), ([[float("nan")]], [0]),
                     ([[0]], [float("inf")]), ([[]], [0])]:
            with self.assertRaises(ValueError):
                fit_tree(X, y)
        for kwargs in [{"max_depth": -1}, {"max_depth": True}, {"min_leaf": 0},
                       {"min_leaf": 1.5}]:
            with self.assertRaises(ValueError):
                fit_tree([[0]], [0], **kwargs)

    def test_predict_rejects_bad_dimensions_and_nonfinite_input(self):
        tree = fit_tree([[0]], [0])
        for values in [[], [1, 2], [float("nan")], [[0]]]:
            with self.assertRaises(ValueError):
                predict_tree(tree, values)
        bad = copy.deepcopy(tree)
        bad["format_version"] = 99
        with self.assertRaises(ValueError):
            predict_tree(bad, [0])


class MatchedLabelTest(unittest.TestCase):
    def test_pairing_uses_seed_not_order_and_means_full_success_difference(self):
        rows = [record(sample=2, arm="replan", success=True),
                record(sample=1, arm="continue", success=True),
                record(sample=2, arm="continue", success=False),
                record(sample=1, arm="replan", success=False),
                record(sample=3, arm="replan", success=True),
                record(sample=3, arm="continue", success=False)]
        result = aggregate_pairs(rows, expected_repeats=3)[0]
        self.assertEqual([pair["sample_seed"] for pair in result["pairs"]], [1, 2, 3])
        self.assertAlmostEqual(result["benefit"], 1 / 3)
        self.assertEqual(rows[0]["arm"], "replan")
        json.dumps(result, allow_nan=False)

    def test_duplicate_unmatched_and_wrong_repeat_records_rejected(self):
        old, new = record(), record(arm="replan")
        for rows in [[old], [old, new, old]]:
            with self.assertRaises(ValueError):
                aggregate_pairs(rows)
        with self.assertRaises(ValueError):
            aggregate_pairs([old, new], expected_repeats=3)

    def test_inconsistent_checkpoint_context_rejected(self):
        for new in [record(arm="replan", episode=1),
                    record(arm="replan", features=[0, 2])]:
            with self.assertRaises(ValueError):
                aggregate_pairs([record(), new])

    def test_invalid_record_values_rejected(self):
        for key, value in [("sample_seed", True), ("arm", "unknown"),
                           ("model_queries", -1), ("model_queries", 1.2),
                           ("success", 2), ("features", [float("nan")])]:
            row = record()
            row[key] = value
            with self.assertRaises(ValueError):
                aggregate_pairs([row])

    def test_all_checkpoints_from_episode_share_split(self):
        cases = [case("a", 0, [0], [False], [True]),
                 case("b", 0, [1], [False], [True]),
                 case("c", 1, [2], [True], [False])]
        train, validation = grouped_split(cases, [0], [1])
        self.assertEqual([item["checkpoint_id"] for item in train], ["a", "b"])
        self.assertEqual([item["checkpoint_id"] for item in validation], ["c"])
        for train_ids, val_ids in [([0, 1], [1]), ([0], [2]), ([0, 1], [])]:
            with self.assertRaises(ValueError):
                grouped_split(cases, train_ids, val_ids)
        with self.assertRaises(ValueError):
            grouped_split(cases + [cases[0]], [0], [1])


class ValidationThresholdTest(unittest.TestCase):
    def test_selects_observed_validation_gain_not_predicted_gain(self):
        tree = fit_tree([[0]] * 3 + [[1]] * 3, [0.2] * 3 + [0.8] * 3)
        validation = [case("harm", 10, [0], [True] * 3, [False] * 3),
                      case("help", 11, [1], [False] * 3, [True] * 3)]
        result = choose_threshold(tree, validation)
        self.assertEqual(result["threshold"], 0.67)
        self.assertEqual(result["selected"]["decisions"], [False, True])
        self.assertAlmostEqual(result["selected"]["mean_success_gain"], 0.5)
        self.assertEqual(result["selected"]["interventions"], 1)

    def test_no_validation_benefit_defaults_to_no_intervention(self):
        tree = fit_tree([[0]] * 3, [1 / 3] * 3)
        validation = [case("neutral", 10, [0], [True], [True])]
        result = choose_threshold(tree, validation)
        self.assertEqual(result["threshold"], 1.1)
        self.assertEqual(result["selected"]["interventions"], 0)

    def test_success_tie_can_prefer_fewer_queries(self):
        tree = fit_tree([[0]] * 3, [1 / 3] * 3)
        validation = [case("cheaper", 10, [0], [True], [True],
                           continue_queries=10, replan_queries=9)]
        result = choose_threshold(tree, validation)
        self.assertEqual(result["threshold"], 0.1)
        self.assertEqual(result["selected"]["mean_model_queries"], 9)

    def test_small_or_negative_predictions_do_not_interrupt(self):
        for prediction in [-1, 0, 0.049, 0.05]:
            tree = fit_tree([[0]], [prediction])
            self.assertFalse(should_replan(tree, [0], threshold=0))
        tree = fit_tree([[0]], [0.051])
        self.assertTrue(should_replan(tree, [0], threshold=0))

    def test_empty_validation_bad_candidates_and_margin_rejected(self):
        tree = fit_tree([[0]], [1])
        validation = [case("x", 0, [0], [False], [True])]
        for cases, thresholds, margin in [([], [0], 0.05), (validation, [], 0.05),
                                           (validation, [float("nan")], 0.05),
                                           (validation, [0], 0)]:
            with self.assertRaises(ValueError):
                choose_threshold(tree, cases, thresholds, margin)


if __name__ == "__main__":
    unittest.main()
