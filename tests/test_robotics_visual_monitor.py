"""CPU-only feature tests; no policy model, GPU, or image truth required."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from robotics_visual_monitor import (
    FEATURE_NAMES, GOAL_POSE, RECTANGLES, RollingVisualFeatures,
    VisualAlignmentEstimator, _rotation,
    _near_target_errors,
)


def estimate(coverage, quality=True):
    return {"quality": quality, "coverage": coverage,
            "position_error": .1, "orientation_error": .2}


class RollingFeaturesTest(unittest.TestCase):
    def test_warning_count_and_net_drawdown(self):
        monitor = RollingVisualFeatures()
        for coverage in [.90, .88, .86, .87]:
            features = monitor.update(estimate(coverage), 3)
        self.assertEqual(features["warning_count"], 2)
        self.assertAlmostEqual(features["drawdown"], .03)
        self.assertAlmostEqual(features["recovery"], -.03)
        self.assertEqual(features["pending_actions"], 3)

    def test_recovery_does_not_accumulate_all_decreases(self):
        monitor = RollingVisualFeatures()
        for coverage in [.8, .7, .8, .7, .85]:
            features = monitor.update(estimate(coverage), 2)
        self.assertEqual(features["drawdown"], 0)
        self.assertEqual(features["warning_count"], 2)
        self.assertAlmostEqual(features["stagnation"], .05)

    def test_missing_frame_clears_history_and_is_no_decision(self):
        monitor = RollingVisualFeatures()
        monitor.update(estimate(.9), 5)
        result = monitor.update(estimate(None, False), 4)
        self.assertFalse(result["quality"])
        self.assertIsNone(result["warning_count"])
        self.assertEqual(monitor.update(estimate(.6), 3)["drawdown"], 0)

    def test_past_only_same_prefix(self):
        left, right = RollingVisualFeatures(), RollingVisualFeatures()
        for value in [.7, .75, .8, .78]:
            self.assertEqual(left.update(estimate(value), 3), right.update(estimate(value), 3))

    def test_fixed_feature_order_is_complete(self):
        row = RollingVisualFeatures().update(estimate(.8), 5)
        self.assertTrue(all(name in row for name in FEATURE_NAMES))

    def test_old_warnings_and_peak_expire(self):
        monitor = RollingVisualFeatures()
        for coverage in [.9, .8] + [.8] * 25:
            features = monitor.update(estimate(coverage), 2)
        self.assertEqual(features["warning_count"], 0)
        self.assertEqual(features["drawdown"], 0)
        self.assertEqual(features["recent_best_gain"], 0)
        self.assertEqual(features["history_steps"], 24)

    def test_rejects_negative_pending_count(self):
        with self.assertRaises(ValueError):
            RollingVisualFeatures().update(estimate(.8), -1)


class ImageEstimatorTest(unittest.TestCase):
    def test_blank_image_abstains(self):
        result = VisualAlignmentEstimator().estimate(np.full((96, 96, 3), 255, np.uint8))
        self.assertFalse(result["quality"])
        self.assertIsNone(result["coverage"])

    def test_rejects_state_or_visualization_instead_of_policy_pixels(self):
        with self.assertRaises(ValueError):
            VisualAlignmentEstimator().estimate(np.zeros((680, 680, 3), np.uint8))

    def test_known_target_shape_is_detected_without_simulator(self):
        image = np.full((96, 96, 3), 255, np.uint8)
        for rectangle in RECTANGLES:
            points = rectangle @ _rotation(GOAL_POSE[2]).T + np.asarray(GOAL_POSE[:2])
            cv2.fillPoly(image, [np.rint(points).astype(np.int32)], (143, 163, 184))
        result = VisualAlignmentEstimator().estimate(image)
        self.assertTrue(result["quality"])
        self.assertGreater(result["coverage"], .9)


class NearTargetAuditTest(unittest.TestCase):
    def test_stratification_and_consecutive_pair_requirement(self):
        rows = [{"step": i + 1, "after": {"coverage": value}}
                for i, value in enumerate([.5, .8, .85, .6, .9])]
        features = [{"step": i + 1, "quality": True, "coverage": value}
                    for i, value in enumerate([.5, .81, .87, .6, .9])]
        errors, changes, eligible = _near_target_errors(rows, features)
        self.assertEqual(eligible, 3)
        np.testing.assert_allclose(errors, [.01, .02, 0])
        np.testing.assert_allclose(changes, [.01])

    def test_invalid_observation_is_not_bridged(self):
        rows = [{"step": i + 1, "after": {"coverage": .9}} for i in range(3)]
        features = [{"step": 1, "quality": True, "coverage": .89},
                    {"step": 2, "quality": False, "coverage": None},
                    {"step": 3, "quality": True, "coverage": .91}]
        errors, changes, eligible = _near_target_errors(rows, features)
        self.assertEqual(eligible, 3)
        self.assertEqual(len(errors), 2)
        self.assertEqual(changes, [])


if __name__ == "__main__":
    unittest.main()
