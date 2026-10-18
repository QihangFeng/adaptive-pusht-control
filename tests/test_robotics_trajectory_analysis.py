"""Artificial traces test frozen exploratory rules without GPU or simulator."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from robotics_trajectory_analysis import analyze_collection, analyze_trace


def trace(coverage, initial=0.5, chunk_length=8):
    rows = []
    before = initial
    for index, after in enumerate(coverage):
        rows.append({"step": index + 1, "chunk": index // chunk_length + 1,
                     "new_prediction": index % chunk_length == 0,
                     "pending_actions": chunk_length - index % chunk_length - 1,
                     "before": {"coverage": before}, "after": {"coverage": after},
                     "pair_contact_points": 1})
        before = after
    return rows


def episode(rows, seed=0, success=False):
    return {"seed": seed, "success": success, "steps": len(rows), "candidates": []}


class TrajectoryRulesTest(unittest.TestCase):
    def analyze(self, coverage, **kwargs):
        rows = trace(coverage, **kwargs)
        return analyze_trace(rows, episode(rows))

    def test_cached_drop_preserves_past_peak_and_contact(self):
        result = self.analyze([0.92, 0.88, 0.86, 0.85, 0.87])
        event = result["regression_events"][0]
        self.assertEqual(event["branch_step"], 3)
        self.assertEqual(event["peak_step"], 1)
        self.assertAlmostEqual(event["coverage_drop"], 0.06)
        self.assertEqual(event["pair_contact_points"], 1)
        self.assertEqual(len(result["regression_events"]), 1)

    def test_new_prediction_drop_is_not_cached_regression(self):
        rows = trace([0.92, 0.92, 0.85, 0.85, 0.85])
        rows[2]["new_prediction"] = True
        # Only the actual action boundary is tested here; repeated plan decline
        # below the same chunk peak may become a later cached-regression event.
        rows[3]["pending_actions"] = 0
        result = analyze_trace(rows, episode(rows))
        self.assertEqual(result["regression_events"], [])

    def test_pending_action_guard(self):
        rows = trace([0.92, 0.86, 0.86])
        rows[1]["pending_actions"] = 0
        self.assertEqual(analyze_trace(rows, episode(rows))["regression_events"], [])

    def test_previous_chunk_peak_does_not_leak(self):
        rows = trace([0.94] + [0.60] * 10)
        for row in rows[1:8]:
            row["pending_actions"] = 0
        self.assertEqual(analyze_trace(rows, episode(rows))["regression_events"], [])

    def test_future_peak_does_not_leak(self):
        self.assertEqual(self.analyze([0.5, 0.44, 0.93])["regression_events"], [])

    def test_chunk_start_state_included(self):
        rows = trace([0.93, 0.87, 0.87], initial=0.94)
        event = analyze_trace(rows, episode(rows))["regression_events"][0]
        self.assertEqual(event["peak_step"], 0)

    def test_final_step_is_not_recovery_branch(self):
        self.assertEqual(self.analyze([0.92, 0.86])["regression_events"], [])

    def test_low_progress_requires_24_actions_and_excludes_done(self):
        self.assertEqual(self.analyze([0.70] * 24)["low_progress_windows"], [])
        result = self.analyze([0.70] * 25, initial=0.70)
        self.assertEqual(len(result["low_progress_windows"]), 1)
        self.assertEqual(result["low_progress_windows"][0]["branch_step"], 24)

    def test_low_progress_uses_global_best_not_endpoint_difference(self):
        result = self.analyze([0.70] * 5 + [0.80] + [0.70] * 20, initial=0.70)
        self.assertEqual(result["low_progress_windows"], [])

    def test_low_progress_endpoint_guard(self):
        result = self.analyze([0.49] * 26, initial=0.49)
        self.assertEqual(result["low_progress_windows"], [])

    def test_low_progress_can_occur_before_eventual_success(self):
        rows = trace([0.70] * 26 + [0.96], initial=0.70)
        record = {"episode": episode(rows, success=True), "rows": rows}
        result = analyze_collection([record])
        self.assertEqual(result["groups"]["success"]["with_low_progress"], 1)
        self.assertEqual(result["groups"]["failure"]["attempts"], 0)
        self.assertEqual(len(result["episodes"][0]["low_progress_spans"]), 1)
        self.assertEqual(result["episodes"][0]["low_progress_spans"][0]["qualifying_windows"], 3)

    def test_nonexclusive_patterns_and_rank_order(self):
        rows = trace([0.93, 0.86] + [0.86] * 25, initial=0.93)
        summary = episode(rows)
        summary["candidates"] = [{"label": "contact_loss_candidate", "end_step": 3}]
        result = analyze_collection([{"episode": summary, "rows": rows}])
        group = result["groups"]["failure"]
        self.assertEqual(group["with_cached_regression"], 1)
        self.assertEqual(group["with_low_progress"], 1)
        self.assertEqual(group["with_contact_loss_candidate"], 1)
        self.assertEqual(result["ranked_regression_events"][0]["branch_step"], 2)

    def test_duplicate_seeds_rejected(self):
        rows = trace([0.6])
        record = {"episode": episode(rows), "rows": rows}
        with self.assertRaises(ValueError):
            analyze_collection([record, copy.deepcopy(record)])

    def test_incomplete_or_invalid_trace_rejected(self):
        rows = trace([0.6, 0.7])
        bad = episode(rows)
        bad["steps"] = 3
        with self.assertRaises(ValueError):
            analyze_trace(rows, bad)
        rows[0]["after"]["coverage"] = float("nan")
        with self.assertRaises(ValueError):
            analyze_trace(rows, episode(rows))


if __name__ == "__main__":
    unittest.main()
