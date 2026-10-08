"""Standard-library tests: no simulator, GPU dataset, or extra packages needed."""

import copy
from pathlib import Path
import unittest

from experiments import delay_acceptability_audit as audit


def example_metrics():
    return dict(integrated_violation_pu_s=0.05, violation_time_s=2,
                worst_vmin=0.945, worst_vmax=1.04, throughput_tps=100,
                itl_violation_fraction=0, batch_switches=3,
                batch_log2_total_variation=3)


def example_result(method, **changes):
    metrics = example_metrics()
    metrics.update(changes)
    return dict(id=method, scenario="example", seed=1, setting="same", delay_s=2,
                method=method, **metrics, **audit.classify(metrics))


class ClassificationTests(unittest.TestCase):
    def test_all_budget_boundaries_are_inclusive(self):
        metrics = example_metrics()
        metrics.update(integrated_violation_pu_s=0.10, worst_vmin=0.94,
                       worst_vmax=1.06, itl_violation_fraction=0.01)
        result = audit.classify(metrics)
        self.assertTrue(result["experimental_budget_pass"])
        self.assertFalse(result["strict_nominal_voltage_pass"])
        self.assertEqual(result["service_capacity_adequacy"], "not_established")

    def test_each_budget_can_independently_fail(self):
        for key, value in (("integrated_violation_pu_s", 0.100001),
                           ("worst_vmin", 0.939999), ("worst_vmax", 1.060001),
                           ("itl_violation_fraction", 0.010001)):
            with self.subTest(key=key):
                metrics = example_metrics()
                metrics[key] = value
                result = audit.classify(metrics)
                self.assertFalse(result["experimental_budget_pass"])
                self.assertEqual(result["experimental_budget_failures"], [key])

    def test_strict_nominal_check_is_separate(self):
        metrics = example_metrics()
        metrics.update(worst_vmin=0.95, worst_vmax=1.05)
        self.assertTrue(audit.classify(metrics)["strict_nominal_voltage_pass"])
        metrics["worst_vmin"] = 0.949999
        self.assertFalse(audit.classify(metrics)["strict_nominal_voltage_pass"])
        self.assertTrue(audit.classify(metrics)["experimental_budget_pass"])

    def test_nonfinite_metrics_are_rejected(self):
        for value in (float("nan"), float("inf")):
            metrics = example_metrics()
            metrics["worst_vmin"] = value
            with self.assertRaises(ValueError):
                audit.classify(metrics)

    def test_large_relative_gain_does_not_imply_acceptability(self):
        baseline = example_result("hold", integrated_violation_pu_s=0.20, worst_vmin=0.93)
        corrected = example_result("predict", integrated_violation_pu_s=0.04, worst_vmin=0.939)
        pair = audit.paired_comparison(baseline, corrected)
        self.assertAlmostEqual(pair["voltage_integral_reduction_percent"], 80)
        self.assertEqual(pair["absolute_budget_outcome"], "neither_passes")

    def test_all_pair_outcomes_and_throughput_boundary(self):
        for base_ok, corrected_ok, outcome in (
            (True, True, "both_pass"), (False, True, "only_correction_passes"),
            (True, False, "only_baseline_passes"), (False, False, "neither_passes"),
        ):
            with self.subTest(outcome=outcome):
                baseline = example_result("hold", worst_vmin=0.945 if base_ok else 0.93)
                corrected = example_result("predict", worst_vmin=0.945 if corrected_ok else 0.93,
                                           throughput_tps=98)
                pair = audit.paired_comparison(baseline, corrected)
                self.assertEqual(pair["absolute_budget_outcome"], outcome)
                self.assertTrue(pair["retains_98_percent_throughput"])
                corrected["throughput_tps"] = 97.999
                self.assertFalse(audit.paired_comparison(baseline, corrected)[
                    "retains_98_percent_throughput"])

    def test_unmatched_pairs_are_rejected(self):
        for key, value in (("scenario", "other"), ("seed", 2),
                           ("setting", "other"), ("delay_s", 0.5)):
            baseline, corrected = example_result("hold"), example_result("predict")
            corrected[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.paired_comparison(baseline, corrected)


class TraceTests(unittest.TestCase):
    def test_trace_metrics_and_inconsistent_rows(self):
        rows = [[i * 0.1, 8, 1000, 0.04, 2, 0.96, 1.04, 0, 400]
                for i in range(3000)]
        result = audit.recompute(rows)
        self.assertEqual(result["throughput_tps"], 400)
        self.assertEqual(result["integrated_violation_pu_s"], 0)
        self.assertEqual(result["batch_switches"], 0)
        self.assertTrue(audit.classify(result)["strict_nominal_voltage_pass"])
        for column, value in ((0, 99), (1, 7), (3, 0), (5, 0.94), (8, 999)):
            bad = copy.deepcopy(rows)
            bad[0][column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                audit.recompute(bad)

    def test_saved_audit_reproduces_and_does_not_hide_failed_seed(self):
        root = Path(__file__).resolve().parents[1]
        saved = audit.load(root / "results/delay-acceptability-audit/summary.json")
        current = audit.audit(root)
        # Text-file byte hashes record the original checkout. Git may convert
        # LF/CRLF in another checkout without changing any numerical evidence.
        # Keep the binary trace hashes and every computed result in this check.
        provenance = {"source_sha256", "auditor_sha256"}
        self.assertEqual({k: v for k, v in current.items() if k not in provenance},
                         {k: v for k, v in saved.items() if k not in provenance})
        self.assertEqual(set(current["source_sha256"]), set(saved["source_sha256"]))
        self.assertEqual(current["verified_run_count"], 48)
        self.assertEqual(current["strict_nominal_voltage_pass_count"], 0)
        self.assertEqual(sum(r["experimental_budget_pass"] for r in current["runs"]), 35)
        evaluated = [r for r in current["runs"] if r["stage"] == "evaluation"
                     and r["method"] == "predict"]
        self.assertEqual(len(evaluated), 4)
        self.assertEqual(sum(r["experimental_budget_pass"] for r in evaluated), 1)
        failed_seed = next(r for r in evaluated if r["scenario"] == "heldout_ramps"
                           and r["seed"] == 2)
        self.assertLess(failed_seed["worst_vmin"], 0.94)
        self.assertEqual(failed_seed["experimental_budget_failures"], ["worst_vmin"])


if __name__ == "__main__":
    unittest.main()
