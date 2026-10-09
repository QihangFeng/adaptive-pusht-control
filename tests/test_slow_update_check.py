"""No simulator dependency; saved traces are checked when the suite exists."""

import unittest

from experiments import delay_acceptability_audit as audit
from experiments import slow_update_check as check


def result(observation, **changes):
    row = dict(id=observation, scenario="example", seed=1, delay_s=2,
               method=observation, setting="slow_update" if observation == "hold" else "half_primal_dual",
               integrated_violation_pu_s=0.05, violation_time_s=2,
               worst_vmin=0.945, worst_vmax=1.04, throughput_tps=100,
               itl_violation_fraction=0, batch_switches=3, batch_log2_total_variation=3)
    row.update(changes)
    return row


class ComparisonTests(unittest.TestCase):
    def test_generated_manifest_is_only_hash_exception(self):
        original = {"profile/_manifest.json": "old", "profile/trace.csv": "trace",
                    "profile/itl_fit.json": "itl", "profile/logistic_fit.json": "power"}
        current = dict(original, **{"profile/_manifest.json": "new"})
        self.assertEqual(check.numerical_input_gate(current, original),
                         {k: v for k, v in original.items() if not k.endswith("_manifest.json")})
        for changed in ({**current, "profile/trace.csv": "bad"},
                        {**current, "extra": "extra"}, {"profile/_manifest.json": "new"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check.numerical_input_gate(changed, original)

    def test_absolute_failures_cannot_be_called_success(self):
        pair = check.compare(result("hold", worst_vmin=0.93, integrated_violation_pu_s=0.2),
                             result("predict", worst_vmin=0.939, throughput_tps=150))
        self.assertEqual(pair["absolute_budget_outcome"], "neither_passes")
        self.assertAlmostEqual(pair["corrected_voltage_integral_reduction_percent"], 75)
        self.assertFalse(pair["additional_value_screen_pass"])

    def test_both_pass_requires_five_percent_gain_and_switch_cap(self):
        slow = result("hold")
        self.assertFalse(check.compare(slow, result("predict", throughput_tps=104.999))[
            "additional_value_screen_pass"])
        self.assertTrue(check.compare(slow, result("predict", throughput_tps=105, batch_switches=5))[
            "additional_value_screen_pass"])
        self.assertFalse(check.compare(slow, result("predict", throughput_tps=105, batch_switches=6))[
            "additional_value_screen_pass"])

    def test_only_correction_passes_requires_preserved_throughput(self):
        slow = result("hold", worst_vmin=0.93)
        pair = check.compare(slow, result("predict", throughput_tps=98))
        self.assertEqual(pair["absolute_budget_outcome"], "only_correction_passes")
        self.assertTrue(pair["additional_value_screen_pass"])
        self.assertFalse(check.compare(slow, result("predict", throughput_tps=97.999))[
            "additional_value_screen_pass"])

    def test_only_simple_control_passes(self):
        pair = check.compare(result("hold"), result("predict", worst_vmin=0.93, throughput_tps=150))
        self.assertEqual(pair["absolute_budget_outcome"], "only_slower_controller_passes")
        self.assertFalse(pair["additional_value_screen_pass"])

    def test_pair_mismatch_and_wrong_methods_rejected(self):
        for change in ({"seed": 2}, {"scenario": "other"}, {"delay_s": 0.5},
                       {"method": "hold"}, {"setting": "reference"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check.compare(result("hold"), result("predict", **change))


class SavedEvidenceTests(unittest.TestCase):
    @unittest.skipUnless((check.ROOT / "results/slow-update-check/summary.json").exists(),
                         "Run bounded six-case suite before checking saved evidence")
    def test_new_traces_and_comparisons_reproduce(self):
        directory = check.ROOT / "results/slow-update-check"
        summary = audit.load(directory / "summary.json")
        protocol = audit.load(directory / "protocol.json")
        self.assertEqual(len(summary["runs"]), 6)
        self.assertEqual(len(summary["compatibility_checks"]), 2)
        self.assertTrue(all(c["matches"] for c in summary["compatibility_checks"]))
        self.assertEqual(summary["prepared_data_sha256_after"], protocol["prepared_data_sha256_before"])
        self.assertTrue(summary["prepared_inputs_unchanged"])
        for row in summary["runs"]:
            metrics, trace_hash = audit.audit_trace(directory, row)
            self.assertEqual(trace_hash, row["trace_file_sha256"])
            self.assertEqual(audit.classify(metrics)["experimental_budget_pass"], row["experimental_budget_pass"])
            self.assertEqual(audit.load(directory / f"{row['id']}.job.json"),
                             {key: row[key] for key in check.JOB_KEYS})
        saved = {r["id"]: r for r in summary["saved_comparators"]}
        runs = {r["id"]: r for r in summary["runs"]}
        for row in saved.values():
            original = audit.load(check.ROOT / "results/delay-prototype" / f"{row['id']}.json")
            self.assertEqual(row, check.checked_result(check.ROOT / "results/delay-prototype", original))
        for compatibility in summary["compatibility_checks"]:
            self.assertEqual(runs[compatibility["id"]]["trajectory_sha256"],
                             saved[compatibility["prior_id"]]["trajectory_sha256"])
        self.assertEqual(len(summary["comparisons"]), 4)
        for pair in summary["comparisons"]:
            self.assertEqual(pair, check.compare(runs[pair["slower_id"]], saved[pair["corrected_id"]]))


if __name__ == "__main__":
    unittest.main()
