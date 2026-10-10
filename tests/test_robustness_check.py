"""Unit checks plus independent saved-trace audit when robustness evidence exists."""

import copy
import hashlib
import math
import statistics
import struct
import unittest

from experiments import delay_acceptability_audit as audit
from experiments import robustness_check as check


def sample_rows():
    return [[i * 0.1, 8, 1000, 0.01 if i < 1500 else 0.02, 1,
             0.94 if i < 20 else 0.96, 1.04, 0.01 if i < 20 else 0,
             800 if i < 1500 else 400] for i in range(3000)]


def trace_fixture():
    rows = sample_rows()
    profile = check.make_profile(3, "variable", "independent")
    observations = [{"time_s": float(i)} for i in range(300)]
    for observation, packet in zip(observations, check.replay_latest_packets(observations, 2, profile)):
        observation["available"] = packet is not None
        if packet is not None:
            measured, arrived = packet
            observation.update(measured_s=measured, arrival_s=arrived,
                               age_s=observation["time_s"] - measured,
                               packet_power_kw=[1 / 3] * 3, current_power_kw=[1 / 3] * 3,
                               packet_voltage=rows[round(measured / 0.1)][5:7],
                               voltage_used=rows[round(measured / 0.1)][5:7])
    external = [check.external_power_kw(row[0], profile) for row in rows]
    trace = dict(rows=rows, observations=observations, sensitivity=[[0, 0, 0], [0, 0, 0]],
                 robustness={**profile, "external_power_kw_per_phase": external,
                             "external_applied_kw_by_phase": [[p] * 3 for p in external]})
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack("<9d", *row))
    result = dict(id="fixture", method="hold", setting="half_primal_dual", seed=3,
                  scenario="abrupt", delay_s=2, control_period_s=1, robustness=profile,
                  trajectory_sha256=digest.hexdigest(), all_solves_converged=True,
                  external_load_readback_matches_profile=True, **check.recompute_metrics(rows))
    return trace, result


class ProfileTests(unittest.TestCase):
    def test_variable_schedule_repeatability_and_balanced_mean(self):
        profile = check.make_profile(3, "variable", "none")
        self.assertEqual(profile, check.make_profile(3, "variable", "none"))
        self.assertNotEqual(profile["delay_blocks_s"],
                            check.make_profile(4, "variable", "none")["delay_blocks_s"])
        self.assertEqual(profile["delay_blocks_s"][0], 2)
        self.assertEqual(sorted(profile["delay_blocks_s"][1:]), [0.5] * 8 + [2] * 8 + [3.5] * 8)
        self.assertEqual(statistics.mean(profile["delay_blocks_s"]), 2)
        self.assertEqual(check.transport_delay(0, profile), 2)
        self.assertEqual(check.transport_delay(11.9, profile), 2)
        self.assertEqual(check.transport_delay(12, profile), profile["delay_blocks_s"][1])
        self.assertEqual(check.transport_delay(299.9, profile), profile["delay_blocks_s"][-1])
        for t in (-0.1, 300, math.inf, math.nan):
            with self.subTest(t=t), self.assertRaises(ValueError):
                check.transport_delay(t, profile)

    def test_local_rng_does_not_mutate_global_numpy_state(self):
        import numpy as np
        np.random.seed(123)
        before = np.random.get_state()
        check.make_profile(3, "variable", "none")
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_external_units_and_event_boundaries(self):
        on = check.make_profile(3, "constant", "independent")
        off = check.make_profile(3, "constant", "none")
        for t, expected in ((74.9, 0), (75, 50), (94.9, 50), (95, 0),
                            (150, 0), (152.5, 50), (155, 100), (164.9, 100),
                            (165, 100), (167.5, 50), (170, 0), (299.9, 0)):
            with self.subTest(t=t):
                self.assertEqual(check.external_power_kw(t, on), expected)
                self.assertEqual(check.external_power_kw(t, off), 0)
        # Scalar is per phase: the three-phase plateau is 300kW, not 100kW.
        self.assertEqual(3 * check.external_power_kw(160, on), 300)

    def test_every_policy_gets_identical_paired_perturbations(self):
        jobs = check.build_jobs()
        self.assertEqual(len(jobs), 72)
        self.assertEqual(len({job["id"] for job in jobs}), 72)
        groups = {}
        for job in jobs:
            p = job["robustness"]
            key = (job["scenario"], job["seed"], p["delay_mode"], p["external_mode"])
            groups.setdefault(key, []).append(job)
        self.assertEqual(len(groups), 24)
        for group in groups.values():
            self.assertEqual(len(group), 3)
            self.assertEqual({check.policy_name(j) for j in group}, set(check.POLICIES))
            self.assertTrue(all(j["robustness"] == group[0]["robustness"] for j in group))
        # Delay schedule ignores workload and external flag, only independent seed matters.
        for seed in check.SEEDS:
            self.assertEqual(len({j["robustness"]["offered_delay_sha256"] for j in jobs
                                  if j["seed"] == seed and j["robustness"]["delay_mode"] == "variable"}), 1)
        with self.assertRaises(ValueError):
            changed = copy.deepcopy(jobs[0]["robustness"])
            changed["delay_blocks_s"][1] = 0.5
            check.validate_profile(changed)

    def test_replay_prefers_newer_sample_over_late_older_arrival(self):
        profile = check.make_profile(3, "variable", "none")
        # Force the first post-startup block to 0.5s while preserving counts.
        index = profile["delay_blocks_s"].index(0.5)
        profile["delay_blocks_s"][1], profile["delay_blocks_s"][index] = (
            profile["delay_blocks_s"][index], profile["delay_blocks_s"][1])
        profile["offered_delay_sha256"] = check.content_hash(profile["delay_blocks_s"])
        check.validate_profile(profile)
        observations = [{"time_s": t} for t in (1, 2, 12, 13, 14)]
        packets = check.replay_latest_packets(observations, 2, profile)
        self.assertIsNone(packets[0])
        self.assertEqual(packets[1], (0, 2))
        self.assertAlmostEqual(packets[2][0], 10)
        self.assertAlmostEqual(packets[3][0], 12.5)
        self.assertAlmostEqual(packets[4][0], 13.5)
        # Check independently against the existing implementation's sensor channel.
        from experiments.delay_screen import VoltageChannel
        channel = VoltageChannel()
        observed = {}
        for i in range(141):
            t = i * 0.1
            channel.publish(t, check.transport_delay(t, profile), [1], [1, 1, 1])
            packet = channel.receive(t)
            if t in (1, 2, 12, 13, 14):
                observed[t] = None if packet is None else (packet.measured, packet.arrives)
        for observation, packet in zip(observations, packets):
            self.assertEqual(observed[observation["time_s"]], packet)


class MetricAndPairTests(unittest.TestCase):
    def test_all_four_metrics_include_startup_and_time_weighting(self):
        metrics = check.recompute_metrics(sample_rows())
        self.assertAlmostEqual(metrics["voltage_violation_fraction_percent"], 2 / 300 * 100)
        self.assertAlmostEqual(metrics["integrated_violation_pu_s"], 0.02)
        self.assertAlmostEqual(metrics["throughput_tps"], 600)
        self.assertAlmostEqual(metrics["mean_modeled_itl_ms"], 15)

    def test_pair_matching_and_zero_baseline(self):
        jobs = check.build_jobs()
        baseline = {**jobs[0], **{key: 100 for key in check.METRICS}}
        corrected = {**jobs[2], **{key: 90 for key in check.METRICS}}
        pair = check.paired_comparison(baseline, corrected)
        self.assertEqual(pair["baseline_policy"], "tuned")
        self.assertAlmostEqual(pair["metrics"]["throughput_tps"]["change_percent"], -10)
        baseline["integrated_violation_pu_s"] = 0
        self.assertIsNone(check.paired_comparison(baseline, corrected)["metrics"][
            "integrated_violation_pu_s"]["change_percent"])
        for key in ("seed", "scenario", "delay_s"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check.paired_comparison(baseline, {**corrected, key: "different"})
        mismatched = copy.deepcopy(corrected)
        mismatched["robustness"]["external_profile_sha256"] = "different"
        with self.assertRaises(ValueError):
            check.paired_comparison(baseline, mismatched)

    def test_aggregation_requires_all_declared_seeds_and_complete_pairs(self):
        runs = [{**j, **{key: float(j["seed"]) for key in check.METRICS},
                 "mean_age_s": 2.0, "max_age_s": 3.5} for j in check.build_jobs()]
        aggregates = check.aggregate_runs(runs)
        self.assertEqual(len(aggregates), 24)
        self.assertEqual(aggregates[0]["metrics"]["mean_modeled_itl_ms"],
                         {"mean": 4.0, "min": 3.0, "max": 5.0, "sample_sd": 1.0})
        self.assertEqual(len(check.build_comparisons(runs)), 48)
        for function in (check.aggregate_runs, check.build_comparisons):
            with self.subTest(function=function), self.assertRaises(ValueError):
                function(runs[:-1])

    def test_trace_audit_rejects_wrong_packet_and_wrong_physical_application(self):
        trace, result = trace_fixture()
        checked = check.validate_trace(trace, result)
        self.assertAlmostEqual(checked["mean_modeled_itl_ms"], 15)
        for failure in ("future_arrival", "older_selected_sample", "wrong_correction",
                        "external_wrong_phase", "changed_trajectory", "wrong_saved_mean"):
            changed_trace, changed_result = copy.deepcopy(trace), copy.deepcopy(result)
            observation = changed_trace["observations"][20]
            if failure == "future_arrival":
                observation["arrival_s"] = 21
            elif failure == "older_selected_sample":
                observation["measured_s"] -= 0.1
                observation["arrival_s"] -= 0.1
                observation["age_s"] += 0.1
            elif failure == "wrong_correction":
                changed_result["method"] = "predict"
                observation["voltage_used"][0] += 0.01
            elif failure == "external_wrong_phase":
                changed_trace["robustness"]["external_applied_kw_by_phase"][750][1] = 0
            elif failure == "changed_trajectory":
                changed_result["trajectory_sha256"] = "different"
            else:
                changed_result["mean_modeled_itl_ms"] = 20
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                check.validate_trace(changed_trace, changed_result)


class SavedEvidenceTests(unittest.TestCase):
    @unittest.skipUnless((check.ROOT / "results/robustness-check/summary.json").exists(),
                         "Run frozen robustness suite before checking saved evidence")
    def test_completed_evidence_recomputes_independently(self):
        directory = check.ROOT / "results/robustness-check"
        summary = audit.load(directory / "summary.json")
        protocol = audit.load(directory / "protocol.json")
        self.assertTrue(summary["complete"])
        self.assertEqual(len(summary["compatibility_checks"]), 4)
        self.assertTrue(all(c["matches"] for c in summary["compatibility_checks"]))
        self.assertEqual(len(summary["runs"]), 72)
        self.assertEqual(len(summary["comparisons"]), 48)
        self.assertEqual(len(summary["aggregates"]), 24)
        self.assertEqual(summary["protocol_sha256"], audit.sha256(directory / "protocol.json"))
        self.assertEqual(summary["prepared_data_sha256_after"], protocol["prepared_data_sha256_before"])
        self.assertTrue(summary["prepared_inputs_unchanged"])
        self.assertTrue(summary["driver_files_unchanged"])
        self.assertEqual(protocol["evaluation_jobs"], check.build_jobs())
        runs = []
        for result in [*summary["compatibility_runs"], *summary["runs"]]:
            saved = audit.load(directory / f"{result['id']}.json")
            self.assertEqual(result, check.checked_result(directory, saved))
            if "robustness" in result:
                runs.append(result)
        self.assertEqual(summary["comparisons"], check.build_comparisons(runs))
        self.assertEqual(summary["aggregates"], check.aggregate_runs(runs))
        for compatibility in summary["compatibility_checks"]:
            saved = audit.load(check.ROOT / "results" / compatibility["prior_directory"]
                               / f"{compatibility['prior_id']}.json")
            self.assertEqual(compatibility["trajectory_sha256"], saved["trajectory_sha256"])


if __name__ == "__main__":
    unittest.main()
