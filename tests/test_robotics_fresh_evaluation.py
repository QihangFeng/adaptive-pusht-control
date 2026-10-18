"""CPU-only protocol, resume and paired-metric checks for fresh PushT scenes."""

import copy
from contextlib import ExitStack, redirect_stdout
import gzip
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_fresh_evaluation as fresh


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="fresh-evaluation-cpu-", dir=parent)


def trial(seed=50, method="fixed8", success=True, **changes):
    steps = 2 if success else 300
    horizon = 8 if method == "fixed8" else 12
    units = [min(horizon, steps - start) for start in range(0, steps, horizon)]
    result = {"seed": seed, "method": method, "success": success,
              "truncated": not success, "steps": steps,
              "model_queries": len(units),
              "initial": {"coverage": .25, "position": [float(seed), 100.]},
              "final_coverage": .96 if success else .90,
              "wall_seconds": 10., "monitor_seconds": 1. if method == "adaptive12" else 0.,
              "unit_lengths": units, "events": [],
              "trace": f"seed-{seed}-{method}.trace.json.gz"}
    result.update(changes)
    return result


def save_trial(directory, result, mutate_trace=None):
    """Create a tiny format-faithful saved artifact, without running physics."""
    directory.mkdir(exist_ok=True)
    boundaries, offset = set(), 1
    for length in result["unit_lengths"]:
        boundaries.add(offset)
        offset += length
    trace = {"initial": result["initial"],
             "steps": [{"step": step, "new_prediction": step in boundaries,
                        "after": {"coverage": result["final_coverage"]}}
                       for step in range(1, result["steps"] + 1)]}
    if mutate_trace:
        mutate_trace(trace)
    with gzip.open(directory / result["trace"], "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False)
    result = {**result, "trace_sha256": fresh.adaptive.base.sha(directory / result["trace"])}
    fresh.adaptive.base.save_json(directory / f"seed-{result['seed']}-{result['method']}.json", result)
    return result


def protocol_fixture(directory):
    development = directory / "development"
    development.mkdir()
    for name, document in (("protocol.json", {"original": True}),
                           ("summary.json", {"complete": True}),
                           ("prefix-smoke.json", {"passed": True})):
        fresh.adaptive.base.save_json(development / name, document)
    stack = ExitStack()
    stack.enter_context(patch.object(fresh, "DEVELOPMENT", development))
    stack.enter_context(patch.object(fresh.adaptive, "verify_protocol"))
    fingerprints = stack.enter_context(patch.object(
        fresh.adaptive, "fingerprints", side_effect=lambda: {"controller.py": "original"}))
    version = stack.enter_context(patch.object(fresh.importlib.metadata, "version", return_value="test-version"))
    return stack, development, fingerprints, version


class FrozenProtocolTest(unittest.TestCase):
    def test_exact_unused_scene_and_method_matrix(self):
        self.assertEqual(fresh.SEEDS, list(range(50, 70)))
        self.assertFalse(set(fresh.SEEDS) & set(range(50)))
        self.assertEqual(fresh.METHODS, ["fixed8", "fixed12", "adaptive12"])
        self.assertEqual(len({(s, m) for s in fresh.SEEDS for m in fresh.METHODS}), 60)

    def test_prepare_binds_unchanged_controller_and_earlier_evidence(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, development, _, _ = protocol_fixture(root)
            with stack:
                original_rules = copy.deepcopy(fresh.adaptive.RULES)
                protocol = fresh.prepare(root / "output")
                self.assertEqual(protocol["rules"], original_rules)
                self.assertIsNot(protocol["rules"], fresh.adaptive.RULES)
                self.assertEqual(protocol["prediction_repeats_per_scene"], 1)
                self.assertEqual(protocol["max_total_actions"], 300)
                self.assertEqual(protocol["checkpoint"]["weights_sha256"], fresh.adaptive.base.WEIGHTS_SHA256)
                for name in ("protocol.json", "summary.json", "prefix-smoke.json"):
                    key = (development / name).as_posix()
                    self.assertEqual(protocol["fingerprints"][key],
                                     fresh.adaptive.base.sha(development / name))
                key = "experiments/robotics_fresh_evaluation.py"
                self.assertEqual(protocol["fingerprints"][key], fresh.adaptive.base.sha(Path(fresh.__file__)))
                self.assertEqual(fresh.adaptive.RULES, original_rules)

    def test_prepare_is_idempotent_and_does_not_rewrite_protocol(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, _, _, _ = protocol_fixture(root)
            with stack:
                output = root / "output"
                first = fresh.prepare(output)
                original = (output / "protocol.json").read_bytes()
                self.assertEqual(fresh.prepare(output), first)
                self.assertEqual(fresh.verify_protocol(output), first)
                self.assertEqual((output / "protocol.json").read_bytes(), original)

    def test_changed_source_packages_or_development_evidence_are_rejected(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, development, fingerprint, version = protocol_fixture(root)
            with stack:
                output = root / "output"
                fresh.prepare(output)
                fingerprint.side_effect = lambda: {"controller.py": "changed"}
                with self.assertRaises(AssertionError):
                    fresh.verify_protocol(output)
                fingerprint.side_effect = lambda: {"controller.py": "original"}
                version.return_value = "changed-version"
                with self.assertRaises(AssertionError):
                    fresh.prepare(output)
                version.return_value = "test-version"
                fresh.adaptive.base.save_json(development / "summary.json", {"complete": False})
                with self.assertRaises(AssertionError):
                    fresh.verify_protocol(output)

    def test_changed_rules_or_seeds_cannot_rebind_existing_protocol(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, _, _, _ = protocol_fixture(root)
            with stack:
                output = root / "output"
                fresh.prepare(output)
                with patch.dict(fresh.adaptive.RULES, {"extension_max_coverage": .80}):
                    with self.assertRaises(AssertionError):
                        fresh.prepare(output)
                with patch.object(fresh, "SEEDS", list(range(70, 90))):
                    with self.assertRaises(AssertionError):
                        fresh.verify_protocol(output)

    def test_unpassed_prefix_smoke_prevents_protocol(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, development, _, _ = protocol_fixture(root)
            with stack:
                fresh.adaptive.base.save_json(development / "prefix-smoke.json", {"passed": False})
                with self.assertRaises(AssertionError):
                    fresh.prepare(root / "output")

    def test_no_new_protocol_can_adopt_existing_trials_or_summary(self):
        for name in ("episodes", "episodes.json", "summary.json"):
            with self.subTest(name=name), test_directory() as directory:
                root = Path(directory)
                stack, _, _, _ = protocol_fixture(root)
                with stack:
                    output = root / "output"
                    output.mkdir()
                    if name == "episodes":
                        (output / name).mkdir()
                    else:
                        fresh.adaptive.base.save_json(output / name, {})
                    with self.assertRaisesRegex(AssertionError, "pre-existing trials"):
                        fresh.prepare(output)

    def test_prior_evidence_directories_and_their_ancestors_are_protected(self):
        for destination in (Path("results/robotics-adaptive-length"),
                            Path("results/robotics-benefit-pilot/new"), Path("results")):
            with self.subTest(destination=destination):
                with self.assertRaises(ValueError):
                    fresh.check_output(destination)


class SavedTrialTest(unittest.TestCase):
    def test_successful_and_full_timeout_trials_are_valid(self):
        with test_directory() as directory:
            output = Path(directory)
            for success in (True, False):
                result = save_trial(output, trial(success=success))
                self.assertEqual(fresh.verify_trial(result, output), result)

    def test_changed_trace_bytes_are_rejected(self):
        with test_directory() as directory:
            output = Path(directory)
            result = save_trial(output, trial())
            (output / result["trace"]).write_bytes(b"changed-trace")
            with self.assertRaises(AssertionError):
                fresh.verify_trial(result, output)

    def test_unsupported_seed_method_and_trace_filename_are_rejected(self):
        with test_directory() as directory:
            output = Path(directory)
            result = save_trial(output, trial())
            for changes in ({"seed": 49}, {"seed": 70}, {"method": "learned"},
                            {"trace": "../elsewhere.trace.json.gz"}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    fresh.verify_trial({**result, **changes}, output)

    def test_changed_initial_or_nonconsecutive_steps_are_rejected(self):
        mutators = (lambda document: document.update(initial={"coverage": 0}),
                    lambda document: document["steps"][1].update(step=3))
        for mutate in mutators:
            with test_directory() as directory:
                output = Path(directory)
                result = save_trial(output, trial(), mutate)
                with self.assertRaises(AssertionError):
                    fresh.verify_trial(result, output)

    def test_query_unit_and_step_counts_must_agree(self):
        with test_directory() as directory:
            output = Path(directory)
            result = save_trial(output, trial())
            for changes in ({"steps": 3}, {"model_queries": 2},
                            {"unit_lengths": [1]}, {"unit_lengths": [0, 2]},
                            {"unit_lengths": [2.]}, {"final_coverage": .97}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    fresh.verify_trial({**result, **changes}, output)

    def test_success_is_strictly_above_ninety_five_percent(self):
        with test_directory() as directory:
            output = Path(directory)
            at_limit = save_trial(output, trial(final_coverage=.95))
            with self.assertRaises(AssertionError):
                fresh.verify_trial(at_limit, output)
            above_limit = save_trial(output, trial(final_coverage=.950001))
            fresh.verify_trial(above_limit, output)

    def test_short_failure_cannot_masquerade_as_completed_timeout(self):
        with test_directory() as directory:
            output = Path(directory)
            result = save_trial(output, trial(success=False, steps=2, unit_lengths=[2], model_queries=1))
            with self.assertRaises(AssertionError):
                fresh.verify_trial(result, output)

    def test_timings_must_be_finite_nonnegative_and_include_monitoring(self):
        with test_directory() as directory:
            output = Path(directory)
            result = save_trial(output, trial())
            for changes in ({"wall_seconds": math.inf}, {"wall_seconds": math.nan},
                            {"wall_seconds": -1.}, {"monitor_seconds": -1.},
                            {"monitor_seconds": 11.}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    fresh.verify_trial({**result, **changes}, output)


class CollectionResumeTest(unittest.TestCase):
    def test_fresh_matrix_calls_the_frozen_evaluator_once_per_missing_trial(self):
        with test_directory() as directory:
            output = Path(directory)
            policy = object()
            expected = [(seed, method) for index, seed in enumerate(fresh.SEEDS)
                        for method in fresh.METHODS[index % 3:] + fresh.METHODS[:index % 3]]

            def collect(actual_policy, seed, method, destination):
                self.assertIs(actual_policy, policy)
                self.assertEqual(destination, output / "episodes")
                return save_trial(destination, trial(seed, method))

            with (patch.object(fresh, "verify_protocol"),
                  patch.object(fresh.torch.cuda, "is_available", return_value=True),
                  patch.object(fresh.adaptive, "load_policy", return_value=(policy, {})) as load,
                  patch.object(fresh.adaptive, "evaluate_episode", side_effect=collect) as evaluate,
                  patch.object(fresh, "summarize"), redirect_stdout(io.StringIO())):
                fresh.run(output)
            self.assertEqual([(call.args[1], call.args[2]) for call in evaluate.call_args_list], expected)
            self.assertEqual(evaluate.call_count, 60)
            load.assert_called_once_with(fresh.adaptive.base.MODEL)
            document = fresh.adaptive.base.read_json(output / "episodes.json")
            self.assertTrue(document["complete"])
            self.assertEqual(len(document["episodes"]), 60)

    def test_completed_trials_are_verified_without_any_policy_loading(self):
        with test_directory() as directory:
            output = Path(directory)
            for seed in fresh.SEEDS:
                for method in fresh.METHODS:
                    save_trial(output / "episodes", trial(seed, method))
            with (patch.object(fresh, "verify_protocol"),
                  patch.object(fresh.torch.cuda, "is_available", return_value=True),
                  patch.object(fresh.adaptive, "load_policy") as load,
                  patch.object(fresh.adaptive, "evaluate_episode") as evaluate,
                  patch.object(fresh, "verify_trial", wraps=fresh.verify_trial) as verify,
                  patch.object(fresh, "summarize"), redirect_stdout(io.StringIO())):
                fresh.run(output)
            load.assert_not_called()
            evaluate.assert_not_called()
            self.assertEqual(verify.call_count, 60)

    def test_orphan_trace_and_wrong_sidecar_reject_resume_before_loading(self):
        for orphan in (True, False):
            with self.subTest(orphan=orphan), test_directory() as directory:
                output = Path(directory)
                episodes = output / "episodes"
                episodes.mkdir()
                if orphan:
                    (episodes / "seed-50-fixed8.trace.json.gz").write_bytes(b"partial")
                else:
                    fresh.adaptive.base.save_json(episodes / "seed-50-fixed8.json", trial(seed=51))
                with (patch.object(fresh, "verify_protocol"),
                      patch.object(fresh.torch.cuda, "is_available", return_value=True),
                      patch.object(fresh.adaptive, "load_policy") as load):
                    with self.assertRaises(AssertionError):
                        fresh.run(output)
                load.assert_not_called()

    def test_cpu_fallback_is_rejected_before_loading_or_collecting(self):
        with test_directory() as directory:
            with (patch.object(fresh, "verify_protocol"),
                  patch.object(fresh.torch.cuda, "is_available", return_value=False),
                  patch.object(fresh.adaptive, "load_policy") as load):
                with self.assertRaisesRegex(AssertionError, "CPU"):
                    fresh.run(Path(directory))
            load.assert_not_called()


class PairedSummaryTest(unittest.TestCase):
    def episodes(self):
        rows = []
        for seed in fresh.SEEDS:
            fixed_success = seed >= 52
            for method in fresh.METHODS:
                success = fixed_success if method != "adaptive12" else seed != 52
                rows.append(trial(seed, method, success,
                                  final_coverage=.90 + .02 * (method == "adaptive12"),
                                  model_queries=20 - 5 * (method != "fixed8"),
                                  wall_seconds=100. - 20. * (method != "fixed8")))
        return rows

    def test_paired_rescue_regression_and_differences_use_scene_pairs(self):
        result = fresh.paired_comparison(self.episodes(), "fixed8", "adaptive12")
        self.assertEqual(result["gained_success_seeds"], [50, 51])
        self.assertEqual(result["lost_success_seeds"], [52])
        self.assertEqual(result["both_success"], 17)
        self.assertEqual(result["both_failure"], 0)
        self.assertEqual(result["exact_two_sided_discordant_success_p"], 1.)
        differences = result["differences_method_minus_reference"]
        self.assertAlmostEqual(differences["success"]["mean_difference"], .05)
        for metric, expected in (("final_coverage", .02), ("model_queries", -5.), ("wall_seconds", -20.)):
            self.assertAlmostEqual(differences[metric]["mean_difference"], expected)
            lower, upper = differences[metric]["paired_bootstrap_95_interval"]
            self.assertAlmostEqual(lower, expected)
            self.assertAlmostEqual(upper, expected)

    def test_bootstrap_is_reproducible_and_independent_of_row_order(self):
        rows = self.episodes()
        self.assertEqual(fresh.paired_comparison(rows, "fixed8", "adaptive12"),
                         fresh.paired_comparison(list(reversed(rows)), "fixed8", "adaptive12"))

    def test_no_discordance_has_unit_p_and_zero_success_difference(self):
        result = fresh.paired_comparison(self.episodes(), "fixed8", "fixed12")
        self.assertEqual(result["exact_two_sided_discordant_success_p"], 1.)
        self.assertEqual(result["gained_success_seeds"], [])
        self.assertEqual(result["lost_success_seeds"], [])
        self.assertEqual(result["differences_method_minus_reference"]["success"]["mean_difference"], 0.)

    def test_strong_discordance_exact_p_matches_binomial_calculation(self):
        rows = self.episodes()
        for row in rows:
            row["success"] = row["method"] == "adaptive12"
        result = fresh.paired_comparison(rows, "fixed8", "adaptive12")
        self.assertAlmostEqual(result["exact_two_sided_discordant_success_p"], 2. / 2 ** 20)

    def test_missing_scene_prevents_pairwise_comparison(self):
        rows = [row for row in self.episodes() if not (row["seed"] == 50 and row["method"] == "adaptive12")]
        with self.assertRaises(AssertionError):
            fresh.paired_comparison(rows, "fixed8", "adaptive12")

    def test_summary_requires_complete_exact_unique_matrix(self):
        rows = self.episodes()
        invalid = ({"complete": False, "episodes": rows},
                   {"complete": True, "episodes": rows[:-1]},
                   {"complete": True, "episodes": rows[:-1] + [rows[0]]})
        for document in invalid:
            with test_directory() as directory:
                output = Path(directory)
                fresh.adaptive.base.save_json(output / "episodes.json", document)
                with patch.object(fresh, "verify_protocol"):
                    with self.assertRaises(AssertionError):
                        fresh.summarize(output)

    def test_summary_averages_monitoring_and_records_all_contrasts(self):
        rows = self.episodes()
        rows[-1]["events"] = [{"kind": "extend"}, {"kind": "interrupt"}]
        with test_directory() as directory:
            output = Path(directory)
            fresh.adaptive.base.save_json(output / "episodes.json", {"complete": True, "episodes": rows})
            with (patch.object(fresh, "verify_protocol"),
                  patch.object(fresh, "verify_trial") as verify, redirect_stdout(io.StringIO())):
                result = fresh.summarize(output)
            self.assertEqual(verify.call_count, 60)
            self.assertEqual(set(result["paired"]), {"fixed12_vs_fixed8", "adaptive12_vs_fixed8", "adaptive12_vs_fixed12"})
            baseline = result["methods"]["fixed8"]
            adaptive = result["methods"]["adaptive12"]
            self.assertEqual(baseline["successes"], 18)
            self.assertEqual(adaptive["successes"], 19)
            self.assertEqual(adaptive["attempts"], 20)
            self.assertEqual(adaptive["mean_model_queries"], 15.)
            self.assertEqual(adaptive["mean_wall_seconds"], 80.)
            self.assertEqual(adaptive["mean_monitor_seconds"], 1.)
            self.assertEqual(adaptive["extensions"], 1)
            self.assertEqual(adaptive["interruptions"], 1)
            self.assertEqual(fresh.adaptive.base.read_json(output / "summary.json"), result)

    def test_summary_rejects_methods_starting_from_different_initial_scene(self):
        rows = self.episodes()
        rows[1]["initial"] = {"coverage": .25, "position": [999., 100.]}
        with test_directory() as directory:
            output = Path(directory)
            fresh.adaptive.base.save_json(output / "episodes.json", {"complete": True, "episodes": rows})
            with patch.object(fresh, "verify_protocol"), patch.object(fresh, "verify_trial"):
                with self.assertRaisesRegex(AssertionError, "same physical scene"):
                    fresh.summarize(output)


if __name__ == "__main__":
    unittest.main()
