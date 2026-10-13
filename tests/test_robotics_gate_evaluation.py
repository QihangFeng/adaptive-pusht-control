"""CPU-only full-plan, camera-gate and frozen fresh-evaluation checks."""

from contextlib import ExitStack, redirect_stdout
import copy
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robotics_gate_evaluation as evaluation
from test_robotics_adaptive_length import FakeEnvironment, FakePolicy, fake_physical_state


CONFIGS = {"weighted": {"family": "weighted", "weights": [.5, .25, .25], "threshold": .5},
           "threshold": {"family": "threshold", "min_warnings": 3, "min_drawdown": .15}}


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="gate-evaluation-cpu-", dir=parent)


class FakeCamera:
    def __init__(self, decline=False):
        self.index, self.decline = -1, decline

    def estimate(self, pixels):
        self.index += 1
        coverage = (.86 if self.index < 4 else .82 if self.index == 4 else .78) if self.decline else .60
        return {"quality": True, "coverage": coverage, "position_error": .1, "orientation_error": .1}


def collect(directory, method="adaptive12", config=None, decline=False, seed=70):
    policy, env, camera = FakePolicy(), FakeEnvironment(), FakeCamera(decline)
    with (patch.object(evaluation.adaptive.gym, "make", return_value=env),
          patch.object(evaluation.adaptive, "observation_batch", side_effect=lambda observation, device: observation),
          patch.object(evaluation.adaptive, "physical_state", side_effect=fake_physical_state),
          patch.object(evaluation.adaptive, "VisualAlignmentEstimator", return_value=camera),
          patch.object(evaluation.torch.cuda, "synchronize")):
        result = evaluation.evaluate_episode(policy, seed, method, directory, config)
    trace = evaluation.verify_trial(result, directory, config)
    return result, trace, policy, env, camera


def rewrite_trace(directory, result, mutate):
    path = directory / result["trace"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    mutate(trace)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False)
    return {**result, "trace_sha256": evaluation.base.sha(path)}


class EpisodeAndGateTest(unittest.TestCase):
    def test_all_three_arms_log_full_plans_initial_camera_and_preserved_history(self):
        traces = []
        for method in evaluation.METHODS:
            with test_directory() as directory:
                result, trace, policy, env, camera = collect(Path(directory), method, CONFIGS.get(method))
                self.assertTrue(env.closed)
                self.assertEqual(policy.resets, 1)
                self.assertEqual(policy.seen, list(range(16)))
                self.assertEqual(list(policy._queues["observation.state"]), [14, 15])
                self.assertEqual(result["unit_lengths"], [12, 4])
                self.assertEqual(result["model_queries"], 2)
                self.assertEqual(result["prediction_seed"], 70)
                self.assertEqual(trace["initial_estimate"]["coverage"], .60)
                self.assertEqual(trace["initial_features"]["coverage"], .60)
                self.assertEqual(camera.index, 15)
                self.assertTrue(all(len(row["predicted_commands"]) == 12 for row in trace["steps"] if row["new_prediction"]))
                self.assertTrue(all("predicted_commands" not in row for row in trace["steps"] if not row["new_prediction"]))
                self.assertTrue(all(key not in trace["steps"][-1] for key in
                                    ("features", "gate", "blocked", "controller_before", "controller_after")))
                traces.append(trace)
        actions = [[r["action"] for r in trace["steps"]] for trace in traces]
        self.assertEqual(actions[0], actions[1])
        self.assertEqual(actions[0], actions[2])

    def test_gate_veto_does_not_consume_interruption_budget_and_still_trims_at_eight(self):
        with test_directory() as directory:
            result, trace, _, _, _ = collect(Path(directory), "weighted",
                                             {"family": "weighted", "weights": [1., 0., 0.], "threshold": 1.}, True)
            blocked = [row for row in trace["steps"] if row.get("blocked")]
            self.assertTrue(blocked)
            for row in blocked:
                self.assertEqual(row["controller_after"]["interruptions"], row["controller_before"]["interruptions"])
                self.assertEqual(row["controller_after"]["last_interrupt"], row["controller_before"]["last_interrupt"])
                self.assertEqual(row["controller_after"]["armed"], row["controller_before"]["armed"])
                self.assertGreater(row["controller_after"]["confirmations"], row["controller_before"]["confirmations"])
            self.assertEqual(result["unit_lengths"], [8, 8])
            self.assertTrue(any(e["kind"] == "trim" and e["step"] == 8 for e in result["events"]))
            self.assertFalse(any(e["kind"] == "interrupt" for e in result["events"]))

    def test_unchanged_arm_does_not_force_interruptions_and_invalid_gate_cannot_leak_truth(self):
        invalid = {"quality": False, "coverage": None, "drawdown": None, "warning_count": None}
        self.assertEqual(evaluation.decision("adaptive12", None, invalid, []),
                         {"allowed": True, "signals": None, "score": None})
        altered = {**invalid, "true_coverage": 1., "contact": False, "success": True}
        for method in ("threshold", "weighted"):
            self.assertEqual(evaluation.decision(method, CONFIGS[method], invalid, []),
                             evaluation.decision(method, CONFIGS[method], altered, []))
            self.assertFalse(evaluation.decision(method, CONFIGS[method], invalid, [1., 1., 1.])["allowed"])
        with self.assertRaises(AssertionError):
            evaluation.decision("weighted", CONFIGS["threshold"], invalid, [])

    def test_trace_verifier_recomputes_queue_controller_gate_and_terminal_contract(self):
        mutations = (lambda trace: trace["steps"][0]["predicted_commands"][0].__setitem__(0, -1.),
                     lambda trace: trace["initial_estimate"].update(coverage=.99),
                     lambda trace: trace["steps"][0]["features"].update(drawdown=.99),
                     lambda trace: trace["steps"][0].update(pending_before=0),
                     lambda trace: trace["steps"][0]["controller_after"].update(interruptions=2),
                     lambda trace: trace["steps"][0]["gate"].update(allowed=True),
                     lambda trace: trace["steps"][-1].update(features={"quality": False}))
        for mutate in mutations:
            with test_directory() as directory:
                output = Path(directory)
                result, _, _, _, _ = collect(output, "weighted", CONFIGS["weighted"])
                changed = rewrite_trace(output, result, mutate)
                with self.assertRaises(AssertionError):
                    evaluation.verify_trial(changed, output, CONFIGS["weighted"])

    def test_trial_seed_prediction_identity_trace_hash_and_metrics_are_checked(self):
        with test_directory() as directory:
            output = Path(directory)
            result, _, _, _, _ = collect(output)
            for changed in ({"seed": 69}, {"seed": 90}, {"prediction_seed": 1},
                            {"trace_sha256": "0" * 64}, {"trace": "../other.trace.json.gz"},
                            {"wall_seconds": float("inf")}, {"monitor_seconds": -1.},
                            {"model_queries": 99}, {"blocked_candidates": 1}, {"success": False}):
                with self.subTest(changed=changed), self.assertRaises(AssertionError):
                    evaluation.verify_trial({**result, **changed}, output)

    def test_saved_collection_does_not_call_policy_and_orphan_trace_prevents_collection(self):
        with test_directory() as directory:
            output = Path(directory)
            result, _, _, _, _ = collect(output)
            with patch.object(evaluation, "evaluate_episode") as evaluate:
                self.assertEqual(evaluation.checked_collect(object(), 70, "adaptive12", output, None, evaluation.SEEDS), result)
                evaluate.assert_not_called()
                (output / "seed-71-adaptive12.trace.json.gz").write_bytes(b"partial")
                with self.assertRaisesRegex(AssertionError, "orphan trace"):
                    evaluation.checked_collect(object(), 71, "adaptive12", output, None, evaluation.SEEDS)
                evaluate.assert_not_called()


class FrozenBindingTest(unittest.TestCase):
    def fixture(self, root):
        old, seen = root / "old-screen", root / "seen"
        for name, value in (("protocol.json", {"fingerprints": {"old.py": "original"}}),
                            ("cases.json", {"cases": []}), ("comparisons.json", {"records": []}),
                            ("summary.json", {"complete": True}), ("independent-audit.json", {"passed": True})):
            evaluation.base.save_json(old / name, value)
        evaluation.base.save_json(seen / "episodes.json", {"episodes": [{"seed": seed} for seed in range(50, 70)]})
        stack = ExitStack()
        stack.enter_context(patch.object(evaluation.screen, "OUTPUT", old))
        stack.enter_context(patch.object(evaluation.screen.fresh, "OUTPUT", seen))
        development = stack.enter_context(patch.object(evaluation, "development_document", return_value={"selected": copy.deepcopy(CONFIGS)}))
        version = stack.enter_context(patch.object(evaluation.importlib.metadata, "version", return_value="test-version"))
        return stack, old, development, version

    def test_exact_unused_seed_matrix_and_overlap_protection(self):
        self.assertEqual(evaluation.SEEDS, list(range(70, 90)))
        self.assertEqual(evaluation.METHODS, ["adaptive12", "threshold", "weighted"])
        self.assertEqual(len({(s, m) for s in evaluation.SEEDS for m in evaluation.METHODS}), 60)
        for output in (Path("results/robotics-decline-screen"), Path("results/robotics-decline-screen/new"),
                       Path("results/robotics-regression-diagnosis"), Path("results")):
            with self.subTest(output=output), self.assertRaises(ValueError):
                evaluation.check_output(output)

    def test_preparation_is_idempotent_and_freezes_models_configs_sources_and_packages(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, _, _, _ = self.fixture(root)
            with stack:
                output = root / "output"
                first = evaluation.prepare(output)
                encoded = (output / "protocol.json").read_bytes()
                self.assertEqual(first["selected"], CONFIGS)
                self.assertEqual(first["normalization"], evaluation.gate.NORMALIZATION)
                self.assertEqual(first["checkpoint"]["weights_sha256"], evaluation.base.WEIGHTS_SHA256)
                self.assertEqual(first["fingerprints"]["experiments/robotics_gate_evaluation.py"],
                                 evaluation.base.sha(Path(evaluation.__file__)))
                self.assertEqual(evaluation.prepare(output), first)
                self.assertEqual(evaluation.verify_protocol(output), first)
                self.assertEqual((output / "protocol.json").read_bytes(), encoded)

    def test_changed_selected_config_package_source_evidence_or_development_rejects_resume(self):
        for changed in ("selected", "package", "source", "development"):
            with self.subTest(changed=changed), test_directory() as directory:
                root = Path(directory)
                stack, old, development, version = self.fixture(root)
                with stack:
                    output = root / "output"
                    evaluation.prepare(output)
                    if changed == "selected":
                        document = {"selected": copy.deepcopy(CONFIGS)}
                        document["selected"]["weighted"]["threshold"] = .65
                        development.return_value = document
                    elif changed == "package":
                        version.return_value = "changed-version"
                    elif changed == "source":
                        evaluation.base.save_json(old / "summary.json", {"complete": False})
                    else:
                        evaluation.base.save_json(output / "development.json", {"selected": {}})
                    with self.assertRaises(AssertionError):
                        evaluation.verify_protocol(output)

    def test_fresh_protocol_and_evaluation_binding_cannot_adopt_existing_episodes(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, _, _, _ = self.fixture(root)
            with stack:
                output = root / "output"
                (output / "episodes").mkdir(parents=True)
                with self.assertRaises(AssertionError):
                    evaluation.prepare(output)
            evaluation.base.save_json(output / "protocol.json", {"frozen": True})
            evaluation.base.save_json(output / "smoke.json", {"passed": True})
            with patch.object(evaluation, "verify_protocol"), patch.object(evaluation, "verify_smoke"):
                with self.assertRaisesRegex(AssertionError, "bind collected episodes"):
                    evaluation.evaluation_binding(output)

    def test_smoke_artifacts_and_evaluation_binding_are_hash_checked(self):
        with test_directory() as directory:
            output = Path(directory)
            artifact = output / "functional-proof.json"
            evaluation.base.save_json(artifact, {"exact": True})
            smoke = {"passed": True, "exact_reference_scene": evaluation.SMOKE_SCENE,
                     "retuning": False, "fingerprints": {artifact.as_posix(): evaluation.base.sha(artifact)}}
            evaluation.base.save_json(output / "smoke.json", smoke)
            self.assertEqual(evaluation.verify_smoke(output), smoke)
            evaluation.base.save_json(output / "protocol.json", {"frozen": True})
            with patch.object(evaluation, "verify_protocol"):
                first = evaluation.evaluation_binding(output)
                self.assertEqual(evaluation.evaluation_binding(output), first)
                evaluation.base.save_json(output / "protocol.json", {"frozen": False})
                with self.assertRaises(AssertionError):
                    evaluation.evaluation_binding(output)
            evaluation.base.save_json(artifact, {"exact": False})
            with self.assertRaises(AssertionError):
                evaluation.verify_smoke(output)


class PairedStatisticsAndRunTest(unittest.TestCase):
    def rows(self):
        return [{"seed": seed, "method": method,
                 "success": (seed >= 72 if method != "weighted" else seed != 72),
                 "initial": {"coverage": .1, "scene": seed},
                 "final_coverage": .90 + .02 * (method == "weighted"),
                 "model_queries": 20 - 4 * (method == "weighted"),
                 "wall_seconds": 100. - 10. * (method == "weighted"), "monitor_seconds": 1.,
                 "events": [], "blocked_candidates": 0}
                for seed in evaluation.SEEDS for method in evaluation.METHODS]

    def test_pairing_reports_rescues_regressions_and_deterministic_metric_intervals(self):
        rows = self.rows()
        paired = evaluation.paired_comparison(rows, "adaptive12", "weighted")
        self.assertEqual(paired["gained_success_seeds"], [70, 71])
        self.assertEqual(paired["lost_success_seeds"], [72])
        self.assertEqual(paired["exact_two_sided_discordant_success_p"], 1.)
        differences = paired["differences_method_minus_reference"]
        self.assertAlmostEqual(differences["success"]["mean_difference"], .05)
        self.assertEqual(differences["model_queries"]["paired_bootstrap_95_interval"], [-4., -4.])
        self.assertEqual(evaluation.paired_comparison(list(reversed(rows)), "adaptive12", "weighted"), paired)

    def test_missing_paired_scene_is_rejected(self):
        rows = self.rows()[1:]
        with self.assertRaises(AssertionError):
            evaluation.paired_comparison(rows, "adaptive12", "weighted")

    def test_summary_requires_complete_unique_matrix_and_matching_initial_scenes(self):
        rows = self.rows()
        documents = ({"complete": False, "episodes": rows}, {"complete": True, "episodes": rows[:-1]},
                     {"complete": True, "episodes": rows[:-1] + [rows[0]]})
        for document in documents:
            with test_directory() as directory:
                output = Path(directory)
                evaluation.base.save_json(output / "episodes.json", document)
                with (patch.object(evaluation, "verify_protocol", return_value={"selected": CONFIGS}),
                      patch.object(evaluation, "evaluation_binding"), self.assertRaises(AssertionError)):
                    evaluation.summarize(output)
        rows[1]["initial"] = {"coverage": .1, "scene": 999}
        with test_directory() as directory:
            output = Path(directory)
            evaluation.base.save_json(output / "episodes.json", {"complete": True, "episodes": rows})
            with (patch.object(evaluation, "verify_protocol", return_value={"selected": CONFIGS}),
                  patch.object(evaluation, "evaluation_binding"), patch.object(evaluation, "verify_trial"),
                  self.assertRaises(AssertionError)):
                evaluation.summarize(output)

    def test_complete_summary_has_three_contrasts_and_monitor_inclusive_metrics(self):
        with test_directory() as directory:
            output = Path(directory)
            evaluation.base.save_json(output / "episodes.json", {"complete": True, "episodes": self.rows()})
            with (patch.object(evaluation, "verify_protocol", return_value={"selected": CONFIGS}),
                  patch.object(evaluation, "evaluation_binding"), patch.object(evaluation, "verify_trial") as verify,
                  redirect_stdout(io.StringIO())):
                result = evaluation.summarize(output)
            self.assertEqual(verify.call_count, 60)
            self.assertEqual(result["methods"]["adaptive12"]["successes"], 18)
            self.assertEqual(result["methods"]["weighted"]["successes"], 19)
            self.assertEqual(result["methods"]["weighted"]["mean_wall_seconds"], 90.)
            self.assertEqual(result["methods"]["weighted"]["mean_monitor_seconds"], 1.)
            self.assertEqual(set(result["paired"]), {"threshold_vs_adaptive12", "weighted_vs_adaptive12", "weighted_vs_threshold"})

    def test_full_run_has_exact_rotated_sixty_keys_and_one_policy_load(self):
        with test_directory() as directory:
            output = Path(directory)
            rows = {(r["seed"], r["method"]): r for r in self.rows()}
            expected = [(seed, method) for index, seed in enumerate(evaluation.SEEDS)
                        for method in evaluation.METHODS[index % 3:] + evaluation.METHODS[:index % 3]]

            def checked(policy, seed, method, directory, config, allowed_seeds):
                self.assertEqual(directory, output / "episodes")
                self.assertEqual(config, CONFIGS.get(method))
                self.assertEqual(allowed_seeds, evaluation.SEEDS)
                return rows[(seed, method)]

            with (patch.object(evaluation, "verify_protocol", return_value={"selected": CONFIGS}),
                  patch.object(evaluation.torch.cuda, "is_available", return_value=True),
                  patch.object(evaluation, "smoke"), patch.object(evaluation, "evaluation_binding"),
                  patch.object(evaluation.adaptive, "load_policy", return_value=(object(), {})) as load,
                  patch.object(evaluation, "checked_collect", side_effect=checked) as collect,
                  patch.object(evaluation, "summarize"), redirect_stdout(io.StringIO())):
                evaluation.run(output)
            self.assertEqual([(call.args[1], call.args[2]) for call in collect.call_args_list], expected)
            load.assert_called_once()
            self.assertTrue(evaluation.base.read_json(output / "episodes.json")["complete"])

    def test_completed_run_and_cpu_fallback_are_rejected(self):
        with test_directory() as directory:
            output = Path(directory)
            with (patch.object(evaluation, "verify_protocol", return_value={"selected": CONFIGS}),
                  patch.object(evaluation.torch.cuda, "is_available", return_value=False),
                  patch.object(evaluation.adaptive, "load_policy") as load):
                with self.assertRaises(AssertionError):
                    evaluation.run(output)
                evaluation.base.save_json(output / "summary.json", {"complete": True})
                with self.assertRaisesRegex(AssertionError, "Completed experiment"):
                    evaluation.run(output)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
