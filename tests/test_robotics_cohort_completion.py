"""CPU-only contracts for completing baselines without rerunning adaptive trials."""

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
import robotics_cohort_completion as completion
from test_robotics_adaptive_length import FakeEnvironment, FakePolicy, fake_physical_state


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="cohort-completion-cpu-", dir=parent)


def state(step=0, coverage=.60):
    return {"coverage": coverage, "block_position": [100., 200.],
            "block_angle": .2, "agent_position": [step, 10.],
            "image_sha256": f"image-{step}"}


def trial_fixture(directory, method="fixed8", seed=70, steps=16, success=True):
    """Construct a valid queue trace without simulator, model or CUDA access."""
    horizon = 8 if method == "fixed8" else 12
    rows = []
    for index in range(steps):
        age = index % horizon + 1
        coverage = .96 if success and index == steps - 1 else .60
        rows.append({"step": index + 1, "action": [float(index), 100.],
                     "new_prediction": age == 1, "plan_age": age,
                     "pending_before": horizon - age, "pending_after": horizon - age,
                     "after": state(index + 1, coverage)})
    lengths = [min(horizon, steps - index) for index in range(0, steps, horizon)]
    trial = {"seed": seed, "method": method, "initial": state(),
             "steps": steps, "success": success, "truncated": not success,
             "final_coverage": rows[-1]["after"]["coverage"],
             "model_queries": len(lengths), "unit_lengths": lengths,
             "events": [], "monitor_seconds": 0., "wall_seconds": 10.,
             "trace": f"seed-{seed}-{method}.trace.json.gz"}
    trace = {"initial": state(), "steps": rows}
    write_trial(directory, trial, trace)
    return trial, trace


def write_trial(directory, trial, trace):
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / trial["trace"], "wt", encoding="utf-8") as stream:
        json.dump(trace, stream)
    trial["trace_sha256"] = completion.base.sha(directory / trial["trace"])
    completion.base.save_json(directory / f"seed-{trial['seed']}-{trial['method']}.json", trial)


def reference_fixture(directory, trial, trace):
    reference = {**trial, "method": "adaptive12",
                 "trace": f"seed-{trial['seed']}-adaptive12.trace.json.gz"}
    write_trial(directory, reference, trace)
    return completion.entry(reference, directory / f"seed-{trial['seed']}-adaptive12.json")


class BaselineContractTest(unittest.TestCase):
    def test_frozen_collector_has_exact_eight_and_twelve_queue_boundaries(self):
        for method, lengths in (("fixed8", [8, 8]), ("fixed12", [12, 4])):
            with self.subTest(method=method), test_directory() as name:
                directory, policy, env = Path(name), FakePolicy(), FakeEnvironment()
                with (patch.object(completion.adaptive.gym, "make", return_value=env),
                      patch.object(completion.adaptive, "observation_batch", side_effect=lambda obs, device: obs),
                      patch.object(completion.adaptive, "physical_state", side_effect=fake_physical_state),
                      patch.object(completion.adaptive.torch.cuda, "synchronize")):
                    trial = completion.adaptive.evaluate_episode(policy, 70, method, directory)
                trace = completion.verify_baseline(trial, directory)
                self.assertTrue(env.closed)
                self.assertEqual(policy.resets, 1)
                self.assertEqual(trial["unit_lengths"], lengths)
                self.assertEqual(trial["model_queries"], 2)
                self.assertEqual([r["plan_age"] for r in trace["steps"][:8]], list(range(1, 9)))
                self.assertEqual(trial["events"], [])
                self.assertEqual(trial["monitor_seconds"], 0.)

    def test_timeout_requires_three_hundred_movements_and_exact_success_threshold(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace = trial_fixture(directory, steps=300, success=False)
            trace["steps"][-1]["after"]["coverage"] = .95
            trial["final_coverage"] = .95
            write_trial(directory, trial, trace)
            completion.verify_baseline(trial, directory)
            for changes in ({"success": True}, {"truncated": False}, {"steps": 299}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    completion.verify_baseline({**trial, **changes}, directory)
            short, _ = trial_fixture(directory, steps=16, success=False)
            with self.assertRaises(AssertionError):
                completion.verify_baseline(short, directory)

    def test_queue_age_query_unit_and_trace_mutations_are_rejected(self):
        changes = (lambda trial, trace: trace["steps"][8].update(plan_age=8),
                   lambda trial, trace: trace["steps"][8].update(new_prediction=False),
                   lambda trial, trace: trace["steps"][0].update(pending_before=0),
                   lambda trial, trace: trace["steps"][0].update(pending_after=0),
                   lambda trial, trace: trace["steps"][0].update(step=2),
                   lambda trial, trace: trial.update(model_queries=99),
                   lambda trial, trace: trial.update(unit_lengths=[7, 9]),
                   lambda trial, trace: trace["steps"][0].update(features={"coverage": .5}),
                   lambda trial, trace: trace["steps"][0].update(predicted_commands=[]),
                   lambda trial, trace: trace["initial"].update(block_angle=.3))
        for change in changes:
            with test_directory() as name:
                directory = Path(name)
                trial, trace = trial_fixture(directory)
                change(trial, trace)
                write_trial(directory, trial, trace)
                with self.assertRaises(AssertionError):
                    completion.verify_baseline(trial, directory)

    def test_events_monitor_wrong_identity_and_nonfinite_cost_are_rejected(self):
        with test_directory() as name:
            directory = Path(name)
            trial, _ = trial_fixture(directory)
            for changes in ({"events": [{"kind": "extend"}]}, {"monitor_seconds": .01},
                            {"method": "adaptive12"}, {"seed": 69}, {"seed": 90},
                            {"wall_seconds": float("inf")}, {"wall_seconds": -.1},
                            {"final_coverage": .99}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    completion.verify_baseline({**trial, **changes}, directory)

    def test_action_shape_and_nonfinite_values_are_rejected(self):
        for action in ([1.], [1., 2., 3.], [float("nan"), 2.], [float("inf"), 2.]):
            with self.subTest(action=action), test_directory() as name:
                directory = Path(name)
                trial, trace = trial_fixture(directory)
                trace["steps"][0]["action"] = action
                write_trial(directory, trial, trace)
                with self.assertRaises(AssertionError):
                    completion.verify_baseline(trial, directory)

    def test_first_successful_observation_must_be_terminal(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace = trial_fixture(directory)
            trace["steps"][5]["after"]["coverage"] = .951
            write_trial(directory, trial, trace)
            with self.assertRaises(AssertionError):
                completion.verify_baseline(trial, directory)

    def test_wrong_trace_path_or_hash_cannot_resume(self):
        with test_directory() as name:
            directory = Path(name)
            trial, _ = trial_fixture(directory)
            for changes in ({"trace": "../other.trace.json.gz"}, {"trace_sha256": "0" * 64}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    completion.verify_baseline({**trial, **changes}, directory)


class PrefixAndResumeTest(unittest.TestCase):
    def test_common_prefix_is_exact_except_existing_truth_coverage_tolerance(self):
        with test_directory() as name:
            trial, trace = trial_fixture(Path(name))
            reference, other = copy.deepcopy(trial), copy.deepcopy(trace)
            reference["method"] = "adaptive12"
            reference["initial"]["coverage"] += 5e-13
            for row in other["steps"][:8]:
                row["after"]["coverage"] += 5e-13
            completion.common_prefix(trial, trace, reference, other)
            # Later movements legitimately diverge after the shared first eight.
            other["steps"][8]["action"] = [-1., -1.]
            completion.common_prefix(trial, trace, reference, other)
            reference["initial"]["coverage"] += 2e-12
            with self.assertRaises(AssertionError):
                completion.common_prefix(trial, trace, reference, other)

    def test_common_prefix_rejects_changed_action_physics_rgb_initial_or_seed(self):
        mutations = (lambda ref, trace: ref.update(seed=71),
                     lambda ref, trace: ref["initial"].update(block_angle=.21),
                     lambda ref, trace: trace["steps"][3]["action"].__setitem__(0, 3.00000000001),
                     lambda ref, trace: trace["steps"][3]["after"].update(block_angle=.21),
                     lambda ref, trace: trace["steps"][3]["after"].update(image_sha256="different"),
                     lambda ref, trace: trace["steps"].__delitem__(slice(7, None)))
        for mutation in mutations:
            with test_directory() as name:
                trial, trace = trial_fixture(Path(name))
                reference, other = copy.deepcopy(trial), copy.deepcopy(trace)
                mutation(reference, other)
                with self.assertRaises(AssertionError):
                    completion.common_prefix(trial, trace, reference, other)

    def test_saved_baseline_is_reused_without_policy_inference(self):
        with test_directory() as name:
            output = Path(name)
            trial, trace = trial_fixture(output / "episodes")
            reference = reference_fixture(output / "reused", trial, trace)
            with patch.object(completion.adaptive, "evaluate_episode") as evaluate:
                result = completion.checked_collect(object(), 70, "fixed8", output, reference)
            evaluate.assert_not_called()
            self.assertEqual(result, completion.entry(trial, output / "episodes" / "seed-70-fixed8.json"))

    def test_orphan_trace_prevents_inference(self):
        with test_directory() as name:
            output = Path(name)
            (output / "episodes").mkdir()
            (output / "episodes" / "seed-70-fixed8.trace.json.gz").write_bytes(b"partial")
            with patch.object(completion.adaptive, "evaluate_episode") as evaluate:
                with self.assertRaisesRegex(AssertionError, "orphan trace"):
                    completion.checked_collect(object(), 70, "fixed8", output, {})
            evaluate.assert_not_called()

    def test_adaptive_rejected_filters_and_out_of_range_seeds_never_collect(self):
        with test_directory() as name:
            with patch.object(completion.adaptive, "evaluate_episode") as evaluate:
                for seed, method in ((70, "adaptive12"), (70, "threshold"), (70, "weighted"),
                                     (69, "fixed8"), (90, "fixed12")):
                    with self.subTest(seed=seed, method=method), self.assertRaises(AssertionError):
                        completion.checked_collect(object(), seed, method, Path(name), {})
            evaluate.assert_not_called()


class ProtocolTest(unittest.TestCase):
    def fixture(self, root):
        reused, old = root / "reused", root / "original"
        artifact = root / "frozen-source.txt"
        artifact.write_text("unchanged", encoding="utf-8")
        packages = {name: "test-version" for name in completion.adaptive.PACKAGES}
        protocol = {"fingerprints": {artifact.as_posix(): completion.base.sha(artifact)},
                    "rules": dict(completion.adaptive.RULES), "packages": packages,
                    "checkpoint": {"model_id": completion.base.MODEL_ID,
                                   "revision": completion.base.MODEL_REVISION,
                                   "weights_sha256": completion.base.WEIGHTS_SHA256}}
        prefix = root / "prefix-proof"
        completion.base.save_json(prefix / "prefix-smoke.json", {"passed": True})
        records = []
        for location, seeds in ((reused, completion.SEEDS), (old, list(range(50, 70)))):
            for filename in ("protocol.json", "episodes.json", "summary.json", "independent-audit.json"):
                completion.base.save_json(location / filename, protocol if filename == "protocol.json" else {"passed": True})
            for seed in seeds:
                sidecar = location / "episodes" / f"seed-{seed}-adaptive12.json"
                completion.base.save_json(sidecar, {"seed": seed})
                trace = sidecar.with_suffix(".trace.json.gz")
                trace.write_bytes(b"frozen trace")
                records.append({"seed": seed, "source_sidecar": sidecar.as_posix(),
                                "trace_directory": sidecar.parent.as_posix(), "trace": trace.name})
        stack = ExitStack()
        stack.enter_context(patch.object(completion, "REUSED", reused))
        stack.enter_context(patch.object(completion.fresh, "OUTPUT", old))
        stack.enter_context(patch.object(completion.adaptive, "OUTPUT", prefix))
        stack.enter_context(patch.object(completion, "reused_adaptive", return_value=records[:20]))
        stack.enter_context(patch.object(completion, "original_cohort", return_value=records[20:]))
        stack.enter_context(patch.object(completion.importlib.metadata, "version", return_value="test-version"))
        stack.enter_context(patch.object(completion, "check_output"))
        return stack, artifact, records

    def test_output_guard_and_exact_new_trial_scope(self):
        completion.check_output(completion.OUTPUT)
        completion.check_output(completion.OUTPUT.with_name(completion.OUTPUT.name + "-repeat"))
        for output in (Path("results"), completion.REUSED, completion.fresh.OUTPUT,
                       completion.OUTPUT / "nested", Path("../outside")):
            with self.subTest(output=output), self.assertRaises(ValueError):
                completion.check_output(output)
        self.assertEqual(completion.SEEDS, list(range(70, 90)))
        self.assertEqual(completion.BASELINES, ["fixed8", "fixed12"])
        self.assertEqual(len({(s, m) for s in completion.SEEDS for m in completion.BASELINES}), 40)

    def test_protocol_binds_reused_sidecars_traces_sources_and_is_idempotent(self):
        with test_directory() as name:
            root = Path(name)
            stack, artifact, records = self.fixture(root)
            with stack:
                output = root / "new"
                first = completion.prepare(output)
                encoded = (output / "protocol.json").read_bytes()
                self.assertEqual(first["new_scored_attempts"], 40)
                self.assertEqual(first["reused_adaptive_attempts"], 20)
                self.assertFalse(first["tuning"])
                self.assertFalse(first["rejected_gate_reruns"])
                self.assertEqual(first["milestone"], completion.MILESTONE)
                for record in records:
                    self.assertIn(record["source_sidecar"], first["fingerprints"])
                    self.assertIn((Path(record["trace_directory"]) / record["trace"]).as_posix(), first["fingerprints"])
                self.assertIn(artifact.as_posix(), first["fingerprints"])
                self.assertEqual(completion.prepare(output), first)
                self.assertEqual((output / "protocol.json").read_bytes(), encoded)

    def test_changed_frozen_source_or_protocol_cannot_resume(self):
        for changed in ("hash", "rules", "seeds", "new_methods", "checkpoint", "statistics", "tuning"):
            with self.subTest(changed=changed), test_directory() as name:
                root = Path(name)
                stack, artifact, _ = self.fixture(root)
                with stack:
                    output = root / "new"
                    document = completion.prepare(output)
                    if changed == "hash":
                        artifact.write_text("changed", encoding="utf-8")
                    else:
                        mutations = {"rules": {}, "seeds": [70], "new_methods": ["adaptive12"],
                                     "checkpoint": {"weights_sha256": "wrong"}, "statistics": {}, "tuning": True}
                        document[changed] = mutations[changed]
                        completion.base.save_json(output / "protocol.json", document)
                    with self.assertRaises(AssertionError):
                        completion.verify_protocol(output)

    def test_unbound_existing_outputs_are_not_adopted(self):
        for existing in ("episodes", "episodes.json", "summary.json"):
            with self.subTest(existing=existing), test_directory() as name:
                output = Path(name)
                if existing == "episodes":
                    (output / existing).mkdir()
                else:
                    completion.base.save_json(output / existing, {})
                with patch.object(completion, "check_output"), patch.object(completion, "protocol_document") as make:
                    with self.assertRaises(AssertionError):
                        completion.prepare(output)
                make.assert_not_called()


class StatisticsAndRunTest(unittest.TestCase):
    def rows(self, seeds=completion.SEEDS):
        return [{"seed": seed, "method": method,
                 "success": (seed >= 72 if method == "fixed8" else seed != 72),
                 "final_coverage": .9 + .02 * (method == "adaptive12"),
                 "model_queries": 20 - 4 * (method == "adaptive12"),
                 "wall_seconds": 100. - 10. * (method == "adaptive12"),
                 "monitor_seconds": 1. if method == "adaptive12" else 0.}
                for seed in seeds for method in completion.METHODS]

    def test_paired_gains_losses_exact_test_and_bootstrap_are_reproducible(self):
        rows = self.rows()
        result = completion.paired(rows, "fixed8", "adaptive12", completion.SEEDS)
        self.assertEqual(result["gained_success_seeds"], [70, 71])
        self.assertEqual(result["lost_success_seeds"], [72])
        self.assertEqual(result["exact_two_sided_discordant_success_p"], 1.)
        differences = result["differences_method_minus_reference"]
        self.assertAlmostEqual(differences["success"]["mean_difference"], .05)
        self.assertEqual(differences["model_queries"]["paired_bootstrap_95_interval"], [-4., -4.])
        self.assertNotIn("wall_seconds", differences)
        self.assertEqual(result["descriptive_runtime_difference"], -10.)
        self.assertEqual(completion.paired(list(reversed(rows)), "fixed8", "adaptive12", completion.SEEDS), result)
        all_gained = copy.deepcopy(rows)
        for row in all_gained:
            row["success"] = row["method"] != "fixed8" if row["seed"] < 74 else False
        gained = completion.paired(all_gained, "fixed8", "adaptive12", completion.SEEDS)
        self.assertEqual(gained["exact_two_sided_discordant_success_p"], .125)

    def test_missing_paired_scene_and_duplicate_or_missing_table_keys_are_rejected(self):
        rows = self.rows()
        with self.assertRaises(AssertionError):
            completion.paired(rows[1:], "fixed8", "adaptive12", completion.SEEDS)
        with self.assertRaises(AssertionError):
            completion.paired(rows + [rows[0]], "fixed8", "adaptive12", completion.SEEDS)
        for invalid in (rows[:-1], rows[:-1] + [rows[0]], rows + [rows[0]]):
            with self.assertRaises(AssertionError):
                completion.metric_table(invalid, completion.SEEDS)

    def test_table_pools_forty_scenes_and_preserves_monitor_inclusive_descriptive_runtime(self):
        seeds = list(range(50, 90))
        result = completion.metric_table(self.rows(seeds), seeds)
        self.assertEqual(set(result["methods"]), set(completion.METHODS))
        self.assertTrue(all(row["attempts"] == 40 for row in result["methods"].values()))
        self.assertEqual(result["methods"]["adaptive12"]["mean_wall_seconds"], 90.)
        self.assertEqual(result["methods"]["adaptive12"]["mean_monitor_seconds"], 1.)
        self.assertEqual(set(result["paired"]), {"fixed12_vs_fixed8", "adaptive12_vs_fixed8", "adaptive12_vs_fixed12"})

    def test_run_collects_only_forty_alternating_baselines_and_reuses_twenty_adaptive(self):
        with test_directory() as name:
            output = Path(name)
            reused = [r for r in self.rows() if r["method"] == "adaptive12"]
            rows = {(r["seed"], r["method"]): r for r in self.rows()}
            expected = [(seed, method) for index, seed in enumerate(completion.SEEDS)
                        for method in (completion.BASELINES if index % 2 == 0 else completion.BASELINES[::-1])]

            def collect(policy, seed, method, destination, reference):
                self.assertEqual(destination, output)
                self.assertEqual(reference["seed"], seed)
                self.assertEqual(reference["method"], "adaptive12")
                return rows[(seed, method)]

            with (patch.object(completion, "verify_protocol"),
                  patch.object(completion.torch.cuda, "is_available", return_value=True),
                  patch.object(completion, "reused_adaptive", return_value=reused),
                  patch.object(completion.adaptive, "load_policy", return_value=(object(), {})) as load,
                  patch.object(completion, "checked_collect", side_effect=collect) as checked,
                  patch.object(completion, "summarize"), redirect_stdout(io.StringIO())):
                completion.run(output)
            self.assertEqual([(call.args[1], call.args[2]) for call in checked.call_args_list], expected)
            load.assert_called_once()
            document = completion.base.read_json(output / "episodes.json")
            self.assertTrue(document["complete"])
            self.assertEqual(document["new_baselines_complete"], 40)
            self.assertEqual(len(document["episodes"]), 60)
            self.assertEqual(document["episodes"][:20], reused)

    def test_completed_comparison_and_cpu_fallback_prevent_policy_load(self):
        with test_directory() as name:
            output = Path(name)
            with (patch.object(completion, "verify_protocol"),
                  patch.object(completion.torch.cuda, "is_available", return_value=False),
                  patch.object(completion.adaptive, "load_policy") as load):
                with self.assertRaises(AssertionError):
                    completion.run(output)
                completion.base.save_json(output / "summary.json", {"complete": True})
                with self.assertRaisesRegex(AssertionError, "Completed comparison"):
                    completion.run(output)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
