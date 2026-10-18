"""CPU-only tests of weight identities and unchanged three-way evaluation."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_recovery_evaluation as evaluation


def temporary():
    directory = Path(__file__).resolve().parents[1] / "tmp"
    directory.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="recovery-evaluation-tests-", dir=directory)


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.diffusion = torch.nn.Module()
        self.diffusion.unet = torch.nn.Module()
        self.diffusion.unet.final_conv = torch.nn.Linear(2, 2)


def context(directory):
    output = directory / "output"
    output.mkdir()
    policy = ToyPolicy()
    original = evaluation.training.tail_state(policy)
    trained = {name: value + 1. for name, value in original.items()}
    head_path = directory / "trained.pt"
    torch.save(trained, head_path)
    protocol = {
        "evaluation": {"scenes": evaluation.SCENES, "draw": evaluation.DRAW,
                       "arms": list(evaluation.ARMS), "max_steps": 300,
                       "rotation": "scene_index_mod_three", "success": "true_coverage_strictly_greater_than_0.95"},
        "milestone": evaluation.gen.MILESTONE,
        "checkpoint": {"weights_sha256": evaluation.base.WEIGHTS_SHA256, "config_sha256": "fake-config"},
        "fingerprints": {},
    }
    evaluation.base.save_json(output / "protocol.json", protocol)
    evaluation.base.save_json(output / "dataset.json", {"complete": True})
    report = {"complete": True, "protocol_sha256": evaluation.base.sha(output / "protocol.json"),
              "dataset_sha256": evaluation.base.sha(output / "dataset.json"),
              "final_head": {"path": head_path.as_posix(), "sha256": evaluation.base.sha(head_path),
                             "tensor_digest": evaluation.training.tail_digest(trained), "optimizer_step": 600}}
    evaluation.base.save_json(output / "training.json", report)
    bindings = {"inputs": {"final_head": report["final_head"]},
                "original_head_digest": evaluation.training.tail_digest(original)}
    evaluation.base.save_json(output / "eval-bindings.json", bindings)
    return output, policy, original, trained, protocol, report, bindings


class ScheduleTest(unittest.TestCase):
    def test_exact_fresh_twenty_by_three_identity_matrix(self):
        schedule = evaluation.schedule()
        self.assertEqual(evaluation.SCENES, list(range(150, 170)))
        self.assertEqual(len(schedule), 60)
        self.assertEqual({(r["scene"], r["draw"], r["arm"]) for r in schedule},
                         {(scene, 50000, arm) for scene in range(150, 170) for arm in evaluation.ARMS})

    def test_rotated_method_order_is_frozen_and_deterministic(self):
        arms = list(evaluation.ARMS)
        for index, scene in enumerate(evaluation.SCENES):
            self.assertEqual([r["arm"] for r in evaluation.schedule() if r["scene"] == scene],
                             arms[index % 3:] + arms[:index % 3])
        self.assertEqual(evaluation.schedule(), evaluation.schedule())

    def test_inner_method_mapping_keeps_both_milestone_controllers_identical(self):
        self.assertEqual(evaluation.ARMS, {"fixed8": "fixed8", "milestone12": "adaptive12", "recovery12": "adaptive12"})
        self.assertEqual(evaluation.STATISTICS["bootstrap_seed"], 20261016)

    def test_output_guard_preserves_old_evidence(self):
        for path in [Path("results"), Path("results/robotics-fresh-evaluation"), Path("../outside")]:
            with self.subTest(path=path), self.assertRaises(AssertionError):
                evaluation.check_output(path)

    def test_gpu_guard_delegates_to_training_resource_and_process_guard(self):
        with patch.object(evaluation.training, "gpu_guard") as guard:
            evaluation.gpu_guard()
            guard.assert_called_once_with()


class HeadSwitchTest(unittest.TestCase):
    def test_original_trained_original_roundtrip_preserves_backbone(self):
        policy = ToyPolicy()
        original = evaluation.training.tail_state(policy)
        changed = {name: value + 1 for name, value in original.items()}
        backbone = evaluation.training.backbone_digest(policy)
        for state in [original, changed, original]:
            evaluation.activate(policy, state, evaluation.training.tail_digest(state))
            self.assertEqual(evaluation.training.backbone_digest(policy), backbone)
            self.assertTrue(all(not m.training for m in policy.modules()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in policy.parameters()))

    def test_rejects_wrong_expected_digest(self):
        policy = ToyPolicy()
        with self.assertRaises(AssertionError):
            evaluation.activate(policy, evaluation.training.tail_state(policy), "bad")

    def test_partial_extra_and_reshaped_heads_are_rejected(self):
        policy = ToyPolicy()
        original = evaluation.training.tail_state(policy)
        first = next(iter(original))
        variants = [{k: v for k, v in original.items() if k != first},
                    {**original, "backbone.weight": torch.ones((2, 2))},
                    {**original, first: torch.ones(7)}]
        for state in variants:
            with self.subTest(keys=list(state)), self.assertRaises(AssertionError):
                evaluation.activate(policy, state, "unused")

    def test_outer_original_arms_share_digest_and_trained_arm_differs(self):
        bindings = {"original_head_digest": "original", "inputs": {"final_head": {"tensor_digest": "trained"}}}
        self.assertEqual(evaluation.head_identity(bindings, "fixed8"), "original")
        self.assertEqual(evaluation.head_identity(bindings, "milestone12"), "original")
        self.assertEqual(evaluation.head_identity(bindings, "recovery12"), "trained")
        with self.assertRaises(AssertionError):
            evaluation.head_identity(bindings, "staged15")


class InputBindingTest(unittest.TestCase):
    def read_inputs(self, output, protocol):
        real_sha = evaluation.base.sha
        config = evaluation.base.MODEL / "config.json"
        with patch.object(evaluation, "check_output"), patch.object(evaluation.pilot, "verify", return_value=protocol), \
                patch.object(evaluation.base, "sha", side_effect=lambda path: "fake-config" if Path(path) == config else real_sha(path)):
            return evaluation.input_document(output)

    def test_final_600_artifact_is_bound_to_protocol_dataset_and_head_file(self):
        with temporary() as name:
            output, _, _, _, protocol, report, _ = context(Path(name))
            document = self.read_inputs(output, protocol)
            self.assertEqual(document["final_head"], report["final_head"])
            self.assertEqual(document["schedule"], evaluation.schedule())

    def test_incomplete_wrong_step_stale_dataset_or_stale_protocol_are_rejected(self):
        with temporary() as name:
            output, _, _, _, protocol, report, _ = context(Path(name))
            changes = [{**report, "complete": False},
                       {**report, "final_head": {**report["final_head"], "optimizer_step": 400}},
                       {**report, "dataset_sha256": "old"}, {**report, "protocol_sha256": "old"}]
            for candidate in changes:
                evaluation.base.save_json(output / "training.json", candidate)
                with self.subTest(candidate=candidate), self.assertRaises(AssertionError):
                    self.read_inputs(output, protocol)

    def test_changed_head_file_is_rejected_before_evaluation(self):
        with temporary() as name:
            output, _, _, _, protocol, report, _ = context(Path(name))
            torch.save({"different": torch.ones(1)}, report["final_head"]["path"])
            with self.assertRaises(AssertionError):
                self.read_inputs(output, protocol)

    def test_changed_test_scene_or_milestone_is_rejected(self):
        with temporary() as name:
            output, _, _, _, protocol, _, _ = context(Path(name))
            for change in [{**protocol, "evaluation": {**protocol["evaluation"], "scenes": [54, 59]}},
                           {**protocol, "milestone": "not-the-designated-milestone"}]:
                with self.subTest(change=change), self.assertRaises(AssertionError):
                    self.read_inputs(output, change)

    def test_cannot_retroactively_label_preexisting_episode_weights(self):
        with temporary() as name:
            output, _, _, _, _, _, bindings = context(Path(name))
            directory = output / "evaluation/recovery12"
            evaluation.base.save_json(directory / "orphan.json", {})
            with self.assertRaises(AssertionError):
                evaluation.condition_binding(output, "recovery12", directory, bindings)

    def test_changed_condition_weight_binding_is_rejected(self):
        with temporary() as name:
            output, _, _, _, _, _, bindings = context(Path(name))
            directory = output / "evaluation/recovery12"
            saved = evaluation.condition_binding(output, "recovery12", directory, bindings)
            evaluation.base.save_json(directory / "model-binding.json", {**saved, "head_digest": "original"})
            with self.assertRaises(AssertionError):
                evaluation.condition_binding(output, "recovery12", directory, bindings)


class CollectionTest(unittest.TestCase):
    def test_smoke_with_loaded_policy_never_repeats_gpu_memory_guard(self):
        with temporary() as name:
            output, policy, _, _, _, _, bindings = context(Path(name))
            with patch.object(evaluation, "gpu_guard") as guard, \
                    patch.object(evaluation.adaptive, "load_policy") as loader, \
                    patch.object(evaluation, "prepare_bindings", return_value=bindings), \
                    patch.object(evaluation, "collect", return_value={}), \
                    patch.object(evaluation, "verify_smoke", return_value={"passed": True}):
                self.assertTrue(evaluation.smoke(output, policy)["passed"])
            guard.assert_not_called()
            loader.assert_not_called()

    def test_each_new_trial_uses_correct_head_and_unchanged_inner_method(self):
        with temporary() as name:
            output, policy, original, trained, _, _, bindings = context(Path(name))
            heads = {"original": original, "trained": trained}
            seen = []
            def evaluator(model, scene, draw, method, directory, max_steps):
                seen.append((method, evaluation.training.tail_digest(evaluation.training.tail_state(model))))
                return {"scene": scene, "draw": draw, "method": method}
            with patch.object(evaluation.gen, "evaluate_episode", side_effect=evaluator), patch.object(evaluation, "verify_record"):
                for arm in evaluation.ARMS:
                    record = evaluation.collect(policy, 150, 50000, arm, output / "evaluation" / arm,
                                                output, bindings, heads)
                    self.assertEqual(record["arm"], arm)
            self.assertEqual(seen, [(evaluation.ARMS[arm], evaluation.head_identity(bindings, arm)) for arm in evaluation.ARMS])

    def test_resume_reads_completed_inner_sidecar_without_inference_or_head_switch(self):
        with temporary() as name:
            output, policy, original, trained, _, _, bindings = context(Path(name))
            directory = output / "evaluation/recovery12"
            evaluation.condition_binding(output, "recovery12", directory, bindings)
            trial = {"scene": 150, "draw": 50000, "method": "adaptive12"}
            evaluation.base.save_json(directory / (evaluation.gen.stem(150, 50000, "adaptive12") + ".json"), trial)
            with patch.object(evaluation.gen, "evaluate_episode") as model_call, patch.object(evaluation, "activate") as activate, \
                    patch.object(evaluation, "verify_record"):
                record = evaluation.collect(policy, 150, 50000, "recovery12", directory, output, bindings,
                                            {"original": original, "trained": trained})
            model_call.assert_not_called()
            activate.assert_not_called()
            self.assertEqual(record["trial"], trial)

    def test_orphan_trace_is_not_overwritten(self):
        with temporary() as name:
            output, policy, original, trained, _, _, bindings = context(Path(name))
            directory = output / "evaluation/fixed8"
            evaluation.condition_binding(output, "fixed8", directory, bindings)
            (directory / (evaluation.gen.stem(150, 50000, "fixed8") + ".trace.json.gz")).touch()
            with patch.object(evaluation.gen, "evaluate_episode") as model_call, self.assertRaises(AssertionError):
                evaluation.collect(policy, 150, 50000, "fixed8", directory, output, bindings,
                                   {"original": original, "trained": trained})
            model_call.assert_not_called()


def artificial_records():
    records = []
    for scene in evaluation.SCENES:
        for arm in evaluation.ARMS:
            records.append({"arm": arm, "trial": {"scene": scene, "draw": 50000,
                "success": (scene % 2 == 0 if arm != "recovery12" else True),
                "final_coverage": .96 if arm == "recovery12" else .90,
                "model_queries": 10 if arm == "recovery12" else 12,
                "wall_seconds": 51. if arm == "recovery12" else 60.,
                "monitor_seconds": 1. if arm != "fixed8" else 0.,
                "prediction_seconds": [5.] * (10 if arm == "recovery12" else 12)}})
    return records


class StatisticsTest(unittest.TestCase):
    def test_metric_table_uses_all_twenty_scenes_and_all_failures(self):
        table = evaluation.metric_table(artificial_records())
        self.assertEqual(table["recovery12"]["successes"], 20)
        self.assertEqual(table["fixed8"]["successes"], 10)
        self.assertEqual(table["fixed8"]["mean_wall_seconds"], 60.)
        self.assertEqual(table["recovery12"]["mean_prediction_seconds"], 5.)

    def test_duplicate_or_missing_condition_cannot_be_summarized(self):
        records = artificial_records()
        for variant in [records[:-1], records[:-1] + [records[0]]]:
            with self.assertRaises(AssertionError):
                evaluation.metric_table(variant)

    def test_paired_differences_and_discordant_success_arithmetic(self):
        result = evaluation.paired(artificial_records(), "milestone12")
        self.assertEqual(result["gained_successes"], list(range(151, 170, 2)))
        self.assertEqual(result["lost_successes"], [])
        self.assertEqual(result["differences_method_minus_reference"]["success"]["mean_difference"], .5)
        self.assertEqual(result["differences_method_minus_reference"]["model_queries"]["mean_difference"], -2.)
        self.assertEqual(result["exact_two_sided_discordant_success_p"], 2. / 1024.)
        self.assertEqual(result, evaluation.paired(artificial_records(), "milestone12"))

    def test_primary_and_secondary_references_are_explicit(self):
        for reference in ["milestone12", "fixed8"]:
            self.assertEqual(evaluation.paired(artificial_records(), reference)["reference"], reference)
        with self.assertRaises(AssertionError):
            evaluation.paired(artificial_records(), "recovery12")


class SmokeVerificationTest(unittest.TestCase):
    def fixture(self):
        initial = {"coverage": .1, "x": 0}
        baseline = [{"action": [float(i), float(i + 1)], "after": initial} for i in range(16)]
        records = [{"label": label, "arm": arm, "trial": {"initial": initial, "unit_lengths": [8, 8]}}
                   for label, arm in [("before", "fixed8"), ("milestone", "milestone12"),
                                      ("trained", "recovery12"), ("restored", "fixed8")]]
        traces = {label: {"steps": copy.deepcopy(baseline)} for label in ["before", "milestone", "trained", "restored"]}
        for row in traces["trained"]["steps"]:
            row["action"] = [999., 999.]
        saved = {"passed": True, "original_backbone_unchanged": True, "records": records}
        return saved, traces, baseline

    def verify(self, saved, traces, baseline):
        with patch.object(evaluation, "verify_bindings", return_value={}), \
                patch.object(evaluation, "verify_record", side_effect=lambda record, *a: traces[record["label"]]), \
                patch.object(evaluation.base, "read_json", side_effect=lambda path: saved if Path(path).name == "smoke.json" else {"traces": {"fixed8": baseline}}):
            return evaluation.verify_smoke(Path("unused"))

    def test_finetuned_actions_may_differ_but_original_roundtrip_must_match(self):
        saved, traces, baseline = self.fixture()
        self.assertTrue(self.verify(saved, traces, baseline)["passed"])
        traces["restored"]["steps"][8]["action"] = [999., 999.]
        with self.assertRaises(AssertionError):
            self.verify(saved, traces, baseline)

    def test_original_control_prefix_must_match_old_saved_baseline(self):
        saved, traces, baseline = self.fixture()
        altered = copy.deepcopy(baseline)
        altered[0]["action"] = [900., 900.]
        with self.assertRaises(AssertionError):
            self.verify(saved, traces, altered)


if __name__ == "__main__":
    unittest.main()
