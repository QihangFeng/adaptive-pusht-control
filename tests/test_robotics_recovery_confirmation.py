"""CPU-only checks for the frozen forty-scene, two-draw confirmation."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_recovery_confirmation as confirmation


def temporary():
    directory = Path(__file__).resolve().parents[1] / "tmp"
    directory.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="recovery-confirmation-tests-", dir=directory)


class ToyPolicy(torch.nn.Module):
    """Exactly six output-block tensors plus a deliberately separate backbone."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.diffusion = torch.nn.Module()
        self.diffusion.unet = torch.nn.Module()
        self.diffusion.unet.final_conv = torch.nn.Sequential(
            torch.nn.Conv1d(2, 2, 3), torch.nn.GroupNorm(1, 2), torch.nn.Conv1d(2, 2, 1))


def context(directory):
    output = directory / "output"
    output.mkdir()
    policy = ToyPolicy()
    original = confirmation.training.tail_state(policy)
    trained = {key: value + 1. for key, value in original.items()}
    head_path = directory / "trained.pt"
    torch.save(trained, head_path)
    bindings = {
        "inputs": {"final_head": {"path": head_path.as_posix(),
                                  "sha256": confirmation.base.sha(head_path),
                                  "tensor_digest": confirmation.training.tail_digest(trained),
                                  "optimizer_step": 600}},
        "original_head_digest": confirmation.training.tail_digest(original)}
    confirmation.base.save_json(output / "eval-bindings.json", bindings)
    return output, policy, original, trained, bindings


def artificial_records():
    records = []
    for item in confirmation.schedule():
        arm, scene, draw = item["arm"], item["scene"], item["draw"]
        trained = arm == "recovery12"
        queries = 10 if trained else 12
        records.append({"arm": arm, "trial": {
            "scene": scene, "draw": draw, "method": confirmation.ARMS[arm],
            "success": True if trained else scene % 2 == 0,
            "final_coverage": .96 if trained else .90,
            "model_queries": queries, "wall_seconds": 51. if trained else 60.,
            "monitor_seconds": 0. if arm == "fixed8" else 1.,
            "prediction_seconds": [5.] * queries}})
    return records


class ScheduleTest(unittest.TestCase):
    def test_forty_distinct_fresh_scenes_two_draws_three_arms(self):
        self.assertEqual(confirmation.SCENES, list(range(170, 210)))
        self.assertEqual(confirmation.DRAWS, [60000, 70000])
        rows = confirmation.schedule()
        self.assertEqual(len(rows), 240)
        expected = {(scene, draw, arm) for scene in range(170, 210)
                    for draw in [60000, 70000] for arm in confirmation.ARMS}
        self.assertEqual({(row["scene"], row["draw"], row["arm"]) for row in rows}, expected)
        for arm in confirmation.ARMS:
            self.assertEqual(sum(row["arm"] == arm for row in rows), 80)

    def test_rotation_balances_order_across_both_draws_and_scenes(self):
        names = list(confirmation.ARMS)
        rows = confirmation.schedule()
        for index, scene in enumerate(confirmation.SCENES):
            for draw_index, draw in enumerate(confirmation.DRAWS):
                offset = (2 * index + draw_index) % 3
                observed = [row["arm"] for row in rows if row["scene"] == scene and row["draw"] == draw]
                self.assertEqual(observed, names[offset:] + names[:offset])
        self.assertEqual(rows, confirmation.schedule())

    def test_inner_methods_do_not_modify_milestone_or_generation_length(self):
        self.assertEqual(confirmation.ARMS, {"fixed8": "fixed8", "milestone12": "adaptive12",
                                             "recovery12": "adaptive12"})
        self.assertEqual(confirmation.STATISTICS["bootstrap_samples"], 10000)
        self.assertEqual(confirmation.STATISTICS["bootstrap_seed"], 20261017)

    def test_output_guard_rejects_existing_pilot_and_other_evidence(self):
        for path in [Path("results"), Path("results/robotics-recovery-pilot"),
                     Path("results/robotics-fresh-evaluation"), Path("../outside")]:
            with self.subTest(path=path), self.assertRaises(AssertionError):
                confirmation.check_output(path)

    def test_only_new_confirmation_output_is_accepted(self):
        confirmation.check_output(confirmation.OUTPUT)


class HeadSwitchTest(unittest.TestCase):
    def test_only_six_final_block_tensors_switch_and_backbone_is_unchanged(self):
        policy = ToyPolicy()
        original = confirmation.training.tail_state(policy)
        self.assertEqual(len(original), 6)
        changed = {name: value + 1. for name, value in original.items()}
        backbone = confirmation.training.backbone_digest(policy)
        for head in [original, changed, original]:
            confirmation.activate(policy, head, confirmation.training.tail_digest(head))
            self.assertEqual(confirmation.training.backbone_digest(policy), backbone)
            self.assertTrue(all(not module.training for module in policy.modules()))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in policy.parameters()))
            for name, value in confirmation.training.tail_state(policy).items():
                self.assertTrue(torch.equal(value, head[name]))

    def test_expected_head_digest_is_mandatory(self):
        policy = ToyPolicy()
        with self.assertRaises(AssertionError):
            confirmation.activate(policy, confirmation.training.tail_state(policy), "wrong")

    def test_missing_extra_or_wrong_shape_output_tensors_are_rejected(self):
        policy = ToyPolicy()
        original = confirmation.training.tail_state(policy)
        first = next(iter(original))
        cases = [{name: value for name, value in original.items() if name != first},
                 {**original, "backbone.weight": torch.ones(2, 2)},
                 {**original, first: torch.ones(1)}]
        for head in cases:
            with self.subTest(keys=list(head)), self.assertRaises(AssertionError):
                confirmation.activate(policy, head, "unused")

    def test_controls_share_original_head_and_candidate_uses_trained_head(self):
        bindings = {"original_head_digest": "original", "inputs": {"final_head": {"tensor_digest": "trained"}}}
        self.assertEqual(confirmation.head_identity(bindings, "fixed8"), "original")
        self.assertEqual(confirmation.head_identity(bindings, "milestone12"), "original")
        self.assertEqual(confirmation.head_identity(bindings, "recovery12"), "trained")
        with self.assertRaises(AssertionError):
            confirmation.head_identity(bindings, "staged15")


def protocol_fixture():
    return {"previous_inputs": {"final_head": {"path": "unused-head.pt", "sha256": "head-sha",
                                                "tensor_digest": "trained", "optimizer_step": 600}},
            "original_head_digest": "original", "fingerprints": {"runner.py": "source-sha"},
            "smoke": {"order": confirmation.SMOKE_ARMS}}


class ProtocolTest(unittest.TestCase):
    def test_cpu_prepare_freezes_protocol_and_never_loads_model(self):
        with temporary() as name:
            output = Path(name) / "output"
            with patch.object(confirmation, "check_output"), \
                    patch.object(confirmation, "protocol_document", side_effect=protocol_fixture), \
                    patch.object(confirmation.adaptive, "load_policy") as loader, \
                    patch.object(confirmation.training, "gpu_guard") as guard:
                confirmation.prepare(output)
                first = (output / "protocol.json").read_bytes()
                confirmation.prepare(output)
                self.assertEqual((output / "protocol.json").read_bytes(), first)
                self.assertEqual(confirmation.verify_bindings(output)["original_head_digest"], "original")
            loader.assert_not_called()
            guard.assert_not_called()

    def test_source_or_trained_artifact_drift_invalidates_frozen_protocol(self):
        for field in ["source", "head"]:
            with self.subTest(field=field), temporary() as name:
                output = Path(name) / "output"
                with patch.object(confirmation, "check_output"), \
                        patch.object(confirmation, "protocol_document", side_effect=protocol_fixture):
                    confirmation.prepare(output)
                changed = protocol_fixture()
                if field == "source":
                    changed["fingerprints"]["runner.py"] = "changed-source"
                else:
                    changed["previous_inputs"]["final_head"]["sha256"] = "changed-head"
                with patch.object(confirmation, "check_output"), \
                        patch.object(confirmation, "protocol_document", return_value=changed), \
                        self.assertRaises(AssertionError):
                    confirmation.verify_protocol(output)

    def test_existing_output_cannot_be_retroactively_frozen(self):
        with temporary() as name:
            output = Path(name)
            existing = output / "summary.json"
            confirmation.base.save_json(existing, {"old": "preserve"})
            before = existing.read_bytes()
            with patch.object(confirmation, "check_output"), \
                    patch.object(confirmation, "protocol_document", side_effect=protocol_fixture), \
                    self.assertRaises(AssertionError):
                confirmation.prepare(output)
            self.assertEqual(existing.read_bytes(), before)
            self.assertFalse((output / "protocol.json").exists())

    def test_tampered_head_or_protocol_evaluation_binding_is_rejected(self):
        for field in ["head", "protocol"]:
            with self.subTest(field=field), temporary() as name:
                output = Path(name) / "output"
                with patch.object(confirmation, "check_output"), \
                        patch.object(confirmation, "protocol_document", side_effect=protocol_fixture):
                    confirmation.prepare(output)
                    saved = confirmation.base.read_json(output / "eval-bindings.json")
                    if field == "head":
                        saved["inputs"]["final_head"]["tensor_digest"] = "wrong-head"
                    else:
                        saved["inputs"]["protocol_sha256"] = "stale-protocol"
                    confirmation.base.save_json(output / "eval-bindings.json", saved)
                    with self.assertRaises(AssertionError):
                        confirmation.verify_bindings(output)

    def test_identical_original_and_trained_head_identities_are_rejected(self):
        with temporary() as name:
            output = Path(name) / "output"
            fake = protocol_fixture()
            fake["previous_inputs"]["final_head"]["tensor_digest"] = "original"
            with patch.object(confirmation, "check_output"), \
                    patch.object(confirmation, "protocol_document", return_value=fake):
                confirmation.prepare(output)
                with self.assertRaises(AssertionError):
                    confirmation.verify_bindings(output)


class CollectionTest(unittest.TestCase):
    def test_each_new_trial_switches_to_correct_head_without_changing_inner_method(self):
        with temporary() as name:
            output, policy, original, trained, bindings = context(Path(name))
            observed = []
            def evaluator(model, scene, draw, method, directory, max_steps):
                observed.append((method, confirmation.training.tail_digest(confirmation.training.tail_state(model))))
                return {"scene": scene, "draw": draw, "method": method}
            with patch.object(confirmation.gen, "evaluate_episode", side_effect=evaluator), \
                    patch.object(confirmation, "verify_record"):
                for arm in confirmation.ARMS:
                    result = confirmation.collect(policy, 170, 60000, arm, output / "evaluation" / arm,
                                                  output, bindings, {"original": original, "trained": trained})
                    self.assertEqual(result["arm"], arm)
            self.assertEqual(observed, [(confirmation.ARMS[arm], confirmation.head_identity(bindings, arm))
                                        for arm in confirmation.ARMS])

    def test_resume_completed_sidecar_does_not_predict_or_switch_weights(self):
        with temporary() as name:
            output, policy, original, trained, bindings = context(Path(name))
            directory = output / "evaluation/recovery12"
            confirmation.condition_binding(output, "recovery12", directory, bindings)
            trial = {"scene": 170, "draw": 60000, "method": "adaptive12"}
            confirmation.base.save_json(directory / (confirmation.gen.stem(170, 60000, "adaptive12") + ".json"), trial)
            with patch.object(confirmation.gen, "evaluate_episode") as predict, \
                    patch.object(confirmation, "activate") as switch, \
                    patch.object(confirmation, "verify_record"):
                result = confirmation.collect(policy, 170, 60000, "recovery12", directory,
                                              output, bindings, {"original": original, "trained": trained})
            predict.assert_not_called()
            switch.assert_not_called()
            self.assertEqual(result["trial"], trial)

    def test_orphan_trace_is_rejected_without_overwrite_or_prediction(self):
        with temporary() as name:
            output, policy, original, trained, bindings = context(Path(name))
            directory = output / "evaluation/fixed8"
            confirmation.condition_binding(output, "fixed8", directory, bindings)
            orphan = directory / (confirmation.gen.stem(170, 60000, "fixed8") + ".trace.json.gz")
            orphan.write_bytes(b"preserve orphan evidence")
            with patch.object(confirmation.gen, "evaluate_episode") as predict, self.assertRaises(AssertionError):
                confirmation.collect(policy, 170, 60000, "fixed8", directory,
                                     output, bindings, {"original": original, "trained": trained})
            predict.assert_not_called()
            self.assertEqual(orphan.read_bytes(), b"preserve orphan evidence")

    def test_existing_unlabelled_episode_cannot_be_assigned_new_weights(self):
        with temporary() as name:
            output, _, _, _, bindings = context(Path(name))
            directory = output / "evaluation/recovery12"
            confirmation.base.save_json(directory / "orphan.json", {})
            with self.assertRaises(AssertionError):
                confirmation.condition_binding(output, "recovery12", directory, bindings)

    def test_changed_condition_head_binding_is_rejected(self):
        with temporary() as name:
            output, _, _, _, bindings = context(Path(name))
            directory = output / "evaluation/recovery12"
            saved = confirmation.condition_binding(output, "recovery12", directory, bindings)
            confirmation.base.save_json(directory / "model-binding.json", {**saved, "head_digest": "original"})
            with self.assertRaises(AssertionError):
                confirmation.condition_binding(output, "recovery12", directory, bindings)


class ScanTest(unittest.TestCase):
    def fixture(self, directory, count=2):
        output, _, _, _, bindings = context(directory)
        rows = []
        for item in confirmation.schedule()[:count]:
            arm = item["arm"]
            folder = output / "evaluation" / arm
            confirmation.condition_binding(output, arm, folder, bindings)
            stem = confirmation.gen.stem(item["scene"], item["draw"], confirmation.ARMS[arm])
            trial = {"scene": item["scene"], "draw": item["draw"], "method": confirmation.ARMS[arm],
                     "trace": stem + ".trace.json.gz"}
            confirmation.base.save_json(folder / (stem + ".json"), trial)
            (folder / trial["trace"]).write_bytes(b"fake trace checked by patched low-level verifier")
            rows.append(confirmation._record(trial, arm, folder, output, bindings))
        return output, bindings, rows

    def test_scans_complete_prefix_and_permits_one_sidecar_ahead_of_index(self):
        with temporary() as name:
            output, bindings, rows = self.fixture(Path(name))
            confirmation.base.save_json(output / "evaluation.json", {"complete": False, "records": rows[:1]})
            with patch.object(confirmation.gen, "verify_trial", return_value={}):
                self.assertEqual(confirmation.scan_records(output, bindings), rows)

    def test_missing_sidecar_is_an_orphan_trace_not_a_missing_trial_to_overwrite(self):
        with temporary() as name:
            output, bindings, rows = self.fixture(Path(name))
            row = rows[1]
            path = Path(row["directory"]) / (confirmation.gen.stem(170, 60000, row["trial"]["method"]) + ".json")
            path.unlink()
            with patch.object(confirmation.gen, "verify_trial", return_value={}), self.assertRaises(AssertionError):
                confirmation.scan_records(output, bindings)

    def test_duplicate_sidecar_is_rejected_and_preserved(self):
        with temporary() as name:
            output, bindings, rows = self.fixture(Path(name))
            duplicate = Path(rows[0]["directory"]) / "duplicate.json"
            confirmation.base.save_json(duplicate, rows[0]["trial"])
            before = duplicate.read_bytes()
            with patch.object(confirmation.gen, "verify_trial", return_value={}), self.assertRaises(AssertionError):
                confirmation.scan_records(output, bindings)
            self.assertEqual(duplicate.read_bytes(), before)

    def test_missing_model_identity_cannot_be_recreated_during_scan(self):
        with temporary() as name:
            output, bindings, rows = self.fixture(Path(name))
            path = Path(rows[0]["directory"]) / "model-binding.json"
            path.unlink()
            with patch.object(confirmation.gen, "verify_trial", return_value={}), self.assertRaises(AssertionError):
                confirmation.scan_records(output, bindings)
            self.assertFalse(path.exists())

    def test_saved_progress_index_must_match_actual_sidecars(self):
        with temporary() as name:
            output, bindings, rows = self.fixture(Path(name))
            changed = copy.deepcopy(rows)
            changed[0]["trial"]["scene"] = 209
            confirmation.base.save_json(output / "evaluation.json", {"complete": False, "records": changed})
            with patch.object(confirmation.gen, "verify_trial", return_value={}), self.assertRaises(AssertionError):
                confirmation.scan_records(output, bindings)

    def test_trial_after_schedule_hole_is_rejected(self):
        with temporary() as name:
            output, bindings, rows = self.fixture(Path(name))
            row = rows[0]
            folder = Path(row["directory"])
            (folder / row["trial"]["trace"]).unlink()
            (folder / (confirmation.gen.stem(170, 60000, row["trial"]["method"]) + ".json")).unlink()
            with patch.object(confirmation.gen, "verify_trial", return_value={}), self.assertRaises(AssertionError):
                confirmation.scan_records(output, bindings)

    def test_smoke_prefix_distinguishes_before_and_restored_identical_trial_ids(self):
        with temporary() as name:
            output, _, _, _, bindings = context(Path(name))
            rows = []
            for label, arm in confirmation.SMOKE_ARMS:
                folder = output / "smoke" / label
                confirmation.condition_binding(output, arm, folder, bindings)
                stem = confirmation.gen.stem(0, 1234, confirmation.ARMS[arm])
                trial = {"scene": 0, "draw": 1234, "method": confirmation.ARMS[arm],
                         "trace": stem + ".trace.json.gz"}
                confirmation.base.save_json(folder / (stem + ".json"), trial)
                (folder / trial["trace"]).write_bytes(b"fake smoke trace")
                rows.append(confirmation._record(trial, arm, folder, output, bindings, label))
                with patch.object(confirmation.gen, "verify_trial", return_value={}):
                    self.assertEqual(confirmation.scan_records(output, bindings, smoke=True), rows)
            self.assertEqual(confirmation.identity(rows[0]), confirmation.identity(rows[-1]))
            self.assertNotEqual(rows[0]["label"], rows[-1]["label"])


class StatisticsTest(unittest.TestCase):
    def test_all_eighty_attempts_count_including_unsuccessful_trials(self):
        result = confirmation.metric_table(artificial_records())
        self.assertEqual(result["recovery12"]["attempts"], 80)
        self.assertEqual(result["recovery12"]["successes"], 80)
        self.assertEqual(result["fixed8"]["successes"], 40)
        self.assertEqual(result["fixed8"]["mean_wall_seconds"], 60.)
        self.assertAlmostEqual(result["recovery12"]["mean_final_coverage"], .96)
        self.assertEqual(result["recovery12"]["mean_prediction_seconds"], 5.)
        self.assertEqual(result["recovery12"]["mean_total_prediction_seconds"], 50.)

    def test_each_prediction_draw_is_reported_separately(self):
        for draw in confirmation.DRAWS:
            result = confirmation.metric_table(artificial_records(), draw=draw)
            self.assertEqual(result["recovery12"]["attempts"], 40)
            self.assertEqual(result["fixed8"]["successes"], 20)

    def test_duplicate_missing_or_unknown_draw_records_are_rejected(self):
        records = artificial_records()
        bad_draw = copy.deepcopy(records)
        bad_draw[0]["trial"]["draw"] = 50000
        for variant in [records[:-1], records[:-1] + [records[0]], bad_draw]:
            with self.subTest(count=len(variant)), self.assertRaises(AssertionError):
                confirmation.metric_table(variant)

    def test_unknown_draw_cannot_be_ignored_as_extra_outside_the_schedule(self):
        records = artificial_records()
        extra = copy.deepcopy(records[0])
        extra["trial"]["draw"] = 50000
        for draw in [None, *confirmation.DRAWS]:
            with self.subTest(draw=draw), self.assertRaises(AssertionError):
                confirmation.metric_table(records + [extra], draw=draw)

    def test_paired_uncertainty_resamples_forty_scenes_not_eighty_attempts(self):
        result = confirmation.paired(artificial_records(), "milestone12")
        difference = result["differences_method_minus_reference"]
        values = np.asarray([float(scene % 2 != 0) for scene in confirmation.SCENES])
        indices = np.random.default_rng(20261017).integers(0, 40, size=(10000, 40))
        expected = np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
        self.assertEqual(difference["success"]["mean_difference"], .5)
        self.assertEqual(difference["success"]["paired_scene_bootstrap_95_interval"], expected)
        self.assertEqual(difference["model_queries"]["mean_difference"], -2.)
        self.assertEqual(difference["wall_seconds"]["mean_difference"], -9.)
        self.assertAlmostEqual(difference["final_coverage"]["mean_difference"], .06)
        self.assertEqual(result, confirmation.paired(artificial_records(), "milestone12"))

    def test_draw_effects_are_averaged_within_scene_before_uncertainty(self):
        records = artificial_records()
        for record in records:
            if record["arm"] == "recovery12":
                record["trial"]["success"] = record["trial"]["draw"] == 60000
            else:
                record["trial"]["success"] = record["trial"]["draw"] == 70000
        difference = confirmation.paired(records, "fixed8")["differences_method_minus_reference"]["success"]
        self.assertEqual(difference["mean_difference"], 0.)
        self.assertEqual(difference["paired_scene_bootstrap_95_interval"], [0., 0.])

    def test_paired_summary_also_rejects_duplicate_and_missing_trials(self):
        records = artificial_records()
        for variant in [records[:-1], records[:-1] + [records[0]]]:
            with self.assertRaises(AssertionError):
                confirmation.paired(variant, "fixed8")

    def test_only_two_predeclared_comparators_are_allowed(self):
        for reference in ["milestone12", "fixed8"]:
            self.assertEqual(confirmation.paired(artificial_records(), reference)["reference"], reference)
        with self.assertRaises(AssertionError):
            confirmation.paired(artificial_records(), "recovery12")


class SmokeTest(unittest.TestCase):
    def fixture(self):
        initial = {"coverage": .1, "x": 0}
        baseline = [{"action": [float(i), float(i + 1)], "after": initial} for i in range(16)]
        rows = [{"label": label, "arm": arm, "trial": {"unit_lengths": [8, 8]}}
                for label, arm in confirmation.SMOKE_ARMS]
        traces = {label: {"initial": initial, "steps": copy.deepcopy(baseline)}
                  for label, _ in confirmation.SMOKE_ARMS}
        for step in traces["trained"]["steps"]:
            step["action"] = [999., 999.]
        saved = {"passed": True, "original_backbone_unchanged": True, "records": rows}
        return saved, rows, traces, baseline

    def verify(self, saved, rows, traces, baseline):
        with patch.object(confirmation, "verify_bindings", return_value={}), \
                patch.object(confirmation, "scan_records", return_value=rows), \
                patch.object(confirmation, "verify_record", side_effect=lambda row, *args: traces[row["label"]]), \
                patch.object(confirmation.base, "read_json", side_effect=lambda path: saved if Path(path).name == "smoke.json"
                             else {"traces": {"fixed8": baseline}}):
            return confirmation.verify_smoke(Path("unused"))

    def test_original_trained_original_restoration_is_required_but_trained_actions_may_differ(self):
        saved, rows, traces, baseline = self.fixture()
        self.assertTrue(self.verify(saved, rows, traces, baseline)["passed"])
        traces["restored"]["steps"][8]["action"] = [123., 456.]
        with self.assertRaises(AssertionError):
            self.verify(saved, rows, traces, baseline)

    def test_original_prefix_must_still_match_historical_baseline(self):
        saved, rows, traces, baseline = self.fixture()
        altered = copy.deepcopy(baseline)
        altered[0]["action"] = [123., 456.]
        with self.assertRaises(AssertionError):
            self.verify(saved, rows, traces, altered)

    def test_original_milestone_must_share_first_actions_with_original_fixed_eight(self):
        saved, rows, traces, baseline = self.fixture()
        traces["milestone"]["steps"][3]["action"] = [123., 456.]
        with self.assertRaises(AssertionError):
            self.verify(saved, rows, traces, baseline)

    def test_loaded_policy_smoke_uses_correct_order_and_never_repeats_gpu_guard(self):
        with temporary() as name:
            output, policy, _, _, bindings = context(Path(name))
            observed = []
            def collect(model, scene, draw, arm, directory, root, bound, heads, max_steps, label):
                observed.append((label, arm, scene, draw, max_steps))
                return {"label": label, "arm": arm}
            with patch.object(confirmation, "verify_bindings", return_value=bindings), \
                    patch.object(confirmation, "scan_records", return_value=[]), \
                    patch.object(confirmation, "collect", side_effect=collect), \
                    patch.object(confirmation, "verify_smoke", return_value={"passed": True}), \
                    patch.object(confirmation.training, "gpu_guard") as guard, \
                    patch.object(confirmation.adaptive, "load_policy") as loader:
                self.assertTrue(confirmation.smoke(output, policy)["passed"])
            guard.assert_not_called()
            loader.assert_not_called()
            self.assertEqual(observed, [(label, arm, 0, 1234, 16) for label, arm in confirmation.SMOKE_ARMS])


if __name__ == "__main__":
    unittest.main()
