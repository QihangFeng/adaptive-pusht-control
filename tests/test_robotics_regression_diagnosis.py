"""CPU-only checks of bounded, matched-state regression diagnostics."""

from collections import deque
from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_regression_diagnosis as diagnosis


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="regression-diagnosis-cpu-", dir=parent)


def features(**changes):
    value = {"quality": True, "warning_count": 2, "drawdown": .04,
             "recent_peak": .86, "recovery": -.02, "history_steps": 8,
             "coverage": .82}
    value.update(changes)
    return value


def commands():
    return np.column_stack((100. + 3. * np.arange(12), np.full(12, 200.))).tolist()


def controller_state(**changes):
    value = dict(vars(diagnosis.adaptive.AdaptiveLengthController()))
    value.update(changes)
    return value


def snapshot(actual=False, **changes):
    step = 193 if actual else 192
    pre = controller_state(confirmations=1 if actual else 0)
    post = controller_state(confirmations=2 if actual else 1,
                            interruptions=1 if actual else 0,
                            last_interrupt=193 if actual else -12,
                            armed=not actual)
    value = {"controller_pre": pre, "controller_post": post,
             "event": {"kind": "interrupt", "reason": "confirmed_regression", "discarded": 7} if actual else None,
             "features": features(), "age": 5 if actual else 4,
             "pending": 7 if actual else 8, "step": step, "commands": commands()}
    value.update(changes)
    return value


def saved_trial(spec, *, success=True):
    prefix = spec.get("checkpoint", 0)
    steps = 2 if success else 300 - prefix
    initial = {"coverage": .80, "position": [100., 200.]}
    rows = [{"step": prefix + index + 1, "new_prediction": index == 0,
             "after": {"coverage": .96 if success else .90}}
            for index in range(steps)]
    result = {**spec, "prefix_step": prefix, "prefix_queries": 12 if prefix else 0,
              "success": success, "truncated": not success, "steps": steps,
              "final_step": prefix + steps, "model_queries": 1,
              "total_model_queries": 13 if prefix else 1, "initial": initial,
              "final_coverage": .96 if success else .90, "unit_lengths": [steps],
              "wall_seconds": 10., "monitor_seconds": 1., "events": []}
    return result, {"initial": initial, "steps": rows}


def artifact(output, spec, *, success=True, mutate_trace=None):
    result, trace = saved_trial(spec, success=success)
    if mutate_trace:
        mutate_trace(trace)
    diagnosis.save_trial(result, trace, output)
    return result


def protocol_fixture(root):
    milestone = root / "milestone"
    source = milestone / "episodes"
    source.mkdir(parents=True)
    for name in ("protocol.json", "summary.json", "episodes.json"):
        diagnosis.base.save_json(milestone / name, {"unchanged": True})
    for scene in diagnosis.CHECKPOINTS:
        for method in ("fixed8", "fixed12", "adaptive12"):
            diagnosis.base.save_json(source / f"seed-{scene}-{method}.json", {"unchanged": True})
    (source / "source.trace.json.gz").write_bytes(b"frozen trace fingerprint fixture")
    stack = ExitStack()
    stack.enter_context(patch.object(diagnosis.fresh, "OUTPUT", milestone))
    stack.enter_context(patch.object(diagnosis, "SOURCE", source))
    stack.enter_context(patch.object(diagnosis.fresh, "verify_protocol"))
    earlier = stack.enter_context(patch.object(
        diagnosis.fresh, "protocol_document", return_value={"fingerprints": {"controller.py": "original"}}))
    stack.enter_context(patch.object(diagnosis, "read_source", return_value=({"trace": "source.trace.json.gz"}, {})))
    stack.enter_context(patch.object(diagnosis, "saved_analysis", return_value={"selection": "frozen"}))
    version = stack.enter_context(patch.object(diagnosis.importlib.metadata, "version", return_value="test-version"))
    return stack, milestone, earlier, version


class TrialDesignTest(unittest.TestCase):
    def test_exact_twenty_six_unique_specs_and_frozen_checkpoints(self):
        self.assertEqual(diagnosis.CHECKPOINTS,
                         {54: {"first_condition": 192, "actual_interruption": 193},
                          59: {"first_condition": 150, "actual_interruption": 151}})
        specs = diagnosis.trial_specs()
        self.assertEqual(len(specs), 26)
        self.assertEqual(len({row["id"] for row in specs}), 26)
        self.assertEqual({kind: sum(row["kind"] == kind for row in specs)
                          for kind in ("control", "repeat", "probe")},
                         {"control": 2, "repeat": 8, "probe": 16})
        self.assertTrue(all(row["scene_seed"] in (54, 59) for row in specs))
        self.assertEqual(specs[:2], [row for row in specs if row["kind"] == "control"])

    def test_randomized_probe_futures_are_paired_and_order_is_rotated(self):
        specs = diagnosis.trial_specs()
        for scene, points in diagnosis.CHECKPOINTS.items():
            for name, step in points.items():
                changed = "force_replan_now" if name == "first_condition" else "defer_current_unit"
                pairs = [row for row in specs if row["kind"] == "probe"
                         and row["scene_seed"] == scene and row["checkpoint"] == step]
                self.assertEqual([row["prediction_seed"] for row in pairs], [9100, 9100, 9101, 9101])
                self.assertEqual([row["arm"] for row in pairs], ["original", changed, changed, "original"])

    def test_full_scene_repeats_separate_scene_and_prediction_seeds(self):
        repeats = [row for row in diagnosis.trial_specs() if row["kind"] == "repeat"]
        for scene in diagnosis.CHECKPOINTS:
            rows = [row for row in repeats if row["scene_seed"] == scene]
            self.assertEqual([row["method"] for row in rows], ["fixed8", "adaptive12", "adaptive12", "fixed8"])
            self.assertTrue(all(row["prediction_seed"] != scene for row in rows))
            self.assertTrue(all("checkpoint" not in row for row in rows))

    def test_previous_evidence_and_fresh_results_are_protected(self):
        for output in (Path("results/robotics-fresh-evaluation"),
                       Path("results/robotics-fresh-evaluation/new"),
                       Path("results/robotics-adaptive-length"), Path("results")):
            with self.subTest(output=output), self.assertRaises(ValueError):
                diagnosis.check_output(output)
        with test_directory() as output:
            diagnosis.check_output(Path(output))


class FrozenDiagnosisProtocolTest(unittest.TestCase):
    def test_prepare_is_idempotent_and_binds_unchanged_rules_and_twenty_six_specs(self):
        with test_directory() as directory:
            root = Path(directory)
            stack, milestone, _, _ = protocol_fixture(root)
            with stack:
                output = root / "output"
                first = diagnosis.prepare(output)
                original = (output / "protocol.json").read_bytes()
                self.assertEqual(first["rules"], diagnosis.adaptive.RULES)
                self.assertEqual(first["trials"], diagnosis.trial_specs())
                self.assertEqual(first["max_total_actions"], 300)
                self.assertEqual(first["restore_truth_coverage_absolute_tolerance"], 1e-12)
                self.assertEqual(first["fingerprints"]["experiments/robotics_regression_diagnosis.py"],
                                 diagnosis.base.sha(Path(diagnosis.__file__)))
                self.assertEqual(first["fingerprints"][(milestone / "summary.json").as_posix()],
                                 diagnosis.base.sha(milestone / "summary.json"))
                self.assertEqual(diagnosis.prepare(output), first)
                self.assertEqual(diagnosis.verify_protocol(output), first)
                self.assertEqual((output / "protocol.json").read_bytes(), original)

    def test_changes_to_controller_package_milestone_or_analysis_are_rejected(self):
        for changed in ("controller", "package", "milestone", "analysis"):
            with self.subTest(changed=changed), test_directory() as directory:
                root = Path(directory)
                stack, milestone, earlier, version = protocol_fixture(root)
                with stack:
                    output = root / "output"
                    diagnosis.prepare(output)
                    if changed == "controller":
                        earlier.return_value = {"fingerprints": {"controller.py": "changed"}}
                    elif changed == "package":
                        version.return_value = "changed-version"
                    elif changed == "milestone":
                        diagnosis.base.save_json(milestone / "summary.json", {"unchanged": False})
                    else:
                        diagnosis.base.save_json(output / "saved-analysis.json", {"selection": "changed"})
                    with self.assertRaises(AssertionError):
                        diagnosis.verify_protocol(output)

    def test_existing_artifacts_cannot_be_bound_to_a_new_protocol(self):
        for name in ("trials", "trials.json", "summary.json"):
            with self.subTest(name=name), test_directory() as directory:
                root = Path(directory)
                stack, _, _, _ = protocol_fixture(root)
                with stack:
                    output = root / "output"
                    output.mkdir()
                    if name == "trials":
                        (output / name).mkdir()
                    else:
                        diagnosis.base.save_json(output / name, {})
                    with self.assertRaises(AssertionError):
                        diagnosis.prepare(output)


class SuppressedDecisionTest(unittest.TestCase):
    def test_normal_decision_matches_unchanged_controller_event_and_state(self):
        normal, wrapper = diagnosis.adaptive.AdaptiveLengthController(), diagnosis.adaptive.AdaptiveLengthController()
        for step, age in ((192, 4), (193, 5), (194, 6)):
            expected = normal.update(features(), age, 12 - age, step, commands())
            actual, blocked = diagnosis.decide(wrapper, features(), age, 12 - age, step, commands())
            self.assertEqual(actual, expected)
            self.assertFalse(blocked)
            self.assertEqual(vars(wrapper), vars(normal))

    def test_suppression_rolls_back_only_interruption_effects_not_confirmation(self):
        controller = diagnosis.adaptive.AdaptiveLengthController()
        controller.confirmations = 1
        before = dict(vars(controller))
        event, blocked = diagnosis.decide(controller, features(), 5, 7, 193, commands(), True)
        self.assertIsNone(event)
        self.assertTrue(blocked)
        self.assertEqual(controller.confirmations, 2)
        for name in ("interruptions", "last_interrupt", "armed"):
            self.assertEqual(getattr(controller, name), before[name])

    def test_suppression_preserves_eight_action_trim_and_unchanged_budget(self):
        controller = diagnosis.adaptive.AdaptiveLengthController()
        controller.confirmations = 1
        event, blocked = diagnosis.decide(controller, features(), 8, 4, 193, commands(), True)
        self.assertTrue(blocked)
        self.assertEqual(event["kind"], "trim")
        self.assertEqual(event["discarded"], 4)
        self.assertEqual(controller.interruptions, 0)
        self.assertEqual(controller.last_interrupt, -12)
        self.assertTrue(controller.armed)

    def test_noninterrupt_extension_is_unchanged_even_when_suppression_is_enabled(self):
        eligible = features(warning_count=0, coverage=.60, drawdown=.01, recovery=0.)
        normal, wrapper = diagnosis.adaptive.AdaptiveLengthController(), diagnosis.adaptive.AdaptiveLengthController()
        expected = normal.update(eligible, 8, 4, 200, commands())
        actual, blocked = diagnosis.decide(wrapper, eligible, 8, 4, 200, commands(), True)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["kind"], "extend")
        self.assertFalse(blocked)
        self.assertEqual(vars(normal), vars(wrapper))

    def test_contact_loss_and_dynamic_truth_do_not_change_decision(self):
        baseline, changed = diagnosis.adaptive.AdaptiveLengthController(), diagnosis.adaptive.AdaptiveLengthController()
        original = features()
        modified = {**original, "contact": False, "contact_loss": True,
                    "true_coverage": 0., "success": False}
        self.assertEqual(diagnosis.regression_condition(original), diagnosis.regression_condition(modified))
        self.assertEqual(diagnosis.decide(baseline, original, 4, 8, 192, commands()),
                         diagnosis.decide(changed, modified, 4, 8, 192, commands()))
        self.assertEqual(vars(baseline), vars(changed))


class MatchedBranchTest(unittest.TestCase):
    def test_original_restores_postdecision_state_and_deepclones_event(self):
        saved = snapshot(actual=True)
        before = copy.deepcopy(saved)
        controller = diagnosis.adaptive.AdaptiveLengthController()
        event, suppressed = diagnosis.branch_decision(saved, "original", controller)
        self.assertEqual(vars(controller), saved["controller_post"])
        self.assertEqual(event, saved["event"])
        self.assertIsNot(event, saved["event"])
        self.assertFalse(suppressed)
        self.assertEqual(saved, before)

    def test_forcing_one_step_earlier_consumes_one_bounded_interruption(self):
        saved = snapshot()
        before = copy.deepcopy(saved)
        controller = diagnosis.adaptive.AdaptiveLengthController()
        event, suppressed = diagnosis.branch_decision(saved, "force_replan_now", controller)
        self.assertEqual(event, {"kind": "interrupt", "reason": "diagnostic_force_replan", "discarded": 8})
        self.assertFalse(suppressed)
        self.assertEqual(controller.interruptions, 1)
        self.assertEqual(controller.last_interrupt, 192)
        self.assertFalse(controller.armed)
        self.assertEqual(controller.confirmations, 1)
        self.assertEqual(saved, before)

    def test_force_respects_budget_cooldown_age_pending_and_actual_condition(self):
        changes = ({"controller_post": controller_state(interruptions=2, confirmations=1)},
                   {"controller_post": controller_state(last_interrupt=181, confirmations=1)},
                   {"controller_post": controller_state(armed=False, confirmations=1)},
                   {"age": 3}, {"pending": 1},
                   {"features": features(warning_count=1)})
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                diagnosis.branch_decision(snapshot(**changed), "force_replan_now",
                                          diagnosis.adaptive.AdaptiveLengthController())

    def test_exact_cooldown_and_budget_boundary_permit_second_interruption(self):
        saved = snapshot(controller_post=controller_state(interruptions=1, last_interrupt=180, confirmations=1))
        controller = diagnosis.adaptive.AdaptiveLengthController()
        diagnosis.branch_decision(saved, "force_replan_now", controller)
        self.assertEqual(controller.interruptions, 2)

    def test_defer_restores_predecision_and_preserves_confirmation_without_cost(self):
        saved = snapshot(actual=True)
        controller = diagnosis.adaptive.AdaptiveLengthController()
        event, suppressed = diagnosis.branch_decision(saved, "defer_current_unit", controller)
        self.assertIsNone(event)
        self.assertTrue(suppressed)
        self.assertEqual(controller.interruptions, 0)
        self.assertEqual(controller.last_interrupt, -12)
        self.assertTrue(controller.armed)
        self.assertEqual(controller.confirmations, 2)

    def test_defer_at_eight_keeps_normal_trim_boundary(self):
        saved = snapshot(actual=True, age=8, pending=4)
        saved["event"]["discarded"] = 4
        event, suppressed = diagnosis.branch_decision(saved, "defer_current_unit",
                                                      diagnosis.adaptive.AdaptiveLengthController())
        self.assertTrue(suppressed)
        self.assertEqual(event["kind"], "trim")
        self.assertEqual(event["discarded"], 4)

    def test_defer_requires_an_actual_interruption_and_unknown_arm_is_rejected(self):
        with self.assertRaises(AssertionError):
            diagnosis.branch_decision(snapshot(), "defer_current_unit", diagnosis.adaptive.AdaptiveLengthController())
        with self.assertRaises(ValueError):
            diagnosis.branch_decision(snapshot(), "unbounded_replanning", diagnosis.adaptive.AdaptiveLengthController())


class QueueAndPrefixTest(unittest.TestCase):
    def test_clone_queues_preserves_lengths_and_independent_cpu_tensor_storage(self):
        original = torch.tensor([[1., 2.]], requires_grad=True)
        policy = SimpleNamespace(_queues={"action": deque([original], maxlen=12),
                                         "observation.state": deque([], maxlen=2),
                                         "unbounded": deque([torch.tensor([3.])])})
        saved = diagnosis.clone_queues(policy)
        self.assertEqual(saved["action"]["maxlen"], 12)
        self.assertEqual(saved["observation.state"], {"maxlen": 2, "values": []})
        self.assertIsNone(saved["unbounded"]["maxlen"])
        self.assertEqual(saved["action"]["values"][0].device.type, "cpu")
        self.assertFalse(saved["action"]["values"][0].requires_grad)
        self.assertNotEqual(saved["action"]["values"][0].data_ptr(), original.data_ptr())
        with torch.no_grad():
            original[0, 0] = 99.
        self.assertEqual(saved["action"]["values"][0].tolist(), [[1., 2.]])

    def test_restoration_clones_again_and_does_not_mutate_saved_snapshot(self):
        policy = SimpleNamespace(_queues={"action": deque([torch.tensor([[1., 2.]])], maxlen=12)})
        saved = diagnosis.clone_queues(policy)
        policy._queues["action"].clear()
        diagnosis.restore_queues(policy, saved, "cpu")
        self.assertEqual(policy._queues["action"].maxlen, 12)
        restored = policy._queues["action"][0]
        self.assertNotEqual(restored.data_ptr(), saved["action"]["values"][0].data_ptr())
        restored[0, 0] = 100.
        self.assertEqual(saved["action"]["values"][0].tolist(), [[1., 2.]])

    def test_same_rows_allows_only_documented_truth_coverage_rounding(self):
        rows = [{"step": 193, "action": [1., 2.],
                 "after": {"coverage": 0., "position": [100., 200.],
                           "image_sha256": "original"}, "features": features()}]
        original = copy.deepcopy(rows)
        for difference in (3.33e-16, 5e-13, 1e-12):
            changed = copy.deepcopy(rows)
            changed[0]["after"]["coverage"] += difference
            diagnosis.same_rows(changed, rows)
            self.assertEqual(rows, original)
            self.assertEqual(changed[0]["after"]["coverage"], difference)
        changed = copy.deepcopy(rows)
        changed[0]["after"]["coverage"] = 1.0001e-12
        with self.assertRaises(AssertionError):
            diagnosis.same_rows(changed, rows)

    def test_same_rows_rejects_other_changed_fields_and_unknown_extra_field(self):
        rows = [{"step": 193, "action": [1., 2.], "new_prediction": False,
                 "pending_before": 7, "pending_after": 0,
                 "after": {"coverage": .82, "position": [100., 200.],
                           "image_sha256": "original"},
                 "features": features()}]
        diagnosis.same_rows(copy.deepcopy(rows), rows)
        mutations = (lambda row: row.update(action=[2., 2.]),
                     lambda row: row.update(pending_after=7),
                     lambda row: row["after"].update(coverage=.820000000002),
                     lambda row: row["after"]["position"].__setitem__(0, 100.00000000000001),
                     lambda row: row["after"].update(image_sha256="changed"),
                     lambda row: row["features"].update(warning_count=3),
                     lambda row: row["features"].update(coverage=.820000000000001),
                     lambda row: row.update(extra="unknown"))
        for mutate in mutations:
            changed = copy.deepcopy(rows)
            mutate(changed[0])
            with self.assertRaises(AssertionError):
                diagnosis.same_rows(changed, rows)
        with self.assertRaises(AssertionError):
            diagnosis.same_rows([], rows)


class SavedArtifactTest(unittest.TestCase):
    def spec(self, kind="probe"):
        return next(row for row in diagnosis.trial_specs() if row["kind"] == kind)

    def test_valid_repeat_and_remaining_budget_probe_are_verified(self):
        with test_directory() as directory:
            output = Path(directory)
            for kind in ("repeat", "probe"):
                result = artifact(output, self.spec(kind))
                self.assertEqual(diagnosis.verify_trial(result, output), result)
            timeout = artifact(output, diagnosis.trial_specs()[-1], success=False)
            self.assertEqual(timeout["steps"], 300 - timeout["checkpoint"])
            diagnosis.verify_trial(timeout, output)

    def test_wrong_spec_or_hash_is_rejected(self):
        with test_directory() as directory:
            output = Path(directory)
            result = artifact(output, self.spec())
            for changed in ({"scene_seed": 55}, {"prediction_seed": 999},
                            {"method": "fixed8"}, {"arm": "force_replan_now"},
                            {"trace": "../different.trace.json.gz"}, {"trace_sha256": "0" * 64}):
                with self.subTest(changed=changed), self.assertRaises(AssertionError):
                    diagnosis.verify_trial({**result, **changed}, output)

    def test_absolute_action_indices_initial_state_and_remaining_cap_are_checked(self):
        mutations = (lambda trace: trace["steps"][0].update(step=1),
                     lambda trace: trace.update(initial={"coverage": .1}),
                     lambda trace: trace["steps"][-1].update(step=301))
        for mutate in mutations:
            with test_directory() as directory:
                output = Path(directory)
                result = artifact(output, self.spec(), mutate_trace=mutate)
                with self.assertRaises(AssertionError):
                    diagnosis.verify_trial(result, output)

    def test_query_units_success_and_finite_inclusive_timing_are_checked(self):
        with test_directory() as directory:
            output = Path(directory)
            result = artifact(output, self.spec())
            for changed in ({"prefix_step": 0}, {"model_queries": 2}, {"total_model_queries": 1},
                            {"unit_lengths": [0, 2]}, {"unit_lengths": [2.]},
                            {"success": False}, {"final_coverage": .95},
                            {"wall_seconds": float("inf")}, {"monitor_seconds": -1.},
                            {"monitor_seconds": 11.}):
                with self.subTest(changed=changed), self.assertRaises(AssertionError):
                    diagnosis.verify_trial({**result, **changed}, output)

    def test_partial_trace_cannot_be_overwritten(self):
        with test_directory() as directory:
            output = Path(directory)
            result, trace = saved_trial(self.spec())
            directory = output / "trials"
            directory.mkdir()
            path = directory / (result["id"] + ".trace.json.gz")
            path.write_bytes(b"partial")
            with self.assertRaisesRegex(AssertionError, "partial trace"):
                diagnosis.save_trial(result, trace, output)
            self.assertEqual(path.read_bytes(), b"partial")

    def test_control_requires_exact_original_suffix_and_event_list(self):
        with test_directory() as directory:
            output = Path(directory)
            spec = self.spec("control")
            result, trace = saved_trial(spec)
            result["exact_suffix_match"] = True
            diagnosis.save_trial(result, trace, output)
            source = {"steps": [{}] * spec["checkpoint"] + copy.deepcopy(trace["steps"])}
            with patch.object(diagnosis, "read_source", return_value=({"events": []}, source)):
                diagnosis.verify_trial(result, output)
                with self.assertRaises(AssertionError):
                    diagnosis.verify_trial({**result, "exact_suffix_match": False}, output)
                with self.assertRaises(AssertionError):
                    diagnosis.verify_trial({**result, "events": [{"kind": "interrupt", "step": 192}]}, output)
            source["steps"][-1]["after"]["coverage"] = .97
            with patch.object(diagnosis, "read_source", return_value=({"events": []}, source)):
                with self.assertRaises(AssertionError):
                    diagnosis.verify_trial(result, output)


class SummaryTest(unittest.TestCase):
    def test_complete_twenty_six_unique_trials_are_required(self):
        rows = [saved_trial(spec)[0] for spec in diagnosis.trial_specs()]
        documents = ({"complete": False, "trials": rows},
                     {"complete": True, "trials": rows[:-1]},
                     {"complete": True, "trials": rows[:-1] + [rows[0]]})
        for document in documents:
            with test_directory() as directory:
                output = Path(directory)
                diagnosis.base.save_json(output / "trials.json", document)
                with patch.object(diagnosis, "verify_protocol"), self.assertRaises(AssertionError):
                    diagnosis.summarize(output)

    def test_two_scene_diagnostic_groups_keep_original_and_changed_arms_separate(self):
        rows = [saved_trial(spec)[0] for spec in diagnosis.trial_specs()]
        with test_directory() as directory:
            output = Path(directory)
            diagnosis.base.save_json(output / "trials.json", {"complete": True, "trials": rows})
            with (patch.object(diagnosis, "verify_protocol"),
                  patch.object(diagnosis, "verify_trial") as verify, redirect_stdout(io.StringIO())):
                summary = diagnosis.summarize(output)
            self.assertEqual(verify.call_count, 26)
            self.assertEqual(summary["exact_controls"], 2)
            self.assertEqual(len(summary["groups"]), 6)
            for group in summary["groups"]:
                self.assertEqual(len(group["methods"]), 2)
                for row in group["methods"].values():
                    self.assertEqual(row["attempts"], 2)
                    self.assertEqual(row["successes"], 2)
                    self.assertEqual(row["mean_wall_seconds"], 10.)
                    self.assertEqual(row["mean_monitor_seconds"], 1.)


if __name__ == "__main__":
    unittest.main()
