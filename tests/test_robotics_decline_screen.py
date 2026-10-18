"""CPU-only checks of past-only decline descriptors and paired evidence reuse."""

from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_decline_screen as screen


POINTS = {53: 131, 54: 193, 59: 151, 62: 85, 65: 160, 67: 146}


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="decline-screen-cpu-", dir=parent)


def features(**changes):
    value = {"quality": True, "coverage": .82, "warning_count": 2,
             "drawdown": .08, "recent_peak": .90, "recovery": -.02,
             "stagnation": .04, "position_error": .1, "orientation_error": .2,
             "pending_actions": 6, "history_steps": 8}
    value.update(changes)
    return value


def cases():
    return [{"scene_seed": scene, "checkpoint": step,
             "source_success": scene not in (54, 59), "features": features(),
             "source_sidecar": f"source/{scene}.json", "source_trace": f"source/{scene}.trace.json.gz"}
            for scene, step in POINTS.items()]


def history(count=12):
    coverage = [.90, .88, .89, .84, .83, .82] + [.81] * max(0, count - 6)
    rows, series = [], []
    for index in range(count):
        step = index + 1
        rows.append({"step": step, "action": [100. + 3. * index, 256.],
                     "plan_age": step, "after": {"coverage": 0., "contact": False}})
        series.append({"step": step, **features(
            coverage=coverage[index], position_error=.1 * step,
            orientation_error=.01 * step, warning_count=2 if step >= 5 else 1)})
    return rows, series


def raw_trial(spec, success=True):
    prefix = spec["checkpoint"]
    count = 2 if success else 300 - prefix
    initial = {"coverage": .82, "position": [100., 200.]}
    trace = {"initial": initial,
             "steps": [{"step": prefix + index + 1, "new_prediction": index == 0,
                        "after": {"coverage": .96 if success else .90}}
                       for index in range(count)]}
    raw = {**spec, "prefix_step": prefix, "prefix_queries": 12,
           "success": success, "truncated": not success, "steps": count,
           "final_step": prefix + count, "model_queries": 1,
           "total_model_queries": 13, "initial": initial,
           "final_coverage": .96 if success else .90, "unit_lengths": [count],
           "wall_seconds": 10., "monitor_seconds": 1., "events": []}
    return raw, trace


def save_record(output, spec, *, success=True):
    raw, trace = raw_trial(spec, success)
    screen.diagnosis.save_trial(raw, trace, output)
    path = output / "trials" / (spec["id"] + ".json")
    return screen.normalize_record(raw, spec, path, False), trace


class PastOnlyFeatureTest(unittest.TestCase):
    def test_known_five_step_drop_error_changes_streak_and_command_descriptors(self):
        rows, series = history()
        result = screen.features_at(rows, series, 6)
        self.assertAlmostEqual(result["accumulated_drop_5"], .09)
        self.assertAlmostEqual(result["position_error_change_3"], .3)
        self.assertAlmostEqual(result["orientation_error_change_3"], .03)
        self.assertEqual(result["warning_streak"], 2)
        self.assertEqual(result["command_consistency"], 1.)
        self.assertAlmostEqual(result["command_goal_progress"], 15. / 512.)
        self.assertEqual(result["plan_age"], 6)
        self.assertEqual(result["remaining_budget"], 294)
        self.assertEqual(screen.features_at(rows, series, 6, age=11)["plan_age"], 11)

    def test_future_rows_features_and_dynamic_truth_are_not_inputs(self):
        rows, series = history()
        original_rows, original_series = copy.deepcopy(rows), copy.deepcopy(series)
        expected = screen.features_at(rows, series, 6)
        for index, row in enumerate(rows):
            row["after"] = {"coverage": 1., "contact": True, "position": [-1000., 1000.]}
            if index >= 6:
                row["action"] = [float("nan"), float("nan")]
                row["plan_age"] = 999
        for item in series[6:]:
            item.update(quality=False, coverage=1., position_error=999., warning_count=100)
        self.assertEqual(screen.features_at(rows, series, 6), expected)
        self.assertEqual(screen.features_at(original_rows[:6], original_series[:6], 6), expected)

    def test_feature_extraction_does_not_mutate_saved_rows_or_series(self):
        rows, series = history()
        before = copy.deepcopy((rows, series))
        screen.features_at(rows, series, 6)
        self.assertEqual((rows, series), before)

    def test_missing_or_invalid_camera_windows_produce_missing_derived_values(self):
        for invalid in ("missing", "invalid"):
            rows, series = history()
            if invalid == "missing":
                series = [r for r in series if r["step"] != 4]
            else:
                series[3]["quality"] = False
            result = screen.features_at(rows, series, 6)
            self.assertIsNone(result["accumulated_drop_5"])
            self.assertIsNone(result["position_error_change_3"])
            self.assertIsNone(result["orientation_error_change_3"])

    def test_invalid_current_or_short_history_is_not_filled_using_future_frames(self):
        rows, series = history()
        series[5]["quality"] = False
        result = screen.features_at(rows, series, 6)
        for field in ("accumulated_drop_5", "position_error_change_3", "orientation_error_change_3"):
            self.assertIsNone(result[field])
        self.assertEqual(result["warning_streak"], 0)
        short = screen.features_at(rows, series, 2)
        self.assertIsNone(short["accumulated_drop_5"])
        self.assertIsNone(short["position_error_change_3"])

    def test_command_descriptors_describe_commands_not_object_motion(self):
        rows, series = history()
        for row in rows[:6]:
            row["action"] = [100., 100.]
        result = screen.features_at(rows, series, 6)
        self.assertIsNone(result["command_consistency"])
        self.assertEqual(result["command_goal_progress"], 0.)
        self.assertEqual(result["accumulated_drop_5"], screen.features_at(*history(), 6)["accumulated_drop_5"])


class TrialMatrixTest(unittest.TestCase):
    def test_selection_includes_all_first_interruptions_regardless_of_original_outcome(self):
        episodes = [{"seed": scene, "method": "adaptive12",
                     "events": [{"kind": "extend", "step": 8},
                                {"kind": "interrupt", "step": step},
                                {"kind": "interrupt", "step": step + 5}]}
                    for scene, step in reversed(list(POINTS.items()))]
        episodes += [{"seed": 50, "method": "adaptive12", "events": []},
                     {"seed": 51, "method": "fixed8", "events": [{"kind": "interrupt", "step": 10}]}]

        def source(scene):
            rows, series = history(POINTS[scene] + 6)
            for row, item in zip(rows, series):
                row["features"] = {key: value for key, value in item.items() if key != "step"}
            return {"success": False, "trace": f"seed-{scene}-adaptive12.trace.json.gz"}, {"steps": rows}

        with (patch.object(screen.base, "read_json", return_value={"complete": True, "episodes": episodes}),
              patch.object(screen.diagnosis, "read_source", side_effect=source)):
            selected = screen.selected_cases()
        self.assertEqual([(c["scene_seed"], c["checkpoint"]) for c in selected], list(POINTS.items()))
        self.assertTrue(all(not c["source_success"] for c in selected))

    def test_exact_thirty_six_pairs_and_eight_reuses_twenty_eight_new(self):
        specs = screen.trial_specs(cases())
        self.assertEqual(len(specs), 36)
        self.assertEqual(len({s["id"] for s in specs}), 36)
        self.assertEqual({s["prediction_seed"] for s in specs}, {9100, 9101, 9102})
        self.assertTrue(all(s["method"] == "adaptive12" and s["kind"] == "probe" for s in specs))
        for case in cases():
            for seed in screen.FUTURE_SEEDS:
                pair = [s for s in specs if s["scene_seed"] == case["scene_seed"] and s["prediction_seed"] == seed]
                self.assertEqual({s["arm"] for s in pair}, {"original", "defer_current_unit"})
                self.assertEqual({s["checkpoint"] for s in pair}, {case["checkpoint"]})
                self.assertEqual({s["decision"] for s in pair}, {"replan_now", "continue_unit"})
        self.assertEqual(sum(screen.reuse_path(s) is not None for s in specs), 8)

    def test_four_new_controls_exclude_two_already_verified_scenes(self):
        controls = screen.control_specs(cases())
        self.assertEqual({s["scene_seed"] for s in controls}, {53, 62, 65, 67})
        self.assertEqual(len(controls), 4)
        self.assertTrue(all(s["prediction_seed"] is None and s["arm"] == "original" for s in controls))

    def test_reuse_is_limited_to_actual_checkpoint_and_two_old_prediction_seeds(self):
        reused = next(s for s in screen.trial_specs(cases()) if screen.reuse_path(s) and s["arm"] == "original")
        expected = screen.OLD / "trials" / "probe-54-193-9100-original.json"
        self.assertEqual(screen.reuse_path(reused), expected)
        with self.assertRaises(AssertionError):
            screen.reuse_path({**reused, "checkpoint": 192})
        self.assertIsNone(screen.reuse_path({**reused, "prediction_seed": 9102}))

    def test_output_cannot_overlap_any_prior_evidence_or_ancestors(self):
        for path in (Path("results/robotics-regression-diagnosis"),
                     Path("results/robotics-regression-diagnosis/new"),
                     Path("results/robotics-fresh-evaluation"),
                     Path("results/robotics-adaptive-length"), Path("results")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                screen.check_output(path)


class FrozenProtocolTest(unittest.TestCase):
    def fixture(self):
        stack = ExitStack()
        stack.enter_context(patch.object(screen.diagnosis, "verify_protocol"))
        stack.enter_context(patch.object(screen, "selected_cases", side_effect=lambda: copy.deepcopy(cases())))
        stack.enter_context(patch.object(screen, "legacy_inventory", return_value={"scope": "fixed8 only", "contexts": []}))
        fingerprint = stack.enter_context(patch.object(screen, "fingerprints", return_value={"frozen.py": "original"}))
        version = stack.enter_context(patch.object(screen.importlib.metadata, "version", return_value="test-version"))
        return stack, fingerprint, version

    def test_prepare_is_idempotent_and_separates_legacy_from_new_contexts(self):
        with test_directory() as directory:
            output = Path(directory)
            stack, _, _ = self.fixture()
            with stack:
                first = screen.prepare(output)
                original = (output / "protocol.json").read_bytes()
                self.assertEqual(first["new_comparisons"], 28)
                self.assertEqual(first["reused_comparisons"], 8)
                self.assertEqual(first["new_controls"], 4)
                self.assertEqual(first["reused_controls"], 2)
                self.assertEqual(first["fixed_feature_directions"], screen.FEATURE_DIRECTIONS)
                self.assertEqual(len(first["cases"]), 6)
                self.assertEqual(screen.prepare(output), first)
                self.assertEqual((output / "protocol.json").read_bytes(), original)
                self.assertEqual(screen.base.read_json(output / "legacy-inventory.json")["scope"], "fixed8 only")

    def test_changed_source_packages_cases_or_inventory_reject_resume(self):
        for changed in ("source", "package", "cases", "legacy"):
            with self.subTest(changed=changed), test_directory() as directory:
                output = Path(directory)
                stack, fingerprint, version = self.fixture()
                with stack:
                    screen.prepare(output)
                    if changed == "source":
                        fingerprint.return_value = {"frozen.py": "changed"}
                    elif changed == "package":
                        version.return_value = "changed-version"
                    elif changed == "cases":
                        screen.base.save_json(output / "cases.json", {"cases": cases()[:-1]})
                    else:
                        screen.base.save_json(output / "legacy-inventory.json", {"scope": "changed"})
                    with self.assertRaises(AssertionError):
                        screen.verify_protocol(output)

    def test_preexisting_trials_comparisons_or_summary_cannot_be_rebound(self):
        for name in ("trials", "comparisons.json", "summary.json"):
            with self.subTest(name=name), test_directory() as directory:
                output = Path(directory)
                stack, _, _ = self.fixture()
                with stack:
                    if name == "trials":
                        (output / name).mkdir()
                    else:
                        screen.base.save_json(output / name, {})
                    with self.assertRaises(AssertionError):
                        screen.prepare(output)


class AssociationTest(unittest.TestCase):
    def test_predefined_direction_has_known_rank_agreement_without_fitting(self):
        contexts = [{"features": {"x": x}, "success_benefit": benefit}
                    for x, benefit in ((1., -1.), (2., 0.), (3., 1.))]
        self.assertEqual(screen.fixed_direction_association(contexts, "x", 1),
                         {"comparable_pairs": 3, "concordant_pairs": 3, "agreement": 1.})
        self.assertEqual(screen.fixed_direction_association(contexts, "x", -1),
                         {"comparable_pairs": 3, "concordant_pairs": 0, "agreement": 0.})

    def test_ties_and_missing_values_are_skipped_not_counted_as_disagreement(self):
        contexts = [{"features": {"x": 1.}, "success_benefit": 0.},
                    {"features": {"x": 1.}, "success_benefit": 1.},
                    {"features": {"x": 2.}, "success_benefit": 1.},
                    {"features": {"x": None}, "success_benefit": -1.},
                    {"features": {}, "success_benefit": -1.}]
        self.assertEqual(screen.fixed_direction_association(contexts, "x", 1),
                         {"comparable_pairs": 1, "concordant_pairs": 1, "agreement": 1.})
        self.assertIsNone(screen.fixed_direction_association(contexts[:1], "x", 1)["agreement"])

    def test_rank_association_is_order_invariant_and_does_not_mutate_contexts(self):
        contexts = [{"features": {"x": x}, "success_benefit": benefit}
                    for x, benefit in ((1., -.5), (2., 1.), (3., .5))]
        original = copy.deepcopy(contexts)
        self.assertEqual(screen.fixed_direction_association(contexts, "x", 1),
                         screen.fixed_direction_association(list(reversed(contexts)), "x", 1))
        self.assertEqual(contexts, original)

    def test_nonfinite_features_or_benefits_are_missing_not_discordant(self):
        contexts = [{"features": {"x": 1.}, "success_benefit": 0.},
                    {"features": {"x": 2.}, "success_benefit": 1.},
                    {"features": {"x": float("nan")}, "success_benefit": -1.},
                    {"features": {"x": float("inf")}, "success_benefit": -1.},
                    {"features": {"x": 3.}, "success_benefit": float("nan")}]
        self.assertEqual(screen.fixed_direction_association(contexts, "x", 1),
                         {"comparable_pairs": 1, "concordant_pairs": 1, "agreement": 1.})
        with self.assertRaises(AssertionError):
            screen.fixed_direction_association(contexts, "x", 0)


class PairSummaryTest(unittest.TestCase):
    def records(self):
        values = {53: ([True, True, True], [True, False, False]),
                  54: ([False, False, False], [True, True, False]),
                  59: ([True, True, True], [True, True, True]),
                  62: ([False, False, False], [False, False, False]),
                  65: ([True, False, False], [False, True, False]),
                  67: ([True, True, False], [True, True, False])}
        records = []
        for spec in screen.trial_specs(cases()):
            index = screen.FUTURE_SEEDS.index(spec["prediction_seed"])
            success = values[spec["scene_seed"]][spec["arm"] != "original"][index]
            records.append({**spec, "success": success, "initial": {"coverage": .82, "scene": spec["scene_seed"]},
                            "final_coverage": .90 + .01 * (spec["arm"] == "original"),
                            "model_queries": 10 + (spec["arm"] == "original"), "wall_seconds": 10.,
                            "reused": screen.reuse_path(spec) is not None})
        return records

    def test_six_contexts_eighteen_futures_keep_gains_and_losses_explicit(self):
        contexts = screen.pair_summary(self.records(), cases())
        self.assertEqual(len(contexts), 6)
        self.assertEqual(sum(len(c["samples"]) for c in contexts), 18)
        self.assertEqual(sum(c["gains"] for c in contexts), 3)
        self.assertEqual(sum(c["losses"] for c in contexts), 3)
        first = contexts[0]
        self.assertEqual((first["continue_successes"], first["replan_successes"]), (1, 3))
        self.assertAlmostEqual(first["success_benefit"], 2. / 3.)
        self.assertAlmostEqual(first["mean_coverage_difference"], .01)
        self.assertEqual(first["mean_calls_difference"], 1.)
        self.assertAlmostEqual(contexts[1]["success_benefit"], -2. / 3.)

    def test_duplicate_missing_or_different_initial_pairs_are_rejected(self):
        rows = self.records()
        bad_sets = [rows + [copy.deepcopy(rows[0])], rows[1:]]
        different = copy.deepcopy(rows)
        different[0]["initial"]["scene"] = 999
        bad_sets.append(different)
        for bad in bad_sets:
            with self.assertRaises(AssertionError):
                screen.pair_summary(bad, cases())

    def test_cached_recovery_stops_at_first_new_prediction(self):
        trace = {"steps": [{"new_prediction": False, "features": features(coverage=.80)},
                           {"new_prediction": False, "features": features(coverage=.91)},
                           {"new_prediction": True, "features": features(coverage=1.)}]}
        result = screen.cached_recovery({"success": False}, trace, cases()[0])
        self.assertEqual(result["cached_movements"], 2)
        self.assertEqual(result["camera_frames"], 2)
        self.assertTrue(result["regained_recent_camera_peak"])
        self.assertAlmostEqual(result["camera_gain_before_prediction"], .09)
        self.assertFalse(result["eventual_success"])
        result = screen.cached_recovery({"success": True}, {"steps": trace["steps"][2:]}, cases()[0])
        self.assertEqual(result["cached_movements"], 0)
        self.assertIsNone(result["camera_gain_before_prediction"])

    def test_summary_requires_complete_unique_comparison_ids(self):
        protocol = {"comparison_specs": screen.trial_specs(cases()), "cases": cases(), "scope": "test"}
        rows = self.records()
        documents = ({"complete": False, "records": rows},
                     {"complete": True, "records": rows[:-1]},
                     {"complete": True, "records": rows[:-1] + [rows[0]]})
        for document in documents:
            with test_directory() as directory:
                output = Path(directory)
                screen.base.save_json(output / "comparisons.json", document)
                with (patch.object(screen, "verify_protocol", return_value=protocol),
                      patch.object(screen, "verify_controls"), self.assertRaises(AssertionError)):
                    screen.summarize(output)

    def test_completed_summary_keeps_six_new_contexts_and_does_not_train_a_classifier(self):
        protocol = {"comparison_specs": screen.trial_specs(cases()), "cases": cases(), "scope": "adaptive contexts only"}
        rows = self.records()
        with test_directory() as directory:
            output = Path(directory)
            screen.base.save_json(output / "comparisons.json", {"complete": True, "records": rows})
            with (patch.object(screen, "verify_protocol", return_value=protocol),
                  patch.object(screen, "verify_controls"),
                  patch.object(screen, "verify_record", return_value={"steps": []}) as verify,
                  redirect_stdout(io.StringIO())):
                result = screen.summarize(output)
            self.assertEqual(verify.call_count, 36)
            self.assertEqual(result["context_count"], 6)
            self.assertEqual(result["paired_futures"], 18)
            self.assertEqual(result["reused_records"], 8)
            self.assertEqual(result["new_records"], 28)
            self.assertEqual((result["gains"], result["losses"]), (3, 3))
            self.assertFalse(result["classifier_trained"])
            self.assertFalse(result["controller_changed"])
            self.assertEqual({c["scene_seed"] for c in result["contexts"]}, set(POINTS))


class RestorationControlTest(unittest.TestCase):
    def test_incomplete_duplicate_or_missing_restoration_controls_are_rejected(self):
        rows = [{"scene_seed": scene} for scene in POINTS]
        documents = ({"complete": False, "controls": rows},
                     {"complete": True, "controls": rows[:-1]},
                     {"complete": True, "controls": rows[:-1] + [rows[0]]})
        for document in documents:
            with test_directory() as directory:
                output = Path(directory)
                screen.base.save_json(output / "controls.json", document)
                with self.assertRaises(AssertionError):
                    screen.verify_controls(output, {"cases": cases(), "new_control_specs": screen.control_specs(cases())})

    def test_four_new_and_two_reused_controls_are_all_verified_against_audits(self):
        with test_directory() as directory:
            root = Path(directory)
            output, old = root / "output", root / "old"
            (old / "trials").mkdir(parents=True)
            controls = [dict(spec) for spec in screen.control_specs(cases())]
            old_audits, new_audits = {}, {}
            for scene, points in screen.diagnosis.CHECKPOINTS.items():
                path = old / "trials" / f"control-{scene}-{points['first_condition']}.json"
                screen.base.save_json(path, {"id": "original-control"})
                audit = {"exact_hidden_state": scene}
                old_audits[str(scene)] = {"checkpoints": {str(points["actual_interruption"]): audit}}
                new_audits[str(scene)] = audit
                controls.append({"scene_seed": scene, "reused": True, "exact_suffix_match": True,
                                 "source_sidecar": path.as_posix(), "sidecar_sha256": screen.base.sha(path),
                                 "covers_checkpoint": points["actual_interruption"]})
            screen.base.save_json(output / "controls.json", {"complete": True, "controls": controls})
            screen.base.save_json(output / "restoration-audit.json", new_audits)
            screen.base.save_json(old / "restore-prefix-audit.json", old_audits)
            protocol = {"cases": cases(), "new_control_specs": screen.control_specs(cases())}
            with (patch.object(screen, "OLD", old), patch.object(screen, "verify_record") as new,
                  patch.object(screen.diagnosis, "verify_trial") as reused):
                result = screen.verify_controls(output, protocol)
                self.assertEqual(set(result), set(POINTS))
                self.assertEqual(new.call_count, 4)
                self.assertEqual(reused.call_count, 2)
                new_audits["54"] = {"exact_hidden_state": -1}
                screen.base.save_json(output / "restoration-audit.json", new_audits)
                with self.assertRaises(AssertionError):
                    screen.verify_controls(output, protocol)


class RecordIntegrityTest(unittest.TestCase):
    def spec(self):
        return next(s for s in screen.trial_specs(cases()) if screen.reuse_path(s) is None)

    def test_new_record_trace_identity_and_remaining_budget_are_valid(self):
        with test_directory() as directory:
            output = Path(directory)
            record, trace = save_record(output, self.spec())
            self.assertEqual(screen.verify_record(record, self.spec(), output), trace)
            timeout_spec = screen.trial_specs(cases())[-1]
            timeout, _ = save_record(output, timeout_spec, success=False)
            screen.verify_record(timeout, timeout_spec, output)
            self.assertEqual(timeout["steps"], 300 - timeout_spec["checkpoint"])

    def test_normalized_old_identity_preserves_raw_data_and_provenance(self):
        with test_directory() as directory:
            root = Path(directory)
            old = root / "old"
            old.mkdir()
            with patch.object(screen, "OLD", old):
                spec = next(s for s in screen.trial_specs(cases()) if screen.reuse_path(s))
                old_spec = next(s for s in screen.diagnosis.trial_specs()
                                if s["scene_seed"] == spec["scene_seed"] and s.get("checkpoint") == spec["checkpoint"]
                                and s["prediction_seed"] == spec["prediction_seed"] and s.get("arm") == spec["arm"])
                raw, trace = raw_trial(old_spec)
                screen.diagnosis.save_trial(raw, trace, old)
                path = screen.reuse_path(spec)
                record = screen.normalize_record(raw, spec, path, True)
                self.assertNotEqual(raw["id"], record["id"])
                self.assertEqual(record["trace"], raw["trace"])
                self.assertEqual(record["model_queries"], raw["model_queries"])
                self.assertEqual(screen.verify_record(record, spec, root / "output"), trace)
                with patch.object(screen.diagnosis, "collect") as collect:
                    self.assertEqual(screen.saved_or_collect(object(), spec, root / "output", {}, {}), record)
                    collect.assert_not_called()
                with self.assertRaises(AssertionError):
                    screen.verify_record({**record, "model_queries": 999}, spec, root / "output")

    def test_changed_sidecar_trace_path_reuse_flag_or_normalized_metadata_is_rejected(self):
        with test_directory() as directory:
            output = Path(directory)
            spec = self.spec()
            record, _ = save_record(output, spec)
            for changed in ({"sidecar_sha256": "0" * 64}, {"trace_sha256": "0" * 64},
                            {"trace_directory": "elsewhere"}, {"source_sidecar": "elsewhere.json"},
                            {"reused": True}, {"checkpoint": 1}, {"decision": "continue_unit"}):
                with self.subTest(changed=changed), self.assertRaises(AssertionError):
                    screen.verify_record({**record, **changed}, spec, output)

    def test_tampered_metric_is_rejected_even_with_valid_trace_and_sidecar_hash(self):
        with test_directory() as directory:
            output = Path(directory)
            spec = self.spec()
            record, _ = save_record(output, spec)
            for changed in ({"wall_seconds": float("inf")}, {"monitor_seconds": -1.},
                            {"final_step": 301}, {"success": False}, {"unit_lengths": [1]}):
                with self.subTest(changed=changed), self.assertRaises(AssertionError):
                    screen.verify_record({**record, **changed}, spec, output)

    def test_partial_new_trace_is_rejected_before_any_collector_call(self):
        with test_directory() as directory:
            output = Path(directory)
            spec = self.spec()
            trials = output / "trials"
            trials.mkdir()
            (trials / (spec["id"] + ".trace.json.gz")).write_bytes(b"partial")
            with patch.object(screen.diagnosis, "collect") as collect:
                with self.assertRaisesRegex(AssertionError, "orphan trace"):
                    screen.saved_or_collect(object(), spec, output, {}, {})
                collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
