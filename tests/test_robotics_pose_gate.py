"""CPU-only precision contracts; not evidence of a useful pushing controller."""

from contextlib import ExitStack, redirect_stdout
import copy
import gzip
import hashlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_pose_gate as gate


def directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="pose-gate-cpu-", dir=parent)


def state(position=(258., 259.), angle=math.pi / 4, coverage=.92, pixels=None):
    pixels = np.zeros((96, 96, 3), dtype=np.uint8) if pixels is None else pixels
    return {"block_pos": list(position), "block_angle": angle,
            "pusher_pos": [120., 220.], "coverage": coverage,
            "image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}


def image_pose(position=(258., 259.), angle=math.pi / 4):
    return [value * gate.visual.SCALE + gate.visual.PIXEL_CENTER_OFFSET
            for value in position] + [angle]


def estimate(position=(258., 259.), angle=math.pi / 4, quality=True):
    return {"quality": quality, "pose_image": image_pose(position, angle) if quality else None}


def sample(seed=0, step=1, stratum="fine_alignment", truth=None, predicted=None):
    truth = state(angle=gate.GOAL[2] + .02) if truth is None else truth
    predicted = estimate(angle=gate.GOAL[2] + .02) if predicted is None else predicted
    estimates = {method: copy.deepcopy(predicted) for method in gate.METHODS}
    return {"seed": seed, "step": step, "stratum": stratum,
            "scene_success": stratum == "completed_control",
            "scores": {method: gate.score_pose_frame(estimates[method], truth)
                       for method in gate.METHODS},
            "estimates": estimates, "fit_seconds": .01}


def run_fixture(root, count=2):
    truth = state(angle=gate.GOAL[2] + .02)
    rows = [{"step": index, "before": copy.deepcopy(truth),
             "after": copy.deepcopy(truth)} for index in range(1, count + 1)]
    cases, samples = [], []
    for index in range(1, count + 1):
        case = {"seed": 0, "step": index, "stratum": "fine_alignment",
                "scene_success": False, "source_trace": "saved.trace.json.gz",
                "image_sha256": truth["image_sha256"],
                "previous_image_sha256": truth["image_sha256"]}
        cases.append(case)
        samples.append({**sample(step=index, truth=truth), **case})
    return {0: rows}, cases, samples, {"frozen": True}


def discovery_fixture(root):
    episodes, hashes = [], {}
    for seed in range(30):
        trace_path = root / f"seed-{seed}.trace.json.gz"
        rows = [{"step": 1, "before": state(coverage=.80), "after": state(coverage=.92)},
                {"step": 2, "after": state(coverage=.96)}]
        with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
            gate.json.dump({"steps": rows}, stream)
        episodes.append({"seed": seed, "source_directory": str(root), "trace": trace_path.name,
                         "steps": 2, "final_coverage": .96, "success": True})
        hashes[trace_path.as_posix()] = gate.sha(trace_path)
    source, validation = root / "source.json", root / "validation.json"
    gate.write(source, {"baseline_complete": True, "episodes": episodes})
    gate.write(validation, {"passed": True, "source_trace_hashes": hashes})
    return source, validation


class PoseMathTest(unittest.TestCase):
    def test_inverse_resize_center_offset_recovers_body_origin(self):
        self.assertEqual(gate.visual.GOAL_POSE[0], 47.59375)
        np.testing.assert_allclose(gate.image_to_world_pose(gate.visual.GOAL_POSE), gate.GOAL)
        np.testing.assert_allclose(gate.image_to_world_pose(image_pose((259., 252.), 1.2)),
                                   [259., 252., 1.2])

    def test_conversion_does_not_use_centroid_or_invert_y(self):
        result = gate.image_to_world_pose(image_pose((256., 261.), -.3))
        self.assertEqual(result, [256., 261., -.3])
        self.assertNotEqual(result[1], 261. + gate.visual.T_CENTROID[1] / gate.visual.SCALE)

    def test_conversion_rejects_nonfinite_or_wrong_shape(self):
        for values in ([1., 2.], [1., 2., 3., 4.], [[1., 2., 3.]],
                       [1., math.nan, .2], [math.inf, 2., .2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                gate.image_to_world_pose(values)

    def test_signed_wrap_preserves_direction_and_full_turn_equivalence(self):
        self.assertAlmostEqual(gate.signed_angle(2 * math.pi + .1), .1)
        self.assertAlmostEqual(gate.signed_angle(-2 * math.pi - .1), -.1)
        self.assertAlmostEqual(gate.signed_angle(math.pi), -math.pi)
        self.assertAlmostEqual(gate.signed_angle(-math.pi), -math.pi)

    def test_goal_residual_has_signed_translation_and_rotation(self):
        result = gate.goal_residual([259., 252., gate.GOAL[2] + .2])
        np.testing.assert_allclose(result, [-3., 4., -.2])
        np.testing.assert_allclose(gate.goal_residual([256., 256., gate.GOAL[2] + 2 * math.pi]),
                                   [0., 0., 0.], atol=1e-14)

    def test_t_shape_does_not_treat_half_turn_as_equivalent(self):
        result = gate.score_pose_frame(estimate(angle=gate.GOAL[2] + math.pi), state())
        self.assertAlmostEqual(result["angle_error_degrees"], 180.)
        self.assertGreater(result["max_vertex_error"], 230.)


class FrameScoringTest(unittest.TestCase):
    def test_exact_fit_has_zero_coordinate_and_vertex_errors(self):
        result = gate.score_pose_frame(estimate(), state())
        self.assertTrue(result["valid"])
        self.assertEqual(result["position_error"], 0.)
        self.assertEqual(result["max_vertex_error"], 0.)
        self.assertEqual(result["angle_error_degrees"], 0.)
        self.assertEqual(result["direction_eligible"], [True, True, False])
        self.assertEqual(result["direction_agreement"], [True, True, None])

    def test_translation_error_is_world_distance_not_normalized_image_error(self):
        result = gate.score_pose_frame(estimate((261., 263.)), state())
        self.assertEqual(result["position_error"], 5.)
        self.assertAlmostEqual(result["max_vertex_error"], 5.)

    def test_angle_error_wraps_boundary_not_raw_subtraction(self):
        truth = state(angle=math.pi - .02)
        result = gate.score_pose_frame(estimate(angle=-math.pi + .03), truth)
        self.assertAlmostEqual(result["angle_error_degrees"], math.degrees(.05))

    def test_same_residual_norm_does_not_imply_same_direction(self):
        truth = state(position=(258., 259.), angle=gate.GOAL[2] + .02)
        predicted = estimate(position=(254., 253.), angle=gate.GOAL[2] - .02)
        result = gate.score_pose_frame(predicted, truth)
        self.assertAlmostEqual(math.dist(result["truth_residual"][:2], [0., 0.]),
                               math.dist(result["estimated_residual"][:2], [0., 0.]))
        self.assertEqual(result["direction_agreement"], [False, False, False])

    def test_deadbands_are_truth_only_and_inclusive(self):
        truth = state(position=(255., 256.5), angle=gate.GOAL[2])
        result = gate.score_pose_frame(estimate(position=(255., 200.), angle=gate.GOAL[2] + 2.), truth)
        self.assertEqual(result["direction_eligible"], [True, False, False])
        self.assertEqual(result["direction_agreement"], [True, None, None])
        result = gate.score_pose_frame(estimate(), state(position=(256., 256.)))
        self.assertEqual(result["direction_eligible"], [False, False, False])
        self.assertEqual(result["direction_agreement"], [None, None, None])

    def test_zero_predicted_residual_is_not_correct_when_truth_has_clear_direction(self):
        result = gate.score_pose_frame(estimate((256., 256.)), state())
        self.assertEqual(result["direction_agreement"], [False, False, None])

    def test_invalid_fit_abstains_instead_of_scoring_zero(self):
        result = gate.score_pose_frame(estimate(quality=False), state())
        self.assertFalse(result["valid"])
        self.assertEqual(result["truth_pose"], [258., 259., gate.GOAL[2]])
        self.assertEqual(result["direction_eligible"], [True, True, False])
        for name in ("estimated_pose", "estimated_residual", "position_error",
                     "angle_error_degrees", "max_vertex_error"):
            self.assertIsNone(result[name])
        self.assertEqual(result["direction_agreement"], [None, None, None])


class SelectionTest(unittest.TestCase):
    @staticmethod
    def rows(values):
        return [{"step": index, "after": {"coverage": value}}
                for index, value in enumerate(values, start=1)]

    def test_first_middle_last_use_temporal_eligibility_not_largest_coverage(self):
        rows = self.rows([.2, .93, .7, .91, .95, .92, .5, .94, .96])
        result = gate.select_steps(rows, True)
        self.assertEqual(result, [{"step": 2, "stratum": "fine_alignment"},
                                  {"step": 5, "stratum": "fine_alignment"},
                                  {"step": 8, "stratum": "fine_alignment"},
                                  {"step": 9, "stratum": "completed_control"}])

    def test_fine_stratum_includes_both_boundaries_and_no_above_threshold_frame(self):
        self.assertEqual(gate.select_steps(self.rows([.899999, .90, .95, .950001]), False),
                         [{"step": 2, "stratum": "fine_alignment"},
                          {"step": 3, "stratum": "fine_alignment"}])

    def test_one_or_two_candidates_are_not_repeated(self):
        self.assertEqual(gate.select_steps(self.rows([.93]), False),
                         [{"step": 1, "stratum": "fine_alignment"}])
        self.assertEqual(len(gate.select_steps(self.rows([.93, .94]), False)), 2)

    def test_absent_fine_frames_remain_absent_not_filled_from_easy_frames(self):
        self.assertEqual(gate.select_steps(self.rows([.2, .5]), False), [])
        self.assertEqual(gate.select_steps(self.rows([.2, .96]), True),
                         [{"step": 2, "stratum": "completed_control"}])

    def test_success_control_requires_strictly_above_point_ninety_five(self):
        with self.assertRaises(AssertionError):
            gate.select_steps(self.rows([.90, .95]), True)


class AggregationTest(unittest.TestCase):
    def test_cold_primary_gate_cannot_be_saved_by_good_warm_fit(self):
        row = sample(predicted=estimate((260., 259.), gate.GOAL[2] + .02))
        row["estimates"]["previous_frame"] = estimate(angle=gate.GOAL[2] + .02)
        row["scores"]["previous_frame"] = gate.score_pose_frame(
            row["estimates"]["previous_frame"], state(angle=gate.GOAL[2] + .02))
        result = gate.summarize([row])
        self.assertFalse(result["passed"])
        self.assertTrue(result["by_stratum"]["fine_alignment"]["previous_frame"]["passed"])

    def test_invalid_frames_stay_in_availability_denominator(self):
        rows = [sample(), sample(step=2, predicted=estimate(quality=False))]
        result = gate.method_metrics(rows, "cold")
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["valid_frames"], 1)
        self.assertEqual(result["valid_fraction"], .5)
        self.assertEqual(result["eligible_valid_direction_components"], 3)
        self.assertEqual(result["direction_agreement"], 1.)
        self.assertFalse(result["passed"])

    def test_empty_or_deadband_only_sample_cannot_pass(self):
        result = gate.method_metrics([], "cold")
        self.assertFalse(result["passed"])
        self.assertIsNone(result["position_error_p95"])
        result = gate.method_metrics([sample(truth=state((256., 256.)),
                                                   predicted=estimate((256., 256.)))], "cold")
        self.assertFalse(result["passed"])
        self.assertIsNone(result["direction_agreement"])

    def test_pooled_frame_mean_and_equal_scene_mean_are_both_explicit(self):
        rows = [sample(seed=0, step=index, predicted=estimate((259., 259.), gate.GOAL[2] + .02))
                for index in (1, 2, 3)] + [sample(seed=1, predicted=estimate((261., 259.), gate.GOAL[2] + .02))]
        result = gate.method_metrics(rows, "cold")
        self.assertEqual(result["position_error_mean"], 1.5)
        self.assertEqual(result["scene_equal_mean_errors"]["position_error"], 2.)
        self.assertEqual(result["per_scene_mean_errors"]["0"]["position_error"], 1.)
        self.assertEqual(result["per_scene_mean_errors"]["1"]["position_error"], 3.)

    def test_controls_do_not_enter_fine_alignment_statistics(self):
        fine = sample()
        control = sample(seed=1, stratum="completed_control", predicted=estimate((308., 259.)))
        result = gate.summarize([fine, control])
        self.assertTrue(result["passed"])
        self.assertEqual(result["by_stratum"]["fine_alignment"]["cold"]["frames"], 1)
        self.assertEqual(result["by_stratum"]["completed_control"]["cold"]["position_error_mean"], 50.)
        self.assertEqual(result["ai_calls"], 0)
        self.assertEqual(result["corrective_actions"], 0)

    def test_per_axis_directions_prevent_translation_from_hiding_wrong_rotation(self):
        truth = state(angle=gate.GOAL[2] + .009)
        rows = [sample(step=index, truth=truth, predicted=estimate(angle=gate.GOAL[2] + .009))
                for index in range(1, 31)]
        rows[0] = sample(truth=truth, predicted=estimate(angle=gate.GOAL[2] - .001))
        result = gate.method_metrics(rows, "cold")
        self.assertAlmostEqual(result["direction_agreement"], 89 / 90)
        self.assertAlmostEqual(result["direction_by_component"]["rotation"]["agreement"], 29 / 30)
        self.assertEqual(result["direction_by_component"]["rotation"]["opposed_frames"], 1)
        self.assertEqual(result["direction_by_component"]["x"]["opposed_frames"], 0)
        # At this sample size both direction conditions pass; a smaller panel
        # exposes the per-axis disagreement even when pooled agreement passes.
        result = gate.method_metrics(rows[:10], "cold")
        self.assertGreaterEqual(result["direction_agreement"], .95)
        self.assertLess(result["direction_by_component"]["rotation"]["agreement"], .95)
        self.assertLess(result["angle_error_degrees_p95"], gate.LIMITS["angle_error_degrees_p95"])
        self.assertLess(result["max_vertex_error_p95"], gate.LIMITS["max_vertex_error_p95"])
        self.assertFalse(result["passed"])

    def test_invalid_fit_is_counted_unavailable_for_each_eligible_axis(self):
        rows = [sample(), sample(step=2, predicted=estimate(quality=False))]
        result = gate.method_metrics(rows, "cold")
        for axis in ("x", "y", "rotation"):
            component = result["direction_by_component"][axis]
            self.assertEqual(component["eligible_frames_including_invalid"], 2)
            self.assertEqual(component["available_frames"], 1)
            self.assertEqual(component["unavailable_frames"], 1)


class PixelInputContractTest(unittest.TestCase):
    def test_cold_and_previous_frame_estimators_have_separate_rgb_only_histories(self):
        instances = []

        class RecordingEstimator:
            def __init__(self):
                self.images = []
                instances.append(self)

            def estimate(self, pixels):
                self.images.append(pixels.copy())
                return {"calls": len(self.images), "image_value": int(pixels[0, 0, 0])}

        pixels = np.full((96, 96, 3), 2, dtype=np.uint8)
        previous = np.full((96, 96, 3), 1, dtype=np.uint8)
        with patch.object(gate.visual, "VisualAlignmentEstimator", RecordingEstimator):
            result = gate.sample_pair(pixels, previous)
        self.assertEqual(len(instances), 2)
        self.assertEqual([len(instance.images) for instance in instances], [1, 2])
        self.assertEqual(result["cold"], {"calls": 1, "image_value": 2})
        self.assertEqual(result["previous_frame"], {"calls": 2, "image_value": 2})
        np.testing.assert_array_equal(instances[1].images[0], previous)
        np.testing.assert_array_equal(instances[1].images[1], pixels)

    def test_scoring_truth_changes_cannot_change_estimator_inputs(self):
        pixels = np.zeros((96, 96, 3), dtype=np.uint8)
        with patch.object(gate.visual.VisualAlignmentEstimator, "estimate", return_value=estimate()) as fit:
            gate.sample_pair(pixels, pixels)
        self.assertEqual(fit.call_count, 3)
        for args, kwargs in fit.call_args_list:
            self.assertEqual(kwargs, {})
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], pixels)


class RecordedReplayTest(unittest.TestCase):
    @staticmethod
    def fixture(count=4, changed_frame=None, changed_physics=None):
        initial_pixels = np.zeros((96, 96, 3), dtype=np.uint8)
        initial = state(pixels=initial_pixels)
        rows = []
        for index in range(1, count + 1):
            pixels = np.full((96, 96, 3), index, dtype=np.uint8)
            after = state(position=(258. + index, 259.),
                          angle=gate.GOAL[2] + .01 * index, pixels=pixels)
            after["pusher_pos"] = [120. + index, 220.]
            rows.append({"step": index, "action": [12.34 + index, 56.78],
                         "before": copy.deepcopy(initial), "after": after})

        class Body:
            position = [0., 0.]
            angle = 0.

        class Environment:
            def __init__(self):
                self.unwrapped = self
                self.block, self.agent = Body(), Body()
                self.closed, self.actions, self.resets = False, [], []
                self.buffer = initial_pixels.copy()

            def _set_state(self, *_):
                raise AssertionError("Replay must not use an altered static restoration.")

            def apply(self, current):
                self.block.position = list(current["block_pos"])
                self.block.angle = current["block_angle"]
                self.agent.position = list(current["pusher_pos"])

            def reset(self, seed):
                self.resets.append(seed)
                self.apply(initial)
                return {"pixels": self.buffer}, {}

            def step(self, action):
                self.actions.append(action.copy())
                index = len(self.actions)
                self.apply(rows[index - 1]["after"])
                self.buffer.fill(index)
                if changed_frame == index:
                    self.buffer[0, 0, 0] += 1
                if changed_physics is not None and changed_physics[0] == index:
                    if changed_physics[1] == "block_position":
                        self.block.position[0] += .00001
                    elif changed_physics[1] == "block_angle":
                        self.block.angle += .00001
                    else:
                        self.agent.position[0] += .00001
                return {"pixels": self.buffer}, 0., False, False, {}

            def close(self):
                self.closed = True

        return rows, Environment()

    def test_checked_pixels_copies_exact_saved_rgb_not_shared_renderer_view(self):
        pixels = np.full((96, 96, 3), 7, dtype=np.uint8)
        result = gate.checked_pixels(pixels, state(pixels=pixels))
        self.assertIsNot(result, pixels)
        np.testing.assert_array_equal(result, pixels)
        pixels.fill(8)
        self.assertEqual(int(result[0, 0, 0]), 7)

    def test_saved_image_hash_mismatch_hard_fails_before_any_fit(self):
        pixels = np.zeros((96, 96, 3), dtype=np.uint8)
        changed = state()
        changed["image_sha256"] = "changed"
        with self.assertRaisesRegex(AssertionError, "Saved RGB mismatch"):
            gate.checked_pixels(pixels, changed)

    def test_wrong_pixel_dtype_or_size_cannot_be_scored(self):
        for pixels in (np.zeros((96, 96, 3), dtype=float),
                       np.zeros((95, 96, 3), dtype=np.uint8)):
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype):
                with self.assertRaises(AssertionError):
                    gate.checked_pixels(pixels, state(pixels=pixels))

    def test_replay_runs_only_to_latest_selection_and_returns_immediate_predecessors(self):
        rows, env = self.fixture()
        with patch.object(gate.gym, "make", return_value=env):
            frames, count = gate.replay_frames(17, rows, [2, 3])
        self.assertEqual(count, 3)
        self.assertEqual(set(frames), {1, 2, 3})
        self.assertEqual(env.resets, [17])
        self.assertEqual(len(env.actions), 3)
        self.assertTrue(env.closed)
        for step, pixels in frames.items():
            self.assertEqual(int(pixels[0, 0, 0]), step)
        self.assertFalse(np.shares_memory(frames[1], frames[2]))
        self.assertFalse(np.shares_memory(frames[2], env.buffer))

    def test_first_selection_includes_copied_initial_image(self):
        rows, env = self.fixture()
        with patch.object(gate.gym, "make", return_value=env):
            frames, count = gate.replay_frames(17, rows, [1])
        self.assertEqual(set(frames), {0, 1})
        self.assertEqual(count, 1)
        self.assertEqual(int(frames[0][0, 0, 0]), 0)
        self.assertEqual(int(frames[1][0, 0, 0]), 1)
        self.assertTrue(env.closed)

    def test_recorded_actions_use_float32_without_new_policy_inference(self):
        rows, env = self.fixture()
        with (patch.object(gate.gym, "make", return_value=env),
              patch.object(gate.visual, "VisualAlignmentEstimator", side_effect=AssertionError("No fit in replay"))):
            gate.replay_frames(17, rows, [4])
        for actual, row in zip(env.actions, rows):
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, np.asarray(row["action"], dtype=np.float32))

    def test_unselected_intermediate_rgb_mismatch_aborts_and_closes(self):
        rows, env = self.fixture(changed_frame=1)
        with patch.object(gate.gym, "make", return_value=env):
            with self.assertRaisesRegex(AssertionError, "Saved RGB mismatch"):
                gate.replay_frames(17, rows, [4])
        self.assertEqual(len(env.actions), 1)
        self.assertTrue(env.closed)

    def test_physical_coordinates_and_angle_have_no_rounding_tolerance(self):
        for coordinate in ("block_position", "block_angle", "pusher_position"):
            with self.subTest(coordinate=coordinate):
                rows, env = self.fixture(changed_physics=(1, coordinate))
                with patch.object(gate.gym, "make", return_value=env):
                    with self.assertRaises(AssertionError):
                        gate.replay_frames(17, rows, [4])
                self.assertEqual(len(env.actions), 1)
                self.assertTrue(env.closed)

    def test_initial_image_mismatch_aborts_before_recorded_action_and_closes(self):
        rows, env = self.fixture()
        rows[0]["before"]["image_sha256"] = "changed"
        with patch.object(gate.gym, "make", return_value=env):
            with self.assertRaisesRegex(AssertionError, "Saved RGB mismatch"):
                gate.replay_frames(17, rows, [4])
        self.assertEqual(len(env.actions), 0)
        self.assertTrue(env.closed)

    def test_initial_physical_mismatch_is_rejected_even_if_rgb_hash_matches(self):
        for coordinate in ("block_position", "block_angle", "pusher_position"):
            with self.subTest(coordinate=coordinate):
                rows, env = self.fixture()
                initial = rows[0]["before"]
                if coordinate == "block_position":
                    initial["block_pos"][0] += .00001
                elif coordinate == "block_angle":
                    initial["block_angle"] += .00001
                else:
                    initial["pusher_pos"][0] += .00001
                with patch.object(gate.gym, "make", return_value=env):
                    with self.assertRaises(AssertionError):
                        gate.replay_frames(17, rows, [4])
                self.assertEqual(len(env.actions), 0)
                self.assertTrue(env.closed)

    def test_invalid_selection_rejected_before_environment_creation(self):
        rows, _ = self.fixture()
        for steps in ([], [0], [5], [1.5], ["1"]):
            with self.subTest(steps=steps), patch.object(gate.gym, "make") as make:
                with self.assertRaises(AssertionError):
                    gate.replay_frames(17, rows, steps)
                make.assert_not_called()


class IntegrityTest(unittest.TestCase):
    def test_discovery_requires_passing_old_gate_and_all_ordered_discovery_seeds(self):
        with directory() as name:
            root = Path(name)
            source, validation = discovery_fixture(root)
            with patch.object(gate, "SOURCE", source), patch.object(gate, "VALIDATION", validation):
                loaded, cases, bindings = gate.discovery()
                self.assertEqual(sorted(loaded), list(range(30)))
                self.assertEqual(len(cases), 60)
                self.assertEqual(len(bindings), 34)
                self.assertEqual([case["stratum"] for case in cases[:2]],
                                 ["fine_alignment", "completed_control"])
                document = gate.read(validation)
                gate.write(validation, {**document, "passed": False})
                with self.assertRaises(AssertionError):
                    gate.discovery()
                gate.write(validation, document)
                document = gate.read(source)
                document["episodes"][0]["seed"] = 1
                gate.write(source, document)
                with self.assertRaises(AssertionError):
                    gate.discovery()

    def test_discovery_rejects_tampered_trace_before_selecting_or_fitting(self):
        with directory() as name:
            root = Path(name)
            source, validation = discovery_fixture(root)
            trace_path = root / "seed-0.trace.json.gz"
            with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
                gate.json.dump({"steps": []}, stream)
            with (patch.object(gate, "SOURCE", source), patch.object(gate, "VALIDATION", validation),
                  patch.object(gate, "select_steps", side_effect=AssertionError("Must first verify source"))):
                with self.assertRaises(AssertionError):
                    gate.discovery()

    def test_output_is_limited_to_dedicated_directory_and_exact_repeat(self):
        with directory() as name, patch.object(gate, "OUTPUT", Path(name) / "pose"):
            gate.check_output(Path(name) / "pose")
            gate.check_output(Path(name) / "pose-repeat")
            for other in ("pose-repeat2", "other", "pose/child"):
                with self.subTest(other=other), self.assertRaises(ValueError):
                    gate.check_output(Path(name) / other)

    def test_prepare_freezes_document_and_verifies_without_rewriting(self):
        with directory() as name:
            root = Path(name) / "pose"
            protocol = {"limits": copy.deepcopy(gate.LIMITS), "fingerprints": {"trace": "original"}}
            with patch.object(gate, "OUTPUT", root), patch.object(gate, "protocol_document", return_value=protocol):
                self.assertEqual(gate.prepare(root), protocol)
                saved = (root / "protocol.json").read_bytes()
                with patch.object(gate, "write", side_effect=AssertionError("No overwrite")):
                    self.assertEqual(gate.prepare(root), protocol)
                self.assertEqual((root / "protocol.json").read_bytes(), saved)
            changed = {**protocol, "fingerprints": {"trace": "tampered"}}
            with patch.object(gate, "OUTPUT", root), patch.object(gate, "protocol_document", return_value=changed):
                with self.assertRaisesRegex(AssertionError, "Frozen pose gate changed"):
                    gate.prepare(root)
            self.assertEqual((root / "protocol.json").read_bytes(), saved)

    def test_prepare_refuses_orphan_samples_or_summary_without_protocol(self):
        for artifact in ("samples.json", "summary.json"):
            with self.subTest(artifact=artifact), directory() as name:
                root = Path(name) / "pose"
                gate.write(root / artifact, {})
                with patch.object(gate, "OUTPUT", root), patch.object(gate, "protocol_document", return_value={}):
                    with self.assertRaises(AssertionError):
                        gate.prepare(root)
                self.assertFalse((root / "protocol.json").exists())

    def test_sample_verification_recalculates_scores_and_checks_case_identity(self):
        with directory() as name:
            loaded, cases, samples, _ = run_fixture(Path(name))
            gate.verify_sample(samples[0], cases[0], loaded[0])
            for mutate in (lambda row: row.update(step=2),
                           lambda row: row["scores"]["cold"].update(position_error=.5),
                           lambda row: row["estimates"].pop("previous_frame"),
                           lambda row: row.update(fit_seconds=-1.),
                           lambda row: row.update(fit_seconds=math.nan)):
                with self.subTest(mutate=mutate):
                    tampered = copy.deepcopy(samples[0])
                    mutate(tampered)
                    with self.assertRaises(AssertionError):
                        gate.verify_sample(tampered, cases[0], loaded[0])

    def test_completed_run_verifies_without_loading_policy_rendering_or_refitting(self):
        with directory() as name:
            root = Path(name)
            loaded, cases, samples, protocol = run_fixture(root)
            expected = gate.summarize(samples)
            gate.write(root / "samples.json", {"complete": True, "samples": samples})
            gate.write(root / "summary.json", expected)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with ExitStack() as stack:
                stack.enter_context(patch.object(gate, "prepare", return_value=protocol))
                stack.enter_context(patch.object(gate, "discovery", return_value=(loaded, cases, {})))
                for owner, name in ((gate.gym, "make"), (gate, "replay_frames"), (gate, "sample_pair"), (gate, "write")):
                    stack.enter_context(patch.object(owner, name, side_effect=AssertionError("No repeat work")))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(gate.run(root), expected)
            self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)

    def test_completed_summary_tampering_does_not_trigger_overwrite_or_new_work(self):
        with directory() as name:
            root = Path(name)
            loaded, cases, samples, protocol = run_fixture(root)
            gate.write(root / "samples.json", {"complete": True, "samples": samples})
            gate.write(root / "summary.json", {**gate.summarize(samples), "passed": False})
            with (patch.object(gate, "prepare", return_value=protocol),
                  patch.object(gate, "discovery", return_value=(loaded, cases, {})),
                  patch.object(gate.gym, "make", side_effect=AssertionError("No new work")),
                  patch.object(gate, "write", side_effect=AssertionError("No overwrite"))):
                with self.assertRaises(AssertionError):
                    gate.run(root)

    def test_resume_preserves_completed_fit_and_collects_only_missing_case(self):
        with directory() as name:
            root = Path(name)
            loaded, cases, samples, protocol = run_fixture(root)
            gate.write(root / "samples.json", {"complete": False, "samples": samples[:1]})

            pixels = np.zeros((96, 96, 3), dtype=np.uint8)
            with (patch.object(gate, "prepare", return_value=protocol),
                  patch.object(gate, "discovery", return_value=(loaded, cases, {})),
                  patch.object(gate.gym, "make", side_effect=AssertionError("Only mocked recorded replay")),
                  patch.object(gate, "replay_frames", return_value=({0: pixels, 1: pixels, 2: pixels}, 2)) as replay,
                  patch.object(gate, "sample_pair", return_value=samples[1]["estimates"]) as fit,
                  redirect_stdout(io.StringIO())):
                result = gate.run(root)
            replay.assert_called_once_with(0, loaded[0], [1, 2])
            self.assertEqual(fit.call_count, 1)
            document = gate.read(root / "samples.json")
            self.assertTrue(document["complete"])
            self.assertEqual(document["samples"][0], samples[0])
            self.assertEqual(result["samples"], 2)
            resources = gate.read(root / "resources-from-001.json")
            self.assertEqual(resources["frames_before"], 1)
            self.assertEqual(resources["frames_complete"], 2)
            self.assertEqual(resources["replayed_movements"], 2)
            self.assertEqual(resources["ai_calls"], 0)
            self.assertEqual(resources["corrective_actions"], 0)

    def test_orphan_resource_file_cannot_be_overwritten_by_resume(self):
        with directory() as name:
            root = Path(name)
            loaded, cases, samples, protocol = run_fixture(root)
            gate.write(root / "samples.json", {"complete": False, "samples": samples[:1]})
            gate.write(root / "resources-from-001.json", {"preserve": True})
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with (patch.object(gate, "prepare", return_value=protocol),
                  patch.object(gate, "discovery", return_value=(loaded, cases, {})),
                  patch.object(gate.gym, "make", side_effect=AssertionError("No collection")),
                  patch.object(gate, "write", side_effect=AssertionError("No overwrite"))):
                with self.assertRaises(AssertionError):
                    gate.run(root)
            self.assertEqual({path.name: path.read_bytes() for path in root.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
