"""CPU-only dataset alignment, decoded-format and protected-output tests."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_recovery_dataset as dataset


def temporary():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="recovery-dataset-test-", dir=parent)


def expert(count=20, episode=0):
    columns = {"frame_index": list(range(count)), "episode_index": [episode] * count,
               "timestamp": [i / 10 for i in range(count)],
               "observation.state": [[float(i), 100.] for i in range(count)],
               "action": [[float(i) + .5, 100.] for i in range(count)],
               "next.done": [False] * (count - 1) + [True]}
    frames = [np.full((96, 96, 3), i, dtype=np.uint8) for i in range(count)]
    return columns, frames, [i / 10 for i in range(count)]


class WindowAlignmentTests(unittest.TestCase):
    def test_native_expert_observation_and_label_alignment(self):
        windows = list(dataset.sample_windows("expert-0", "expert", "train", 20, 20))
        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[0], {"episode_id": "expert-0", "group": "expert", "split": "train",
                                     "first_observation_index": 0, "current_observation_index": 1,
                                     "action_start_index": 0, "action_stop_index": 16})
        self.assertEqual(windows[-1]["action_stop_index"], 20)

    def test_recovery_preserves_full_prefix_and_post_branch_anchor(self):
        windows = list(dataset.sample_windows("r", "recovery", "train", 285, 284, 157))
        self.assertEqual(len(windows), 113)
        self.assertEqual(windows[0]["first_observation_index"], 156)
        self.assertEqual(windows[0]["current_observation_index"], 157)
        self.assertEqual(windows[0]["action_start_index"], 156)
        self.assertEqual(windows[0]["action_stop_index"], 172)
        self.assertEqual(windows[-1]["current_observation_index"], 269)
        self.assertEqual(windows[-1]["action_stop_index"], 284)

    def test_sixteen_executed_labels_give_one_window_not_sixteen_future_steps(self):
        self.assertEqual(len(list(dataset.sample_windows("e", "expert", "train", 16, 16))), 1)
        self.assertEqual(len(list(dataset.sample_windows("r", "recovery", "train", 17, 16))), 1)

    def test_no_end_padding_when_fewer_than_sixteen_labels(self):
        self.assertEqual(list(dataset.sample_windows("e", "expert", "train", 15, 15)), [])

    def test_observation_arrays_cannot_have_arbitrary_length(self):
        with self.assertRaises(AssertionError): list(dataset.sample_windows("e", "expert", "train", 22, 20))

    def test_selection_disjoint_and_expert_rehearsal_not_baseline_replacement(self):
        contract = dataset.selection_contract()
        self.assertFalse(set(contract["recovery_train_scenes"]) & set(contract["recovery_validation_scenes"]))
        self.assertEqual(set(contract["recovery_train_scenes"] + contract["recovery_validation_scenes"]),
                         {2, 3, 4, 6, 7, 15, 16, 18})
        self.assertEqual(contract["expert_train_episodes"], list(range(64)))
        self.assertEqual(contract["expert_validation_episodes"], list(range(64, 80)))
        self.assertEqual(contract["expert_revision"], "a16737ffb6c8dbe935291b9fd2f36d14c9bf0820")


class ExpertDecodedFormatTests(unittest.TestCase):
    def test_valid_decoding_has_expected_dtypes(self):
        columns, frames, pts = expert()
        pixels, states, actions = dataset.validate_expert_arrays(columns, frames, pts, 0)
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertEqual(states.dtype, np.float32)
        self.assertEqual(actions.dtype, np.float32)
        self.assertEqual(pixels.shape, (20, 96, 96, 3))

    def test_mismatching_decoded_frame_count_rejected(self):
        columns, frames, pts = expert()
        with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames[:-1], pts[:-1], 0)

    def test_missing_or_repeated_frame_index_rejected(self):
        columns, frames, pts = expert()
        columns["frame_index"][4] = 3
        with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_cross_episode_rows_rejected(self):
        columns, frames, pts = expert()
        columns["episode_index"][5] = 1
        with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_video_pts_and_parquet_timestamps_must_both_match(self):
        for key in ["pts", "parquet"]:
            with self.subTest(key=key):
                columns, frames, pts = expert()
                if key == "pts": pts[3] += .1
                else: columns["timestamp"][3] += .1
                with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_float32_timestamp_rounding_is_allowed(self):
        columns, frames, pts = expert()
        columns["timestamp"] = np.asarray(columns["timestamp"], dtype=np.float32).tolist()
        dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_early_terminal_flag_rejected(self):
        columns, frames, pts = expert()
        columns["next.done"][4] = True
        with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_duplicate_terminal_endpoint_is_truncated_after_full_validation(self):
        columns, frames, pts = expert()
        columns["next.done"][-2] = True
        arrays = dataset.validate_expert_arrays(columns, frames, pts, 0)
        self.assertEqual([len(value) for value in arrays], [19, 19, 19])
        self.assertEqual(len(list(dataset.sample_windows("e", "expert", "train", 19, 19))), 4)
        metadata = dataset.expert_terminal_handling(columns)
        self.assertEqual(metadata["original_rows"], 20)
        self.assertEqual(metadata["retained_rows"], 19)
        self.assertEqual(metadata["discarded_rows"], 1)
        self.assertEqual(metadata["first_done_row_index"], 18)
        self.assertEqual(metadata["original_done_row_indices"], [18, 19])

    def test_missing_done_retains_the_whole_demonstration(self):
        columns, frames, pts = expert()
        columns["next.done"] = [False] * 20
        arrays = dataset.validate_expert_arrays(columns, frames, pts, 0)
        self.assertEqual([len(value) for value in arrays], [20, 20, 20])
        self.assertIsNone(dataset.expert_terminal_handling(columns)["first_done_row_index"])

    def test_success_flags_do_not_filter_or_truncate_expert_data(self):
        columns, frames, pts = expert()
        columns["next.done"][-2] = True
        columns["next.success"] = [False] * 20
        first = dataset.validate_expert_arrays(columns, frames, pts, 0)
        columns["next.success"] = [True] * 20
        second = dataset.validate_expert_arrays(columns, frames, pts, 0)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(dataset.expert_terminal_handling(columns)["success_used_for_filtering"])

    def test_bad_action_in_discarded_endpoint_still_fails_full_validation(self):
        columns, frames, pts = expert()
        columns["next.done"][-2] = True
        columns["action"][-1][0] = np.nan
        with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_bad_pts_in_discarded_endpoint_still_fails_full_validation(self):
        columns, frames, pts = expert()
        columns["next.done"][-2] = True
        pts[-1] += .1
        with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_missing_terminal_metadata_rows_are_not_ignored(self):
        columns, frames, pts = expert()
        columns["next.done"].pop()
        with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_nonfinite_or_wrong_dimension_states_actions_rejected(self):
        for field in ["observation.state", "action"]:
            for changed in [[np.nan, 0.], [np.inf, 0.], [1., 2., 3.]]:
                with self.subTest(field=field, changed=changed):
                    columns, frames, pts = expert()
                    columns[field] = [changed] * len(frames)
                    with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_wrong_rgb_shape_or_dtype_rejected(self):
        for array in [np.zeros((96, 96), dtype=np.uint8), np.zeros((96, 96, 3), dtype=np.float32)]:
            columns, frames, pts = expert()
            frames = [array] * len(frames)
            with self.assertRaises(AssertionError): dataset.validate_expert_arrays(columns, frames, pts, 0)

    def test_expert_paths_are_pinned_first_eighty_only(self):
        self.assertEqual(dataset.expert_paths(79)[0], "data/chunk-000/episode_000079.parquet")
        for index in [-1, 80, 206]:
            with self.assertRaises(AssertionError): dataset.expert_paths(index)


class ReplayTests(unittest.TestCase):
    def fake_replay(self, count=20):
        def state(i):
            pixels = np.full((96, 96, 3), i, dtype=np.uint8)
            return {"pusher_pos": [float(i), 10.], "coverage": .96 if i == count else .1,
                    "image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
        initial = state(0)
        rows = [{"step": i, "action": [float(i), 10.], "new_prediction": (i - 1) % 8 == 0,
                 "after": state(i)} for i in range(1, count + 1)]

        class Environment:
            def __init__(self): self.index = 0; self.closed = False
            def obs(self): return {"pixels": np.full((96, 96, 3), self.index, dtype=np.uint8)}
            def reset(self, seed): self.seed = seed; return self.obs(), {}
            def step(self, action):
                self.index += 1
                return self.obs(), 0., self.index == count, False, {}
            def close(self): self.closed = True

        return initial, rows, Environment(), state

    def test_replay_has_one_reset_and_observation_after_every_executed_command(self):
        initial, rows, env, state = self.fake_replay()
        record = {"id": "r", "scene": 2, "branch_after_step": 8}
        with patch.object(dataset, "load_record", return_value=(initial, rows)), \
                patch("gymnasium.make", return_value=env), \
                patch.object(dataset, "physical_state", side_effect=lambda e, o: state(e.index)):
            (pixels, states, actions), verification = dataset.replay_record(record)
        self.assertEqual((len(pixels), len(states), len(actions)), (21, 21, 20))
        self.assertFalse(verification["reset_at_branch"])
        self.assertEqual(verification["verified_recorded_movements"], 20)
        self.assertEqual(env.seed, 2)
        self.assertTrue(env.closed)

    def test_replay_fails_hard_and_closes_environment_on_rgb_mismatch(self):
        initial, rows, env, state = self.fake_replay()
        rows[5]["after"]["image_sha256"] = "f" * 64
        with patch.object(dataset, "load_record", return_value=(initial, rows)), \
                patch("gymnasium.make", return_value=env), \
                patch.object(dataset, "physical_state", side_effect=lambda e, o: state(e.index)):
            with self.assertRaises(AssertionError): dataset.replay_record({"id": "r", "scene": 2, "branch_after_step": 8})
        self.assertTrue(env.closed)


class StorageAndDownloadTests(unittest.TestCase):
    def test_npz_is_lossless_and_deterministic(self):
        columns, frames, pts = expert()
        arrays = dataset.validate_expert_arrays(columns, frames, pts, 0)
        with temporary() as directory:
            first, second = Path(directory) / "a.npz", Path(directory) / "b.npz"
            dataset.deterministic_npz(first, *arrays)
            dataset.deterministic_npz(second, *arrays)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with np.load(first, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {"pixels", "states", "actions"})
                for key, original in zip(["pixels", "states", "actions"], arrays):
                    np.testing.assert_array_equal(saved[key], original)

    def test_prepared_npz_cannot_be_overwritten(self):
        with temporary() as directory:
            path = Path(directory) / "a.npz"
            path.write_bytes(b"existing")
            with self.assertRaises(AssertionError): dataset.deterministic_npz(path, np.zeros(1), np.zeros(1), np.zeros(1))
            self.assertEqual(path.read_bytes(), b"existing")

    def test_matching_json_is_verified_and_changed_json_rejected(self):
        with temporary() as directory:
            path = Path(directory) / "value.json"
            dataset.write_new(path, {"a": 1})
            first = path.read_bytes()
            dataset.write_new(path, {"a": 1})
            self.assertEqual(first, path.read_bytes())
            with self.assertRaises(AssertionError): dataset.write_new(path, {"a": 2})

    def test_verified_download_and_existing_file_reuse(self):
        payload = b"small public expert file"
        digest = hashlib.sha256(payload).hexdigest()
        with temporary() as directory:
            path = Path(directory) / "source.bin"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as request:
                result = dataset.fetch_public("https://example.org", path, digest, len(payload))
                self.assertEqual(request.call_count, 1)
            with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected new request")):
                self.assertEqual(dataset.fetch_public("https://example.org", path, digest, len(payload)), result)

    def test_bad_download_never_creates_partial_or_overwrites(self):
        with temporary() as directory:
            path = Path(directory) / "source.bin"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(AssertionError): dataset.fetch_public("https://example.org", path, "f" * 64)
            self.assertFalse(path.exists())
            path.write_bytes(b"existing bad")
            with self.assertRaises(AssertionError): dataset.fetch_public("https://example.org", path, "f" * 64)
            self.assertEqual(path.read_bytes(), b"existing bad")

    def test_network_retry_is_bounded_without_writing_incomplete_file(self):
        with temporary() as directory:
            path = Path(directory) / "source.bin"
            with patch("urllib.request.urlopen", side_effect=OSError("network unavailable")) as request, \
                    patch.object(dataset.time, "sleep"):
                with self.assertRaises(OSError): dataset.fetch_public("https://example.org", path)
                self.assertEqual(request.call_count, 3)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
