"""CPU checks; no pretrained-model load or GPU inference."""

import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from robotics_baseline_check import (
    local_probes, physical_state, passive_simulator_test, replay_prefix, screen_candidates, states_match, vertex_motion,
)
import gymnasium as gym
import numpy as np


def row(step, contact=0, chunk=1, pending=4, command=10, motion=0, gap=2):
    before = {"block_vertices": [[0, 0], [10, 0]], "coverage": 0.5, "pusher_block_gap": gap}
    after = {"block_vertices": [[motion, 0], [10 + motion, 0]],
             "coverage": 0.5, "pusher_block_gap": gap}
    return {"step": step, "chunk": chunk, "pending_actions": pending,
            "before": before, "after": after, "pair_contact_points": contact,
            "command_distance": command}


class CandidateRulesTest(unittest.TestCase):
    def test_approach_is_not_contact_loss(self):
        self.assertEqual(screen_candidates([row(i) for i in range(1, 5)]), [])

    def test_prior_contact_same_chunk_required(self):
        rows = [row(1, contact=1)] + [row(i) for i in range(2, 5)]
        self.assertEqual(screen_candidates(rows)[0]["label"], "contact_loss_candidate")
        rows[0]["chunk"] = 0
        self.assertEqual(screen_candidates(rows), [])

    def test_far_away_no_contact_is_not_flagged(self):
        rows = [row(1, contact=1)] + [row(i, gap=30) for i in range(2, 5)]
        self.assertEqual(screen_candidates(rows), [])

    def test_contact_stall(self):
        rows = [row(i, contact=1) for i in range(1, 4)]
        self.assertEqual(screen_candidates(rows)[0]["label"], "contact_stall_candidate")
        moving = copy.deepcopy(rows)
        moving[-1]["after"]["block_vertices"][0][0] = 5
        self.assertEqual(screen_candidates(moving), [])

    def test_rotation_is_motion(self):
        a = {"block_vertices": [[0, 0], [10, 0]]}
        b = {"block_vertices": [[0, 0], [0, 10]]}
        self.assertGreater(vertex_motion(a, b), 10)

    def test_low_command_distance_not_flagged(self):
        self.assertEqual(screen_candidates([row(i, contact=1, command=1) for i in range(1, 4)]), [])

    def test_chunk_boundary_and_empty_remainder_not_flagged(self):
        self.assertEqual(screen_candidates([row(1, contact=1, chunk=1),
                                           row(2, contact=1, chunk=2), row(3, contact=1, chunk=2)]), [])
        self.assertEqual(screen_candidates([row(i, contact=1, pending=1) for i in range(1, 4)]), [])

    def test_duplicate_flags_suppressed(self):
        self.assertEqual(len(screen_candidates([row(i, contact=1) for i in range(1, 7)])), 1)

    def test_passive_logging_preserves_physics(self):
        self.assertEqual(passive_simulator_test()["identical_steps"], 100)

    def test_replay_preserves_branch_state_and_observation_history(self):
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
        env.reset(seed=2)
        rows, observations = [], []
        for index in range(6):
            action = np.array([250 + index, 200 + index], dtype=np.float32)
            observation, *_ = env.step(action)
            observations.append(observation)
            rows.append({"action": action.tolist(), "after": physical_state(env, observation)})
        replayed, current, history = replay_prefix(2, rows, 6)
        self.assertEqual(physical_state(replayed, current), rows[-1]["after"])
        self.assertEqual(len(history), 2)
        np.testing.assert_array_equal(history[0]["pixels"], observations[3]["pixels"])
        np.testing.assert_array_equal(history[1]["pixels"], observations[4]["pixels"])
        env.close()
        replayed.close()

    def test_probe_skips_a_candidate_at_episode_end(self):
        candidate = {"end_step": 1, "pending_actions": 2}
        episode = {"seed": 2, "candidates": [candidate], "trace": "trace.json.gz"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with gzip.open(path / episode["trace"], "wt", encoding="utf-8") as stream:
                json.dump({"steps": [{"step": 1}]}, stream)
            result = local_probes(None, [episode], path)
        self.assertTrue(result[0]["skipped"])

    def test_only_coverage_rounding_is_tolerated(self):
        a = {"coverage": 0.5, "pusher_pos": [1, 2], "image_sha256": "same"}
        b = dict(a, coverage=0.5 + 1e-16)
        self.assertTrue(states_match(a, b))
        self.assertFalse(states_match(a, dict(a, coverage=0.51)))
        self.assertFalse(states_match(a, dict(a, pusher_pos=[1, 2.001])))
        self.assertFalse(states_match(a, dict(a, image_sha256="different")))


if __name__ == "__main__":
    unittest.main()
