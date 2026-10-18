"""CPU-only contracts for the bounded recovery-training feasibility probe.

Numbered observations/actions expose temporal-label mistakes. A tiny policy
double exercises real autograd and Adam state without loading the checkpoint.
"""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_recovery_feasibility as recovery


def recording(length=30):
    rows = [{"step": step, "action": [float(step), 100. + step],
             "predicted_commands": [[-999., -999.]],
             "after": {"coverage": .50, "block_pos": [999., 999.]}}
            for step in range(1, length + 1)]
    frames = {step: {"pixels": np.full((96, 96, 3), step, dtype=np.uint8),
                     "agent_pos": np.asarray([step + .25, step + .75], dtype=np.float32)}
              for step in range(length + 1)}
    return rows, frames


def training_batch():
    rows, frames = recording()
    return recovery.make_training_sample(rows, frames, 5)


class TrainingSampleTest(unittest.TestCase):
    def test_observations_precede_future_actions_and_first_raw_target_is_historical(self):
        rows, frames = recording()
        batch = recovery.make_training_sample(rows, frames, 5)
        self.assertEqual(set(batch), {"observation.image", "observation.state", "action", "action_is_pad"})
        self.assertEqual(tuple(batch["observation.image"].shape), (1, 2, 3, 96, 96))
        self.assertEqual(tuple(batch["observation.state"].shape), (1, 2, 2))
        self.assertEqual(tuple(batch["action"].shape), (1, 16, 2))
        self.assertEqual(tuple(batch["action_is_pad"].shape), (1, 16))
        self.assertEqual(batch["action_is_pad"].dtype, torch.bool)
        self.assertFalse(bool(batch["action_is_pad"].any()))
        self.assertEqual(batch["observation.image"].dtype, torch.float32)
        torch.testing.assert_close(batch["observation.image"][0, 0], torch.full((3, 96, 96), 4 / 255.))
        torch.testing.assert_close(batch["observation.image"][0, 1], torch.full((3, 96, 96), 5 / 255.))
        torch.testing.assert_close(batch["observation.state"], torch.tensor([[[4.25, 4.75], [5.25, 5.75]]]))
        torch.testing.assert_close(batch["action"], torch.tensor([
            [[float(step), 100. + step] for step in range(5, 21)]]))
        self.assertEqual(batch["action"][0, 0].tolist(), [5., 105.])
        self.assertEqual(batch["action"][0, 1].tolist(), [6., 106.])

    def test_first_valid_and_last_complete_window_do_not_pad_or_cross_reset(self):
        rows, frames = recording(30)
        first = recovery.make_training_sample(rows, frames, 1)
        last = recovery.make_training_sample(rows, frames, 15)
        self.assertEqual(first["action"][0, 0].tolist(), [1., 101.])
        self.assertEqual(last["action"][0, -1].tolist(), [30., 130.])
        for step in (0, -1, 16, 30, 31):
            with self.subTest(step=step), self.assertRaises((AssertionError, ValueError)):
                recovery.make_training_sample(rows, frames, step)

    def test_missing_history_or_current_rgb_is_rejected(self):
        for missing in (4, 5):
            rows, frames = recording()
            del frames[missing]
            with self.subTest(missing=missing), self.assertRaises((AssertionError, KeyError, ValueError)):
                recovery.make_training_sample(rows, frames, 5)

    def test_nonconsecutive_duplicate_or_reset_steps_are_not_a_training_trajectory(self):
        for index, step in ((0, 2), (5, 1), (9, 9), (29, 31)):
            rows, frames = recording()
            rows[index]["step"] = step
            with self.subTest(index=index, step=step), self.assertRaises((AssertionError, ValueError)):
                recovery.make_training_sample(rows, frames, 5)

    def test_targets_are_executed_actions_not_discarded_predictions_or_truth(self):
        rows, frames = recording()
        expected = recovery.make_training_sample(rows, frames, 5)
        altered = copy.deepcopy(rows)
        for row in altered:
            row["predicted_commands"] = [[1234., 5678.]] * 16
            row["after"] = {"coverage": .999, "block_pos": [-999., -999.], "success": False}
        actual = recovery.make_training_sample(altered, frames, 5)
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]))
        del altered[4]["action"]
        with self.assertRaises((AssertionError, KeyError, ValueError)):
            recovery.make_training_sample(altered, frames, 5)

    def test_invalid_executed_action_or_agent_state_is_rejected(self):
        for value in ([1.], [1., 2., 3.], [float("nan"), 1.], [float("inf"), 1.]):
            rows, frames = recording()
            rows[8]["action"] = value
            with self.subTest(action=value), self.assertRaises((AssertionError, ValueError)):
                recovery.make_training_sample(rows, frames, 5)
        for value in (np.asarray([1.]), np.asarray([float("nan"), 2.])):
            rows, frames = recording()
            frames[5]["agent_pos"] = value
            with self.subTest(state=value), self.assertRaises((AssertionError, ValueError)):
                recovery.make_training_sample(rows, frames, 5)

    def test_pixels_must_be_original_uint8_camera_shape(self):
        for pixels in (np.zeros((96, 96, 3), dtype=np.float32),
                       np.zeros((95, 96, 3), dtype=np.uint8),
                       np.zeros((3, 96, 96), dtype=np.uint8)):
            rows, frames = recording()
            frames[5]["pixels"] = pixels
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype), self.assertRaises((AssertionError, ValueError)):
                recovery.make_training_sample(rows, frames, 5)

    def test_batch_construction_does_not_mutate_or_alias_recording(self):
        rows, frames = recording()
        original_rows, original_frames = copy.deepcopy(rows), copy.deepcopy(frames)
        batch = recovery.make_training_sample(rows, frames, 5)
        self.assertEqual(rows, original_rows)
        for step in frames:
            np.testing.assert_array_equal(frames[step]["pixels"], original_frames[step]["pixels"])
            np.testing.assert_array_equal(frames[step]["agent_pos"], original_frames[step]["agent_pos"])
        batch["action"][0, 0, 0] = -5.
        batch["observation.image"][0, 0, 0, 0, 0] = 1.
        batch["observation.state"][0, 0, 0] = -5.
        self.assertEqual(rows, original_rows)
        np.testing.assert_array_equal(frames[4]["pixels"], original_frames[4]["pixels"])
        np.testing.assert_array_equal(frames[4]["agent_pos"], original_frames[4]["agent_pos"])


class TinyUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(2, 4)
        self.final_conv = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))

    def forward(self, actions, timesteps=None, global_cond=None):
        return self.final_conv(self.trunk(actions))


class TinyDiffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = TinyUnet()
        self.encoder = torch.nn.Linear(2, 2)
        self.register_buffer("frozen_calibration", torch.tensor([3.]))


class TinyPolicy(torch.nn.Module):
    """A single-noise-time loss with the same named trainable boundary."""

    def __init__(self):
        super().__init__()
        self.diffusion = TinyDiffusion()
        self.loss_calls = 0
        self.random_timesteps = []

    def forward(self, batch):
        self.loss_calls += 1
        timesteps = torch.randint(0, 100, (batch["action"].shape[0],))
        self.random_timesteps.append(timesteps.clone())
        actions = batch["action"] / 100.
        prediction = self.diffusion.unet(actions, timesteps)
        loss = torch.nn.functional.mse_loss(prediction, torch.zeros_like(prediction))
        return loss, None


class TailTrainingTest(unittest.TestCase):
    def test_only_final_convolution_parameters_are_trainable_in_eval_mode(self):
        policy = TinyPolicy().train()
        selected = recovery.configure_tail_only(policy)
        expected = [name for name, _ in policy.named_parameters() if name.startswith("diffusion.unet.final_conv.")]
        self.assertEqual(selected, expected)
        self.assertTrue(expected)
        self.assertTrue(all(not module.training for module in policy.modules()))
        for name, parameter in policy.named_parameters():
            self.assertEqual(parameter.requires_grad, name in expected)

    def test_one_update_changes_only_tail_allocates_adam_and_uses_one_denoiser_forward(self):
        policy = TinyPolicy()
        selected = recovery.configure_tail_only(policy)
        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        with patch.object(torch, "save") as save:
            result = recovery.train_one_update(policy, training_batch())
        save.assert_not_called()
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertGreater(result["gradient_l2_norm"], 0.)
        self.assertTrue(np.isfinite(result["gradient_l2_norm"]))
        self.assertEqual(result["denoiser_forward_calls"], 1)
        self.assertEqual(policy.loss_calls, 1)
        self.assertEqual(len(policy.random_timesteps), 1)
        self.assertEqual(tuple(policy.random_timesteps[0].shape), (1,))
        self.assertGreater(result["adam_state_count"], 0)
        self.assertTrue(result["trainable_changed_names"])
        self.assertTrue(set(result["trainable_changed_names"]).issubset(selected))
        self.assertGreaterEqual(result["seconds"], 0.)
        for name, value in policy.state_dict().items():
            if name not in selected:
                self.assertTrue(torch.equal(value, before[name]), name)
        for name, parameter in policy.named_parameters():
            if name not in selected:
                self.assertIsNone(parameter.grad, name)

    def test_training_refuses_unfrozen_trunk(self):
        policy = TinyPolicy()
        recovery.configure_tail_only(policy)
        policy.diffusion.unet.trunk.weight.requires_grad_(True)
        with self.assertRaises((AssertionError, ValueError)):
            recovery.train_one_update(policy, training_batch())


class SnapshotRestoreTest(unittest.TestCase):
    def test_restore_recovers_tail_modes_flags_and_cpu_rng_after_training(self):
        policy = TinyPolicy().train()
        policy.diffusion.encoder.eval()
        policy.diffusion.encoder.weight.requires_grad_(False)
        original_state = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        original_modes = {name: module.training for name, module in policy.named_modules()}
        original_flags = {name: parameter.requires_grad for name, parameter in policy.named_parameters()}
        rng = torch.get_rng_state().clone()
        with (patch.object(torch.cuda, "is_initialized", return_value=False),
              patch.object(torch.cuda, "get_rng_state_all") as cuda_rng):
            snapshot = recovery.snapshot_training_state(policy)
        cuda_rng.assert_not_called()
        recovery.configure_tail_only(policy)
        recovery.train_one_update(policy, training_batch())
        recovery.restore_training_state(policy, snapshot)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[name]), name)
        self.assertEqual({name: module.training for name, module in policy.named_modules()}, original_modes)
        self.assertEqual({name: p.requires_grad for name, p in policy.named_parameters()}, original_flags)
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))


class NativeLossPathTest(unittest.TestCase):
    def test_installed_ddpm_training_loss_samples_one_time_and_calls_denoiser_once(self):
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

        model = TinyDiffusion()
        model.config = SimpleNamespace(horizon=16, n_obs_steps=2,
                                       prediction_type="epsilon", do_mask_loss_for_padding=False)
        model._prepare_global_conditioning = lambda batch: torch.zeros((1, 4))
        model.noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=100))
        model.noise_scheduler.add_noise = lambda actions, noise, timestep: actions + noise
        batch = training_batch()
        batch["observation.environment_state"] = torch.zeros((1, 2, 2))
        with (patch.object(torch, "randint", wraps=torch.randint) as random_time,
              patch.object(model.noise_scheduler, "add_noise", wraps=model.noise_scheduler.add_noise) as add_noise,
              patch.object(model.unet, "forward", wraps=model.unet.forward) as denoise):
            loss = DiffusionModel.compute_loss(model, batch)
        random_time.assert_called_once()
        self.assertEqual(random_time.call_args.kwargs["high"], 100)
        self.assertEqual(random_time.call_args.kwargs["size"], (1,))
        add_noise.assert_called_once()
        denoise.assert_called_once()
        self.assertEqual(tuple(denoise.call_args.args[1].shape), (1,))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(bool(torch.isfinite(loss)))


if __name__ == "__main__":
    unittest.main()
