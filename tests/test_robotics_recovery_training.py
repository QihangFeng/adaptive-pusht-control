"""CPU-only contracts for tail fine-tuning, balanced data and exact resumption."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robotics_recovery_training as training
from test_robotics_recovery_feasibility import TinyPolicy, training_batch


def directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="recovery-training-cpu-", dir=parent)


class RandomLossPolicy(TinyPolicy):
    def forward(self, batch):
        self.loss_calls += 1
        actions = batch["action"] / 100.
        timestep = torch.randint(0, 100, (1,))
        predicted = self.diffusion.unet(actions, timestep)
        return torch.nn.functional.mse_loss(predicted, torch.randn_like(predicted)), None


class ToyDataset:
    def choose(self, split, group, rng):
        return (0 if group == "expert" else 100) + int(rng.integers(10))

    def batch(self, index, device):
        result = training_batch()
        result["action"] = result["action"] + index / 10.
        return {name: value.to(device) for name, value in result.items()}


def toy_settings(steps=2):
    return {**training.SETTINGS, "optimizer_steps": steps,
            "validation_windows_per_group": 2,
            "validation_steps": [0, *range(1, steps + 1)],
            "checkpoint_steps": list(range(1, steps + 1))}


def optimizer(policy):
    training.feasibility.configure_tail_only(policy)
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.)


class TailArtifactTest(unittest.TestCase):
    def test_tail_is_cloned_cpu_only_and_digests_include_names_shapes_values(self):
        policy = TinyPolicy()
        state = training.tail_state(policy)
        expected = {name for name, _ in policy.named_parameters() if name.startswith(training.TAIL)}
        self.assertEqual(set(state), expected)
        self.assertTrue(all(value.device.type == "cpu" and not value.requires_grad for value in state.values()))
        digest = training.tail_digest(state)
        self.assertEqual(digest, training.tail_digest(dict(reversed(list(state.items())))))
        name = next(iter(state))
        state[name].add_(1.)
        self.assertNotEqual(digest, training.tail_digest(state))
        self.assertNotEqual(state[name].data_ptr(), dict(policy.named_parameters())[name].data_ptr())
        self.assertNotEqual(digest, training.tail_digest({**state, name + "different": state[name]}))

    def test_load_tail_changes_only_existing_tail_without_changing_flags_or_normalization(self):
        policy = TinyPolicy().train()
        state = training.tail_state(policy)
        for value in state.values():
            value.add_(.01)
        before = training.backbone_digest(policy)
        flags = {name: p.requires_grad for name, p in policy.named_parameters()}
        training.load_tail(policy, state)
        self.assertEqual(training.tail_digest(training.tail_state(policy)), training.tail_digest(state))
        self.assertEqual(training.backbone_digest(policy), before)
        self.assertTrue(policy.training)
        self.assertEqual({name: p.requires_grad for name, p in policy.named_parameters()}, flags)

    def test_incomplete_extra_wrong_shape_dtype_or_nonfinite_tail_is_rejected_atomically(self):
        policy = TinyPolicy()
        state = training.tail_state(policy)
        name = next(iter(state))
        invalid = [dict(list(state.items())[1:]), {**state, "diffusion.encoder.weight": torch.ones(2)},
                   {**state, name: torch.ones(1)}, {**state, name: state[name].double()},
                   {**state, name: torch.full_like(state[name], float("nan"))}]
        original = training.tail_digest(state)
        for changed in invalid:
            with self.subTest(keys=list(changed)), self.assertRaises(AssertionError):
                training.load_tail(policy, changed)
            self.assertEqual(training.tail_digest(training.tail_state(policy)), original)

    def test_saved_tail_contains_no_backbone_and_hash_is_required(self):
        with directory() as name:
            path = Path(name) / "tail.pt"
            state = training.tail_state(TinyPolicy())
            training.atomic_torch_save(path, state)
            loaded = training.read_tail(path, training.feasibility.sha(path))
            self.assertEqual(training.tail_digest(loaded), training.tail_digest(state))
            self.assertEqual(set(loaded), set(state))
            with self.assertRaises(AssertionError):
                training.read_tail(path, "0" * 64)

    def test_partial_artifact_is_never_overwritten(self):
        with directory() as name:
            path = Path(name) / "tail.pt"
            partial = path.with_suffix(".pt.tmp")
            partial.write_bytes(b"incomplete")
            with self.assertRaisesRegex(AssertionError, "Inspect incomplete"):
                training.atomic_torch_save(path, {})
            self.assertEqual(partial.read_bytes(), b"incomplete")
            self.assertFalse(path.exists())


class UpdateTest(unittest.TestCase):
    def test_four_microbatch_gradients_average_before_one_adam_update(self):
        policy = RandomLossPolicy()
        reference = copy.deepcopy(policy)
        opt, other = optimizer(policy), optimizer(reference)
        batches = [training_batch() for _ in range(4)]
        seed = torch.get_rng_state().clone()
        result = training.training_update(policy, opt, batches)
        torch.set_rng_state(seed)
        other.zero_grad(set_to_none=True)
        for batch in batches:
            loss, _ = reference(batch)
            (loss / 4).backward()
        other.step()
        for (name, actual), (name2, expected) in zip(policy.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, name2)
            self.assertTrue(torch.equal(actual, expected), name)
        self.assertEqual(result["denoiser_forward_calls"], 4)
        self.assertEqual(result["adam_state_count"], len(training.tail_state(policy)))
        self.assertEqual(len(result["microbatch_losses"]), 4)
        self.assertGreater(result["gradient_l2_norm"], 0.)
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))

    def test_frozen_parameters_and_buffers_never_change_during_update(self):
        policy = RandomLossPolicy()
        opt = optimizer(policy)
        frozen = training.backbone_digest(policy)
        tail = training.tail_digest(training.tail_state(policy))
        training.training_update(policy, opt, [training_batch()] * 4)
        self.assertEqual(training.backbone_digest(policy), frozen)
        self.assertNotEqual(training.tail_digest(training.tail_state(policy)), tail)
        self.assertTrue(all(not module.training for module in policy.modules()))

    def test_training_rejects_unfrozen_parameters_training_mode_and_wrong_optimizer(self):
        policy = RandomLossPolicy()
        opt = optimizer(policy)
        policy.diffusion.encoder.weight.requires_grad_(True)
        with self.assertRaises(AssertionError):
            training.training_update(policy, opt, [training_batch()] * 4)
        optimizer(policy)
        policy.train()
        with self.assertRaises(AssertionError):
            training.training_update(policy, opt, [training_batch()] * 4)
        optimizer(policy)
        wrong = torch.optim.AdamW(list(policy.parameters()), lr=1e-5)
        with self.assertRaises(AssertionError):
            training.training_update(policy, wrong, [training_batch()] * 4)

    def test_nonfinite_loss_stops_without_update_and_removes_hook_gradients(self):
        class BadPolicy(RandomLossPolicy):
            def forward(self, batch):
                loss, _ = super().forward(batch)
                return loss * float("nan"), None

        policy = BadPolicy()
        opt = optimizer(policy)
        original = training.tail_digest(training.tail_state(policy))
        with self.assertRaisesRegex(AssertionError, "Nonfinite"):
            training.training_update(policy, opt, [training_batch()] * 4)
        self.assertEqual(training.tail_digest(training.tail_state(policy)), original)
        self.assertFalse(policy.diffusion.unet._forward_hooks)
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))
        self.assertFalse(opt.state)


class ValidationTest(unittest.TestCase):
    def test_fixed_window_noise_validation_is_repeatable_and_preserves_training_rng(self):
        policy, dataset = RandomLossPolicy(), ToyDataset()
        optimizer(policy)
        plan = training.validation_plan(dataset, toy_settings())
        state = torch.get_rng_state().clone()
        first = training.validate(policy, dataset, plan)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(first, training.validate(policy, dataset, plan))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        self.assertEqual(first["windows_per_group"], {"expert": 2, "recovery": 2})
        self.assertEqual(first["balanced_mean_loss"], np.mean(list(first["groups"].values())))
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))
        self.assertEqual(training.validation_plan(dataset, toy_settings()), plan)

    def test_validation_rng_is_restored_even_if_forward_fails(self):
        state = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with training.fixed_noise(2, torch.device("cpu")):
                torch.rand(50)
                raise RuntimeError("synthetic validation error")
        self.assertTrue(torch.equal(torch.get_rng_state(), state))


class ResumeAndLoopTest(unittest.TestCase):
    def test_resuming_restores_exact_adam_tail_and_both_sampling_rngs(self):
        with directory() as name:
            root = Path(name)
            source = root / "source.txt"
            source.write_text("frozen", encoding="utf-8")
            bindings = {source.as_posix(): training.feasibility.sha(source)}
            policy = RandomLossPolicy()
            opt = optimizer(policy)
            rng = np.random.default_rng(42)
            training.training_update(policy, opt, [training_batch()] * 4)
            saved_tail = training.tail_state(policy)
            plan = [{"group": "expert", "window": 1, "noise_seed": 2}]
            with patch.object(torch.cuda, "is_initialized", return_value=False):
                training.save_resume(root / "resume.pt", policy, opt, rng, 1, toy_settings(), bindings, plan, {"updates": [1]})
            expected_numpy, expected_torch = rng.integers(100000), torch.rand(3)
            fresh = RandomLossPolicy()
            fresh_opt, fresh_rng = optimizer(fresh), np.random.default_rng(999)
            step, history = training.load_resume(root / "resume.pt", fresh, fresh_opt, fresh_rng,
                                                  toy_settings(), bindings, plan)
            self.assertEqual(step, 1)
            self.assertEqual(history, {"updates": [1]})
            self.assertEqual(fresh_rng.integers(100000), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(3), expected_torch))
            self.assertEqual(training.tail_digest(training.tail_state(fresh)), training.tail_digest(saved_tail))
            for state in fresh_opt.state.values():
                self.assertEqual(float(state["step"]), 1.)
            training.training_update(policy, opt, [training_batch()] * 4)

    def test_changed_binding_settings_or_validation_plan_refuses_resume(self):
        with directory() as name:
            root = Path(name)
            source = root / "source.txt"
            source.write_text("frozen", encoding="utf-8")
            bindings = {source.as_posix(): training.feasibility.sha(source)}
            policy, rng = RandomLossPolicy(), np.random.default_rng(1)
            opt = optimizer(policy)
            with patch.object(torch.cuda, "is_initialized", return_value=False):
                training.save_resume(root / "resume.pt", policy, opt, rng, 0, toy_settings(), bindings, [], {})
            for settings, different_bindings, plan in (({**toy_settings(), "lr": 2e-5}, bindings, []),
                                                       (toy_settings(), {}, []), (toy_settings(), bindings, [1])):
                with self.assertRaises(AssertionError):
                    training.load_resume(root / "resume.pt", policy, opt, rng, settings, different_bindings, plan)
            source.write_text("changed", encoding="utf-8")
            with self.assertRaises(AssertionError):
                training.load_resume(root / "resume.pt", policy, opt, rng, toy_settings(), bindings, [])

    def test_resume_optimizer_hyperparameters_cannot_override_frozen_settings(self):
        with directory() as name:
            root = Path(name)
            source = root / "source.txt"
            source.write_text("frozen", encoding="utf-8")
            bindings = {source.as_posix(): training.feasibility.sha(source)}
            policy, rng = RandomLossPolicy(), np.random.default_rng(1)
            opt = optimizer(policy)
            with patch.object(torch.cuda, "is_initialized", return_value=False):
                training.save_resume(root / "resume.pt", policy, opt, rng, 0, toy_settings(), bindings, [], {})
            saved = torch.load(root / "resume.pt", map_location="cpu", weights_only=True)
            saved["optimizer"]["param_groups"][0]["lr"] = 1e-3
            torch.save(saved, root / "resume.pt")
            with self.assertRaisesRegex(AssertionError, "Resume optimizer settings"):
                training.load_resume(root / "resume.pt", policy, opt, rng, toy_settings(), bindings, [])

    def test_bounded_loop_has_exact_mixture_checkpoints_and_final_not_validation_selection(self):
        with directory() as name:
            root = Path(name)
            source = root / "source.txt"
            source.write_text("frozen", encoding="utf-8")
            bindings = {source.as_posix(): training.feasibility.sha(source)}
            policy, dataset = RandomLossPolicy(), ToyDataset()
            original = copy.deepcopy(policy)
            with patch.object(torch.cuda, "is_initialized", return_value=False):
                result = training.train(policy, dataset, root / "trained", toy_settings(), bindings)
            self.assertTrue(result["complete"])
            self.assertEqual([row["step"] for row in result["history"]["validation"]], [0, 1, 2])
            self.assertEqual(len(result["history"]["updates"]), 2)
            for row in result["history"]["updates"]:
                self.assertEqual(row["groups"], ["expert", "recovery", "expert", "recovery"])
                self.assertEqual(row["denoiser_forward_calls"], 4)
            self.assertEqual(result["final_head"]["optimizer_step"], 2)
            self.assertNotEqual(result["original_tail_digest"], result["final_head"]["tensor_digest"])
            loaded = training.read_tail(result["final_head"]["path"], result["final_head"]["sha256"])
            self.assertEqual(training.tail_digest(loaded), training.tail_digest(training.tail_state(policy)))
            self.assertEqual(source.read_text(encoding="utf-8"), "frozen")
            # A completed local state is reused, not trained further or selected by validation.
            with (patch.object(training, "training_update") as update,
                  patch.object(training, "validate") as validate,
                  patch.object(torch.cuda, "is_initialized", return_value=False)):
                resumed = training.train(original, dataset, root / "trained", toy_settings(), bindings)
            update.assert_not_called()
            validate.assert_not_called()
            self.assertEqual(result["final_head"], resumed["final_head"])


def prepared_fixture(root):
    episodes, windows = [], []
    for index, (group, split) in enumerate((('expert', 'train'), ('expert', 'validation'),
                                            ('recovery', 'train'), ('recovery', 'validation'))):
        count = 20
        frames = count if group == "expert" else count + 1
        path = root / f"episode-{index}.npz"
        np.savez_compressed(path, pixels=np.full((frames, 96, 96, 3), index, dtype=np.uint8),
                            states=np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
                            actions=np.arange(count * 2, dtype=np.float32).reshape(count, 2))
        episodes.append({"episode_id": f"episode-{index}", "group": group, "split": split,
                         "scene": index, "file": path.as_posix(), "n_frames": frames,
                         "n_actions": count, "sha256": training.feasibility.sha(path)})
        for j in range(count - 15):
            windows.append({"episode_id": f"episode-{index}", "group": group, "split": split,
                            "first_observation_index": j, "current_observation_index": j + 1,
                            "action_start_index": j, "action_stop_index": j + 16})
    path = root / "dataset.json"
    path.write_text(json.dumps({"episodes": episodes, "windows": windows}), encoding="utf-8")
    return path


class PreparedDataTest(unittest.TestCase):
    def test_native_windows_and_lru_cache_never_use_dynamic_truth(self):
        with directory() as name:
            root = Path(name)
            dataset = training.PreparedWindows(prepared_fixture(root), cache_size=1)
            batch = dataset.batch(3, torch.device("cpu"))
            torch.testing.assert_close(batch["observation.state"], torch.tensor([[[6., 7.], [8., 9.]]]))
            torch.testing.assert_close(batch["action"][0], torch.arange(6., 38.).reshape(16, 2))
            self.assertEqual(set(batch), {"observation.image", "observation.state", "action", "action_is_pad"})
            dataset.batch(7, torch.device("cpu"))
            self.assertEqual(len(dataset.cache), 1)
            self.assertEqual(list(dataset.cache), ["episode-1"])

    def test_sampler_uses_requested_split_and_group_without_scene_overlap(self):
        with directory() as name:
            dataset = training.PreparedWindows(prepared_fixture(Path(name)))
            rng = np.random.default_rng(1)
            for split in ("train", "validation"):
                for group in training.GROUPS:
                    for _ in range(10):
                        window = dataset.windows[dataset.choose(split, group, rng)]
                        self.assertEqual((window["split"], window["group"]), (split, group))

    def test_scene_overlap_invalid_window_or_changed_archive_are_rejected(self):
        for mutation in (lambda data: data["episodes"][1].update(scene=0),
                         lambda data: data["windows"][0].update(action_start_index=1),
                         lambda data: data["windows"][0].update(current_observation_index=0),
                         lambda data: data["episodes"][0].update(sha256="changed")):
            with directory() as name:
                path = prepared_fixture(Path(name))
                data = json.loads(path.read_text(encoding="utf-8"))
                mutation(data)
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(AssertionError):
                    training.PreparedWindows(path)


class ResourceGuardTest(unittest.TestCase):
    def environment(self, other=False, available=4 * 2**30, free=3500 * 2**20):
        process = SimpleNamespace(pid=100, parents=lambda: [SimpleNamespace(pid=99)])
        processes = [SimpleNamespace(info={"pid": 100, "name": "python.exe"}),
                     SimpleNamespace(info={"pid": 99, "name": "python.exe"})]
        if other:
            processes.append(SimpleNamespace(info={"pid": 200, "name": "python.exe"}))
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(training.torch.cuda, "is_available", return_value=True))
        stack.enter_context(patch.object(training.psutil, "Process", return_value=process))
        stack.enter_context(patch.object(training.psutil, "process_iter", return_value=processes))
        stack.enter_context(patch.object(training.psutil, "virtual_memory", return_value=SimpleNamespace(available=available)))
        stack.enter_context(patch.object(training.torch.cuda, "mem_get_info", return_value=(free, 4096 * 2**20)))
        fraction = stack.enter_context(patch.object(training.torch.cuda, "set_per_process_memory_fraction"))
        return stack, fraction

    def test_python_launcher_ancestors_are_allowed_and_allocator_is_capped(self):
        stack, fraction = self.environment()
        with stack:
            result = training.gpu_guard()
        self.assertEqual(result["allocator_budget_bytes"], 2800 * 2**20)
        fraction.assert_called_once_with(2800 / 4096)

    def test_other_python_low_ram_and_low_vram_stop_before_allocator_setting(self):
        for kwargs in ({"other": True}, {"available": 2 * 2**30}, {"free": 1500 * 2**20}):
            stack, fraction = self.environment(**kwargs)
            with stack:
                with self.assertRaises(AssertionError):
                    training.gpu_guard()
            fraction.assert_not_called()

    def test_cpu_fallback_is_not_used(self):
        with (patch.object(training.torch.cuda, "is_available", return_value=False),
              patch.object(training.psutil, "Process") as process):
            with self.assertRaisesRegex(AssertionError, "No silent CPU fallback"):
                training.gpu_guard()
        process.assert_not_called()


class CliBindingTest(unittest.TestCase):
    def fixture(self, root):
        protocol, dataset, report = root / "protocol.json", root / "dataset.json", root / "training.json"
        source = root / "source.txt"
        source.write_text("frozen", encoding="utf-8")
        protocol.write_text("frozen protocol", encoding="utf-8")
        dataset.write_text("frozen dataset", encoding="utf-8")
        document = {"training": training.SETTINGS, "fingerprints": {source.as_posix(): training.feasibility.sha(source)},
                    "data_selection": {"split": "whole scenes"}}
        prepared = SimpleNamespace(manifest={"complete": True, "selection": document["data_selection"],
                                             "protocol_sha256": training.feasibility.sha(protocol)},
                                   bindings={dataset.as_posix(): training.feasibility.sha(dataset)})
        argv = ["training", "--protocol", str(protocol), "--dataset", str(dataset),
                "--report", str(report), "--artifacts-dir", str(root / "artifacts")]
        return protocol, dataset, report, document, prepared, argv

    def test_mismatched_dataset_protocol_or_split_stops_before_gpu(self):
        for field in ("protocol_sha256", "selection"):
            with directory() as name:
                protocol, dataset, report, document, prepared, argv = self.fixture(Path(name))
                prepared.manifest[field] = "changed"
                with (patch.object(sys, "argv", argv),
                      patch("robotics_recovery_pilot.verify", return_value=document),
                      patch.object(training, "PreparedWindows", return_value=prepared),
                      patch.object(training, "gpu_guard") as gpu):
                    with self.assertRaises(AssertionError):
                        training.main()
                gpu.assert_not_called()

    def test_completed_training_verifies_bound_tail_without_gpu_work(self):
        with directory() as name:
            root = Path(name)
            protocol, dataset, report, document, prepared, argv = self.fixture(root)
            tail = root / "tail.pt"
            state = training.tail_state(TinyPolicy())
            training.atomic_torch_save(tail, state)
            bindings = {**document["fingerprints"], protocol.as_posix(): training.feasibility.sha(protocol),
                        **prepared.bindings}
            saved = {"complete": True, "bindings": bindings, "settings": training.SETTINGS,
                     "protocol_sha256": training.feasibility.sha(protocol),
                     "dataset_sha256": training.feasibility.sha(dataset),
                     "final_head": {"optimizer_step": 600, "path": tail.as_posix(),
                                    "sha256": training.feasibility.sha(tail),
                                    "tensor_digest": training.tail_digest(state)}}
            report.write_text(json.dumps(saved), encoding="utf-8")
            with (patch.object(sys, "argv", argv),
                  patch("robotics_recovery_pilot.verify", return_value=document),
                  patch.object(training, "PreparedWindows", return_value=prepared),
                  patch.object(training, "gpu_guard") as gpu,
                  patch.object(training, "train") as train):
                training.main()
            gpu.assert_not_called()
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
