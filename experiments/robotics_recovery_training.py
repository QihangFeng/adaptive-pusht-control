"""Bounded final-convolution recovery fine-tuning with original-demo rehearsal.

Only a native H16 policy's existing final convolution is updated. Persistent
artifacts contain its tail, Adam state and RNG state, never a second/full model.
The final update is deployed; validation is diagnostic, not model selection.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch

import robotics_recovery_feasibility as feasibility


TAIL = feasibility.TAIL
SETTINGS = {"optimizer_steps": 600, "accumulation": 4, "lr": 1e-5,
            "weight_decay": 0., "seed": 51000, "validation_seed": 51001,
            "validation_windows_per_group": 32, "validation_steps": [0, 200, 400, 600],
            "checkpoint_steps": [200, 400, 600], "selection": "final_600_only"}
GROUPS = ("expert", "recovery")


def tail_state(policy):
    state = {name: parameter.detach().cpu().clone()
             for name, parameter in policy.named_parameters() if name.startswith(TAIL)}
    assert state, "Existing final convolution was not found."
    return state


def tail_digest(state):
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        assert name.startswith(TAIL) and torch.is_tensor(value)
        digest.update(name.encode())
        digest.update(feasibility.tensor_hash(value).encode())
    return digest.hexdigest()


def load_tail(policy, state):
    expected = {name: parameter for name, parameter in policy.named_parameters() if name.startswith(TAIL)}
    assert set(state) == set(expected) and expected, "Tail keys differ from the existing model."
    for name, parameter in expected.items():
        value = state[name]
        assert torch.is_tensor(value) and value.shape == parameter.shape and value.dtype == parameter.dtype
        assert bool(torch.isfinite(value).all()), "Nonfinite tail tensor."
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(parameter.device))


def read_tail(path, expected_sha256):
    assert feasibility.sha(path) == expected_sha256, "Learned tail file hash changed."
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and state
    assert all(isinstance(name, str) and name.startswith(TAIL) and torch.is_tensor(value)
               and bool(torch.isfinite(value).all()) for name, value in state.items())
    return state


def backbone_digest(policy):
    return feasibility.state_fingerprint(policy, exclude_tail=True)


def verify_bindings(bindings):
    assert bindings
    for path, expected in bindings.items():
        assert feasibility.sha(path) == expected, f"Frozen training input changed: {path}"


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    assert not temporary.exists(), f"Inspect incomplete local training artifact: {temporary}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
    temporary.replace(path)


class PreparedWindows:
    """Hierarchical scene -> episode -> window sampling, with bounded RGB cache."""

    def __init__(self, manifest_path, cache_size=8):
        self.path = Path(manifest_path)
        self.manifest = feasibility.read(self.path)
        self.episodes = {row["episode_id"]: row for row in self.manifest["episodes"]}
        assert len(self.episodes) == len(self.manifest["episodes"])
        self.windows = self.manifest["windows"]
        self.cache_size, self.cache, self.groups = cache_size, OrderedDict(), {}
        assert cache_size > 0
        for index, window in enumerate(self.windows):
            episode = self.episodes[window["episode_id"]]
            assert window["group"] == episode["group"] and window["split"] == episode["split"]
            assert window["group"] in GROUPS and window["split"] in ("train", "validation")
            j = window["first_observation_index"]
            assert isinstance(j, int) and j >= 0
            assert window["current_observation_index"] == j + 1 < episode["n_frames"]
            assert window["action_start_index"] == j and window["action_stop_index"] == j + 16 <= episode["n_actions"]
            key = window["split"], window["group"]
            self.groups.setdefault(key, {}).setdefault(episode["scene"], {}).setdefault(episode["episode_id"], []).append(index)
        for group in GROUPS:
            assert self.groups.get(("train", group)) and self.groups.get(("validation", group))
            assert not set(self.groups["train", group]) & set(self.groups["validation", group])
        for episode in self.episodes.values():
            assert feasibility.sha(episode["file"]) == episode["sha256"]
        self.bindings = {self.path.as_posix(): feasibility.sha(self.path),
                         **{episode["file"]: episode["sha256"] for episode in self.episodes.values()}}

    def choose(self, split, group, rng):
        scenes = self.groups[split, group]
        scene = sorted(scenes)[int(rng.integers(len(scenes)))]
        episodes = scenes[scene]
        episode = sorted(episodes)[int(rng.integers(len(episodes)))]
        windows = episodes[episode]
        return windows[int(rng.integers(len(windows)))]

    def batch(self, index, device):
        window = self.windows[index]
        identifier = window["episode_id"]
        if identifier not in self.cache:
            episode = self.episodes[identifier]
            with np.load(episode["file"], allow_pickle=False) as archive:
                pixels, states, actions = (archive[key].copy() for key in ("pixels", "states", "actions"))
            assert pixels.shape == (episode["n_frames"], 96, 96, 3) and pixels.dtype == np.uint8
            assert states.shape == (episode["n_frames"], 2) and actions.shape == (episode["n_actions"], 2)
            assert states.dtype == actions.dtype == np.float32
            assert np.isfinite(states).all() and np.isfinite(actions).all()
            rows = [{"step": step + 1, "action": action} for step, action in enumerate(actions)]
            self.cache[identifier] = pixels, states, rows
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(identifier)
        pixels, states, rows = self.cache[identifier]
        j = window["first_observation_index"]
        frames = {step: {"pixels": pixels[step], "agent_pos": states[step]} for step in (j, j + 1)}
        sample = feasibility.make_training_sample(rows, frames, j + 1)
        return {key: value.to(device) for key, value in sample.items()}


def validation_plan(dataset, settings):
    rng = np.random.default_rng(settings["validation_seed"])
    return [{"group": group, "window": dataset.choose("validation", group, rng),
             "noise_seed": settings["validation_seed"] + 1000 + index + group_index * 1000}
            for group_index, group in enumerate(GROUPS)
            for index in range(settings["validation_windows_per_group"])]


@contextmanager
def fixed_noise(seed, device):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)


def validate(policy, dataset, plan):
    assert all(not module.training for module in policy.modules())
    device = next(policy.parameters()).device
    losses = {group: [] for group in GROUPS}
    with torch.no_grad():
        for item in plan:
            batch = dataset.batch(item["window"], device)
            with fixed_noise(item["noise_seed"], device):
                loss, _ = policy(batch)
            assert loss.ndim == 0 and bool(torch.isfinite(loss))
            losses[item["group"]].append(float(loss))
    assert all(losses.values())
    means = {group: float(np.mean(values)) for group, values in losses.items()}
    return {"groups": means, "balanced_mean_loss": float(np.mean(list(means.values()))),
            "windows_per_group": {group: len(values) for group, values in losses.items()}}


def training_update(policy, optimizer, batches):
    assert batches and all(not module.training for module in policy.modules())
    selected = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    assert selected and all(name.startswith(TAIL) for name, _ in selected)
    parameters = [parameter for _, parameter in selected]
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in parameters}
    calls = []
    hook = policy.diffusion.unet.register_forward_hook(lambda *_: calls.append(1))
    losses, started = [], time.perf_counter()
    try:
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            assert batch["action"].shape == (1, 16, 2)
            assert all(not value.requires_grad for value in batch.values())
            loss, _ = policy(batch)
            assert loss.ndim == 0 and bool(torch.isfinite(loss)), "Nonfinite supervised loss."
            losses.append(float(loss.detach()))
            (loss / len(batches)).backward()
        assert len(calls) == len(batches), "One denoiser call per microbatch, not a reverse sampling chain."
        assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
        assert all(parameter.grad is None for _, parameter in policy.named_parameters() if not parameter.requires_grad)
        norm = float(torch.sqrt(sum(parameter.grad.detach().float().square().sum() for parameter in parameters)))
        assert math.isfinite(norm) and norm > 0
        optimizer.step()
        assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
        return {"loss": float(np.mean(losses)), "microbatch_losses": losses,
                "gradient_l2_norm": norm, "denoiser_forward_calls": len(calls),
                "adam_state_count": len(optimizer.state), "seconds": time.perf_counter() - started}
    finally:
        hook.remove()
        optimizer.zero_grad(set_to_none=True)


def save_resume(path, policy, optimizer, rng, step, settings, bindings, plan, history):
    assert all(parameter.grad is None for parameter in policy.parameters())
    atomic_torch_save(path, {"format": 1, "step": step, "settings": settings, "bindings": bindings,
                            "tail": tail_state(policy), "optimizer": optimizer.state_dict(),
                            "numpy_rng": rng.bit_generator.state,
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                            "validation_plan": plan, "history": history})


def load_resume(path, policy, optimizer, rng, settings, bindings, plan):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["format"] == 1 and saved["settings"] == settings and saved["bindings"] == bindings
    assert saved["validation_plan"] == plan
    assert 0 <= saved["step"] <= settings["optimizer_steps"]
    verify_bindings(bindings)
    load_tail(policy, saved["tail"])
    optimizer.load_state_dict(saved["optimizer"])
    assert all(group["lr"] == settings["lr"] and group["weight_decay"] == settings["weight_decay"]
               for group in optimizer.param_groups), "Resume optimizer settings differ from the frozen protocol."
    rng.bit_generator.state = saved["numpy_rng"]
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"] is not None:
        assert torch.cuda.is_initialized(), "Resume needs the original CUDA RNG device."
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["step"], saved["history"]


def train(policy, dataset, artifacts_dir, settings, bindings):
    """Resume only complete optimizer boundaries; final step is never selected by test results."""
    started = time.perf_counter()
    verify_bindings(bindings)
    settings = dict(settings)
    assert settings["accumulation"] == 4 and settings["optimizer_steps"] > 0
    assert settings["weight_decay"] == 0 and settings["lr"] > 0
    assert settings["validation_steps"][0] == 0 and settings["validation_steps"][-1] == settings["optimizer_steps"]
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    selected = feasibility.configure_tail_only(policy)
    original_backbone = backbone_digest(policy)
    original_tail = tail_digest(tail_state(policy))
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=settings["lr"], weight_decay=0.)
    rng = np.random.default_rng(settings["seed"])
    plan = validation_plan(dataset, settings)
    torch.random.default_generator.manual_seed(settings["seed"])
    device = next(policy.parameters()).device
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(settings["seed"])
    resume = artifacts_dir / "resume.pt"
    history = {"updates": [], "validation": []}
    start = 0
    if resume.exists():
        start, history = load_resume(resume, policy, optimizer, rng, settings, bindings, plan)
    else:
        assert not any(artifacts_dir.iterdir()), "Unbound training artifacts require inspection."
        history["validation"].append({"step": 0, **validate(policy, dataset, plan)})
        save_resume(resume, policy, optimizer, rng, 0, settings, bindings, plan, history)
    for step in range(start + 1, settings["optimizer_steps"] + 1):
        groups = ["expert", "recovery", "expert", "recovery"]
        indices = [dataset.choose("train", group, rng) for group in groups]
        batches = [dataset.batch(index, device) for index in indices]
        update = training_update(policy, optimizer, batches)
        history["updates"].append({"step": step, "groups": groups, "windows": indices, **update})
        if step in settings["validation_steps"]:
            assert backbone_digest(policy) == original_backbone, "Frozen backbone/normalization changed."
            history["validation"].append({"step": step, **validate(policy, dataset, plan)})
            verify_bindings(bindings)
            save_resume(resume, policy, optimizer, rng, step, settings, bindings, plan, history)
        if step % 20 == 0:
            print(f"Tail training {step}/{settings['optimizer_steps']}: loss={update['loss']:.6f}", flush=True)
    assert backbone_digest(policy) == original_backbone
    verify_bindings(bindings)
    assert len(history["updates"]) == settings["optimizer_steps"]
    assert sum(row["denoiser_forward_calls"] for row in history["updates"]) == settings["optimizer_steps"] * settings["accumulation"]
    assert [item["step"] for item in history["validation"]] == settings["validation_steps"]
    final = artifacts_dir / "final-tail.pt"
    state = tail_state(policy)
    assert tail_digest(state) != original_tail, "The final tail must contain an actual learned update."
    if final.exists():
        assert tail_digest(read_tail(final, feasibility.sha(final))) == tail_digest(state)
    else:
        atomic_torch_save(final, state)
    return {"complete": True, "settings": settings, "bindings": bindings,
            "wall_seconds": time.perf_counter() - started,
            "timing_scope": "Training phase including validation and checkpoint I/O; excludes download, dataset preparation and model loading.",
            "trainable_names": selected,
            "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in policy.parameters()),
            "original_tail_digest": original_tail, "backbone_digest": original_backbone,
            "frozen_backbone_and_normalization_unchanged": True,
            "validation_plan": plan, "history": history,
            "final_head": {"path": final.as_posix(), "sha256": feasibility.sha(final),
                           "tensor_digest": tail_digest(state), "optimizer_step": settings["optimizer_steps"]},
            "resume": {"path": resume.as_posix(), "sha256": feasibility.sha(resume)},
            "selection": "Final optimizer step only; validation losses never select a checkpoint."}


def gpu_guard():
    assert torch.cuda.is_available(), "No silent CPU fallback for the training pilot."
    own = psutil.Process(os.getpid())
    allowed = {own.pid, *(process.pid for process in own.parents())}
    other = [process.info["pid"] for process in psutil.process_iter(["pid", "name"])
             if (process.info["name"] or "").lower() in ("python", "python.exe", "pythonw.exe")
             and process.info["pid"] not in allowed]
    assert not other, f"Inspect other Python processes before using the GPU: {other}"
    assert psutil.virtual_memory().available >= 3 * 2**30, "Less than 3GiB available system RAM."
    free, total = torch.cuda.mem_get_info()
    budget = min(2800 * 2**20, free - 384 * 2**20)
    assert budget >= 1600 * 2**20, "Insufficient free GPU memory for the bounded pilot."
    torch.cuda.set_per_process_memory_fraction(budget / total)
    return {"allocator_budget_bytes": budget, "free_gpu_bytes_before_loading": free,
            "total_gpu_bytes": total, "other_python_processes": other}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("results/robotics-recovery-pilot/protocol.json"))
    parser.add_argument("--dataset", type=Path, default=Path("results/robotics-recovery-pilot/dataset.json"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("data/prepared/robotics/recovery-pilot/training"))
    parser.add_argument("--report", type=Path, default=Path("results/robotics-recovery-pilot/training.json"))
    args = parser.parse_args()
    import robotics_recovery_pilot as pilot
    protocol = pilot.verify(args.protocol.parent)
    assert protocol["training"] == SETTINGS
    bindings = dict(protocol["fingerprints"])
    bindings[args.protocol.as_posix()] = feasibility.sha(args.protocol)
    bindings[args.dataset.as_posix()] = feasibility.sha(args.dataset)
    verify_bindings(bindings)
    dataset = PreparedWindows(args.dataset)
    assert dataset.manifest["complete"]
    assert dataset.manifest["protocol_sha256"] == feasibility.sha(args.protocol)
    assert dataset.manifest["selection"] == protocol["data_selection"]
    bindings.update(dataset.bindings)
    if args.report.exists():
        saved = feasibility.read(args.report)
        assert saved["complete"] and saved["bindings"] == bindings and saved["settings"] == SETTINGS
        assert saved["protocol_sha256"] == feasibility.sha(args.protocol)
        assert saved["dataset_sha256"] == feasibility.sha(args.dataset)
        assert saved["final_head"]["optimizer_step"] == SETTINGS["optimizer_steps"]
        assert feasibility.sha(saved["final_head"]["path"]) == saved["final_head"]["sha256"]
        assert tail_digest(read_tail(saved["final_head"]["path"], saved["final_head"]["sha256"])) == saved["final_head"]["tensor_digest"]
        print("Completed training remains unchanged; no GPU work repeated.", flush=True)
        return
    guard = gpu_guard()
    torch.set_num_threads(4)
    from robotics_baseline_check import load_policy
    policy, _ = load_policy(Path("data/raw/robotics/diffusion_pusht"))
    with feasibility.MemorySampler() as memory:
        torch.cuda.reset_peak_memory_stats()
        result = train(policy, dataset, args.artifacts_dir, SETTINGS, bindings)
    result["protocol_sha256"] = feasibility.sha(args.protocol)
    result["dataset_sha256"] = feasibility.sha(args.dataset)
    pilot.verify(args.protocol.parent)
    result["resources"] = {"peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "peak_process_rss_bytes": memory.peak_rss,
                           "minimum_system_available_bytes": memory.minimum_available,
                           "guard": guard}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": True, "final_head": result["final_head"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
