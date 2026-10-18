"""One reversible, real-data supervised update; not recovery-policy training.

CPU replay reconstructs an existing successful continuation. Only the current
RGB/pusher observation pair and executed commands enter the training loss.
The original checkpoint, frozen experiments and milestone controller stay intact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import psutil
import torch

from robotics_feasibility import MemorySampler, WEIGHTS_SHA256, gpu_status


OUTPUT = Path("results/robotics-recovery-feasibility")
PREFIX = Path("results/robotics-baseline-check/seed-2.trace.json.gz")
RECOVERY = Path("results/robotics-failure-patterns/recovery/seed-2-sample-2001-replan_once.trace.json.gz")
SIDECAR = RECOVERY.with_name(RECOVERY.name.replace(".trace.json.gz", ".json"))
TAIL = "diffusion.unet.final_conv."
OBSERVATION_STEP = 157


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def validate_rows(rows):
    if not rows or [r.get("step") for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Executed movements must be consecutive from one, with no reset boundary.")
    for row in rows:
        action = np.asarray(row.get("action"), dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Every label must be an actual, finite executed two-coordinate command.")


def make_training_sample(rows, frames, observation_step):
    """At state after t movements: observations t-1,t; commands t..t+15.

    Raw target slot zero is the preceding command. Slot one is the next command,
    matching the checkpoint's n_obs_steps=2, horizon=16 convention. No padding.
    """
    validate_rows(rows)
    t = observation_step
    if isinstance(t, bool) or not isinstance(t, int) or t < 1 or t + 15 > len(rows):
        raise ValueError("Need two real observations and all sixteen executed action labels.")
    images, positions = [], []
    for step in (t - 1, t):
        observation = frames[step]
        pixels = np.asarray(observation["pixels"])
        position = np.asarray(observation["agent_pos"], dtype=np.float32)
        if pixels.shape != (96, 96, 3) or pixels.dtype != np.uint8:
            raise ValueError("Expected the unchanged 96x96 uint8 RGB observation.")
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("Expected the pusher's own finite two-coordinate observation.")
        images.append(torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.)
        positions.append(torch.from_numpy(position.copy()))
    actions = np.asarray([r["action"] for r in rows[t - 1:t + 15]], dtype=np.float32)
    return {"observation.image": torch.stack(images).unsqueeze(0),
            "observation.state": torch.stack(positions).unsqueeze(0),
            "action": torch.from_numpy(actions).unsqueeze(0),
            "action_is_pad": torch.zeros((1, 16), dtype=torch.bool)}


def configure_tail_only(policy):
    policy.eval()  # Preserve frozen image encoding, including deterministic center crop.
    selected = []
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith(TAIL))
        if parameter.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError("The existing final convolution block was not found.")
    return selected


def snapshot_training_state(policy):
    if any(p.grad is not None for p in policy.parameters()):
        raise ValueError("This isolated gate requires no pre-existing parameter gradients.")
    return {"tail": {n: p.detach().cpu().clone() for n, p in policy.named_parameters()
                     if n.startswith(TAIL)},
            "requires_grad": {n: p.requires_grad for n, p in policy.named_parameters()},
            "modes": {n: m.training for n, m in policy.named_modules()},
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_training_state(policy, snapshot):
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in snapshot["tail"]:
                parameter.copy_(snapshot["tail"][name].to(parameter.device))
            parameter.grad = None
            parameter.requires_grad_(snapshot["requires_grad"][name])
    # Set each flag directly, preserving mixed child modes without recursive overrides.
    for name, module in policy.named_modules():
        module.training = snapshot["modes"][name]
    torch.set_rng_state(snapshot["cpu_rng"])
    if snapshot["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda_rng"])


def tensor_hash(tensor):
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{cpu.dtype}:{tuple(cpu.shape)}:".encode())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def state_fingerprint(policy, *, exclude_tail=False):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        if exclude_tail and name.startswith(TAIL):
            continue
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def train_one_update(policy, batch, lr=1e-5):
    """One random-timestep denoising loss, backward and first AdamW update.

    This intentionally does not call an action-generation/inference function.
    The caller snapshots/restores state; no learned checkpoint is saved.
    """
    names = [n for n, p in policy.named_parameters() if p.requires_grad]
    if not names or any(not n.startswith(TAIL) for n in names):
        raise ValueError("Only the existing final convolution block may be trainable.")
    if any(module.training for module in policy.modules()):
        raise ValueError("Keep this bounded frozen-encoder gate in evaluation mode.")
    if any(value.requires_grad for value in batch.values()):
        raise ValueError("Training inputs must not retain a model-backbone gradient graph.")
    parameters = [p for p in policy.parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
    calls = []

    def record_call(_module, inputs, _output):
        calls.append(inputs[1].detach().cpu().tolist() if len(inputs) > 1 else None)

    hook = policy.diffusion.unet.register_forward_hook(record_call)
    device = parameters[0].device
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy(batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Nonfinite or nonscalar supervised denoising loss.")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("Every selected tail parameter needs a finite gradient.")
        norm = float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in parameters)))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Need a finite, nonzero training gradient.")
        optimizer.step()  # Allocates real Adam state; forward/backward alone is insufficient.
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        changed = [n for n, p in policy.named_parameters()
                   if n in before and not torch.equal(before[n], p.detach())]
        if not changed or len(calls) != 1:
            raise ValueError("Expected an actual tail update and one denoiser forward, not a sampling chain.")
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError("Optimizer produced nonfinite parameters.")
        return {"loss": float(loss.detach()), "gradient_l2_norm": norm,
                "trainable_changed_names": changed, "adam_state_count": len(optimizer.state),
                "adam_state_bytes": sum(v.numel() * v.element_size()
                                        for state in optimizer.state.values()
                                        for v in state.values() if torch.is_tensor(v)),
                "denoiser_forward_calls": len(calls), "sampled_diffusion_timesteps": calls,
                "seconds": elapsed, "learning_rate": lr}
    finally:
        hook.remove()
        optimizer.zero_grad(set_to_none=True)


def recovery_rows():
    from robotics_baseline_check import states_match
    prefix, branch, metadata = read(PREFIX), read(RECOVERY), read(SIDECAR)
    assert metadata["seed"] == 2 and metadata["branch_step"] == OBSERVATION_STEP
    assert metadata["success"] and metadata["identical_branch_state"]
    assert metadata["arm"] == "replan_once"
    rows = prefix["steps"][:OBSERVATION_STEP]
    assert states_match(rows[-1]["after"], branch["branch_state"])
    rows = rows + branch["steps"]
    validate_rows(rows)
    assert rows[-1]["after"]["coverage"] > .95 and len(rows) == metadata["total_actions"]
    return rows


def reconstruct_sample():
    from robotics_baseline_check import physical_state, states_match
    import gymnasium as gym
    rows = recovery_rows()
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    frames, compared = {}, 0
    try:
        observation, _ = env.reset(seed=2)
        assert states_match(physical_state(env, observation), rows[0]["before"])
        for row in rows[:OBSERVATION_STEP]:
            observation, _, terminated, truncated, _ = env.step(np.asarray(row["action"], dtype=np.float32))
            assert not terminated and not truncated, "A sample cannot cross a terminal/reset boundary."
            assert states_match(physical_state(env, observation), row["after"]), "Recorded RGB/physics replay mismatch."
            compared += 1
            if row["step"] in (OBSERVATION_STEP - 1, OBSERVATION_STEP):
                frames[row["step"]] = {"pixels": observation["pixels"].copy(),
                                       "agent_pos": observation["agent_pos"].copy()}
    finally:
        env.close()
    sample = make_training_sample(rows, frames, OBSERVATION_STEP)
    return sample, {"scene": 2, "successful_recovery_sample": 2001,
                    "observation_after_movements": [156, 157],
                    "executed_target_movements": list(range(157, 173)),
                    "historical_target_slot": 0, "first_future_target_slot": 1,
                    "cpu_replay_commands_checked": compared, "rgb_and_physics_exact": True,
                    "coverage_only_tolerance": 1e-12, "padding": False,
                    "inputs": list(sample), "sample_hashes": {k: tensor_hash(v) for k, v in sample.items()},
                    "success_is_selection_only_not_model_input": True}


def run(args):
    import draccus
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    import lerobot.policies.diffusion.modeling_diffusion as installed
    import lerobot.policies.diffusion.configuration_diffusion as installed_config
    import lerobot.policies.normalize as installed_normalize
    import robotics_baseline_check as baseline
    import robotics_feasibility as feasibility

    target = args.output_dir / "training-gate.json"
    if target.exists():
        raise FileExistsError("Completed gate exists; inspect it or choose a disposable *-repeat output.")
    assert torch.cuda.is_available(), "CUDA is required; no silent CPU fallback."
    # Windows venv python.exe redirects through a parent Python launcher. It is
    # this invocation, not a second collector; unrelated Python jobs still block.
    own_processes = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    assert not any(p.info["pid"] not in own_processes for p in psutil.process_iter(["pid", "name"])
                   if (p.info["name"] or "").lower() in ("python.exe", "pythonw.exe")), "Another Python process needs inspection before GPU use."
    torch.set_num_threads(4)
    torch.manual_seed(20261015)
    model_dir = args.model_dir.resolve()
    paths = [PREFIX, RECOVERY, SIDECAR, Path(__file__).resolve(),
             Path(baseline.__file__), Path(feasibility.__file__), Path(installed.__file__),
             Path(installed_config.__file__), Path(installed_normalize.__file__),
             model_dir / "config.json", model_dir / "model.safetensors"]
    bindings = {str(path): sha(path) for path in paths}
    assert bindings[str(model_dir / "model.safetensors")] == WEIGHTS_SHA256
    raw = read(model_dir / "config.json")
    assert (raw["horizon"], raw["n_obs_steps"], raw["num_train_timesteps"],
            raw["noise_scheduler_type"], raw["prediction_type"]) == (16, 2, 100, "DDPM", "epsilon")
    assert psutil.virtual_memory().available >= 3 * 2**30, "Less than 3GiB free system RAM."
    sample, sample_info = reconstruct_sample()
    config = draccus.decode(DiffusionConfig, {k: v for k, v in raw.items() if k != "type"})
    config.device = "cuda"
    free, total = torch.cuda.mem_get_info()
    budget = min(int(2800 * 2**20), free - 384 * 2**20)
    assert budget >= 1600 * 2**20, "Insufficient free GPU memory for the bounded gate."
    torch.cuda.set_per_process_memory_fraction(budget / total)
    result = {"purpose": "one reversible supervised update, not a trained controller",
              "passed": False, "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name)
                           for name in ("torch", "lerobot", "gym-pusht", "gymnasium", "numpy", "diffusers")},
              "source_bindings": bindings, "sample": sample_info,
              "gpu_before": gpu_status(), "cuda_allocator_budget_mib": budget / 2**20,
              "deployment": {"horizon": 16, "denoising_iterations": 100,
                             "controller": "unchanged milestone 8/12", "extra_models": 0},
              "no_policy_predictions_or_alternative_actions": True,
              "no_learned_checkpoint_saved": True}
    policy = DiffusionPolicy.from_pretrained(model_dir, config=config, strict=True).to("cuda").eval()
    initial = state_fingerprint(policy)
    frozen = state_fingerprint(policy, exclude_tail=True)
    snapshot = snapshot_training_state(policy)
    try:
        names = configure_tail_only(policy)
        result["trainable_names"] = names
        result["trainable_parameters"] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        result["total_parameters"] = sum(p.numel() for p in policy.parameters())
        batch = {k: v.to("cuda") for k, v in sample.items()}
        torch.cuda.reset_peak_memory_stats()
        try:
            result["update"] = train_one_update(policy, batch)
            result["frozen_parameters_and_buffers_unchanged"] = state_fingerprint(policy, exclude_tail=True) == frozen
            assert result["frozen_parameters_and_buffers_unchanged"]
            result["passed"] = True
        except torch.cuda.OutOfMemoryError as error:
            result["failure"] = {"type": "CUDA out of memory", "message": str(error)}
        finally:
            result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    finally:
        restore_training_state(policy, snapshot)
        result["entire_model_restored_exactly"] = state_fingerprint(policy) == initial
        assert result["entire_model_restored_exactly"]
        result["source_files_unchanged"] = all(sha(path) == digest for path, digest in bindings.items())
        assert result["source_files_unchanged"]
    result["gpu_after"] = gpu_status()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/raw/robotics/diffusion_pusht"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with MemorySampler() as memory:
        result = run(args)
    result.update(peak_process_rss_mib=memory.peak_rss / 2**20,
                  minimum_system_available_ram_mib=memory.minimum_available / 2**20)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training-gate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "trainable_parameters", "update",
                                           "peak_cuda_allocated_mib", "entire_model_restored_exactly")
                      if k in result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
