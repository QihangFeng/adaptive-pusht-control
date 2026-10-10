"""Bounded laptop check, not a controller experiment or success-rate benchmark.

Integration follows LeRobot v0.3.3/examples/2_evaluate_pretrained_policy.py.
The Windows config loader avoids reopening an upstream NamedTemporaryFile.
Only camera pixels and the pusher's own position reach the frozen policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gym_pusht  # noqa: F401 -- registers the environment
import gymnasium as gym
import numpy as np
import psutil
import torch

MODEL_ID = "lerobot/diffusion_pusht"
MODEL_REVISION = "84a7c23178445c6bbf7e1a884ff497017910f653"
WEIGHTS_SHA256 = "995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8"


def summarize(values):
    if not values:
        return None
    return {"count": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "min": min(values), "max": max(values)}


class MemorySampler:
    """Sample process RSS and system availability; GPU peaks use PyTorch counters."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_rss = 0
        self.minimum_available = psutil.virtual_memory().available
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
            self.stop_event.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()


def gpu_status():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def observation_batch(observation, device):
    image = torch.from_numpy(observation["pixels"]).to(dtype=torch.float32)
    image = image.permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    state = torch.from_numpy(observation["agent_pos"]).to(dtype=torch.float32)
    return {"observation.image": image, "observation.state": state.unsqueeze(0).to(device)}


def simulator_check():
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    initial, _ = env.reset(seed=42)
    repeated, _ = env.reset(seed=42)
    assert np.array_equal(initial["pixels"], repeated["pixels"])
    assert initial["pixels"].shape == (96, 96, 3)
    assert initial["pixels"].dtype == np.uint8
    assert initial["agent_pos"].shape == (2,)
    steps = []
    for _ in range(10):
        start = time.perf_counter()
        current, _, _, _, _ = env.step(np.array([256, 256], dtype=np.float32))
        steps.append(time.perf_counter() - start)
    changed = bool(np.any(initial["pixels"] != current["pixels"]))
    assert changed, "Camera image must change after physical movement."
    result = {"observation_shape": list(initial["pixels"].shape),
              "image_changes_after_actions": changed, "same_seed_same_initial_image": True,
              "step_seconds": summarize(steps), "simulated_hz": env.metadata["render_fps"]}
    env.close()
    return result


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(4)
    packages = ["torch", "torchvision", "lerobot", "gym-pusht", "gymnasium", "pymunk",
                "numpy", "diffusers", "huggingface-hub", "safetensors", "psutil",
                "opencv-python", "opencv-python-headless"]
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in packages},
              "installed_packages": {dist.metadata["Name"]: dist.version
                                     for dist in importlib.metadata.distributions()},
              "gpu_before": gpu_status(), "simulator": simulator_check()}
    if args.sim_only:
        return result
    assert torch.cuda.is_available(), "CUDA unavailable; do not silently switch to CPU."
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    import draccus

    model_dir = args.model_dir.resolve()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False,
                          allow_patterns=["config.json", "model.safetensors"], local_dir=model_dir)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    with (model_dir / "model.safetensors").open("rb") as weights:
        weights_hash = hashlib.file_digest(weights, "sha256").hexdigest()
    assert weights_hash == WEIGHTS_SHA256, "Weights differ from the pinned public checkpoint."
    config = draccus.decode(DiffusionConfig, {key: value for key, value in raw.items() if key != "type"})
    config.device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    policy = DiffusionPolicy.from_pretrained(model_dir, config=config, strict=True).to("cuda").eval()
    torch.cuda.synchronize()
    result["policy"] = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                        "strict_load": True, "config": raw, "load_seconds": time.perf_counter() - load_start,
                        "parameters": sum(p.numel() for p in policy.parameters()),
                        "weights_bytes": (model_dir / "model.safetensors").stat().st_size,
                        "weights_sha256": weights_hash,
                        "config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()}
    # Normalizers are learned checkpoint buffers. Infinity would mean missing statistics.
    for name, buffer in policy.named_buffers():
        assert torch.isfinite(buffer).all(), f"Non-finite checkpoint buffer: {name}"
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    observation, _ = env.reset(seed=42)
    warm_times = []
    with torch.inference_mode():
        for index in range(4):
            policy.reset()  # force a real neural prediction, not a cached action
            batch = observation_batch(observation, "cuda")
            torch.cuda.synchronize()
            tick = time.perf_counter()
            action = policy.select_action(batch)
            torch.cuda.synchronize()
            warm_times.append(time.perf_counter() - tick)
            assert action.shape == (1, 2) and torch.isfinite(action).all()
            print(f"Prediction check {index + 1}/4: {warm_times[-1]:.3f}s", flush=True)
    result["cold_prediction_seconds"] = warm_times[0]
    result["warm_prediction_seconds"] = summarize(warm_times[1:])
    episodes = []
    all_queries, all_cached = [], []
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        policy.reset()
        observation, _ = env.reset(seed=seed)
        queries, cached, simulation = [], [], []
        rewards = []
        episode_start = time.perf_counter()
        for step in range(300):
            batch = observation_batch(observation, "cuda")
            new_prediction = len(policy._queues["action"]) == 0
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(batch)
            torch.cuda.synchronize()
            (queries if new_prediction else cached).append(time.perf_counter() - tick)
            action_np = action.squeeze(0).cpu().numpy()
            assert np.isfinite(action_np).all()
            tick = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action_np)
            simulation.append(time.perf_counter() - tick)
            rewards.append(float(reward))
            if (step + 1) % 80 == 0:
                print(f"Seed {seed}: {step + 1} steps, {len(queries)} AI predictions", flush=True)
            if terminated or truncated:
                break
        elapsed = time.perf_counter() - episode_start
        episodes.append({"seed": seed, "steps": step + 1, "success": bool(terminated),
                         "truncated": bool(truncated), "max_reward": max(rewards),
                         "final_reward": rewards[-1], "wall_seconds": elapsed,
                         "simulated_seconds": (step + 1) / env.metadata["render_fps"],
                         "model_queries": len(queries), "query_seconds": summarize(queries),
                         "cached_action_seconds": summarize(cached), "environment_step_seconds": summarize(simulation)})
        all_queries.extend(queries)
        all_cached.extend(cached)
        print(f"Seed {seed} complete: success={terminated}, {step + 1} steps, {elapsed:.1f}s", flush=True)
    env.close()
    result.update(episodes=episodes, episode_query_seconds=summarize(all_queries),
                  cached_action_seconds=summarize(all_cached),
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  gpu_after=gpu_status(), total_run_seconds=time.perf_counter() - started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/raw/robotics/diffusion_pusht"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results/robotics-feasibility/summary.json"))
    args = parser.parse_args()
    with MemorySampler() as memory:
        result = run(args)
    result.update(peak_process_rss_mib=memory.peak_rss / 2**20,
                  minimum_system_available_ram_mib=memory.minimum_available / 2**20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
