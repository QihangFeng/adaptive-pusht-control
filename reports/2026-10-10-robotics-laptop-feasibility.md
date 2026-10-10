# PushT robotics: laptop feasibility check

Date: October 10, 2026

## Conclusion

**The laptop can run the PushT simulator and the public pretrained Diffusion Policy, and can support a bounded, offline course-project evaluation.** All three complete rollouts finished without a software crash or GPU out-of-memory error. Two reached the target; one exhausted the 300-step limit.

This passes the hardware/software feasibility stage. It does **not** establish that our proposed contact-aware feedback rule is useful, that it improves on tuned baselines, or that this model is suitable for real-time deployment. No new controller was implemented.

## What was actually tested

The setup uses the GTX 1650 with 4 GiB graphics memory and its existing NVIDIA driver 552.22. The machine reports 15.92 GiB system RAM; its CPU is the user's reported i5-9300H. Python is the existing permanent installation, version 3.11.9.

PushT is a small **two-dimensional manipulation task**, not a full 3D robot arm. A simulated pusher moves a T-shaped block toward a target. Its physics engine updates the scene after actions, and the environment returns a new camera image. The policy receives that image plus the pusher's own position—not the block's privileged simulator pose or contact count. [Simulator](https://github.com/huggingface/gym-pusht), [official integration example](https://github.com/huggingface/lerobot/blob/v0.3.3/examples/2_evaluate_pretrained_policy.py).

The simulator check verified 96 × 96 RGB observations, identical initial images for repeated seed 42, and changed images after physical movement. Its ten smoke-test steps averaged 2.53 ms each. The model then ran four standalone prediction checks and three complete episodes with seeds 0, 1 and 2.

The frozen model retained its downloaded configuration: batch size one, FP32 without mixed precision, DDPM with 100 denoising steps, two observation frames, a 16-step prediction horizon, and eight executed actions per new prediction. We did not reduce denoising steps or substitute a smaller model to make the test pass. These settings come from the [checkpoint configuration](https://huggingface.co/lerobot/diffusion_pusht/blob/84a7c23178445c6bbf7e1a884ff497017910f653/config.json); the handling of a null inference-step setting is in [LeRobot's implementation](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py).

## Measured results

The authoritative results are in [summary.json](../results/robotics-feasibility/summary.json).

| Check | Measured result | Interpretation |
|---|---|---|
| Peak Python-process RAM | 2.57 GiB | Fits within the laptop's RAM; at least 3.37 GiB system RAM remained available during the sampled run |
| Peak PyTorch GPU allocation / reservation | 1.07 / 1.15 GiB | The tested model and inference fit on the GTX 1650 |
| Total GPU memory, before / final snapshot | 1008 / 2240 MiB | Includes desktop/background use and GPU runtime overhead; these are snapshots, not a measured total-memory peak |
| New AI prediction | Mean 4.80 s, median 4.77 s, p95 5.01 s | Based on 87 actual episode predictions, not cached actions |
| Cached next-action retrieval | Mean 1.00 ms | Retrieving an already-planned movement is not a new AI prediction |
| Complete test duration | 7.76 min | Includes simulator checks, model loading, four initial predictions and the three episodes; excludes installation/downloads and initial module imports |

Memory sampling covered the Python process every 100 ms. GPU allocation and reservation are PyTorch high-water counters, not all graphics-memory consumption. Prediction timing synchronizes CUDA and excludes observation preparation and output transfer; episode wall time includes them. These are measurements under the current desktop workload, not guaranteed timings on every run.

| Environment / policy seed | Steps | New AI predictions | Wall time | Task outcome |
|---|---:|---:|---:|---|
| 0 | 187 | 24 | 1.90 min | Target reached |
| 1 | 194 | 25 | 2.01 min | Target reached |
| 2 | 300 | 38 | 3.11 min | Step limit reached; task not completed |

The environment requires more than 95% target coverage for success. Seed 2 reached approximately 94.88% at its best point and ended at approximately 87.57%. Its failure was a **task failure**, not a hardware failure. We have not diagnosed its cause, and cannot label it a missed-contact failure from these summary metrics. Three seeds are too few to estimate a reliable success rate.

## Is the necessary evaluation practical?

Yes, provided we keep it bounded and run episodes sequentially. A full 300-step baseline episode took about three minutes. At that cost, 30 full episodes take approximately 1.5 hours per method, and four methods with similar query counts take approximately six hours. These are planning estimates, not completed comparisons.

More frequent replanning costs more: at the measured prediction speed, predicting every four actions would require about six minutes for a full episode; predicting every action would require about 24 minutes. Early success can shorten episodes, while thermal throttling and other GPU activity can lengthen them. Avoid a large parameter sweep before identifying the useful comparisons.

**Real-time control is not verified.** PushT advances simulated time only when stepped. Eight actions correspond to 0.8 simulated seconds, while a new prediction takes about 4.8 wall-clock seconds on this laptop. The offline simulator waits for inference; a moving physical robot would not automatically wait. We must not claim that these timings demonstrate a deployable real-time controller.

No policy training, large vision-language model, 3D manipulation benchmark, cloud service or paid API was tested or required. The future monitor's cost and usefulness remain untested.

## Working setup and reproduction

The robotics environment is separate from the original grid environment:

- `.venv-robotics/`: Python 3.11.9; GPU PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, LeRobot 0.3.3, Gym-PushT 0.1.5 and Pymunk 6.11.1.
- [requirements-robotics.txt](../requirements-robotics.txt): key compatibility pins. The full installed package inventory is recorded in the result JSON; this requirements file is not a complete transitive lockfile.
- `data/raw/robotics/diffusion_pusht/`: public model files, Git-ignored. No training dataset was downloaded.
- [robotics_feasibility.py](../experiments/robotics_feasibility.py): simulator assertions, strict checkpoint loading, timing, memory sampling and rollout evaluation.

The checkpoint is pinned to `84a7c23178445c6bbf7e1a884ff497017910f653`. Its weights are 1,050,862,408 bytes, with SHA-256 `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`; the script verifies that hash. The policy contains 262,709,044 parameters.

Two compatibility details matter:

1. LeRobot 0.3.3 retains the checkpoint's internal input/action normalization. Its newer processor-based release must not be substituted without a proper migration. The script supplies a directly decoded configuration to avoid the older loader's temporary-file reopening pattern on Windows. [Versioned policy source](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py), [configuration loader](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/configs/policies.py).
2. The first model attempt failed because an unbounded dependency selected PyArrow 26, which required NumPy 2 at runtime. PyArrow 20 and pandas 2.2.3 are now pinned with NumPy 1.26.4. After this correction, dependency checks and all three rollouts passed. The separate `simulator-check.json` records the earlier simulator-only smoke test; use `summary.json` for the final working environment.

For a **fresh** environment, from the project root:

```powershell
& 'D:\DevTools\Python\Python311\python.exe' -m venv .venv-robotics
.\.venv-robotics\Scripts\python.exe -m pip install -r requirements-robotics.txt
.\.venv-robotics\Scripts\python.exe -m pip check
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HOME = Join-Path (Get-Location) 'data\raw\huggingface'
.\.venv-robotics\Scripts\python.exe experiments\robotics_feasibility.py --download
```

For the already-installed environment, skip setup and downloads. Save a repeat separately:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_feasibility.py --output results\robotics-feasibility-repeat\summary.json
```

No system Python or GPU driver was changed. The existing `.venv`, grid simulator and pre-existing grid changes were preserved. No commit or push was made during the feasibility check itself.

## Recommended next step

Move to the unchanged-baseline failure check: record and inspect trajectories to determine whether missed contact or ineffective pushing actually occurs, and whether a new prediction can recover. Then decide which feedback rule and comparisons are justified. Do not infer a meaningful controller improvement from this feasibility test alone.

The policy and simulator are reused infrastructure. Our proposed contribution remains a separate execution-feedback rule and its evaluation—not building this simulator or inventing Diffusion Policy. For the underlying method, see [Diffusion Policy, Chi et al.](https://arxiv.org/abs/2303.04137). For the compatible GPU package pair, see [PyTorch's official version matrix](https://pytorch.org/get-started/previous-versions/).
