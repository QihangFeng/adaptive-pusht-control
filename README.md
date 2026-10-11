# Robotic Manipulation with Visual Feedback

ECE 730 course project: computation and AI for cyber-physical systems.

This branch explores feedback around a frozen robot-action model. In the PushT simulator, a camera observes a pusher and a T-shaped block. A pretrained Diffusion Policy predicts movements; the simulator applies them and returns updated images.

The candidate contribution is lightweight execution feedback that decides when to interrupt an existing movement plan and request a new prediction, potentially adapting the number of actions executed between predictions. Its usefulness has **not** been established. The baseline screen found task failures, but contact loss alone was not a reliable failure signal, and short local replanning probes did not improve alignment.

## Current status

Laptop feasibility is verified on a GTX 1650 (4 GiB), 16 GB RAM and Python 3.11.9. Three complete simulator runs finished without crashes or GPU-memory errors; two reached the target. New predictions averaged 4.80 seconds, and full runs took 1.9–3.1 minutes.

This supports bounded **offline simulation**, not real-time deployment, large 3D robot models or policy training. See the [feasibility report](reports/2026-10-10-robotics-laptop-feasibility.md) for evidence, limitations, sources and setup instructions.

The [baseline diagnostic report](reports/2026-10-10-robotics-baseline-check.md) covers six unchanged-policy episodes (three successes, three timeouts), contact/trajectory analysis, and 15 short same-state replanning probes. The next investigation should compare fixed execution lengths and clarify the trigger before implementing the proposed method.

## Where to start

| Location | Purpose |
|---|---|
| [requirements-robotics.txt](requirements-robotics.txt) | Compatible simulator and inference dependencies |
| [experiments/robotics_feasibility.py](experiments/robotics_feasibility.py) | Verified baseline integration and resource check |
| [experiments/robotics_baseline_check.py](experiments/robotics_baseline_check.py) | Passive baseline recordings and local replanning probes |
| [reports/](reports/) | Findings and reproduction instructions |
| [results/](results/README.md) | Small, shareable experiment evidence |
| `.venv-robotics/` | Local robotics environment; Git-ignored |
| `data/raw/robotics/diffusion_pusht/` | Pinned model download; Git-ignored |

Use `.venv-robotics`, not another local environment. For an already-installed setup, repeat the check without overwriting the original results:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_feasibility.py --output results\robotics-feasibility-repeat\summary.json
```

For first-time installation and downloading, follow the report. No training dataset or paid API is required for this baseline.

## Branch separation

- `topic/robotic-manipulation` contains the robotics work.
- [`main`](https://github.com/QihangFeng/delay-aware-ai-grid-control/tree/main) preserves the completed power-grid work, including its robustness study. Grid files are intentionally absent from this branch.

Keep these as separate topic snapshots; merging this cleanup into `main` would remove its grid files.

Switching branches restores tracked topic files. Ignored local environments, downloaded data and course materials are not switched by Git; they are retained separately from the published source tree.

The simulator and pretrained policy are reused work, not our contribution: [Gym-PushT](https://github.com/huggingface/gym-pusht), [Diffusion Policy](https://arxiv.org/abs/2303.04137), and the [public checkpoint](https://huggingface.co/lerobot/diffusion_pusht).
