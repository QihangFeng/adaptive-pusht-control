# Robotic Manipulation with Visual Feedback

ECE 730 course project: computation and AI for cyber-physical systems.

This branch explores feedback around a frozen robot-action model. In the PushT simulator, a camera observes a pusher and a T-shaped block. A pretrained Diffusion Policy predicts movements; the simulator applies them and returns updated images.

The candidate contribution is lightweight execution feedback that decides when to interrupt an existing movement plan and request a new prediction, potentially adapting the number of actions executed between predictions. Its usefulness has **not** been established. The completed investigation found recovery opportunities, but interruption also harmed successful continuations. The adaptive controller is not implemented.

## Current status

Laptop feasibility is verified on a GTX 1650 (4 GiB), 16 GB RAM and Python 3.11.9. Three complete simulator runs finished without crashes or GPU-memory errors; two reached the target. New predictions averaged 4.80 seconds, and full runs took 1.9–3.1 minutes.

This supports bounded **offline simulation**, not real-time deployment, large 3D robot models or policy training. See the [feasibility report](reports/2026-10-10-robotics-laptop-feasibility.md) for evidence, limitations, sources and setup instructions.

The [baseline diagnostic report](reports/2026-10-10-robotics-baseline-check.md) preserves the original six unchanged-policy episodes and 15 short probes. The [expanded failure/recovery report](reports/2026-10-11-robotics-failure-patterns.md) covers 30 baseline attempts (20 successes, 10 timeouts) and 24 matched-state continuations. Immediate replanning helped in some selected failed scenes but disrupted the successful control: overall success was 5/12 under each arm, with more AI calls for replanning. This is exploratory evidence, not a validated controller improvement. Clarifying the interruption trigger should precede implementation or further sweeps.

The [expanded replanning literature review](reports/2026-10-11-robotics-replanning-literature.md) compares 17 timing, monitoring and recovery papers, with code-access and laptop-fit caveats. SGAC and DCDP directly study Diffusion Policy on PushT; observation-driven interruption also has established precedent. The review recommends keeping the frozen-policy setup while discussing a narrow method, not claiming that adaptive replanning itself is new.

## Where to start

| Location | Purpose |
|---|---|
| [requirements-robotics.txt](requirements-robotics.txt) | Compatible simulator and inference dependencies |
| [experiments/robotics_feasibility.py](experiments/robotics_feasibility.py) | Verified baseline integration and resource check |
| [experiments/robotics_baseline_check.py](experiments/robotics_baseline_check.py) | Passive baseline recordings and local replanning probes |
| [experiments/robotics_failure_investigation.py](experiments/robotics_failure_investigation.py) | Resumable 30-attempt investigation and full remaining-episode recovery pilot |
| [experiments/robotics_trajectory_analysis.py](experiments/robotics_trajectory_analysis.py) | Frozen exploratory rules applied to saved traces |
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
