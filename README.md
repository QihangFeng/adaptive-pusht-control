# Robotic Manipulation with Visual Feedback

ECE 730 course project: computation and AI for cyber-physical systems.

This branch explores feedback around a frozen robot-action model. In the PushT simulator, a camera observes a pusher and a T-shaped block. A pretrained Diffusion Policy predicts movements; the simulator applies them and returns updated images.

The candidate contribution is lightweight execution feedback that decides when to reuse more predicted movements or request a new prediction. The completed benefit-aware pilot tests one early interruption per attempt. A separate adaptive-length screen tests camera/path-guided eight-versus-twelve-action execution and bounded interruptions. Both are preliminary investigations, not proven general improvements.

## Current status

Laptop feasibility is verified on a GTX 1650 (4 GiB), 16 GB RAM and Python 3.11.9. Three complete simulator runs finished without crashes or GPU-memory errors; two reached the target. New predictions averaged 4.80 seconds, and full runs took 1.9–3.1 minutes.

This supports bounded **offline simulation**, not real-time deployment, large 3D robot models or policy training. See the [feasibility report](reports/2026-10-10-robotics-laptop-feasibility.md) for evidence, limitations, sources and setup instructions.

The [baseline diagnostic report](reports/2026-10-10-robotics-baseline-check.md) preserves the original six unchanged-policy episodes and 15 short probes. The [expanded failure/recovery report](reports/2026-10-11-robotics-failure-patterns.md) covers 30 baseline attempts (20 successes, 10 timeouts) and 24 matched-state continuations. Immediate replanning helped in some selected failed scenes but disrupted the successful control: overall success was 5/12 under each arm, with more AI calls for replanning. This is exploratory evidence, not a validated controller improvement.

The [completed benefit-aware pilot](reports/2026-10-11-robotics-benefit-pilot.md) contains camera validation, 102 matched continuations from 17 checkpoints, and 80 full held-out attempts. The learned gate finished 16/20 cases versus 15/20 for fixed-eight and the warning rule, and 5/20 for fixed-four. It saved only five AI calls overall and ran slightly slower once monitoring was included. All eight interruptions addressed stagnation. This is preliminary evidence, not a strong or robust improvement; a simpler same-timing comparator remains necessary. All 91 tests and independent result audits passed.

The [adaptive action-reuse screen](reports/2026-10-12-robotics-adaptive-length.md) completed 24 attempts on eight reused development scenes. Adaptive execution achieved 6/8 successes versus fixed-eight 3/8, with 37.55% fewer predictions and 31.65% shorter runtime including monitoring. However, mean final alignment was slightly lower: 92.14% versus 93.43%, with one baseline success lost. Fixed-twelve was fastest and achieved 5/8 successes, but only 82.05% mean alignment. This supports further investigation of the quality/computation trade-off, not a reliable general improvement. All 124 robotics tests and the independent artifact audit passed; the earlier frozen pilot remains unchanged.

The [fresh-scene evaluation](reports/2026-10-12-robotics-fresh-evaluation.md) kept that controller unchanged and completed 60 attempts on new seeds 50–69. Adaptive execution finished 18/20 pushes versus fixed-eight 15/20, with 95.60% versus 92.88% mean alignment, 22.22% fewer predictions and 16.35% shorter runtime including monitoring. It rescued five baseline failures but lost two successes; the success-rate gain remains uncertain. Fixed-twelve was cheaper (16/20 successes, 88.61% mean alignment), so adaptive execution is a quality/computation trade-off, not dominance over every comparator. All 152 tests and the independent audit passed. The evaluation took 97.21 minutes on the laptop; interruption-specific benefit and repeated-randomness reliability remain untested.

The [expanded replanning literature review](reports/2026-10-11-robotics-replanning-literature.md) compares 17 timing, monitoring and recovery papers, with code-access and laptop-fit caveats. SGAC and DCDP directly study Diffusion Policy on PushT; observation-driven interruption also has established precedent. The review recommends keeping the frozen-policy setup while discussing a narrow method, not claiming that adaptive replanning itself is new.

The designated **milestone** is commit [`0f40449`](https://github.com/QihangFeng/delay-aware-ai-grid-control/commit/0f4044966646df4cc38b0373fa52c0b733506e15), the completed fresh-scene evaluation. Subsequent work continues through new commits on this branch; this recovery reference is preserved, not an instruction to reset or discard later work.

The [regression diagnosis](reports/2026-10-12-robotics-regression-diagnosis.md) completed 26 bounded trials on lost baseline-success scenes 54/59. New prediction samples changed outcomes: unchanged adaptive execution succeeded in 3/4 repeats versus fixed-eight 1/4. Neither one-movement-earlier replanning nor deferring interruption rescued a tested local failure; both changes lost the same successful scene-59 continuation. Keep the milestone controller unchanged. These selected-case results explain sensitivity, not general reliability or a validated fix. Both restoration controls, all 183 tests and the independent audit passed.

The [alignment-decline screen](reports/2026-10-12-robotics-decline-screen.md) compared every first interruption in the saved adaptive attempts: six situations, three paired prediction samples each, reusing eight completed comparisons. Continuing and replanning both succeeded 7/18 times. Smaller two-warning declines favored continuing (7/9 versus 2/9); larger three-warning declines favored replanning (5/9 versus 0/9). These correlated signals support discussing a simple ongoing-decline check, not a validated classifier or controller fix. All six restoration controls, 213 tests and the independent audit passed; no controller or model was changed.

The [weighted-versus-simple gate evaluation](reports/2026-10-13-robotics-weighted-gate.md) completed a bounded development-only search and 60 full attempts on new seeds 70–89. Both gates succeeded 16/20 times versus the unchanged adaptive controller's 15/20, rescuing 72/81 but losing 73. The simple gate had 94.58% mean final alignment versus weighted 92.13% and unchanged 92.02%; both gates used slightly more predictions and runtime than unchanged. Weighting did not demonstrate an advantage, and the success gain remains uncertain. Keep the milestone controller unchanged. All 247 tests and the independent audit passed; settings were not retuned using these outcomes.

The [saved decision review](reports/2026-10-13-robotics-decision-review.md) inspected all 131 milestone extensions and 358 fresh-plan transitions without new GPU runs. It did not identify a clearly harmful earlier extension or a common abrupt-transition defect explaining losses 54/59. Both severe declines later regained their pre-drop overlap, but neither attempt ever exceeded 95% true coverage. Final positioning is therefore a better-supported subject for further discussion than labeling these dips unrecoverable. The milestone stays unchanged; development/retesting of the rejected threshold and weighted filters has stopped.

The [completed baseline comparison](reports/2026-10-13-robotics-cohort-completion.md) added the missing fixed-eight/fixed-twelve attempts on already-inspected scenes 70–89, without rerunning adaptive execution or changing any rule. The original improvement did not repeat: adaptive finished 15/20 versus fixed-eight 18/20. Pooled across 40 scenes, both finished 33/40; adaptive averaged 93.81% alignment versus 94.05% and 7.38% fewer predictions, but the prediction-saving interval includes zero. Runtime is descriptive only because collection sessions/logging differ. All 294 tests and the independent audit passed. Preserve the milestone, but do not claim a demonstrated general upgrade; no new controller or rejected filter is adopted.

## Where to start

| Location | Purpose |
|---|---|
| [requirements-robotics.txt](requirements-robotics.txt) | Compatible simulator and inference dependencies |
| [experiments/robotics_feasibility.py](experiments/robotics_feasibility.py) | Verified baseline integration and resource check |
| [experiments/robotics_baseline_check.py](experiments/robotics_baseline_check.py) | Passive baseline recordings and local replanning probes |
| [experiments/robotics_failure_investigation.py](experiments/robotics_failure_investigation.py) | Resumable 30-attempt investigation and full remaining-episode recovery pilot |
| [experiments/robotics_trajectory_analysis.py](experiments/robotics_trajectory_analysis.py) | Frozen exploratory rules applied to saved traces |
| [experiments/robotics_visual_monitor.py](experiments/robotics_visual_monitor.py) | Camera-only alignment estimation and offline accuracy checks |
| [experiments/robotics_benefit_model.py](experiments/robotics_benefit_model.py) | Small outcome-trained decision tree and validation-only threshold selection |
| [experiments/robotics_benefit_pilot.py](experiments/robotics_benefit_pilot.py) | Resumable, one-decision pilot with separate held-out evaluation |
| [experiments/robotics_adaptive_length.py](experiments/robotics_adaptive_length.py) | Frozen development screen of cached action reuse, camera/path gating and bounded interruptions |
| [experiments/robotics_fresh_evaluation.py](experiments/robotics_fresh_evaluation.py) | Frozen three-method evaluation on 20 new scenes, with paired uncertainty and strict resumability |
| [experiments/robotics_regression_diagnosis.py](experiments/robotics_regression_diagnosis.py) | Selected-case sampling and matched-state diagnosis of lost baseline successes, with restoration controls |
| [experiments/robotics_decline_screen.py](experiments/robotics_decline_screen.py) | Frozen matched-state screen of when camera-observed declines benefit from replanning, with reuse of completed trials |
| [experiments/robotics_weighted_gate.py](experiments/robotics_weighted_gate.py) | Explicit weighted/trend and simple-threshold interruption gates, with a bounded development search |
| [experiments/robotics_gate_evaluation.py](experiments/robotics_gate_evaluation.py) | Frozen three-method gate comparison, whole-trajectory preflight, complete plan logging and paired statistics |
| [experiments/robotics_decision_review.py](experiments/robotics_decision_review.py) | CPU-only retrospective extension/transition and final-alignment review; no new predictions or controller changes |
| [experiments/robotics_cohort_completion.py](experiments/robotics_cohort_completion.py) | Frozen completion of two missing baseline arms on inspected scenes, with adaptive evidence reuse and paired 40-scene synthesis |
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
