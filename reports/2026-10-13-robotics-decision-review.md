# Do earlier extensions or new-plan transitions explain losses 54 and 59?

Date: October 13, 2026

## Conclusion first

**The saved trajectories do not identify a clearly harmful earlier extension or a common abrupt-plan-transition defect. Do not launch another eight-versus-twelve experiment merely because these two attempts failed.** Keep the milestone controller unchanged; do not restart the rejected threshold/weighted gate development.

There is a more important finding: **both severe declines later regained their pre-drop alignment, but neither attempt completed the final positioning above 95% coverage.** Calling these declines “unrecoverable” would confuse recovery of overlap with successful task completion. Final fine alignment is a better-supported subject for our next discussion, not a proven new controller or solution.

This completed task is CPU-only inspection of existing evidence. There were no new predictions, simulator runs, rendering, controller changes, alternative-action continuations or trained models. The original milestone and all prior frozen sources/results remain unchanged.

## What was inspected

The source is the [milestone evaluation](2026-10-12-robotics-fresh-evaluation.md), commit `0f40449`: scenes 50–69, all 60 recorded fixed-eight/fixed-twelve/adaptive attempts. The focused panel covers all **20 adaptive attempts**, **131 extensions** and **358 transitions to a fresh prediction**, rather than selecting examples only from the two failures.

We checked 176 source/evidence bindings, each compressed trace hash, sidecars against the complete inventory, movement indices, prediction counts and terminal coverage. Reused source-code bindings were checked without importing or executing those collectors. The earlier [regression diagnosis](2026-10-12-robotics-regression-diagnosis.md) already documented the destructive prefixes and the failure of two local timing changes; those trials were not repeated here.

The comparisons below are retrospective and outcome-defined. Multiple decisions from one scene are correlated; they are not independent experimental trials. Different methods usually reach different later states, so their saved outcomes cannot replace a matched-state intervention.

## 1. Were the extra four movements visibly harmful?

For every recorded `extend` decision, inspect the actual next four executed commands before any new prediction. In all 131 cases the full four-command suffix was executed. Compare true coverage after movement eight with coverage over those four commands. Simulator truth is used only for this offline diagnosis; the milestone's runtime decisions use camera estimates.

| Adaptive group | Scenes | Extended units | Suffix net drops greater than 1.5 pp | Suffix net drops greater than 3 pp |
|---|---:|---:|---:|---:|
| Lost original-baseline successes: 54/59 | 2 | 21 | 1 | 0 |
| Rescued original-baseline failures | 5 | 32 | 0 | 0 |
| Retained original-baseline successes | 13 | 78 | 3 | 1 |

“pp” means percentage points. The 1.5/3-point bins are descriptive checks, **not new controller thresholds, failure labels or a parameter search**. Net drop and maximum within-window drawdown are stored separately in [analysis.json](../results/robotics-decision-review/analysis.json).

The last extensions before the previously studied large declines were:

| Lost scene | Extend after movement | Cached movements executed | True coverage across the four movements | First substantial later decline |
|---|---:|---|---:|---:|
| 54 | 160 | 161–164 | Unchanged at 66.23%; block position and angle unchanged | 191 |
| 59 | 100 | 101–104 | Unchanged at 71.58%; block position and angle unchanged | 149 |

Both attempts improved alignment after those suffixes. Thus these are not visibly destructive four-movement extensions. The pusher still moves, and earlier reuse can indirectly change subsequent observations and stochastic plans; a stationary block does not prove that an extension has no causal effect. The saved data simply do not identify a specific extension worth blaming or overriding.

A useful counterexample is retained-success **scene 62, extension after movement 20**: coverage falls from **36.09% to 27.12%** during movements 21–24, a larger immediate drop than any extended suffix in the two lost cases. Nevertheless, that attempt succeeds at movement 194. A local drop during extension is therefore not a reliable terminal-failure signal.

## 2. Did a sudden change between plans explain the destructive pushing?

We compared the final **executed command target** of an old plan with the first target of a new plan. Its displacement is a command-target difference—not physical speed, and not disagreement with an unlogged discarded old-plan suffix. A negative direction cosine means a reversal relative to the preceding executed-command difference. These are retrospective diagnostics, not safety criteria.

| Lost scene | Old plan ended | New plan begins | Command-target displacement | Direction cosine | Observed prefix before interruption |
|---|---|---:|---:|---:|---|
| 54 | Ordinary trim after 188 | 189 | 1.16 coordinate units | +0.700 | Movements 189–193; coverage 79.68% → 60.22% |
| 59 | Ordinary trim after 144 | 145 | 31.88 coordinate units | +0.843 | Movements 145–151; coverage 92.49% → 51.56% |

Coordinates are the simulator/policy's 512-coordinate action space, **not camera pixels**. Both plans start after ordinary eight-action trimming. Neither boundary reverses the preceding command direction. Scene 54 has a small boundary change; scene 59 has a larger one. This does not support a common large-jump/reversal explanation for both failures.

Successful runs also have similar or larger boundary changes:

- Retained-success **62, new plan at 114**: before coverage 93.09%, command displacement 27.58 coordinate units, and physical speed about 167.90 units/s. Scene 59's corresponding speed is 168.11 units/s. Yet scene 62's first-eight coverage drop is only 2.70 points and it eventually succeeds.
- Retained-success **52, new plan at 189**, and **62, new plan at 142** have command displacements **33.22** and **49.02** units—larger than scene 59's—but only about **0.15** and **0.14** points of first-prefix drop.
- Rescued-success **67, new plan at 141** drops from **92.43% to 69.92%** before interruption after 146, but subsequently completes the task. A severe decline inside a fresh prefix is also not a terminal-failure label.

Across adaptive attempts, command-direction reversals occur at **13/64** boundaries in the lost cases, **21/107** in rescued successes and **20/187** in retained successes. Outcome-dependent counts and unequal trajectory lengths do not establish a classifier or causal effect. We found no evidence strong enough to propose a boundary-jump/reversal filter or plan blending as a fix.

## 3. The declines recover, but final positioning does not finish

The following steps refer to the original saved adaptive attempts, not new repeats. Recovery here means regaining the true overlap immediately before the studied decline; it does not mean the same physical pose or guaranteed success.

| Scene | Before the large decline | Lowest later coverage | First regained pre-drop coverage | Best coverage in entire attempt | Final coverage |
|---|---|---|---|---|---:|
| 54 | 79.68% after 190 | 43.54% after 198 | 79.83% after 242 | 94.58% after 280 | 94.49% |
| 59 | 92.49% after 148 | 21.18% after 155 | 92.82% after 197 | 94.53% after 135 | 91.87% |

Scene 59's best coverage occurred **before** the large decline; the table deliberately retains that distinction. Both attempts time out at 300 movements because neither ever exceeds the simulator's strict **95%** success threshold. The important distinction is:

> Regaining a previous alignment level is not the same as completing the final alignment.

This also explains why diagnosing only the biggest decline can miss another problem: a run can recover that decline and still struggle with the last few percentage points.

### Camera estimates near completion

The saved camera estimates sometimes exceed 95% while true coverage does not:

| Scene and movement | Camera-estimated coverage | True coverage at the same movement |
|---|---:|---:|
| 54 after 284 | 96.90% | 94.49% |
| 59 after 135 | 95.43% | 94.53% |

These are each scene's maximum logged camera estimate, not maximum true coverage or calibrated confidence. The milestone does **not** use that camera estimate as the simulator's success checker, and the frozen action model does not receive this derived coverage scalar as its input. Therefore this observation is not proof that camera overestimation caused either timeout. It is relevant when considering a future camera-based final-alignment monitor, whose apparent “finished” state could be inaccurate.

## Decision and next discussion

1. **Keep milestone `0f40449` as the main method.** Stop development/retesting of the rejected weighted/threshold filters; preserve their negative evidence for reference.
2. **Do not launch the proposed earlier-extension counterfactual now.** This inspection did not find a specific visibly harmful extension or common boundary defect to justify selecting one. Indirect effects remain possible, but “the run eventually failed” is not a sufficient selection argument.
3. **Discuss final-alignment difficulty as the next success-focused investigation.** Inspect late positioning in failed and successful attempts: camera error, position/orientation error, repeated nearly identical motion and how much movement budget remains. Any proposed signal must be available from images/current commands, not simulator truth or future outcomes. Reuse the existing evidence first; this task does not authorize a new controller, classifier or broad GPU search.

The objective remains more completed pushes, with computation as a secondary trade-off. Keep the existing 95% success criterion and 300-movement budget; do not redefine the task to turn a timeout into a success. The frozen policy may itself limit final precision; changing query timing alone might not resolve that limitation. Any revised method would need multiple prediction samples and fresh evaluation scenes, not a single successful rerun of 54/59. All scenes inspected here are already-used development information for such a revision.

## Reproduction and verification

Run from the project directory with the existing Python environment:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_decision_review.py
```

The collector verifies an existing matching analysis rather than overwriting it. For a disposable repeat, append `--output-dir results\robotics-decision-review-repeat`; output cannot overlap earlier evidence directories. This is standard-library CPU analysis: no PyTorch inference, simulator, RGB fitting or paid API is required.

Inputs, definitions, every derived extension/transition and selected loss windows are in [analysis.json](../results/robotics-decision-review/analysis.json). All **273 robotics tests** passed, including 26 new synthetic CPU checks, both in the working tree and the staged checkout. The staged collector reproduced the existing analysis exactly without rewriting it.

The [independent audit](../results/robotics-decision-review/independent-audit.json) passed: 176 SHA bindings, all 60 raw sidecars/traces, 10,691 movements, 131 extensions and 1,095 noninitial transitions across the three methods (358 adaptive). It independently recomputed window arithmetic, method/group summaries, medians, drop bins and the selected extension/handover/recovery/camera-peak examples without importing this review module. Limits include missing full discarded plans, no matched-state alternative execution and no reconstruction of latent observation/RNG histories. Camera features are read from earlier validated recordings rather than re-estimated here; no RGB fitting or physical replay was repeated.

## Sources

- [Milestone fresh-scene report](2026-10-12-robotics-fresh-evaluation.md), [all original attempts](../results/robotics-fresh-evaluation/episodes.json), [original protocol](../results/robotics-fresh-evaluation/protocol.json), and [original independent audit](../results/robotics-fresh-evaluation/independent-audit.json): authoritative recorded commands, physics diagnostics, camera features and outcomes.
- [Adaptive action-reuse report](2026-10-12-robotics-adaptive-length.md): unchanged extension/interruption rules and their assumptions.
- [Completed regression diagnosis](2026-10-12-robotics-regression-diagnosis.md): earlier destructive-plan windows, prediction-sample sensitivity and unsuccessful timing changes.
- [Completed threshold/weighted comparison](2026-10-13-robotics-weighted-gate.md): rejected extensions, retained as evidence rather than adopted methods.

The contribution of this task is a reproducible diagnostic review and a narrower problem interpretation—not a new robot policy, proven causal failure mechanism or controller improvement.
