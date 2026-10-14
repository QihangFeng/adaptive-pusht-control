# Completing baseline comparisons for the unchanged milestone

Date: October 13, 2026

## Conclusion and question

**The milestone's improvement over fixed-eight did not repeat in the additional 20-scene comparison.** On scenes 70–89, unchanged adaptive execution succeeded in 15/20 attempts versus fixed-eight's 18/20. Across all 40 scenes, both succeeded in 33/40 attempts. Adaptive execution averaged 7.38% fewer predictions, but that estimated saving remains uncertain; alignment and observed runtime did not improve. The original positive milestone result remains valid for its original scenes, not an established general advantage.

All 40 missing baseline attempts are complete. All 294 robotics CPU tests and the independent artifact audit passed. No controller rule, checkpoint, target criterion or movement budget was changed. Milestone `0f40449` remains preserved; no revised method is adopted.

Does the milestone's quality/computation trade-off appear again in a second complete comparison, rather than only in the original 20-scene evaluation?

This is **completion of an already-inspected cohort**, not a new blind holdout, new method, rejected-rule retest or comparison across different initial scenes.

## Fixed scope

| Evidence | Scenes | Methods | Attempts newly collected |
|---|---|---|---:|
| Original milestone | 50–69 | Fixed-eight, fixed-twelve, adaptive | 0; reuse all 60 |
| Existing adaptive records | 70–89 | Unchanged adaptive eight/twelve | 0; reuse all 20 |
| New baseline collection | 70–89 | Fixed-eight, fixed-twelve | 40 |

The second cohort contains 60 method attempts on 20 different initial scenes. The pooled comparison contains 120 method attempts on **40 different scenes**. No previously completed attempt was rerun. Each method has one stochastic prediction sample per scene, so this does not establish reliability across repeated samples or other manipulation tasks.

## Preserved methods and conditions

- Use the original frozen `robotics_adaptive_length.evaluate_episode` for the two new baselines. Fixed-eight executes eight cached commands before predicting again; fixed-twelve executes twelve. Neither has the added camera monitor or interruption rule.
- Reuse only the `adaptive12` arm of the completed [gate comparison](2026-10-13-robotics-weighted-gate.md). Its controller is unchanged from milestone `0f40449`; no threshold/weighted predictions or development searches are run.
- Scene seed and prediction seed both equal the scene index. Reset prediction/observation queues through the original horizon configuration helper. Noise is paired by prediction-call order, not physical movement time.
- Keep the pinned public Diffusion Policy checkpoint and existing package versions, CUDA inference, four CPU threads, 300 total movements and strict true-coverage **greater than 95%** success criterion.
- Alternate the two new baseline orders by scene index. Keep every success and timeout; do not select scenes or rerun unfavorable predictions.
- Run one GPU collector. Freeze the new runner, reused evidence hashes, packages and checkpoint before collection; do not edit frozen sources while it runs.

## Integration and resume checks

CPU preparation verifies all 20 reused adaptive sidecars/traces and their initial camera history, actions, queues, guards and decisions. It binds the original 60 milestone attempts by reference, not by copying or editing their files. Existing passed horizon/prefix smoke evidence is reused rather than launching extra GPU checks.

Every newly scored baseline must start at the same physical state/image and execute the **same first eight commands and physical/image states** as the saved adaptive record for that scene. Only simulator truth coverage uses the existing `1e-12` rounding tolerance; commands, coordinates and RGB hashes must remain exact. The saved adaptive first unit contains twelve movements on every scene in this cohort, so there is no pre-eight intervention to invalidate this comparison.

New baseline records must reconstruct their exact eight/twelve command-queue boundaries, ages, pending counts, query counts and terminal unit lengths. Intervention events and monitoring time must be zero. A successful attempt first exceeds 95% at termination; every timeout has 300 movements. An existing completed record is verified, not rerun; an orphan trace or changed hash stops collection for inspection.

The first new scored attempts provide the integration checks. They remain part of the 40 attempts; there are no hidden unscored trials.

## Metrics and statistical plan

Primary outcomes are **successful pushes and average final alignment**. Prediction counts and runtime remain secondary trade-offs. Average over all attempts, including timeouts, with equal weight per initial scene.

Report the 70–89 cohort separately, followed by a pooled 50–89 synthesis. Compare adaptive with both fixed intervals and report gained/lost successes for each paired scene. For success, final coverage and query count, use 10,000 whole-scene-pair bootstrap resamples with seed `20261013`, plus the exact two-sided discordant-success test. Intervals are exploratory and have no multiplicity correction; identical observed outcomes do not prove equivalence.

**Runtime is descriptive only.** Adaptive runs were collected in an earlier session with additional trace logging and decision bookkeeping; baseline runs use the original collector now. Hardware load, temperature and logging differences are confounded with method. Report raw mean runtime but do not attach an inferential runtime-benefit claim or bootstrap interval to this mixed-session comparison. AI-call counts provide a cleaner computational measurement.

Already-inspected scenes and the pooled earlier cohort are not a newly untouched validation set. The unchanged milestone was frozen before these outcomes, so the comparison is informative, but it is not evidence about new tasks, real cameras, disturbances or deployment. A stronger prospective check, if later justified, would require genuinely new scenes and all methods collected together.

## Reproduction

Use `.venv-robotics`, the existing pinned checkpoint and [feasibility setup](2026-10-10-robotics-laptop-feasibility.md). Disposable repeats must use the separate Git-ignored output directory in every stage:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_cohort_completion.py --stage prepare --output-dir results\robotics-cohort-completion-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_cohort_completion.py --stage run --output-dir results\robotics-cohort-completion-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_cohort_completion.py --stage summarize --output-dir results\robotics-cohort-completion-repeat
```

Authoritative new evidence belongs to `results/robotics-cohort-completion/`. Existing adaptive and original-cohort traces remain in their source directories, referenced by frozen hashes. A completed summary cannot be silently rerun or overwritten. No training, paid API, new dataset or new hardware is needed.

## Results and decision

### Original milestone: scenes 50–69

These unchanged results are included to show the contrast, not to replace or rewrite the earlier report.

| Method | Successful pushes | Average final alignment | Average AI predictions | Average runtime |
|---|---:|---:|---:|---:|
| Fixed-eight | 15/20 | 92.88% | 24.30 | 119.51 s |
| Fixed-twelve | 16/20 | 88.61% | 14.55 | 71.57 s |
| Unchanged adaptive eight/twelve | 18/20 | 95.60% | 18.90 | 99.97 s |

Adaptive execution improved all four observed means against fixed-eight in this original cohort, while losing two baseline successes. That was useful preliminary evidence, but the next cohort tests whether the pattern persists.

### Completed comparison: scenes 70–89

| Method | Successful pushes | Average final alignment | Average AI predictions | Average runtime* |
|---|---:|---:|---:|---:|
| Fixed-eight | 18/20 | 95.22% | 21.10 | 103.25 s |
| Fixed-twelve | 13/20 | 88.37% | 18.55 | 90.98 s |
| Unchanged adaptive eight/twelve | 15/20 | 92.02% | 23.15 | 125.91 s |

Against fixed-eight, adaptive execution rescued scene **89** but lost successes **78, 80, 81 and 82**. Scene **72** failed under both methods. Adaptive execution had three fewer successes, 3.20 percentage points less mean alignment and 9.72% more predictions. These are observed differences, not proof that it is generally inferior: the paired uncertainty intervals include no difference.

Against fixed-twelve, adaptive execution rescued **70, 77, 86 and 87**, but lost **72 and 80**. Its higher observed success/alignment came with 4.60 additional predictions per attempt on average.

### Pooled comparison: all 40 scenes, 50–89

| Method | Successful pushes | Average final alignment | Average AI predictions | Average runtime* |
|---|---:|---:|---:|---:|
| Fixed-eight | 33/40 | 94.05% | 22.70 | 111.38 s |
| Fixed-twelve | 29/40 | 88.49% | 16.55 | 81.27 s |
| Unchanged adaptive eight/twelve | 33/40 | 93.81% | 21.03 | 112.94 s |

*Runtime is a mixed-session descriptive measurement, not a clean causal comparison. Reused adaptive traces had extra bookkeeping; baseline attempts were collected in a later session. Times include unsuccessful attempts and adaptive monitoring. Average adaptive monitoring was 11.59 s in the additional cohort and 9.60 s in the pooled comparison. No claim of statistically faster or slower execution is made from these times.*

Relative to fixed-eight, adaptive execution rescued **six failures** (53, 55, 57, 58, 67, 89) and lost **six successes** (54, 59, 78, 80, 81, 82). Equal success totals do not mean identical behavior or establish equivalence. Adaptive used **841 predictions versus 908**, a 67-call saving overall (7.38%), with slightly lower average alignment and no observed mean-runtime saving. The prediction-saving estimate is not yet reliably separated from zero.

Relative to fixed-twelve, adaptive execution gained seven successes and lost three: gains **55, 63, 68, 70, 77, 86, 87**; losses **59, 72, 80**. Its observed quality is higher but it uses **27.04% more predictions**. This is a possible quality/computation trade-off, not demonstrated dominance or a reliable success-rate increase.

### Paired uncertainty

Differences below are adaptive minus the reference method. Alignment and success are in percentage points; prediction differences are calls per complete attempt. Intervals are 95% paired-scene bootstrap intervals, not repeat-prediction reliability intervals.

| Cohort and reference | Success difference [interval] | Alignment difference [interval] | Prediction difference [interval] | Exact success-test p |
|---|---:|---:|---:|---:|
| 70–89, fixed-eight | −15 [−35, +5] | −3.20 [−9.50, +0.66] | +2.05 [−2.35, +6.80] | 0.375 |
| 70–89, fixed-twelve | +10 [−15, +35] | +3.65 [−6.46, +14.93] | +4.60 [+1.00, +8.35] | 0.6875 |
| 50–89, fixed-eight | 0 [−17.5, +17.5] | −0.24 [−4.03, +2.67] | −1.68 [−4.88, +1.58] | 1.0 |
| 50–89, fixed-twelve | +10 [−5, +25] | +5.32 [−1.32, +12.82] | +4.48 [+2.00, +6.98] | 0.34375 |

Success and alignment differences include zero throughout. Prediction cost relative to fixed-twelve is higher in both comparisons under these exploratory intervals. Full fixed-twelve versus fixed-eight statistics are retained in [summary.json](../results/robotics-cohort-completion/summary.json). No multiplicity correction or equivalence test was applied; non-significance is not evidence that the methods are identical.

### Verification and laptop resources

All 40 newly collected attempts finished without GPU-memory errors, excluded records or selective reruns. The collector took **65.12 minutes**, including preparation/verification inside the invocation. Peak process RAM was **2.52 GiB**, minimum available system memory **2.53 GiB**, and peak PyTorch GPU allocated/reserved memory **1.07/1.15 GiB**. These measurements confirm feasibility for this bounded offline experiment, not real-time deployment.

All **294 CPU tests** passed before and after collection. The [independent audit](../results/robotics-cohort-completion/independent-audit.json) passed, independently checking:

- 369 frozen source/evidence bindings and all 120 scored/reused records;
- 22,657 movements and 2,411 predictions, including 7,725 movements and 793 predictions in the 40 new baselines;
- all 40 exact first-eight-command/physics/RGB comparisons against the reused adaptive attempts;
- unchanged adaptive camera/controller arithmetic, queue boundaries and complete saved plans;
- all-attempt means, paired gains/losses, bootstrap intervals, exact tests and resource arithmetic.

This is an artifact/provenance audit. It did not rerun neural inference, physical/RGB replay or camera fitting. The prior milestone audit's limits for older unlogged discarded suffixes remain explicit. The independent auditor's SHA-256 is recorded in the audit artifact; its local diagnostic helper is not part of the method.

### Decision and remaining limits

**Preserve the milestone, but do not claim a demonstrated general improvement over fixed-eight.** The additional comparison did not reproduce the original simultaneous quality/efficiency gain, and pooling does not hide that disagreement. This task supplies a more complete, reproducible comparison; it does not supply a new controller or a validated fix.

No threshold/weighted filter was retested, no parameter was tuned and no success threshold was relaxed. The results do not establish which component helps, why each regression happens, performance under disturbances, real-camera accuracy or generalization outside this 2D PushT setup. There is one prediction sample per method per scene; all scenes are now inspected, and the pooled analysis is retrospective.

Before another controller change or large evaluation, discuss these mixed results with Qihang. If further failure diagnosis is selected, it should include the four newly lost successes, not only 54/59, and successful counterexamples. Do not automatically reopen the rejected filters, fit a fix to these scenes and call them unseen evaluation, or keep expanding trials until a favorable result appears.

## Sources

- [Original adaptive development screen](2026-10-12-robotics-adaptive-length.md): unchanged baseline loops and extension/interruption rules.
- [Milestone evaluation](2026-10-12-robotics-fresh-evaluation.md), [original protocol](../results/robotics-fresh-evaluation/protocol.json), and [all original attempts](../results/robotics-fresh-evaluation/episodes.json): first 20-scene comparison and all reused 50–69 records.
- [Completed gate comparison](2026-10-13-robotics-weighted-gate.md), [protocol](../results/robotics-weighted-gate/protocol.json), and [recorded attempts](../results/robotics-weighted-gate/episodes.json): unchanged `adaptive12` records on 70–89; rejected methods are not rerun.
- [Saved decision review](2026-10-13-robotics-decision-review.md): motivation for verifying the existing contribution before further tuning.
- [Completed comparison protocol](../results/robotics-cohort-completion/protocol.json), [attempt index](../results/robotics-cohort-completion/episodes.json), [summary](../results/robotics-cohort-completion/summary.json), [independent audit](../results/robotics-cohort-completion/independent-audit.json), and [resource measurements](../results/robotics-cohort-completion/run-resources-from-00.json): authoritative evidence for this report.

The simulator and pretrained policy remain reused infrastructure. This task contributes a reproducible completion of missing comparator evidence, not a new algorithm or a predetermined claim of improvement.
