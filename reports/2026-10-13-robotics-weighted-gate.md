# Weighted evidence for interrupting robot movement plans

Date: October 13, 2026

## Current conclusion

**The weighted rule did not demonstrate an advantage over the simpler threshold rule. Keep the milestone controller unchanged.** On 20 new scenes, both gates succeeded 16/20 times versus the unchanged controller's 15/20. Both rescued failures 72/81 but lost success 73 and used slightly more predictions and runtime. The simple gate had higher observed average alignment than weighting; neither gate is a proven upgrade.

The approved experiment is complete: a development-only parameter search, frozen settings, three preflight attempts and 60 fresh full attempts. All 247 robotics tests and the independent artifact audit passed. No completed trial in this new experiment was rerun, no fresh outcomes were used to retune, and no earlier frozen controller or result was changed. Preflight deliberately reproduced one earlier unchanged-controller trajectory as an integration check.

The approved contribution is a lightweight mathematical decision rule around the frozen Diffusion Policy—not a newly trained robot model. It decides whether an already-confirmed alignment decline should interrupt cached movements. We preserve the original eight/twelve-action reuse logic, camera estimator and milestone `0f40449`.

## Question and proposed methods

Can combining decline size, warning count and recent improvement/worsening reduce harmful interruptions without losing useful replanning, compared with the unchanged adaptive controller and a simpler two-threshold gate?

Let \(\hat c_t\) be camera-estimated target coverage after movement \(t\). The existing monitor supplies:

- \(d_t\): recent peak over eight camera values minus current coverage;
- \(n_t\): number of drops greater than 1.5 percentage points among the latest five changes.

The new trend is \(r_t=\hat c_t-\hat c_{t-2}\). It uses three consecutive valid camera values. Positive values indicate recent improvement; negative values indicate deterioration. A negative value closer to zero means slower decline, **not recovery**. Missing camera history is not bridged.

Normalize the signals using fixed, round scales specified before the search:

\[
D_t=\min(d_t/0.20,1),\qquad W_t=n_t/5,\qquad
R_t=\operatorname{clip}(r_t/0.10,-1,1).
\]

The scales mean 20 percentage points of drawdown and 10 percentage points of net change over two movements. They are engineering choices for this pilot, not statistically established optimal scales. Decline, warnings and trend overlap; normalization does not remove that correlation.

The weighted gate permits interruption when

\[
S_t=w_DD_t+w_WW_t-w_RR_t\geq\tau.
\]

The selected configuration is **\((w_D,w_W,w_R)=(0.50,0.25,0.25)\), \(\tau=0.50\)**. Recovery reduces the score; ongoing deterioration raises it. This score is an engineering heuristic, not a calibrated probability that interruption will help. The simpler selected gate requires **at least three warnings AND at least 15 percentage points of drawdown**. Mixed cases that fail either threshold keep ordinary scheduled feedback.

Neither gate can force an interruption that the original controller would reject. Original confirmation, minimum plan age, pending-action count, recent-peak requirement, interruption cap and cooldown all remain in place. A veto preserves confirmation history but does not consume interruption budget or cooldown. It is reconsidered at the next movement, and normal trimming at eight actions still occurs. Thus “continue” does not mean execute indefinitely without observing or replanning.

## Bounded development selection

Reuse the [completed decline screen](2026-10-12-robotics-decline-screen.md): six saved contexts, three paired prediction samples each, 36 existing continuation records. Do not rerun those trials or pool the older fixed-eight panel into this search.

The fixed grid has twelve weighted configurations: four weight triples crossed with thresholds 0.50, 0.35 and 0.65. Six simple configurations cross warning limits 3/2 with drawdown limits 0.15/0.10/0.20. Select separately within each family by conditional successes first, fewer AI predictions second, higher final alignment third, then fixed candidate order. Runtime is not a selection criterion because some saved records come from different sessions.

Both selected configurations accept scenes 54/59/67 and veto scenes 53/62/65. **Eight weighted configurations and three simple configurations tie with that same winning decision pattern.** Particular weights are therefore not uniquely identified.

Choosing the corresponding saved arm gives a conditional development objective of 12/18 successes, 198 predictions and 90.61% mean final alignment. **These are not measured results of either new controller.** The reused “continue” arm suppresses interruption until the next prediction; the new gates reconsider each movement. Their later decisions, prediction timing and interruption budgets can differ. Selection on already-inspected outcomes is optimistic, and this six-context objective is only a development proxy.

No neural policy or separate learned classifier is trained. This is outcome-based tuning of a small, explicit controller parameter grid, not a claim that weights were chosen independently of labels.

## Frozen full-attempt evaluation plan

Before fresh collection, freeze the selected configurations, formulas, checkpoint, packages, sources/evidence hashes and trial identities. Preflight comprises an unchanged-controller whole-trajectory reproduction of saved scene 62, plus functional full attempts of both new gates on previously used development scene 0. These do not authorize another tuning round. CPU tests cover accepted/vetoed interruptions, restored budget/cooldown, missing images and the age-eight fallback even if a smoke scene lacks an eligible decline.

Then run **20 unused initial scenes, seeds 70–89, three methods, 60 full attempts**:

1. unchanged milestone adaptive controller;
2. threshold-gated adaptive controller;
3. weighted-gated adaptive controller.

Use the same scene and prediction seed across arms, rotate method order, and retain the 300-total-movement limit. Prediction randomness is paired by AI-call order, not physical time. There is one prediction sample per scene/method; this cannot establish reliability across repeated samples. Keep these fresh outcomes out of all parameter selection.

Primary comparison metrics are task success, average final alignment, average AI predictions and average runtime including monitoring. Include failures in all means. Report gained/lost paired successes, exact discordant-success tests and paired-scene bootstrap intervals. Interruption/extension/veto counts are explanatory diagnostics, not additional optimization targets. Compare weighting directly with the threshold rule, not just the old controller.

Success means the simulator terminates after coverage exceeds 95%; otherwise the attempt times out at 300 movements. Final alignment is terminal object/target overlap coverage, scored using simulator truth, not the camera estimate. AI predictions count new twelve-command plans, not executed movements or language-model tokens. Runtime is measured full-attempt wall time including monitoring; it is not a real-time deployment measurement.

Every fresh prediction logs its complete twelve-command plan, including later discarded suffixes. Camera history and gate values are logged separately from simulator truth, which remains scoring/validation-only. Saved records are verified before resuming; completed trials are not rerun. Run only one GPU collector, and do not edit frozen sources during collection.

## Reproduction

Use the existing `.venv-robotics`, pinned checkpoint and [laptop setup](2026-10-10-robotics-laptop-feasibility.md). For disposable reproduction, use a separate Git-ignored output directory in every stage:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_gate_evaluation.py --stage prepare --output-dir results\robotics-weighted-gate-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_gate_evaluation.py --stage smoke --output-dir results\robotics-weighted-gate-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_gate_evaluation.py --stage run --output-dir results\robotics-weighted-gate-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_gate_evaluation.py --stage summarize --output-dir results\robotics-weighted-gate-repeat
```

Authoritative evidence is saved under `results/robotics-weighted-gate/`; the default completed collector refuses to rerun it. Reproduction uses the same inspected scenes and is not a new generalization test. No paid API, new dataset or new hardware is required.

## Results and next decision

### Completed preflight

All **247 robotics tests** passed, including 34 new CPU tests. An independent preparation check verified 223 frozen source/evidence bindings and recomputed all eighteen candidate configurations, selection objectives and ties.

The new unchanged-controller loop reproduced all **194 movements** of saved scene 62: actions, physical coordinates, RGB hashes, camera features, events and query counts remained exact; the largest simulator-coverage rounding difference was only \(4.44\times10^{-16}\). Both new gates also completed development scene 0 without execution errors, although both timed out with 66.80% final alignment and 28 predictions. They logged zero vetoes in that scene, so actual veto/fallback coverage comes from the passed CPU tests rather than these two GPU attempts.

The independent smoke check audited three records, 794 movements and 79 complete predicted plans, including discarded suffixes. Preflight took **7.21 minutes**, with peak process RAM **2.54 GiB** and peak allocated GPU memory **1.07 GiB**. This is functional evidence, not proof of improvement. Settings were not changed after preflight.

### Fresh results

All 60 attempts completed. Means include successes and timeouts. The reference here is **our unchanged adaptive eight/twelve-action controller**, not the published fixed-eight baseline.

| Method | Successful pushes | Average final alignment | Average AI predictions | Average runtime, including monitoring |
|---|---:|---:|---:|---:|
| Unchanged milestone controller | 15/20 (75%) | 92.02% | 23.15 | 125.91 s |
| Simple threshold gate | 16/20 (80%) | 94.58% | 23.55 | 128.14 s |
| Weighted gate | 16/20 (80%) | 92.13% | 23.45 | 127.72 s |

Both gates gained scenes **72 and 81** and lost **73** relative to the unchanged controller. Weighted and simple gating had identical success/failure outcomes on every scene. A net gain of one success out of twenty is not strong evidence of a reliable improvement.

The following paired differences are method minus reference. Intervals resample 20 scene pairs 10,000 times; alignment and success changes are percentage points, not relative percentages.

| Comparison | Success change, 95% interval | Alignment change, 95% interval | Prediction change, 95% interval | Runtime change, 95% interval |
|---|---:|---:|---:|---:|
| Simple minus unchanged | +5.00 [−10.00, +20.00] pp | +2.56 [−0.58, +8.10] pp | +0.40 [−0.30, +1.30] | +2.23 [−1.70, +7.08] s |
| Weighted minus unchanged | +5.00 [−10.00, +20.00] pp | +0.11 [−0.37, +0.71] pp | +0.30 [−0.35, +1.20] | +1.81 [−1.98, +6.72] s |
| Weighted minus simple | 0.00; identical observed outcomes | −2.46 [−7.91, +0.55] pp | −0.10 [−0.25, 0.00] | −0.42 [−2.04, +1.08] s |

The exact two-sided discordant-success test is \(p=1.0\) for each comparison. The gates' empirical success-difference bootstrap interval is [0, 0] because their observed binary outcomes match; **this does not prove population equivalence**. All other intervals include zero. These are exploratory comparisons without multiplicity adjustment, not proof that methods are equivalent or that a gain cannot exist.

### Why fewer interruptions did not guarantee better results

The unchanged controller interrupted 11 times, the simple gate twice and the weighted gate five times. The gates vetoed 27 and 11 per-movement candidates respectively; these are repeated decision checks, not distinct failure cases. Rejecting an interruption can change the subsequent trajectory and lead to more predictions later, so a veto is not automatically a saved AI call.

Two examples show why mild-looking declines do not reliably determine which choice is better:

| Scene | First changed decision, after movement | Signals at that decision | Unchanged controller | Both gates |
|---|---:|---|---|---|
| 72 | 146; six cached movements remain | Two warnings; 5.20 pp drawdown; latest two-movement change −0.13 pp | Interrupt; eventually fails, 34 predictions | Continue; eventually succeed, 31 predictions |
| 73 | 144; eight cached movements remain | Two warnings; 6.48 pp drawdown; latest two-movement change −0.13 pp | Interrupt; eventually succeeds, 27 predictions | Continue; eventually fail, 34 predictions |

Within each scene, actions, physical coordinates and RGB observations are identical through the listed decision; the first changed action is the next movement. These are observed complete-trajectory comparisons, not claims that the selected moment was objectively the start of failure. Similar local signals led to opposite eventual effects.

The weighted and simple gates executed identically on **18 of the 20 scenes**. Their two effective differences explain the average-alignment gap:

| Scene | First different decision, after movement | Weighted choice | Simple choice | Final alignment: weighted / simple |
|---|---:|---|---|---:|
| 80 | 148 | Interrupt: score 0.576 ≥ 0.50 | Continue: drawdown 13.40 pp < 15 pp | 39.63% / 92.40% |
| 82 | 170 | Interrupt: score 0.645 ≥ 0.50 | Continue: drawdown 12.73 pp < 15 pp | 84.65% / 80.99% |

Both scenes failed under both gates. Scene 80 accounts for −2.64 percentage points of the weighted-minus-simple mean; scene 82 offsets +0.18 points. Thus weighting permitted an extra interruption that was harmful in one observed trajectory and somewhat helpful in another. This compares the complete decision rules; it **does not isolate the recovery/trend term's causal effect** or establish a universal failure pattern.

### Verification and laptop cost

The [independent audit](../results/robotics-weighted-gate/independent-audit.json) passed: 223 frozen bindings, all 63 preflight/fresh records, 13,653 movements, 1,482 complete predicted plans, gate formulas, camera-window arithmetic, controller guards and veto bookkeeping, prediction queues, metrics and paired statistics. All **247 robotics tests** passed after collection. The audit independently recomputed logged evidence; it did not rerun neural inference, physics, rendering or RGB fitting, or reconstruct latent observation/RNG bytes.

Fresh collection took **127.65 minutes**, with peak process RAM **2.51 GiB**, minimum system-available RAM **1.73 GiB**, and peak allocated/reserved GPU memory **1.07/1.15 GiB**. These measurements cover all 60 attempts in one collector invocation; preflight's 7.21 minutes is separate. No collection or GPU-memory errors occurred. See [summary.json](../results/robotics-weighted-gate/summary.json), [episodes.json](../results/robotics-weighted-gate/episodes.json) and [run-resources.json](../results/robotics-weighted-gate/run-resources.json).

### Decision and limitations

**Do not replace the milestone with the weighted gate.** It added no observed successes beyond the simpler rule and had lower observed average alignment; its 0.10-prediction and 0.42-second advantages over the simple rule are too small and uncertain to justify the added score. The simple gate is a more promising quality-focused candidate in these data, but its slight success/alignment gains over the unchanged controller remain uncertain and cost slightly more computation. It is not an adopted upgrade either.

This answers the approved question without training a larger classifier: combining these three signals did not demonstrate a useful advantage in this bounded experiment. It does not mean recovery information can never help. Six development contexts, tied configurations, a one-shot selection proxy used for a per-movement gate, and one stochastic prediction sample per evaluation scene limit that conclusion. The fixed, rendered 2D task does not establish real-camera robustness, transfer to other manipulation tasks, deployment safety or training feasibility.

Preserve the completed evidence and milestone `0f40449`. Seeds 70–89 are now inspected evaluation data, not unseen scenes for a future revision. Before any further experiment, discuss whether to pursue a more informative decision signal or focus the course project on the existing adaptive action-reuse quality/computation trade-off. Do not merely retune on scene 80 or promise to preserve every successful push.

## Sources and reused work

- [Diffusion Policy](https://arxiv.org/abs/2303.04137), [LeRobot](https://github.com/huggingface/lerobot), and the [pinned checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653): frozen action predictor and public weights.
- [Gym-PushT](https://github.com/huggingface/gym-pusht): simulator, rendered observations and task-success definition.
- [Our adaptive-length screen](2026-10-12-robotics-adaptive-length.md) and [fresh-scene milestone](2026-10-12-robotics-fresh-evaluation.md): unchanged action reuse, interruption safeguards and reference controller.
- [Our loss diagnosis](2026-10-12-robotics-regression-diagnosis.md): reused interruption-veto/state-restoration helper.
- [Our decline screen](2026-10-12-robotics-decline-screen.md): development inputs and matched continuation outcomes.

The new contribution being tested is the weighted/threshold decision extension and its controlled comparison, not a claim that weighted event triggering or adaptive replanning is entirely new.
