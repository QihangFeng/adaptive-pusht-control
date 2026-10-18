# When does an alignment decline justify replanning?

Date: October 12–13, 2026 (specified October 12; completed October 13)

## Conclusion first

**We found a promising distinction, but not a validated controller fix.** In the three situations with two warnings and smaller alignment drops, continuing succeeded **7/9** times versus **2/9** after immediate replanning. In the three situations with three warnings and larger drops, continuing succeeded **0/9** times versus **5/9** after replanning. These are six saved situations, each tested with three future prediction samples—not eighteen different scenes.

Replanning was not universally better: both choices succeeded **7/18** times overall. Warning count, decline size and current alignment are correlated here, so we cannot identify one as the cause or claim that the split will hold elsewhere. The older fixed-eight comparisons also contain counterexamples to a simple “larger decline means replan” rule.

The useful next discussion is a **small additional check before interrupting**, rather than training a larger classifier: is alignment still falling substantially now, or is the warning mainly remembering earlier drops? A post hoc check supports investigating this distinction, not adopting it yet. The milestone controller and model weights remain unchanged; this task does not establish a fix for the original losses 54/59.

## Purpose

This investigation asks whether information already available to the controller can distinguish alignment decreases where early replanning helps from those where continuing is preferable. It does **not** train a classifier, change the milestone controller, or claim that a state is absolutely unrecoverable.

The approved scope is: inspect saved declines in successful and failed attempts, compare matched continuing/replanning choices with multiple prediction samples, and examine warning count, accumulated drop, recovery and movement descriptors. The designated milestone remains `0f40449`; all earlier frozen source and results remain unchanged.

Two questions must stay separate: whether an attempt can succeed after continuing the current unit, and whether interrupting improves success or computation cost. Continuing may fail while replanning also fails; both may succeed while one uses fewer predictions. “Recovery” here permits ordinary scheduled feedback afterward—it does not mean executing forever without another prediction.

## Existing evidence, reused rather than rerun

The [earlier benefit-aware pilot](2026-10-11-robotics-benefit-pilot.md) already contains five decline checkpoints and twelve stagnation checkpoints. The five declines have three matched future samples each, giving **15 pairs / 30 saved continuations**, not 15 independent scenes. Both choices return to fixed-eight behavior after the initial decision.

| Discovery scene and movement | Continue successes | Replan successes | Paired success gains / losses |
|---|---:|---:|---:|
| 2, after 157 | 0/3 | 2/3 | 2 / 0 |
| 9, after 153 | 0/3 | 0/3 | 0 / 0 |
| 15, after 218 | 1/3 | 1/3 | 1 / 1 |
| 16, after 132 | 2/3 | 1/3 | 1 / 2 |
| 18, after 97 | 3/3 | 3/3 | 0 / 0 |

Together, continuing succeeded 6/15 times and replanning 7/15, with four gains and three losses. Scene 16 had the largest camera drawdown, yet replanning reduced its observed success count. Scene 18 had the smallest drawdown; both choices succeeded, but replanning saved 9–11 predictions. A larger decline is therefore not already established as a reliable trigger, and zero success difference does not imply zero efficiency value.

At scenes 15 and 16, the same past observations led to both beneficial and harmful replans under different future random samples. No function of that identical prefix can perfectly predict each sampled future. These old data already influenced training/validation of the earlier gate and are development evidence, not new validation.

The [loss diagnosis](2026-10-12-robotics-regression-diagnosis.md) also supplies eight directly relevant adaptive comparisons: at the actual interruptions in scenes 54/59, two future seeds compare immediate replanning against finishing the current unit. Its earlier adjacent checkpoints instead test one-action timing differences and are **not** added as independent contexts here.

The old fixed-eight panel and the adaptive panel below remain separate because their future controller behavior differs.

## Adaptive screen, specified before new outcomes

Select **every first actual interruption** in the milestone's 20 adaptive attempts. This deterministic rule yields six contexts, without selecting on future comparison outcomes:

| Scene | Interrupt after movement | Original full-attempt outcome |
|---|---:|---|
| 53 | 131 | Success |
| 54 | 193 | Failure |
| 59 | 151 | Failure |
| 62 | 85 | Success |
| 65 | 160 | Success |
| 67 | 146 | Success |

Original outcomes provide context only, not decision features or outcome labels for the new futures. Selecting actual interruptions restricts conclusions to moments accepted by the existing warning rule; it does not test whether the rule finds every useful decline or other replanning opportunity. Only the first interruption is selected when an episode contains two.

Each context has future prediction seeds **9100, 9101, 9102** and two choices:

- **Replan now:** apply the existing interruption, discard cached actions, consume its original interruption budget/cooldown, and retain observation history.
- **Continue the current unit:** suppress interruption until the next prediction while preserving ordinary eight-action trimming/extension; do not consume the suppressed interruption's budget. Then resume the same frozen adaptive controller.

This comparison includes the downstream consequences of budget/cooldown consumption. It is not a permanent switch-off of interruption, and continuing does not necessarily execute every pending movement. Both choices retain **300 total movements including the prefix**, not 300 additional movements. Future randomness is paired by prediction-call order, not physical time.

There are **36 comparison records / 18 paired futures / six distinct source contexts**. Eight records from scenes 54/59 and two earlier restoration controls are reused by hash-bound reference; no completed trial is rerun. Collect only **28 missing comparisons and four new restoration controls: 32 new trials**. Exact source-prefix regeneration is still necessary to recover hidden cached actions and observation/RNG state; it is setup, not another scored full attempt.

Before collecting missing comparisons, regenerate all six prefixes, match the published old snapshot audits for 54/59, verify reused controls, and pass the four new unchanged original-RNG suffix controls. Only simulator truth coverage uses its existing `1e-12` rounding tolerance; actions, physical coordinates, RGB hashes, camera features, queues and events remain exact.

## Observable inputs and analysis

We keep the existing camera features: coverage, warning count, drawdown, recent recovery, position/orientation error, recent gain, pending actions. Add only inexpensive **past-only** descriptors:

- Accumulated drop: sum of negative coverage changes over the latest five changes, without subtracting recoveries.
- Position/orientation error trend: current camera error minus the error three actions earlier.
- Warning persistence: consecutive complete existing regression conditions ending at the current movement.
- Plan age and remaining execution budget.
- Command-direction consistency: average cosine between neighboring nonzero displacement vectors among the latest six executed pusher target commands.
- Command progress toward the static goal: decrease in distance of the commanded pusher targets to the known goal, divided by world width 512.

The command descriptors describe the **pusher commands**, not actual block motion or proof of a useful push. No dynamic simulator pose, contact, true coverage, original success/failure or future observations enter these features. Invalid recent camera frames produce missing derived features; invalid commands are rejected. Actual physics coverage remains scoring/diagnostic evidence only.

Primary outcomes are paired success gains/losses and per-context successful futures. Supporting outcomes are final alignment, remaining predictions and remaining runtime, keeping failed attempts. If both choices succeed, check efficiency separately. Camera monitoring is included in runtime; model loading, prefix generation/replay and trace writing are excluded. Eight reused comparison runtimes come from a prior session, so pooled timing is secondary descriptive evidence rather than a controlled hardware comparison.

Also distinguish **recovery before the next scheduled prediction** from **eventual task success after later feedback**. For the continue arm, measure whether cached movements regain the recent camera peak before its first new prediction. A missing or invalid cached camera frame leaves a negative result unknown unless an earlier measured frame already shows recovery; the frozen collector omits monitoring on terminal rows. This is an outcome used for analysis, never a decision input.

Before new outcomes, freeze seven hypothesized feature directions: more warnings, greater drawdown/accumulated drop, more negative recovery, increasing position/orientation errors, and less consistent command direction might favor replanning. For each, report the fraction of non-tied context pairs whose feature ordering agrees with ordering of the empirical success benefit. Missing values and ties are skipped. These are small, exploratory rank associations—not classifier accuracy, causal feature effects, tuned thresholds, calibrated probabilities or independent validation. There is no multiplicity correction.

The key question is whether useful interruptions and harmful interruptions recur across distinct situations and have reasonably consistent observable differences. Opposite outcomes under identical inputs show uncertainty that a more complicated classifier cannot eliminate. Any favorable association would justify discussing a further study, not adopting a rule fitted to these six contexts.

## Reproduction and evidence

Use the pinned `.venv-robotics` environment and checkpoint from the [laptop feasibility report](2026-10-10-robotics-laptop-feasibility.md). No training dependencies, paid APIs or extra hardware are needed. Keep one GPU collector and do not edit frozen code during collection.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_decline_screen.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_decline_screen.py --stage run
.\.venv-robotics\Scripts\python.exe experiments\robotics_decline_screen.py --stage summarize
```

The separate protocol binds all context/spec identities, reuse paths, formulas, package versions and source/evidence hashes before new scored trials. Resume verifies saved trials and never silently reruns a completed summary; orphan traces require inspection. Disposable reproduction uses a separate Git-ignored `*-repeat/` directory. The new runner imports the earlier diagnosis's restoration/collection helpers without editing them.

## Results

### Which choice helped in each situation?

All prescribed comparisons and restoration checks completed. Each fraction below counts three future prediction samples from the same saved situation. Alignment drop is the camera-estimated difference from the recent peak, in percentage points (pp); it is not a success label.

| Scene, decision after movement | Warnings | Alignment drop | Continue successes | Replan successes | Observed better choice |
|---|---:|---:|---:|---:|---|
| 53, after 131 | 2 | 8.48 pp | 1/3 | 0/3 | Continue |
| 54, after 193 | 3 | 18.84 pp | 0/3 | 1/3 | Replan |
| 59, after 151 | 3 | 41.34 pp | 0/3 | 1/3 | Replan |
| 62, after 85 | 2 | 10.49 pp | 3/3 | 0/3 | Continue |
| 65, after 160 | 2 | 9.28 pp | 3/3 | 2/3 | Continue |
| 67, after 146 | 3 | 22.96 pp | 0/3 | 3/3 | Replan |
| **Total** | — | — | **7/18** | **7/18** | Neither universally |

There were **five paired success gains and five losses** from replanning; eight pairs had equal success outcomes. Scene 67 originally succeeded under the adaptive controller, yet deferring its interruption failed in all three new samples. Therefore, original full-attempt success is not a reason to classify every decline in that attempt as harmless.

For a concrete example, restore scene 62 **after movement 85**. Continuing executes cached movements **86–88**, then trims normally and requests updated predictions. With prediction seed 9100 it succeeds at movement **101**, using two remaining AI predictions. Immediate replanning at the same starting point times out at movement 300, using 27 predictions. All three future samples favored continuing here. This does **not** mean cached movements alone repaired the alignment: they did not regain the earlier peak; later ordinary visual feedback completed the task.

Indeed, **none of the eighteen continue-arm futures regained the recent camera peak before their first new prediction**, although seven eventually succeeded. “Better to continue this unit” and “the object immediately returns to its earlier alignment” are different claims.

Supporting whole-continuation means, including failures:

| Metric | Continue current unit | Replan now |
|---|---:|---:|
| Final alignment, simulator scoring | 88.68% | 87.49% |
| Remaining AI predictions | 12.56 | 15.61 |
| Remaining runtime, including monitoring | 66.36 s | 82.62 s |

These are selected-state continuation metrics, not a new full-controller evaluation. Eight timings are reused from the earlier session; runtime is descriptive. The efficiency result does not justify disabling interruption, given its success gains in the other three situations.

### Do the observable signals distinguish benefit?

The pre-specified directional associations were strongest for warning count: **9/9** non-tied comparisons between context averages agreed with the hypothesis that more warnings favor replanning. Drawdown, accumulated drop, recent decline and orientation-error trend agreed in **10/13**; position-error trend and command consistency in **9/13**. These comparisons reuse the same six contexts: **9/9 is not 100% prediction accuracy**. Warning persistence was two at every selected checkpoint, so it did not distinguish them.

The warning groups also separate decline size and current coverage. Two-warning contexts have coverage **72.26–83.33%** and drawdown **8.48–10.49 pp**; three-warning contexts have coverage **50.79–71.09%** and drawdown **18.84–41.34 pp**. We cannot tell which correlated feature matters most. Three samples estimate benefit imprecisely, and the older fixed-eight panel does not reproduce the same warning/magnitude ordering. No calibrated probability, universal threshold or held-out classifier performance is established.

### A possible explanation: old warnings can outlast steep decline

The following check was added **after seeing outcomes**, so it is exploratory, not part of the frozen primary test. Subtract the previous camera coverage from the coverage at each decision point, using only the saved source prefix:

| Scene | Latest one-movement camera change |
|---|---:|
| 53 | −1.448 pp |
| 54 | −7.818 pp |
| 59 | −14.404 pp |
| 62 | −0.059 pp |
| 65 | −1.305 pp |
| 67 | −9.249 pp |

Every latest change is still negative. However, the declines have become much smaller at the three points where continuing did better. Our warning count remembers significant drops over the latest five changes, so an interruption can remain eligible even after the current decline slows. In scene 62, for example, the camera barely changes at movement 85 while two earlier warnings remain active.

This suggests checking **ongoing decline**, not interpreting every accumulated warning as a new emergency. It does not prove intentional repositioning, physical recovery or a causal mechanism. Camera-change errors matter near a threshold: scene 53's latest simulator-scored drop is 2.416 pp versus the camera's 1.448 pp. The earlier near-target validation recorded a 1.126 pp 95th-percentile camera-change error and 4.498 pp maximum. Simulator truth is diagnostic only and cannot be used by a camera-only gate.

### Integrity and laptop cost

All **213 robotics tests** passed. The independent audit verified all 36 comparison records, all six restoration controls, 160 frozen bindings, 4,871 adaptive movements and 597 exact restored suffix movements; it separately checked the 30 older fixed-eight continuations. No provenance error was found. It did not rerun neural inference, physics or RGB fitting, and latent queue/RNG payloads are hash inventories rather than independently reconstructed bytes. Twenty-seven camera-eligible trims retain unlogged discarded-suffix limits. The post hoc latest-change table is a separate direct calculation from saved source prefixes, not a frozen primary association.

The new collector took **46.97 minutes**, including prefix regeneration, with peak process RAM **2.06 GiB**, minimum available system RAM **3.33 GiB**, peak allocated GPU memory **1.07 GiB**, and reserved GPU memory **1.15 GiB**. This remains feasible on the laptop; no additional training or hardware was needed.

Evidence: [frozen protocol](../results/robotics-decline-screen/protocol.json), [selected cases](../results/robotics-decline-screen/cases.json), [legacy inventory](../results/robotics-decline-screen/legacy-inventory.json), [all comparison records](../results/robotics-decline-screen/comparisons.json), [summary and cached-recovery checks](../results/robotics-decline-screen/summary.json), [restoration audit](../results/robotics-decline-screen/restoration-audit.json), [independent audit](../results/robotics-decline-screen/independent-audit.json), and [resources](../results/robotics-decline-screen/resources.json).

## Recommended next discussion, not an adopted method

Discuss one lightweight additional gate on an already-confirmed interruption: require evidence that a substantial decline is **still happening**, using the latest camera change together with warning count/accumulated drop. Reusing the existing 1.5 pp warning scale is a candidate starting point, not a validated threshold. Adding such a gate is different from simply increasing the original confirmation length, because that can change which observations are available before the decision.

Do not train a larger classifier from six contexts, claim that an outcome-selected combination of arms is a new controller result, or promise to preserve every baseline success. A revised gate can change future interruption budgets and trajectories, so it would need complete-attempt testing on new scene/prediction seeds after we agree on its definition. The present screen supports that next discussion; it does not already establish the revision's benefit. Preserve the milestone and all frozen evidence.

## Sources

- [Diffusion Policy](https://arxiv.org/abs/2303.04137), [LeRobot implementation](https://github.com/huggingface/lerobot), and the [pinned checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653): reused action predictor and weights.
- [Gym-PushT](https://github.com/huggingface/gym-pusht): reused simulator, rendering and success definition.
- [Benefit-aware pilot](2026-10-11-robotics-benefit-pilot.md), [milestone fresh evaluation](2026-10-12-robotics-fresh-evaluation.md), and [loss diagnosis](2026-10-12-robotics-regression-diagnosis.md): prior source trajectories, labels, controller and restoration evidence.

This task contributes a controlled diagnostic comparison and observable-signal analysis, not a new robot policy or a claim of algorithmic novelty.
