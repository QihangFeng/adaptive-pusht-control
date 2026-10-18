# Fresh-scene confirmation of recovery-rehearsal fine-tuning

## Conclusion and question

**The exact recovery-trained model retained a success and prediction-efficiency advantage in this larger, frozen confirmation.** It completed **60/80 pushes**, versus **48/80** for the unchanged milestone and **46/80** for the original fixed-eight baseline. Compared with the milestone, this is a **15-percentage-point success increase**, **11.15% fewer predictions** and **11.06% shorter average recorded runtime**. The trained version's success advantage appeared in both prediction draws. Its small average alignment increase remains uncertain, and some previously successful scene/draw pairs still failed.

These results are stronger evidence than the earlier twenty-scene pilot, not universal superiority or a new adopted milestone. All **240 predetermined attempts finished**, with no completed attempt replaced or excluded. There are **forty distinct scenes**, not eighty independent scenes per condition. The previous pilot and designated milestone `0f40449` remain unchanged.

Question: does the exact recovery-trained checkpoint retain its observed success/efficiency advantage on forty new initial scenes and more than one prediction sample per scene?

The [previous pilot](2026-10-16-robotics-recovery-pilot.md) observed trained/milestone/fixed-eight successes of 17/13/14 out of twenty. Those inspected outcomes motivated this confirmation; they are not independent confirmation data and were not pooled into the primary table.

## Locked comparison

| Condition | Model weights | Execution |
|---|---|---|
| Original fixed-eight | Original public pretrained checkpoint | Original fixed-eight rule |
| Unchanged milestone | Same original checkpoint | Unchanged camera/path-guided 8/12 rule |
| Recovery-trained milestone | Exact final-update-600 output block from the previous pilot | Same unchanged 8/12 rule |

The comparison used scene seeds **170–209**, two prediction draw offsets **60000/70000**, and all three conditions for every scene/draw pair: **240 new attempts, eighty per condition, forty distinct scenes**. Actual neural prediction RNG seeds are initial-scene seed plus draw offset. Initial physical configurations depend only on the scene seed. Method order rotates across scene/sample pairs.

No training, new data selection, threshold adjustment, generation-length change, denoising reduction or additional online model/candidate search occurs. The final trained head, original full checkpoint, camera estimator and both execution rules remain frozen. The new runner imports the unchanged episode evaluator; it does not modify old sources or patch their global settings.

All attempts use H16 generation, two observations, 100 DDPM denoising iterations, FP32 inference, the existing simulator and a 300-total-movement limit. Success requires simulator target coverage strictly greater than 95%. Simulator truth is scoring/validation-only, not an added controller input. Simulation pauses during model inference; this is not real-time robot deployment.

## Checks before and during collection

- Freeze a separate source/configuration/weight/protocol binding before fresh-scene predictions. Verify the previous pilot's trained artifact rather than creating another checkpoint.
- Validate original/trained/original head switching on an inspected scene. Original restored actions must match the existing original control; the genuinely changed trained model need not reproduce original actions.
- Load one GPU model and copy only the six existing output-block tensors between conditions. Check the frozen backbone and original weight file, and avoid concurrent GPU collectors.
- Retain the existing process/RAM/GPU resource guard. Resource or provenance failures stop execution rather than silently changing hardware or settings.
- Save each completed attempt, compressed trace and model-condition binding. Resume verifies existing evidence before collecting missing trials; do not selectively repeat outcomes or overwrite partial/orphan artifacts.
- Check common initial states for all methods and both draws, plus identical original-control first prediction prefixes for each draw. Sampling is paired by prediction-call order, not by physical time.

### Collection continuity

The first collector stopped after 127 completed attempts; its termination cause is unconfirmed. The unchanged CPU verifier accepted all 127 saved records and found no orphan/partial artifacts before the collector was relaunched to collect only the remaining **113 attempts**. No completed attempt was rerun, and no source, weight or setting changed.

The interrupted invocation did not leave its whole-invocation resource record. Saved per-attempt timings remain available, but complete collector elapsed time and whole-run RAM peaks cannot be reconstructed. The resumed invocation's resource record covers only its own segment. Its final backbone-unchanged assertion applies to the resumed model instance; it does not replace the missing first segment's end-of-invocation check. Recorded per-attempt weight-version checks, model identities, source/weight bindings and the original-head restoration smoke remain available. Runtime comparisons span collection sessions and should be interpreted descriptively, not as a controlled hardware-speed benchmark.

## Evaluation fixed before outcomes

Primary metric: successful pushes. Supporting metrics: average final alignment, average local policy predictions and average complete-attempt runtime. Means include failures. Prediction time and GPU memory are supporting checks of per-prediction computation. Episode time includes monitoring/decisions/physics and excludes loading, initialization and trace writing. Available collector resources are recorded separately. An AI call here means local Diffusion Policy inference, not an LLM API request, dollar cost or measured energy.

The primary contrast is trained milestone minus unchanged milestone. The original fixed-eight comparison is the secondary reference. Show both prediction samples separately as well as the complete eighty-attempt table, and report gained/lost successes rather than only net totals.

For uncertainty, first average the two paired condition differences within each initial scene, then bootstrap the **forty complete scene clusters**, keeping the two samples together. Use 10,000 resamples and statistics seed `20261017`. The same paired scene resamples apply to all four metrics. Do not treat eighty correlated attempts as eighty independent scenes or apply the old single-draw pooled exact success test to them.

Collect the complete predetermined comparison regardless of favorable or unfavorable intermediate results. No outcome-based parameter changes, optional scene additions, checkpoint selection or early statistical stopping. These are descriptive/exploratory intervals without multiplicity adjustment, not guaranteed statistical power or a calibrated pass/fail claim.

## Results

### All forty scenes and both draws

| Method | Successful pushes | Average final alignment | Average AI calls | Average runtime |
|---|---:|---:|---:|---:|
| Original fixed-eight baseline | 46/80 (57.5%) | 90.96% | 29.0250 | 143.26 s |
| Unchanged milestone 8/12 | 48/80 (60.0%) | 91.59% | 25.3375 | 135.95 s |
| Same recovery-trained model + unchanged 8/12 | **60/80 (75.0%)** | **91.88%** | **22.5125** | **120.92 s** |

Compared with the milestone, the trained version completed twelve more pushes and required 2.825 fewer predictions per attempt on average. Compared with the original fixed-eight baseline, it completed fourteen more pushes, used **22.44% fewer predictions** and took **15.60% less recorded time** on average. These are comparisons of all predetermined attempts, including failures; neither favorable scenes nor successful attempts alone were selected.

Between the trained and unchanged milestone, the action model's architecture, H16 generation, 100 denoising steps and execution rule did not change. Episode-weighted mean fresh-prediction times were **4.8885 s** for fixed-eight, **4.8831 s** for the milestone and **4.8748 s** for the trained version. This statistic first averages predictions within each episode, then averages episodes; it is not a pooled per-call benchmark. The maximum recorded per-attempt peak allocated GPU memory was **1,095.41 MiB in every arm**. The main observed efficiency saving is fewer predictions per attempt, not a reduced neural inference budget. The resumed-session timing limitation still applies.

### Prediction draw 60000

| Method | Successful pushes | Average final alignment | Average AI calls | Average runtime |
|---|---:|---:|---:|---:|
| Original fixed-eight | 22/40 | 90.61% | 29.400 | 145.14 s |
| Unchanged milestone 8/12 | 24/40 | 93.26% | 25.575 | 137.12 s |
| Recovery-trained model + unchanged 8/12 | **31/40** | **93.68%** | **22.200** | **119.66 s** |

### Prediction draw 70000

| Method | Successful pushes | Average final alignment | Average AI calls | Average runtime |
|---|---:|---:|---:|---:|
| Original fixed-eight | 24/40 | **91.31%** | 28.650 | 141.37 s |
| Unchanged milestone 8/12 | 24/40 | 89.93% | 25.100 | 134.79 s |
| Recovery-trained model + unchanged 8/12 | **29/40** | 90.07% | **22.825** | **122.17 s** |

The trained version has more successes and fewer calls in each draw. Alignment is not uniformly better: in the second draw the fixed-eight baseline's mean alignment exceeds the trained version's. Both draws use the same forty initial physical scenes, so these tables are not two independent forty-scene replications.

### Paired uncertainty and regressions

Each interval below resamples forty scene clusters after averaging the two paired draw differences within each scene. A positive success/alignment difference favors training; a negative call/time difference favors training.

| Trained minus reference | Metric | Observed difference | Paired-scene bootstrap 95% interval |
|---|---|---:|---:|
| Unchanged milestone | Success rate | +15.00 percentage points | +3.75 to +27.50 percentage points |
| Unchanged milestone | Final alignment | +0.29 percentage points | -3.08 to +3.78 percentage points |
| Unchanged milestone | Calls per attempt | -2.825 | -5.0375 to -0.7375 |
| Unchanged milestone | Runtime per attempt | -15.04 s | -26.76 to -4.11 s |
| Original fixed-eight | Success rate | +17.50 percentage points | +3.75 to +31.25 percentage points |
| Original fixed-eight | Final alignment | +0.91 percentage points | -2.64 to +4.38 percentage points |
| Original fixed-eight | Calls per attempt | -6.5125 | -8.95 to -4.1625 |
| Original fixed-eight | Runtime per attempt | -22.34 s | -34.70 to -10.64 s |

The success and prediction-count intervals exclude zero against both references in this comparison, unlike the earlier small pilot's primary intervals. This provides stronger within-task evidence for the frozen candidate. Alignment intervals include zero; the evidence does **not** establish better average final alignment. Runtime intervals also exclude zero, but bootstrap sampling does not remove hardware/session confounding.

Paired outcomes retain real regressions:

| Reference | Failures changed to successes | Successes changed to failures | Net extra successes |
|---|---:|---:|---:|
| Unchanged milestone | 19 | 7 | 12 |
| Original fixed-eight | 23 | 9 | 14 |

These counts refer to scene/draw pairs, not distinct scenes. Against the milestone, lost pairs were `(173,70000)`, `(177,70000)`, `(183,60000)`, `(196,70000)`, `(198,70000)`, `(202,60000)` and `(206,70000)`. No pair was selectively rerun to rescue a failure. The [full summary](../results/robotics-recovery-confirmation/summary.json) preserves every gained/lost pair, per-scene difference and both comparisons. We cannot claim a zero-regression improvement.

### Recorded laptop resources

The [resumed invocation record](../results/robotics-recovery-confirmation/evaluation-resources-from-127.json) covers completed-attempt counts **127 to 240 only**. That segment took **15,628.72 seconds (4.34 hours)** including loading, verification and collection, with sampled peak process RSS **2.55 GiB** and minimum available system RAM **4.30 GiB**. These are not the full experiment's elapsed time or peak RAM. The missing interrupted-invocation resource record is preserved as a limitation rather than estimated from the later segment. No new training occurred in this confirmation; the previous pilot's offline training cost remains separate.

## Interpretation and decision boundaries

This confirmation supports one frozen checkpoint's success/prediction-efficiency improvement within the same clean two-dimensional PushT task. It does not establish transfer to new tasks, shapes, real cameras, disturbances or new training seeds. New scene seeds are unseen in this project's tuning, not evidence that they were unseen in original model pretraining. Two prediction samples improve the check of sampling sensitivity but do not eliminate stochastic uncertainty.

The separate expert/recovery contributions remain unisolated: this is a confirmation of the combined fine-tuning recipe, not an expert-only or recovery-only ablation. It does not show that recovery labels specifically caused the gain, or that each component of the unchanged 8/12 controller is necessary. There is still only one training seed and one chosen final600 checkpoint. Sparse, correlated recovery data and the expert-video limitations documented in the previous pilot remain relevant.

Preserve the exact checkpoint, controller and evidence as a supported candidate, and preserve milestone `0f40449` as the designated recovery reference. This report does not adopt a replacement milestone, alter the method or authorize another experiment. Scenes 170–209 are now inspected evidence; do not tune on them and then call them fresh. Any follow-up design or evaluation requires discussion first.

## Verification

All **693 CPU tests passed before collection**, including 38 new tests for the confirmation wrapper. They cover the exact schedule, rotation, protected outputs, source/weight bindings, safe resume, orphan/partial-record refusal, original-head restoration and scene-cluster statistics. An independent code review found no blocking issue. No GPU inference or training was performed by the tests. Actual-GPU original/trained/original switching checks passed before scored scenes, including exact original-head restoration and the saved original baseline prefix.

The [independent completed-results audit](../results/robotics-recovery-confirmation/independent-audit.json) passed. It checked **67 protocol fingerprints**, all **240 scored record/sidecar/trace bindings**, **53,358 movements**, **6,150 predictions**, four smoke records and saved controller-event arithmetic. It independently reproduced comparison tables, scene-cluster statistics and gained/lost pairs, and checked the matched initial states/original-control prefixes. It did not repeat neural inference, physical simulation or hardware timing. The full CPU suite was rerun after collection: **all 693 tests passed**, with zero failures/errors and no frozen-source changes.

## Reproduction

Use the existing `.venv-robotics` environment. Do not launch another GPU collector while this run is active. Sources must remain frozen after preparation.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_confirmation.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_confirmation.py --stage run
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_confirmation.py --stage verify
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_confirmation.py --stage summarize
```

Shareable evidence belongs in `results/robotics-recovery-confirmation/`; local public/trained weights remain Git-ignored. The new collector must preserve all earlier results, especially `results/robotics-recovery-pilot/`.

## Sources

- [Previous recovery-rehearsal pilot](2026-10-16-robotics-recovery-pilot.md): frozen training recipe, data provenance, final-update-600 checkpoint and initial results.
- [Milestone evaluation](2026-10-12-robotics-fresh-evaluation.md): unchanged camera/path-guided 8/12 controller and metrics.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137), [pinned checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653), [LeRobot implementation](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py), and [Gym-PushT](https://github.com/huggingface/gym-pusht): reused neural policy, inference and physical simulator.
