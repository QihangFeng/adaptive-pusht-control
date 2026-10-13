# Does adaptive action reuse help on fresh pushing scenes?

Date: October 12, 2026

## Conclusion first

**The unchanged adaptive controller's favorable trade-off persisted on fresh scenes: better observed pushing results and lower computation cost than fixed-eight.** Across 20 previously unused scenes, it finished 18/20 pushes versus 15/20, increased average final alignment from 92.88% to 95.60%, used 22.22% fewer AI predictions, and ran 16.35% faster including monitoring.

This is stronger evidence than our eight reused development scenes, but not proof of a reliable success-rate increase: it rescued five baseline failures and lost two baseline successes. The success difference remains uncertain with this small sample. Fixed-twelve is still cheaper, with 16/20 successes and 88.61% average alignment. Our method therefore offers a quality/computation trade-off, not dominance over every comparator.

My recommendation is to keep this simple method for the course project, preserve these results without tuning on them, and next discuss a bounded ablation of extension versus interruption. The current evidence supports useful adaptive action reuse, not a separate benefit from the seven interruptions or a new adaptive-chunk algorithm.

The evaluation plan below was saved before collecting new outcomes. All 60 predetermined attempts completed in one collector invocation with the controller, trial selection and statistical plan unchanged.

## Question and frozen comparison

Does the unchanged camera/path-guided eight/twelve-action controller improve pushing quality or computation cost on previously unused initial scenes, compared with fixed-eight and fixed-twelve execution?

We use **20 predetermined fresh scene seeds, 50–69**, with all three methods on each scene: **60 complete attempts**. These seeds are separate from discovery seeds 0–29, previous held-out seeds 30–49, and the adaptive screen's reused development seeds 0–7. They represent new initial configurations of the same PushT task, not new tasks, objects, renderers or disturbances.

| Method | Behavior |
|---|---|
| Fixed-eight baseline | Predict, execute up to eight movements, then predict again; no added camera monitor |
| Fixed-twelve | Execute up to twelve movements per prediction; no added camera monitor |
| Adaptive eight/twelve | At movement eight, retain the four cached movements only if the existing camera/path rule permits; otherwise predict again, with the existing bounded early-interruption rule |

All methods directly reuse `evaluate_episode` from the frozen [adaptive-length runner](../experiments/robotics_adaptive_length.py). The [development report](2026-10-12-robotics-adaptive-length.md) defines its exact thresholds, extension/interruption guards, camera inputs, existing prefix checks and limitations. No parameters, neural weights, observation history, denoising schedule or source files from the previous pilots are changed.

The checkpoint is the pinned [LeRobot Diffusion Policy model](https://huggingface.co/lerobot/diffusion_pusht), revision `84a7c23178445c6bbf7e1a884ff497017910f653`. Each attempt has the same 300-movement limit and succeeds only when physical target coverage is strictly greater than 95%. Random generators reset to the scene seed before each method; sampling noise is aligned by prediction-call order, not physical time. There is **one prediction-randomness sample per method per scene**, not repeated independent policy seeds. Method order rotates across scenes to reduce systematic runtime-order bias.

## Main metrics and interpretation

| Metric | Definition |
|---|---|
| Successful pushes | Number of complete attempts reaching strictly greater than 95% target coverage |
| Average final alignment | Final fraction of target area covered by the block, averaged equally across all 20 attempts, including failures |
| Average AI calls | Number of local Diffusion Policy predictions, averaged across all 20 attempts; these are not LLM tokens or API calls |
| Average complete-attempt runtime | Wall-clock time for actions, physics, diagnostics, monitoring and decisions, averaged across all attempts; excludes model loading, environment creation and trace writing |

Camera monitoring cost is included in adaptive runtime and also reported separately. Simulator truth is used for scoring and diagnostic traces, not as an input to the adaptive decisions. This is offline simulation: physical time does not advance while the AI computes a prediction.

A quicker failed attempt is not automatically an improvement. We will retain every outcome, show gained and lost baseline successes, and discuss cost alongside success and final alignment. Fixed-twelve is a required comparator because longer action reuse alone may explain much of any efficiency gain. The combined rule does not isolate extension from interruption; any component-specific claim needs a later ablation.

## Statistical plan fixed before collection

The comparison units are the **20 initial scenes**, not the 60 method attempts. For each pair of methods, compute per-scene differences in success, final coverage, prediction calls and runtime. Report the mean difference and a paired percentile bootstrap 95% interval, using 10,000 resamples of whole scene pairs with random seed `20261012`. All contrasts use the same resampling indices.

Also report the exact two-sided discordant-success test, comparing gained versus lost successes within scene pairs. These intervals and tests are exploratory, with no multiplicity adjustment. Twenty scenes and one policy sample per scene do not establish universal benefit or isolate robustness to prediction randomness. A success improvement remains uncertain if the evidence includes plausible zero or negative benefit; we will not tune the controller or selectively rerun unfavorable scenes to change that conclusion.

## Evidence and safeguards

Before the first new prediction, [the protocol](../results/robotics-fresh-evaluation/protocol.json) binds the 60 scene/method pairs, unchanged rules, statistical plan, package versions, checkpoint identity, source hashes and previous evidence. The new wrapper has its own fresh-seed trial verifier and does not change the old runner's seed list. Each completed attempt has a hash-bound compressed trace and sidecar. Resuming verifies completed trials; orphan partial traces require inspection rather than silent replacement.

The completed audit verified all 60 unique pairs, trace hashes, identical initial physical states and camera-image hashes across each scene's three methods, recorded queue events, summary arithmetic and paired comparisons. Audit limits remain explicit: the initial camera fit and some unexecuted discarded suffixes are not fully reconstructable from the existing trace format. No concurrent GPU collector, package change or frozen-source edit occurred during collection.

## Results: all 20 fresh scenes

All metrics average equally across all attempts, including failures; no trials were excluded or selectively rerun. A successful attempt stops when coverage is strictly greater than 95%; an unsuccessful attempt reaches 300 movements.

| Method | Successful pushes | Average final alignment | Average AI calls | Average complete-attempt runtime |
|---|---:|---:|---:|---:|
| Fixed-eight baseline | 15/20 (75%) | 92.88% | 24.30 | 119.51 s |
| Fixed-twelve | 16/20 (80%) | 88.61% | 14.55 | 71.57 s |
| Adaptive eight/twelve | 18/20 (90%) | 95.60% | 18.90 | 99.97 s |

Evidence: [summary.json](../results/robotics-fresh-evaluation/summary.json), [all attempts](../results/robotics-fresh-evaluation/episodes.json), and [independent audit](../results/robotics-fresh-evaluation/independent-audit.json).

Relative to fixed-eight, adaptive execution saved **108 predictions and 390.72 seconds** across the complete evaluation. It used fewer predictions in 15/20 scenes and less runtime in 14/20. The savings are not solely faster failures: on the 13 scenes where both methods succeeded, mean calls fell from 18.69 to 15.38 and runtime from 91.27 to 81.07 seconds. This outcome-selected subgroup is descriptive, not a replacement for the all-scene comparison or an independent statistical test.

Relative to fixed-twelve, adaptive execution improved observed alignment by 6.99 percentage points and produced two additional successes, but used **29.90% more predictions and 39.69% more runtime**. Fixed-twelve remains the strongest efficiency comparator. Its zero-alignment failure in scene 68 has substantial influence on its mean alignment; that failure is retained rather than removed as an outlier.

### Which scenes changed outcome?

The seed is the reproducible initial-scene index, not a movement number. Each row starts from the same physical scene and image under all three methods.

| Scene | Fixed-eight | Fixed-twelve | Adaptive eight/twelve |
|---|---|---|---|
| 50 | Success | Success | Success |
| 51 | Success | Success | Success |
| 52 | Success | Success | Success |
| 53 | Failure | Success | Success |
| 54 | Success | Failure | Failure |
| 55 | Failure | Failure | Success |
| 56 | Success | Success | Success |
| 57 | Failure | Success | Success |
| 58 | Failure | Success | Success |
| 59 | Success | Success | Failure |
| 60 | Success | Success | Success |
| 61 | Success | Success | Success |
| 62 | Success | Success | Success |
| 63 | Success | Failure | Success |
| 64 | Success | Success | Success |
| 65 | Success | Success | Success |
| 66 | Success | Success | Success |
| 67 | Failure | Success | Success |
| 68 | Success | Failure | Success |
| 69 | Success | Success | Success |

Adaptive execution rescued all five baseline failures—53, 55, 57, 58 and 67—but lost baseline successes 54 and 59. In scene 54, final alignment fell from 95.26% to 94.49%, with five more predictions and 37.94 seconds more runtime. In scene 59, it fell from 95.36% to 91.87%, with eight more predictions and 52.51 seconds more runtime. These regressions rule out claiming that our method preserves every baseline success or saves computation in every scene.

Compared with fixed-twelve, adaptive execution gained successes 55, 63 and 68 and lost success 59. In scene 68, fixed-twelve ended at 0% alignment while fixed-eight and adaptive execution finished at 95.10% and 95.41%. This demonstrates a severe failure of longer fixed execution in one scene, not proof that camera feedback prevents every such failure.

### How certain are the differences?

Differences below are **adaptive minus comparator**. Positive success/alignment and negative calls/runtime favor adaptive execution. Coverage and success are shown in percentage points, not relative percentages.

| Comparator | Metric | Mean difference | Exploratory paired 95% interval |
|---|---|---:|---:|
| Fixed-eight | Success rate | +15.00 points | −10.00 to +40.00 |
| Fixed-eight | Final alignment | +2.72 points | +0.10 to +6.26 |
| Fixed-eight | AI calls | −5.40 | −9.20 to −1.80 |
| Fixed-eight | Runtime | −19.54 s | −38.48 to −1.14 s |
| Fixed-twelve | Success rate | +10.00 points | −10.00 to +30.00 |
| Fixed-twelve | Final alignment | +6.99 points | +0.09 to +17.52 |
| Fixed-twelve | AI calls | +4.35 | +0.95 to +7.75 |
| Fixed-twelve | Runtime | +28.41 s | +10.99 to +45.90 s |

The paired exact success tests give p = 0.453 against fixed-eight and p = 0.625 against fixed-twelve. Therefore, **do not call the success-rate gains statistically established**. The favorable cost and alignment intervals are useful exploratory evidence, but have no multiplicity correction; several bounds are close to zero. One stochastic policy sample per scene also leaves prediction-randomness reliability untested.

The defensible conclusion is: the fresh evaluation supports the observed efficiency/alignment trade-off against fixed-eight, while a reliable population success advantage and superiority over a well-chosen fixed interval remain unproven. It is not evidence about new tasks, real cameras, contact disturbances or real-time deployment.

## What the controller actually did

Adaptive execution extended 131 plans to twelve actions, trimmed 220 plans back to the eight-action interval, and interrupted seven times across six scenes. Average executed actions per prediction were 9.14 for adaptive execution, 7.87 for fixed-eight and 11.71 for fixed-twelve, calculated from total movements divided by total predictions. Partial final plans and interruptions explain values below the maximum lengths.

| Scene | Interrupt after movement(s) | Final adaptive outcome |
|---|---|---|
| 53 | 131 | Success |
| 54 | 193 | Failure |
| 59 | 151 | Failure |
| 62 | 85, 125 | Success |
| 65 | 160 | Success |
| 67 | 146 | Success |

Both adaptive failures had an interruption, but that observation does not establish that interruption caused them; changed action reuse also changes the subsequent trajectory. Scene 62 still succeeded, but used six more predictions and 37.36 seconds more runtime than fixed-eight. Scene 65 saved one prediction but ran 3.38 seconds slower once monitoring was included. These examples reinforce why call counts and complete-attempt runtime should both be measured.

Monitoring and decisions averaged **7.62 seconds per adaptive attempt**, already included in its 99.97-second runtime. Neither these results nor the development screen isolates extension from interruption. Do not describe all recovered pushes as interruption recoveries.

## Verification and laptop cost

All **152 robotics tests passed before and after collection**, including 28 new CPU-only tests for the fresh wrapper, protocol freezing, protected outputs, trial integrity, resumability, complete paired matrices and statistics. An independent reviewer checked the wrapper before collection; it directly invokes the unchanged frozen evaluator rather than duplicating or changing the controller.

The independent saved-artifact audit verified:

- all 60 sidecars and compressed trace hashes, 10,691 executed movements and queue transitions;
- all 3,435 logged camera-feature rows, independently recalculated from recorded camera coverage;
- all 131 fully executed retained command paths and all seven interruption guards;
- 51 source/evidence bindings, installed packages and checkpoint weights;
- all 20 matching initial scene/image trios, 20 unique new images distinct from the previous 50 scenes, and 160 matching first-prediction actions/states across the methods;
- all metric means, gained/lost successes, paired bootstrap intervals and exact success tests.

Audit limitations: no RGB estimate was re-fitted, and the 20 unlogged initial camera estimates were inferred only to check feature arithmetic. Full rejected suffixes are not saved; the command-path reasons for 48 camera-eligible trims cannot be independently reconstructed from the artifacts. These limitations are preserved in [independent-audit.json](../results/robotics-fresh-evaluation/independent-audit.json), not hidden by the audit's passing status.

A clean staged checkout preserved all 51 fresh-evaluation bindings and 60 trace hashes. The previous development screen's 47 bindings and 24 traces, and the earlier pilot's 50 bindings and 182 traces, also remained valid. Earlier frozen source and evidence were not changed.

The single collector invocation took **97.21 minutes**. Peak process memory was **2.57 GiB**, minimum available system memory **3.37 GiB**, and peak PyTorch GPU allocation/reservation **1.07/1.15 GiB**. GPU figures exclude some driver/desktop usage. All 60 attempts completed without GPU-memory errors, resumption or reruns. See [run-resources.json](../results/robotics-fresh-evaluation/run-resources.json). This confirms feasibility for the bounded offline evaluation on the GTX 1650, not real-time robot control or policy training.

## Recommended next decision

Keep adaptive action reuse as the course method: the fresh data provide a meaningful, reproducible efficiency/alignment result against the original baseline, with an explicit cheaper quality/computation comparator. Do not add a learned controller or tune thresholds merely to increase these observed scores.

Before claiming a contribution from interruption, discuss a small extension-only versus combined-rule ablation. Repeating prediction seeds under the same frozen rules would separately test stochastic reliability. Those are proposed next tasks, not experiments included here. Seeds 50–69 are now used evaluation data; any method revised after inspecting them needs a separate future evaluation rather than calling these scenes fresh again.

## Reproduction

Use the existing `.venv-robotics`, public checkpoint and laptop setup from the [feasibility report](2026-10-10-robotics-laptop-feasibility.md). No training or paid API is required. Keep reruns in the Git-ignored repeat directory so the authoritative evidence is not overwritten.

```powershell
.\.venv-robotics\Scripts\python.exe -m unittest discover -s tests -p 'test_robotics_*.py'
.\.venv-robotics\Scripts\python.exe experiments\robotics_fresh_evaluation.py --stage prepare --output-dir results/robotics-fresh-evaluation-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_fresh_evaluation.py --stage run --output-dir results/robotics-fresh-evaluation-repeat
```

`run` resumes verified completed attempts, but refuses to rerun a directory that already contains a completed summary. Use `--stage summarize` to inspect completed evidence. Resource measurements describe the latest collector invocation; if recovery requires resuming, distinguish those measurements from total collection time.

## Attribution

The frozen action predictor is reused [Diffusion Policy, Chi et al.](https://arxiv.org/abs/2303.04137); the physics, rendered camera and success criterion are reused [Gym-PushT](https://github.com/huggingface/gym-pusht). Our course contribution is the lightweight execution rule and its evaluation, not the simulator or pretrained policy. Adaptive replanning and action chunks have established precedent; see the [literature review](2026-10-11-robotics-replanning-literature.md). This fresh evaluation tests our adaptation, not a claim that adaptive execution itself is new.
