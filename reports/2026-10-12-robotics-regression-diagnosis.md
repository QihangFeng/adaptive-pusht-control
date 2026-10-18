# Why did adaptive execution lose baseline successes 54 and 59?

Date: October 12, 2026

## Conclusion first

**Keep the milestone controller unchanged for now. Neither tested timing change fixed a failed remaining attempt, and both changes spoiled the same successful scene-59 continuation.** Replanning one movement earlier is not an established fix; simply delaying interruption is not an established fix either.

The two milestone losses are also **sensitive to prediction randomness**. With two new prediction seeds per scene, unchanged adaptive execution succeeded in 1/2 scene-54 repeats and 2/2 scene-59 repeats; fixed-eight succeeded in 0/2 and 1/2. This does not prove the losses were “just randomness”: adaptive action reuse changes the observations and timing of subsequent stochastic predictions. We cannot cleanly blame either randomness alone or a single controller decision.

All 26 planned trials completed, both restoration controls passed, and the independent artifact audit passed. The corrected collector took 44.63 minutes on the GTX 1650. All 183 robotics tests passed. No controller parameters, model weights or earlier frozen results were changed.

## Question and preserved milestone

The completed fresh-scene evaluation is the designated **milestone**, commit `0f40449` (full `0f4044966646df4cc38b0373fa52c0b733506e15`). It remains unchanged. This investigation uses separate code and evidence, and will produce a new commit rather than reset the milestone.

Both scene 54 and scene 59 succeeded with fixed-eight execution but failed with adaptive eight/twelve. We investigate whether those losses are sensitive to prediction randomness and whether a specific local decision can improve recovery. This is a **selected-case diagnosis**, not a generalization test, a component ablation, or adoption of a revised controller.

## What the existing traces show

The large near-target alignment drops happened while the pusher was still touching the block. They began within the first eight movements of a newly predicted plan, not during its extra movements 9–12. The existing interruption occurred after the drop had started. Therefore, neither lost contact nor the interruption itself explains the start of that drop.

| Scene | Newly predicted plan begins | First full existing warning condition | Existing interruption | Normal eight-movement boundary |
|---|---:|---:|---:|---:|
| 54 | Movement 189 | After 192 | After 193 | After 196 |
| 59 | Movement 145 | After 150 | After 151 | After 152 |

For scene 54, actual alignment fell from 79.68% after movement 190 to 75.48%, 68.30%, and 60.22% after 191–193. For scene 59, it fell from 92.49% after 148 to 80.88%, 66.35%, and 51.56% after 149–151. These are scoring/diagnostic measurements, not runtime controller inputs; the unchanged controller uses camera estimates.

A dip alone does not prove a failed plan. In the existing fixed-twelve run, scene 59 recovered from a large decline and eventually succeeded. We must compare complete remaining attempts, not only the next movement or a short-term alignment increase.

Sources: [milestone report](2026-10-12-robotics-fresh-evaluation.md), [saved source analysis](../results/robotics-regression-diagnosis/saved-analysis.json), and the hash-bound source traces identified in the new protocol.

## Bounded plan, specified before collecting new outcomes

The plan contains **26 trials**, with no outcome-dependent additions:

1. **Two restoration controls.** Regenerate the original adaptive prefix for each scene and restore the earlier checkpoint. Continue with the original random-generator state and unchanged controller. Every remaining movement, physical/image state, visual feature, queue boundary and decision event must match the milestone trace, not just its final outcome. Only simulator truth coverage uses its existing absolute rounding tolerance of `1e-12`; everything else stays exact. Stop the local experiments if restoration fails.
2. **Eight full-scene sampling repeats.** For each scene, compare fixed-eight and unchanged adaptive execution using two new prediction seeds: `54001, 54002` for scene 54 and `59001, 59002` for scene 59. The simulator scene seed stays fixed. This tests sensitivity, not a reliable success probability.
3. **Sixteen matched-state remaining-episode probes.** At each of four checkpoints, use future prediction seeds `9100, 9101` and compare two options:
   - After 192/150: keep the existing decision, or replan immediately, one movement earlier. The forced replan consumes one interruption and respects the original cap, minimum age, pending-action and cooldown guards.
   - After 193/151: apply the existing interruption, or defer interruption until this unit ends. Deferral preserves ordinary eight-action trimming/extension; it does **not** blindly execute all twelve movements. The unchanged rule resumes at the next prediction, and a suppressed interruption does not consume its budget.

Policy sampling is stochastic Diffusion Policy prediction, **not LLM generation**. Fresh prediction seeds are paired by future prediction-call order, not by physical movement number. Changing prediction timing also changes subsequent observations and actions; these effects interact.

All trials retain the original **300 total movements**, including the restored prefix. We restore all action/observation queues, visual pose/history, controller state and random generators. Only the pending action queue is discarded for replanning; observation history is preserved. No simulator contact/pose information enters the decisions.

The two checkpoints per scene share a prefix and are not independent scenes. Local calls/runtime refer to the remaining part only; they must not be directly compared with full-scene calls/runtime. Model loading, prefix regeneration, state replay and trace writing are excluded from per-trial runtime; camera monitoring is included. Method/arm order alternates across the two prediction seeds. No training, paid API or additional hardware is needed.

## Reproduction

Use the pinned `.venv-robotics` environment and checkpoint from the [feasibility report](2026-10-10-robotics-laptop-feasibility.md). Do not run another GPU collector concurrently or modify any frozen sources during collection.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_regression_diagnosis.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_regression_diagnosis.py --stage run
.\.venv-robotics\Scripts\python.exe experiments\robotics_regression_diagnosis.py --stage summarize
```

The protocol freezes all 26 identities, settings, package/source hashes and milestone evidence before collecting scored trials. Completed trials resume only after verification; orphan partial traces require inspection. A completed summary cannot be silently rerun or overwritten. Disposable reruns require a separate `*-repeat/` directory, never an earlier evidence directory.

An initial restoration gate stopped before any trial was saved because it incorrectly required bitwise equality of simulator coverage. A focused replay of all 108 remaining movements found only coverage-rounding differences, at most `3.33e-16`; commands, physical coordinates, RGB hashes, visual features and queue boundaries stayed exact. Gym-PushT unions shapes in set iteration order. We corrected the checker to reuse the already-established `states_match` tolerance, reran tests, and froze the corrected protocol before collecting the 26 trials. No controller rule, checkpoint, prediction seed or outcome selection was changed.

## Results

### 1. Full-scene repeats: prediction randomness matters

Each row below averages two complete attempts on the **same initial scene**, using the two new prediction seeds. Failures are retained. The original milestone attempts are not included in these averages.

| Scene | Method | Successful pushes | Average final alignment | Average AI predictions | Average complete-attempt runtime |
|---|---|---:|---:|---:|---:|
| 54 | Fixed-eight | 0/2 | 93.72% | 38.00 | 182.91 s |
| 54 | Unchanged adaptive eight/twelve | 1/2 | 94.08% | 31.00 | 161.46 s |
| 59 | Fixed-eight | 1/2 | 82.52% | 26.50 | 128.21 s |
| 59 | Unchanged adaptive eight/twelve | 2/2 | 95.46% | 20.00 | 104.66 s |

The original fixed-eight run succeeded on both scenes, but three of these four new baseline repeats failed. The original adaptive run failed on both, but three of four new adaptive repeats succeeded. Therefore the original labels “baseline success” and “adaptive failure” are outcomes of particular prediction samples, not guaranteed properties of those scenes.

In the four new paired samples, adaptive gained two successes and lost none. This is descriptive evidence from two retrospectively selected scenes, not a new held-out success-rate estimate or proof of dominance. The samples remain useful for diagnosing sensitivity.

### 2. Local comparisons: neither proposed change established recovery

These start at **saved moments in the original failed adaptive trajectories**, not at the beginning of a new scene. Each method is tested with future prediction seeds `9100` and `9101`.

| Scene and decision point | Option | Successful remaining attempts | Average final alignment | Average remaining AI predictions | Average remaining runtime |
|---|---|---:|---:|---:|---:|
| 54, after movement 192 | Original decision | 0/2 | 64.53% | 12.50 | 65.08 s |
| 54, after movement 192 | Replan immediately, one movement earlier | 0/2 | 87.26% | 12.00 | 62.67 s |
| 54, after movement 193 | Original interruption | 0/2 | 64.53% | 12.50 | 65.22 s |
| 54, after movement 193 | Finish the current unit before predicting again | 0/2 | 60.89% | 12.00 | 62.78 s |
| 59, after movement 150 | Original decision | 1/2 | 95.15% | 13.50 | 70.57 s |
| 59, after movement 150 | Replan immediately, one movement earlier | 0/2 | 92.17% | 18.00 | 93.77 s |
| 59, after movement 151 | Original interruption | 1/2 | 95.15% | 13.50 | 70.65 s |
| 59, after movement 151 | Finish the current unit before predicting again | 0/2 | 93.08% | 18.00 | 93.73 s |

Earlier replanning improved final alignment in **both** scene-54 samples, by 4.88 and 40.59 percentage points, but neither reached success within the remaining budget. The higher average alignment is a real conditional result, not a demonstrated rescue or a general improvement.

Deferral in scene 54 had opposing outcomes: final alignment fell by 56.91 points under seed `9100`, but rose by 49.64 points under `9101`. Neither succeeded. Its average hides substantial sample sensitivity.

The clearest counterexample is **scene 59, future seed `9101`**:

- From after movement 150, the original choice executed movement 151, interrupted, and ultimately succeeded at movement 231 with 96.25% alignment. Replanning before movement 151 timed out at movement 300 with 89.89% alignment.
- From after movement 151, the original interruption reproduced that same successful future. Deferral instead executed the remaining cached eighth movement, 152, before the ordinary trim and next prediction; it timed out with 93.14% alignment.

These are two alternative changes to the **same successful continuation**, not two independent successful scenes. The adjacent original branches produced identical movements from 152 onward for each paired future seed. They must not be counted as independent evidence.

“Finish the unit” did not mean execute every cached movement: scene 54 executed cached movements 194–196, and scene 59 executed cached movement 152, before normal trimming. Both then resumed the unchanged adaptive controller. In scene 59, changing even this one movement altered the later stochastic trajectory substantially.

### 3. Restoration, integrity and hardware

The two unchanged original-RNG controls reproduced all **258 remaining movements** from the milestone: 108 for scene 54 and 150 for scene 59. Commands, physical positions/velocities, images, visual features, queue boundaries and events matched exactly; simulator coverage differed only within the stated `1e-12` bound. Their original failures and total prediction counts were reproduced.

The independent audit verified all 26 prescribed identities, 4,120 recorded movements, 3,078 recorded camera-feature rows, 67 frozen source/evidence bindings, paired initial states, intervention guards, cached deferred commands, summary arithmetic and package/model identity. It found no provenance errors. It did not independently rerun the neural model, physics or RGB fitting; latent observation-queue/RNG payloads are represented by published hashes, not their full bytes. Thirty-one camera-eligible trims still have unlogged discarded paths, so the audit does not claim to reconstruct every rejected suffix.

The corrected collector completed in **44.63 minutes**, including prefix regeneration but excluding model loading and the earlier preflight investigation. Peak process RAM was **2.06 GiB**, minimum available system RAM **4.74 GiB**, peak allocated GPU memory **1.07 GiB**, and reserved GPU memory **1.15 GiB**. No GPU-memory errors or concurrent collectors occurred.

Evidence: [summary](../results/robotics-regression-diagnosis/summary.json), [all 26 trial records](../results/robotics-regression-diagnosis/trials.json), [restored-prefix audit](../results/robotics-regression-diagnosis/restore-prefix-audit.json), [independent audit](../results/robotics-regression-diagnosis/independent-audit.json), and [resource measurements](../results/robotics-regression-diagnosis/resources.json). Each trial has its own hash-bound compressed movement trace and sidecar under `trials/`.

## Interpretation and recommended next decision

1. **Do not change the warning confirmation length or disable interruption based on these results.** Earlier replanning helps alignment in one scene but harms a successful continuation in the other. Deferral is similarly inconsistent.
2. **Keep the milestone as the working method, not a guarantee of never losing a baseline success.** The original 20-scene quality/computation results are preserved. The new data explain why a single success or failure cannot establish a deterministic controller defect.
3. **If we continue improving the method, inspect the earlier action-reuse decisions and transitions between predictions before choosing another fix.** The first command divergence was movement 9 in both scenes, while the large declines occurred much later, inside fresh eight-action prefixes. Our local results do not establish whether earlier extensions set up those later states or whether the frozen policy would also fail under similar observations. Compare saved loss trajectories with rescued and retained-success trajectories; do not infer a solution from losses alone.

This recommendation is for a focused diagnostic discussion, **not authorization to run a broad ablation or implement a new risk model**. A substantive revised rule should be discussed first and evaluated later on new scene/prediction seeds. Scenes 54/59 and these probe outcomes are now development information; they are not unseen evaluation data for a method designed using this report.

We have not found a reliable fix for both losses. That is a useful negative result: it prevents adopting two plausible but unsupported changes, while leaving our positive milestone intact.

## References and reused work

- [Diffusion Policy](https://arxiv.org/abs/2303.04137): frozen action predictor; we do not retrain it.
- [LeRobot implementation](https://github.com/huggingface/lerobot) and [pinned public checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653): reused policy implementation/weights.
- [Gym-PushT](https://github.com/huggingface/gym-pusht): reused physical simulator, rendering and success criterion.
- [Our frozen adaptive-length screen](2026-10-12-robotics-adaptive-length.md) and [fresh-scene milestone](2026-10-12-robotics-fresh-evaluation.md): existing controller and original regression evidence.

Our contribution in this task is the controlled diagnosis and reproducible evidence, not a new robot policy or a claimed novel interruption algorithm.
