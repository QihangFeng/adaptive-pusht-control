# PushT failure patterns and a bounded recovery investigation

Date: October 11, 2026

## Conclusion

**The unchanged baseline has recurring failures, and earlier replanning can sometimes recover. However, this simple regression-triggered interruption has not established a net benefit and can harm an otherwise successful continuation. Do not adopt it as the final controller yet.**

We completed all three planned steps: 30 baseline attempts (20 successes, 10 timeouts), failed/successful trajectory comparison, and 24 matched-state continuations across four selected scenes. In the three originally failed scenes, replanning succeeded in 4/9 trials versus 2/9 for continuing. In the originally successful control, it succeeded in 1/3 versus 3/3. Overall, both arms succeeded in 5/12 continuations, while replanning required more AI calls.

This supports studying **when to interrupt**, not claiming that interruption is already an improvement. Adaptive action-chunk length, a visual monitor and held-out controller evaluation remain unimplemented and unvalidated.

## Question and scope

Does the unchanged pretrained baseline exhibit a recurring execution pattern that an earlier prediction can help recover from?

We reuse the public Diffusion Policy checkpoint and PushT setup from the [feasibility report](2026-10-10-robotics-laptop-feasibility.md) and [six-attempt baseline screen](2026-10-10-robotics-baseline-check.md). This is failure diagnosis and a recovery opportunity test, not the final interruption/adaptive-length controller and not a reproduction of the original paper's reported benchmark score.

## Discovery design

- Thirty total attempts use predetermined seeds 0-29. Seeds 0-5 reuse their existing recordings; only seeds 6-29 are newly simulated.
- The checkpoint, two observation frames, 16-step prediction horizon, eight-action execution queue, 100 DDPM denoising steps, FP32 inference, 300-action episode budget, and absence of injected disturbances remain unchanged.
- Successful and failed attempts are both retained. Success means target coverage strictly greater than 95%; a timeout is a task failure, not a hardware failure.
- Simulator contact, geometry and coverage are privileged **diagnostics**, not extra inputs to the frozen policy or an implemented visual monitor.
- Initial scenes and diffusion sampling both vary with the episode seed; this collection does not isolate their individual causal effects.
- These are discovery attempts. Later controller comparisons require separate, unused evaluation seeds.

The authoritative [protocol](../results/robotics-failure-patterns/protocol.json) was saved before the new collection, including original-evidence hashes, seed lists, model configuration, pattern rules and recovery selection. The original six recordings remain in their original directory rather than being rerun or duplicated. Their hashes are checked again at completion.

### Exploratory patterns, not failure causes

The existing six attempts informed two rules, frozen before collecting the 24 new attempts:

1. **Near-target cached-action regression:** within one action chunk, past coverage reached at least 90%; on a cached action, coverage is at least five percentage points below that peak, with at least one action still pending. Retain the first qualifying event per chunk; use only information through the detected event, not a future peak. The peak can include the state immediately before the chunk's first action, so the rule does not attribute the entire decline to cached actions.
2. **Low-progress window:** over 24 actions, the global best coverage improves by at most one percentage point while coverage at both endpoints is at least 50%. Overlapping windows are retained as overlapping measurements, not counted as independent failures; merged spans aid review.

Original contact-loss and contact-stall candidates remain supporting information. None of these flags is an exclusive or sufficient failure label. Temporary regression can occur during useful repositioning or rotation, and an attempt can later recover. Pattern association alone does not establish the cause of a timeout.

## Completed baseline findings

All 30 predetermined attempts completed without a hardware failure: **20 successes and 10 timeouts**. The original six contribute three successes and three timeouts; the 24 new attempts contribute 17 successes and seven timeouts. The 24 new runs took 47.20 minutes in aggregate, with new predictions averaging 4.84 seconds (weighted by prediction count).

Every timeout reached at least 90% target coverage, but none crossed the required 95%. In plain language, the recurring broad problem is **getting close without finishing the final alignment**, sometimes followed by losing alignment again. This is not the same as failing to reach or touch the block.

| Pattern present at least once | Timed-out attempts | Successful attempts | Interpretation |
|---|---:|---:|---|
| Near-target cached-action regression | 6/10 | 1/20 | A promising recovery candidate, but not a sufficient failure label |
| Low-progress window | 10/10 | 8/20 | Common in failures, but also part of successful execution |
| Contact-loss candidate | 9/10 | 7/20 | Supporting information; unsuitable as the sole interruption trigger |
| Contact-stall candidate | 2/10 | 0/20 | Rare under the existing rule; not a general explanation |

There are 11 near-target regression events across seven scenes. All have actual pusher-block contact at detection, so **contact loss would not detect these particular regressions**. The rules were informed by the original six attempts; among the 24 new attempts, regression appears in four of seven timeouts and one of 17 successes. This is descriptive discovery evidence, not a statistically validated classifier.

Timeouts last 300 actions, whereas successful attempts often stop earlier. Longer trajectories have more opportunities to contain a flag, so these presence counts alone cannot establish predictive reliability. Likewise, stationary endings in failures cannot be directly compared with successful episodes that terminate as soon as the target is reached.

| Failed seed | Best coverage | Final coverage | Best-coverage action | Near-target regression events |
|---|---:|---:|---:|---:|
| 2 | 94.88% | 87.57% | 281 | 3 |
| 3 | 94.62% | 93.64% | 168 | 0 |
| 4 | 94.65% | 90.43% | 222 | 1 |
| 6 | 94.48% | 94.48% | 284 | 1 |
| 7 | 92.19% | 90.25% | 249 | 1 |
| 9 | 90.33% | 64.70% | 129 | 0 |
| 11 | 94.77% | 66.48% | 161 | 0 |
| 15 | 94.04% | 73.72% | 288 | 2 |
| 16 | 92.75% | 83.48% | 121 | 2 |
| 18 | 94.77% | 82.00% | 134 | 0 |

The pattern does not explain all timeouts. For example, seed 6 recovered after an earlier regression but then made no further block movement after action 284 through timeout, despite new predictions at actions 289 and 297. Its next regular prediction after the earlier regression, at action 177, also worsened coverage before later recovery. Therefore, recognizing a bad-looking movement does not guarantee that the frozen policy can correct it.

## Conditional recovery pilot

If the near-target cached regression occurs in at least two eligible failed scenes, select the first three qualifying failed seeds in numerical order and the first qualifying successful seed as a comparator, if available. Use the first eligible event chronologically in each scene, with at least 16 actions left before the original deadline. Do not retune thresholds or choose the largest drop to manufacture a promising example.

For each selected branch, replay the saved action prefix. Position, velocity, block geometry and camera hash must match the recording; only coverage permits the existing 1e-12 floating-point tolerance. Restore the two-frame observation history.

Each scene uses all three fresh diffusion seeds, 2000, 2001 and 2002, and two arms:

| Arm | Decision at the branch | Subsequent execution |
|---|---|---|
| Continue cached | Restore and execute the still-pending original actions | New predictions use the original fixed-eight setting |
| Replan once | Discard the pending actions and request one prediction immediately | New predictions use the original fixed-eight setting |

Both arms stop at success or **300 total actions**, including the shared prefix. They do not receive 300 additional actions. The continue arm must reproduce the recorded physical states until its first new prediction or recorded success, directly checking queue restoration.

The branch occurs after the qualifying decline has already happened: this tests recovery, not prevention of that decline. One interruption also shifts subsequent fixed-eight prediction times, so its effect includes the resulting change in prediction timing rather than only an isolated extra AI call.

The same fresh random seed is reset independently for each arm. This aligns randomness by prediction order, not by control time: the two arms make predictions at different times and observations. The continue arm's later predictions are a **fresh stochastic continuation**, not an exact replay of the original future random draws. The recorded suffix therefore remains a third reference, not the sole comparison. A successful restart against an old failed recording is insufficient if the fresh continue arm also succeeds.

Report all paired outcomes: success, final and post-branch best coverage, total actions, total AI calls and wall time. Interruption does not necessarily add exactly one total call: it shifts query timing and may change the time to success. The selected scenes, not the multiple random trials within a scene, are the scenario units. Scene selection uses simulator diagnostics and known episode outcomes, so this is an **oracle-selected opportunity test**, not a deployable controller or held-out evaluation.

### Prespecified selected scenes

| Seed | Recorded outcome | Branch action | Previous within-chunk peak → branch coverage | Pending actions discarded by replanning |
|---|---|---:|---|---:|
| 2 | Timeout | 157 | 93.58% → 86.19% | 3 |
| 4 | Timeout | 223 | 94.65% → 85.30% | 1 |
| 6 | Timeout | 172 | 90.77% → 81.93% | 4 |
| 28 | Success control | 142 | 94.35% → 89.02% | 2 |

These follow the frozen selection order, not the largest observed loss. The successful comparator subsequently reached 96.16% coverage in its original recording, showing that a regression can be temporary and self-correcting under fixed-eight execution.

## Recovery results

All 24 planned continuations completed. Each table entry averages all three samples in that scene, including timeouts; no best restart is selected. Target coverage measures how much of the target region is covered by the block. A high average coverage is not equivalent to a high success rate.

| Selected scene | Continue: successes | Replan: successes | Mean final coverage: continue → replan | Mean total AI calls: continue → replan |
|---|---:|---:|---|---|
| Failed seed 2 | 1/3 | 2/3 | 93.91% → 94.44% | 36.00 → 36.67 |
| Failed seed 4 | 0/3 | 0/3 | 86.51% → 82.00% | 38.00 → 38.00 |
| Failed seed 6 | 1/3 | 2/3 | 93.95% → 95.48% | 38.00 → 34.67 |
| Successful seed 28 | 3/3 | 1/3 | 95.71% → 94.99% | 23.33 → 34.33 |

### Every paired outcome

“Replan only” means only immediate replanning reached the target; “continue only” means only continuing reached it. Each pair starts from the same recorded physical state and uses the same fresh diffusion seed.

| Scene | Fresh sample | Continue final coverage | Replan final coverage | Task-success outcome |
|---|---:|---:|---:|---|
| 2 | 2000 | 95.17% | 92.37% | Continue only |
| 2 | 2001 | 93.68% | 95.23% | Replan only |
| 2 | 2002 | 92.88% | 95.71% | Replan only |
| 4 | 2000 | 92.19% | 92.89% | Neither |
| 4 | 2001 | 87.40% | 89.31% | Neither |
| 4 | 2002 | 79.93% | 63.81% | Neither |
| 6 | 2000 | 92.78% | 95.02% | Replan only |
| 6 | 2001 | 93.74% | 96.66% | Replan only |
| 6 | 2002 | 95.32% | 94.75% | Continue only |
| 28 | 2000 | 95.20% | 91.83% | Continue only |
| 28 | 2001 | 96.82% | 94.96% | Continue only |
| 28 | 2002 | 95.11% | 98.18% | Both |

Four pairs favor replanning, four favor continuing, three fail under both, and one succeeds under both. The overall 5/12 tie is **not proof of equivalence**; the scenes are few and deliberately selected. Conversely, reporting only the failed-scene subset would conceal the harmful interventions in the successful control.

Across the 12 continuations per arm, replanning uses **167 remaining AI calls versus 142** for continuing (25 more, about 17.6%). Including the identical prefixes, mean total calls are 35.92 versus 33.83. Mean measured continuation wall time is 69.20 versus 59.01 seconds; this excludes model loading, prefix replay and history restoration, and is not full-episode runtime or real-time robot latency. The 24 continuations took 25.64 minutes in aggregate, in addition to baseline collection and analysis.

All per-trial actions, prediction counts, final/post-branch best/full-episode best coverage, success flags and timings are retained in [recovery.json](../results/robotics-failure-patterns/recovery.json) and its referenced compressed traces. The original baseline recordings are a separate reference, not the counterfactual arm.

### Visual checks

CPU-only replay regenerated the selected baseline states and all 24 continuation endpoints, checking physical state and camera hashes against the saved traces. Frame sheets: [seed 2](../results/robotics-failure-patterns/figures/seed-2-recovery-frames.png), [seed 4](../results/robotics-failure-patterns/figures/seed-4-recovery-frames.png), [seed 6](../results/robotics-failure-patterns/figures/seed-6-recovery-frames.png), and [seed 28](../results/robotics-failure-patterns/figures/seed-28-recovery-frames.png). Images illustrate the recorded behavior; they do not reveal the policy's intended motion or independently establish its cause.

## Decision and next discussion

**Keep the robotics problem, but keep the proposed interruption/adaptive-length method tentative.** We have established that the baseline sometimes fails near the target and that changing prediction timing can rescue some matched continuations. We have not established a reliable observable interruption signal or a better controller.

Before implementing the extension, discuss how a monitor could distinguish persistent loss of alignment from temporary useful repositioning, using camera-accessible information rather than privileged simulator contact/coverage. A decline alone is insufficient. Simply learning to shorten future chunks after every such flag could carry the same false-alarm problem into the adaptive rule; this experiment does not test that rule.

If we agree on a defensible trigger, later evaluation should compare it with reasonably tuned fixed execution lengths, include successful trajectories and separate unused seeds, and count both policy and monitoring computation. Those are future experiments, not results from this investigation. No new controller, parameter sweep or scope change was implemented here.

## Verification and evidence

- All 30 baseline artifacts and 24 recovery artifacts passed independent summary/trace consistency checks, including ordered/unique trial keys, coverage, success/truncation, query counts and action budgets.
- Original six-attempt protocol, summary and trace hashes remain unchanged; the frozen checkpoint/configuration and experiment rules match the protocol.
- Continued cached actions reproduced their recorded physical states until the first fresh prediction. Observation-history restoration was reviewed in the code; history frames are not separately persisted in each trial file.
- All 37 robotics unit tests passed. They cover existing baseline helpers, frozen trace rules, selection and resume guards; they do not constitute a statistical validation of the proposed controller.
- Shareable evidence: [baseline summary](../results/robotics-failure-patterns/summary.json), [protocol](../results/robotics-failure-patterns/protocol.json), [trace analysis](../results/robotics-failure-patterns/analysis.json), and [recovery results](../results/robotics-failure-patterns/recovery.json).

## Reproduction

Use the existing `.venv-robotics` environment and pinned local checkpoint. No training, paid API or new dependency is required. Preserve the authoritative output; use a separate `*-repeat/` directory for a fresh collection.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_failure_investigation.py --stage collect --output-dir results\robotics-failure-patterns-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_failure_investigation.py --stage analyze --output-dir results\robotics-failure-patterns-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_failure_investigation.py --stage recover --output-dir results\robotics-failure-patterns-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_failure_investigation.py --stage review --output-dir results\robotics-failure-patterns-repeat
.\.venv-robotics\Scripts\python.exe -m unittest discover -s tests -p 'test_robotics_*.py' -v
```

`collect --resume` resumes a matching protocol without rerunning already-saved attempts. Recovery also resumes completed trial records, checking case selection and evidence files. Original six-attempt references remain necessary for reproducibility. Full new baseline videos are local under `tmp/robotics-failure-pattern-videos/`; compressed traces, summaries and selected frames are shareable under `results/`.

## References

1. Chi et al. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023; [authors' project](https://diffusion-policy.cs.columbia.edu/).
2. Hugging Face. [LeRobot v0.3.3 Diffusion Policy implementation](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py), including observation/action queues and inference-time sampling.
3. Hugging Face. [Public Diffusion Policy PushT checkpoint](https://huggingface.co/lerobot/diffusion_pusht), pinned revision `84a7c23178445c6bbf7e1a884ff497017910f653`; weight hash recorded in the protocol.
4. Hugging Face. [Gym-PushT simulator](https://github.com/huggingface/gym-pusht), installed version 0.1.5. The simulator advances only when stepped; this remains offline evaluation, not proof of real-time robot deployment.
