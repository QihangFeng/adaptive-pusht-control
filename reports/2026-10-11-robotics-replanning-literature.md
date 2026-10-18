# When should a frozen robot policy replan? An expanded literature review

Date: October 11, 2026

## Conclusion

**Our question is reasonable, but it is not an unexplored question. There are established methods for choosing execution length, detecting trouble during execution, and preserving useful motion when generating a new plan. PACE is only one reference.**

This targeted review covers **17 relevant papers beyond the original Diffusion Policy**, including published work and preprints. The closest references to our current setup are **SGAC**, which studies adaptive execution of Diffusion Policy on PushT, and **DCDP**, which corrects actions within chunks on dynamic PushT. **React When You Need To** and **SV-VLA** directly study observation-driven early replanning. They show that using updated observations to interrupt a plan is not, by itself, a new contribution.

My recommendation is to **keep the verified frozen Diffusion Policy and PushT environment, but keep our interruption rule tentative**. Read SGAC and React When You Need To first, then consider whether our course contribution should emphasize better interruption decisions or less disruptive replanning. We do not need a publication-level gap; we do need a clearly defined modification, appropriate comparisons, and evidence that it helps or explains a meaningful trade-off.

No new controller, dependency installation, policy training, or simulator experiment was performed for this review. Public code access is not proof that its full setup runs on our laptop.

## Scope and evidence level

The search covered adaptive action chunking, execution-horizon selection, event-triggered inference, temporal action consistency, runtime failure monitoring, and recovery around pretrained robot policies. Sources are primary papers, proceedings, authors' project pages, and author-linked repositories. Key method sections and implementation/release information were inspected where available; this is not an exhaustive systematic review or reproduction of the papers.

Publication labels below distinguish proceedings-confirmed venues, author-reported acceptance, and preprints. Where no venue was verified, the paper is listed as a preprint rather than assuming acceptance or rejection. Release status is the status observed during this review and may change. Laptop-fit judgments are **our inference**, unless explicitly identified as measurements from our existing experiments.

The three decisions must be kept separate:

1. **When to ask AI again:** decide how many movements to execute before a fresh prediction.
2. **Whether to interrupt now:** use new observations to decide whether the pending movements remain appropriate.
3. **How to recover or generate a better next plan:** a new prediction can still fail or disrupt useful motion.

## 1. Choose execution length from the proposed plan

These methods normally choose a prefix when the AI returns a plan. They do not all monitor subsequent object movement inside that prefix.

| Method and status | Understandable explanation | Policy/training requirements and evidence | Fit with our laptop and frozen policy |
|---|---|---|---|
| **PACE**, May 2026 preprint [1] | Find a pronounced slowdown in the predicted movement sequence and ask for updated feedback at that transition. | No policy retraining or internal signals. Demonstrated with RoboTwin2.0 and real ALOHA/Franka robots; not our PushT checkpoint. The paper calibrates its valley threshold using training demonstrations. | A planned-target-distance adaptation is inexpensive, but our short queue may contain few meaningful valleys. No author implementation was verified in the inspected sources; any local version must be called PACE-inspired unless it faithfully follows the method. |
| **AAC**, CVPR 2026 [2] | Ask the policy for several possible plans. Use their disagreement to decide how long to trust the predictions. | Training-free selection based on action entropy; published VLA evaluations use multiple samples, including a default of 20. Public GR00T/LIBERO/RoboCasa repositories were verified through the [author code organization](https://github.com/orgs/Adaptive-Action-Chunking/repositories). | Sampling is possible without training our policy, but repeated predictions are expensive. Published small-overhead results on an A800 do not establish low overhead on a GTX 1650. |
| **DVAC**, June 2026 preprint [3] | While AI produces a plan, inspect whether its final refinement iterations agree about each future movement. Execute the stable prefix. | Uses clean-action estimates from one flow-matching inference and a rolling variance threshold. Evaluated on LIBERO, RoboTwin, CALVIN and real robots. No official code release was verified in the inspected paper. | Promising low-extra-arithmetic idea, but our model is stochastic **DDPM**, not the paper's flow policy. Extracting analogous estimates would be an adaptation, not a ready-made uncertainty measure. |
| **DEHP**, June 2026 preprint [4] | Train a small separate model to choose how many movements to execute from each AI plan. | Freezes Diffusion Policy, trains the horizon head with online PPO. Evaluations use state-based precision/assembly tasks; training includes 1,000 parallel environments. No downloadable code link was verified on the [project page](https://dehp-chunking.github.io/). | A trained head could be inexpensive at inference, but obtaining it requires many expensive policy rollouts. Not recommended as our first implementation. |
| **AutoHorizon**, ECCV 2026 [5] | Read the AI model's internal action-attention patterns to estimate how far its plan remains useful. | Training-free but requires action-token self-attention in flow-based VLAs. Evaluated on LIBERO, RoboTwin and real tasks. [Official code](https://github.com/hatchetProject/AutoHorizon) is public. | Not directly compatible: our CNN/U-Net Diffusion Policy lacks the relevant action-token attention. Switching to a large VLA solely for this method would expand the project substantially. |
| **ChunkTrust**, September 2026 preprint [6] | Combine how stable action generation looks, how well the new plan connects with recent motion, and memory of previously suitable execution lengths. | Training-free AHS selector; optional learned QHA prior. Evaluated on RoboTwin2.0, RoboCasa and real tasks. [Official selector code and integration documentation](https://github.com/hf618/ChunkTrust) are public. | Useful reference for continuity and adaptive memory, but its internal flow-generation evidence is not directly equivalent to our DDPM noise predictions. More complex than a first lightweight wrapper. |

## 2. Observe execution and decide whether to keep or replace the plan

This family is particularly relevant to our discussion about temporary alignment decreases. However, monitoring an event and proving that another prediction will recover are different achievements.

| Method and status | Understandable explanation | Evidence and implementation access | Relevance and limitation |
|---|---|---|---|
| **SGAC**, NeurIPS 2025 [7] | Compare the next old movement with a fresh AI prediction. Keep the old queue when they agree; replace it when they disagree. A separate self-guidance component changes sampling. | Diffusion Policy **and PushT** evaluations. [Official code](https://github.com/junhyukso/SGAC) includes separate adaptive-chunking and self-guidance evaluation scripts. No extra training required for these components. | **Closest timing/continuity reference.** The consistency check itself needs fresh predictions, so keeping an old queue does **not** mean fewer AI calls than our fixed-eight baseline. Its PushT results use coverage, not our binary success definition. |
| **React When You Need To**, September 2026 preprint [8] | Compare the current camera observation with the observation at the last prediction. Trigger early inference after the event score is high twice consecutively. | Trained event decoder on frozen VLA visual features; real Franka dynamic tasks using fine-tuned pi0.5. Authors report RTX 3090 inference. The [project page](https://react-when-you-need-to.github.io/) promises code upon acceptance; no release was verified. | Direct precedent for **observation change + confirmation + early replanning**. Its asynchronous robot continues moving while AI thinks; our current simulator pauses. A simpler local adaptation would not reproduce its trained estimator or deployment results. |
| **SV-VLA**, April 2026 preprint [9] | A small learned verifier uses fresh images to estimate a suitable next movement. If it disagrees with the pending movement, discard the suffix and ask the expensive planner again. | LIBERO/OpenVLA-OFT, not PushT. [Public training/evaluation code](https://github.com/edsad122/SV-VLA) lists roughly **16 GB VRAM** for LIBERO inference. | Strong conceptual reference, not directly runnable on our 4 GB GPU as released. The paper improves over fixed-64 execution but remains below fixed-8 success: 79.5% fixed-64, 90.9% verified-64, 96.0% fixed-8. Its benefit includes a speed trade-off, not universally better success. |
| **Sentinel**, CoRL 2024 / PMLR 2025 [10] | Check whether successive action distributions behave inconsistently, and separately use a vision-language model to check task progress. | Diffusion-policy failure detection including PushT and manipulation tasks. [Official code](https://github.com/agiachris/sentinel) is public. | Important distinction: a robot can be consistent yet make no useful progress. Detecting these failures does not establish that immediate resampling fixes them. Multiple action samples and VLM calls must be counted. |
| **FIPER**, NeurIPS 2025 [11] | Raise an alarm when observations remain unfamiliar **and** predicted actions remain uncertain, using thresholds calibrated on successful attempts. | Includes PushT; [public detector code](https://github.com/learnsyslab/fiper). Its published PushT setup uses 256 action samples per query and a 90% success threshold, unlike our 95% threshold. | Useful failure-monitor reference, but the full online entropy computation is a poor first fit for our budget. Evidence is for detection, not a complete successful recovery controller. |
| **PATCH**, June 2026 preprint [12] | Predict how relevant image regions should change under the current plan. Accumulate persistent, unexplained changes near the planned motion, while filtering robot self-motion. | Real ACT/Piper-arm intervention and policy resumption. Requires dense visual features and a trained latent dynamics model. The [author repository](https://github.com/YananZHOU5555/PATCH) says research code, data and checkpoints are coming soon. | Relevant to avoiding unnecessary interruptions, but it targets external disturbances, not simply decreasing alignment. Reproducing its full monitor needs additional models, training and geometry; laptop feasibility is unverified. |

## 3. Related alternatives: improve continuity, correct actions, or change recovery

These are not all answers to *when* to replan. They address why merely requesting a fresh plan can fail.

| Method and status | Understandable explanation | Evidence and access | Assessment for this project |
|---|---|---|---|
| **BID**, ICLR 2025 [13] | Generate several possible new plans and choose one that connects well with the previous plan while favoring plausible future actions. | Diffusion Policy and **PushT**, plus other simulated/real tasks. Author page links [Diffusion code](https://github.com/YuejiangLIU/bid_diffusion) and [LeRobot code](https://github.com/Jubayer-Hamid/bid_lerobot). | Strong continuity baseline/reference. Multiple samples and stronger/weaker-policy comparisons make full reproduction more costly than a simple scheduler. Do not treat arbitrary averaging of incompatible pushing modes as equivalent. |
| **Self-Guided Action Diffusion**, August 2025 preprint [14] | Guide a new prediction toward useful overlapping movements from the previous plan, instead of discarding that information completely. | Includes **PushT** and other diffusion-policy benchmarks. [Author-linked code](https://github.com/rhea-mal/self-guided-action-diffusion) is public, but setup/checkpoint portability was not verified. | **Worth reading as a related alternative:** it tackles disruptive plan switching with fewer samples. It changes inference sampling, not just timing. Published cost claims need measurement under our DDPM implementation. |
| **DCDP**, ICRA 2026 acceptance reported on arXiv; preprint reviewed [15] | Keep the expensive AI plan, but use a faster image-based model to correct each movement while it executes. | Frozen Diffusion Policy with **dynamic PushT**, including 16 predicted / 8 executed actions and >95% overlap success. [Code](https://github.com/wupengyuan/dcdp) is public; inspected checkpoint directory contained no usable auxiliary weights. | Closely matched alternative question: **cheap action correction versus full replanning**. Despite the title's "training-free" wording, the auxiliary dynamics encoder and action VAE are trained. Not yet a ready-to-run laptop replacement. |
| **RTC**, 2025 preprint version reviewed [16] | Generate the next plan while the robot follows the current one, preserving movements that will execute before the new plan is ready. | Flow/diffusion inference-time inpainting; Kinetix and real bimanual tasks. [Simulation code](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) is public. | Useful systems reference, not our first project change. Full published training is much larger than our resources, and our current paused simulator does not model motion during inference. Asynchrony requires an explicit timing model. |
| **Rewind-IL**, April 2026 preprint version reviewed [17] | Detect inconsistent plans, return toward a previously identified safe intermediate state, and restart the policy there. | ACT/flow policies, simulated and real long-horizon manipulation; no PushT evaluation verified. Uses offline VLM-selected checkpoints. The [project page](https://sjay05.github.io/rewind-il/) lists code as coming soon. | Raises a useful question: **where should recovery start**, not just when? For pushing, returning the pusher to an old position does not restore the displaced block. Simulator state restoration would be privileged experimental machinery, not a physical recovery method. |

## What this changes in our interpretation

**Our earlier distinction between PACE and our observation-based idea was correct but incomplete.** PACE analyzes planned motion. Other work already uses observations, repeated warnings, plan disagreement, and execution verification. Therefore, the following are not sufficient novelty claims: "use visual feedback," "replan adaptively," "wait for repeated warnings," or "keep the base policy frozen."

Nevertheless, a technically sound course contribution can adapt an established idea to our constrained setup, add a justified modification, and critically evaluate its costs and failures. The review does not prove that our exact future rule is already published, nor does it prove a gap. We have not specified that rule fully enough to make either claim.

Three lessons are especially relevant:

- **Temporary worsening is not a failure label.** A useful pushing or rotation sequence may initially reduce overlap. Plan continuity, action disagreement, and persistent task-relevant changes provide alternatives to an immediate overlap-drop alarm; none is guaranteed to be reliable in our setup.
- **A monitor can secretly cost as much as replanning.** Comparing an old movement with a freshly sampled one, or estimating uncertainty from many plans, requires policy computation even when the old queue is retained.
- **Timing and action quality are different problems.** If the frozen policy keeps producing ineffective actions at a difficult state, changing prediction timing alone may not solve it. Guidance, fast corrections or a different recovery state may be needed.

## What our existing evidence establishes

The [completed failure-pattern report](2026-10-11-robotics-failure-patterns.md) remains unchanged: the baseline has 20 successes and 10 failures across 30 attempts. All failures get close to the target without completing alignment. The selected-state pilot gives **5/12 successes for continuing and 5/12 for one-shot replanning**, with more remaining AI calls under replanning.

That pilot demonstrates possible recoveries and possible harmful interruptions. It is not a full monitor evaluation, does not explain the policy's intentions, and does not show that any reviewed method improves our checkpoint. We did not rerun completed experiments for this review.

## Laptop and implementation constraints

Current local inspection confirms two observation frames, a 16-slot diffusion output, eight exposed executable actions, and 100 DDPM iterations. The standard generation function retains slots 1 through 8; one of the original 16 slots is historical. A carefully written interface could expose up to 15 current/future slots, but longer-than-eight execution quality is unvalidated. See the [pinned implementation](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py) and [checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653).

- **Most plausible low-cost candidates:** a planned-movement timing rule or a small camera-based execution monitor, retaining the existing policy. This is an interface/complexity assessment, not measured new-method feasibility.
- **More costly candidates:** fresh-prediction agreement checks and multi-sample uncertainty. Our recorded mean is 4.84 seconds per new prediction. Four sequential predictions would be roughly 19 seconds, assuming unchanged per-call cost; this is an estimate, not a new benchmark. Batched memory/runtime are unverified.
- **Not ready-made:** attention-based VLA methods, learned horizon heads, world/latent dynamics monitors and auxiliary action correctors. Some require a different architecture; others require training or unreleased weights.
- **Visual inputs must stay honest:** our simulator coverage, contact and exact block pose are diagnostic ground truth, not automatic inputs to a camera-based controller. A lightweight visual alignment/motion estimate would need its own validation.
- **Queue handling matters:** changing `n_action_steps` alone is unsafe because the action deque retains its original capacity. Interruption should discard pending actions while preserving observation history; resetting the entire policy clears both. No runtime changes were made here.
- **Offline versus real time:** the simulator currently waits during AI inference. Good offline results cannot establish responsiveness on a continuously moving physical robot.

## Recommended choices for discussion

These are options, not newly selected project requirements.

| Possible focus | Course-sized question | Main references | Recommendation |
|---|---|---|---|
| **Lightweight execution-aware replanning** | Can a camera-based decision rule improve pushing success without interrupting useful adjustments or using excessive AI computation? | SGAC, React When You Need To, SV-VLA, PACE | Best fit with our current infrastructure. Persistence alone is established; the actual signal and rule need justification. |
| **Continuity-aware replanning** | When we request a new plan, can retaining useful information from the old plan reduce harmful switching? | SGAC, BID, Self-Guided Action Diffusion | Serious alternative if changing timing alone is insufficient. Requires an agreed sampling/selection change and fresh feasibility checks. |
| **Fast correction inside an AI plan** | Can a small feedback module fix movements more cheaply than requesting a whole new plan? | DCDP | Strong CPS/AI connection, but higher setup risk because trained auxiliary weights are not verified. Keep as a fallback, not an immediate pivot. |

For a fair later evaluation, start with reasonably tuned fixed intervals, then select only a small number of compatible references. Use unused evaluation seeds, include success cases as well as failure cases, and count **all** policy predictions and monitoring computation. Report success rate and final alignment with cost; do not infer a gain by comparing only with a poorly chosen long interval or selecting only rescued failures.

### Reading order

1. **SGAC, Section 3.2 and AC-only ablation:** understand keeping versus replacing a movement queue, and why its checks still cost AI calls.
2. **React When You Need To, Section IV:** understand observation-based triggering, confirmation, learned event scores and asynchronous assumptions.
3. **BID and Self-Guided Action Diffusion, method sections:** understand why a new prediction may disrupt a useful ongoing movement and how prior plans can inform the next one.
4. **PACE, Section 3:** retain a simple planned-motion comparator; distinguish an inexpensive adaptation from reproducing all the authors' settings.

After that discussion, choose one narrow method to investigate. Do not implement every method in this review or claim that literature alone proves our proposal will improve pushing.

## References

1. Nie et al. [PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking](https://arxiv.org/abs/2606.00537), preprint, May 30, 2026. Reviewed [v1 method](https://arxiv.org/html/2606.00537v1).
2. Liang et al. [Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](https://arxiv.org/abs/2604.04161), CVPR 2026. [Official proceedings paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Adaptive_Action_Chunking_at_Inference-time_for_Vision-Language-Action_Models_CVPR_2026_paper.pdf).
3. Feng et al. [Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies](https://arxiv.org/abs/2606.03847), preprint, June 2, 2026. Reviewed [v1 method](https://arxiv.org/html/2606.03847v1).
4. Zhao et al. [Dynamic Execution Horizon Prediction for Chunk-based Robot Policies](https://arxiv.org/abs/2606.11408), preprint, June 9, 2026. Reviewed [v1 method](https://arxiv.org/html/2606.11408v1) and [v2](https://arxiv.org/html/2606.11408v2).
5. Wang et al. [VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies](https://arxiv.org/abs/2602.21445), ECCV 2026 per [authors' project page](https://hatchetproject.github.io/autohorizon/).
6. Huang et al. [ChunkTrust: Adapting Execution Horizons for Robot Policies with Action-Expert Evidence](https://arxiv.org/abs/2609.39754), preprint, September 30, 2026. Reviewed [v1 method](https://arxiv.org/html/2609.39754v1).
7. So et al. [Improving Generative Behavior Cloning via Self-Guidance and Adaptive Chunking](https://proceedings.neurips.cc/paper_files/paper/2025/hash/79ce24f9e8d3c4ff5919240eac78a782-Abstract-Conference.html), NeurIPS 2025. Reviewed [adaptive-chunking method](https://arxiv.org/html/2510.12392v1).
8. Wu et al. [React When You Need To: Event-Triggered Asynchronous Inference for VLA Policies](https://arxiv.org/abs/2609.22587), preprint, September 18, 2026. Reviewed [v1 method](https://arxiv.org/html/2609.22587v1).
9. Wang et al. [Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA](https://arxiv.org/abs/2604.02965), preprint, April 3, 2026. Reviewed [v1 method](https://arxiv.org/html/2604.02965v1).
10. Agia et al. [Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress](https://proceedings.mlr.press/v270/agia25a.html), CoRL 2024 proceedings, PMLR 270, 2025.
11. Römer et al. [Failure Prediction at Runtime for Generative Robot Policies](https://arxiv.org/abs/2510.09459), NeurIPS 2025; venue also identified by the [official repository](https://github.com/learnsyslab/fiper). Reviewed [v1 method](https://arxiv.org/html/2510.09459v1).
12. Zhou et al. [PATCH: Action-Chunk-Conditioned Latent Patch Innovation Monitoring for Robot Manipulation](https://arxiv.org/abs/2606.16690), preprint, June 15, 2026. Reviewed [v1 method](https://arxiv.org/html/2606.16690v1).
13. Liu et al. [Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling](https://bid-robot.github.io/), ICLR 2025. Earlier arXiv title used "Closed-Loop Resampling"; [paper record](https://arxiv.org/abs/2408.17355).
14. Malhotra, Liu and Finn. [Self-Guided Action Diffusion](https://arxiv.org/abs/2508.12189), preprint version reviewed, August 17, 2025. [Method](https://arxiv.org/html/2508.12189v1), [author page](https://rhea-mal.github.io/selfgad.github.io/).
15. Wu et al. [Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy](https://arxiv.org/abs/2603.01953), first preprint March 2, 2026; arXiv comments report ICRA 2026 acceptance. Reviewed [v1 method](https://arxiv.org/html/2603.01953v1).
16. Black, Galliker and Levine. [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339), first preprint June 9, 2025; reviewed [v2 method and compute appendix](https://arxiv.org/html/2506.07339v2).
17. Zheng et al. [Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning](https://arxiv.org/abs/2604.16683), first preprint April 17, 2026; reviewed [v1 method](https://arxiv.org/html/2604.16683v1). Authors' later venue claims were not independently checked against proceedings.
18. Chi et al. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), RSS 2023; reused action policy, not our replanning contribution.
