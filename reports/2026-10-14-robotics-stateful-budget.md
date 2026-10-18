# Persistent execution length: eight, twelve or fifteen movements

Started: October 14, 2026. Completed: October 15, 2026 (UTC).

## Conclusion

**The persistent 8/12/15 controller did not establish an overall improvement over the milestone.** On twenty new scenes, both succeeded 17/20 times. The new controller had slightly higher mean final alignment, but used 18.93% more AI calls and took 18.58% longer. Keep the milestone unchanged; this candidate is not adopted.

In plain words: remembering and increasing the movement-unit length worked mechanically, but this rule rarely allowed long execution. It mostly behaved like eight-movement execution with extra camera monitoring. This particular implementation did not deliver the intended computation saving; that does not prove every persistent adaptive-length method would fail.

All 60 scored attempts, five mechanics checks, 482 CPU tests and the independent final audit are complete. No settings were changed after observing results, and no completed trial was replaced.

## Question

Can persistent, camera-guided execution lengths preserve or improve pushing success while reducing predictions and total computation relative to fixed-eight and the milestone?

## What changes, and what does not

Every condition uses the same pinned pretrained Diffusion Policy, two-frame observation history, H16 generated positions, 96×96 policy images, 100 DDPM denoising iterations and four CPU threads. This generation provides **15 usable future movements**. No retraining, changed model weights, resized policy inputs or paid API is introduced.

| Method | Execution rule |
|---|---|
| Fixed-eight baseline | Execute eight movements, then request a new prediction |
| Unchanged milestone | Retain twelve; after eight, use its existing camera/path check to continue or replan, with its original bounded regression interruption |
| Persistent 8/12/15 candidate | Start at eight; retain the chosen budget across predictions, increase after two reliable completed units, and return to eight when longer execution is not eligible |

The candidate changes **how much of a generated plan is executed**, not the distance of each movement or the number of positions generated. Adding persistent memory and the 15-movement option are our proposed modification; the policy, simulator, camera estimator, eligibility thresholds and regression guard are reused work.

## Predeclared controller

Let \(b_k\in\{8,12,15\}\) be the execution budget of prediction \(k\), and \(s_k\) the consecutive eligible-unit counter. Initially \(b_0=8\), \(s_0=0\).

1. Generate all 15 usable commands, record them, then retain only the first \(b_k\) for execution. Do not reset observation history between predictions. Generated-but-unscheduled commands never count as pending for the interruption guard.
2. Keep the milestone's original regression thresholds, confirmations, arming, cooldown and maximum two interruptions. A confirmed interruption discards the scheduled remainder and resets the next budget and counter to eight/zero.
3. For a twelve- or fifteen-movement unit, check after movement eight. Continue only if the original camera eligibility and command-smoothness criteria hold for the proposed suffix. Otherwise discard the remainder and reset to eight/zero.
4. At a completed nonterminal unit, increment the counter only if the endpoint is camera-eligible, the executed path is smooth, and estimated alignment is strictly greater than at the unit's start. Otherwise reset to eight/zero. After two consecutive qualifying units, increase one level (8→12→15, saturating at 15) and clear the counter.

Near-target or invalid observations therefore cause fallback at the eight-movement checkpoint or unit end, **not an immediate new interruption at every movement**. Between those boundaries, only the original confirmed-regression guard can interrupt. This retains the distinction between a scheduling decision and a validated failure signal.

The reused eligibility requires valid camera estimates, at least eight history movements, estimated coverage below 75%, zero current warning flags, drawdown at most 1.5 percentage points and recovery at least −1 percentage point. Smoothness reuses minimum command separation two units and adjacent direction cosine at least 0.5. Paths start at command index four, as in the milestone, and extend through the selected budget. These are eligibility heuristics, not safety guarantees or failure labels.

Two-unit persistence and positive net progress are proposed conventions, not optimized or established settings. They were not adjusted after seeing the evaluation. A positive visual change can be within estimator error; good past progress does not guarantee that the next longer plan is appropriate. Temporary decreases can be useful, and near-target failures may persist despite returning to eight. No rejected weighted/threshold interruption gate was reopened.

## Locked evaluation

- **Twenty new initial scenes: seeds 90–109**, not the previously inspected scenes. A preflight filename search found no existing recorded attempt for these seeds.
- One predetermined policy draw per scene, seed **30000 + scene**, used by all three methods: **60 scored attempts**. Noise is paired by prediction-call order, not physical movement time; different execution schedules need not produce identical later trajectories.
- Rotate method order by scene index. Run one GPU collector, keep every success and timeout, and do not select or repeat unfavorable trials.
- Keep the 300-movement limit and success criterion: true block coverage **strictly greater than 95%**. Dynamic simulator truth is scoring/trace validation only, never a controller input. Camera feedback uses RGB, static task geometry and scheduled pending-action count.
- Primary metrics: successful pushes and average final alignment. Secondary metrics: average AI calls and whole-attempt runtime including monitoring/decision cost. Loading/reset and compressed trace writing are excluded consistently. Also report actual selected budgets, realized unit lengths and intervention/transition counts; a nominal three-length controller may seldom use fifteen.
- Freeze source hashes, methods, checkpoint/configuration, installed generation code, packages, rules and trial order before GPU preflight or evaluation.
- Report paired candidate-minus-baseline and candidate-minus-milestone differences, gained/lost successes and 10,000 whole-scene-pair bootstrap intervals (seed 20261014). These are exploratory intervals without multiplicity correction. One sample per scene is not repeated-randomness reliability, proof of equivalence, real-camera generalization or deployment evidence.

## Mechanics gate before scoring

Run 16-movement continuations on scene zero, policy seed 1234, for all three real methods. Verify matching initial generated first-eight commands and physical/RGB prefixes. If an approved guard interrupts before eight, verify the shared prefix only up to the intentional divergence. Commands, physical coordinates and RGB hashes are exact; only existing diagnostic-coverage rounding permits `1e-12` tolerance.

Also run the candidate's adapter with feedback disabled and forced budgets eight and fifteen. Forced-eight must exactly reproduce the baseline across **16 movements and two predictions**, demonstrating preserved observation history. Forced-fifteen must execute fifteen cached movements before its second prediction. Verify raw generated `(1,16,2)` tensors, finite commands, queue counts and unchanged parameter-version guards. These mechanics controls are not scored candidate outcomes.

Existing matching trials are verified on resume, not replaced. Orphan traces, partial sidecars, modified protocols or changed results stop the runner for inspection. If the mechanics gate fails, stop before evaluation; do not change weights or silently weaken the checks.

## Results

### Completed mechanics gate

All five 16-movement mechanics records passed: real fixed-eight/milestone/candidate units were `[8,8]`, `[12,4]` and `[8,8]`. Forced-eight reproduced baseline commands, coordinates and RGB across all 16 movements and two predictions. Forced-fifteen used `[15,1]` units. All ten recorded neural generations remained `(1,16,2)`; commands were finite and parameter-version guards passed.

The independent CPU smoke audit verified 50 frozen input bindings, two installed model sources, configuration/checkpoint/packages, action queues, past-only rolling camera features and controller events without repeating inference, image fitting or physics. The complete smoke stage took 54.83 seconds, peak process RAM 2.56 GiB and minimum available system RAM 3.61 GiB. These are mechanics/resource checks, not pushing improvement or real-time deployment evidence.

### Scored comparison

These are **twenty new scenes, seeds 90–109, with one prediction sample per method per scene**: sixty attempts, not sixty independent scenes. Earlier milestone tables used different scenes and samples; their numbers should not replace these matched controls.

| Method | Successful pushes | Average final alignment | Average AI calls | Average runtime |
|---|---:|---:|---:|---:|
| Fixed-eight baseline | 14/20 | 94.27% | 24.60 | 121.92 s |
| Unchanged milestone, adaptive 8/12 | 17/20 | 94.30% | 20.60 | 110.81 s |
| New persistent 8/12/15 | 17/20 | 95.06% | 24.50 | 131.39 s |

Runtime includes camera monitoring and controller decisions. Monitoring averaged 9.20 seconds for the candidate and 8.68 seconds for the milestone. These are offline simulation costs: the simulator waits during prediction, so they do not measure a real robot's response latency.

Relative to the milestone, the candidate gained success in **scene 90** but lost it in **scene 96**. Relative to fixed-eight, it gained **91, 102, 104 and 108** but lost **109**. Thus the candidate did not preserve every baseline success, and its matching 17/20 total does not mean it solved the same scenes as the milestone.

### Uncertainty

The locked 10,000 scene-pair bootstrap gives the following candidate-minus-reference differences and exploratory 95% intervals. Alignment and success differences are in percentage points; calls and runtime are per attempt.

| Metric | New minus fixed-eight, estimate [interval] | New minus milestone, estimate [interval] |
|---|---:|---:|
| Success rate | +15 [−5, +35] pp | 0 [−15, +15] pp |
| Final alignment | +0.79 [−1.06, +2.77] pp | +0.76 [−1.06, +3.47] pp |
| AI calls | −0.10 [−5.00, +4.90] | +3.90 [−0.75, +8.15] |
| Runtime | +9.48 [−15.65, +35.23] s | +20.59 [−3.59, +42.38] s |

Every interval includes zero. The observed means do not establish a stable success/alignment gain, a reliable computation penalty, or equivalence between controllers. They nevertheless give no reason to replace the simpler milestone with this more expensive observed candidate. One draw per scene does not measure sensitivity to repeated prediction randomness.

### Did the execution length really change?

Yes. Across the candidate's 490 predictions, its selected budget changed 57 times between consecutive units. However, selecting twelve or fifteen did not guarantee executing that many movements: the existing checkpoint could still stop the unit at eight.

| Selected execution budget | Units selected | Units executing the full selected budget | Units stopped at eight by the checkpoint |
|---|---:|---:|---:|
| 8 | 447 | 425 | Not applicable |
| 12 | 36 | 19 | 17 |
| 15 | 7 | 2 | 5 |

Another 22 eight-budget units ended before eight through an interruption or episode ending. Overall, only **21/490 units executed more than eight movements**, and only **2/490 executed fifteen**. The candidate had 22 checkpoint-triggered replans and ten original-guard interruptions. These are controller events, not verified physical failures.

The evidence supports a narrow explanation: this candidate made little use of its long-execution options. A plausible reason is that two consecutive eligible units with positive net alignment is restrictive: useful approach/repositioning can have flat or decreasing alignment. This is a design limitation and hypothesis, **not a causal diagnosis from an ablation**. The saved data do not establish which condition should be relaxed, or whether doing so would improve success.

### Verification and laptop resources

- All **482 CPU tests passed**, including 55 new tests for budget selection, queue handling, controller transitions, resumability and result checking.
- The [independent audit](../results/robotics-stateful-budget/independent-audit.json) passed for all 60 scored and five mechanics records: 50 frozen input bindings, two installed model sources, checkpoint/configuration/packages, 136 artifact fingerprints, 11,530 scored movements and 1,394 predictions. Queue arithmetic, rolling camera histories, original guard/persistent transitions, common prefixes, summary means and paired statistics reconciled.
- All three methods shared the exact first eight generated/executed commands and physical/RGB prefix in all twenty scenes. This verifies common initial conditions, not identical later stochastic trajectories.
- The scored collector took **121.62 minutes**, with peak process RAM **2.55 GiB**, minimum available system RAM **2.82 GiB**, and maximum recorded PyTorch GPU allocation **1.07 GiB**. Allocation is not total device usage. No memory failure or CPU fallback occurred.

The independent audit did not rerun inference, physics, image fitting or timing. Raw neural tensor shapes, finite-generation and unchanged-parameter checks remain source-bound collector evidence, not independently repeated captures. The simulated camera and frozen policy do not establish real-camera robustness, deployment feasibility or general improvement beyond these scenes.

## Decision

Preserve milestone `0f40449` and its controller unchanged. Archive this completed candidate and its evidence rather than adopting it or tuning it on scenes 90–109 and calling those scenes fresh again. Any revised promotion/fallback rule requires discussion before another experiment. The rejected weighted/threshold interruption filters remain stopped.

## Reproduction

Use the established `.venv-robotics` and pinned model download. Keep all earlier frozen experiments untouched.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_stateful_budget.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_stateful_budget.py --stage smoke
.\.venv-robotics\Scripts\python.exe experiments\robotics_stateful_budget.py --stage run
```

The output directory is `results/robotics-stateful-budget/`. For a disposable repeat, add `--output-dir results\robotics-stateful-budget-repeat` at every stage. The runner does not overwrite prior completed trials or edit the downloaded configuration. CPU tests use the existing `unittest` suite with a project-local temporary directory.

## Sources

- [Diffusion Policy](https://arxiv.org/abs/2303.04137), Chi et al., and the [pinned public checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653): reused neural policy and original generation/observation settings.
- [Gym-PushT](https://github.com/huggingface/gym-pusht): reused physical simulation and coverage/termination definition.
- Installed LeRobot `modeling_diffusion.py`: generation slicing, cached action queues and observation-history semantics; source hashes are frozen in the protocol.
- [Milestone development](2026-10-12-robotics-adaptive-length.md), [milestone evaluation](2026-10-12-robotics-fresh-evaluation.md) and [second-cohort comparison](2026-10-13-robotics-cohort-completion.md): original eligibility/guard and its inconsistent quality/efficiency findings.
- [Generation-length screen](2026-10-14-robotics-generation-length.md): distinguishes generation from execution, confirms H16 future-command capacity and warns that frequent replanning does not automatically help.
- [New runner](../experiments/robotics_stateful_budget.py): this candidate and locked comparison; [reused camera monitor](../experiments/robotics_visual_monitor.py) and [unchanged milestone controller](../experiments/robotics_adaptive_length.py).
