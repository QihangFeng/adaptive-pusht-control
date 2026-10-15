# Staged execution: eight, twelve or fifteen movements

Started and completed: October 15, 2026 (UTC).

## Conclusion

**Keep the unchanged milestone 8/12. The new staged 8/12/15 controller did not establish a useful upgrade.** On the forty fresh scenes, it succeeded 26/40 times versus the milestone's 30/40, with 89.91% versus 93.70% mean final alignment. Calls and runtime were almost identical: 21.775 versus 22.025 calls and 119.27 versus 119.54 seconds. The small observed cost saving does not justify the lower observed quality.

In plain words: the robot really did execute the extra three movements, but doing so did not improve its overall pushing results. This is not the earlier persistent controller, which rarely used fifteen. This staged version completed **326 fifteen-movement units**. Allowing more cached movement is therefore not enough by itself; the current continuation check did not establish when the extra movement is beneficial.

All **180 compared attempts on sixty scenes** are complete: 140 new scored runs and forty reused controls. All 529 CPU tests, five mechanics records and the independent final audit passed. No settings changed after collection began; no trial was excluded or selectively repeated. The designated milestone `0f40449` and earlier frozen evidence remain unchanged. This candidate is not adopted.

## Question

Question: can executing the last three already-generated movements improve pushing performance or reduce prediction/computation cost without sacrificing quality?

## Method in plain words

Every new prediction generates fifteen usable future movements. Keep twelve scheduled initially and reserve movements thirteen–fifteen separately. After eight, use the exact milestone camera/path check to replan or continue through twelve. After twelve, inspect current feedback and the remaining path: either request a new prediction or append the three cached movements. Appending makes no new AI call and does not reset observation history.

There is no four-movement checkpoint, learned rule, changed generation horizon or remembered execution budget. The milestone's existing bounded interruption guard remains active, with its original state and scheduled-pending counts through twelve. Dynamic simulator state is used for scoring/trace validation, never for runtime control.

Let \(\hat c_t\) denote camera-estimated target coverage, \(w_t\) warning count, \(d_t\) recent drawdown and \(r_t\) recent recovery. The reused camera eligibility is

\[
E_t=\mathrm{valid}_t\land(h_t\ge8)\land(\hat c_t<0.75)
\land(w_t=0)\land(d_t\le0.015)\land(r_t\ge-0.01).
\]

After movement twelve, extend only if \(E_t\) and the original smoothness test hold on commands thirteen–fifteen together with the last four executed commands. With zero-based indexing this is `commands[8:15]`. The original smoothness thresholds are command separation at least two units and adjacent direction cosine at least 0.5. These are heuristics, not safety guarantees or failure labels. The eight-step check continues to assess only the original twelve-command path, not the newly available tail.

## Locked comparison

| Method | Execution |
|---|---|
| Original fixed-eight | Request a new prediction after eight |
| Unchanged milestone 8/12 | After eight, replan or continue through twelve |
| Staged 8/12/15 | Same eight-step check; after twelve, replan or execute the last three |

- **Sixty scenes per method, 180 compared attempts, 140 new scored runs.**
- Inspected development scenes **90–109**, prediction seed `30000 + scene`: reuse their twenty fixed-eight and twenty milestone attempts by reference/hash, and collect twenty new staged attempts. Do not rerun or relabel old controls.
- **Primary fresh panel: forty new scenes 110–149**, prediction seed `40000 + scene`, all three methods: 120 new attempts. Rotate method order by scene index. These scenes must not be used to tune settings after inspection. The pooled sixty-scene comparison is secondary.
- Same checkpoint/weights, H16 generation, two observations, 96×96 images, 100 DDPM denoising iterations, four CPU threads, one GPU collector and 300-movement cap. Success requires simulator coverage **strictly greater than 95%**. No failures are excluded or selectively repeated.
- Primary quality outcomes: successful pushes and average final alignment. Secondary costs: average AI calls and whole-attempt runtime including monitoring/decision overhead. Loading/reset and compressed trace writing are excluded consistently. Fresh-cohort rotated-order timing is the primary runtime comparison; reused earlier-session controls make development/pooled runtime descriptive.
- Pair noise by prediction-call order, not by physical movement time. One policy draw per scene is not repeated-randomness reliability.
- Freeze rules, cohorts, order, checkpoint/configuration, packages, installed generation sources and reused sidecar/trace hashes before GPU use. No tuning between cohorts. Stop on modified bindings, orphan traces or partial results; resume by verifying completed trials, not replacing them.
- Report candidate-minus-fixed-eight and candidate-minus-milestone differences, gained/lost successes, exact two-sided discordant-success tests and 10,000 scene-pair bootstrap intervals (seed 20261015). Intervals are exploratory without multiplicity correction; no equivalence or universal improvement claim follows from nonsignificance.

## Mechanics gate

Before scoring, run five 32-movement controls on scene zero, prediction seed 1234:

1. All three actual methods.
2. Staged adapter with late extension disabled: must reproduce the unchanged milestone's actions, physical/RGB states, camera features and original events across multiple predictions.
3. Staged adapter with late extension forced: must execute all first fifteen cached commands, append only after twelve, and request its next prediction at movement sixteen.

The overrides are mechanics-only, never scored. Raw generation must remain `(1,16,2)`, output finite, and parameter-version guards unchanged. Only existing diagnostic-coverage rounding permits `1e-12` tolerance; commands, physical coordinates, RGB hashes and camera decisions remain exact. Normal candidate trajectories must match their paired milestone until the first approved late extension; with none, the whole attempt must match.

## Results

### Primary comparison: forty fresh scenes

Seeds 110–149 were unseen when the rule was frozen. All three methods were collected in the same invocation, with rotated method order and paired prediction seeds. These are forty independent initial scenes, not 120 independent scenes.

| Method | Successful pushes | Average final alignment | Average AI calls | Average runtime |
|---|---:|---:|---:|---:|
| Original fixed-eight | 26/40 | 92.08% | 27.050 | 133.61 s |
| Unchanged milestone 8/12 | 30/40 | 93.70% | 22.025 | 119.54 s |
| New staged 8/12/15 | 26/40 | 89.91% | 21.775 | 119.27 s |

Relative to the milestone, the new version used only **0.25 fewer calls (1.14%)** and **0.27 fewer seconds (0.22%)** per attempt, while its mean alignment was 3.79 percentage points lower and it completed four fewer pushes. These are observed differences, not proof of general inferiority.

Relative to fixed-eight, staged execution saved **5.275 calls (19.50%)** per attempt, with the same total success count and lower observed alignment. Thus it does offer evidence of fewer predictions against fixed-eight, but the existing milestone already delivers nearly the same prediction cost with better observed quality. Runtime includes camera/decision overhead; the simulator waits during inference, so it is not a real robot's response latency.

### Paired uncertainty and changed successes

The predeclared 10,000 scene-pair bootstrap gives the following candidate-minus-reference differences and exploratory 95% intervals. Success and alignment differences are percentage points (pp); all costs are per attempt.

| Metric | New minus fixed-eight, estimate [interval] | New minus milestone, estimate [interval] |
|---|---:|---:|
| Success rate | 0 [−20, +20] pp | −10 [−30, +10] pp |
| Final alignment | −2.17 [−7.60, +2.37] pp | −3.79 [−8.78, +0.33] pp |
| AI calls | −5.275 [−9.100, −1.500] | −0.250 [−3.775, +3.300] |
| Runtime | −14.34 [−33.81, +5.12] s | −0.27 [−18.87, +18.44] s |

Only the call-saving interval against fixed-eight excludes zero. All comparisons against the milestone include zero; the exact two-sided discordant-success test is **p = 0.4545**. This does not prove equal performance or universal harm. It gives no established improvement supporting adoption of this candidate. The exact test against fixed-eight is p = 1.0.

On fresh scenes, staged execution gained milestone failures **110, 111, 116, 118, 124 and 134**, but lost milestone successes **114, 120, 123, 128, 130, 133, 140, 141, 143 and 148**: six gains, ten losses. Against fixed-eight it gained nine successes and lost nine. These labels describe the recorded paired sample, not a guarantee that a scene always succeeds or fails under other prediction draws.

### Inspected development scenes and pooled results

The twenty development scenes were already inspected before this method was proposed. Their fixed-eight/milestone controls were reused by hash-bound reference, not rerun. The pooled sixty-scene view is secondary; its mixed-session runtime is descriptive rather than a clean speed comparison.

| Panel | Method | Successful pushes | Average alignment | Average AI calls | Average runtime |
|---|---|---:|---:|---:|---:|
| Inspected twenty | Fixed-eight | 14/20 | 94.27% | 24.600 | 121.92 s |
| Inspected twenty | Milestone 8/12 | 17/20 | 94.30% | 20.600 | 110.81 s |
| Inspected twenty | Staged 8/12/15 | 14/20 | 92.16% | 21.300 | 115.03 s |
| Pooled sixty | Fixed-eight | 40/60 | 92.81% | 26.233 | 129.71 s |
| Pooled sixty | Milestone 8/12 | 47/60 | 93.90% | 21.550 | 116.63 s |
| Pooled sixty | Staged 8/12/15 | 40/60 | 90.66% | 21.617 | 117.86 s |

Pooled, staged execution gained seven milestone failures but lost fourteen successes. Its mean alignment difference is −3.24 pp, with exploratory interval [−6.75, −0.39] pp; this secondary result includes inspected development data and must not replace the primary fresh-scene analysis. Full per-cohort statistics and scene identities are saved in [summary.json](../results/robotics-staged-extension/summary.json).

### Were the extra movements actually executed?

Yes. All sixty candidate attempts made at least one approved twelve-to-fifteen extension. Candidate trajectories matched their paired milestone exactly until that first intentional extension: movement twelve in 57 scenes, twenty in scenes 94/120, and forty in scene 128.

| Candidate execution | Fresh forty scenes | All sixty scenes |
|---|---:|---:|
| Total prediction units | 871 | 1,297 |
| Full fifteen-movement units | 232 | 326 |
| Full twelve-movement units | 33 | 53 |
| Eight-movement units | 550 | 835 |
| Other lengths from interruption/episode ending | 56 | 83 |

Every one of the 326 approved late extensions completed all fifteen movements. The candidate also recorded 53 twelve-step replan decisions and 35 original-guard interruptions. These controller events are not physical failure labels. Unlike the persistent-budget experiment, lack of actual fifteen-step execution does not explain this result.

A plausible limitation is that a smooth predicted path and favorable recent feedback do not establish whether the last three commands will improve physical alignment. Extra execution changes the state used for the next prediction, so its later consequences can be beneficial or harmful. The paired results establish the net outcome for this panel, **not the exact causal failure pattern**. No new matched-state probes, threshold changes or recovery method were added after viewing the results.

### Mechanics and independent verification

The five mechanics records passed. Actual fixed-eight, milestone and staged unit lengths were `[8,8,8,8]`, `[12,12,8]` and `[15,15,2]`. Disabling late extension exactly reproduced the milestone across all 32 movements and original events. Forcing late extension used `[15,15,2]`; the original fifteen commands executed without a new prediction, with the next prediction at movement sixteen. The normal candidate first diverged only after its approved checkpoint at movement twelve. Mechanics took 90.61 seconds and peak process RAM 2.58 GiB.

All **529 CPU tests passed**, including 47 new staged-controller tests. The [independent final audit](../results/robotics-staged-extension/independent-audit.json) verified all 131 frozen input bindings, two installed model sources, checkpoint/configuration/packages, forty reused controls, 180 scored comparisons and five mechanics records. It reconciled **36,848 scored movements, 4,164 predictions and 377 artifact hashes**, including cached queues, rolling camera histories, original/staged events, all sixty milestone-equivalent prefixes and all cohort statistics.

The scored collector took **4.79 hours (287.31 minutes)**, with peak process RAM **2.51 GiB**, minimum available system RAM **3.08 GiB**, and maximum recorded PyTorch GPU allocation **1.07 GiB**. Allocation is not total device usage. No memory failure or CPU fallback occurred. The audit did not repeat inference, image fitting, physics or timing; raw neural shapes/finite outputs and parameter-version guards remain source-bound collection evidence, not independently repeated tensor captures.

One policy sample per scene, a simulated camera and one frozen PushT policy do not establish repeated-randomness reliability, real-camera robustness, deployment feasibility or universal superiority/inferiority. The user-provided nine-hour window was not used to tune, extend sampling after seeing results or discard unfavorable trials.

## Decision

Keep the milestone 8/12 unchanged. Preserve this completed staged candidate and its negative comparison rather than adopting it or changing its rules on the same scenes. Scenes 110–149 are now inspected evaluation data for any future revision. No four-step controller, persistent-budget variant, rejected threshold/weighted gate or additional experiment was introduced. Any next methodological change should be discussed first.

## Reproduction

Run from the project root using `.venv-robotics` and the pinned Git-ignored model download. Do not edit earlier frozen experiments.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_staged_extension.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_staged_extension.py --stage smoke
.\.venv-robotics\Scripts\python.exe experiments\robotics_staged_extension.py --stage run
```

Use `--output-dir results\robotics-staged-extension-repeat` at every stage for a disposable repeat. Existing matching authoritative trials are verified rather than overwritten.

## Sources and attribution

- [Diffusion Policy](https://arxiv.org/abs/2303.04137), [pinned public checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653), [Gym-PushT](https://github.com/huggingface/gym-pusht): reused pretrained policy, simulator and original task.
- [Milestone development](2026-10-12-robotics-adaptive-length.md) and [evaluation](2026-10-12-robotics-fresh-evaluation.md): reused camera eligibility, smoothness and original interruption guard.
- [Persistent-budget comparison](2026-10-14-robotics-stateful-budget.md): reused matched controls for inspected scenes 90–109; its candidate is not included here.
- [New runner](../experiments/robotics_staged_extension.py), [unchanged milestone](../experiments/robotics_adaptive_length.py), [camera estimator](../experiments/robotics_visual_monitor.py) and installed LeRobot generation code (protocol-hashed): implementation and source attribution. Our modification is the staged twelve-to-fifteen execution gate, not adaptive replanning in general.
