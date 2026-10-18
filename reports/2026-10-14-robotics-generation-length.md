# Does shorter neural prediction improve pushing efficiency?

Date: October 14, 2026

## Conclusion

**Shorter generation works technically, but this tested setup does not provide a useful improvement. Do not adopt it in place of the milestone.** All 64 attempts completed. Short H8/E4 and matched H16/E4 each succeeded 6/16 times, versus the original baseline's 11/16 and milestone's 10/16. Short generation did not clearly reduce the cost of an AI prediction, while asking again every four movements made whole attempts much more expensive.

This conclusion concerns an inference-only H8 override on the existing H16 checkpoint, with four-action execution and 100 denoising iterations. It does not prove that all adaptive generation, retrained short-horizon policies or phase-specific schedules fail. No new controller is adopted; milestone `0f40449` and all earlier evidence remain unchanged.

## Question and scope

This is an approved, bounded development screen of **generated prediction length**, not another interruption filter or final-positioning corrector. Both tested horizons are fixed within each attempt; a dynamically varying generation-length controller has not been implemented.

Can the existing pretrained Diffusion Policy generate a shorter trajectory without retraining, and does that reduce computation while preserving pushing success and alignment?

The original model generates 16 positions. Because the observation history has two frames, future movements start at generated index one: a generated length of eight supplies **seven usable future movements**, not eight. The installed implementation validates generated lengths as multiples of eight. Its temporal network is convolutional, and sampling allocates a tensor using the configured length, so changing that setting changes actual generation rather than merely cropping the output. Mechanical compatibility does not establish prediction quality: the checkpoint's original horizon is 16, and shorter generation is a zero-shot inference change.

## Predeclared comparison

| Condition | Generated positions | Movements retained for execution | Purpose |
|---|---:|---:|---|
| Short prediction (`h8_e4`) | 8 | 4 | Candidate |
| Matched control (`h16_e4`) | 16 | 4 | Isolate generated-length effect |
| Original baseline (`fixed8`) | 16 | 8 | Project reference |
| Unchanged milestone (`adaptive12`) | 16 | Up to 12, with the original eight-action check and bounded interruption | Project reference |

Only short versus matched control isolates generated length. Comparisons with the original baseline and milestone also change feedback frequency; they measure overall behavior, not generation length alone.

- Eight already-inspected development scenes, seeds **0–7**. These are not untouched evaluation data.
- Two new prediction samples per scene: diffusion seeds **10000 + scene** and **20000 + scene**. All four conditions use both samples, giving **64 scored attempts**. Previous completed episodes are preserved, not rerun to replace their results.
- Keep the pinned checkpoint, two-frame observation history, 96×96 policy input, 100 DDPM denoising iterations, four CPU threads and 300-movement limit unchanged. No training, optimizer, additional training dataset or paid API is used.
- Success remains simulator coverage **strictly greater than 95%**. True block pose/coverage is used for scoring and trace validation, never as a runtime decision input. The unchanged milestone receives the existing camera features only.
- Rotate condition order within scene/sample blocks. Run one GPU collector. Reset model observation/action queues at each attempt, not at each prediction; retain history during execution.
- Freeze runner, reused sources, installed generation/configuration code, conditions and schedule before smoke or scored inference. Do not tune or replace unfavorable attempts after seeing results.

Matching prediction seeds does **not** make trajectories identical across different generated lengths: tensor shapes and random-number consumption change. Scene matching and repeated samples address comparison uncertainty without claiming identical AI plans.

## Mechanics and timing gate

Before scoring the 64 attempts, run four eight-movement smoke continuations on scene 0 with prediction seed 1234. Capture the raw sampler's actual tensor shapes and retained queue lengths; require finite commands and unchanged parameter versions. The H16 arms must have exact first-four commands, physical coordinates and RGB hashes, and baseline/milestone first-eight prefixes must match the already-saved original smoke. The existing state comparison permits only a diagnostic-coverage rounding tolerance of `1e-12`; commands, coordinates and RGB remain exact. The short H8 prediction is not required to match H16 commands.

On the same reset camera observation, perform **two warmups and three measured predictions per condition**, rotating order and synchronizing CUDA. Record prediction time and peak allocated GPU memory. These are mechanics/resource probes, not additional scored pushing attempts. They do not establish improvement.

If mechanics fails, stop before the scored comparison, preserve diagnostic evidence and explain the failure. Do not alter network weights or architecture merely to force a passing result.

## Metrics and interpretation

Primary outcomes: successful pushes and average final alignment. Secondary trade-offs: average AI calls and full-attempt runtime, including the milestone's monitoring/decision cost. Loading the checkpoint, creating/resetting the environment and writing compressed traces are excluded from episode time for all conditions.

Also report CUDA-synchronized prediction time, counting only fresh neural predictions, and per-attempt peak allocated GPU memory. The episode mean prediction time is averaged equally across attempts; a count of calls alone is no longer an adequate computation comparison when each call can have different cost. Initial-observation warmed timings are separate from the differing observations encountered during pushing.

Report both prediction samples separately as well as the combined table. For paired comparisons, average the two sample differences within each scene first; bootstrap **eight whole scene clusters**, not 16 supposedly independent scenes (10,000 resamples, seed 20261014). These are small-sample exploratory intervals without multiplicity correction. No holdout-generalization, real-time deployment, variable-horizon-controller or algorithmic-novelty claim is justified by this screen.

Trace/sidecar pairs retain initial state, actual commands, recorded raw-generation shapes, queue boundaries, cached plans, timing and unchanged milestone decisions. Completed sidecars/traces must verify before resume; orphan or altered files stop the collector rather than being overwritten.

## Results

### Completed mechanics and resource probe

The actual raw sampler produced `(1,8,2)` for short prediction and `(1,16,2)` for all reference conditions. Retained queues, finite actions and parameter-version guards passed. H16 first-four movements were exact, and baseline/milestone first-eight movements matched the original saved smoke. The independent smoke-only audit also passed without rerunning GPU inference or physics.

| Condition | Median warmed prediction time, three calls | Maximum allocated GPU memory |
|---|---:|---:|
| Short H8/E4 | 4.872 s | 1095.2 MiB |
| Matched H16/E4 | 4.834 s | 1095.3 MiB |
| Baseline H16/E8 | 4.789 s | 1095.3 MiB |
| Milestone H16/E12, prediction only | 4.874 s | 1095.3 MiB |

The shorter output ran successfully, but this small warmed probe showed **no clear per-call speed or allocated-memory saving**. It does not establish statistical equivalence or tell us whether pushing quality improves. Full-attempt comparisons remain necessary; we did not change the protocol after these timings.

The complete smoke stage took 132.59 seconds with peak process RAM 2.53 GiB and minimum observed available system RAM 3.70 GiB. GPU peaks use PyTorch allocated-memory counters, not total device memory used by other applications. This verifies laptop feasibility for this inference override, not real-time deployment or training.

### Scored pushing comparison

Each method has **16 attempts: eight previously inspected scenes, with two new prediction samples per scene**. These are not 16 independent or untouched scenes. All averages include failures; success requires true coverage strictly greater than 95% before the unchanged 300-movement limit.

| Method | Successful pushes | Average final alignment | Average AI calls | Average attempt runtime |
|---|---:|---:|---:|---:|
| Short generation: H8/E4 | 6/16 | 87.63% | 61.06 | 301.72 s |
| Matched control: H16/E4 | 6/16 | 87.38% | 62.50 | 310.57 s |
| Original baseline: H16/E8 | 11/16 | 93.57% | 27.50 | 138.29 s |
| Unchanged milestone: H16/adaptive 8–12 | 10/16 | 89.56% | 23.94 | 130.11 s |

The short setup took **2.18 times the baseline's average runtime**. Both four-movement methods made substantially more predictions than the references. This supports rejecting these constant-four-movement setups as an improvement in this screen, not claiming that generation length alone caused their worse performance: feedback frequency also differs from the baseline and milestone.

Results vary with the prediction sample:

| Method | Sample `10000 + scene` | Sample `20000 + scene` | Combined |
|---|---:|---:|---:|
| Short H8/E4 | 2/8 | 4/8 | 6/16 |
| Matched H16/E4 | 4/8 | 2/8 | 6/16 |
| Original baseline | 4/8 | 7/8 | 11/16 |
| Unchanged milestone | 6/8 | 4/8 | 10/16 |

These new outcomes do not replace the earlier milestone table or establish a new general success rate. Starting scenes and prediction draws differ from its 20-scene evaluation.

### Does changing generation alone help?

Compare **H8/E4 against H16/E4**, which asks again at the same four-movement interval. Short generation gained four successes and lost four successes, leaving the total unchanged. Its mean alignment was only 0.25 percentage points higher; calls were 1.44 lower and runtime 8.85 seconds shorter. All exploratory scene-cluster intervals include zero:

| Short minus matched control | Observed mean difference | 95% scene-cluster bootstrap interval |
|---|---:|---:|
| Success rate | 0.00 percentage points | −25.00 to +25.00 |
| Final alignment | +0.25 percentage points | −12.41 to +12.09 |
| AI calls | −1.44 | −9.88 to +7.25 |
| Runtime | −8.85 s | −50.62 to +34.05 s |

This is **no demonstrated benefit**, not proof of equivalence. With only eight scene clusters and no multiplicity correction, the intervals are exploratory. Identical seed labels do not imply identical H8/H16 sampled plans.

During full attempts, average fresh-prediction times were **4.917 / 4.942 / 4.970 / 4.961 seconds** for short / matched / baseline / milestone, respectively. H8 and H16 mean peak allocated GPU memory were **1095.3 and 1095.4 MiB**. These observations, together with the initial-observation warmed probe, show no clear practically useful per-call or memory saving. They do not establish equivalence or explain the underlying network timing causally.

The full collector took **235.08 minutes (3.92 hours)**, with peak process RAM **2.51 GiB** and minimum observed available system RAM **2.72 GiB**. Every method completed on the GTX 1650 without a recorded memory failure. The simulator waits during neural inference, so these are offline computation costs, not demonstrated real-time robot control.

### Verification and decision

The independent CPU audit passed for all 64 attempts, **14,821 movements, 2,800 predictions** and 367 unchanged milestone decisions (124 extensions, 231 trims, 12 interruptions). It checked queues and retained actions, rolling camera-feature histories, shared initial/H16 prefixes, all four smoke continuations, 12 measured smoke calls, summary arithmetic and scene-cluster statistics, 49 frozen input bindings, two installed model-source hashes, checkpoint/configuration/packages and 142 artifact fingerprints.

The audit did **not** repeat neural inference, physics, image fitting or timing. Recorded raw tensor shapes, finite-output checks and parameter-version guards are hash-bound assertions from the collector, not independently recaptured tensors. GPU counters do not include other applications' allocations; balanced ordering reduces but does not eliminate laptop load or thermal effects.

All **427 CPU tests passed**, including 43 new generation-screen tests. These use model/environment doubles to check configuration, history/queues, shape capture, unchanged-controller decisions, result integrity, resumability and statistics; passing them is not evidence of pushing improvement.

**Recommendation:** retain the unchanged milestone as our project reference; do not build the next controller around an assumed saving from shorter generation. Changing how many cached movements we execute remains a separate possible design question, requiring discussion before implementation. No threshold/weighted filter, camera corrector, training or variable-length controller was introduced here.

Authoritative evidence: [summary](../results/robotics-generation-length/summary.json), [frozen protocol](../results/robotics-generation-length/protocol.json), [all attempts](../results/robotics-generation-length/episodes.json), [independent audit](../results/robotics-generation-length/independent-audit.json) and [resource measurements](../results/robotics-generation-length/run-resources.json).

## Reproduction

Run from the repository root with the existing `.venv-robotics`. Keep prior frozen experiments untouched.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_generation_length.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_generation_length.py --stage smoke
.\.venv-robotics\Scripts\python.exe experiments\robotics_generation_length.py --stage run
```

The default results directory is `results/robotics-generation-length/`. For a disposable repeat, add `--output-dir results\robotics-generation-length-repeat` to every stage. Matching completed trials are verified, not replaced. The original model configuration/download remains unmodified; overrides exist only in the new collector's in-memory policy.

The CPU suite uses Python's built-in `unittest`, not `pytest`. Use a project-local temporary directory to avoid Windows sandbox temporary-folder replacement errors:

```powershell
.\.venv-robotics\Scripts\python.exe -c "import pathlib,tempfile,unittest; p=pathlib.Path('tmp/robotics-adaptive-length-tests').resolve(); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

## Sources

- [Diffusion Policy](https://arxiv.org/abs/2303.04137), Chi et al.: reused neural action policy, not the project's own trained model.
- [Pinned public checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653): original H16/two-observation configuration and weights; model identity is retained in the frozen protocol.
- Installed LeRobot `modeling_diffusion.py`, `conditional_sample`, `generate_actions` and `select_action`; `configuration_diffusion.py` horizon validation: primary implementation evidence for actual generation shapes, history-aligned slicing and supported lengths. Their hashes and installed package versions are frozen in the protocol.
- [Milestone development method](2026-10-12-robotics-adaptive-length.md), [designated milestone evaluation](2026-10-12-robotics-fresh-evaluation.md), and [completed second-cohort comparison](2026-10-13-robotics-cohort-completion.md): preserved controller/reference evidence and uncertainty.
- [New runner](../experiments/robotics_generation_length.py): scoped implementation of this screen; its H8 inference override is the new experimental change. No previous controller is edited or adopted anew.
