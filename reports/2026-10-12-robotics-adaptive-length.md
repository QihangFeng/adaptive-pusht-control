# Can longer cached movement plans reduce AI computation?

Date: October 12, 2026

## Conclusion first

**Adaptive action reuse is a more promising efficiency direction than our previous one-shot interruption pilot, but it is not an improvement in every metric.** In this eight-scene development screen, it finished 6/8 pushes versus the baseline's 3/8, used 37.55% fewer AI predictions and ran 31.65% faster including monitoring. However, average final alignment fell from 93.43% to 92.14%, mainly because it turned one baseline success into a much worse failure.

Simply executing twelve actions also helped substantially: 5/8 successes and the lowest computation cost, but only 82.05% average final alignment. Our adaptive rule produced one additional success and better average alignment than fixed-twelve, at higher computation cost. Therefore, longer action reuse is worth investigating; the camera rule's general benefit is not established.

My recommendation is to keep this modest rule unchanged for the next fresh-scene comparison, not add another learned model yet. These are eight reused development scenes, not proof of robust improvement or evidence comparable to the previous pilot's different held-out scenes.

## Question and scope

The [completed benefit-aware pilot](2026-10-11-robotics-benefit-pilot.md) showed one additional successful push, almost unchanged AI calls, and slightly slower execution. Its controller could interrupt once but could not extend useful plans. This separate investigation tests a more direct efficiency mechanism: **reuse twelve actions from a prediction when camera feedback and the predicted path support continuing; otherwise predict again after eight.**

This is a **development screen**, not a new held-out evaluation. We use predetermined discovery seeds 0–7, not the previous held-out seeds 30–49. Three methods on each scene give 24 complete attempts. Rules are frozen before collection, with no parameter search or changes after seeing outcomes. The previous pilot's code, evidence and model remain unchanged.

The aim is fewer AI calls and shorter total execution time **without sacrificing pushing success or final alignment**. Faster failure is not an improvement. A small eight-scene screen can identify an implementation problem or promising trade-off; it cannot establish a reliable general improvement.

## What the saved discovery trajectories show

We inspected camera-derived features from all 30 discovery scenes without simulator replay or AI inference. At nonterminal eight-action boundaries with four subsequent recorded actions available, 427 meet the proposed camera condition. A proxy using the five latest executed commands finds smooth, non-tiny movement at 363 of those boundaries.

| Development scene | Camera candidates | Also prefix-smooth | Subsequent four-action decreases greater than three points |
|---|---:|---:|---:|
| 0 | 13 | 13 | 1 |
| 1 | 12 | 10 | 0 |
| 2 | 15 | 13 | 1 |
| 3 | 16 | 13 | 0 |
| 4 | 19 | 15 | 0 |
| 5 | 13 | 10 | 0 |
| 6 | 12 | 12 | 0 |
| 7 | 20 | 16 | 1 |

The baseline's subsequent four actions come from a **fresh prediction**, not the discarded suffix of the previous prediction. These observations establish opportunities for testing, not the quality of cached actions 9–12. Across all 30 scenes, eight camera candidates have subsequent decreases greater than three points; seven also pass the prefix-smooth proxy. Smooth commands and low warnings are not proven safety signals.

Temporary deterioration also occurs in successful attempts. In discovery scene 8, estimated coverage falls from a recent peak of 88.97% to 64.19% after action 110, yet the baseline eventually succeeds. Therefore, the new interruption rule remains a bounded hypothesis, not a failure detector.

Evidence: [saved discovery analysis](../results/robotics-adaptive-length/discovery-analysis.json). The analysis uses past-only features to identify candidates; later changes are descriptive diagnostics, not runtime inputs.

## Proposed adaptive execution rule

The frozen Diffusion Policy samples a 16-position trajectory. With its two-observation history, at most 15 positions are usable future actions. Its default retains eight. We retain twelve, without changing the neural weights, observation history, denoising schedule or prediction horizon. This changes the execution queue, not the size of the AI model.

After executing eight actions from a prediction, keep the four remaining cached actions only if all conditions hold:

- The camera estimate is valid and has at least eight preceding changes.
- Estimated target coverage is below 75%, avoiding extension during final alignment.
- There are no recent warnings; recent drawdown is at most 1.5 percentage points, and the latest three-action coverage change is at least −1 point.
- Planned differences from the final four prefix commands through the four-action suffix are each at least two simulator coordinate units, with consecutive direction cosine at least 0.5. This checks the prefix/suffix join and rejects planned slowdown or sharp reversal.

The plan condition uses predicted commands already available to the controller, not future physical observations. The two-unit threshold is in the policy's 512-coordinate action space, not camera pixels. Eligibility is not proof that extending is safe or productive.

Otherwise, discard the four pending actions and predict again. Only the action queue is cleared; visual observation history is retained. A missing camera estimate causes the conservative eight-action choice, not continuation to twelve.

### Bounded early interruption

One warning means estimated coverage fell by more than 1.5 percentage points in one movement. The warning count covers up to the latest five movements. Drawdown is the best estimated coverage among up to the latest eight observations minus current coverage; recovery is the sum of up to the latest three coverage changes. These past-only signals describe movement, not the robot's intention.

The old learned tree is not reused repeatedly. A new hand rule requests feedback early only when the camera reports at least two warnings, at least a three-point drawdown, a recent peak of at least 80%, and a latest-three-action change below −0.5 points. The condition must hold for two consecutive observations, with at least four actions already executed from the current prediction and at least two actions pending.

At most two early interruptions are allowed per attempt. They must be at least twelve executed actions apart. The trigger must clear on a valid observation before it rearms; unchanged old warnings cannot immediately discard successive predictions. These guards limit repeated triggering, but do not guarantee useful recovery.

No controller training, repeated application of the old learned gate, adaptive denoising, disturbance injection or new model download is included.

## Comparisons and implementation checks

| Method | Execution behavior |
|---|---|
| Fixed-eight | Original eight-action interval, no added monitor |
| Fixed-twelve | Execute twelve actions before the next prediction, no added monitor |
| Adaptive-twelve | Camera/path check at action eight; retain or discard the final four, with bounded early interruption |

All methods use the same pinned checkpoint, initial scenes, 300-action limit and strict greater-than-95% coverage success criterion. Random generators reset to each scene's seed; sampling noise is aligned by prediction order, not physical time. Method order rotates across scenes. Total runtime includes monitoring and decisions, but excludes model loading, environment creation and trace writing.

The GPU smoke check passed before full collection: identical first-eight commands and physical states for fixed-eight, fixed-twelve and forced-trim-twelve. A forced-trim-twelve wrapper also matched two complete eight-action intervals, proving that retaining more predicted actions and discarding them does not corrupt the prefix or observation history. Fixed-twelve executed its cached suffix without an additional prediction. This verifies mechanics, not performance. See [prefix-smoke.json](../results/robotics-adaptive-length/prefix-smoke.json).

The new runner records plan lengths, retained/discarded actions, interruption reasons, camera features, AI calls, timings and compressed action/state traces. Dynamic simulator geometry and coverage are diagnostic/scoring data only. The protocol binds source files, discovery evidence, package versions and checkpoint identity; completed attempts are verified before resuming.

## Results

All 24 predetermined attempts completed with the rules unchanged. Alignment is the final fraction of target area covered by the block, averaged equally across all eight attempts, including failures. Success requires coverage strictly greater than 95%. Calls and runtime are also averaged across all attempts; runtime includes camera checking and decisions.

| Method | Successful pushes | Average final alignment | Average AI calls | Average complete-attempt runtime |
|---|---:|---:|---:|---:|
| Fixed-eight baseline | 3/8 | 93.43% | 32.63 | 162.85 s |
| Fixed-twelve | 5/8 | 82.05% | 18.50 | 92.76 s |
| Adaptive eight/twelve | 6/8 | 92.14% | 20.38 | 111.31 s |

Evidence: [summary.json](../results/robotics-adaptive-length/summary.json), [all attempts](../results/robotics-adaptive-length/episodes.json), and [independent audit](../results/robotics-adaptive-length/independent-audit.json). No screen trials were excluded or selectively rerun after seeing their outcomes.

### Which scenes improved or worsened?

Each row is the same initial scene under three methods. The seed is its reproducible scene index, not a timestep.

| Scene seed | Fixed-eight | Fixed-twelve | Adaptive eight/twelve |
|---|---|---|---|
| 0 | Success | Failure | Failure |
| 1 | Success | Success | Success |
| 2 | Failure | Failure | Success |
| 3 | Failure | Success | Success |
| 4 | Failure | Success | Success |
| 5 | Success | Success | Success |
| 6 | Failure | Success | Success |
| 7 | Failure | Failure | Failure |

The adaptive rule rescued baseline failures 2, 3, 4 and 6, but lost baseline success 0: four gains and one loss, not three gains with no downside. Scene 0's final alignment fell from 95.85% to 66.80%. This large regression explains why a higher success count did not also produce higher average alignment. Fixed-twelve was worse still in that scene, and its failed scenes 0 and 7 ended at roughly 45% and 50% coverage.

Across all scenes, adaptive execution saved 98 predictions and 412.34 seconds versus fixed-eight. It did not save computation in every scene: scene 0 used four more predictions and took 34.79 seconds longer than the successful baseline. Thus, we cannot claim that the controller preserves every successful baseline behavior.

As a supporting check, both fixed-eight and adaptive execution succeeded in scenes 1 and 5. On that two-scene subset, average calls fell from 23.5 to 17 and average runtime from 118.26 to 93.53 seconds. These savings do not come from replacing those successes with faster failures. Nevertheless, this outcome-selected subgroup is descriptive, not independent causal evidence; the all-scene table remains primary.

### What the controller actually did

- It retained the four extra actions 75 times and trimmed them 80 times.
- Average executed actions per prediction increased from 7.88 to 9.68, using total actions divided by total predictions. Partial final plans explain why the fixed-eight value is below eight.
- It interrupted early only once: scene 6 after movement 72, four movements into that prediction, discarding eight pending movements. There were three recent warnings and a 10.38-point camera-estimated drawdown.
- Fixed-twelve also succeeded in scene 6. Therefore, the observed recovery cannot be credited specifically to that interruption without an ablation.
- Monitoring and decisions cost an average of 9.71 seconds per adaptive attempt, already included in the runtime comparison.

Relative to fixed-twelve, adaptive execution used 10.14% more predictions and ran 20.00% longer, while improving success from 5/8 to 6/8 and average final alignment by 10.09 percentage points. This is a quality/computation trade-off, not a method that dominates both baselines. The combined rule and changed trajectories do not establish which component caused each gain.

## Verification and laptop cost

All **124 robotics tests passed**, including 33 new tests of thresholds, plan continuity, queue trimming, observation-history preservation, interruption guards, source binding and resumability. The GPU prefix smoke check passed separately.

The independent arithmetic/artifact audit verified all 24 trace hashes, 5,392 executed actions and queue transitions, 1,570 saved camera-feature rows, all 75 complete extended command paths and the one interruption. All 2,057 fixed-eight actions and originally recorded state/image fields matched the original discovery recordings. It also checked 47 source/evidence hashes, installed package versions, checkpoint weights and all summary values.

Audit limits matter: it recomputed camera arithmetic from recorded estimated coverage, not new RGB estimation. The initial camera estimate is not saved and was inferred from the first logged change, so its image fit was not independently verified. Full discarded plans are not saved, so the command-path reason for 23 camera-eligible trims cannot be independently reconstructed. Unit tests and inspected source support that mechanism; the artifact audit does not prove those hidden suffixes individually. No simulator replay or additional inference was used for this audit.

A fresh staged checkout preserved all 47 manifest files and 24 new trace hashes. The previous pilot's 50 bindings and 182 trace hashes also remained valid; its frozen evidence and source were not changed.

The collector took **49.70 minutes** on the GTX 1650. Peak process memory was **2.94 GiB**; minimum available system memory was **3.02 GiB** during the run. PyTorch peak GPU allocation/reservation was **1.07/1.15 GiB**, which excludes some driver/desktop GPU memory. A separate running-system spot check showed about 1.9 GiB total GPU usage out of 4 GiB. All attempts completed without GPU-memory errors. See [run-resources.json](../results/robotics-adaptive-length/run-resources.json).

This confirms laptop feasibility for this bounded offline experiment, not real-time control, training the robot policy or a much larger experiment budget.

## Recommended next decision

Keep adaptive action reuse as the provisional method, but do not tune these rules on the eight outcomes or add controller complexity now. The useful next test is the same paired three-method comparison on genuinely fresh scenes with the rules frozen. If its success/computation trade-off persists, compare extension alone with extension plus interruption to identify what helps. The scene-0 regression and mean-alignment decrease must remain visible, not be dismissed because the success count increased.

## Reproduction

Use the existing `.venv-robotics` and pinned checkpoint. No dependency installation or paid API is required. Put reruns in the ignored repeat directory so the original trial evidence and resource measurements remain unchanged.

```powershell
.\.venv-robotics\Scripts\python.exe -m unittest discover -s tests -p 'test_robotics_*.py' -v
.\.venv-robotics\Scripts\python.exe experiments\robotics_adaptive_length.py --stage analyze --output-dir results/robotics-adaptive-length-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_adaptive_length.py --stage prepare --output-dir results/robotics-adaptive-length-repeat
.\.venv-robotics\Scripts\python.exe experiments\robotics_adaptive_length.py --stage run --output-dir results/robotics-adaptive-length-repeat
```

`run` performs the smoke check if needed and resumes verified completed attempts. Do not edit frozen runner code, change the environment or launch concurrent GPU collectors during collection. Rerunning discovery seeds is not a new independent evaluation. Stage resource files describe their latest invocation; do not run a completed collector in the authoritative directory merely to inspect its summary.

## Attribution and limitations

- The frozen predictor is reused [Diffusion Policy, Chi et al.](https://arxiv.org/abs/2303.04137), accessed through the pinned [LeRobot checkpoint](https://huggingface.co/lerobot/diffusion_pusht).
- Physics, actions, success scoring and camera rendering come from [Gym-PushT](https://github.com/huggingface/gym-pusht).
- Camera fitting and its accuracy evidence are reused from our [previous pilot](2026-10-11-robotics-benefit-pilot.md). Its clean-renderer accuracy does not establish real-camera robustness.
- Adaptive execution and replanning have established precedent, reviewed in our [literature report](2026-10-11-robotics-replanning-literature.md). This rule is our simple course adaptation, not a claim that adaptive action chunks are new.

The methods change future observations and trajectories; identical initial noise does not make all future actions equivalent. Eight reused scenes and one stochastic sample per method are insufficient for a general improvement claim. These first eight discovery scenes are also relatively difficult: fixed-eight succeeds in 3/8, compared with 20/30 across the complete discovery collection. We did not select them by the new methods' outcomes, but they are not a representative or held-out sample. Extension and interruption are combined, so the favorable success/computation result motivates later ablations and genuinely fresh evaluation—not a claim that every metric improves or that adaptive chunks are a new algorithm.
