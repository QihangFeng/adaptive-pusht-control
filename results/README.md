# Robotics experiment evidence

## Laptop feasibility

- [robotics-feasibility/summary.json](robotics-feasibility/summary.json): authoritative working environment, pinned model identity, simulator assertions, GPU/RAM measurements, prediction timing and three complete baseline episodes.
- [robotics-feasibility/simulator-check.json](robotics-feasibility/simulator-check.json): earlier simulator-only smoke test. Its package inventory predates the PyArrow/pandas compatibility correction; use `summary.json` for the final environment.
- [Feasibility report](../reports/2026-10-10-robotics-laptop-feasibility.md): understandable conclusions, measurement definitions, limitations and reproduction instructions.

These results establish offline laptop feasibility, not a reliable success-rate estimate or an improvement from our proposed controller. Preserve the original evidence; put disposable reruns in a `*-repeat/` directory, which is Git-ignored.

## Baseline diagnosis

- [robotics-baseline-check/summary.json](robotics-baseline-check/summary.json): six fixed-eight episodes, feasibility compatibility checks, candidate windows, and 15 same-state local replanning trials.
- [robotics-baseline-check/protocol.json](robotics-baseline-check/protocol.json): unchanged policy settings, diagnostic rules, episode seeds, and prespecified probe selection/seeds.
- `robotics-baseline-check/seed-*.trace.json.gz`: commands, prediction boundaries, actual pusher-block contact, physical motion, and coverage. Physical state is diagnostic evidence, not an input to the policy.
- `robotics-baseline-check/seed-*-frames.png` and [coverage/contact plot](robotics-baseline-check/coverage-contact-review.png): small visual evidence for reviewing the candidate windows.
- [Baseline report](../reports/2026-10-10-robotics-baseline-check.md): findings, limitations, references, and reproduction commands.

Contact-loss candidates are not confirmed failures. The short probes did not demonstrate immediate alignment improvement and do not evaluate full-episode recovery or an adaptive controller. Full videos remain local under `tmp/robotics-baseline-videos/`; rerunning the experiment regenerates them.

## Expanded failure and recovery investigation

- [robotics-failure-patterns/summary.json](robotics-failure-patterns/summary.json): 30 completed fixed-eight baseline attempts, reusing the original six and adding seeds 6-29. Outcomes: 20 successes, 10 timeouts.
- [robotics-failure-patterns/protocol.json](robotics-failure-patterns/protocol.json): frozen settings, exploratory rules, predetermined seeds, recovery selection and original-evidence hashes.
- [robotics-failure-patterns/analysis.json](robotics-failure-patterns/analysis.json): nonexclusive trajectory patterns in failures and successes; simulator-only diagnostics, not a visual controller.
- [robotics-failure-patterns/recovery.json](robotics-failure-patterns/recovery.json): all 24 matched-state continuations, including harmful and neutral interventions. The two arms each succeed in 5/12 continuations; replanning uses more remaining AI calls (167 versus 142).
- `robotics-failure-patterns/episodes/` and `recovery/`: per-attempt summaries and compressed action/state traces; original seeds 0-5 remain referenced in `robotics-baseline-check/`.
- [robotics-failure-patterns/figures/](robotics-failure-patterns/figures/): replay-verified selected baseline frames and every recovery endpoint.
- [Failure/recovery report](../reports/2026-10-11-robotics-failure-patterns.md): conclusions, all paired outcomes, limitations and reproduction instructions.

These deliberately selected continuations demonstrate some recovery opportunities, not a net controller improvement, a population success rate or equivalence of the methods. Adaptive chunk length and a deployable monitor are not evaluated. New baseline videos remain local under `tmp/robotics-failure-pattern-videos/`.

## Benefit-aware replanning pilot

- [robotics-benefit-pilot/visual-validation.json](robotics-benefit-pilot/visual-validation.json): CPU-only replay of 30 saved attempts, exact camera-hash checks and camera-estimator accuracy. No new AI predictions.
- [robotics-benefit-pilot/near-target-validation.json](robotics-benefit-pilot/near-target-validation.json): additional accuracy check where simulator truth coverage is at least 75%. Truth is used only to validate the estimator.
- [robotics-benefit-pilot/visual-features.json.gz](robotics-benefit-pilot/visual-features.json.gz): past-only, camera-derived features; no future outcomes or dynamic simulator truth.
- [robotics-benefit-pilot/protocol.json](robotics-benefit-pilot/protocol.json): 17 frozen checkpoints, episode-separated training/validation, fresh label seeds and held-out seeds 30–49; original evidence and code fingerprints.
- `robotics-benefit-pilot/labels/` and [labels.json](robotics-benefit-pilot/labels.json): all 102 matched continuations, three repeats per arm per checkpoint.
- [model.json](robotics-benefit-pilot/model.json): frozen depth-two tree, empirical benefit targets and validation-only threshold selection. Validation showed no success gain.
- [evaluation-protocol.json](robotics-benefit-pilot/evaluation-protocol.json): binds held-out evaluation to the frozen protocol and model.
- `robotics-benefit-pilot/evaluation/` and [evaluation.json](robotics-benefit-pilot/evaluation.json): all 80 full held-out attempts with sidecars, trace hashes, decisions and query counts.
- [summary.json](robotics-benefit-pilot/summary.json): four-method metrics and paired outcomes. Gate 16/20 successes; fixed-eight and warning rule 15/20; fixed-four 5/20.
- `*-resources.json`: stage memory and duration measurements. Evaluation measurements cover the resumed 45-attempt invocation; per-attempt timings cover all 80 attempts.
- [Pilot report](../reports/2026-10-11-robotics-benefit-pilot.md): conclusions, method, training/validation results, complete held-out comparison, individual interruptions, limitations and reproduction instructions.

The implemented controller makes at most one early decision per attempt. Its observed improvement was one additional success, with only five AI calls saved overall and slightly slower execution after monitoring. All eight interventions addressed stagnation, not alignment-warning events. This small pilot does not establish robust improvement or learning's advantage over an untested same-timing simple rule. The 91 tests and independent audits of all trials passed. Keep the frozen source, environment and evidence unchanged when resuming; never launch concurrent collectors.

## Adaptive action-reuse development screen

- [robotics-adaptive-length/discovery-analysis.json](robotics-adaptive-length/discovery-analysis.json): CPU-only analysis of the 30 saved discovery trajectories. Later baseline movements are fresh predictions, not evidence about discarded cached suffixes.
- [protocol.json](robotics-adaptive-length/protocol.json): predetermined discovery seeds 0–7, fixed-eight/fixed-twelve/adaptive methods, frozen rules, source/evidence hashes, package versions and checkpoint identity.
- [prefix-smoke.json](robotics-adaptive-length/prefix-smoke.json): GPU verification of identical initial prefixes, exact forced-trim matching over sixteen actions and actual cached-suffix execution.
- `robotics-adaptive-length/episodes/` and [episodes.json](robotics-adaptive-length/episodes.json): all 24 complete attempts, compressed action/state traces, actual unit lengths, query counts, camera features and controller events.
- [summary.json](robotics-adaptive-length/summary.json): all-scene metrics. Adaptive 6/8 successes, fixed-eight 3/8 and fixed-twelve 5/8. Adaptive uses fewer calls and less runtime than fixed-eight, but its mean final alignment is lower.
- [independent-audit.json](robotics-adaptive-length/independent-audit.json): direct fingerprint checks and independent arithmetic verification of all 5,392 actions/queue transitions, all 75 extended paths, the interruption and all summary values. Discarded suffixes and RGB estimates cannot be fully rechecked from these records.
- [run-resources.json](robotics-adaptive-length/run-resources.json): the complete collector invocation, 49.70 minutes, peak process memory 2.94 GiB and peak PyTorch GPU allocation 1.07 GiB.
- [Adaptive-length report](../reports/2026-10-12-robotics-adaptive-length.md): conclusion-first comparison, paired outcomes, trade-offs, audit limits and reproduction instructions.

These eight scenes are reused development data, with one stochastic sample per method. Adaptive execution rescued four baseline failures and lost one baseline success. Fixed-twelve provides a substantial efficiency gain without the monitor, so the adaptive rule's benefit must be assessed against it, not just fixed-eight. All 124 robotics tests passed. Preserve this frozen screen and the earlier pilot; use an ignored `*-repeat/` directory for reruns and fresh scenes for a future generalization test.

## Fresh-scene adaptive action-reuse evaluation

- [robotics-fresh-evaluation/protocol.json](robotics-fresh-evaluation/protocol.json): frozen seeds 50–69, three unchanged methods, rules, checkpoint/packages, 51 source/evidence bindings and paired statistical plan, saved before collection.
- `robotics-fresh-evaluation/episodes/` and [episodes.json](robotics-fresh-evaluation/episodes.json): all 60 complete attempts, sidecars and compressed traces; no excluded or selectively repeated trials.
- [summary.json](robotics-fresh-evaluation/summary.json): four-metric means, paired gained/lost successes, 10,000-scene-pair bootstrap intervals and exact discordant-success tests. Adaptive 18/20 successes versus fixed-eight 15/20 and fixed-twelve 16/20; success advantages remain uncertain.
- [independent-audit.json](robotics-fresh-evaluation/independent-audit.json): all 60 hashes, 10,691 movements/queue transitions, 3,435 camera rows, 131 retained paths, seven interruptions, source/packages/weights, 20 unique initial scenes and summary/statistical arithmetic. Hidden discarded suffixes and RGB fits retain explicit audit limits.
- [run-resources.json](robotics-fresh-evaluation/run-resources.json): one complete 97.21-minute collector invocation, peak process memory 2.57 GiB and PyTorch GPU allocation 1.07 GiB.
- [Fresh-scene report](../reports/2026-10-12-robotics-fresh-evaluation.md): fixed evaluation plan, results, every scene's outcome, uncertainty, regressions, attribution and reproduction instructions.

Adaptive execution improved all four observed means versus fixed-eight, with 22.22% fewer calls and 16.35% shorter runtime, but lost baseline successes 54 and 59. Fixed-twelve remains cheaper. All 152 tests and independent checks passed; neither the combined controller's interruption-specific benefit nor reliability across repeated prediction seeds is established. Preserve this evaluation and all earlier frozen evidence. Seeds 50–69 are no longer unseen data for a method revised after inspecting these results.

## Lost-success diagnosis

- [robotics-regression-diagnosis/protocol.json](robotics-regression-diagnosis/protocol.json): 26 fixed diagnostic trial identities, source/evidence hashes, distinct scene/prediction seeds, bounded intervention definitions and the existing truth-coverage rounding tolerance.
- [saved source analysis](robotics-regression-diagnosis/saved-analysis.json): selected failure moments and milestone comparisons, prepared before new scored trials.
- [restored-prefix audit](robotics-regression-diagnosis/restore-prefix-audit.json): hidden cached commands, controller states and complete RNG/queue hash inventories for four checkpoints.
- [all trials](robotics-regression-diagnosis/trials.json) and `robotics-regression-diagnosis/trials/`: two unchanged restoration controls, eight complete sampling repeats, sixteen remaining-episode probes, sidecars and hash-bound traces.
- [summary](robotics-regression-diagnosis/summary.json), [independent audit](robotics-regression-diagnosis/independent-audit.json), and [resources](robotics-regression-diagnosis/resources.json): verified comparisons, explicit audit limitations and laptop measurements.
- [Diagnosis report](../reports/2026-10-12-robotics-regression-diagnosis.md): understandable results, decision-point examples and interpretation.

Prediction samples changed both baseline and adaptive outcomes. Neither earlier replanning nor finishing the current unit established successful recovery; both changes lost the same successful scene-59 continuation. The milestone controller stays unchanged. These are conditional selected-case diagnostics, not a new held-out evaluation or an adopted controller revision. Local runtime/call counts exclude the prefix and must not be mixed with complete-attempt metrics. Preserve commit `0f40449` as the designated milestone and keep scenes 54/59 out of unseen evaluation for revisions informed by this diagnosis.

## Alignment-decline benefit screen

- [robotics-decline-screen/protocol.json](robotics-decline-screen/protocol.json), [cases.json](robotics-decline-screen/cases.json), and [legacy inventory](robotics-decline-screen/legacy-inventory.json): frozen selection of all six first adaptive interruptions, past-only features, three future samples and references to completed evidence.
- [comparisons.json](robotics-decline-screen/comparisons.json) and `robotics-decline-screen/trials/`: 36 comparison records, eight reused by hash-bound reference and 28 newly collected; four new restoration controls bring new collection to 32 trials.
- [restoration audit](robotics-decline-screen/restoration-audit.json) and [controls.json](robotics-decline-screen/controls.json): all six regenerated prefixes, four new exact suffix controls and two reused controls.
- [summary.json](robotics-decline-screen/summary.json), [independent audit](robotics-decline-screen/independent-audit.json), and [resources](robotics-decline-screen/resources.json): per-context outcomes, feature rank associations, cached versus eventual recovery, integrity checks and the 46.97-minute collection cost.
- [Decline-screen report](../reports/2026-10-12-robotics-decline-screen.md): conclusion-first tables, separate older fixed-eight evidence, post hoc latest-change exploration, limitations and reproduction commands.

Both choices succeeded 7/18 times, with five gains and five losses from replanning. Two-warning/smaller-drop contexts favored continuing; three-warning/larger-drop contexts favored replanning. These are six source situations, not eighteen independent scenes, and the signals are correlated. No classifier was trained or controller changed. All 213 tests, six restoration controls and independent checks passed. The milestone remains `0f40449`; these inspected scenes are development evidence for any subsequent revision.

## Weighted versus simple interruption gates

- [robotics-weighted-gate/development.json](robotics-weighted-gate/development.json) and [protocol.json](robotics-weighted-gate/protocol.json): twelve weighted and six threshold candidates evaluated only on six existing development contexts; selected settings, ties, proxy limitations and 223 frozen bindings.
- [smoke.json](robotics-weighted-gate/smoke.json), `robotics-weighted-gate/smoke/` and [smoke-resources.json](robotics-weighted-gate/smoke-resources.json): exact 194-movement unchanged-controller reproduction plus two functional gate attempts; no retuning after preflight.
- [evaluation-protocol.json](robotics-weighted-gate/evaluation-protocol.json), [episodes.json](robotics-weighted-gate/episodes.json) and `robotics-weighted-gate/episodes/`: all 60 full attempts on seeds 70–89, three methods, complete twelve-command plans including discarded suffixes, camera histories, decisions and hash-bound traces.
- [summary.json](robotics-weighted-gate/summary.json): success, alignment, prediction/runtime means, paired gains/losses, exact discordant tests and scene-pair bootstrap intervals.
- [independent-audit.json](robotics-weighted-gate/independent-audit.json): all 63 preflight/fresh records, 13,653 movements and 1,482 plans; independent source/evidence checks, gate/queue bookkeeping and summary/statistical arithmetic. Physics, inference and RGB fitting were not rerun.
- [run-resources.json](robotics-weighted-gate/run-resources.json): complete 127.65-minute fresh collector invocation; peak RAM 2.51 GiB and allocated GPU memory 1.07 GiB.
- [Gate-evaluation report](../reports/2026-10-13-robotics-weighted-gate.md): formulas, parameter selection, four-metric results, uncertainty, exact decision-point examples, audit limits and reproduction instructions.

Both gates achieved 16/20 successes versus unchanged adaptive 15/20, gained scenes 72/81 and lost 73. Simple gating had 94.58% mean alignment versus weighted 92.13% and unchanged 92.02%; both gates cost slightly more predictions/runtime than unchanged. Weighting shows no demonstrated advantage. All 247 tests and the independent audit passed. Neither gate is adopted; preserve the milestone and frozen evidence. Seeds 70–89 are now inspected data, not unseen evaluation for a revised method.

## Saved extension and plan-transition review

- [robotics-decision-review/analysis.json](robotics-decision-review/analysis.json): verifies all 60 milestone attempts and 176 source/evidence bindings; derives every extension and noninitial prediction transition, separates camera inputs from true diagnostic outcomes, and records exact loss/recovery windows.
- [independent-audit.json](robotics-decision-review/independent-audit.json): independently derived arithmetic, source hashes and explicit inspection limits.
- [Decision-review report](../reports/2026-10-13-robotics-decision-review.md): understandable findings, successful counterexamples, exact movement indices, final-positioning interpretation and CPU reproduction instructions.

All 131 adaptive extended suffixes executed four movements. Losses 54/59 contain 21 extensions, with no immediate suffix drop greater than three percentage points; a successful scene has a much larger suffix decline. Their destructive fresh plans followed ordinary eight-action trimming, not immediate extensions, and do not share a large-jump/reversal defect. Both later severe declines regain pre-drop coverage but never finish above 95%. All 273 tests and the independent audit passed. This is retrospective inspection, not causal proof, a new controller or fresh evaluation. The milestone remains unchanged; no GPU, simulator, rendering or threshold/weighted retest was run.

## Completed comparator evidence and pooled synthesis

- [robotics-cohort-completion/protocol.json](robotics-cohort-completion/protocol.json): 369 frozen bindings, unchanged rules/checkpoint/packages, baseline-only seeds 70–89, exact prefix requirements and paired statistical plan.
- `robotics-cohort-completion/episodes/` and [episodes.json](robotics-cohort-completion/episodes.json): all 40 new fixed-eight/fixed-twelve sidecars and compressed traces, plus hash-bound references to 20 existing unchanged adaptive attempts. Original 50–69 evidence remains in its source directory; nothing is overwritten or selectively rerun.
- [summary.json](robotics-cohort-completion/summary.json): separate additional-20-scene and pooled-40-scene tables, paired gains/losses, whole-scene bootstrap intervals and exact discordant-success tests. Runtime is descriptive only because sessions/logging differ.
- [independent-audit.json](robotics-cohort-completion/independent-audit.json): 120 records, 22,657 movements, 2,411 predictions, 40 exact first-eight-command/physics/RGB pairs and independently checked arithmetic. No inference/physical replay/camera refitting was repeated by the audit.
- [run-resources-from-00.json](robotics-cohort-completion/run-resources-from-00.json): complete 65.12-minute invocation, peak RAM 2.52 GiB and GPU allocation 1.07 GiB.
- [Completion report](../reports/2026-10-13-robotics-cohort-completion.md): conclusion-first comparison, uncertainty, resource checks, limitations and reproduction instructions.

The original milestone improvement did not repeat: additional-cohort adaptive/fixed-eight successes were 15/20 versus 18/20. Pooled, both finished 33/40, with six adaptive gains and six losses. Adaptive averaged 7.38% fewer predictions, but its saving interval includes zero; alignment and observed mean runtime did not improve. Fixed-twelve finished 29/40 with fewer predictions and lower alignment, a possible quality/cost trade-off rather than adaptive dominance. All 294 tests and the independent audit passed. Preserve milestone `0f40449`; no controller, scoring criterion or rejected filter was changed. These 40 initial scenes are inspected data, not a new untouched holdout or repeat-sampling reliability test.

## Six-lost-success saved-evidence review

- [robotics-loss-review/analysis.json](robotics-loss-review/analysis.json): 457 source/evidence bindings, all 120 saved attempt summaries, 289 extension windows, exact common prefixes, terminal-48-movement diagnostics, physical pose/path errors, chronological drawdown/recovery, camera errors and nonterminal stationary controls.
- [independent-audit.json](robotics-loss-review/independent-audit.json): independently recomputed diagnostics/groups and raw provenance, including 22,657 movements, 2,411 predictions and all 40 common-prefix pairs. No inference, physics or image fitting was rerun.
- [Lost-success report](../reports/2026-10-13-robotics-lost-success-review.md): understandable six-case table, exact movement indices, successful counterexamples, limitations and CPU reproduction command.

All six lost successes approach but never exceed 95%. Cases 78/81 have final block-stationary intervals despite 12/nine further predictions; 80/82 suffer renewed destructive fresh-plan pushing. Successful attempts also contain long nonterminal stalls and substantial drops, so the diagnostics are not new failure labels or a validated recovery rule. All 336 tests and the independent audit passed. Earlier 54/59 trials and extension windows were reused, not rerun; the milestone, old sources/results, controller and completion criterion remain unchanged.

## Camera precision before final-action correction

- [robotics-pose-gate/protocol.json](robotics-pose-gate/protocol.json): 34 frozen source/evidence bindings, package/renderer identity, discovery-only selection of 76 fine-alignment frames plus 20 terminal controls, and accuracy/direction targets declared before production fitting.
- [samples.json](robotics-pose-gate/samples.json): all 96 frame identities, exact original RGB hashes, cold/two-image estimates, 192 scored poses and offline truth comparisons; no new controller or policy calls.
- [summary.json](robotics-pose-gate/summary.json): separate fine/control strata, availability, position/rotation/vertex precision, per-axis direction counts, frame-weighted and scene-equal errors; the primary and sensitivity gates both fail.
- [independent-audit.json](robotics-pose-gate/independent-audit.json) and [resources-from-000.json](robotics-pose-gate/resources-from-000.json): independent statistical/provenance checks and the 36.65-second CPU replay/fit invocation, peak observed RAM 140.27 MiB.
- [Pose-gate report](../reports/2026-10-13-robotics-camera-pose-gate.md): conclusion-first accuracy table, stopped static preflight/replay fallback, interpretation, limitations and repeat commands.

Near-target direction agreement is 184/184 eligible components, but cold 95th-percentile position/rotation/vertex errors are 1.48 units/0.98°/1.99 units, above the 1-unit/0.5°/1-unit targets. All 384 tests and the independent audit passed. Stop before corrective pushes; no training, action model, controller adoption or success improvement is established. Keep the milestone and all earlier evidence unchanged. The preflight archive is recoverable locally; production contains every selected frame, with no threshold changes or selectively discarded fits.

Power-grid results belong to the repository's `main` branch and are intentionally absent here.
