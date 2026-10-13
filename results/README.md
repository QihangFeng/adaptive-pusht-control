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

Power-grid results belong to the repository's `main` branch and are intentionally absent here.
