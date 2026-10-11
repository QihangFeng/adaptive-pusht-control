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

Power-grid results belong to the repository's `main` branch and are intentionally absent here.
