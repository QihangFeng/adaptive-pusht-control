# Does correcting old voltage information actually help enough?

Date: October 8, 2026

Follow-up completed later the same day: the [environment is restored](2026-10-08-simulator-restoration.md), and the [four slower-controller comparisons are complete](2026-10-08-slow-update-comparison.md). The audit below is retained as the earlier evidence and pre-comparison decision rule.

## Conclusion

**Correction helps in the saved simulations, but we have not shown that it is necessary or that it reliably meets our requirements.** This distinction matters more than the large percentage improvements reported earlier.

- A reasonably tuned controller using old readings already meets our experimental voltage and modeled-latency budgets on all four delayed development cases. A slower-update alternative also passes all four.
- On the four evaluation cases, that selected old-reading controller passes none. Correction passes **one of four**, not four of four. It reduces accumulated voltage violations in every case, but three cases still dip below our experimental minimum voltage.
- The slower-update alternative has **not** been evaluated on those four cases. We therefore cannot yet say that correction is better than the relevant simpler alternative.
- None of the 48 saved runs keeps voltage entirely inside the nominal 0.95–1.05 range. This includes fresh-feedback runs. Mean-ITL requirements pass throughout, but that does not establish adequate real service latency or capacity.

**Recommendation:** retain the topic as a conditional robustness study, but do not treat correction as a proven solution or add controller complexity yet. First complete the missing slower-controller comparison. Current evidence does not show that we have solved voltage compliance or established a widespread AI-center problem.

Status: this report completes an independently reproducible **audit of saved evidence**, not new simulations. New simulation work is blocked by missing local simulator files and prepared input data; Section 5 explains the recovery needed.

## 1. What are we trying to achieve?

Our goal is: **process as much AI work as possible while meeting voltage and service requirements.** Voltage is a constraint; throughput is the performance we want to preserve or increase within that constraint. Improving one average metric does not necessarily make a controller adequate.

Three questions are distinct:

1. **Acceptability:** does a controller meet the stated requirements?
2. **Added value:** does correction usefully improve on a reasonable uncorrected controller?
3. **Practical relevance:** does the assumed delayed-information architecture occur in the system we want to study?

The first two can be assessed together, but requirements must be specified independently of which controller wins. The third remains conditional: the literature review did not establish how often this particular remote-voltage delay occurs in real AI centers. [R1]

### Requirements used in this audit

We retain the original development-selection budgets without changing them to fit the results. This is a retrospective application of existing budgets to saved evaluation runs, not a newly preregistered experiment. [R2]

| Check | Requirement | Meaning and limitation |
|---|---|---|
| Strict nominal voltage | Every monitored voltage remains between 0.95 and 1.05 pu | Separate, stricter check; no excursions in the saved samples |
| Experimental voltage budgets | Accumulated violation ≤0.10 pu·s; minimum ≥0.94 pu; maximum ≤1.06 pu | Earlier screening assumptions permitting some excursions outside the nominal band; not an operational safety standard |
| Modeled service latency | At most 1% of samples exceed the 80 ms model-average inter-token-latency deadline | Does not measure individual-request latency, queueing, time to first token, or p99 latency |
| Throughput | Report simulated tokens/second; compare after checking constraints | No arrival-rate requirement or minimum service-capacity target was defined |
| Cost of correction | Report throughput changes and batch switching | Earlier useful-prototype rule required retaining ≥98% of baseline throughput and limited additional switching |

Here, 1 pu is the nominal voltage reference. Accumulated violation adds the distance outside 0.95–1.05 over monitored bus-phases and time. A 70% decrease in this metric is **not** a 70% improvement in voltage or proof of acceptable voltage.

“Pass” below means all experimental voltage and modeled-latency budgets pass. It does not mean strict nominal compliance or proven production-service adequacy.

## 2. What the saved results show

The audit reads the completed 30-run baseline screen and 18-run prototype stage. It recalculates metrics from all 48 compressed trajectories, checks trajectory hashes, cross-checks jobs/results/summaries, and verifies causal observation records. No simulation was rerun. [R2–R4]

### A. Old readings can already be adequate under the experimental budgets

For each controller below, the four delayed development cases comprise two workload schedules at 0.5-second and 2-second delay.

| Controller | Passes all four delayed development cases? | Lowest throughput across those cases, million tokens/s | Largest accumulated violation, pu·s |
|---|---:|---:|---:|
| Selected tuned controller, no correction | Yes | 13.598 | 0.06044 |
| Slower-update controller, no correction | Yes | 13.190 | 0.04923 |

The selected tuning has slightly higher worst-case development throughput; the slower controller has lower worst-case accumulated violation. These are development observations, not proof that either generalizes. The slower controller still uses the existing optimizer; it avoids the correction component but is not a new elementary rule-based controller. [R2]

Correction also has a development trade-off. At two-second delay, it gives 51.8% and 26.9% higher throughput than the slower alternative on the two schedules, but makes 19 batch changes versus 9 and 10. Both comparisons exceed the earlier switching cap, `max(baseline + 2, 1.1 × baseline)`. This is not improvement on every criterion. [R2–R4]

### B. Improvement is not the same as meeting requirements

All evaluation runs below use the frozen `half_primal_dual` setting. “Fresh” is a diagnostic reference with zero delay; “old” and “corrected” use two-second delay. Each run simulates 300 seconds. Individual seeds are shown because an average can hide a failure. [R3–R4]

| Workload / seed | Observation | Accumulated violation, pu·s | Minimum voltage, pu | Throughput, million tokens/s | Experimental budgets |
|---|---|---:|---:|---:|---|
| Abrupt / 1 | Fresh | 0.05014 | 0.937673 | 15.006 | Fail: minimum voltage |
| Abrupt / 1 | Old | 0.20845 | 0.936510 | 10.122 | Fail: accumulated violation and minimum |
| Abrupt / 1 | Corrected | 0.05044 | 0.937673 | 15.069 | Fail: minimum voltage |
| Abrupt / 2 | Fresh | 0.05393 | 0.937568 | 13.737 | Fail: minimum voltage |
| Abrupt / 2 | Old | 0.20457 | 0.937531 | 10.086 | Fail: accumulated violation and minimum |
| Abrupt / 2 | Corrected | 0.04754 | 0.937568 | 15.307 | Fail: minimum voltage |
| Ramps / 1 | Fresh | 0.05508 | 0.940053 | 24.136 | Pass |
| Ramps / 1 | Old | 0.16912 | 0.938019 | 24.301 | Fail: accumulated violation and minimum |
| Ramps / 1 | Corrected | 0.05508 | 0.940053 | 24.105 | Pass |
| Ramps / 2 | Fresh | 0.05701 | 0.939684 | 24.141 | Fail: minimum voltage |
| Ramps / 2 | Old | 0.16743 | 0.938304 | 24.307 | Fail: accumulated violation and minimum |
| Ramps / 2 | Corrected | 0.05881 | 0.939684 | 24.126 | Fail: minimum voltage |

Correction reduces accumulated violation by approximately 65–77% in these four comparisons. It raises throughput by approximately 49–52% in the abrupt cases and reduces throughput by less than 1% in the ramp cases. Those remain genuine results. However:

- Only ramp seed 1 becomes acceptable under all experimental budgets, with a 0.81% throughput cost and unchanged switching.
- In the other three pairs, **neither** old nor corrected feedback meets every budget.
- Ramp seed 2 reaches 0.939684 pu: slightly below 0.94, but still a failure under the stated rule. Rounding it to 0.940 would conceal that distinction. Two seeds and one near-boundary pass do not establish robust compliance.
- Fresh feedback passes the same one of four cases. Correction's remaining minimum-voltage failures also occur without measurement delay in these runs. Removing the delay penalty is therefore insufficient to demonstrate full constraint satisfaction. Fresh-feedback failure does not prove physical infeasibility or rule out another controller succeeding.

Across all 48 runs, 35 pass the relaxed experimental budgets, zero pass strict nominal voltage, and all have zero modeled-ITL misses. The 35/48 count mixes development and evaluation; it is not an independent generalization success rate. [R4]

## 3. What changes in our earlier interpretation?

The earlier report correctly stated that all four pairs passed its **relative useful-prototype rule**: enough accumulated-violation reduction, preserved throughput, and limited extra switching. That rule did not require the corrected run to meet every absolute development-selection budget. [R5]

The clearer conclusion is:

> Correction reduces the damage caused by old measurements in the tested fixed-background simulations. It has not demonstrated reliably acceptable operation, and its advantage over the slower alternative on evaluation cases remains untested.

This does not invalidate the method or make the topic worthless. It narrows the claim. Our contribution could be determining **when correction is useful, unnecessary, or insufficient**, with mathematical explanation and fair comparisons. It should not assume that old measurements always require correction.

## 4. What should we test next?

### First: the missing simpler alternative

After restoring the environment, run frozen `slow_update` on the existing two evaluation schedules and seeds 1 and 2, at two-second delay: **four new runs**. Do not choose a different winning setting for each case. Reuse saved comparable runs only after compatibility checks pass.

Judge each comparison in this order:

1. Which controllers meet the declared budgets?
2. Among those that qualify, which delivers more throughput, at what switching cost?
3. If neither qualifies, what failure remains? Do not call a relative gain an acceptable solution.

If simple stale control meets requirements with comparable throughput, correction is optional in that setting. If correction uniquely meets the budgets while retaining ≥98% of comparator throughput, it demonstrates additional capability in that case. If both qualify, use a 5% throughput increase as a working signal of additional value for this **future** comparison, retaining the earlier switching cap. This 5% threshold is a course-study choice, not an industry requirement or a claim preregistered for existing results.

### Then: assumptions most favorable to correction

The current estimate is `old voltage + sensitivity × (current local power − old local power)`. It corrects changes at the AI center. It cannot directly observe a new load change elsewhere while the relevant voltage packet is still in transit. [R1, R6]

If the first comparison justifies continuing, freeze a small follow-up protocol before new runs:

- Retain the selected and slower uncorrected controllers, existing correction, and fresh-feedback reference. Use the same plant, workload and all-bus evaluation limits.
- Test variable transport delay bounded by 0.5–2 seconds; report actual delivered measurement age and preserve out-of-order packet handling. This remains a sensitivity assumption, not a measured field distribution.
- Introduce separately scheduled changes to background grid loads, excluding AI-center loads. Separate variable-delay-only, external-change-only, and combined cases. Record exact schedules and magnitudes before execution; do not increase severity until correction looks attractive.
- Use matched seeds and fixed settings. Count failures. Verify external changes occur before the intended power-flow solve and are not exposed directly to the predictor.
- If fresh feedback also fails, inspect control authority before blaming delay. A low-batch diagnostic helps but is not a feasibility proof.

Fresh local-only control remains an unresolved architectural alternative. Do not claim remote feedback is necessary until that comparison is justified and performed. Request-level latency and sufficient service capacity also remain outside the present model.

## 5. Why new simulations have not been run

The documented setup lived under `C:\Users\冯同学\AppData\Local\Temp\ece730-feasibility-580bb4f1`. On inspection:

- `env-grid\Scripts\python.exe` is absent.
- `openg2g` contains a leftover cache, not the simulator source checkout.
- `measured-results\specs` is absent.
- Project `.venv` runs Python 3.11.9 but lacks NumPy, pytest, OpenG2G, OpenDSSDirect, mlenergy and huggingface_hub.
- No replacement was identified in the project, inspected course directory, matching temporary directories, or default Hugging Face cache inventory. This was a targeted check, not a search of every drive.

This is a missing-environment/input issue, **not evidence that the laptop lacks computing capacity**. Historical resource measurements remain historical evidence, not a currently runnable installation.

Recovery requires a saved copy or recreation of pinned OpenG2G and prepared measured GPU profiles, in a persistent location. The protocol records source commit `c582a5115cfd49dcdfb140f1837224e7cbf0cfb1`, dataset revision `fea6cb99d037d11f86f9c45a6991b9e5764f4529`, and input hashes. [R2–R3]

Before comparing new and saved results, match those inputs and reproduce one zero-delay and one two-second-delay trajectory. If they differ, diagnose the mismatch or establish a new common baseline; do not silently combine changed inputs with old results. No replacement data, synthetic profiles, package installation, or simulator restoration was performed here.

## 6. Reproduction and verification

The audit and tests use only Python's standard library and run in the existing project environment:

```powershell
# Run from the repository root. Use a new output file; evidence is not overwritten.
.\.venv\Scripts\python.exe experiments\delay_acceptability_audit.py --output results\delay-acceptability-audit-repeat\summary.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_delay_acceptability_audit.py -v
```

**Verification: 9 audit tests passed.** They cover inclusive budget boundaries, individual failures, strict versus relaxed checks, nonfinite values, matched comparisons, large relative gains that still fail requirements, metric consistency, and reproduction of numerical findings across all 48 trajectories. Output includes source/trace hashes and per-run failures. Text-file hashes record this checkout's bytes; tests permit Git's line-ending conversion while requiring matching binary trace hashes and numerical findings. The original simulator-dependent suite was not rerun because its dependencies are missing.

The audit independently aggregates exported per-step voltage-violation sums; it does not recompute physical power flows. Historical convergence flags are checked, not newly established. This is verification of saved evidence, not reproduction of the simulations.

## References

No new literature search was performed. New empirical conclusions come from saved code/results, not additional field evidence.

- **[R1]** [Telemetry-delay practicality review](2026-10-07-telemetry-delay-practicality-review.md), Sections 6–8: primary-literature references and limits of real-world motivation.
- **[R2]** Baseline screen: [protocol](../results/delay-screen/protocol.json), [selection](../results/delay-screen/selection.json), [summary](../results/delay-screen/summary.json), and per-run traces in that directory.
- **[R3]** Prototype/evaluation: [protocol](../results/delay-prototype/protocol.json), [summary](../results/delay-prototype/summary.json), and per-run traces in that directory.
- **[R4]** This audit: [implementation](../experiments/delay_acceptability_audit.py), [machine-readable findings](../results/delay-acceptability-audit/summary.json), [tests](../tests/test_delay_acceptability_audit.py).
- **[R5]** [Earlier delay go/no-go report](2026-09-20-delay-go-no-go.md): relative useful-prototype criteria and limitations. Original numerical findings are retained; this report adds the absolute-budget interpretation.
- **[R6]** [Delay experiment and correction implementation](../experiments/delay_screen.py): `ObservationView`, `VoltageChannel`, and `select_setting`. The simulator is reused OpenG2G; the correction's established estimation precedent is discussed in [R1].
