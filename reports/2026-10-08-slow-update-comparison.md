# Does a slower controller make voltage correction unnecessary?

Date: October 8, 2026

Subsequent metric agreement: use the [four comparative evaluation metrics](evaluation-metrics.md), including accumulated violation severity. The updated comparison below uses those metrics; the original experimental-budget analysis is retained separately as historical evidence.

## Updated four-metric comparison

Each value below is the mean across seeds 1 and 2 for a 300-second run with a two-second measurement delay. All 12 saved traces were checked against their trajectory hashes and stored duration, severity and throughput metrics. Average modeled inter-token latency was calculated directly from the equally spaced trace samples. No new simulations were needed for this table.

| Workload | Method | Violation time / total time, % ↓ | Average throughput, million tokens/s ↑ | Average modeled ITL, ms ↓ | Accumulated severity, pu·s ↓ |
|---|---|---:|---:|---:|---:|
| Abrupt | Tuned old-reading baseline | 2.30 | 10.104 | 11.38 | 0.2065 |
| Abrupt | Slower old-reading controller | 2.95 | 10.146 | 11.28 | 0.1972 |
| Abrupt | Power-history correction | 1.02 | 15.188 | 14.44 | 0.0490 |
| Ramps | Tuned old-reading baseline | 3.27 | 24.304 | 31.52 | 0.1683 |
| Ramps | Slower old-reading controller | 3.30 | 24.195 | 31.17 | 0.1809 |
| Ramps | Power-history correction | 2.80 | 24.115 | 31.30 | 0.0569 |

Correction improves both voltage metrics relative to both alternatives. Against the tuned baseline, abrupt-case severity falls by 76.3% and throughput rises by 50.3%, but modeled ITL rises by 26.9%. Ramp-case severity falls by 66.2%, while throughput falls by 0.8% and ITL falls by 0.7%. This is an improvement in the observed voltage–performance trade-off, not simultaneous improvement in every metric or a guarantee of acceptable voltage.

Throughput is the simulated fleet aggregate; ITL is the time average of modeled time between generated tokens, not request response time. The tuned baseline and correction share the one-second update setting; the slower alternative also changes tuning and updates every two seconds. Sources: [original evaluation traces](../results/delay-prototype/summary.json), [slower-controller traces](../results/slow-update-check/summary.json), and the [saved-trace auditor](../experiments/delay_acceptability_audit.py). The historical budgets below are not additional requirements for claiming comparative improvement.

## Conclusion

**No, slowing the tested controller does not resolve the problem in these four evaluation cases. But correction is still not a complete solution.**

- The frozen slower controller using two-second-old readings fails the experimental voltage budgets in **all four** cases.
- Correction reduces accumulated voltage violations by **67.5–79.3%** relative to that alternative. It processes approximately **48–51% more tokens** in the abrupt cases and approximately **0.3–0.4% fewer** in the ramp cases, with one or two additional batch changes.
- Correction meets all the experimental voltage and modeled-latency budgets in **one of four** cases. That case also passes the predeclared throughput-preservation and switching checks. The other three still fail the minimum-voltage requirement.

The new result closes one important gap: the earlier improvements are not explained away by this already-qualified slower-update alternative. It does **not** show that every simpler controller fails, establish reliable voltage compliance, or prove that the assumed delay is common in real AI centers.

**Recommendation:** retain the topic as a focused robustness study. Do not advertise the current correction as a proven safe controller. Before adding a new mechanism, understand why the fresh-feedback reference also crosses the minimum-voltage limit and then assess the correction under variable delay and independent grid changes. Those are research questions still to answer, not completed validations.

## 1. What was tested—and what was reused

This follow-up implements the four-run comparison specified in the [acceptability audit](2026-10-08-delay-acceptability-audit.md). The protocol was saved before the first new simulation. [R1]

| Item | Fixed choice |
|---|---|
| New baseline | Existing `slow_update` setting, uncorrected two-second-old voltage; controller updates every two seconds |
| Correction comparator | Previously frozen `half_primal_dual` setting with power-history correction; controller updates every second |
| Cases | `abrupt` and `heldout_ramps`, each with seeds 1 and 2 |
| Shared plant/data | Pinned OpenG2G, IEEE 13-bus feeder, same measured GPU profiles, workload schedules and batch options |
| Requirements | Accumulated voltage violation ≤0.10 pu·s, minimum ≥0.94 pu, maximum ≤1.06 pu, modeled 80 ms ITL miss fraction ≤1% |
| Extra-value check | Correction must meet every budget, retain ≥98% of baseline throughput, and use no more than `max(baseline switches + 2, 1.1 × baseline switches)`; if both controllers pass, also require ≥5% throughput gain |

Both controller settings were already selected or identified from development data; neither was retuned using these evaluation outcomes. They differ in tuning/update period as well as correction, so this is an **alternative-policy comparison**, not an isolated causal ablation of correction. The earlier same-setting comparison provides that ablation. [R2]

The four new baseline runs reuse the saved correction runs only after two compatibility simulations reproduce exact saved trajectories, one with fresh feedback and one with two-second delay. Numerical prepared-input hashes match exactly. The generated provenance manifest differs only in its creation timestamp, and Python is now 3.11.9 rather than 3.11.16. Details and reproducible setup are in the [restoration report](2026-10-08-simulator-restoration.md).

## 2. Absolute results

Each case lasts 300 simulated seconds. Throughput refers to virtual replicas using measured H100 profiles, not inference on this laptop. “Pass” means all experimental budgets above, **not** the stricter requirement to remain inside 0.95–1.05 pu at every sample.

| Workload / seed | Controller | Accumulated violation, pu·s | Minimum voltage, pu | Throughput, million tokens/s | Batch changes | Experimental budgets |
|---|---|---:|---:|---:|---:|---|
| Abrupt / 1 | Slower, uncorrected | 0.24361 | 0.936731 | 10.165 | 8 | Fail |
| Abrupt / 1 | Corrected | 0.05044 | 0.937673 | 15.069 | 10 | Fail |
| Abrupt / 2 | Slower, uncorrected | 0.15078 | 0.937568 | 10.127 | 8 | Fail |
| Abrupt / 2 | Corrected | 0.04754 | 0.937568 | 15.307 | 10 | Fail |
| Ramps / 1 | Slower, uncorrected | 0.18065 | 0.938019 | 24.177 | 5 | Fail |
| Ramps / 1 | Corrected | 0.05508 | 0.940053 | 24.105 | 6 | Pass |
| Ramps / 2 | Slower, uncorrected | 0.18120 | 0.938304 | 24.213 | 5 | Fail |
| Ramps / 2 | Corrected | 0.05881 | 0.939684 | 24.126 | 6 | Fail |

Every slower-controller case fails both the accumulated-violation and minimum-voltage budgets. Its time with any nominal-band violation is 8.9, 8.8, 9.9 and 9.9 seconds in the table's case order. None of the slower runs has an upper-voltage-budget failure or modeled-ITL miss. [R1]

The corrected cases all pass the accumulated-violation budget, but three still fail the minimum-voltage budget. Ramp seed 2 is close to the threshold, yet 0.939684 is below 0.94 and must count as a failure. Reporting individual seeds prevents this from being hidden by rounding or averaging. The 80 ms modeled-ITL constraint is nonbinding for both methods; request-level latency remains unevaluated. [R1–R2]

## 3. Does the correction add value?

| Case | Less accumulated violation with correction | Throughput change versus slower controller | Extra batch changes | Decision under the stated rule |
|---|---:|---:|---:|---|
| Abrupt / 1 | 79.3% | +48.24% | +2 | Both fail absolute budgets; improved but not an acceptable solution |
| Abrupt / 2 | 68.5% | +51.16% | +2 | Both fail absolute budgets; improved but not an acceptable solution |
| Ramps / 1 | 69.5% | −0.30% | +1 | Only correction passes; additional-value check passes |
| Ramps / 2 | 67.5% | −0.36% | +1 | Both fail absolute budgets; improved but not an acceptable solution |

All four comparisons preserve at least 98% of slower-controller throughput and stay within the switching cap. Only one passes the full added-value rule because **absolute acceptability comes first**. No case has both controllers passing, so the proposed 5% throughput-gain threshold does not determine any outcome here.

Lower accumulated violation is useful evidence, but it is not equivalent to “voltage is always acceptable.” In particular, the minimum voltage in each corrected evaluation case matches the fresh-feedback reference to the displayed precision. The fresh reference also passes only ramp seed 1. The residual problem is therefore not exclusively caused by measurement delay in these runs. This does not prove physical infeasibility or rule out a different controller, reserve margin, or operating point. [R2]

## 4. What is now settled, and what remains open?

**Settled within this experiment:**

- The current laptop can run this evaluation economically without its GPU.
- The environment is restored reproducibly enough to match two complete historical trajectories and all numerical input files.
- The frozen slower stale-feedback alternative does not meet the selected evaluation budgets. Correction improves the observed trade-off relative to it, with one qualifying case.

**Still open:**

- Why the remaining transient dips also occur with fresh feedback, and whether the desired requirements can be met at acceptable throughput with the available batch-size actions.
- Whether other simple policies or a fresh local-voltage architecture avoid the need for correction.
- Whether benefits survive independent outside-grid changes, varying measurement age, imperfect sensitivity, and less-ideal local power telemetry.
- Whether throughput meets an actual service-demand target and whether request-level latency is acceptable.
- How often the assumed remote-measurement scenario occurs in practice; the earlier literature review remains conditional. [R3]

The next step should clarify those mechanisms and assumptions, not enlarge delays or relax thresholds to obtain a positive result. Any new margin/filter/control design should be discussed before implementation.

## 5. Verification and reproduction

Six new simulations completed: two exact compatibility checks plus four missing baseline cases. Prepared inputs and source remained unchanged. Every new trace passed independent metric, hash and causal-observation checks. **32 project tests and 25 subtests passed; no tests were skipped.**

The simulations used approximately 139–141 MiB peak sampled process-tree memory and 5.8–8.9 seconds each. Their combined child-run wall time was approximately 40.4 seconds. These are simulation-only figures; setup and data preparation are excluded.

```powershell
# From the project root after restoring the environment and data:
.\.venv\Scripts\python.exe experiments\slow_update_check.py --output results\slow-update-check-repeat
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests -q
```

The wrapper refuses nonempty output directories, preserves the original drivers, and stops before the four comparisons if either historical compatibility check fails.

## References

- **[R1]** New evidence: [frozen protocol](../results/slow-update-check/protocol.json), [compatibility checks](../results/slow-update-check/compatibility.json), [summary and per-case comparisons](../results/slow-update-check/summary.json), [runner](../experiments/slow_update_check.py), [tests](../tests/test_slow_update_check.py). Per-run compressed traces are stored alongside the summary.
- **[R2]** [Acceptability audit](2026-10-08-delay-acceptability-audit.md), [saved evaluation results](../results/delay-prototype/summary.json), and [development selection](../results/delay-screen/selection.json).
- **[R3]** [Telemetry-delay practicality review](2026-10-07-telemetry-delay-practicality-review.md): motivation, existing prediction/control precedent and deployment uncertainty. No new literature search was needed for this bounded experimental comparison.
- **[R4]** [Restoration report](2026-10-08-simulator-restoration.md): pinned source/data references, timestamp treatment, environment differences and setup instructions.
