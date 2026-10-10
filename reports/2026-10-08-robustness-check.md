# Robustness check: changing measurement delays and outside-grid demand

Date: October 8, 2026.

Status: all four compatibility checks and 72 evaluation runs completed. The design was fixed before evaluation. All 42 project tests and 52 subtests passed; none were skipped.

## Conclusion

**The voltage benefit survives variable delay with unchanged outside demand, but does not consistently survive independent outside-demand changes.** Keep the topic, but do not describe the current correction as universally better or robust against arbitrary grid changes.

Against the tuned old-reading baseline, the counts below cover two workloads, two delay conditions and three new seeds per background condition. There are no ties. These are counts of the tested cases, not probabilities of success in deployment.

| Background | Lower accumulated severity | Less violation time | Higher throughput | Lower modeled ITL |
|---|---:|---:|---:|---:|
| Unchanged outside demand | 12/12 | 12/12 | 6/12 | 6/12 |
| Added independent demand changes | 7/12 | 6/12 | 10/12 | 2/12 |

The strongest counterexample is the ramp workload with constant two-second delay and added outside demand: in **all three seeds**, correction gives worse voltage metrics, higher throughput and higher modeled ITL. This is a real limitation and trade-off, not a smaller version of the earlier voltage improvement.

The appropriate current claim is: **power-history correction reduces voltage violations in the tested unchanged-background conditions, including variable delay; outside-grid changes can reverse that advantage.** A higher-throughput result alone does not establish better overall control.

## Question

Does the existing power-history correction still improve the voltage–AI-performance trade-off when the age of voltage information changes and electricity demand elsewhere on the grid changes independently?

Keep the existing method and both comparator settings unchanged. This tests the assumptions behind the promising fixed-delay, fixed-background result; it is not another parameter search or a new controller.

## Bounded experiment

The experiment used the pinned OpenG2G source and prepared measured H100 inference profiles, IEEE 13-bus grid, datacenter at bus 671, 300 simulated seconds and 0.1-second physical/sensor samples. Both `abrupt` and `heldout_ramps` workloads ran with additional seeds 3, 4 and 5.

| Component | Fixed design |
|---|---|
| Tuned old-reading baseline | `hold`, frozen `half_primal_dual`, one-second decisions |
| Slower old-reading alternative | `hold`, frozen `slow_update`, two-second decisions; also different tuning |
| Our method | `predict`, frozen `half_primal_dual`, one-second decisions; unchanged sensitivity correction |
| Constant transport delay | Two seconds per voltage packet |
| Variable transport delay | Twenty-five 12-second blocks: first block two seconds; remaining blocks are a separately seeded permutation of eight each at 0.5, 2 and 3.5 seconds |
| Fixed background | Original static feeder loads, no additional load |
| Changing background | Additional demand at bus 675, independent of the datacenter's actions |

This is a 2 × 2 comparison of constant/variable delay and fixed/changing background. With two workloads, three seeds and three methods, there are **72 evaluation runs**. Four preliminary compatibility runs reproduced saved trajectories exactly for fresh hold, delayed tuned hold, correction and slower hold before evaluation started.

The added load is a square pulse from 75 to 95 seconds at **50 kW per phase**, followed by a trapezoid rising from 150 to 155 seconds, holding until 165 seconds, then returning to zero by 170 seconds at a peak of **100 kW per phase**. These are **150 and 300 kW aggregate**, because the pinned backend applies the returned power to each of three phases. Its power factor is fixed at 0.96, with per-phase reactive demand `Q = P × tan(acos(0.96))`. The pulse profile is identical across paired methods and is never given to their controller.

The external-load test adds energy demand as well as variation; it is not a matched-mean background comparison. These synthetic delay and demand profiles are declared sensitivity cases, not measured production distributions. Their sizes will not be increased merely to obtain a favorable result.

## Evaluation and correctness

Use our [four agreed metrics](evaluation-metrics.md): violation time / simulated duration, average simulated tokens/s, average modeled inter-token latency, and accumulated voltage violation severity. Report paired changes, seed ranges and trade-offs; no new absolute throughput or latency threshold is introduced.

The variable profile has an offered mean transport delay of two seconds, but that does not guarantee a two-second age for the newest usable sample. Report actual mean/max sample age as supporting diagnostics. Older packets arriving late must not replace newer data.

Verification included exact historical trajectory compatibility, unchanged source and prepared inputs, identical offered delays and outside loads within each paired case, independent replay of packet arrivals, historical power/voltage consistency, unchanged correction arithmetic, converged power flows and independent recomputation of all four metrics. Changing-load runs also read back all three applied phase demands from OpenDSS. Historical voltage checks compare the recorded physical extrema; full per-timestep bus-phase physical snapshots are not independently reconstructed by the trace audit.

## Complete four-metric comparison

Values are means across seeds 3, 4 and 5, including startup. The [saved summary](../results/robustness-check/summary.json) retains individual runs, paired changes and each group's minimum, maximum and sample standard deviation. Do not interpret the means alone as consistency across seeds. "Outside" means the additional bus-675 demand profile above; "none" retains the original static feeder loads.

| Workload | Delay | Outside changes | Method | Violation fraction, % ↓ | Throughput, million tokens/s ↑ | Modeled ITL, ms ↓ | Severity, pu·s ↓ |
|---|---|---|---|---:|---:|---:|---:|
| Abrupt | Constant | None | Tuned | 2.28 | 10.120 | 11.38 | 0.20819 |
| Abrupt | Constant | None | Slower | 2.97 | 10.167 | 11.29 | 0.23938 |
| Abrupt | Constant | None | Corrected | 0.99 | 14.994 | 14.28 | 0.05029 |
| Abrupt | Variable | None | Tuned | 2.29 | 12.747 | 12.70 | 0.25140 |
| Abrupt | Variable | None | Slower | 2.27 | 10.521 | 11.45 | 0.23025 |
| Abrupt | Variable | None | Corrected | 0.99 | 14.775 | 14.01 | 0.05024 |
| Abrupt | Constant | Outside | Tuned | 2.61 | 9.811 | 11.39 | 0.20830 |
| Abrupt | Constant | Outside | Slower | 3.12 | 9.534 | 11.24 | 0.24041 |
| Abrupt | Constant | Outside | Corrected | 2.46 | 11.571 | 12.07 | 0.06657 |
| Abrupt | Variable | Outside | Tuned | 3.09 | 10.708 | 11.72 | 0.25891 |
| Abrupt | Variable | Outside | Slower | 2.69 | 10.157 | 11.44 | 0.23348 |
| Abrupt | Variable | Outside | Corrected | 2.56 | 11.732 | 12.06 | 0.07046 |
| Ramps | Constant | None | Tuned | 3.24 | 24.283 | 31.53 | 0.16918 |
| Ramps | Constant | None | Slower | 3.59 | 24.206 | 31.25 | 0.18120 |
| Ramps | Constant | None | Corrected | 2.76 | 24.092 | 31.27 | 0.05799 |
| Ramps | Variable | None | Tuned | 3.12 | 24.262 | 31.51 | 0.14904 |
| Ramps | Variable | None | Slower | 3.41 | 24.136 | 31.09 | 0.15514 |
| Ramps | Variable | None | Corrected | 2.76 | 24.092 | 31.27 | 0.05799 |
| Ramps | Constant | Outside | Tuned | 4.16 | 12.281 | 12.56 | 0.02738 |
| Ramps | Constant | Outside | Slower | 4.41 | 12.423 | 12.54 | 0.02796 |
| Ramps | Constant | Outside | Corrected | 4.51 | 19.017 | 17.40 | 0.05705 |
| Ramps | Variable | Outside | Tuned | 4.80 | 16.384 | 15.08 | 0.07769 |
| Ramps | Variable | Outside | Slower | 4.21 | 14.717 | 13.79 | 0.04431 |
| Ramps | Variable | Outside | Corrected | 4.52 | 18.837 | 17.08 | 0.05082 |

Against the slower alternative, both voltage metrics also improve in all 12 unchanged-background cases. With outside changes, severity improves in seven of 12 and violation time in seven of 12; throughput rises in all 12, but modeled ITL also rises in all 12. This remains an alternative-policy comparison, not a correction-only ablation.

### Counterexamples and variability

- **Ramps, constant delay, outside changes:** average severity rises from 0.02738 to 0.05705 pu·s and violation fraction from 4.16% to 4.51%. Throughput rises from 12.281 to 19.017 million tokens/s, but modeled ITL rises from 12.56 to 17.40 ms. The same directions occur in all three seeds.
- **Ramps, variable delay, outside changes:** the mean severity looks better, but the individual tuned comparisons are worse in seeds 3 and 4 (+3.1% and +92.9%) and better in seed 5 (−73.0%). The favorable seed pulls down the mean; this is not consistent robustness.
- **Abrupt, constant delay, outside changes, seed 4:** severity improves, but violation fraction rises from 2.57% to 2.77%. Less accumulated severity does not necessarily mean shorter violations.
- **Abrupt, variable delay, outside changes, seed 3:** severity improves, but throughput falls by 3.7%. With the ramp variable-delay outside case in seed 5, these are the two disturbed cases with lower throughput than the tuned baseline.

Actual mean measurement age at controller decisions ranges from **1.920 to 1.963 seconds** in the variable-delay runs; every such run reaches a maximum age of 3.5 seconds. Constant-delay runs use two-second-old measurements. Matching the offered mean transport delay therefore does not match the controller's exact age distribution.

## Interpretation and recommendation

The correction uses local power history:

`corrected voltage = old voltage + initial sensitivity × (current local power − old local power)`.

It cannot reconstruct an unseen power change elsewhere on the grid. In the earlier illustrative error decomposition, the unobserved external-change term remains even with perfect local telemetry. This is an established limitation, not a new algorithmic discovery. [Earlier formulation](2026-09-20-delay-go-no-go.md)

There is also a control trade-off. In ramp seed 3 with constant delay and outside demand, the tuned baseline's mean batch is about 70, versus 127 with correction. The corrected run has higher mean datacenter power (3.531 versus 3.357 MW). Those trace diagnostics support the interpretation that correction preserves more AI work while the old-reading controller throttles more. They do not prove that estimation error alone explains every voltage difference. Within the corrected trajectory, correction still reduces voltage-estimation RMSE relative to the raw old packet; a better estimate alone does not guarantee better closed-loop voltage metrics.

**Recommendation:** retain the topic as a conditional robustness study, and retain this simple correction as the current prototype. We now have repeatable benefits and explicit counterexamples. Do not present it as uniformly improving voltage regulation or all AI metrics. Before proposing any new controller, discuss whether our final contribution is to characterize this trade-off or to improve it at comparable voltage performance. No new margin, fallback, tuning search or controller redesign was introduced in this check.

## Verification, resources and reproduction

All 76 simulations completed with converged power flows. All four historical trajectory hashes matched exactly. The pinned upstream source stayed clean; prepared-input and driver hashes remained unchanged. Independent review recomputed all four metrics and checked hashes for all 76 saved traces and all 24 aggregate means.

The final project test suite passed **42 tests and 52 subtests with no skips**. It includes causal-channel, profile/unit, pairing, altered-readback and saved-evidence checks as well as the earlier project tests. This is not a rerun of every upstream OpenG2G test.

Evaluation runs took **6.2–9.9 seconds each**, with **140.0–143.3 MiB** peak sampled process-tree memory. Combined child-run wall time for the compatibility and evaluation runs was **512.7 seconds**, approximately 8.5 minutes. These measurements exclude implementation, review, tests and report writing. No GPU inference or training was run.

```powershell
# From the project root with the persistent setup already restored:
.\.venv\Scripts\python.exe experiments\robustness_check.py --output results\robustness-check-repeat
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests -q
```

The runner refuses populated output directories, freezes its protocol before simulation, and saves an explicitly incomplete summary until every planned run and verification succeeds. Authoritative evidence is in [protocol.json](../results/robustness-check/protocol.json), [summary.json](../results/robustness-check/summary.json) and adjacent per-run job/result/trace files.

## Scope limits

This check does not establish formal system stability, production deployment safety, a real-world delay distribution, request-level latency or behavior under arbitrary model errors. Local power telemetry and the initial sensitivity estimate retain the earlier favorable assumptions. Three additional seeds provide a bounded sensitivity check, not broad statistical generalization.

## Sources

- [Earlier method and fixed-delay evaluation](2026-09-20-delay-go-no-go.md): mathematical correction, reused infrastructure and original settings.
- [Latest comparison](2026-10-08-slow-update-comparison.md): the existing three-method, four-metric result and its trade-offs.
- [Practicality and prior-work review](2026-10-07-telemetry-delay-practicality-review.md): why unseen outside-grid changes are an important limitation and why the assumed architecture remains conditional.
- [Pinned OpenG2G load implementation](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/grid/opendss.py): `attach_load`, per-phase power application and grid-before-controller timing, checked in the local source.
- [Existing delayed-observation adapter](../experiments/delay_screen.py): causal channel, restricted observation interface and frozen controller.
- [Robustness runner](../experiments/robustness_check.py) and [tests](../tests/test_robustness_check.py): predeclared jobs, source/input gates, independent packet replay, paired metrics and saved-evidence checks.
