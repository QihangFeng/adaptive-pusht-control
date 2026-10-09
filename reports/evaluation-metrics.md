# Agreed evaluation metrics

Compare controllers using these four metrics. Improvements are assessed against the same baseline, workload, grid and simulation duration; a minimum throughput target is not required for this comparison.

| Metric | Definition | Unit | Better direction |
|---|---|---|---|
| Voltage violation fraction | Total time with any monitored voltage outside the limits, divided by total simulated duration | % | Lower |
| Average simulated throughput | Time-weighted average of the aggregate modeled tokens/second | Tokens/s | Higher |
| Average modeled inter-token latency | Time-weighted average of recorded modeled fleet inter-token latency | ms | Lower |
| Accumulated violation severity | Sum of voltage distances outside the limits, multiplied by timestep duration, across all monitored bus-phases and timesteps | pu·s | Lower |

## Shared definitions

Use the current nominal voltage band, **0.95–1.05 pu**, and the same monitored bus-phases for every controller. A sample is violating when at least one monitored voltage is below 0.95 or above 1.05. Values exactly on either boundary are inside the band.

The denominator is the **simulated duration**, not the laptop's wall-clock execution time. Include the full run, including startup, consistently across methods. Current runs use 300 seconds and equally spaced 0.1-second samples.

### Voltage violation fraction

For equally spaced samples:

- Violation time = number of violating samples × timestep duration.
- Violation fraction (%) = 100 × violation time / simulated duration.

When several grid locations violate simultaneously, that timestep contributes only once to the total system violation time.

### Throughput and latency averages

With equally spaced samples, take the arithmetic mean of the per-step throughput and modeled inter-token-latency values. With unequal timestep durations, weight each value by the duration it represents and divide by total duration.

Throughput is aggregated across the simulated replicas. Latency is the time average of the recorded modeled fleet inter-token latency for the current single-model study; convert seconds to milliseconds when reporting. These quantities describe the trace-based simulation. They do not provide request queueing, complete response-time, or request-level tail-latency measurements.

### Accumulated violation severity

At each monitored bus-phase and timestep, compute:

`distance outside limits = max(0.95 − voltage, 0) + max(voltage − 1.05, 0)`

Multiply by timestep duration and sum over all monitored bus-phases and timesteps. A voltage within the limits contributes zero.

For one location staying at 0.94 pu for five seconds, the contribution is `(0.95 − 0.94) × 5 = 0.05 pu·s`. Staying at 0.90 pu for five seconds contributes `0.25 pu·s`. Thus this metric captures depth and duration even when the violation fractions are identical.

The existing output field `integrated_violation_pu_s` already represents this quantity. Compare it using the same monitored locations and voltage limits; adding more monitored locations can change the sum.

## Reading the comparisons

Report the four values together and explain their trade-offs. Shorter or shallower voltage violations can be useful even if throughput changes slightly; throughput gains can be useful at comparable voltage performance. Do not require every metric to improve simultaneously or combine them into an arbitrary weighted score.

The earlier reports' experimental-budget pass/fail labels remain a separate description of absolute requirements. They do not replace the four comparative metrics agreed here. Minimum and maximum voltage and batch switching can remain supporting diagnostics.

## Implementation references

- [Existing delay experiment](../experiments/delay_screen.py): per-step voltage, modeled latency and throughput traces, violation duration and accumulated severity.
- [Saved-trace audit](../experiments/delay_acceptability_audit.py): independent recomputation of the existing duration, severity and throughput metrics.
- [Latest comparison report](2026-10-08-slow-update-comparison.md): completed comparisons and their original experimental budgets. This document records the agreed metrics; it does not recalculate that historical report.
