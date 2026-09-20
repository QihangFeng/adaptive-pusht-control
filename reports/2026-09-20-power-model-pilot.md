# Power-model sensitivity pilot

## Conclusion

**Keep the question provisionally, but narrow it to limited profiling, power-curve shape, and controller tuning. Do not implement calibration yet.**

The pilot found a conditional sensitivity worth investigating, not a general failure of the existing controller. A fit made without one measured batch size degraded the integrated voltage/throughput trade-off under the development-selected setting in both evaluation scenarios. Offset errors had no effect, and gain errors were usually much less consequential. Results depend strongly on the controller setting and on which grid metric is considered.

No adaptive method was implemented. These results do not show that calibration would outperform robust static tuning, establish a new research gap, or validate a real deployment uncertainty distribution.

## Protocol recorded before running

Purpose: determine whether inaccurate power-response estimates justify developing a calibration controller. This pilot implements no calibration and makes no novelty claim.

- Fixed environment: existing measured Llama-3.1-8B/H100 profile, IEEE 13-bus feeder, one data center at bus 671, 500 kW base load per phase, 4,800 virtual-GPU capacity, 300 seconds, 0.1-second simulation steps, one-second control updates, 80 ms model-average ITL target.
- Controller-only variants: nominal fitted power curve; +50 W offset; 0.9/1.1 multiplicative gains; fit omitting batch 64. The omitted middle-range point is selected before any control results. Other six batch measurements train that fit. Latency/throughput fits and plant traces are unchanged.
- Gain/offset errors are synthetic sensitivity tests, not measured deployment uncertainty. Omitting one batch represents limited profiling within this dataset, not an independently measured distribution shift.
- Five nominal-fit configurations are compared on a separate development ramp (2,400 to 4,200 replicas and back, seed 0). Reference settings come from the earlier feasibility experiment, not untouched `OFOConfig` defaults. Vary primal step or voltage-term gain by a factor of two, one at a time.
- Selection: highest development throughput among configurations with integrated voltage violation <=0.05 pu*s, minimum voltage >=0.94 pu, maximum <=1.06 pu, and ITL-target miss fraction <=1%. If none qualifies, select lowest integrated violation. These are pilot comparison budgets, not electrical safety guarantees. Persist the choice before evaluation.
- Held-out evaluation: a fast ramp to 4,800 replicas and a two-pulse ramp to 4,500, both seed 1. Run all five fits using reference and selected settings. If they coincide, do not duplicate them. Also run fixed batches 8 and 512 for context. Maximum: 29 full simulations.
- Metrics: integrated voltage deviation, violation time, minimum/maximum voltage, estimated throughput, model-average ITL misses, batch switching, runtime and memory. Compare both sides of the trade-off, not only a relative change in a near-zero voltage metric.
- Interpretation trigger: a >=20% relative AND >=0.01 pu*s absolute increase in integrated violation, or >=5% throughput loss without a compensating grid benefit, merits a closer look. This is an exploratory screening rule, not a statistical significance test. Do not increase perturbations after observing weak effects merely to force a failure.
- Sanity checks: all power-flow solves converge; values are finite; batches are supported; independent metric calculations agree; prepared input hashes remain unchanged; constant-offset trajectories exactly match nominal ones for each scenario/configuration.
- Scope: one measured profile, one evaluation seed, two related load scenarios, a coarse five-point development sweep. This cannot establish broad robustness or whether calibration beats a thoroughly tuned baseline.

## Results

### 1. What was tested

The complete suite contains 29 distinct runs: five development configurations, 20 controller/fit/scenario evaluations, and four fixed-batch references. The fixed references establish the endpoints of the service/grid trade-off, not competitive adaptive baselines.

The five-point development sweep selected `half_voltage_gain`: primal step 0.1 and voltage-gradient scale 500,000, instead of the reference's 1,000,000. Its development throughput was 24.793 versus 24.385 million simulated tokens/s, while integrated voltage violation rose from 0.02373 to 0.03676 pu*s, still inside the declared 0.05 budget. Thus, selection intentionally preferred throughput within that budget; it did not identify a universally superior or robust setting.

All configurations in this pilot had zero model-average ITL-target violations. This means latency was nonbinding in these cases, not that request-level tail latency is guaranteed. On held-out abrupt-load evaluation, both reference and selected settings exceeded the development voltage budget. No retuning was performed using those evaluation results.

### 2. Size and meaning of the model differences

| Controller power fit | Maximum power error against the seven benchmark means | Gradient difference from nominal fit | Interpretation |
|---|---:|---:|---|
| Nominal, all seven batches | 1.14% | 0% | Baseline fit, not perfect knowledge of the replay plant. |
| Nominal +50 W | 10.61% | 0% | Prediction level changes, derivative does not. |
| Nominal multiplied by 0.9 | 10.85% | 10% | Synthetic gain sensitivity. |
| Nominal multiplied by 1.1 | 11.25% | 10% | Synthetic gain sensitivity. |
| Refit without batch 64 | 3.20% | 15.34% | Limited profiling from six measured batches, with the omitted batch retained in the plant. |

Gradient difference is the relative L2 difference on 121 equally spaced points in log2(batch) from 3 to 9. It is a difference from the nominal fit, **not an error against a measured ground-truth derivative**. Maximum power error combines training-point and omitted-point errors; it is not an independent workload-generalization score.

All seven possible single-batch omissions are characterized in the manifest. Batch 64 was chosen before simulation outcomes, not because it produced the largest control degradation. This is a same-dataset profiling exercise, not evidence that an actual software or workload change has this effect.

### 3. Main comparison: omit one profiling point

Lower integrated voltage violation and higher throughput are preferable. Voltage integration sums deviations outside 0.95-1.05 pu across monitored bus phases and time; it is not a percentage of time outside bounds.

| Scenario | Controller setting | Power fit | Integrated violation (pu*s) | Throughput (million simulated tokens/s) | Time outside limits (s) |
|---|---|---|---:|---:|---:|
| Abrupt ramp | Reference | Nominal | 0.05735 | 13.969 | 3.9 |
| Abrupt ramp | Reference | Omit batch 64 | 0.06007 | 13.892 | 3.9 |
| Abrupt ramp | Development-selected | Nominal | 0.06684 | 16.394 | 5.2 |
| Abrupt ramp | Development-selected | Omit batch 64 | 0.10771 | 15.724 | 5.0 |
| Two pulses | Reference | Nominal | 0.04701 | 14.841 | 15.3 |
| Two pulses | Reference | Omit batch 64 | 0.05811 | 18.437 | 12.1 |
| Two pulses | Development-selected | Nominal | 0.06643 | 18.978 | 20.6 |
| Two pulses | Development-selected | Omit batch 64 | 0.09342 | 17.798 | 11.2 |

Under the development-selected setting, omitting batch 64 increased integrated violation by **0.04087 pu*s (61.15%)** in the abrupt scenario and **0.02699 pu*s (40.62%)** in the two-pulse scenario. Throughput fell **4.09%** and **6.22%**, respectively. Both satisfy the predeclared integrated-violation screening trigger.

However, violation duration decreased in both comparisons, especially in the two-pulse case. The latter also had a worse minimum voltage: 0.94088 rather than 0.94235 pu. These are mixed outcomes across grid metrics, not evidence that every measure deteriorated. The sizeable percentages also start from small integrated values; the absolute changes above must accompany them.

The reference controller tells a different story: the omitted-point fit has only a small effect under the abrupt ramp, and under two pulses it increases both integrated violation and throughput (the latter by 24.23%). An inaccurate fit can therefore change the operating trade-off rather than uniformly make control worse. A matched-fit run is not an optimality benchmark.

### 4. Offset and gain controls

- The +50 W offset produced **exactly the same saved trajectories** as the nominal fit in all four scenario/controller pairs. This matches the derivative-based update and confirms that ordinary prediction error alone is not a sufficient diagnostic.
- Under the reference setting, +/-10% gain errors did not change the abrupt-scenario summary metrics. Under two pulses, their throughput changes were between -0.75% and +1.44%.
- Under the development-selected setting, gain effects were larger but mixed. For example, gain 0.9 under two pulses reduced throughput by 6.06%, barely increased integrated violation, and substantially reduced violation duration. Do not describe it as uniformly worse grid performance.
- A constant multiplier of the power derivative enters the voltage-gradient term like a gain adjustment. These tests therefore reinforce the need for a tuned static baseline; they do not by themselves justify online calibration.

### 5. Feasibility and implementation checks

Fixed batch 8 had zero voltage violations in both evaluation scenarios, but only 3.62-3.71 million simulated tokens/s. Fixed batch 512 provided 35.08-36.05 million tokens/s with much larger integrated violations (8.64 and 3.36 pu*s). A voltage-feasible operating point therefore exists in these sampled scenarios; the challenge is preserving service while responding to changing load.

Each run completed 3,000 steps, all power-flow solves converged, and finite-value/batch-range checks passed. The driver independently checked its voltage integration and throughput against upstream metrics. Plant-input hashes were unchanged, and offset-control trajectory hashes matched nominal ones. Saved traces allow the reported metrics to be audited.

The full 29-case suite was repeated after fixing process-tree memory accounting. All 29 numerical trajectory hashes matched exactly across the two suites. The repeat is a reproducibility check with the same seeds, **not additional independent statistical evidence**.

Verified per-run wall time was 5.65-7.60 seconds; sampled peak process-tree RSS was 139.28-141.90 MiB. Summed child-run wall time for the 29-case verified suite was 174.82 seconds, excluding parent fitting/orchestration overhead. No 180-second or 3.5-GiB per-run limit was hit. This confirms the reduced pilot is practical on the laptop; it does not predict the cost of larger studies.

Four unit tests and one saved-artifact audit passed (`5 passed`). The artifact audit checks all 29 compressed traces, summary metrics, batch support, repeatable tuning selection, and the recorded driver hash.

Two implementation issues were corrected transparently: an initial simulation could not export to a relative output path after OpenDSS changed its working directory; output paths are now absolute. The first complete suite's RAM sampler observed only the Windows venv launcher, so its approximately 4-MiB readings are invalid for workload memory. The repeated suite measures launcher plus recursive children. Initial artifacts are retained separately and are not the authoritative resource evidence.

## Recommendation after the pilot

The more defensible provisional question is now:

> How do limited GPU profiling and power-curve gradient errors affect the voltage-throughput trade-off of grid-aware LLM batch control, and does measurement-based correction add value beyond robust static tuning?

Before implementing correction:

1. Repeat the critical nominal/omitted-fit comparisons on additional seeds and development schedules. Use a small Pareto comparison across static controller settings rather than treating the single selected setting as best overall.
2. If the effect persists, check another measured profile or additional profiling omissions. Separate limited-data effects from simulator/model-specific behavior.
3. Only then choose a minimal correction method and compare it against the robustly tuned static baseline. An affine correction cannot generally repair a curve-shape error; do not assume the earlier two-parameter proposal is sufficient.

This is an initial screening result, not a justification for a large experiment campaign. One evaluation seed, one model/hardware profile, two related load scenarios, nonbinding latency constraints, and five nominal-only tuning candidates limit the conclusion. The recommendation is to keep investigating this narrow effect, not to declare calibration necessary or superior.

## Artifacts and reproduction

- Driver: [power_model_pilot.py](../experiments/power_model_pilot.py).
- Tests: [test_power_model_pilot.py](../tests/test_power_model_pilot.py).
- Authoritative repeated suite: [summary](../results/power-model-pilot-verified/summary.json), [manifest](../results/power-model-pilot-verified/manifest.json), and [frozen development selection](../results/power-model-pilot-verified/selection.json).
- Each run also has its input job, full OFO configuration, compressed numerical trajectory, resource measurement, and log. The manifest records source/data revisions, input hashes, fit parameters, all profiling-omission diagnostics, package versions, and the exact driver hash.

Run from the project directory, using the existing validated environment and prepared data:

```powershell
$pilotRoot = 'C:\Users\冯同学\AppData\Local\Temp\ece730-feasibility-580bb4f1'
& "$pilotRoot\env-grid\Scripts\python.exe" experiments/power_model_pilot.py suite `
  --openg2g-root "$pilotRoot\openg2g" `
  --prepared-data "$pilotRoot\measured-results\specs" `
  --fit-check "$pilotRoot\measured-results\fit-check.json" `
  --output results/power-model-pilot-rerun
& "$pilotRoot\env-grid\Scripts\python.exe" -m pytest -q -p no:faulthandler tests/test_power_model_pilot.py
```

The output directory must be new/empty. For the artifact test to audit a different output directory, set `PILOT_RESULTS` to that path before running pytest. The `no:faulthandler` option avoids the previously observed Windows/OpenDSS native floating-point exception diagnostics; it does not disable Python assertions, solver checks, or pytest failures.

No packages were installed and the project `.venv` was not modified. This pilot used Python 3.11.16 in the already-tested environment, not the new Python 3.11.9 `.venv`. NumPy 2.4.6, mlenergy 0.1.0, mlenergy-data 0.4.0, and OpenDSSDirect.py 0.9.4 were used; the full version inventory is in the manifest.

This is reproducible with the existing local cache, not yet a self-contained fresh-machine package. On another machine, obtain the pinned OpenG2G checkout and regenerate the selected measured profile using its [data pipeline](https://gpu2grid.io/openg2g/guide/data-pipeline/) and the dataset access already obtained. The temporary runtime/cache should be migrated to stable storage before a longer project. No raw gated measurement files or LLM weights were added to this repository.

## Sources

- [Research question review](2026-09-20-research-question-review.md): literature, assumptions, and proposed question.
- [OpenG2G OFO implementation](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/controller/ofo.py): reused controller, fitted derivatives, and tuning parameters.
- [OpenG2G offline backend](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/datacenter/offline.py): reused measured-trace plant.
- [GPU-to-Grid paper](https://arxiv.org/html/2602.05116v1): underlying batch-control formulation.
- [ML.ENERGY Benchmark v3](https://huggingface.co/datasets/ml-energy/benchmark-v3/tree/fea6cb99d037d11f86f9c45a6991b9e5764f4529): measured source data; no model weights or raw gated measurements are added to the project.
