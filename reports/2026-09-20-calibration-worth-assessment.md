# Is power-model calibration worth making our core contribution?

## Decision

**Do not commit to the original simple calibration method as the final project contribution yet.** Keep the AI-datacenter/grid-control direction, but treat calibration as an unproven candidate. This review and a completed static-tuning follow-up weaken its current justification.

Three findings drive this assessment:

1. Existing feedback already reacts to measured voltage. A power prediction error does not automatically cause a control failure.
2. The proposed affine power correction mostly overlaps with an existing tuning parameter. It cannot generally repair a wrong curve shape.
3. A fixed controller setting chosen using development data removed the nominal-versus-incomplete-fit difference in both new abrupt-load runs. It also improved the integrated-voltage/throughput trade-off over the earlier selected setting in every incomplete-fit comparison, although it increased switching and sometimes violation duration.

This is not evidence that adaptation can never help. It is evidence that our current pilot does not establish a need for it. No adaptive controller was implemented or evaluated.

## 1. What prior work already provides

| Source | Evidence relevant to our idea | Implication |
|---|---|---|
| [GPU-to-Grid, 2026](https://arxiv.org/html/2602.05116v1), Sections III-B/C | Uses measured voltage/latency feedback and batch-dependent derivatives to update batch sizes; explicitly discusses robustness to modeling imperfections. | We must not describe the baseline as blindly trusting power predictions. |
| [Picallo et al., 2022](https://arxiv.org/html/2110.00954v5), Section III | Combines online feedback optimization with recursive sensitivity estimation; learning requires informative changes in control inputs. | Adding an online estimator is established methodology. The grid sensitivity estimated there is not the GPU power curve, so this is close methodological precedent rather than an identical LLM application. |
| [Zagorowska et al., 2026 preprint](https://arxiv.org/html/2604.12863v1), Sections 2-3 | Adapts scaling and step size for a different OFO formulation, demonstrated on process-control examples. | Adaptive tuning is also related work. Its algorithm and guarantees are not directly interchangeable with OpenG2G's discrete primal-dual controller. |
| [OpenG2G, 2026](https://arxiv.org/html/2605.05519v1), Sections 3-4 and Appendix K | Provides trace-based workloads, interchangeable controllers, and grid/service metrics. The default grid backend models quasi-steady voltage. | A suitable reused testbed, not our contribution. Results here are not production serving or transient-stability guarantees. |
| [ML.ENERGY Benchmark, 2025](https://arxiv.org/html/2505.06371v1), Section 4 and Appendix B | Shows that energy outcomes depend on model, hardware, task and serving configuration. | Supports studying modeling assumptions, but does not establish that our particular omitted-profile error represents a real time-varying deployment change. Energy variation is not itself evidence of a changed power-curve slope. |

The focused search did not establish publication-level novelty for our proposed GPU-specific extension. It is not an exhaustive literature review. A carefully adapted method can still be a good course contribution, but its practical benefit needs evidence.

## 2. Why the original mathematical correction is weak

The earlier candidate was

\[
\widehat p_t(x)=a_t p_0(x)+c_t,\qquad x=\log_2(b),
\]

where the controller learns a multiplier and an offset from power observations.

The inspected [pinned OFO implementation](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/controller/ofo.py) uses the power derivative in its voltage-gradient term. Consequently,

\[
\frac{d\widehat p_t}{dx}=a_t\frac{dp_0}{dx},
\qquad
\beta\frac{d\widehat p_t}{dx}=(\beta a_t)\frac{dp_0}{dx},
\]

where \(\beta\) is the existing voltage-gradient scale. For this single-model controller, with everything else held fixed, a constant multiplier is equivalent in that term to rescaling the controller gain. The offset disappears entirely. This concerns the controller's estimated curve, not a physical change in the plant.

An online-changing multiplier could still be useful when the response really changes over time; a single fixed gain need not reproduce that behavior. But our experiments have not tested such a change. An affine correction also cannot generally repair a saturation region appearing at the wrong batch size.

In plain language: we proposed teaching the controller a better electricity guide, but part of that teaching just turns an adjustment knob the controller already has.

## 3. Completed follow-up: give static tuning a fairer chance

### Protocol

Used the unchanged pilot simulator, same measured profile and same five existing parameter settings. No new error magnitude, physical model, calibration, dependency installation, or GPU inference was introduced.

- Reused five completed nominal-fit development runs, checking the original driver and prepared-data hashes.
- Added five development runs with the preselected omitted-batch-64 fit.
- Required both development fits to satisfy the original voltage/ITL screening budgets. Selected the setting with the highest worst-fit throughput. The fallback was minimum worst-fit integrated violation; it was not needed.
- Recorded the protocol before new simulations and froze the selection before new evaluation.
- Evaluated reference, earlier nominal-selected, and newly selected settings with both fits, two existing load schedules and new seeds 2/3: 24 evaluation runs.
- Total: **29 new simulations**, not a repeat of the original suite.

The new setting was `double_step`: primal step 0.2, voltage-gradient scale 1,000,000. The earlier nominal-only selection was `half_voltage_gain`: step 0.1, scale 500,000. These labels describe fixed parameters, not new algorithms. The new selection used knowledge of the limited-profiling uncertainty during development; it is not a test of unseen error types.

### Main results

The table uses the incomplete power fit throughout. Values are arithmetic means over seeds 2 and 3, not confidence intervals. Integrated violation measures accumulated voltage excursion severity over bus phases and time; lower is better. Throughput is the simulator's token-rate estimate; higher is better.

| Load schedule | Fixed controller setting | Integrated violation (pu*s) | Throughput (million tokens/s) | Time outside voltage limits (s) | Batch switches |
|---|---|---:|---:|---:|---:|
| Abrupt load | Reference | 0.06135 | 13.801 | 4.00 | 17.0 |
| Abrupt load | Earlier selection | 0.11687 | 14.598 | 5.95 | 18.5 |
| Abrupt load | New development selection | 0.06683 | 18.681 | 6.15 | 34.0 |
| Two pulses | Reference | 0.04810 | 17.375 | 9.45 | 20.5 |
| Two pulses | Earlier selection | 0.08304 | 17.409 | 10.40 | 19.0 |
| Two pulses | New development selection | 0.06154 | 20.940 | 17.15 | 35.5 |

Compared with the earlier selection under the same incomplete fit, the new fixed setting reduced integrated violation by **23-43%** and increased throughput by **16-28%** across the four individual evaluation cases. However, it switched more and spent longer outside voltage limits in these comparisons. It is not uniformly better, and the reference setting retained lower integrated violations with less throughput.

For the new setting, nominal and incomplete fits produced **identical saved numerical trajectories in both abrupt-load seeds**. In the two-pulse cases the fit still changed behavior: mean integrated violation rose from 0.05433 to 0.06154, while throughput also rose from 18.942 to 20.940 million tokens/s. That is a different trade-off, not uniform deterioration from an inaccurate model.

All evaluation runs had zero model-average ITL target misses. Latency was nonbinding; this is not a per-request or tail-latency guarantee. The new setting still exceeded the development integrated-voltage budget on evaluation, and abrupt cases reached voltages below 0.94 pu. No electrical safety guarantee follows.

### Scope and checks

This is exploratory: one profile, one development schedule, one omitted point and five settings. The evaluation schedules were already known from the pilot; new seeds are not independent workload types. We have not exhausted static tuning, measured a realistic temporal change, or tested adaptation. There are only seven supported batch sizes in this reduced case, so we must also justify why obtaining a complete offline profile would be impractical before claiming that online learning is necessary.

All power-flow solves converged and prepared input hashes remained unchanged. Saved-trace audits independently verified hashes, finite values, throughput, integrated voltage violation, violation duration and voltage extrema. The original and follow-up tests passed: **7 passed**. Per-run wall time was 5.33-9.57 seconds, sampled process-tree peak RAM 138.61-141.03 MiB; summed child-run wall time was 170.38 seconds, excluding parent overhead. Laptop compute remains feasible.

## 4. Recommendation and decision boundary

For a method-centered course project, **do not proceed directly to the two-parameter calibration controller**. Its current justification is weaker after this test. Neither a poor baseline setting nor an artificially enlarged error should be used to create an apparent research gap.

If we retain calibration, the defensible question would be narrower:

> Can measurement-based correction of batch-dependent response errors improve the voltage-throughput trade-off beyond uncertainty-aware static tuning, without excessive batch switching or full re-profiling?

This is a proposed question, not a demonstrated gap. It needs a measured or otherwise defensible source of curve-shape uncertainty and a reason offline profiling is insufficient. A real workload-change study would also need coherent latency/throughput changes, rather than changing power alone and calling it realistic.

The next decision should therefore be between **reformulating this candidate with that evidence** and **reviewing the previously identified delayed-feedback candidate**. Do not start a more complex estimator merely to preserve the original idea. Any eventual adaptive method must face the fixed-tuning alternatives above, appropriate established estimation baselines, equal tuning budgets, and matched grid/service/switching trade-offs.

## Artifacts and reproduction

- [Follow-up driver](../experiments/static_tuning_check.py)
- [Selection and artifact tests](../tests/test_static_tuning_check.py)
- [Pre-run protocol](../results/static-tuning-check/protocol.json)
- [Frozen development selection](../results/static-tuning-check/selection.json)
- [Complete results](../results/static-tuning-check/summary.json), with per-run jobs, logs and compressed traces in the same directory.
- [Original pilot report](2026-09-20-power-model-pilot.md)

Using the previously validated temporary runtime/cache, from the project root:

```powershell
$checkRoot = 'C:\Users\冯同学\AppData\Local\Temp\ece730-feasibility-580bb4f1'
& "$checkRoot\env-grid\Scripts\python.exe" experiments/static_tuning_check.py `
  --openg2g-root "$checkRoot\openg2g" `
  --prepared-data "$checkRoot\measured-results\specs" `
  --prior-results results/power-model-pilot-verified `
  --output results/static-tuning-check-rerun
& "$checkRoot\env-grid\Scripts\python.exe" -m pytest -q -p no:faulthandler tests/test_power_model_pilot.py tests/test_static_tuning_check.py
```

Use a new output directory. The saved-artifact test currently targets the canonical `results/static-tuning-check` directory, not the rerun directory. No raw gated benchmark files were copied into the project. The local cache dependency remains; this is not yet a fresh-machine reproduction package. No existing controller or earlier result was modified, and nothing was committed or pushed.
