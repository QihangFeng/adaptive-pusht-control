# Delay-aware AI batch control: gap audit and bounded feasibility study

Follow-up: the [October 7 practicality review](2026-10-07-telemetry-delay-practicality-review.md) adds field evidence, clarifies methodological precedent, and qualifies the deployment motivation. The simulation results below are unchanged; typical AI-center remote-voltage delays remain unverified.

## Decision

**GO: select this as a course-project topic, framed as a meaningful extension of existing control, not a new general delay-compensation algorithm.** The completed 48-run study demonstrates an important delay effect and a useful inexpensive remedy in the declared simulation setting. We have enough preliminary evidence to move from topic searching to a focused research project.

Recommended title: **Delay-Aware Batch-Size Control for Grid-Responsive AI Inference**.

Recommended starting method: **power-history-compensated online feedback optimization (OFO)**. Do not yet build the more elaborate age-margin/discrete-optimization proposal. First characterize the simple method's limits; extra machinery should solve a demonstrated limitation.

This is not an unconditional success claim. The application-specific opening is supported by a bounded literature search, not exhaustive novelty proof. The tested transport delays are assumed, external grid disturbances were absent, and the controller still permits some voltage violations. These are explicit research limitations, not reasons to claim real-world safety.

## Scope and question

This follows the [initial delay assessment](2026-09-20-delayed-feedback-assessment.md). It tests the candidate rather than treating it as the final project topic.

> With fresh local datacenter telemetry but delayed remote feeder-voltage measurements, does power-history compensation improve the voltage-throughput-switching trade-off over reasonably tuned batch control?

The proposed contribution is an application-specific control extension and critical evaluation, **not the invention of prediction, delay compensation, state estimation, or discrete control**.

## 1. What the deeper literature audit establishes

### Practical motivation, with important limits

Ortmann et al.'s real distribution-grid deployment received measurements through an archive/CSV pipeline once per minute; grid sensors were also event-triggered. Its approximately four-minute setpoint-to-response lag concerns the actuation path, **not a measured four-minute voltage-packet delay**. No voltage-violation events occurred during the reported deployment, so that capability was not empirically demonstrated. This supports concern about reporting age and interfaces, not our particular 0.5/2-second assumptions. [R3, Sections IV-V]

Comden et al. explicitly separate remote grid measurements, coordinator messages, and local power measurements. Their numerical study uses an approximately 2,000-node feeder, 163 PV units, and 140 batteries. For the delay study, remote messages are sent every 10 seconds and local messages every 2 seconds. Table I's 60-second voltage-channel limit is a **tolerance in their simulation**, not typical observed network latency and not transferable to OpenG2G. [R4, Sections II, V-A, V-D and Table I]

Conversely, GridPilot reports a fast GPU dispatch path on three V100s. **October 7 clarification:** the [authors' artifact](https://github.com/denisa-c/gridpilot#reading-the-latency-result) states that the roughly 97.2 ms headline interval ends when the final cap write returns; delivered-power settling is estimated, not independently measured in that interval. This is neither remote feeder-voltage measurement age nor measured physical power-response time. Large-facility behavior is extrapolated. [R7, Sections 2, 4-6; artifact clarification]

**Interpretation:** studying stale feedback is defensible under a stated telemetry architecture. We have not established a field delay distribution for an AI computing center. Our delay values are sensitivity cases, not industry statistics. Reporting interval, transport delay, and actuation delay must remain distinct.

### Closest work and the remaining opening

| Source and review scope | Already addressed | Implication for our project |
|---|---|---|
| GPU-to-Grid **v2, July 22, 2026**, Sections III-IV [R1] | Measured GPU power/performance, discrete batch actuation, voltage/mean-ITL feedback, one-second control. Discusses implementation mismatch. | No dedicated stale-voltage-packet comparison was identified in these sections. Earlier review used v1; this audit checks the newer revision. |
| OpenG2G v1 and pinned local controller/coordinator source [R2] | Trace-driven datacenter/grid simulation and reusable controllers. | No explicit delayed-observation channel identified in the inspected implementation. Missing infrastructure alone is not a research contribution. |
| Magnusson, Qu and Li, algorithm and theorem [R5] | Delay/asynchrony-tolerant voltage control with fresh local voltage and delayed neighbor messages. | Different information structure and continuous power inputs; nevertheless, delay-aware voltage control is established. |
| Ortmann et al., Section III, plus Picallo et al., publisher abstract/introduction [R3, R6] | Discrete voltage-control inputs; connecting estimated state to feedback optimization. | Neither discretization nor placing an estimator before OFO is new by itself. The Picallo theorem was not independently rederived. |
| Delay-Aware Semantic Sampling, Sections IV-VI [R8] | Local signal reconstruction using inner-loop information and freshness, tested in power-electronic secondary control. | Even freshness-aware local reconstruction already exists. Its DER dynamics differ from our batch-controlled consumption model. |
| GridPilot, architecture/evaluation/limitations [R7] | Predictive coordination, power caps and fast response to grid triggers. | Predictive AI-grid control broadly is not a gap. Its evaluated target is not delayed feeder-voltage feedback driving LLM batch sizes. |

The earlier assessment also covers Zhan et al.'s delayed **dual-message** tests and why conservative tuning is mandatory. Those experiments were not repeated here.

Searches included exact OpenG2G/GPU-to-Grid titles with delay, LLM/GPU voltage control with delayed or stale measurements, predictive data-center grid control, and prediction/state estimation with feedback optimization. No directly matching solution was found in this bounded search. **This is a scoped search result, not proof that no such paper exists.**

The defensible possible opening is narrow: characterize and mitigate the impact of stale remote voltage information on a batch-controlled inference load, including when a cheap local predictor fails. An incremental, carefully evaluated extension can meet the course requirement; it should not be sold as a fundamentally new control algorithm.

## 2. Smallest method worth testing

At time \(t\), the newest delivered voltage sample was measured at \(s_t\), with age \(A_t=t-s_t\). Use

\[
\widetilde v_t = y_{s_t} + \widehat H(P_t-P_{s_t}).
\]

- \(y_{s_t}\): the actual old voltage packet, not a current simulator reading.
- \(P\): measured three-phase datacenter consumption in **kW**, including the same power components at both timestamps.
- \(\widehat H\): one common initial voltage sensitivity in pu/kW, frozen for all methods.
- The unchanged OFO controller receives \(\widetilde v_t\) instead of the held old sample. Throughput/latency fits, objectives and batch choices are unchanged.

In ordinary words: account for the power changes the center has already made before reacting again to an old grid message. No LLM training is needed.

### Why it could work, and why it may fail

For an illustrative local model \(v_t=H_*P_t+w_t\), where \(w_t\) includes external grid changes, and \(y_s=v_s+e_s\), the estimation error is

\[
v_t-\widetilde v_t=(H_*-\widehat H)(P_t-P_s)+(w_t-w_s)-e_s.
\]

This algebra is an explanation, not a nonlinear-grid stability proof. It shows that the predictor can remove the effect of known local power changes, but cannot recover unseen changes elsewhere. A nonlinear approximation adds a residual term. An age-dependent margin would need justified bounds or calibrated residuals; it is **not implemented** in this screen.

The first test is deliberately favorable: fixed feeder configuration/background, one measured model profile, accurate timestamps and fresh local power. Failure here is strong evidence against investing in this simple method. Success would still need external-disturbance, sensitivity-mismatch and variable-age evaluation later.

## 3. Predeclared experiment and correctness checks

Driver: `experiments/delay_screen.py`. Tests: `tests/test_delay_screen.py`.

- Same pinned OpenG2G commit and prepared ML.ENERGY data as the previous pilot; no installed/upstream files changed.
- IEEE 13-bus feeder; datacenter on bus 671; 300 seconds at 0.1-second physical/sensor ticks; same measured Llama-3.1-8B/H100 profile and seven batches.
- **30 baseline runs:** five settings x two development schedules x delays 0, 0.5, 2 seconds; seed 0. Settings: reference, earlier robust double primal step, half voltage gain, half primal and dual steps, and a two-second control period. Slower control retains the same physical transport delay.
- One setting is selected across all four delayed development cases. Each must have integrated voltage violation <=0.10 pu*s, minimum voltage >=0.94, maximum <=1.06 and mean-ITL miss fraction <=1%. Among eligible settings maximize minimum throughput; otherwise minimize worst integrated violation. These are **screening budgets**, not operational safety standards. All metrics remain visible.
- Conditional second stage: six development predictor runs plus twelve evaluation runs (fresh, delayed, predicted at two seconds; two schedules x two seeds). Settings freeze before that stage. `heldout_ramps` is new; `abrupt` and its seed 1 were used historically in other experiments, so only the former is a genuinely new schedule. Both are excluded from this delay-specific selection.
- A useful prototype signal was defined before screening: >=20% and >=0.01 pu*s lower integrated violation, >=98% baseline throughput, and switches <=max(baseline+2, 1.1 x baseline). Also inspect violation duration, worst voltage and ITL; this is not a proof of safety or statistical significance.
- No future packets; copied immutable arrays; older out-of-order packets cannot replace newer data. The first batch is held until data arrives, and startup remains in metrics.
- The optimizer receives a restricted observation interface without a physical-grid handle. Sensitivity is initialized once using common initial model information, not recomputed from live grid state. True voltage is logged only for diagnostics and evaluation.
- Before interpreting delay effects, the zero-delay reference must match the earlier saved 3,000-step trajectory exactly.

Protocols, selections, all individual results, compressed trajectories and observation timestamps are retained under `results/delay-screen/` and, if executed, `results/delay-prototype/`. Each process has a 180-second / 3.5-GiB guard. These bounds are not expected resource consumption.

## 4. Completed results

### Baseline screening: tuning helps, but does not erase the issue

All 30 planned baseline runs completed. The predeclared selection chose `half_primal_dual`: primal step 0.05, voltage/latency dual steps 0.5, one-second updates. Both this setting and slower updates met the development budgets; the former had higher minimum throughput. Three more aggressive settings failed the worst-case integrated-violation budget.

For the selected setting on the repeated-load development schedule:

| Voltage observation | Integrated voltage violation (pu*s) | Throughput (million tokens/s) | Batch changes |
|---|---:|---:|---:|
| Fresh | 0.02199 | 15.256 | 22 |
| 0.5-second delay | 0.04833 | 16.863 | 14 |
| 2-second delay | 0.06044 | 13.598 | 17 |

At two seconds the voltage-violation measure is about 2.75 times the fresh case and throughput is 10.9% lower. At 0.5 seconds, however, throughput is higher while voltage performance is worse. Other schedules also show mixed effects. **Delay changes a trade-off; it is not uniformly harmful on every metric.** Five settings are a reasonable small screen, not exhaustive tuning or proof that no static controller can work.

### Minimal prototype: improvement extends beyond the selection cases

All 18 second-stage runs completed: six development checks and twelve evaluation cases. The table reports means over seeds 1 and 2, with the same frozen setting throughout. Lower voltage violation is better; higher throughput is better. The voltage integral sums excursions outside 0.95-1.05 pu across monitored bus-phases and time; it is **not a percentage voltage drop**.

| Evaluation schedule | Observation method | Voltage integral (pu*s) | Throughput (million tokens/s) | Batch changes | Any-voltage-violation duration (s) |
|---|---|---:|---:|---:|---:|
| Abrupt load changes | Fresh, diagnostic reference | 0.05204 | 14.372 | 11.5 | 3.50 |
| Abrupt load changes | Old voltage, 2-second delay | 0.20651 | 10.104 | 8.5 | 6.90 |
| Abrupt load changes | Power-history compensation, 2-second delay | 0.04899 | 15.188 | 10.0 | 3.05 |
| New ramp schedule | Fresh, diagnostic reference | 0.05604 | 24.138 | 6.0 | 8.05 |
| New ramp schedule | Old voltage, 2-second delay | 0.16827 | 24.304 | 6.0 | 9.80 |
| New ramp schedule | Power-history compensation, 2-second delay | 0.05695 | 24.115 | 6.0 | 8.40 |

Relative to the tuned delayed baseline, in **each** evaluation seed:

- Abrupt schedule: 75.8-76.8% lower voltage integral, 48.9-51.8% higher throughput, and only one or two additional batch changes.
- New ramp schedule: 64.9-67.4% lower voltage integral, 0.75-0.81% lower throughput, and unchanged switching.
- All four comparisons pass the predeclared useful-prototype criteria. Violation duration also decreases in each, and worst minimum voltage is no worse.
- Within the predictor's own trajectories, correcting the received packet reduces observation RMSE by approximately 94% compared with using that packet uncorrected. This is a diagnostic on the same trajectory, not a comparison between different plants.
- All 48 runs have zero **model-average ITL** misses. The 80 ms latency constraint is therefore nonbinding here; this does not establish request-level/p99 latency quality.

The apparent occasional improvement beyond fresh feedback is not evidence that delay is beneficial: discrete actions, transient history and finite trajectories can change the operating path. Fresh feedback is an unequal-information reference, not a competitor the method must beat.

### Correctness and laptop feasibility

- **13 tests passed**, including prior pilot/static-tuning tests and new channel/artifact checks. No stages were skipped in the final run.
- Zero-delay reference matches the prior 3,000-step trajectory byte-for-byte; zero-delay prediction also matches ordinary OFO for both development schedules.
- Recorded packet power/timestamps match the actual historical trajectory. All noninitial packet voltage extrema match their historical physical snapshots within 1e-8 pu. At t=0, OpenDSS's initial sensitivity-estimation re-solve occurs after the stored grid snapshot, as in upstream OFO; the screen's maximum extrema residual is 2.15e-5 pu. That startup exception is isolated and tested, not silently applied to later samples.
- All power-flow solves converged; independent trace integration reproduces the reported metrics; prepared data hashes remain unchanged.
- Per-run sampled process-tree RAM: **141.3-143.6 MiB**. Wall time: **6.0-12.5 seconds** per 300-second simulated case. Total child-run wall time: approximately **355 seconds** for 48 cases. The full investigation also includes reading, implementation and validation time.
- GTX 1650 not used. These are CPU simulations replaying H100 measurements across thousands of virtual replicas, **not live LLM throughput measured on the laptop**.

## 5. What this does and does not justify

| Decision question | Finding |
|---|---|
| Is there a demonstrable problem? | Yes, within the declared stale-remote-voltage architecture and tested load schedules; it survives the tested conservative settings. |
| Is there a reasonable mathematical method? | Yes. A causal sensitivity-based correction improves all four evaluation comparisons under the stated acceptance rule. |
| Is there a research opening? | A credible application/robustness extension, with no direct matching solution found in the scoped search. General prediction and delay-aware control are not novel. |
| Can the laptop evaluate it? | Yes for this CPU-based study. Larger scenario sweeps appear affordable; no local model training is required. |
| Is deployment readiness or general safety proven? | No. Some undervoltage remains: even the compensated evaluation cases have minima around 0.938-0.940 pu, below the nominal 0.95 limit. |

**We should now commit to the research question, not promise a particular final performance percentage.** The initial contribution can be the adapted estimator-controller, its mathematical assumptions/error decomposition, and a rigorous characterization of when it improves or fails. This aligns with the course's allowance for a meaningful extension of prior work.

Remaining work belongs to the actual project, not another broad topic-selection cycle:

1. Define the target telemetry architecture with the instructor: remote multi-bus measurements versus local measurements, timestamp synchronization, and whether reporting age or transport delay is the main limitation. Do not portray the assumed two-second delay as an observed AI-center norm.
2. Test independent external load/generation changes, imperfect sensitivity, variable measurement age and binding service constraints. The current fixed-background setting is favorable to local power compensation. Two seeds and two evaluation schedules are not statistical evidence of generalization.
3. Compare against a wider but bounded set of tuned and rule-based baselines, using equal information and a shared development/test split. Add an age-dependent margin or constrained batch filter only if these tests reveal a specific need; retain prediction alone as a strong ablation.

No stability theorem, safe operating guarantee, production serving-system integration, carbon benefit, or new general-purpose algorithm has been established by this study.

## Reproduction

Use the already working external environment; no new packages were installed. The prepared data and external checkout are still in a temporary directory, so this is reproducible **in the verified local setup**, not yet a portable clean-machine package. Move/cache them permanently as a separate reproducibility task before final submission.

```powershell
$delayPython = 'C:\Users\冯同学\AppData\Local\Temp\ece730-feasibility-580bb4f1\env-grid\Scripts\python.exe'
$delayRoot = 'C:\Users\冯同学\AppData\Local\Temp\ece730-feasibility-580bb4f1\openg2g'
$delayData = 'C:\Users\冯同学\AppData\Local\Temp\ece730-feasibility-580bb4f1\measured-results\specs'

# Run from the project root. Use new output directories; do not overwrite evidence.
& $delayPython experiments\delay_screen.py screen --openg2g-root $delayRoot --prepared-data $delayData --prior-results results\power-model-pilot-verified --output results\delay-screen-repeat
& $delayPython experiments\delay_screen.py prototype --openg2g-root $delayRoot --prepared-data $delayData --prior-results results\power-model-pilot-verified --screen-results results\delay-screen-repeat --output results\delay-prototype-repeat
& $delayPython -m pytest tests -q
```

The pytest artifact tests audit the authoritative `delay-screen` and `delay-prototype` directories; repeat directories can be compared independently. Read each stage's `protocol.json`, `selection.json` (screen), `summary.json`, and per-run `.trace.json.gz` for exact settings/results. The screen must pass before undertaking the conditional prototype stage.

## Sources

- **R1:** Liang et al. [GPU-to-Grid: Voltage Regulation via GPU Utilization Control, v2](https://arxiv.org/html/2602.05116v2), 2026.
- **R2:** Chung et al. [OpenG2G](https://arxiv.org/html/2605.05519v1), 2026; [pinned OFO implementation](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/controller/ofo.py).
- **R3:** Ortmann et al. [Deployment of an Online Feedback Optimization Controller for Reactive Power Flow Optimization in a Distribution Grid](https://arxiv.org/abs/2305.06702), 2023. Full-text Sections III-V read; pages 4-6 visually inspected, including Figures 3, 4 and 7.
- **R4:** Comden, Wang and Bernstein. [Evaluation of Communication Issues in Primal-Dual-Based DERMS](https://docs.nlr.gov/docs/fy23osti/84706.pdf), 2023. Full-text relevant sections read; physical PDF pages 5-8 (printed 2-5) visually inspected, including channel architecture and Table I.
- **R5:** Magnusson, Qu and Li. [Distributed Optimal Voltage Control with Asynchronous and Delayed Communication](https://arxiv.org/html/1903.01065v3), 2020.
- **R6:** Picallo, Bolognani and Dörfler. [Closing the loop: Dynamic state estimation and feedback optimization of power grids](https://doi.org/10.1016/j.epsr.2020.106753), 2020. Publisher abstract/introduction reviewed; not a full theorem audit.
- **R7:** Constantinescu and Atienza. [GridPilot: Real-Time Grid-Responsive Control for AI Supercomputers](https://arxiv.org/html/2605.26384v1), 2026 preprint.
- **R8:** [Delay-Aware Semantic Sampling in Power Electronic Systems](https://arxiv.org/html/2402.13586v2), 2024 preprint.
