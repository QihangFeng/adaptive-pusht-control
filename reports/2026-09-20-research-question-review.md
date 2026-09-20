# Selecting a research question for OpenG2G-based LLM inference control

Date: 2026-09-20

Purpose: focused literature and implementation review for the ECE 730 project. This is a recommendation for discussion, not a finalized topic or a claim of established novelty.

## 1. Recommendation

**Prioritize power-response model uncertainty, with delayed feedback as the backup. Do not select smoother batch-size control in its original broad form.**

Recommended research question:

> How sensitive is grid-aware LLM batch-size control to inaccurate GPU power-response curves, and when does lightweight online calibration improve the voltage-throughput trade-off compared with a well-tuned static controller?

In plain language: the controller needs to know how much electricity demand will change when it changes the number of requests processed together. What happens when that estimate is wrong? Can it learn a useful correction from observed power without unnecessarily reducing AI throughput?

This is narrower than “make AI computing centers energy efficient.” It gives us one modeling assumption to investigate, an existing baseline, and a plausible small implementation contribution. It also connects Python-based AI-system modeling, online estimation, control, testing, and reproducible performance evaluation. That is useful experience to discuss in interviews, although it is simulation research, not production GPU-cluster operation.

**Important qualification:** model adaptation and robust feedback control already have substantial literature. Our possible contribution is an application-specific investigation and extension for discrete LLM batch control, not the invention of adaptive control. Whether calibration helps remains an experimental question.

## 2. The system in understandable terms

An LLM service runs already-trained models to generate responses. Adjusting its batch size changes its workload and electricity demand. OpenG2G connects a simulated data center, an electrical-network model, and a controller. It allows us to investigate how software decisions affect grid voltage and AI-service performance. [OpenG2G, Section 3][R1]

We would reuse the simulator, measured GPU traces, electrical-network model, and existing controllers. Our work would be the uncertainty experiment, a small controller modification if justified, and the analysis.

Terminology:

- **Throughput:** total generated tokens per second across the simulated service.
- **Inter-token latency (ITL):** time between successive generated tokens; not the same as total request response time.
- **Power-response curve:** the relationship between batch size and GPU power.
- **OFO, online feedback optimization:** a controller that repeatedly adjusts its decision using current measurements and an optimization update.
- **Plant:** the system being controlled; here, the simulated computing center and grid.

## 3. Comparison of the three candidates

The assessments below are judgments from this focused review, not measured project outcomes.

| Candidate | Narrow question and plain-language meaning | Prior-work overlap | Implementation and verdict |
|---|---|---|---|
| **Power-response model uncertainty** | Which errors in the batch-to-power relationship actually hurt control, and can online calibration help? The controller may misjudge the effect of changing batch size. | GPU-to-Grid already motivates OFO partly through robustness to modeling imperfections [R2]. Adaptive sensitivity estimation also exists [R3]. “Handle model errors” alone is not novel. | **First choice.** Separate the controller's power fit from the trace-driven plant, introduce controlled fit errors, and test a bounded calibration method. Moderate implementation effort; no new simulator or GPU training required. |
| **Delayed-feedback robustness** | How does stale voltage information affect decisions, and can a timestamp-aware update improve the trade-off? | Grid-control research already identifies systematic communication delays as a weakness [R4]. GPU-to-Grid also mentions actuation delays [R2]. | **Good backup.** Add an observation-history buffer and compare unmodified, slower-updating, and delay-aware controllers. Moderate effort, with particular care around simulator timing and information leakage. |
| **Less disruptive batch control** | Can the controller avoid repeated large batch changes without sacrificing too much performance? | The original objective already penalizes changes [R2]. The implementation includes switching regularization, an existing switching-weight sweep, and a rule-based deadband [C1], [C6], [C7]. | **Not recommended as currently phrased.** Adding a penalty, deadband, or sweeping its weight would overlap too directly. A genuine actuator-cost or transition-dynamics study would require extra modeling and evidence. |

All three can use the same CPU-based environment. The main difference is research scope and validity, not the GTX 1650's capacity.

## 4. What the related work establishes

| Source | Relevant finding | Consequence for our project |
|---|---|---|
| **OpenG2G**, Chung et al., 2026 [R1] | Provides the shared simulation platform and compares existing control approaches. Its default grid backend models quasi-steady voltages, not fast electrical transients. | Reuse the platform; do not present it as our contribution or claim grid-frequency/transient stability results. |
| **GPU-to-Grid**, Liang et al., 2026 [R2] | Fits batch-dependent power, latency, and throughput relationships and formulates constrained batch optimization. It already includes switching regularization and discusses feedback-based robustness. | Investigate the limits of an existing method. Do not claim the baseline assumes perfect behavior or that adding smoothness is new. |
| **Adaptive real-time grid operation via OFO with sensitivity estimation**, Picallo et al., 2022 [R3] | Learns grid input-output sensitivities using recursive estimation. Its method needs informative changes in control inputs, and its guarantees depend on explicit assumptions. | Useful methodological precedent. We would instead examine the GPU batch-to-power component. Its theoretical guarantees do not automatically transfer to discrete, nonlinear LLM control. |
| **Robustness assessment of primal-dual gradient projection-based OFO**, Zhan et al., 2025 [R4] | Reports robustness to several disturbances but sensitivity to systematic communication delays in its distribution-grid setting. | Strong motivation for the delay candidate, but not evidence that the OpenG2G controller fails at the same delay. Review scope here: institutional abstract and publication record, not a complete reproduction of its method. |
| **The ML.ENERGY Benchmark**, Chung et al., 2025 [R5] | Explains measurement design and the importance of hardware, software, workload, and service metrics when interpreting inference energy results. | Supports careful profile selection. It does not establish that an arbitrary 20% power perturbation represents a real deployment change. |

### What we can and cannot call a gap

I did not identify a dedicated evaluation of GPU power-curve mismatch plus online GPU-curve calibration in the inspected OpenG2G sections and code paths. This is a **candidate application-specific gap**, not proof that no paper has studied it.

The focused search covered OpenG2G/GPU-to-Grid, OFO model mismatch and sensitivity estimation, communication delays, and inference energy measurement. It is not an exhaustive systematic review. Before claiming publication-level novelty, we would need a broader citation search and a current upstream check.

For this course, a carefully formulated extension or critical evaluation can still be meaningful even when the underlying estimation algorithm is established.

## 5. What the simulator code actually supports

Code inspection used the previously validated, unmodified OpenG2G checkout at commit `c582a5115cfd49dcdfb140f1837224e7cbf0cfb1`. The links below pin that revision rather than a moving branch. These are inspection findings; no new uncertainty or delay experiment was run for this report.

| Code evidence | What was verified | Practical consequence |
|---|---|---|
| [OFO model store and optimizer][C1] | Controller fits are supplied separately. The update explicitly uses the derivative of the fitted power curve. Grid sensitivity can be re-estimated independently. | A wrong GPU power response and a wrong grid sensitivity are different problems. Start by changing only the GPU fit. |
| [Offline data-center backend][C2] | Power comes from batch-indexed trace templates. The state exposes power per model, active replicas, applied batches, and sampled ITL. | Keep plant traces fixed while changing controller knowledge. A calibration method can use exposed observations, with its telemetry assumptions stated. |
| [Power augmentation configuration][C3] | Per-server scaling and random noise are already supported. | Adding random noise is not itself a contribution. Fleet-wide systematic mismatch is not equivalent to independent per-server noise. |
| [Coordinator timing][C4] and [offline command handling][C2] | The coordinator advances the data center and grid before calling controllers. Commands are dispatched in the same tick; offline batch changes affect the next data-center step. The offline backend rejects ramp-rate commands. | A control interval is not a communication delay. Additional delays or gradual actuation need an explicit wrapper/extension, not just a configuration change. |
| [Performance metrics][C5] | Throughput is estimated from batch size, observed ITL, and active replicas. ITL violations count model/timestep samples. | Do not label these metrics measured request throughput, p99 request latency, or per-request deadline success. |

A project-owned controller module should reuse the upstream components. Avoid editing installed package files. A small subclass or adapted controller may be needed because OFO reads its backend objects directly; arbitrary observation wrappers are not automatically plug-compatible.

### Implementation outline for each candidate

- **Model uncertainty:** construct a separate controller model store; preserve the plant's trace store; log prediction residuals; implement calibration inside a project-owned controller. Unit-test that perturbing controller knowledge does not change the physical replay itself.
- **Delayed feedback:** store timestamped snapshots and expose only the allowed historical measurement to the controller. Keep actuation immediate initially, and record true current voltages separately for evaluation. Test buffer warm-up, zero delay, and the absence of current-state leakage. Observation delay and command delay should be separate experiments.
- **Smoother control:** first compare against existing switching regularization and the rule-based deadband. Do not equate fewer switches with measured user-experience improvement: the offline backend does not model request-level transition overhead.

## 6. Recommended question: a controlled, small study

Working title: **Power-Model Calibration for Grid-Aware LLM Batch Control**.

### 6.1 What exactly is uncertain?

Hold the grid, replay data, latency model, throughput model, and workload schedule fixed initially. Change only the controller's estimate of the power-response curve.

The key distinction is **power level versus power response**. A constant error in predicted watts does not necessarily create an error in the derivative used by OFO. A controller may still respond appropriately through measured voltage feedback. Conversely, a wrong slope can make a batch change appear more or less effective than it really is. This interpretation follows the inspected optimizer update [C1].

Use three diagnostic error types:

1. **Offset error:** shift the estimated curve up or down without changing its slope. This is a useful negative control, not automatically a difficult case.
2. **Gain error:** overestimate or underestimate how strongly power responds to batching.
3. **Shape error:** misplace the saturation region so the response error varies with batch size.

Keep all altered curves positive and physically plausible within the supported batch range. Example perturbations such as a 0.7 or 1.3 gain are sensitivity-test settings, not measured real-world uncertainty levels. Choose final magnitudes after inspecting data and, where possible, held-out fit errors.

**Do not alter both the controller fit and the plant identically:** that would remove the intended mismatch. Equally, do not describe a power-only perturbation as a faithful simulation of a software upgrade or workload change, which could also change latency and throughput.

### 6.2 A small proposed method

A starting method is a bounded online correction of the nominal power curve:

`corrected_power(batch) = a * nominal_power(batch) + c`

Estimate `a` and `c` from observed inference power and the actually applied batch size, using a lightweight recursive or rolling-window fit. This is a proposed adaptation of established estimation ideas, not a new estimator [R3].

For the initial single-model experiment:

- Normalize observed inference power by the active replica count; do not include cooling/base load in a GPU-only fit.
- Use the applied discrete batch, not the optimizer's unexecuted continuous decision.
- Keep calibration causal and bounded; reset it between independent runs.
- Freeze updates when batches lack enough variation, replicas are inactive, or observations are unreliable. Repeated measurements at one batch cannot identify both gain and offset reliably.
- Retain a documented nominal fallback rather than claiming a safety guarantee.

An affine correction can address gain and offset errors but cannot generally fix a misplaced saturation region. Shape error is therefore an important limitation test. A more complex estimator is optional, only if the simple method shows a useful effect.

### 6.3 Baselines and fair comparison

Compare:

1. Fixed batch: no coordination, using a documented, latency-feasible batch selected before testing.
2. Existing OFO with the inaccurate fit.
3. Static OFO tuned on separate development scenarios, including step-size and voltage-gradient-scale tuning.
4. The existing rule-based controller, which does not require fitted power curves [C7].
5. OFO with the proposed online calibration.

A matched-fit OFO run can be a diagnostic reference, but is not an optimality bound or proof of perfect plant knowledge.

The tuned static baseline matters particularly because a uniform power-slope multiplier can act like changing the voltage-term gain. If adaptation only beats a poorly tuned baseline, that is weak evidence for a meaningful method improvement.

Use equal tuning budgets, common evaluation scenarios and seeds, and the same allowed observations. Freeze hyperparameters before final testing; do not retune separately using each test scenario's hidden error. Report any additional power telemetry required by calibration as an explicit assumption.

### 6.4 Minimum evaluation

Start with one model/hardware profile and the IEEE 13-bus feeder. Include normal operation, a load-increase event, and load relief or a separate overvoltage scenario. Check whether any allowed batch could satisfy the constraints in each scenario; physically infeasible cases must not be mislabeled as controller failures.

| Metric | What it answers |
|---|---|
| Integrated voltage violation and worst deviation | How severe are the excursions, not just how often they occur? |
| Time outside voltage limits | How long does the grid operate outside the configured range? |
| Mean throughput or integrated generated-token estimate | How much AI service capacity is retained? |
| Model-average ITL-target violation fraction | Does the simulator's latency target remain satisfied? |
| Batch-change count and magnitude | Does the method introduce excessive control movement? |
| Power prediction error and calibration behavior | Did the estimator actually correct the intended mismatch? |
| Wall time and peak memory | Is the evaluation practical on the laptop? |

Plot voltage violation against throughput rather than declaring a winner from only one metric. Compare at similar latency performance. Include calibration-disabled and no-mismatch cases, and report failures or negligible benefits honestly.

A second measured profile or held-out workload would strengthen the final result if accessible. Keep it an extension, not a prerequisite for the first pilot. Do not fit and evaluate on identical records and call that independent generalization.

## 7. Laptop feasibility and scope boundaries

The existing local feasibility report records a reduced measured-data experiment: preparation peaked around 1.65 GiB RAM; individual 300-second simulations took approximately 5-9 seconds and 139 MiB RAM. These are earlier one-off checks, not new measurements or a guarantee for larger experiments. The report and underlying artifacts were inspected again for this review. [Local validation record][L1]

Our selected profile replays H100 measurements; the laptop does not run the corresponding LLM. The current project `.venv` configuration identifies Python 3.11.9. The earlier simulator checks used a separate Python 3.11.16 environment; installing and checking the selected dependencies in the new environment remains a later setup task.

Recommended limits:

- CPU simulation, one feeder, one model initially; run experiments sequentially.
- Start with at most roughly 30 short pilot runs, then size the final study from measured runtime and memory.
- Reuse a selected data subset; no complete benchmark download is required for the initial study.
- No LLM training, full PPO reproduction, cloud deployment, dashboard, battery co-design, or multi-center coordination in the minimum project.
- Report simulation-based voltage regulation and service proxies, not certified electrical safety or production serving performance.

## 8. Decision and immediate next step

**Recommended choice for approval:** power-response model uncertainty, with lightweight calibration as a candidate method rather than a promised successful solution.

After agreeing on that question:

1. Write a one-page experiment specification defining the uncertainty, available telemetry, baselines, and success criteria.
2. Run a small baseline-only sensitivity pilot. First establish whether plausible slope/shape errors create a meaningful trade-off after reasonable static tuning.
3. Implement calibration only if the pilot justifies it. If the baseline already handles the tested errors well, report that finding and reconsider the method or the delayed-feedback backup; do not exaggerate perturbations merely to manufacture failure.

The intended contribution is a clear answer about **which model errors matter, when calibration is useful, and where it fails**, supported by a reproducible implementation. That is stronger course work and interview material than simply running an existing simulator.

## References and inspection sources

Sources accessed or local artifacts inspected on 2026-09-20. Paper versions and code revision are pinned where available. Literature statements above are separated from proposed experiments; none of the proposed methods has been implemented or evaluated for this report.

- **[R1]** Jae-Won Chung et al. *OpenG2G: A Simulation Platform for AI Datacenter-Grid Runtime Coordination*. arXiv:2605.05519v1, 2026. [Paper][R1]. Relevant sections: 3-4; Appendices B, E, F, J, K.
- **[R2]** Zhirui Liang et al. *GPU-to-Grid: Voltage Regulation via GPU Utilization Control*. arXiv:2602.05116v1, 2026. [Paper][R2]. Relevant sections: II-C, III-A/B, IV; especially objective (4a) and the feedback update.
- **[R3]** Miguel Picallo, Lukas Ortmann, Saverio Bolognani, Florian Dörfler. *Adaptive real-time grid operation via Online Feedback Optimization with sensitivity estimation*. Electric Power Systems Research 212, 108405, 2022. [Author preprint, v5][R3]; [publisher record](https://www.sciencedirect.com/science/article/pii/S0378779622005582). Relevant sections: II-D and III, including excitation requirements and conditional convergence results.
- **[R4]** Sen Zhan et al. *Robustness Assessment of Primal-dual Gradient Projection-based Online Feedback Optimization for Real-time Distribution Grid Management*. Electric Power Systems Research 242, 111468, 2025. [Institutional record and abstract][R4]; [DOI](https://doi.org/10.1016/j.epsr.2025.111468). Used for the reported findings, not detailed implementation claims.
- **[R5]** Jae-Won Chung et al. *The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization*. arXiv:2505.06371v1, 2025. [Reviewed preprint][R5]. Relevant sections: 2 and 3. This paper describes an earlier benchmark iteration; our selected simulation data are from benchmark v3.
- **[C1]** OpenG2G `controller/ofo.py`: model store, fitted derivatives, switching term, and observations. [Pinned source][C1]. Key lines: 93-165, 442-617, 646-810.
- **[C2]** OpenG2G `datacenter/offline.py`: replay/state generation and batch-command behavior. [Pinned source][C2]. Key lines: 213-313.
- **[C3]** OpenG2G `datacenter/config.py`: power augmentation settings. [Pinned source][C3]. Key lines: 431-458.
- **[C4]** OpenG2G `coordinator.py`: tick ordering and command dispatch. [Pinned source][C4]. Key lines: 263-343.
- **[C5]** OpenG2G `metrics/performance.py`: throughput and ITL definitions. [Pinned source][C5]. Key lines: 19-40, 98-142.
- **[C6]** OpenG2G `examples/offline/sweep_ofo_parameters.py`: existing switching-weight sweep. [Pinned source][C6]. See line 173.
- **[C7]** OpenG2G `controller/rule_based.py`: model-free proportional baseline, deadband, and latency guard. [Pinned source][C7]. Key lines: 32-68.
- **[L1]** Local *OpenG2G measured-data feasibility validation*, dated 2026-09-19. [Validation record][L1]. Internal feasibility evidence, not an external publication; the temporary path may not remain available indefinitely. The essential resource results are preserved in Section 7 above.

[R1]: https://arxiv.org/html/2605.05519v1
[R2]: https://arxiv.org/html/2602.05116v1
[R3]: https://arxiv.org/html/2110.00954v5
[R4]: https://research.tue.nl/en/publications/robustness-assessment-of-primal-dual-gradient-projection-based-on/
[R5]: https://arxiv.org/html/2505.06371v1
[C1]: https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/controller/ofo.py#L93
[C2]: https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/datacenter/offline.py#L213
[C3]: https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/datacenter/config.py#L431
[C4]: https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/coordinator.py#L263
[C5]: https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/metrics/performance.py#L19
[C6]: https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/examples/offline/sweep_ofo_parameters.py#L173
[C7]: https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/controller/rule_based.py#L32
[L1]: <C:/Users/冯同学/AppData/Local/Temp/ece730-feasibility-580bb4f1/measured-results/VALIDATION.md>
