# Do delayed grid measurements justify our AI-inference control project?

Date: 2026-10-07. This is a focused literature review of practical motivation and prior work. It adds no simulation results.

## 1. Conclusion and recommendation

**Slow or stale telemetry is a documented issue in real AI infrastructure. However, the reviewed literature does not establish that 0.5-2 second delays in remote grid-voltage measurements are a common operational problem for AI inference centers.** We cannot currently estimate how often our exact assumed situation occurs.

The evidence supports the following conclusions:

| Question | Assessment |
|---|---|
| Do real AI facilities face measurement-timing limitations? | **Yes.** A production AI-cluster study reports slow power reads and monitoring resolution that restricted a control design. [R1] |
| Do AI centers provide useful services to electricity grids? | **Yes.** Physical demonstrations show workload-based power flexibility. Their interfaces mostly use power targets and dispatch events. [R2, R3] |
| Is delayed remote voltage feedback a demonstrated, widespread AI-inference problem? | **Not established by this review.** Our specific measurement architecture and delay values remain experimental assumptions. |
| Is our predictor a new general method? | **No.** Propagating voltage estimates using known power changes and feeding estimates into feedback optimization has established precedent. [R10] |
| Is there still a worthwhile course project? | **Yes, conditionally:** a carefully formulated robustness study and application-specific controller extension, with meaningful alternatives and explicit failure cases. |

My recommendation is to retain the AI-grid direction and continue the current question **as a conditional robustness study**. Before substantial controller development, test whether remote information is actually needed in our scenarios and whether compensation improves on simpler alternatives. The existing feasibility results justify that bounded investigation; they do not settle practical deployment value.

If the selection criterion is specifically “a problem already observed in production using this exact signal and control action,” our current topic has not met it. Internal AI-facility power control has stronger direct evidence, but changing to that problem would require a separate formulation and decision.

This qualifies the practical interpretation of the earlier [topic decision](2026-09-20-delay-go-no-go.md). Its simulation results remain useful.

## 2. What the delay problem actually means

Our controller uses voltage observations to decide how many inference requests to process together. If it reduces electricity consumption but then receives a voltage sample taken before that reduction, it can react again to an outdated problem. Conversely, an old acceptable voltage can hide a recent deterioration. This mechanism depends on how much changes while information is old.

Several different quantities are easily confused:

| Quantity | Meaning | Why it matters here |
|---|---|---|
| Sampling/reporting interval | Time between acquired samples or published updates | The controller may hold a value between updates even with fast transmission. |
| Transport/processing delay | Time from a sample's timestamp to its availability | The delivered reading is already old. |
| Filtering window | Time interval summarized by an averaged measurement | A recent message may describe a smoothed history. |
| Measurement age | Current time minus the timestamp of the newest usable sample | Describes freshness at a decision, assuming meaningful synchronized timestamps. |
| Actuation delay | Time from a command to the physical response | Requires a different timing model from old observations alone. |
| Dispatch lead time and duration | Notice before an event and how long it lasts | A scheduled three-hour reduction is a different control service from one-second voltage regulation. |

For a usable sample timestamped at $s_t$, its age is $A_t=t-s_t$. A stream delayed by two seconds differs from fresh samples published only once every two seconds. Filtered telemetry also needs a definition of which instant its timestamp represents.

Compare age with the control interval and the rate of relevant physical change. A two-second age spans two updates of a one-second controller; it may matter little to an hourly scheduler. This is a reasoning aid, not a stability criterion. EPRI's flexibility framework likewise distinguishes service needs, lead times, duration, and location; its qualitative asset table is not a measurement-latency dataset. [R13, pp. 4-5, 10]

## 3. Evidence from real AI and data-center systems

These are author/operator reports, with the study type and limits kept explicit.

| Source and evidence type | What was observed or implemented | Implication for our question |
|---|---|---|
| **Ardestani et al., Meta production AI-cluster study, 2026 preprint** [R1, Sections 5.1, 5.4, 6] | A 150 MW facility with 83K GB200 GPUs. Critical electrical equipment can be polled every three seconds. Reported read-latency statistics include a median average below one second, a median maximum slightly above one second, and rare outliers near 4.5 seconds. Reading dominates the control loop. Monitoring resolution was insufficient for a telemetry-triggered power-smoothing approach. | Strong direct evidence that telemetry timing affects AI-facility control. These are **internal power readings and training/power-capping workloads**, not external feeder-voltage measurements or inference batch control. Read latency alone is not total measurement age. |
| **Williams et al., physical AI-cluster demonstration, 2026 preprint** [R2, Sections 3.1, 4, 4.2-4.3] | A 130 kW, 96-GPU London cluster received 22 dispatch events based on historical grid scenarios. Signals specified power targets, timing, and ramps. GPU measurements had second-level granularity; rack data had 20-second and minute-level granularity. | Demonstrates grid coordination and mixed-rate telemetry on real hardware. The recorded intervals are **sampling/reporting granularity**, not measured network delays. Replayed grid events do not establish actual delayed voltage feedback. |
| **Colangelo et al., Phoenix field demonstration, 2025 preprint** [R3, Sections 2.2, 4.3, 5] | A 256-A100 cluster delivered 25% curtailment for three hours, with 15-minute ramps, while meeting the stated service commitments. Grid forecasts and historical load information came through an API. | Strong evidence for useful AI demand response, but no reported distribution of remote voltage delay. The demonstrated operating timescale differs substantially from ours. |
| **Google operator report, 2023** [R4, “How does it work?” and “Our work in action”] | Forecast event notices generated hour-by-hour instructions for nonurgent computing; the operator described pilots and regional deployments, including day-ahead demand response. | An important counterexample: valuable data-center grid participation does not necessarily require fast raw voltage feedback. This publication concerns broader data-center workloads, not specifically our inference controller. |

**Interpretation:** the general concern about imperfect telemetry is stronger than our earlier review established. The missing connection is still specific: evidence of an AI inference operator using remote feeder voltages, their timestamp-to-use ages, and harmful decisions attributable to those ages. None of these AI/data-center field sources supplies that complete chain.

The studies also describe different workloads, constraints, and measurement locations. Combining their numbers into a supposed “typical AI-center delay” would be unsupported. We have no representative survey or public timestamped dataset from which to estimate prevalence.

## 4. What grid research contributes, and what it cannot establish

| Source and study type | Relevant finding | Boundary on its use |
|---|---|---|
| **Ortmann et al., operational distribution-grid deployment** [R5, Sections IV-A, V] | A controller for 16 inverters received archive/CSV measurements once per minute. Sensors also used a greater-than-1% change trigger. Approximately four minutes separated setpoint changes from inverter response. | The minute is a reporting cadence; the four minutes is command-to-response lag. Neither measures AI-center voltage transport delay. The deployment experienced no voltage violations during the reported analysis, so it did not demonstrate mitigation of such events. |
| **Comden et al., DERMS simulation** [R6, Sections II, V-A, V-D, Table I] | Different communication paths had different delay tolerances. The voltage channel tolerated 60 seconds in that study; local power feedback tolerated less than 0.2 seconds. Remote and local messages in the delay experiment were sent every 10 and two seconds respectively. | These are **simulated tolerance thresholds**, under that controller and its functionality criteria, not observed latency statistics. They cannot calibrate our remote-voltage delay distribution. |
| **LBNL distribution PMU event library, measured data** [R11, README] | Distribution-grid recordings at 120 Hz capture sub-second events. | Fast measurement technology exists. Sampling resolution alone does not guarantee equally fresh data at a remote controller. We should include fresh-information cases, not assume all grids have slow sensing. |
| **GridPilot, small GPU testbed and authors' artifact** [R12, README “Reading the latency result”] | The artifact explains that its roughly 97 ms headline measures the dispatch path through GPU cap writes; the interval ends when the final write returns. Delivered-power settling is estimated rather than independently measured in that interval. | It is not remote grid-sensor latency. This corrects the earlier report's description of the number as trigger-to-GPU-power response. It also cautions against treating every software/control path as inherently slow. |

A factory study of utility communication hardware was also screened: Otani et al., CIGRE CSE 025 (2022), [Sections 4.1-4.2](https://cse.cigre.org/cse-n025/a-method-for-managing-setting-values-of-voltage-control-in-iec-61850-based-distribution-automation-systems.html). Its approximately two-second setting/report transactions were single tests involving controller settings and a simulated regulator. They do not establish voltage-sample latency and were not used to choose our numerical assumptions.

## 5. Does the original AI-grid architecture require remote measurements?

**GPU-to-Grid provides a coherent research setting, not evidence of a deployed telemetry system.** Its optimization constrains voltages across the distribution network; the measurement step uses three-phase voltages at all buses, and the evaluated controller updates every second. [R7, Sections III-A, III-B, IV-C]

Our interpretation is that a center relying on these observations needs information beyond its own connection point. The paper does not establish how a utility would expose those measurements, their end-to-end freshness, or how commonly this architecture is used. Providing authorized remote measurements, useful timestamps, and suitable model information is part of our declared scenario.

There are substantive alternatives:

| Architecture | What the center receives | Consequence for the project |
|---|---|---|
| Remote multi-bus voltage feedback | Voltage observations from several grid locations | Our present setting; freshness could affect network-wide decisions. |
| Fresh local voltage feedback | Voltage at the center's connection point | May avoid the remote-observation problem. Test whether it adequately protects all monitored buses under the same workloads. |
| Utility-issued power target or budget | A requested power trajectory or limit | Matches the interface of several field demonstrations. The center's task becomes tracking that target; raw grid-voltage estimation may belong elsewhere. |
| Slower or less aggressive feedback | The same information with more cautious decisions | Can reduce repeated reactions to old data, with a possible performance cost. It is a serious baseline. |

Chen and Zhang explicitly study data-center voltage control with **local voltage measurements**, without assuming real-time communication, using GPU frequency control and reactive-power adjustment. [R9, Sections II-B, III] This is a genuine alternative architecture. Its actuators and service model differ from our batch-control problem, so an adapted local-sensing batch controller must be labeled as our comparison, not a reproduction of their method.

Also, “local controller” does not necessarily mean “local-only measurements.” OpenG2G's droop formulation uses a global bus-voltage vector even though actions are taken at individual centers. A local-only baseline requires an explicitly restricted observation interface. [R8, Appendix B.1]

## 6. How much of our methodology is already known?

Our prototype corrects an old voltage observation using the center's intervening power change:

$$
\widetilde v_t = y_{s_t} + \widehat H(P_t-P_{s_t}).
$$

Here $y_{s_t}$ is a received historical voltage sample, $P_t$ is current measured local power, and $\widehat H$ is an approximate power-to-voltage sensitivity. It assumes power at both timestamps is comparable and sufficiently fresh.

Picallo, Bolognani, and Dörfler already model voltage evolution with a known controlled-power-change term plus uncertain external-load changes. Their equations (7)-(8) use that term in state prediction; equation (9) feeds the estimate into feedback optimization. [R10, Section III] **Our correction is a simplified application of this established prediction principle across an observation gap.** The full text makes this overlap clearer than the earlier abstract-level review. Its theoretical guarantees do not automatically apply to our discrete batch choices and delayed packets.

The plausible contribution is therefore a mathematically specified adaptation and critical evaluation for measured AI-inference profiles: discrete batch actions, service-performance trade-offs, realistic observation timing, and limits under outside disturbances. A new general predictor or a proof of deployment safety has not been established.

Our existing error decomposition explains the key limitation: local power changes can be estimated, but an unseen load change elsewhere on the grid cannot be recovered from local power alone. Thus, holding the external background fixed makes the initial experiment favorable. That is precisely why the next evaluation must vary it.

## 7. What our existing experiment proves

The [saved study](2026-09-20-delay-go-no-go.md) includes 30 baseline screening runs and 18 prototype/evaluation runs. It demonstrated improvement after a small conservative tuning screen. It did not measure any real telemetry delay.

For a concrete, independently rechecked saved pair, `heldout_ramps`, seed 1:

| Metric | Tuned controller with two-second-old observations | Same tuning with power-history correction |
|---|---:|---:|
| Integrated voltage violation | 0.16912 pu*s | 0.05508 pu*s |
| Time with any voltage violation, out of 300 seconds | 10.6 s | 8.8 s |
| Aggregate simulated throughput | 24.301 million tokens/s | 24.105 million tokens/s |
| Batch changes | 6 | 6 |
| Minimum voltage | 0.93802 pu | 0.94005 pu |

The integrated measure improved by 67.4%, while throughput fell 0.81%. Its magnitude combines excursions across monitored bus-phases and time; it is not a percentage improvement in voltage. Both runs still crossed the nominal 0.95 pu lower bound. The throughput describes virtual replicas driven by measured GPU profiles, not laptop inference performance.

Sources: [baseline JSON](../results/delay-prototype/eval-heldout_ramps-s1-hold-d2.json), [compensated JSON](../results/delay-prototype/eval-heldout_ramps-s1-predict-d2.json).

Five settings were screened; this does not exclude all useful static tuning. Slower updates already passed the development budgets, with lower minimum throughput than the selected tuning. Model-average service latency was nonbinding in the study, so it provides limited evidence about difficult inference-service trade-offs.

## 8. A bounded next step and a decision rule

No new experiment is performed in this report. The next investigation should resolve these three questions before adding controller complexity:

1. **Is remote information needed?** Compare a fresh local-voltage controller with the current remote-feedback architecture. Use the same plant, workload, batch options, and network-wide evaluation constraints. Label the information difference explicitly. Include a case where an outside disturbance affects a remote bus; verify that the center has enough physical influence to help.
2. **Does the benefit survive realistic uncertainty?** Retain the current fresh/0.5/2-second cases as declared sensitivity assumptions, then distinguish periodic reporting from pure transport delay. Add independent grid-load changes, imperfect sensitivity, and some delay/noise in local power. Do not transplant internal-facility read statistics into an external-voltage delay distribution.
3. **Does compensation beat sensible alternatives?** Compare tuned stale feedback, slower updates, and the simple predictor under common development/evaluation splits. Keep fresh remote feedback as a diagnostic reference and fresh local control as an architectural comparison. Report absolute violation severity/duration, worst voltage, throughput, service latency, and switching.

Predeclare acceptable voltage and service-performance budgets before inspecting evaluation results; distinguish these experimental budgets from operational standards. The relevant result is a useful trade-off under plausible assumptions, not just a large relative percentage.

**Proceed with the narrow delay topic** if there is a defensible remote-measurement scenario, simple alternatives leave a meaningful trade-off, and compensation remains useful with independent outside disturbances. A useful outcome may include clear conditions where it fails.

**Reframe or stop the narrow claim** if fresh local control is adequate, cautious tuning performs comparably, or gains require almost-perfect fresh local power and a fixed outside grid. Increasing delay until an attractive failure appears would not resolve the motivation problem.

The next decision is about the signal and architecture, not laptop capability: the existing CPU environment is adequate for these checks. An optional instructor discussion can help establish whether the conditional robustness framing meets the expected research emphasis; it cannot substitute for field evidence.

## 9. Search scope and remaining uncertainty

This was a targeted review completed on October 7, 2026, using primary papers, author artifacts, laboratory data documentation, and operator/research-institute publications. Searches included:

- `"data center" "voltage" "measurement" "communication delay"`
- `"AI" "datacenter" "grid" "telemetry" latency`
- `"GPU-to-Grid" delayed measurements` and `"OpenG2G" "delay" voltage`
- `"AI data center" "demand response" field demonstration 2025 2026`
- `"data center" "grid" "stale" measurement`
- targeted searches for distribution-grid communication, field deployments, and local-voltage alternatives.

The review followed relevant references to earlier estimation/control work and compared field measurements with simulated delay studies. Relevant full-text sections were inspected; this is not a complete audit of every paper's theorem or implementation. EPRI PDF pages 4 and 10 were rendered and visually checked so that its qualitative table was not mistaken for numerical latency measurements. The accessible Phoenix preprint supplies our detailed locators; search results for its journal version were not treated as an independent experiment.

No directly matching field study was identified in the reviewed material. That is a scoped search finding, not proof of absence or an estimate of prevalence. Deployment details may be unpublished, and recent preprints can change. The most useful missing evidence would be timestamped remote-voltage observations and decision logs from an AI-grid controller, or a documented architecture with an explicit freshness requirement.

## References and reading locations

All web sources accessed 2026-10-07. Preprint status is stated where applicable; section references identify the material used.

- **[R1]** Ardestani et al. *Provisioning to Runtime Optimization of a 100 MW-Scale AI Cluster*. 2026 preprint, v2, May 26. [Full text](https://arxiv.org/html/2605.24461v2). Sections 5.1, 5.4, 6, especially “Dimmer latencies.”
- **[R2]** Williams et al. *Power-Flexible AI Data Centers: A New Paradigm for Grid-Responsive Compute*. 2026 preprint, v1, June 23. [Full text](https://arxiv.org/html/2606.25098v1). Sections 3.1, 4, 4.2-4.3, 5. The paper distinguishes 22 dispatch events from 200+ individual targets/ramp constraints.
- **[R3]** Colangelo et al. *Turning AI Data Centers into Grid-Interactive Assets: Results from a Field Demonstration in Phoenix, Arizona*. 2025 preprint, v1, July 1. [Full text](https://arxiv.org/html/2507.00909v1). Sections 2.2, 4.3, 5.
- **[R4]** Mehra and Hasegawa, Google. *Supporting power grids with demand response at Google data centers*. October 3, 2023. [Operator report](https://cloud.google.com/blog/products/infrastructure/using-demand-response-to-reduce-data-center-power-consumption). “How does it work?”, deployment examples, and the utility's day-ahead pilot account.
- **[R5]** Ortmann et al. *Deployment of an Online Feedback Optimization Controller for Reactive Power Flow Optimization in a Distribution Grid*. 2023. [Full text](https://arxiv.org/html/2305.06702). Sections IV-A, V; distinguish acquisition/reporting from actuator response.
- **[R6]** Comden, Wang, and Bernstein. *Evaluation of Communication Issues in Primal-Dual-Based Distributed Energy Resource Management Systems (DERMS)*. 2023, arXiv v1. [Full text](https://arxiv.org/html/2305.03854). Sections II, V-A, V-D and Table I. Numerical simulation.
- **[R7]** Liang et al. *GPU-to-Grid: Voltage Regulation via GPU Utilization Control*. 2026 preprint, v2, July 22. [Full text](https://arxiv.org/html/2602.05116v2). Sections III-A, III-B, IV-C.
- **[R8]** Chung et al. *OpenG2G: A Simulation Platform for AI Datacenter-Grid Runtime Coordination*. 2026 preprint, v1, May 6. [Full text](https://arxiv.org/html/2605.05519v1). Section 3 and Appendices B.1-B.2, K.
- **[R9]** Chen and Zhang. *Voltage Regulation in Distribution Systems with Data Center Loads*. 2025 preprint, v1, July 8. [Full text](https://arxiv.org/html/2507.06416v1). Sections II-B, III, IV-A; local-voltage/DVFS alternative.
- **[R10]** Picallo, Bolognani, and Dörfler. *Closing the Loop: Dynamic State Estimation and Feedback Optimization of Power Grids*. [Full preprint, v2, September 2019](https://arxiv.org/html/1909.02753v2); [2020 journal publication](https://doi.org/10.1016/j.epsr.2020.106753). Section III-A, equations (7)-(8), and Section III-B, equation (9).
- **[R11]** Lawrence Berkeley National Laboratory. *Distribution PMU Event Library*. [Repository README](https://github.com/LBNL-ETA/pmu_event_library/blob/master/README.md). “General,” “LBNL PMU Configuration,” and data structure; primary measured-data documentation.
- **[R12]** Constantinescu and Atienza. *GridPilot: Real-Time Grid-Responsive Control for AI Supercomputers*. [2026 preprint](https://arxiv.org/html/2605.26384v1); [authors' artifact](https://github.com/denisa-c/gridpilot#reading-the-latency-result). The artifact's “Reading the latency result” limits the interpretation of the headline timing.
- **[R13]** EPRI. *Grid Flexibility Needs and Data Center Characteristics*. June 2025, report 3002031504. [Public PDF](https://restservice.epri.com/publicdownload/000000003002031504/0/Product). Printed pages 4-5 and 10, including Tables 1 and 4.
