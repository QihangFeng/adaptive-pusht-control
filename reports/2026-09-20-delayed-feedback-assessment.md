# Delayed voltage feedback: research-question assessment

## Conclusion

**This is worth a bounded baseline experiment, but not yet selection as our final contribution.** Delayed information has a clearer documented failure mechanism than our simple power-calibration proposal. However, delay-tolerant control is established research, and neither a practically important delay nor an improvement has been demonstrated in our OpenG2G setup.

Recommended candidate question:

> When grid-voltage measurements arrive late, can timestamp-aware batch control retain more AI throughput at comparable voltage violations and batch switching than a conservatively tuned existing controller?

Keep the general AI-computing-center direction. Keep calibration on hold. Do not start a large controller implementation or claim novelty from adding delay to a simulator.

This report contains literature review, inspection of the existing implementation, and a proposed methodology/evaluation design. **No delayed-feedback simulation or new controller was implemented in this review.**

## 1. Explanation in ordinary words

Imagine the computing center receives a message saying the grid voltage is too low. It reduces the number of AI requests processed together to reduce electricity demand. But the message describes the grid several moments ago. The center may already have reduced its demand since that measurement was taken. Acting repeatedly on old information can produce unnecessary reductions or back-and-forth adjustments.

That is an illustrative mechanism, not an observed result from our simulator. A more careful controller could ask: **How old is this measurement, and what have I changed since it was taken?**

The trade-off is important: simply reacting more slowly can reduce overreaction, but may also delay a useful response when electricity demand genuinely changes. We need to test whether a more informed method improves that trade-off, rather than merely beating an aggressive default setting.

## 2. What the related work actually establishes

| Related work | Evidence and review scope | Consequence for us |
|---|---|---|
| **GPU-to-Grid**, Liang et al., 2026 [R1] | Sections III-IV inspected. Uses voltage/latency feedback and discrete batch actions; experiments use one-second controller updates. Mentions implementation mismatch, including actuation delays. | The existing controller is not claimed to require perfect behavior. Mentioning delay robustness is not a dedicated evaluation of stale voltage packets. |
| **OpenG2G**, Chung et al., 2026 [R2] | Controller architecture, evaluation and limitations inspected, alongside pinned local source. | Reusable environment. I did not identify a dedicated measurement-transport-delay experiment in the inspected sections or a built-in delay channel in the pinned package. This is not an exhaustive novelty claim. |
| **Robustness assessment of primal-dual OFO**, Zhan et al., 2025 [R3] | Full-text Sections 4.1, 4.3, 5.5-5.6 and 5.9; Figures 7 and 9 visually inspected. | Systematic delays can damage a related grid controller. The authors also test conservative scaling and recommend slower updates, so those are necessary competing explanations/baselines. |
| **Distributed optimal voltage control with asynchronous and delayed communication**, Magnusson, Qu and Li, 2020 [R4] | Algorithm and Theorem 1 inspected. Uses local voltage measurements plus delayed neighbor messages; proves convergence under a linear power-flow model, bounded delays and a sufficiently small step size. | Delay-aware voltage control is not new. Its continuous power decisions and communication structure differ from centralized delayed voltage observations and discrete LLM batches. Its theorem does not automatically apply here. |
| **Delay-robust primal-dual dynamics**, Sen et al., 2026 preprint [R5] | Problem formulation and proposed augmentation inspected. Adds auxiliary dynamics with delay-dependent gain conditions for continuous-time convex, equality-constrained distributed optimization. | Another methodological precedent, not a drop-in controller or transferable stability guarantee for our nonlinear, discrete, inequality-constrained setting. |

### Important correction to the earlier literature summary

Zhan et al. delay the **broadcast dual information used by the primal update**, not raw voltage samples. Their normal iteration period is **six seconds**. In one tested configuration, performance fails beyond two delayed iterations; changing scaling substantially changes that behavior. Their result motivates investigating communication timing, but gives **neither an OpenG2G failure threshold nor evidence that a two-second measurement delay is typical**. [R3, Sections 4.1, 4.3 and 5.6]

We must distinguish measurement delay, delayed controller messages, delayed actuation and a slow update interval. They place old information at different points in the control loop.

## 3. Practical importance: plausible, but not fully established

A defensible architecture is a controller located at the computing center, with fresh local power/batch information but remotely received feeder-voltage measurements. Measurement reporting, aggregation and transport determine their age. A local voltage sensor at the center is not automatically equivalent to fresh voltage measurements at every monitored grid bus.

This architecture is a **project assumption**, not a measured deployment. The literature above supports communication limitations in grid control; it does not establish the delay distribution for an AI computing center. We have not obtained field telemetry that would justify saying that multi-second delays are common in such systems.

Therefore:

- Study delay relative to the controller's update interval and report actual measurement age in seconds.
- Treat proposed numerical delays as sensitivity cases, not observed networking statistics.
- Include short delays, not only extreme cases selected to force failure.
- If a practical deployment provides sufficiently fresh measurements and the effect is negligible, the relevance of this particular contribution is weaker.
- Keep the claim to quasi-steady distribution-voltage regulation. OpenG2G's default backend cannot establish fast electrical-transient or grid-frequency stability. [R2]

## 4. A possible mathematical contribution

This is a design candidate for discussion, not an implemented or proven method. The building blocks are established model-based prediction and constrained optimization. The potential course contribution would be their carefully justified adaptation to delayed observations, discrete AI batches and service constraints.

### Information available to the controller

At decision time \(t\), the latest delivered voltage packet has measurement time \(s_t\):

\[
y_t=v(s_t)+e_t,\qquad A_t=t-s_t\geq0.
\]

Here \(A_t\) is measurement age, not LLM response latency. Initially assume synchronized timestamps and delay only the voltage channel; local power, applied batch, active replica count and ITL remain available. This assumption must be shared across compared methods.

### Proposed prediction and batch decision

Start with the incremental voltage-power approximation underlying GPU-to-Grid [R1, Eq. 12-14]. Let \(P_t\) be measured three-phase datacenter active-power consumption in kW, and \(\widehat H\) an offline-estimated voltage sensitivity at the same fixed power factor. Estimate current voltage from the old packet and measured local power change:

\[
\widehat v_t=y_t+\widehat H(P_t-P_{s_t}).
\]

For candidate batch \(b\), with current replica count \(n_t\) and actually applied batch \(b_t\), predict the next operating voltage:

\[
\widehat v_t^+(b)=\widehat v_t+
\widehat H\bigl[\widehat P(b,n_t)-\widehat P(b_t,n_t)\bigr].
\]

The measured power difference covers what has already happened; the fitted difference predicts only the proposed next action. This avoids counting the same batch change twice. The sensitivity sign must match consumption, not injection, and all powers must have consistent units and phase ordering.

A small discrete optimization could then maximize estimated throughput minus a batch-change penalty, subject to:

\[
\underline v+\epsilon(A_t)\leq\widehat v_t^+(b)
\leq\overline v-\epsilon(A_t),\quad
\widehat{\mathrm{ITL}}(b)\leq L_{\max},\quad b\in\mathcal B.
\]

\(\epsilon(A_t)\) is an age-dependent prediction-error allowance, selected using development data and checked on held-out scenarios. Older information may warrant more caution. With the current seven supported batch sizes, the optimization can enumerate candidates; it does not require training an LLM or a large optimization solver.

**Limits:** this does not reveal unknown changes in other loads, generation or tap positions. They become prediction error. The next-step approximation also assumes limited change during actuation. A fitted error allowance is not a certified bound, so it provides no unconditional safety guarantee. If no candidate passes, the method must report that condition and use an explicitly defined least-predicted-violation fallback; it cannot claim to enforce infeasible constraints.

The smallest initial extension could instead feed the compensated voltage into existing OFO. Before choosing a discrete optimization/filter, compare this simpler alternative. To justify the age-dependent allowance, it must add value beyond prediction alone and a fixed allowance. Adding a switching penalty alone is not new: the original formulation already includes one.

## 5. Implementation feasibility on our laptop

Code inspected at unchanged OpenG2G commit `c582a5115cfd49dcdfb140f1837224e7cbf0cfb1`:

| Verified source behavior | Required implementation work |
|---|---|
| `GridState` includes `time_s` and per-bus/per-phase voltages. [C1] | A bounded packet buffer can store copied values, measurement timestamps and arrival timestamps. |
| Coordinator advances datacenter and grid before controllers. [C2] | Record packets at grid ticks; release only those whose arrival time has passed. Evaluate performance using the true current grid state separately. |
| OFO directly calls `grid.voltages_vector()` and can call `grid.estimate_sensitivity()`. [C3] | A project-owned observation interface/controller adapter is needed. Merely attaching timestamps to logs does not delay its observations. Precompute/freeze sensitivity to prevent live simulator access from leaking current information. |
| Rule-based control reads `grid.state.voltages` through a different path. [C4] | Route that baseline through the same delayed channel too. |
| Datacenter states expose timestamped power, applied batch and replica counts. [C5] | Retain aligned local power history for causal compensation. Do not use unexecuted optimizer decisions as applied batches. |

Do not overwrite the simulator's true state globally or edit installed package files. A copied array is needed because a frozen state object does not make nested dictionaries immutable.

The earlier completed CPU experiments used about 139-141 MiB per simulation and seconds per 300-second scenario [L1]. A short packet history and seven-candidate calculation appear inexpensive by inspection, but **their integrated runtime has not yet been measured**. Your GTX 1650 is not required for this approach.

Essential correctness tests before interpreting results:

1. Zero delay reproduces the existing controller's saved trajectory under the same configuration.
2. A packet is unavailable before its arrival; the latest delivered measurement timestamp never lies in the future.
3. A late, older packet does not replace a newer one when testing variable delays.
4. Specify startup without fictional fresh measurements; include startup in reported metrics or define a common pre-event warm-up explicitly.
5. True current grid voltage cannot be reached through another controller or sensitivity-estimation path.
6. Log generation time, arrival time and age. Keep measurement period, transport delay, control period and actuation delay separate.

## 6. The next experiment and decision rule

**Next: baseline-only screening, not implementation of the proposed compensator yet.** Preserve the measured plant, keep the power model fixed, and introduce only the timestamped voltage channel.

Suggested first screening is at most **30 simulations**: five predeclared existing static configurations, three delays (0, 0.5 and 2 seconds), and two development load schedules, using one fixed seed. Keep the original one-second control interval initially. These delay values probe fractions/multiples of a control interval; they are not measured deployment delays. Reuse an existing zero-delay run only when the full configuration and seed match, after verifying the adapter's zero-delay equivalence.

This screen is exploratory and cannot support a final performance claim. Inspect voltage severity, worst voltage, violation duration, throughput, ITL misses, and switching jointly. Do not select a winner from a single relative voltage metric or give each test scenario its own retuned controller.

Continue only if delayed feedback creates a meaningful loss after reasonable static tuning. Before final evaluation, declare absolute/relative improvement criteria and freeze settings on separate development scenarios. If the effect appears only at extreme delays with no defensible use case, reconsider the question rather than escalating delay until something breaks.

Any eventual method comparison should include:

- Current-feedback OFO as a diagnostic reference, not an equal-information competitor.
- Delayed-feedback OFO with shared, development-tuned parameters.
- Smaller primal/dual steps and a slower-update baseline with unchanged physical transport delay. Account for the different number of updates per second and tune accordingly.
- Rule-based control on the same delayed observations.
- Simple power-history prediction without the age-dependent allowance, then a fixed allowance, to isolate our proposed addition.
- Separate held-out scenarios/seeds and, later, variable delays or external load changes that the local power history cannot explain.

**Go/no-go assessment now:** the problem is plausible, the environment is reusable and the mathematical scope is manageable. The empirical need, deployment-relevant delay range and benefit beyond tuned baselines remain unproven. This is a stronger reason to run one bounded screening study, not yet a reason to promise a successful new method.

## Sources

- **[R1]** Liang et al. *GPU-to-Grid: Voltage Regulation via GPU Utilization Control*, 2026. [Paper](https://arxiv.org/html/2602.05116v1).
- **[R2]** Chung et al. *OpenG2G: A Simulation Platform for AI Datacenter-Grid Runtime Coordination*, 2026. [Paper](https://arxiv.org/html/2605.05519v1).
- **[R3]** Zhan et al. *Robustness Assessment of Primal-dual Gradient Projection-based Online Feedback Optimization for Real-time Distribution Grid Management*, EPSR 242, 111468, 2025. [Publication record](https://research.tue.nl/en/publications/robustness-assessment-of-primal-dual-gradient-projection-based-on/), [university repository author manuscript](https://pure.tue.nl/ws/portalfiles/portal/351180729/main.pdf). Physical PDF pages 7-11 (printed pages 6-10) inspected, including complete figures and captions. This improves on the earlier abstract-only review.
- **[R4]** Magnusson, Qu and Li. *Distributed Optimal Voltage Control with Asynchronous and Delayed Communication*, IEEE TSG 11(4), 3469-3482, 2020. [Author manuscript](https://arxiv.org/html/1903.01065v3).
- **[R5]** Sen et al. *Delay-Robust Primal-Dual Dynamics for Distributed Optimization*, 2026 preprint. [Paper](https://arxiv.org/html/2603.18236v1).
- **[C1]** [Pinned grid state/interface](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/grid/base.py), especially `GridState`.
- **[C2]** [Pinned coordinator](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/coordinator.py), `step` and controller dispatch.
- **[C3]** [Pinned OFO controller](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/controller/ofo.py), `OFOBatchSizeController.step`.
- **[C4]** [Pinned rule-based controller](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/controller/rule_based.py), `RuleBasedBatchSizeController.step`.
- **[C5]** [Pinned datacenter state/interface](https://github.com/gpu2grid/openg2g/blob/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1/openg2g/datacenter/base.py), `DatacenterState` and `LLMDatacenterState`.
- **[L1]** [Completed calibration-worth assessment](2026-09-20-calibration-worth-assessment.md) and its saved results. Resource evidence is from those completed runs, not a new delay experiment.
