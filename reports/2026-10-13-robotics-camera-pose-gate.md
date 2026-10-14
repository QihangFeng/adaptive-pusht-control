# Is the camera precise enough for small final-positioning corrections?

Date: October 13, 2026

## Conclusion and purpose

**The current 96×96 pose estimator did not meet our predeclared fine-correction accuracy targets.** It identified the correct direction toward the goal in all 184 eligible near-target component comparisons, but its 95th-percentile position, rotation and vertex errors were **1.48 units, 0.98° and 1.99 units**, exceeding the targets of **1 unit, 0.5° and 1 unit**. The previous-frame warm start also failed these precision targets.

Therefore, **stop before adding corrective pushes**. We need to improve or reconsider pose estimation before implementing this particular fine-correction design. This is not proof that camera-based correction is impossible or that a different uncertainty-aware design cannot work. The thresholds are an engineering screen, not a theorem about controller success.

This completed first gate concerns the proposed **fixed-eight plus camera-based final-positioning corrector**. It is not a new adaptive timing rule, policy training run or controller-improvement evaluation. No local action model, corrected movement or recovery pilot was implemented. The milestone, fixed-eight baseline, estimator and earlier results remain unchanged.

Existing coverage accuracy checks do not answer whether the camera can identify the remaining translation and rotation precisely enough to guide small pushes. We measure those errors directly before building a local action model.

## Frozen selection and inputs

- Use the 30 existing unchanged fixed-eight discovery attempts, seeds 0–29: 20 successes and ten timeouts. These are already-inspected development scenes, not a holdout.
- For every scene, select up to three recorded frames with true coverage in **[90%, 95%]**: first, middle and last eligible temporal indices. Selection uses truth only for this offline validation. No frame is selected because its visual fit looks good.
- Include each successful terminal frame with true coverage strictly **above 95%** as a completed-task control.
- This yields **96 selected frames: 76 fine-alignment frames and 20 completed-task controls**. All 30 scenes contribute fine-alignment frames: 46 samples from successful scenes and 30 from failed scenes. Some pass through the band in only one or two observations, so not every scene supplies three samples.
- The estimator receives only the original **96×96 RGB image**, with the already-known static T silhouette and goal. Dynamic true pose is used only for image reconstruction and scoring; it is not an estimator input.

We obtain the saved observations by **CPU replay of recorded commands**, without loading or calling the action policy. Each replayed image must match the original RGB SHA-256 exactly, and recorded pusher/block coordinates and block angle must also remain exact, including the initial state. Both each selected image and its immediately preceding image are checked before fitting; only the selected image receives an accuracy score. This replay reconstructs old observations, **not the model observation/RNG state for a future alternative pushing continuation**.

An initial fast static-image preflight matched four selected frames, but stopped at the first RGB mismatch on scene 1. No mismatching image was fitted or scored. Its source, frozen protocol and four verified measurements remain recoverable locally under `tmp/robotics-pose-gate-static-preflight/`. The production collector uses recorded-action replay instead. Sample identities, estimation variants, deadbands and accuracy targets were not changed; all 96 predeclared frames remain included. The four matching preflight frames are refitted in the production screen, not selectively removed or treated as independent extra samples.

## Estimation variants

The primary candidate is a **cold, stateless** fit on the selected image. A separate sensitivity check fits the immediately preceding RGB image first, then the selected image. This two-image warm start is **not** a reproduction of the earlier monitor's full chronological history. Neither variant uses true pose to initialize the estimator or changes its segmentation/fit settings.

## Metrics and predeclared gate

Convert the estimated image pose back to simulator coordinates using the existing pixel-center offset and scale. The target body origin is `(256,256)` with angle `π/4`. Angles wrap modulo `2π`; the T does not have 180-degree symmetry.

Measure Euclidean body-origin position error, wrapped angle error, and maximum corresponding T-vertex error. The vertex metric combines translation and rotation. Also compare the **signed direction toward the goal** in x, y and rotation; two vectors can have similar magnitudes while pointing in opposite directions.

| Primary fine-alignment screen | Frozen target |
|---|---:|
| Valid estimate fraction | At least 98% |
| 95th-percentile Euclidean position error | At most 1 coordinate unit |
| 95th-percentile maximum vertex error | At most 1 coordinate unit |
| 95th-percentile wrapped rotation error | At most 0.5° |
| Direction agreement, pooled and each x/y/rotation component | At least 95% |

Direction is evaluated outside true residual deadbands of **one coordinate unit** for x/y and **0.5°** for rotation. Invalid fits remain in availability counts; unavailable or near-zero directions are not counted as correct. Each directional component needs available observations to pass. These are diagnostic deadbands, not controller triggers.

These targets are a conservative engineering screen tied to resolving an intended three-coordinate-unit pose correction with roughly one-unit uncertainty. They are **not** a guarantee about contact dynamics, pusher-command effects or task success. Limits and selection are frozen before fitting; do not loosen them after unfavorable results.

Report frame-weighted errors, scene-equal mean errors, per-scene errors, valid availability, and separate x/y/rotation agreement. Frames within a scene are correlated. Completed-task controls are reported separately, not mixed into the primary fine-alignment gate.

## Results and decision

### Primary fine-alignment frames

Both variants have valid estimates for all 76 frames. The primary method is the stateless fit; the warm-start variant is a sensitivity check, not a newly selected controller.

| Metric | Frozen target | Cold primary | Previous-frame sensitivity |
|---|---:|---:|---:|
| Valid estimate fraction | ≥98% | 100% | 100% |
| Mean Euclidean position error | Diagnostic | 0.72 units | 0.72 units |
| 95th-percentile position error | ≤1 unit | **1.48 units** | **1.48 units** |
| 95th-percentile rotation error | ≤0.5° | **0.98°** | **0.93°** |
| 95th-percentile maximum vertex error | ≤1 unit | **1.99 units** | **1.96 units** |
| Signed direction agreement outside deadbands | ≥95%, each component | 184/184 | 184/184 |
| Overall precision gate | All criteria pass | **Fail** | **Fail** |

The directional comparisons are **57 x components, 62 y components and 65 rotation components**; each has 100% agreement in this sample. Components within the one-unit/0.5° deadbands are excluded, not counted as correct. This observed agreement is encouraging for coarse directional information, but does not establish perfect direction prediction in new scenes or determine the size of a physically effective push.

Cold scene-equal mean errors are **0.73 units** for position, **0.33°** for rotation and **1.09 units** for vertices. Frame-weighted means are **0.72 units, 0.30° and 1.05 units**. Per-scene means are retained in the summary; the difference reflects unequal sample counts, not additional independent scenes.

### Completed-task controls

The 20 successful terminal frames have cold 95th-percentile errors of **1.30 position units, 0.80° rotation and 1.75 vertex units**. One eligible y-direction comparison is opposed: **12/13** agree, while x is 4/4 and rotation 14/14. Previous-frame results are nearly identical. These completed controls are not mixed into the primary gate, and the benchmark terminates before any additional corrective action. They nevertheless show why near-completion perception must not be treated as an exact goal sensor.

### Interpretation and next discussion

In plain words: **the estimator usually knows which way the object needs to move, but its estimate of exactly how far and how much to rotate is not yet precise enough for our chosen small-correction target.** Mean errors alone would hide the less accurate cases captured by the 95th percentile.

The gate failed before any correction model or action was introduced. Do not loosen the frozen limits, declare the passing direction metric sufficient, or claim a success-rate improvement. A possible next investigation is better image fitting or retaining a higher-resolution frame for the auxiliary monitor while the pretrained policy keeps its 96×96 input. That would be a perception/design change requiring discussion first; neither change is implemented here. Higher resolution also does not automatically solve occlusion, contact modeling or correction selection.

If a later perception version passes an appropriately declared accuracy screen, evaluate one bounded correction against unchanged fixed-eight continuation from matched states. Include successful temporary stalls, several prediction samples and the original remaining movement budget, then use genuinely new scenes for any frozen method evaluation. Those steps remain conditional and unperformed.

### Reproducibility, resource use and audit

Production replay verified all **5,557 executed recorded movements plus 30 initial observations** against saved RGB hashes and pusher/block coordinates/angle. There were **192 scored pose estimates** on 96 frames and **288 estimator calls**, including the predecessor-image warm starts. No new neural-policy calls or corrective commands occurred.

The complete production invocation took **36.65 seconds**, with peak observed process RAM **140.27 MiB**. This CPU observation check is laptop-feasible; it does not establish the feasibility or benefit of a future learned action model or full correction evaluation.

All **384 robotics CPU tests passed**, including 48 focused tests. The [independent audit](../results/robotics-pose-gate/independent-audit.json) passed: it independently reconstructed selection, recomputed 192 pose scores, directional denominators, per-scene/stratum summaries and gate flags, checked 34 source/evidence bindings plus package/two-renderer-source identities, and verified resource accounting. It did not rerun rendering, fitting or physics; exact RGB/coordinate recovery is asserted by the hash-bound collector. The four valid static-preflight estimates and scores match their production refits exactly; timings are not treated as additional observations.

## Reproduction

Use the existing `.venv-robotics` environment. No model loading, GPU inference, paid API or new training data is required.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_pose_gate.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_pose_gate.py --stage run
```

Completed matching results are verified, not overwritten or refitted. For a disposable repeat, add `--output-dir results\robotics-pose-gate-repeat` to **both** stages. The repeat directory is Git-ignored.

## Sources

- [Lost-success investigation](2026-10-13-robotics-lost-success-review.md): motivation for testing final action effectiveness instead of another interruption filter.
- [Discovery attempt inventory](../results/robotics-failure-patterns/summary.json) and [original camera validation](../results/robotics-benefit-pilot/visual-validation.json): saved scene/frame provenance and exact RGB trace bindings.
- [Camera estimator](../experiments/robotics_visual_monitor.py) and [benefit-pilot report](2026-10-11-robotics-benefit-pilot.md): reused geometry, image-to-world transformation and earlier coverage-only accuracy checks.
- [Frozen protocol](../results/robotics-pose-gate/protocol.json), [all selected measurements](../results/robotics-pose-gate/samples.json), [summary](../results/robotics-pose-gate/summary.json), [independent audit](../results/robotics-pose-gate/independent-audit.json), and [resource record](../results/robotics-pose-gate/resources-from-000.json): authoritative evidence for this completed gate.
- Installed `gym-pusht 0.1.5` source, `envs/pusht.py` and `envs/pymunk_override.py`: static renderer behavior, body origin, action coordinates and strict success criterion. Their hashes and package versions are recorded in the new protocol, not vendored or modified.

The contribution of this stage is a reproducible observability screen, not a corrected robot policy or demonstrated success-rate improvement.
