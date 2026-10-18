# Why do six baseline successes fail under adaptive execution?

Date: October 13, 2026

## Conclusion first

**We found different observable failure patterns, not one common interruption mistake.** Some attempts get almost aligned and then produce no effective final push. Others get almost aligned, push the object away again and fail to regain their best alignment before the time limit. All six lost successes reach between **93.62% and 94.81%** true coverage, but never exceed the unchanged **95%** success criterion.

The clearest new evidence is in **78 and 81**: the object stops moving even though the system requests **12 and nine additional predictions**. Near the end, both already use eight-movement intervals. Asking for a new plan earlier is therefore not, by itself, an established solution to their final stalls.

**Keep milestone `0f40449` unchanged.** Do not restart the rejected threshold/weighted filters. A more useful next discussion is effective final-positioning recovery, rather than another rule that merely asks the same model to replan sooner. This report identifies problems to investigate; it does not establish a fix or prove what caused each regression.

## Scope and preserved work

This is CPU-only inspection of **120 completed attempts on 40 initial scenes, 50–89**: fixed-eight, fixed-twelve and unchanged adaptive eight/twelve. It focuses on all six lost baseline successes **54, 59, 78, 80, 81 and 82**, while retaining the six rescued successes, 27 retained successes and shared failure **72** as comparison information.

No new prediction, simulator run, RGB fitting, training, recovery probe, altered criterion or controller change was performed. The completed 54/59 diagnosis and earlier extension inspection were not rerun; their findings and hash-bound extension windows were reused. Only new saved-evidence diagnostics and the four additional lost cases were examined.

The runner verifies 457 source/evidence bindings, sidecars, compressed-trace hashes, movement/query accounting and first-success termination. It reuses earlier audits for full controller/queue provenance rather than claiming a new physics or neural replay. Every fixed-eight/adaptive pair shares the same initial physical/image state and at least eight exactly matching executed movements; actions, coordinates and image hashes stay exact. In all six lost cases, the first differing command is movement **9**. Later failure states are not matched alternatives. Only existing truth-coverage rounding tolerance `1e-12` is retained.

## 1. What actually fails in each case?

Coverage measures how much the block overlaps the target. Every adaptive attempt below times out after 300 movements. The baseline column records when fixed-eight completes successfully; it is not a continuation from the adaptive attempt's later state.

| Case | Fixed-eight succeeds after movement | Adaptive best coverage / movement | Adaptive final coverage | What the saved adaptive attempt does |
|---|---:|---:|---:|---|
| 54 | 215 | 94.58% / 280 | 94.49% | Recovers the previously studied drop, but reaches almost-complete alignment late and does not finish. |
| 59 | 201 | 94.53% / 135 | 91.87% | Regains the overlap immediately before the studied drop, not its whole-attempt best; late pushing makes little progress. |
| 78 | 136 | 94.57% / 197 | 94.57% | Block pose remains unchanged after 197 through 300, despite further predictions. No early interruption occurs anywhere. |
| 80 | 128 | 94.61% / 122 | 39.63% | Fresh plans push the block away from near-target alignment; two interruptions do not establish recovery. Later pushing worsens it again. |
| 81 | 119 | 93.62% / 234 | 93.62% | Block pose remains unchanged after 234 through 300. The earlier interruption at 114 is far before this final plateau. |
| 82 | 112 | 94.81% / 125 | 84.65% | A fresh eight-movement prefix disrupts near-target alignment; the next prediction also reduces overlap. |

All six fail to **ever** cross 95%, not merely to end above it. We must not relax the criterion or add movement budget to turn these timeouts into successes.

The fixed-twelve control does not supply one universal answer either: it succeeds on 59 and 80, but fails on 54, 78, 81 and 82. Differences across these later trajectories are descriptive, not matched-state action comparisons.

## 2. A concrete stall example: predictions continue, but the block does not move

### Case 78

- After movement **197**, true coverage is **94.5736%**.
- The block's recorded position and angle remain exactly unchanged through movement **300**: **103 additional movements**.
- During that interval, new predictions occur before **205, 213, 221, 229, 237, 245, 253, 261, 269, 277, 285 and 293**: **12 calls**.
- The pusher travels **106.11 simulator coordinate units** while the block stays still. This is not a paused simulation or missing model calls.
- The last twelve-movement extension was after **136**. There is no early interruption in this attempt; its near-target stage already predicts again every eight movements.

In plain words: **the AI keeps proposing movements and the pusher keeps moving, but those movements do not produce the final effective push.** The immediate late-stage problem is not that a twelve-movement unit prevents another prediction. Earlier reuse may still have changed the trajectory indirectly; the saved data do not rule that out.

### Case 81

After **234**, the block stays at **93.6170%** coverage through **300**, spanning **66 additional movements**. There are **nine** new predictions, before **235, 243, 251, 259, 267, 275, 283, 291 and 299**. The pusher travels **162.11 coordinate units** while the block does not move. Its last extension was after **96**.

The diagnostic runner uses block-origin translation and wrapped angle changes of at most `1e-9` per movement to summarize stationary intervals. The two examples above also pass exact recorded pose equality in the independent raw inspection. This is block stationarity, not an inference about contact force or the whole scene being still.

## 3. A different problem: new plans can keep pushing in an unhelpful way

**Case 80:** after **144**, coverage is **91.75%**. The new plan begins at **145** and lowers coverage to **80.61%** after **148**, when the controller interrupts. A later new plan starts at **157** and lowers it to **41.81%** after **160**, triggering the second interruption. The attempt eventually ends at **39.63%**.

**Case 82:** after ordinary eight-action trimming at **164**, coverage is **94.67%**. The new plan starts at **165**, lowering coverage to **81.76%** after **170**, when interruption occurs. The new prediction beginning at **171** continues the decline to **60.27%** after **175**. The attempt ends at **84.65%**, below its earlier **94.81%** peak.

These examples occur inside **fresh predictions**, not simply the extra four cached movements of a twelve-movement extension. They show that requesting a new plan does not guarantee an effective correction. They do not show whether a particular alternative action would have succeeded.

## 4. Why not label every stall or large decline as failure?

Successful attempts contain both patterns:

| Successful adaptive case | Temporary problem | What eventually happens |
|---|---|---|
| 53 | Block stays fixed after 174–209 at 91.75%, with four new predictions during 35 stationary movements. | Succeeds after 219. |
| 75 | Block stays fixed after 131–168 at 90.19%, with four new predictions during 37 stationary movements. | Succeeds after 170. |
| 84 | Coverage drops from 92.37% after 93 to 42.66% after 116. | Regains the earlier overlap after 220 and succeeds after 245. |
| 89 | Coverage drops from 94.00% after 146 to 45.06% after 172. | Succeeds after 275. |

Across all 40 adaptive attempts, near-target stationary runs of at least 24 movements occur in **four lost cases** (59, 78, 81, 82), **two rescued successes** (53, 67), and **four retained successes** (61, 73, 75, 87). Thus a stall alone is not a reliable terminal-failure label. The 24-movement and 90% bins are exploratory descriptions, not newly proposed controller thresholds.

Do not compare only stationary **tails** of timeouts against successful endings: successful recordings stop at the first crossing of 95%, which inherently biases that comparison. Nonterminal successful stalls above are included explicitly to avoid this mistake. Likewise, a low endpoint displacement can conceal substantial out-and-back movement, so the runner retains total motion path separately.

Earlier suffix inspection also has counterexamples. Lost **78**, movements **21–24**, drops from **31.45% to 14.75%**. But successful **76**, movements **129–132**, drops from **70.12% to 55.96%** during an extension and still completes at **212**. A local suffix decrease is not proof that extending caused terminal failure.

## 5. Does inaccurate camera feedback explain all six?

No. Camera coverage exceeds 95% while true coverage remains at or below 95% in lost **54, 59 and 82** (19, one and one saved samples), but never in lost **78, 80 or 81**. The same overestimation occurs in successful **65, 69 and 84** (15, one and 22 samples).

Camera estimates therefore have near-target limitations, but overestimation is neither common to every failure nor unique to failures. The simulator's true coverage, not the derived camera scalar, checks success. The action model itself does not receive this derived scalar. Do not claim that a false camera “finished” reading caused these timeouts.

For context, final physical pose errors in 78/81 are only **2.20/1.51 coordinate units** and **2.52°/1.18°** relative to the target's body origin `(256,256)` and angle `π/4`. These are offline truth diagnostics, not privileged runtime inputs. Translation and orientation interact through the T shape: neither error alone is a monotonic replacement for coverage.

## What we can and cannot conclude

**Established observations:** all six approach the target without finishing; repeated near-target predictions can produce no object motion; new plans can continue destructive pushing; successful attempts can exhibit the same temporary symptoms.

**Not established:** a single failure classifier, a common causal defect in the adaptive controller, a camera-error explanation, or a recovery method that improves success. Identical scene/prediction seeds pair noise by prediction-call order, not physical time. Once methods execute different actions or query at different times, subsequent observations and predictions differ. Saved baseline success is not proof that forcing one change at a saved adaptive moment will recover it. Earlier 54/59 resampling already showed prediction sensitivity; those completed trials were not repeated.

My recommendation is to discuss a **bounded final-positioning recovery strategy** for repeated ineffective pushes, while separately considering destructive final pushes. Any proposal must use observable images/current commands, retain useful natural repositioning, and be tested from matched states against continuing unchanged. Merely adding another alignment-decline threshold is not supported, and the rejected filters remain stopped. No recovery strategy is implemented or adopted here.

Keep the milestone as the preserved main method, but retain the [completed comparison's mixed conclusion](2026-10-13-robotics-cohort-completion.md): its original improvement did not replicate. These diagnoses do not convert it into a demonstrated general improvement. All inspected scenes are development information for any revised method, not a fresh evaluation set.

## Reproduction and verification

From the project directory:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_loss_review.py
```

This standard-library CPU command verifies an existing matching result rather than overwriting it. For a disposable repeat, use `--output-dir results\robotics-loss-review-repeat`. It does not load the neural policy or create new model calls.

[analysis.json](../results/robotics-loss-review/analysis.json) contains all definitions, groups, 120 attempt summaries, source references, 289 extension windows, 22,657 recorded movements and 2,411 recorded predictions. The original 50–69 extension windows are reused from the earlier verified analysis; no completed GPU task was redone.

All **336 robotics CPU tests passed**, including 42 new tests for indexing, strict completion, wrapped pose/path calculations, recovery versus success, finite camera rows, stationary intervals, source integrity and protected outputs. The [independent audit](../results/robotics-loss-review/independent-audit.json) passed: it independently recomputed every diagnostic and group summary from raw evidence without importing the review module, checked all 457 bindings and 40 common-prefix pairs, and recorded the analysis/auditor hashes. It did not rerun inference, physics, image fitting or latent observation/RNG histories; prior audit limitations remain explicit.

The staged checkout reproduced the saved analysis exactly and passed all 336 tests. Checkout attributes preserve the original byte format of the earlier audit referenced by this review; its contents and all original experiment evidence remain unchanged.

## Sources

- [Completed baseline comparison](2026-10-13-robotics-cohort-completion.md), [attempt index](../results/robotics-cohort-completion/episodes.json), and [independent audit](../results/robotics-cohort-completion/independent-audit.json): paired outcomes and new fixed-eight/fixed-twelve evidence on 70–89.
- [Milestone evaluation](2026-10-12-robotics-fresh-evaluation.md) and [original attempts](../results/robotics-fresh-evaluation/episodes.json): preserved 50–69 baseline/adaptive records.
- [Gate comparison](2026-10-13-robotics-weighted-gate.md) and [attempt index](../results/robotics-weighted-gate/episodes.json): unchanged adaptive records on 70–89, reused without rerunning rejected methods.
- [Completed decision review](2026-10-13-robotics-decision-review.md), [saved analysis](../results/robotics-decision-review/analysis.json), and [earlier regression diagnosis](2026-10-12-robotics-regression-diagnosis.md): completed 54/59 evidence and explicit distinction between overlap recovery and task success.
- [Camera estimator implementation](../experiments/robotics_visual_monitor.py): known static target/geometry, derived camera inputs and estimator limitations; no new image fitting is performed here.

This task contributes a reproducible failure diagnosis, not a new policy, proven remedy or causal experiment.
