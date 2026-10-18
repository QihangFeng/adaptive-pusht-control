# Recovery-rehearsal fine-tuning pilot

## Conclusion and question

**The recovery-trained version improved all four observed metrics in this twenty-scene pilot, with essentially unchanged observed time per prediction.** It finished 17/20 pushes, versus the unchanged milestone's 13/20 and the original baseline's 14/20. This is a promising candidate, not yet a demonstrated stable improvement: the sample is small, and it still lost some successes. Preserve the designated milestone unchanged as the recovery reference.

Question: can a small offline update teach more useful pushing movements while retaining the milestone's prediction architecture and 8/12 execution rule?

The comparison uses twenty new simulator scenes, seeds 150–169, with one prediction sample per method: sixty full attempts. All three methods start from the same physical scene. The main comparison is the trained model plus milestone execution versus the original model plus milestone execution; the original fixed-eight baseline is the second reference.

## What changes—and what does not

| Method | Neural weights | Execution rule |
|---|---|---|
| Original baseline | Original public checkpoint | Predict, execute eight commands, predict again |
| Unchanged milestone | Original public checkpoint | Original camera/path-guided 8/12 reuse and bounded interruptions |
| Recovery-trained milestone | Only the existing final convolution block updated | Exactly the same milestone rule |

There is no new online model, candidate search, controller threshold, generation horizon or denoising schedule. Every prediction still generates H16 using 100 DDPM iterations. The two-observation convention leaves fifteen future commands available; execution still uses the existing eight/twelve mechanism.

This is a course-project adaptation of recovery-data policy refinement, not a claim that recovery fine-tuning is a new algorithm. ReGuide provides relevant precedent for mixing successful self-collected trajectories with original demonstrations and then deploying the updated policy without online guidance [1]. Our data collection, small final-block update and PushT evaluation are different from its world-model/guidance pipeline. SIME provides related self-improvement precedent [2].

## Frozen data and training protocol

The [previous feasibility task](2026-10-15-robotics-recovery-feasibility.md) already verified one real update on the GTX 1650. This task does not repeat that experiment.

- Recovery data: 31 previously successful continuations from eight originally failed discovery scenes, selected by final task completion—not by immediate alignment gain. The failed prefix supplies observation history; only complete post-branch target windows enter the recovery pool.
- Recovery training scenes: 2, 3, 4, 6 and 7. Recovery validation scenes: 15, 16 and 18. Entire initial scenes remain on one side of the split.
- Expert rehearsal: the first eighty public PushT demonstration episodes at the pinned LeRobot dataset revision [3]. Episodes 0–63 are training; 64–79 are validation. Selection is by index, not outcome. Expert videos use lossy AV1; they are not exact simulator RGB. The checkpoint may already have seen these demonstrations during original pretraining.
- Twenty saved ordinary baseline successes are prepared as diagnostic controls only. They are not substitutes for original expert demonstrations and do not enter training windows.
- Recovery images are reconstructed by CPU replay of the actually executed commands. All saved RGB hashes and physical quantities must match; only the pre-existing coverage-rounding tolerance of `1e-12` is allowed. The replay does not reset at a branch.
- One example contains observations at indices `j,j+1` and executed command targets `j:j+16`. Target index zero is historical; index one is the next usable future command. No padding or episode crossing is allowed.

The initial preparation stopped before training because expert `next.done` was true in the last two rows, not only the final row. This occurred in all eighty episodes. The [official converter](https://github.com/huggingface/lerobot/blob/a4d77b99f0d731a5e82b0feab7eb8341de475fb3/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py#L143) marks the final row done, then shifts/repeats that flag for `next.done`; this is not evidence of an internal reset or task success. The revised preparation validates the full video's chronology, then conservatively retains rows through the first done flag inclusive. It drops one final row per expert episode and eighty possible windows overall, retaining all eighty demonstrations without success filtering. The [stopped protocol](../results/robotics-recovery-pilot/preflight/README.md) and local partial cache are preserved; the new protocol is frozen before any training or test outcome.

Prepared training/validation pools:

| Pool | Eligible recorded trajectories | Complete H16 windows |
|---|---:|---:|
| Expert training | 64 | 6,818 |
| Expert validation | 16 | 1,677 |
| Recovery training | 18 from five initial scenes | 1,737 |
| Recovery validation | 11 from three initial scenes | 1,023 |

All 31 recovery recordings are preserved and replay-verified; two have too few remaining executed commands for a complete post-branch H16 target. They provide no training window rather than receiving invented padding. The windows overlap and are not independent demonstrations. All eighty expert videos passed full original-frame checks; the largest video/parquet timestamp difference was `7.63e-7` seconds. Raw expert parquet/video downloads total 3,484,755 bytes, excluding metadata.

Only `diffusion.unet.final_conv` is trainable. The encoder, earlier denoiser blocks, normalization and model structure are frozen. All modules remain in evaluation mode so that image encoding/cropping stays deterministic; evaluation mode does not disable gradients for the selected output block.

The loss is the checkpoint's existing normalized epsilon-prediction loss [4]:

\[
\mathcal L(\theta_{\rm tail})=
\tfrac12\mathbb E_{(o,a)\sim D_{\rm expert},k,\epsilon}
\tfrac1{32}\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_k}a+
\sqrt{1-\bar\alpha_k}\epsilon,k,o)\|_2^2
+\tfrac12\mathbb E_{(o,a)\sim D_{\rm recovery},k,\epsilon}
\tfrac1{32}\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_k}a+
\sqrt{1-\bar\alpha_k}\epsilon,k,o)\|_2^2.
\]

The factor `1/32` averages over sixteen two-coordinate commands, matching the implementation's mean-squared error. Here `o` contains only RGB and the pusher's own position. Simulator block state, contact and coverage are selection/validation/scoring information, not neural inputs. Each supervised microbatch uses one randomly sampled noise timestep; it does not backpropagate through a 100-step action-generation chain.

The fixed budget is 600 AdamW updates, learning rate `1e-5`, zero weight decay. Each update accumulates four microbatches of one example: two expert and two recovery. Sampling is uniform over initial scene, then episode and complete window, to avoid giving a scene extra weight just because it has more saved branches. Training RNG seed is 51000.

Validation uses 32 fixed sample draws per group and fixed noise seeds, with seed 51001, at updates 0/200/400/600. The 64 draws cover 63 distinct windows. Validation is diagnostic only. **The final update 600 is the only deployed checkpoint**; there is no checkpoint or hyperparameter selection using validation or fresh-test outcomes. Nonfinite values, memory exhaustion or provenance/shape mismatches stop execution.

Only the six output-block tensors and the local optimizer/RNG resume state are saved. The original public checkpoint file is never overwritten. Local model/data artifacts remain Git-ignored; shareable manifests identify their hashes and reproduction instructions.

## Comparable evaluation

The evaluator reuses the unchanged [generated-length experiment's episode collector](../experiments/robotics_generation_length.py), with its original `fixed8` or `adaptive12` inner method. Separate outer directories distinguish the original and trained neural weights; the original collector is not rewritten or relabeled.

All sixty attempts are collected after training, sequentially on the same GPU. Method order rotates by scene. Each method uses prediction RNG seed `scene + 50000`, a 300-command limit and the existing success definition: simulator target coverage strictly greater than 95%. Paired seeds align random generation by call order, not by physical time or identical resulting trajectories.

Before fresh evaluation, a scene-0 switching check runs original fixed-eight, original milestone, trained milestone and restored original fixed-eight. Restored original actions must exactly reproduce the original and saved baseline smoke. The trained model is not required to reproduce original actions: its weights intentionally differ. Both original controls must share their first generated/executed prefix, and all methods must start from the same scene. Frozen backbone hashes and output-block identities guard against trained-weight leakage into controls.

Main reported metrics are successful pushes, mean final alignment, mean policy calls and mean complete-attempt runtime. Fresh-prediction time is a required supporting cost check. Episode timing includes monitoring/bookkeeping but excludes model loading and trace-file writing. One-time data preparation and training are reported separately, not hidden as free computation.

The primary contrast is trained milestone minus unchanged milestone. Paired scene bootstrap intervals are descriptive uncertainty estimates; twenty initial scenes, one training seed and one prediction sample per scene cannot establish stable general superiority. Fresh simulator seeds are unseen in this project's tuning, not proof of unseen original-model pretraining data.

## Results

All sixty fresh attempts finished. No scored attempt was excluded or replaced.

| Method | Successful pushes | Average final alignment | Average AI calls | Average runtime |
|---|---:|---:|---:|---:|
| Original fixed-eight baseline | 14/20 | 92.87% | 25.05 | 122.89 s |
| Unchanged milestone 8/12 | 13/20 | 88.75% | 22.50 | 118.95 s |
| Recovery-trained model + unchanged 8/12 | **17/20** | **94.47%** | **18.40** | **97.62 s** |

Compared with the milestone, the trained version completed four more pushes, used 18.22% fewer predictions and took 17.94% less time on average. Compared with the original baseline, it completed three more pushes, used 26.55% fewer predictions and took 20.57% less time. Average time per fresh prediction was 4.8449 seconds for the trained model, 4.8453 for the milestone and 4.8781 for fixed-eight. Thus the observed saving comes from fewer predictions per attempt, not cheaper neural predictions or an altered execution rule. Mean executed movements were 175.60, 207.25 and 196.35, respectively.

The paired outcomes show both gains and regressions:

| Reference | Scenes changed from failure to success | Scenes changed from success to failure |
|---|---|---|
| Unchanged milestone | 151, 152, 153, 157, 163, 166 | 150, 167 |
| Original fixed-eight | 151, 152, 156, 169 | 167 |

For example, scene 167 succeeded under both original controls but failed with the trained model. The pilot therefore does **not** establish that training preserves every previously successful case, or fixes the older inspected losses such as 54/59.

The primary paired uncertainty estimates remain broad:

| Trained minus unchanged milestone | Observed difference | Paired-scene bootstrap 95% interval |
|---|---:|---:|
| Success rate | +20 percentage points | -5 to +45 percentage points |
| Final alignment | +5.72 percentage points | -0.40 to +14.74 percentage points |
| AI calls per attempt | -4.10 | -9.60 to +1.40 |
| Runtime per attempt | -21.34 s | -50.40 to +7.81 s |

All four primary intervals include zero. The exact two-sided discordant-success test gives `p=0.289` against the milestone and `p=0.375` against fixed-eight. Against fixed-eight, the call and runtime difference intervals exclude zero in this sample, but the success and alignment intervals do not. These results support continuing with this frozen candidate; they do not justify calling its success advantage statistically established or generally reliable. See the [complete summary](../results/robotics-recovery-pilot/summary.json) for both paired comparisons.

### Offline training and laptop cost

The run completed exactly 600 optimizer updates and 2,400 supervised denoiser forwards. It updated 1,313,282 of 262,709,044 parameters (about 0.50%) and verified that every frozen parameter/buffer remained unchanged. Training, diagnostic validation and checkpoint I/O took 201.12 seconds, excluding downloads, data preparation and model loading. Peak allocated/reserved GPU memory was 1,111.08/1,182 MiB; sampled peak process RAM was 1.73 GiB, with 3.96 GiB minimum available system RAM. This is a completed laptop training result, not just the earlier one-update memory gate.

| Fixed-noise validation loss | Original | Update 200 | Update 400 | Final update 600 |
|---|---:|---:|---:|---:|
| Expert | 0.002244 | 0.002974 | 0.003554 | 0.004444 |
| Recovery | 0.018247 | 0.016977 | 0.016980 | 0.017235 |
| Equal-group mean | 0.010246 | 0.009976 | 0.010267 | 0.010839 |

The final recovery loss fell slightly, while expert loss increased and the balanced loss was worse than the original. These mixed diagnostics do not establish task-level forgetting or predict the observed pushing improvement. The predeclared final update 600 is the only evaluated learned checkpoint; update 200 was not selected after seeing these numbers.

The full evaluation invocation took 114.65 minutes, including model loading, the switching smoke check, scored attempts and verification, with sampled peak process RAM of 2.54 GiB and minimum available system RAM of 3.00 GiB. All three methods peaked at 1,095.41 MiB allocated GPU memory. This verifies the complete bounded preparation/training/evaluation workflow on the laptop, not large-model training or real-time control.

### Verification

An independent CPU data audit verified all 460 source bindings, 131 NPZ hashes and 11,255 complete windows, with no diagnostic controls in training. It also compared all 10,715 prepared recovery/control RGB frames against recorded hashes and all 10,664 executed commands/pusher positions against original joined traces. Expert labels/terminal truncation were independently checked against the eighty parquets; the independent audit did not repeat AV1 decoding. All 655 CPU tests passed before training.

The independent results audit checked all sixty record/sidecar/trace bindings, 11,584 movements, 1,319 predictions and 780 saved controller-event decisions. It independently reproduced the comparison means, gained/lost cases, bootstrap intervals and exact tests, and verified common initial scenes and the original controls' shared first prediction prefix. Training and learned-weight checks are recorded in the [independent audit](../results/robotics-recovery-pilot/independent-audit.json). These checks do not repeat neural inference, physics, camera fitting or hardware timing.

## Interpretation and limitations

The change addresses a different limitation from our earlier timing experiments: it changes the movements the frozen architecture proposes, rather than only deciding when to ask it again. In this pilot, the updated model reached the task's success threshold more often and with fewer predictions. The online system remains the same size and uses the same 8/12 controller. The additional cost is offline data preparation and about 3.35 minutes of bounded training, excluding preparation/loading.

However, this experiment isolates the combined fine-tuning recipe from unchanged weights—not the separate effects of recovery data, expert rehearsal or the chosen trainable block. Without expert-only/recovery-only comparisons, we cannot claim recovery labels specifically caused the gain. We also have only one training seed, one prediction sample for each of twenty scenes, correlated overlapping recovery windows from few discovery scenes, and clean two-dimensional simulation. Expert AV1 images differ from exact simulator RGB. Hardware timing is specific to this laptop, and simulation pauses during inference; this is not a real-time or physical-robot result.

Recommendation: retain the exact trained checkpoint and protocol as a promising candidate, keep milestone `0f40449` intact, and discuss a further frozen evaluation before claiming general improvement. Seeds 150–169 are now inspected evidence, not fresh data for a revision. No threshold/weighted interruption rule or additional online candidate search is adopted.

## Reproduction

Use the existing Python 3.11 `.venv-robotics` environment. Do not run another GPU collector concurrently. Commands are resumable and refuse mismatched frozen inputs.

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_dataset.py --stage download
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_pilot.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_dataset.py --stage prepare
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_training.py
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_evaluation.py --stage run
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_dataset.py --stage verify
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_pilot.py --stage verify
```

## References

1. Lin et al., [ReGuide: From Test-Time Guidance to Self-Improving Diffusion Policies](https://arxiv.org/html/2606.28939v1), 2026 preprint; [official implementation](https://github.com/tzuhsiangl/reguide). Recovery-rehearsal inspiration, not an off-the-shelf PushT or 4 GB recipe.
2. [SIME: Enhancing Policy Self-Improvement with Modal-level Exploration](https://arxiv.org/html/2505.01396v1), 2025; [official implementation](https://github.com/EricJin2002/SIME). Related precedent; its exploration/training scale is not reproduced here.
3. [Pinned public PushT dataset](https://huggingface.co/datasets/lerobot/pusht/tree/a16737ffb6c8dbe935291b9fd2f36d14c9bf0820), LeRobot v2.0 conversion of the original expert demonstrations. Used as rehearsal data, with explicit AV1 and alignment checks.
4. Chi et al., [Diffusion Policy](https://arxiv.org/abs/2303.04137); [LeRobot v0.3.3 implementation](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py), [pinned checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653) and [Gym-PushT](https://github.com/huggingface/gym-pusht). These supply the pretrained model, loss, normalization, task and physical simulation.
