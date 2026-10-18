# Can recovery-focused fine-tuning fit our laptop?

Date: October 15, 2026 (UTC).

## Conclusion

**A limited update is feasible on the GTX 1650, and existing recordings provide a small recovery-data starting pool. We have not trained or evaluated an improved pushing policy.**

One genuine supervised update of the existing model's final output block completed with **1,100.42 MiB peak allocated GPU memory** and **2,579.91 MiB peak process RAM**. The update changed only the selected block; the entire model was then restored exactly. No learned checkpoint was saved. The milestone controller, downloaded checkpoint and earlier experiments remain unchanged.

In plain words: we can afford to try teaching a small part of the model better movements. We also have examples where the robot recovered from difficult situations. These checks do **not** tell us whether learning from those examples will increase successful pushes.

## Objective and scope

The primary objective is more successful pushes than the milestone, while keeping per-prediction computation and average total runtime approximately at the milestone's level. Extra candidate predictions and speed-only changes are not the selected direction.

The proposed method would update a limited subset of existing policy weights offline, then deploy the same network architecture, H16 generation, 100 DDPM denoising iterations and unchanged 8/12 execution controller. No additional network or candidate search is planned at deployment. Actual whole-attempt calls/runtime must still be evaluated: unchanged architecture does not guarantee unchanged episode length.

This task completes only two feasibility checks:

1. Audit existing successful recovery recordings and their training-label provenance.
2. Perform one reversible supervised gradient/optimizer update using a real recorded recovery example.

There was no full training run, new pushing attempt, policy prediction, fine-correction controller, hyperparameter search or adoption of a new method. One-time data preparation/training costs are separate from deployment cost and must not be hidden in future comparisons.

## Existing data audit

The audit uses only discovery scenes **0–29**, reusing the original failure-pattern investigation and benefit-pilot continuations. Previously inspected evaluation scenes 30–149 are excluded from the initial candidate pool. They also cannot become unseen evaluation data for a later revision.

| Existing recording category | Count | Interpretation |
|---|---:|---|
| Successful continuations of originally failed baseline cases | 31 | Potential recovery examples, from eight scenes and nine branch contexts |
| Complete, unpadded native H16 windows in those continuations | 2,760 | Overlapping windows, not independent demonstrations |
| Exact-content duplicate recovery windows | 0 | Different hashes do not prove meaningful diversity |
| Ordinary successful baseline episodes | 20 | Separate potential rehearsal controls, not recovery examples |
| Complete windows from ordinary successful baselines | 2,686 | Not a replacement for obtaining the original expert dataset |
| Successful continuations of already-successful cases | 36 | Useful controls; not rescues of a failed original |
| Unsuccessful selected continuations | 59 | Excluded from the positive imitation-label pool |

Recovery scenes are **2, 3, 4, 6, 7, 15, 16 and 18**. The audit verifies 293 source/evidence bindings, recorded branch-state continuity, paired arms, actual executed commands, final success and complete label windows. It does not infer that replanning caused recovery or that every command in a successful trajectory is optimal.

The traces contain **image hashes, not RGB arrays**. Therefore, their existence alone does not provide a ready-to-train image dataset. Images need reconstruction by exact recorded-action CPU replay. For this task, only the selected smoke example's required observations were reconstructed and checked; the whole candidate pool was not rendered or exported.

Eight scenes are enough to motivate a bounded pilot, not to establish training-data adequacy or generalization. Branches share histories, and neighboring windows overlap heavily. Any future training/validation split must be by complete initial scene, not random windows. Earlier pilot partitions are historical context, not a new holdout.

## Correct image/action alignment

The selected example is **scene 2, prediction sample 2001, `replan_once`, branch after movement 157**. Its original baseline failed; the saved continuation succeeded after 284 total movements.

| Item | Recorded movement index |
|---|---|
| Previous camera/pusher observation | After 156 |
| Current camera/pusher observation | After 157 |
| Sixteen executed-command targets | 157 through 172, inclusive |
| Target slot zero | Historical command 157 |
| First usable future target | Command 158, at slot one |

This matches the pretrained model's two-observation history and H16 output convention. Targets are **actually executed commands**, not a predicted chunk's unused tail. All sixteen labels exist; no initial/end padding or invented post-terminal actions are used.

CPU replay verified the original reset plus 157 recorded movements. RGB hashes and physical coordinates match exactly; only the existing diagnostic coverage rounding allows `1e-12` tolerance. Block state and coverage are validation/selection information, never model inputs. The model receives only RGB images, the pusher's own positions and executed-command training labels, through its existing checkpoint normalization.

## Real limited-update gate

The gate freezes all parameters except `diffusion.unet.final_conv`, the existing final convolution block. This includes a 512-to-512 convolution/normalization block and the final 512-to-2 projection, not merely the last projection. All modules remain in evaluation mode to preserve frozen image encoding and deterministic center cropping. Gradients still update the selected parameters in evaluation mode.

The loss uses the installed LeRobot supervised denoising path: add noise at **one sampled diffusion timestep**, predict that noise, and compute mean-squared error. It does **not** backpropagate through a 100-step generation chain.

| Measured/checkable item | Result |
|---|---:|
| Full policy parameters | 262,709,044 |
| Updated final-block parameters | 1,313,282, approximately 0.50% |
| Optimizer | AdamW, learning rate `1e-5`, weight decay zero |
| Denoiser forward evaluations in the update | 1 |
| Sampled training diffusion timestep | 42 of 100 |
| Supervised loss | 0.10540016 |
| Gradient L2 norm | 2.30667162, finite and nonzero |
| Parameters with allocated Adam state | Six tensors |
| Adam state storage | 10,506,280 bytes |
| Isolated first update time | 1.30 seconds |
| Peak allocated / reserved GPU memory | 1,100.42 / 1,182.00 MiB |
| Peak observed process RAM | 2,579.91 MiB |
| Minimum observed available system RAM | 3,366.07 MiB |
| Frozen parameters and buffers unchanged | Passed |
| Entire model restored exactly afterward | Passed |
| Bound source/checkpoint files unchanged | Passed |

The optimizer's first real step allocates its state, so this is stronger than merely loading the model or computing a loss. A 2,800 MiB allocator limit and a free-system-RAM check bounded the probe. The first preflight stopped before model loading because it counted Windows' own virtual-environment Python launcher as a second job; inspection identified that parent, and the guard was corrected to exclude only this invocation and its ancestors while still blocking unrelated Python jobs.

The 1.30-second figure is a cold, single supervised update, excluding loading, replay, hashing and restoration. It is **not** prediction latency, training convergence speed or an estimate of whole-experiment duration. Neither this loss value nor its gradient establishes improved behavior.

## Decision and next boundary

**Proceed to planning a small recovery-focused training pilot, not a full-scale training run.** The hardware gate passes for one sample and this exact final-block update. The data audit establishes availability of candidate executed labels, not that 2,760 windows will be sufficient.

Before a pilot, prepare and verify the remaining RGB training examples, obtain compatible original expert demonstrations for rehearsal, and freeze whole-scene training/validation splits. Then define a small training budget and separate fresh evaluation scenes. More successful pushes is the primary decision criterion; actual per-prediction time, number of predictions and total runtime remain required cost checks.

Limited final-block training may lack enough flexibility, and successful recordings may not contain the corrections needed for other failures. Mixing original demonstrations can reduce forgetting but does not guarantee its absence. No new fine-tuned policy is adopted, and no reported success-rate gain follows from this task.

## Reproduction and evidence

Use `.venv-robotics`. The CPU-only audit verifies an existing matching output rather than overwriting it:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_data_audit.py
```

The GPU update refuses to overwrite its completed gate. For a deliberate disposable repeat, close unrelated Python/GPU collectors and use a new directory:

```powershell
.\.venv-robotics\Scripts\python.exe experiments\robotics_recovery_feasibility.py --output-dir results\robotics-recovery-feasibility-repeat
New-Item -ItemType Directory -Path tmp\recovery-feasibility-tests -Force | Out-Null
.\.venv-robotics\Scripts\python.exe -c "import tempfile,unittest,sys; from pathlib import Path; tempfile.tempdir=str(Path('tmp/recovery-feasibility-tests').resolve()); result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests')); sys.exit(not result.wasSuccessful())"
```

Evidence: [data audit](../results/robotics-recovery-feasibility/data-audit.json), [training gate](../results/robotics-recovery-feasibility/training-gate.json), and [independent audit](../results/robotics-recovery-feasibility/independent-audit.json). The independent audit inspects records, provenance and implementation; it does not repeat physics, gradients or GPU work.

All **572 CPU tests passed**, including 43 new data-alignment, provenance, optimizer and restoration tests. The first full-suite invocation encountered Windows sandbox temporary-directory rename permissions in 14 older tests; pointing Python's `tempfile.tempdir` at the project-local `tmp/recovery-feasibility-tests/` resolved them without editing frozen code. The successful suite ran in 22.94 seconds. This is test-fixture verification, not additional pushing experiments.

## Sources and reused work

1. Lin et al., [ReGuide: From Test-Time Guidance to Self-Improving Diffusion Policies](https://arxiv.org/html/2606.28939v1), 2026 preprint; [official implementation](https://github.com/tzuhsiangl/reguide). Inspiration: absorb successful recovery rollouts into an existing policy using original-data rehearsal, then evaluate without online guidance. Their Robomimic results do not establish our PushT or 4 GB training feasibility. This task does not reproduce their world-model/guidance pipeline.
2. [SIME: Enhancing Policy Self-Improvement with Modal-level Exploration](https://arxiv.org/html/2505.01396v1), 2025; [official implementation](https://github.com/EricJin2002/SIME). Related precedent for selecting useful self-collected trajectories for policy refinement; their training scale is not our laptop recipe.
3. Chi et al., [Diffusion Policy](https://arxiv.org/abs/2303.04137); [LeRobot v0.3.3 implementation](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/diffusion/modeling_diffusion.py), [pinned public checkpoint](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653), and [Gym-PushT](https://github.com/huggingface/gym-pusht). These supply the policy, supervised loss, normalization and physical environment. Our new contribution at this stage is the bounded provenance/resource feasibility check, not an improved learned controller.
