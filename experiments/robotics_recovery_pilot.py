"""Freeze the bounded recovery-rehearsal pilot before preparing/training data.

Only the final convolution block is learned; the comparison reuses the original
fixed-eight and milestone evaluators. No adaptive controller is changed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path


OUTPUT = Path("results/robotics-recovery-pilot")
SOURCES = ["robotics_recovery_pilot.py", "robotics_recovery_dataset.py",
           "robotics_recovery_training.py", "robotics_recovery_evaluation.py",
           "robotics_recovery_feasibility.py", "robotics_recovery_data_audit.py"]
TRAINING = {"optimizer_steps": 600, "accumulation": 4, "lr": 1e-5,
            "weight_decay": 0., "seed": 51000, "validation_seed": 51001,
            "validation_windows_per_group": 32,
            "validation_steps": [0, 200, 400, 600],
            "checkpoint_steps": [200, 400, 600], "selection": "final_600_only"}
EVALUATION = {"scenes": list(range(150, 170)), "draw": 50000,
              "arms": ["fixed8", "milestone12", "recovery12"], "max_steps": 300,
              "rotation": "scene_index_mod_three",
              "success": "true_coverage_strictly_greater_than_0.95"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f"Frozen artifact changed: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_output(output):
    assert Path(output).resolve() == OUTPUT.resolve(), "Use the dedicated recovery-pilot directory."


def protocol_document():
    import robotics_generation_length as gen
    import robotics_recovery_dataset as dataset
    import lerobot.policies.normalize as normalization

    inherited = gen.protocol_document()
    sources = {**inherited["fingerprints"],
               **{f"experiments/{name}": sha(Path("experiments") / name) for name in SOURCES},
               "results/robotics-recovery-feasibility/data-audit.json":
                   sha("results/robotics-recovery-feasibility/data-audit.json"),
               (gen.base.MODEL / "config.json").as_posix(): sha(gen.base.MODEL / "config.json"),
               (gen.base.MODEL / "model.safetensors").as_posix(): gen.base.WEIGHTS_SHA256,
               "requirements-robotics.txt": sha("requirements-robotics.txt")}
    return {
        "scope": "One bounded final-block recovery-rehearsal training pilot; 60 new evaluation attempts",
        "date_utc": "2026-10-16", "milestone": gen.MILESTONE,
        "data_selection": dataset.selection_contract(), "training": TRAINING,
        "training_method": {
            "trainable_prefix": "diffusion.unet.final_conv.",
            "microbatch": 1, "microbatches_per_update": {"expert": 2, "recovery": 2},
            "sampling": "uniform scene, then episode within scene, then complete native H16 window",
            "loss": "existing checkpoint-normalized random-timestep epsilon MSE, equal group weighting",
            "all_modules_eval": True, "encoder_frozen": True,
            "selection": "Final update600 only; validation losses diagnostic, not checkpoint selection",
            "stop": "Nonfinite loss/gradient/weights, memory exhaustion or provenance/shape mismatch",
            "no_test_tuning": True, "no_online_learning": True},
        "evaluation": EVALUATION,
        "evaluation_method": {
            "fixed8": "Original checkpoint + unchanged fixed-eight execution",
            "milestone12": "Original checkpoint + unchanged milestone8/12 execution",
            "recovery12": "Final600 trained output block + same unchanged milestone8/12 execution",
            "primary_comparison": "recovery12 minus milestone12; fixed8 secondary reference",
            "prediction_seed": "scene plus50000; pair randomness by call order, not identical trajectories",
            "new_attempts": 60, "samples_per_scene_per_arm": 1,
            "smoke": "Original->trained->original head switching on inspected scene0, draw1234; sixteen movements",
            "test_data": "Fresh simulator initial-scene seeds150..169; no claim of unseen original pretraining data",
            "primary_metric": "successful pushes",
            "other_metrics": ["mean final coverage", "mean policy calls", "mean complete-attempt seconds",
                              "mean fresh-prediction seconds"],
            "uncertainty": "Paired scene bootstrap; twenty scenes and one model-training seed are a pilot",
            "statistics_seed": 20261016, "bootstrap_samples": 10000,
            "runtime_inputs": "RGB, own pusher position and unchanged camera-monitor features; no block truth"},
        "generation": {"horizon": 16, "observations": 2, "denoising_steps": 100,
                       "extra_candidates": 0, "extra_learned_models": 0},
        "reused_milestone_rules": gen.adaptive.RULES,
        "checkpoint": inherited["checkpoint"],
        "packages": {**inherited["packages"],
                     **{name: importlib.metadata.version(name) for name in ["av", "pyarrow", "pandas"]}},
        "model_sources": {**inherited["model_sources"],
                          normalization.__name__: sha(inspect.getfile(normalization))},
        "fingerprints": dict(sorted(sources.items())), "timing": inherited["timing"],
        "limitations": ["Correlated successful self-generated recovery labels are not expert corrections.",
                        "Public expert AV1 is lossy; local recovery RGB is exact-replay verified.",
                        "Expert rehearsal split is episode-disjoint, not independent of pretrained checkpoint data.",
                        "Unchanged architecture does not guarantee unchanged episode length or wall runtime.",
                        "A negative pilot does not refute every fine-tuning design; no hidden hyperparameter search."]}


def verify(output=OUTPUT):
    check_output(output)
    document = read(Path(output) / "protocol.json")
    assert document == protocol_document(), "Frozen protocol, sources, dependencies or settings changed."
    return document


def prepare(output=OUTPUT):
    check_output(output)
    path = Path(output) / "protocol.json"
    if path.exists():
        return verify(output)
    assert not any((Path(output) / name).exists() for name in
                   ["dataset.json", "training.json", "eval-bindings.json", "episodes.json", "summary.json"])
    document = protocol_document()
    write_once(path, document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "verify"], required=True)
    args = parser.parse_args()
    document = globals()[args.stage]()
    print(json.dumps({"training": document["training"], "evaluation": document["evaluation"],
                      "bound_sources": len(document["fingerprints"])}, indent=2))


if __name__ == "__main__":
    main()
