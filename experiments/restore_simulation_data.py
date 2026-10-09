"""Restore only the measured benchmark subset used by the original pilot.

Uses the normal local Hugging Face login; never accepts or prints credentials.
Raw and prepared data remain in ignored, persistent project directories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
REVISION = "fea6cb99d037d11f86f9c45a6991b9e5764f4529"
COMMIT = "c582a5115cfd49dcdfb140f1837224e7cbf0cfb1"
BATCHES = (8, 16, 32, 64, 128, 256, 512)
SPEC_HASH = "b473220852891f26"
ORIGINAL_WRITTEN_UTC = "2026-09-20T05:14:00+00:00"


def selected_files():
    prefix = ("llm/h100/current/run/llm/lm-arena-chat/results/"
              "meta-llama/Llama-3.1-8B-Instruct/H100/")
    files = ["runs/llm.parquet"]
    for batch in BATCHES:
        repeats = 1 if batch <= 128 else 2
        files.append(prefix + f"num_gpus+1+max_num_seqs+{batch}+num_request_repeats+{repeats}"
                     "+dataset_split+train+num_requests+1024+seed+48105/results.json")
    return files


def digest(path):
    # Raw result files can be tens of megabytes; hash without a second full copy.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(directory):
    return {p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def original_timestamp_hash(manifest_bytes):
    """Normalize provenance in memory only, never backdate the restored file."""
    current = json.loads(manifest_bytes)["written_utc"]
    return hashlib.sha256(manifest_bytes.replace(
        current.encode("utf-8"), ORIGINAL_WRITTEN_UTC.encode("utf-8"), 1
    )).hexdigest()


def fetch(cache):
    from huggingface_hub import hf_hub_download

    for i, filename in enumerate(selected_files(), 1):
        path = Path(hf_hub_download(
            repo_id="ml-energy/benchmark-v3", repo_type="dataset", revision=REVISION,
            filename=filename, cache_dir=str(cache),
        ))
        print(f"Cached selected file {i}/8 ({path.stat().st_size:,} bytes)", flush=True)
    return cache / "datasets--ml-energy--benchmark-v3" / "snapshots" / REVISION


def prepare(snapshot, prepared, verification):
    from mlenergy.data.records import LLMRuns
    from openg2g.controller.ofo import LogisticModelStore
    from openg2g.datacenter.config import InferenceModelSpec
    from openg2g.datacenter.workloads.inference import InferenceData

    if verification.exists() or (prepared.exists() and any(prepared.iterdir())):
        raise ValueError("Use new/empty prepared and verification paths; existing evidence is preserved")
    source = ROOT / "external/openg2g"
    git = ["git", "-C", str(source)]
    if subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("Simulator source revision does not match the original pilot")
    if subprocess.check_output(git + ["status", "--porcelain"], text=True).strip():
        raise ValueError("Simulator checkout contains changes")
    spec = InferenceModelSpec(
        model_label="Llama-3.1-8B", model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_model="H100", task="lm-arena-chat", precision="bfloat16",
        gpus_per_replica=1, tensor_parallel=1, itl_deadline_s=0.08,
        batch_sizes=BATCHES, feasible_batch_sizes=BATCHES,
    )
    if spec.cache_hash() != SPEC_HASH:
        raise ValueError("Model specification hash changed")
    runs = (LLMRuns.from_directory(snapshot, stable_only=False)
            .task(spec.task).model_id(spec.model_id).gpu_model(spec.gpu_model)
            .num_gpus(1).max_num_seqs(*BATCHES))
    if len(runs) != 7 or not all(run.is_stable for run in runs):
        raise ValueError("Expected exactly seven stable benchmark runs")
    if sorted(run.max_num_seqs for run in runs) != list(BATCHES):
        raise ValueError("Selected batch coverage changed")
    raw_before = {name: digest(snapshot / name) for name in selected_files()}
    data = InferenceData.generate((spec,), runs=runs, dt_s=0.1, seed=0, itl_sample_cap=2048)
    data.save(prepared, plot=False)
    models = LogisticModelStore.generate((spec,), runs=runs)
    models.save(prepared, (spec,), plot=False)
    current = fingerprint(prepared)
    original = json.loads((ROOT / "results/power-model-pilot-verified/manifest.json")
                          .read_text(encoding="utf-8"))
    expected = original["prepared_data_sha256"]
    manifest_key = f"{SPEC_HASH}/_manifest.json"
    numerical_matches = {k: current.get(k) == v for k, v in expected.items() if k != manifest_key}
    manifest_bytes = (prepared / manifest_key).read_bytes()
    timestamp_only = original_timestamp_hash(manifest_bytes) == expected[manifest_key]
    raw_unchanged = raw_before == {name: digest(snapshot / name) for name in selected_files()}
    required_versions = original["package_versions"]
    versions = {name: importlib.metadata.version(name) for name in required_versions}
    version_differences = {name: {"original": expected_version, "current": versions[name]}
                           for name, expected_version in required_versions.items()
                           if versions[name] != expected_version}
    record = dict(
        openg2g_commit=COMMIT, dataset_repo="ml-energy/benchmark-v3", dataset_revision=REVISION,
        python=sys.version, original_python=original["python"], package_versions=versions,
        package_version_differences=version_differences, selected_raw_sha256=raw_before,
        raw_inputs_unchanged=raw_unchanged,
        selected_raw_bytes=sum((snapshot / name).stat().st_size for name in selected_files()),
        prepared_data_sha256=current, numerical_artifacts_match_original=numerical_matches,
        manifest_matches_after_timestamp_only_normalization=timestamp_only,
        current_written_utc=json.loads(manifest_bytes)["written_utc"],
        original_written_utc=ORIGINAL_WRITTEN_UTC,
        manifest_policy="Retain new timestamp on disk; compare old timestamp in memory only",
        unexpected_prepared_files=sorted(set(current) - set(expected)),
    )
    verification.parent.mkdir(parents=True, exist_ok=True)
    with verification.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    if not (all(numerical_matches.values()) and timestamp_only and raw_unchanged
            and set(current) == set(expected) and not version_differences):
        raise RuntimeError("Restored inputs differ; inspect verification before running comparisons")
    print("All three numerical input files exactly match the original hashes.", flush=True)
    print("Manifest differs only by its truthful regeneration timestamp.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "prepare"))
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/huggingface")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--verification", type=Path,
                        default=ROOT / "results/simulator-restoration/verification.json")
    args = parser.parse_args()
    cache = args.cache.resolve()
    if args.action == "fetch":
        print(fetch(cache))
    else:
        snapshot = cache / "datasets--ml-energy--benchmark-v3" / "snapshots" / REVISION
        prepare(snapshot, args.prepared.resolve(), args.verification.resolve())


if __name__ == "__main__":
    main()
