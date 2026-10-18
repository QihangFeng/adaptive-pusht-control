"""Prepare original expert rehearsal and exact-replayed discovery recovery data.

No neural policy, inference, gradient update, image fitting or new trial. Public
expert RGB is decoded lossy AV1; locally replayed RGB must match old hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import numpy as np

import robotics_recovery_data_audit as audit


OUTPUT = Path("results/robotics-recovery-pilot")
CACHE = Path("data/prepared/robotics/recovery-pilot")
RAW = Path("data/raw/robotics/pusht-expert/hf-v2-a16737f")
REPO = "lerobot/pusht"
REVISION = "a16737ffb6c8dbe935291b9fd2f36d14c9bf0820"
EXPERT_EPISODES = list(range(80))
RECOVERY_TRAIN = [2, 3, 4, 6, 7]
RECOVERY_VALIDATION = [15, 16, 18]
EXPERT_TRAIN = list(range(64))
EXPERT_VALIDATION = list(range(64, 80))
HORIZON = 16


def read(path, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_new(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f"Existing artifact differs: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fetch_public(url, destination, expected_sha=None, expected_size=None):
    """Download bytes before exclusive write; never overwrite or leave partials."""
    destination = Path(destination)
    if destination.exists():
        payload = destination.read_bytes()
    else:
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        assert payload is not None
    if expected_size is not None:
        assert len(payload) == expected_size, f"Public source size differs: {destination}"
    if expected_sha is not None:
        assert hashlib.sha256(payload).hexdigest() == expected_sha, f"Public source hash differs: {destination}"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def expert_paths(episode):
    assert episode in EXPERT_EPISODES
    name = f"episode_{episode:06d}"
    return [f"data/chunk-000/{name}.parquet", f"videos/chunk-000/observation.image/{name}.mp4"]


def download_expert(raw_dir=RAW):
    raw_dir = Path(raw_dir)
    tree_url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"
    fetch_public(tree_url, raw_dir / "tree.json")
    tree = read(raw_dir / "tree.json")
    entries = {entry["path"]: entry for entry in tree}
    assert len(entries) == len(tree)
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
    for relative in ["meta/info.json", "meta/episodes.jsonl", "README.md"]:
        fetch_public(base + relative, raw_dir / relative)
    tasks = []
    for episode in EXPERT_EPISODES:
        for relative in expert_paths(episode):
            entry = entries[relative]
            assert entry["lfs"]["size"] == entry["size"]
            tasks.append((base + relative, raw_dir / relative, entry["lfs"]["oid"], entry["size"]))
    with ThreadPoolExecutor(max_workers=8) as executor:
        sizes = list(executor.map(lambda args: fetch_public(*args), tasks))
    assert len(sizes) == 160
    return {"repo_id": REPO, "revision": REVISION, "episodes": EXPERT_EPISODES,
            "data_video_files": len(sizes), "data_video_bytes": sum(v["bytes"] for v in sizes),
            "format": "206 original expert demonstrations converted to pinned LeRobot v2.0; first80 selected by index, not outcome"}


def expert_terminal_handling(columns):
    """Keep the first terminal transition, not the repeated endpoint row.

    The original LeRobot converter marks the last demonstration state done,
    shifts that vector by one for next.done, and repeats its last value. Hence
    pinned expert episodes have a trailing pair of True flags, not a reset.
    Success flags are never used to select or truncate expert demonstrations.
    """
    count = len(columns["frame_index"])
    done = columns["next.done"]
    assert len(done) == count and all(isinstance(v, (bool, np.bool_)) for v in done)
    indices = [i for i, value in enumerate(done) if value]
    if indices:
        assert indices == list(range(indices[0], count)), "Nonterminal rows follow a done flag; inspect reset semantics."
    retained = indices[0] + 1 if indices else count
    assert not any(done[:retained - 1])
    return {"original_rows": count, "retained_rows": retained, "discarded_rows": count - retained,
            "first_done_row_index": indices[0] if indices else None,
            "original_done_row_indices": indices, "success_used_for_filtering": False,
            "rule": "Validate all original frames/rows first; retain through first next.done inclusive, or all rows if none.",
            "semantics_source": "https://raw.githubusercontent.com/huggingface/lerobot/a4d77b99f0d731a5e82b0feab7eb8341de475fb3/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py"}


def validate_expert_arrays(columns, frames, presentation_times, episode, fps=10):
    assert len(frames) == len(columns["frame_index"]) > 0
    count = len(frames)
    assert columns["frame_index"] == list(range(count)), "Expert frame indices have gaps/reset."
    assert set(columns["episode_index"]) == {episode}
    timestamps = np.asarray(columns["timestamp"], dtype=np.float64)
    pts = np.asarray(presentation_times, dtype=np.float64)
    expected = np.arange(count, dtype=np.float64) / fps
    assert timestamps.shape == pts.shape == (count,)
    assert np.isfinite(timestamps).all() and np.isfinite(pts).all()
    assert np.max(np.abs(timestamps - expected)) <= 1e-5
    assert np.max(np.abs(pts - expected)) <= 1e-5, "Video PTS and parquet row alignment differ."
    states = np.asarray(columns["observation.state"], dtype=np.float32)
    actions = np.asarray(columns["action"], dtype=np.float32)
    assert states.shape == actions.shape == (count, 2)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    pixels = np.stack(frames)
    assert pixels.shape == (count, 96, 96, 3) and pixels.dtype == np.uint8
    # Validate the FULL decoded chronology before excluding any endpoint rows.
    # Cropping is a format correction, never outcome selection or end padding.
    retained = expert_terminal_handling(columns)["retained_rows"]
    return pixels[:retained], states[:retained], actions[:retained]


def load_expert(episode, raw_dir=RAW):
    import av
    import pyarrow.parquet as parquet

    table_path, video_path = [Path(raw_dir) / p for p in expert_paths(episode)]
    columns = parquet.read_table(table_path).to_pydict()
    frames, pts = [], []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        assert abs(float(stream.average_rate) - 10.) < 1e-9
        for frame in container.decode(video=0):
            assert frame.pts is not None and frame.time_base is not None
            pts.append(float(frame.pts * frame.time_base))
            frames.append(frame.to_ndarray(format="rgb24"))
    arrays = validate_expert_arrays(columns, frames, pts, episode)
    return arrays, {"frame_count": len(arrays[0]), "decoded_frame_count": len(frames), "fps": 10,
                    "terminal_handling": expert_terminal_handling(columns),
                    "maximum_timestamp_alignment_error_seconds":
                    float(np.max(np.abs(np.asarray(columns["timestamp"]) - np.asarray(pts)))),
                    "image_origin": "Pinned original-expert AV1 video decoded to RGB24; lossy pixels, not exact simulator RGB",
                    "decoder": "PyAV sequential decode with explicit per-frame presentation-time checks"}


def physical_state(env, observation):
    """Same recorded quantities as the unchanged original collector, no fitting."""
    base = env.unwrapped
    radius = next(iter(base.agent.shapes)).radius
    gap = min(shape.point_query(base.agent.position).distance - radius for shape in base.block.shapes)
    shapes = sorted(base.block.shapes, key=lambda shape: tuple(tuple(v) for v in shape.get_vertices()))
    vertices = [list(shape.body.local_to_world(v)) for shape in shapes for v in shape.get_vertices()]
    return {"pusher_pos": list(base.agent.position), "pusher_velocity": list(base.agent.velocity),
            "block_pos": list(base.block.position), "block_angle": float(base.block.angle),
            "block_velocity": list(base.block.velocity), "block_angular_velocity": float(base.block.angular_velocity),
            "block_vertices": vertices, "pusher_block_gap": float(gap), "coverage": float(base._get_coverage()),
            "image_sha256": hashlib.sha256(observation["pixels"].tobytes()).hexdigest()}


def load_record(record):
    trace = read(record["source"], compressed=True)
    if record["branch_after_step"]:
        prefix = read(record["prefix_source"], compressed=True)
        initial = prefix["initial"]
        rows = audit.join_recovery_rows(prefix["steps"], trace["steps"], trace["branch_state"],
                                        record["branch_after_step"])
    else:
        initial, rows = trace["initial"], trace["steps"]
    assert len(rows) == record["total_executed_movements"]
    assert rows[-1]["after"]["coverage"] > .95
    return initial, rows


def replay_record(record):
    import gymnasium as gym
    import gym_pusht  # noqa: F401: environment registration only

    initial, rows = load_record(record)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    frames, states = [], []
    try:
        observation, _ = env.reset(seed=record["scene"])
        assert audit.states_match(physical_state(env, observation), initial), "Reset state/RGB mismatch."
        frames.append(np.asarray(observation["pixels"]).copy())
        states.append(np.asarray(initial["pusher_pos"], dtype=np.float32))
        for index, row in enumerate(rows):
            observation, _, terminated, truncated, _ = env.step(np.asarray(row["action"], dtype=np.float32))
            actual = physical_state(env, observation)
            assert audit.states_match(actual, row["after"]), \
                f"Exact recorded-action replay mismatch: {record['id']} movement{row['step']}"
            frames.append(np.asarray(observation["pixels"]).copy())
            states.append(np.asarray(actual["pusher_pos"], dtype=np.float32))
            assert not (terminated or truncated) or index == len(rows) - 1
        assert terminated and not truncated, "Successful source did not exactly replay to success."
    finally:
        env.close()
    pixels = np.stack(frames)
    positions = np.stack(states)
    actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
    assert pixels.shape == (len(rows) + 1, 96, 96, 3) and pixels.dtype == np.uint8
    assert positions.shape == (len(rows) + 1, 2) and actions.shape == (len(rows), 2)
    return (pixels, positions, actions), {"verified_recorded_movements": len(rows),
                                        "verified_rgb_frames_including_reset": len(rows) + 1,
                                        "replayed_full_original_prefix": record["branch_after_step"],
                                        "reset_at_branch": False,
                                        "tolerance": "existing diagnostic coverage1e-12 only; coordinates and RGB hashes exact"}


def sample_windows(episode_id, group, split, n_frames, n_actions, branch_after_step=0):
    assert n_frames in [n_actions, n_actions + 1]
    first = max(0, branch_after_step - 1)
    for start in range(first, n_actions - HORIZON + 1):
        assert start + 1 < n_frames
        yield {"episode_id": episode_id, "group": group, "split": split,
               "first_observation_index": start, "current_observation_index": start + 1,
               "action_start_index": start, "action_stop_index": start + HORIZON}


def deterministic_npz(path, pixels, states, actions):
    assert not Path(path).exists(), "Do not overwrite a prepared episode."
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                                                       compresslevel=6) as archive:
        for key, value in [("pixels", pixels), ("states", states), ("actions", actions)]:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue(), compresslevel=6)


def versions():
    return {name: importlib.metadata.version(name) for name in
            ["numpy", "gym-pusht", "gymnasium", "pymunk", "av", "pyarrow"]}


def selection_contract():
    return {"recovery_train_scenes": RECOVERY_TRAIN, "recovery_validation_scenes": RECOVERY_VALIDATION,
            "expert_repo_id": REPO, "expert_revision": REVISION,
            "expert_train_episodes": EXPERT_TRAIN, "expert_validation_episodes": EXPERT_VALIDATION,
            "expert_terminal_handling": "Full decoded chronology validated before first-next.done-inclusive truncation; require contiguous terminal-True tail; no success filtering."}


def prepare(output=OUTPUT, cache=CACHE, raw_dir=RAW):
    output, cache, raw_dir = Path(output), Path(cache), Path(raw_dir)
    protocol = read(output / "protocol.json")
    for key, value in selection_contract().items():
        assert protocol["data_selection"][key] == value, f"Unexpected frozen selection: {key}"
    document = read(audit.OUTPUT / "data-audit.json")
    bindings = dict(document["source_bindings"])
    for relative, expected in bindings.items():
        assert audit.sha(relative) == expected, f"Discovery source binding changed: {relative}"
    bindings.update({(audit.OUTPUT / "data-audit.json").as_posix(): audit.sha(audit.OUTPUT / "data-audit.json"),
                     (output / "protocol.json").as_posix(): audit.sha(output / "protocol.json"),
                     "experiments/robotics_recovery_dataset.py": audit.sha(__file__)})
    info = read(raw_dir / "meta/info.json")
    assert info["total_episodes"] == 206 and info["total_frames"] == 25650 and info["fps"] == 10
    assert info["features"]["observation.image"]["shape"] == [96, 96, 3]
    tree = {x["path"]: x for x in read(raw_dir / "tree.json")}
    for relative in ["tree.json", "meta/info.json", "meta/episodes.jsonl", "README.md"]:
        path = raw_dir / relative
        bindings[path.as_posix()] = audit.sha(path)
    for episode in EXPERT_EPISODES:
        for relative in expert_paths(episode):
            path = raw_dir / relative
            assert path.stat().st_size == tree[relative]["size"]
            digest = audit.sha(path)
            assert digest == tree[relative]["lfs"]["oid"]
            bindings[path.as_posix()] = digest
    protocol_sha = audit.sha(output / "protocol.json")
    code_sha = audit.sha(__file__)
    package_versions = versions()
    episodes, windows = [], []

    def materialize(identifier, group, split, scene, filename, source, branch, producer):
        path = cache / filename
        metadata_path = path.with_suffix(".json")
        signature = audit.fingerprint({"source": source, "protocol_sha256": protocol_sha,
                                       "dataset_code_sha256": code_sha, "packages": package_versions})
        if metadata_path.exists():
            metadata = read(metadata_path)
            assert metadata["source_signature"] == signature
            assert path.exists() and audit.sha(path) == metadata["sha256"]
        else:
            assert not path.exists(), "Orphan prepared NPZ needs inspection; do not replace."
            arrays, verification = producer()
            pixels, states, actions = arrays
            deterministic_npz(path, pixels, states, actions)
            metadata = {"episode_id": identifier, "group": group, "split": split, "scene": scene,
                        "file": path.as_posix(), "sha256": audit.sha(path), "bytes": path.stat().st_size,
                        "n_frames": len(pixels), "n_actions": len(actions), "branch_after_step": branch,
                        "source": source, "source_signature": signature, "verification": verification}
            write_new(metadata_path, metadata)
        assert (metadata["episode_id"], metadata["group"], metadata["split"], metadata["scene"],
                metadata["file"], metadata["branch_after_step"]) == (identifier, group, split, scene,
                                                                      path.as_posix(), branch)
        episodes.append(metadata)
        if group in ["recovery", "expert"]:
            windows.extend(sample_windows(identifier, group, split, metadata["n_frames"],
                                          metadata["n_actions"], branch))
        print(f"Prepared {identifier}: {metadata['n_actions']} actions, {split}", flush=True)

    positives = [r for r in document["records"] if r["category"] == "successful_continuation_of_failed_original"]
    controls = [r for r in document["records"] if r["category"] == "ordinary_successful_baseline"]
    assert len(positives) == 31 and len(controls) == 20
    for index, record in enumerate(positives + controls):
        positive = index < len(positives)
        group = "recovery" if positive else "ordinary_control"
        scene = record["scene"]
        assert scene in audit.SCENES
        split = ("train" if scene in RECOVERY_TRAIN else "validation") if positive else "diagnostic"
        if positive:
            assert scene in RECOVERY_TRAIN + RECOVERY_VALIDATION
        identifier = f"{group}-{index:03d}"
        source = {"record_id": record["id"], "trace": record["source"],
                  "trace_sha256": bindings[record["source"]], "scene": scene,
                  "category": record["category"], "branch_after_step": record["branch_after_step"]}
        if positive:
            source.update({"prefix_trace": record["prefix_source"],
                           "prefix_trace_sha256": bindings[record["prefix_source"]]})
        materialize(identifier, group, split, scene, identifier + ".npz", source,
                    record["branch_after_step"], lambda r=record: replay_record(r))
    for episode in EXPERT_EPISODES:
        group, split = "expert", "train" if episode in EXPERT_TRAIN else "validation"
        paths = [raw_dir / relative for relative in expert_paths(episode)]
        source = {"repo_id": REPO, "revision": REVISION, "expert_episode_index": episode,
                  "files": {p.as_posix(): bindings[p.as_posix()] for p in paths}}
        materialize(f"expert-{episode:06d}", group, split, episode, f"expert-{episode:06d}.npz",
                    source, 0, lambda e=episode: load_expert(e, raw_dir))
    counts = {}
    for group in ["recovery", "expert"]:
        for split in ["train", "validation"]:
            selected = [w for w in windows if w["group"] == group and w["split"] == split]
            counts[f"{group}_{split}"] = {"episodes": len({w["episode_id"] for w in selected}),
                                            "scenes": sorted({e["scene"] for e in episodes
                                                              if e["group"] == group and e["split"] == split}),
                                            "windows": len(selected)}
    assert counts["recovery_train"]["windows"] + counts["recovery_validation"]["windows"] == 2760
    manifest = {"schema_version": 1, "complete": True, "selection": selection_contract(),
                "protocol_sha256": protocol_sha, "dataset_code_sha256": code_sha,
                "packages": package_versions, "source_bindings": dict(sorted(bindings.items())),
                "episodes": episodes, "windows": windows, "counts": counts,
                "ordinary_controls": {"episodes": len(controls), "used_for_training": False},
                "array_contract": {"pixels": "uint8[S,96,96,3], RGB", "states": "float32[S,2], observed pusher",
                                   "actions": "float32[A,2], actual executed commands or original human demonstration labels",
                                   "recovery": "S=A+1; full reset prefix retained; only post-branch anchors sampled",
                                   "expert": "S=A; observation/action paired at each original dataset row",
                                   "window": "obs indices j,j+1; actions[j:j+16]; current observation j+1; first usable future output index1",
                                   "padding": "none; full16 labels only; never cross episodes or duplicate branch reset images"},
                "limitations": ["Public expert AV1 is lossy, unlike exactly verified local simulated RGB.",
                                "Successful continuation actions are not proven optimal or causally effective corrections.",
                                "Overlapping windows and reused contexts are not independent demonstrations.",
                                "Discovery scene split separates initial scenes, not object distribution or real-world domain.",
                                "Ordinary baseline successes are diagnostic controls, not replacements for original expert rehearsal.",
                                "Preparation performs no policy inference, neural training, RGB fitting or new experiment."]}
    write_new(output / "dataset.json", manifest)
    return manifest


def verify_prepared(output=OUTPUT):
    output = Path(output)
    manifest = read(output / "dataset.json")
    assert manifest["complete"]
    for relative, expected in manifest["source_bindings"].items():
        assert audit.sha(relative) == expected, f"Prepared source changed: {relative}"
    for episode in manifest["episodes"]:
        assert audit.sha(episode["file"]) == episode["sha256"]
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["download", "prepare", "verify"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == "download":
        print(json.dumps(download_expert(), indent=2))
    elif args.stage == "prepare":
        print(json.dumps(prepare(args.output_dir)["counts"], indent=2))
    else:
        print(json.dumps(verify_prepared(args.output_dir)["counts"], indent=2))


if __name__ == "__main__":
    main()
