"""Recover PhysX 5.9 Diffuse render labels from native frames on the GPU.

The PhysX CUDA solver classifies a Diffuse particle from the number of primary
fluid neighbors inside ``2 * particleContactOffset``: fewer than four is spray,
four through seven is foam, and eight or more is bubble.  The class itself is a
temporary kernel value and is not present in the public Diffuse output buffer.

This tool consumes same-frame native primary and Diffuse arrays exported by
``render_swamp_physx_fluid_preview.py``.  It never uses height, water-surface
distance, velocity, or camera position to manufacture a class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--only-frames", nargs="*", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--warp-package-parent",
        type=Path,
        default=Path(r"Y:\isaacsim\extscache\omni.warp.core-1.13.0+wx64"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


args = parse_args()
raw_directory = args.raw_directory.resolve()
output_directory = args.output_directory.resolve()
if output_directory.exists():
    raise FileExistsError(f"Refusing to overwrite output directory: {output_directory}")
if not args.warp_package_parent.is_dir():
    raise FileNotFoundError(args.warp_package_parent)
sys.path.insert(0, str(args.warp_package_parent.resolve()))

import warp as wp


@wp.kernel
def recover_kernel(
    grid: wp.uint64,
    primary: wp.array(dtype=wp.vec3),
    diffuse: wp.array(dtype=wp.vec3),
    radius: float,
    radius_squared: float,
    counts: wp.array(dtype=wp.int32),
    labels: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    query_position = diffuse[index]
    neighbors = wp.hash_grid_query(grid, query_position, radius)
    count = int(0)
    for primary_index in neighbors:
        delta = primary[primary_index] - query_position
        if wp.dot(delta, delta) < radius_squared:
            count = count + 1
    if count > 16:
        count = 16
    counts[index] = count
    if count < 4:
        labels[index] = 0
    elif count < 8:
        labels[index] = 1
    else:
        labels[index] = 2


def gpu_recover(primary: np.ndarray, diffuse: np.ndarray, radius: float, device: str):
    primary_xyz = np.ascontiguousarray(primary[:, :3], dtype=np.float32)
    diffuse_xyz = np.ascontiguousarray(diffuse[:, :3], dtype=np.float32)
    primary_wp = wp.array(primary_xyz, dtype=wp.vec3, device=device)
    diffuse_wp = wp.array(diffuse_xyz, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(128, 128, 128, device=device)
    grid.build(primary_wp, radius)
    counts_wp = wp.empty(len(diffuse_xyz), dtype=wp.int32, device=device)
    labels_wp = wp.empty(len(diffuse_xyz), dtype=wp.int32, device=device)
    wp.launch(
        recover_kernel,
        dim=len(diffuse_xyz),
        inputs=[grid.id, primary_wp, diffuse_wp, radius, radius * radius],
        outputs=[counts_wp, labels_wp],
        device=device,
    )
    wp.synchronize_device(device)
    counts = counts_wp.numpy().astype(np.uint8, copy=False)
    labels = labels_wp.numpy().astype(np.uint8, copy=False)
    return labels, counts


def synthetic_gate(device: str) -> dict[str, object]:
    centres = np.asarray(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    expected_counts = np.asarray([3, 4, 7, 8], dtype=np.uint8)
    primary = []
    for centre, count in zip(centres, expected_counts, strict=True):
        for sample_index in range(int(count)):
            angle = 2.0 * np.pi * sample_index / max(int(count), 1)
            primary.append(centre + np.asarray([0.5 * np.cos(angle), 0.5 * np.sin(angle), 0.0]))
        primary.append(centre + np.asarray([1.0, 0.0, 0.0]))
    labels, counts = gpu_recover(
        np.asarray(primary, dtype=np.float32), centres, 1.0, device
    )
    expected_labels = np.asarray([0, 1, 1, 2], dtype=np.uint8)
    passed = bool(
        np.array_equal(counts, expected_counts)
        and np.array_equal(labels, expected_labels)
    )
    if not passed:
        raise RuntimeError(
            f"Warp synthetic threshold gate failed: counts={counts}, labels={labels}"
        )
    return {
        "passed": True,
        "strict_radius_exclusion_tested": True,
        "expected_counts": expected_counts.tolist(),
        "expected_labels": expected_labels.tolist(),
    }


manifest_path = raw_directory / "manifest.json"
raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not raw_manifest.get("state", {}).get("complete"):
    raise RuntimeError("Raw native Diffuse manifest is incomplete")
if raw_manifest.get("product") != "swamp_physx_native_diffuse_frames":
    raise ValueError("Unexpected raw cache product")
selected = set(args.only_frames) if args.only_frames is not None else None
records = [
    record
    for record in raw_manifest["frames"]
    if selected is None or int(record["output_frame"]) in selected
]
if not records:
    raise ValueError("No raw frames matched the requested selection")
if selected is not None and {int(record["output_frame"]) for record in records} != selected:
    raise ValueError("One or more requested output frames are not in the raw cache")

output_directory.mkdir(parents=True)
output_manifest_path = output_directory / "manifest.json"
wp.init()
device = wp.get_device(args.device)
if not device.is_cuda:
    raise RuntimeError(f"GPU label recovery requires CUDA, got {device}")
gate = synthetic_gate(args.device)
output_manifest = {
    "schema": 1,
    "product": "physx_diffuse_recovered_render_labels",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "state": {"complete": False, "frames": 0},
    "source": {
        "raw_directory": str(raw_directory),
        "raw_manifest": str(manifest_path),
        "raw_manifest_sha256": sha256_file(manifest_path),
    },
    "method": {
        "authority": "PhysX 5.9 diffuseParticles.cu neighbor thresholds",
        "label_source": "same-frame native primary neighbor count",
        "mapping": {"0": "spray (<4)", "1": "foam (4..7)", "2": "bubble (>=8)"},
        "maximum_neighbors": 16,
        "strict_radius_comparison": "distance_squared < radius_squared",
        "height_or_surface_partition_used": False,
        "velocity_partition_used": False,
        "device": str(device),
        "warp_version": wp.__version__,
        "synthetic_gate": gate,
    },
    "frames": [],
}
atomic_json(output_manifest_path, output_manifest)

for sequence_index, record in enumerate(records, start=1):
    source_path = raw_directory / record["file"]
    if sha256_file(source_path).lower() != record["sha256"].lower():
        raise RuntimeError(f"Raw frame hash mismatch: {source_path}")
    with np.load(source_path, allow_pickle=False) as cache:
        primary = np.asarray(cache["primary_position_inv_mass"], dtype=np.float32)
        diffuse_positions = np.asarray(
            cache["diffuse_position_lifetime"], dtype=np.float32
        )
        diffuse_velocity = np.asarray(cache["diffuse_velocity"], dtype=np.float32)
        radius = float(cache["diffuse_neighbor_radius"])
        output_frame = int(cache["output_frame"])
        physics_step = int(cache["physics_step"])
        simulation_time = float(cache["simulation_time"])
    if len(primary) != int(record["primary_active_count"]):
        raise RuntimeError(f"Primary count mismatch in {source_path}")
    if len(diffuse_positions) != int(record["diffuse_active_count"]):
        raise RuntimeError(f"Diffuse count mismatch in {source_path}")
    if diffuse_velocity.shape != diffuse_positions.shape:
        raise RuntimeError(f"Diffuse velocity shape mismatch in {source_path}")
    started = time.perf_counter()
    labels, counts = gpu_recover(primary, diffuse_positions, radius, args.device)
    elapsed = time.perf_counter() - started
    histogram = {
        "spray": int(np.count_nonzero(labels == 0)),
        "foam": int(np.count_nonzero(labels == 1)),
        "bubble": int(np.count_nonzero(labels == 2)),
    }
    if sum(histogram.values()) != len(diffuse_positions):
        raise RuntimeError(f"Label conservation failed in {source_path}")
    output_path = output_directory / f"diffuse_labels_{output_frame:04d}.npz"
    atomic_npz(
        output_path,
        schema=np.asarray("physx_diffuse_render_labels/1"),
        output_frame=np.asarray(output_frame, dtype=np.int64),
        physics_step=np.asarray(physics_step, dtype=np.int64),
        simulation_time=np.asarray(simulation_time, dtype=np.float64),
        source_raw_sha256=np.asarray(record["sha256"]),
        diffuse_neighbor_radius=np.asarray(radius, dtype=np.float32),
        diffuse_position_lifetime=np.ascontiguousarray(diffuse_positions),
        diffuse_velocity=np.ascontiguousarray(diffuse_velocity),
        recovered_label=np.ascontiguousarray(labels),
        primary_neighbor_count=np.ascontiguousarray(counts),
    )
    output_manifest["frames"].append(
        {
            "output_frame": output_frame,
            "physics_step": physics_step,
            "simulation_time_s": simulation_time,
            "source_raw_file": record["file"],
            "source_raw_sha256": record["sha256"],
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "active_count": len(diffuse_positions),
            "histogram": histogram,
            "neighbor_count_range": [int(counts.min()), int(counts.max())],
            "gpu_seconds": elapsed,
        }
    )
    output_manifest["state"]["frames"] = len(output_manifest["frames"])
    atomic_json(output_manifest_path, output_manifest)
    print(
        f"[physx-labels] frame={output_frame:04d} count={len(labels)} "
        f"histogram={histogram} gpu={elapsed:.3f}s "
        f"sequence={sequence_index}/{len(records)}",
        flush=True,
    )

output_manifest["state"].update(
    {
        "complete": True,
        "frames": len(output_manifest["frames"]),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
)
atomic_json(output_manifest_path, output_manifest)
