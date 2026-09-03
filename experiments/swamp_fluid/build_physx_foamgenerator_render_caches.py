#!/usr/bin/env python3
"""Convert audited FoamGenerator/PhysX BGEO states into Blender render caches.

The converter does not invent trajectories or classifications.  It samples the
120 Hz closed-loop state at exact 30 FPS boundaries, preserves stable ids,
positions, and velocities, and only adds explicitly render-only radius, shape,
and lifecycle-opacity attributes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from foam_bgeo_io import PARTIO_FLOAT, PARTIO_INT, PARTIO_VECTOR, read_bgeo, require_schema


KINDS = ("foam", "spray", "bubbles")
TYPE_BY_KIND = {"foam": 0, "spray": 1, "bubbles": 2}
REQUIRED = {
    "position": (PARTIO_VECTOR, 3),
    "velocity": (PARTIO_VECTOR, 3),
    "id": (PARTIO_INT, 1),
    "remaining_lifetime": (PARTIO_FLOAT, 1),
    "birth_frame": (PARTIO_INT, 1),
    "particle_type": (PARTIO_INT, 1),
    "source_particle_index": (PARTIO_INT, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--start-output-frame", type=int, required=True)
    parser.add_argument("--end-output-frame", type=int, required=True)
    parser.add_argument("--source-fps", type=int, default=120)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--spray-radius", type=float, default=0.00125)
    parser.add_argument("--foam-radius", type=float, default=0.0024)
    parser.add_argument("--bubble-radius-min", type=float, default=0.00055)
    parser.add_argument("--bubble-radius-max", type=float, default=0.0022)
    parser.add_argument("--birth-fade-seconds", type=float, default=0.05)
    parser.add_argument("--death-fade-seconds", type=float, default=0.20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def hash01(ids: np.ndarray, salt: int) -> np.ndarray:
    values = ids.astype(np.uint64) + np.uint64(salt)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return ((values >> np.uint64(11)).astype(np.float64) / float(1 << 53)).astype(np.float32)


def smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0).astype(np.float32)
    return value * value * (np.float32(3.0) - np.float32(2.0) * value)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_kind(path: Path, expected_type: int) -> dict[str, np.ndarray]:
    if not path.exists():
        return {
            "position": np.empty((0, 3), dtype=np.float32),
            "velocity": np.empty((0, 3), dtype=np.float32),
            "id": np.empty(0, dtype=np.int32),
            "remaining_lifetime": np.empty(0, dtype=np.float32),
            "birth_frame": np.empty(0, dtype=np.int32),
            "source_particle_index": np.empty(0, dtype=np.int32),
        }
    schema, arrays = read_bgeo(path)
    require_schema(path, schema, REQUIRED)
    count = len(arrays["id"])
    if np.any(arrays["particle_type"] != expected_type):
        raise ValueError(f"{path}: particle_type disagrees with split filename")
    if len(np.unique(arrays["id"])) != count or np.any(arrays["id"] < 0):
        raise ValueError(f"{path}: invalid stable ids")
    if not np.all(np.isfinite(arrays["position"])) or not np.all(np.isfinite(arrays["velocity"])):
        raise ValueError(f"{path}: non-finite motion state")
    order = np.argsort(arrays["id"], kind="stable")
    return {name: np.asarray(arrays[name])[order] for name in (
        "position", "velocity", "id", "remaining_lifetime", "birth_frame", "source_particle_index"
    )}


def main() -> int:
    args = parse_args()
    input_directory = args.input_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_directory}")
    if args.end_output_frame < args.start_output_frame:
        raise ValueError("Invalid output frame range")
    if args.source_fps <= 0 or args.output_fps <= 0 or args.source_fps % args.output_fps:
        raise ValueError("source_fps must be a positive integer multiple of output_fps")
    if not (0.0 < args.bubble_radius_min <= args.bubble_radius_max):
        raise ValueError("Invalid bubble radius range")
    if min(args.spray_radius, args.foam_radius, args.birth_fade_seconds, args.death_fade_seconds) <= 0:
        raise ValueError("Radii and fade durations must be positive")

    scale = args.source_fps // args.output_fps
    output_directory.mkdir(parents=True, exist_ok=False)
    for kind in KINDS:
        (output_directory / kind).mkdir()

    frame_records: list[dict] = []
    totals = {kind: 0 for kind in KINDS}
    seen_source_files: dict[str, str] = {}
    for output_frame in range(args.start_output_frame, args.end_output_frame + 1):
        source_frame = output_frame * scale
        counts: dict[str, int] = {}
        ids_across_kinds: list[np.ndarray] = []
        for kind in KINDS:
            source_path = input_directory / f"secondary_{source_frame:06d}_{kind}.bgeo"
            data = read_kind(source_path, TYPE_BY_KIND[kind])
            ids = np.asarray(data["id"], dtype=np.int32)
            positions = np.asarray(data["position"], dtype=np.float32)
            velocities = np.asarray(data["velocity"], dtype=np.float32)
            remaining = np.asarray(data["remaining_lifetime"], dtype=np.float32)
            birth_frames = np.asarray(data["birth_frame"], dtype=np.int32)
            source_indices = np.asarray(data["source_particle_index"], dtype=np.int32)
            age_seconds = np.maximum(0, source_frame - birth_frames).astype(np.float32) / np.float32(args.source_fps)
            opacity = smoothstep01(age_seconds / np.float32(args.birth_fade_seconds))
            opacity *= smoothstep01(remaining / np.float32(args.death_fade_seconds))

            if kind == "spray":
                radius = np.full(len(ids), args.spray_radius, dtype=np.float32)
            elif kind == "foam":
                variation = np.float32(0.82) + np.float32(0.36) * hash01(ids, 0xF04A)
                radius = np.float32(args.foam_radius) * variation
            else:
                # Log-uniform radii keep the numerous small bubbles while still
                # admitting a sparse resolvable tail for close-up inspection.
                unit = hash01(ids, 0xBABB1E)
                ratio = np.float32(args.bubble_radius_max / args.bubble_radius_min)
                radius = np.float32(args.bubble_radius_min) * np.power(ratio, unit).astype(np.float32)
            shape = np.ones((len(ids), 3), dtype=np.float32)

            destination = output_directory / kind / f"frame_{output_frame:04d}.npz"
            atomic_npz(
                destination,
                id=ids,
                position=positions,
                velocity=velocities,
                radius=radius,
                opacity=opacity.astype(np.float32),
                shape=shape,
                remaining_lifetime=remaining,
                birth_frame=birth_frames,
                source_particle_index=source_indices,
                source_frame=np.asarray(source_frame, dtype=np.int32),
                output_frame=np.asarray(output_frame, dtype=np.int32),
            )
            if source_path.exists():
                seen_source_files[str(source_path)] = sha256_file(source_path)
            counts[kind] = len(ids)
            totals[kind] += len(ids)
            ids_across_kinds.append(ids)

        combined_ids = np.concatenate(ids_across_kinds)
        if len(np.unique(combined_ids)) != len(combined_ids):
            raise ValueError(f"Output frame {output_frame}: stable id occurs in multiple kinds")
        frame_records.append(
            {
                "output_frame": output_frame,
                "source_frame": source_frame,
                "counts": counts,
                "total": int(len(combined_ids)),
            }
        )

    manifest = {
        "schema": "foamgenerator-physx-blender-render-cache/v1",
        "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "input_directory": str(input_directory),
            "output_frame_range": [args.start_output_frame, args.end_output_frame],
            "source_fps": args.source_fps,
            "output_fps": args.output_fps,
            "exact_source_frame_scale": scale,
            "source_frame_mapping": "source_frame = output_frame * exact_source_frame_scale",
            "spray_radius_m": args.spray_radius,
            "foam_radius_m": args.foam_radius,
            "bubble_radius_range_m": [args.bubble_radius_min, args.bubble_radius_max],
            "birth_fade_seconds": args.birth_fade_seconds,
            "death_fade_seconds": args.death_fade_seconds,
            "render_only_attributes": ["radius", "opacity", "shape"],
            "authoritative_attributes_preserved": [
                "id", "position", "velocity", "remaining_lifetime", "birth_frame", "source_particle_index", "particle_type"
            ],
        },
        "state": {
            "complete": True,
            "completed_frames": len(frame_records),
            "total_frames": len(frame_records),
            "frames": frame_records,
            "particle_frame_totals": totals,
        },
        "source_files": seen_source_files,
        "finite_difference_velocity_used": False,
        "trajectory_resampling_used": False,
        "classification_regenerated": False,
    }
    atomic_json(output_directory / "secondary_manifest.json", manifest)
    print(json.dumps({"valid": True, "frames": len(frame_records), "particle_frame_totals": totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
