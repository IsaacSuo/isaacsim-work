#!/usr/bin/env python3
"""Audit Blender caches against their authoritative split BGEO source."""

from __future__ import annotations

import argparse
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
    parser.add_argument("cache_directory", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def empty() -> dict[str, np.ndarray]:
    return {
        "position": np.empty((0, 3), dtype=np.float32),
        "velocity": np.empty((0, 3), dtype=np.float32),
        "id": np.empty(0, dtype=np.int32),
        "remaining_lifetime": np.empty(0, dtype=np.float32),
        "birth_frame": np.empty(0, dtype=np.int32),
        "source_particle_index": np.empty(0, dtype=np.int32),
    }


def main() -> int:
    args = parse_args()
    directory = args.cache_directory.resolve()
    manifest_path = directory / "secondary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("valid") or not manifest.get("state", {}).get("complete"):
        raise ValueError("Render-cache manifest is incomplete")
    configuration = manifest["configuration"]
    input_directory = Path(configuration["input_directory"])
    scale = int(configuration["exact_source_frame_scale"])
    start_frame, end_frame = map(int, configuration["output_frame_range"])

    records: list[dict] = []
    particle_states = 0
    for output_frame in range(start_frame, end_frame + 1):
        source_frame = output_frame * scale
        for kind in KINDS:
            source_path = input_directory / f"secondary_{source_frame:06d}_{kind}.bgeo"
            if source_path.exists():
                schema, source = read_bgeo(source_path)
                require_schema(source_path, schema, REQUIRED)
                if np.any(source["particle_type"] != TYPE_BY_KIND[kind]):
                    raise ValueError(f"{source_path}: type mismatch")
                order = np.argsort(source["id"], kind="stable")
                source = {key: np.asarray(value)[order] for key, value in source.items()}
            else:
                source = empty()

            cache_path = directory / kind / f"frame_{output_frame:04d}.npz"
            with np.load(cache_path, allow_pickle=False) as archive:
                cache = {key: np.asarray(archive[key]) for key in archive.files}
            for name in ("id", "position", "velocity", "remaining_lifetime", "birth_frame", "source_particle_index"):
                if not np.array_equal(cache[name], source[name]):
                    raise ValueError(f"{cache_path}: {name} differs from authoritative BGEO")
            if int(cache["source_frame"]) != source_frame or int(cache["output_frame"]) != output_frame:
                raise ValueError(f"{cache_path}: frame mapping metadata is wrong")
            count = len(cache["id"])
            if cache["radius"].shape != (count,) or cache["opacity"].shape != (count,) or cache["shape"].shape != (count, 3):
                raise ValueError(f"{cache_path}: render-only arrays are misaligned")
            if (
                np.any(cache["radius"] <= 0.0)
                or np.any(cache["opacity"] < 0.0)
                or np.any(cache["opacity"] > 1.0)
                or np.any(cache["shape"] <= 0.0)
                or not np.all(np.isfinite(cache["radius"]))
                or not np.all(np.isfinite(cache["opacity"]))
                or not np.all(np.isfinite(cache["shape"]))
            ):
                raise ValueError(f"{cache_path}: invalid render-only attributes")
            particle_states += count
            records.append(
                {
                    "output_frame": output_frame,
                    "source_frame": source_frame,
                    "kind": kind,
                    "particles": count,
                    "cache": cache_path.name,
                    "cache_sha256": sha256_file(cache_path),
                    "authoritative_attributes_bit_exact": True,
                }
            )

    report = {
        "schema": "foamgenerator-physx-blender-render-cache-audit/v1",
        "valid": True,
        "cache_directory": str(directory),
        "manifest_sha256": sha256_file(manifest_path),
        "frame_range": [start_frame, end_frame],
        "exact_source_frame_scale": scale,
        "files_audited": len(records),
        "particle_states_audited": particle_states,
        "authoritative_ids_bit_exact": True,
        "authoritative_positions_bit_exact": True,
        "authoritative_velocities_bit_exact": True,
        "authoritative_lifecycle_bit_exact": True,
        "finite_difference_velocity_used": False,
        "trajectory_resampling_used": False,
        "classification_regenerated": False,
        "render_only_attributes_valid": True,
        "files": records,
    }
    atomic_json(args.output_report.resolve(), report)
    print(json.dumps({"valid": True, "files_audited": len(records), "particle_states_audited": particle_states}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
