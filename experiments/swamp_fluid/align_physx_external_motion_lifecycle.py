#!/usr/bin/env python3
"""Align cached PhysX secondary motion to FoamGenerator's lifecycle boundary.

PhysX remains authoritative for every retained position and velocity.  This
utility only removes one-frame stragglers caused by performing FoamGenerator's
single-precision lifetime countdown in double precision in an older bridge.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from audit_foamgenerator_external_motion import read_split
from foam_bgeo_io import (
    PARTIO_INT,
    PARTIO_VECTOR,
    read_bgeo,
    require_schema,
    write_motion_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-directory", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    return parser.parse_args()


def reference_ids(directory: Path, frame: int) -> np.ndarray:
    try:
        return read_split(directory, frame)["id"].astype(np.int32, copy=False)
    except FileNotFoundError:
        return np.empty(0, dtype=np.int32)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("Invalid frame range")
    source = args.external_directory.resolve()
    reference = args.reference_directory.resolve()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)

    previous_reference_ids = np.empty(0, dtype=np.int32)
    corrections: list[dict] = []
    total_particles = 0
    for frame in range(args.start_frame, args.end_frame + 1):
        source_path = source / f"external_{frame:06d}.bgeo"
        output_path = output / source_path.name
        schema, arrays = read_bgeo(source_path)
        require_schema(
            source_path,
            schema,
            {
                "position": (PARTIO_VECTOR, 3),
                "velocity": (PARTIO_VECTOR, 3),
                "id": (PARTIO_INT, 1),
            },
        )
        ids = arrays["id"].astype(np.int32, copy=False)
        expected_ids = reference_ids(reference, frame)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{source_path}: duplicate stable ids")

        expected_set = set(map(int, expected_ids))
        source_set = set(map(int, ids))
        missing = sorted(expected_set - source_set)
        extras = sorted(source_set - expected_set)
        if missing:
            raise ValueError(f"Frame {frame}: PhysX motion is missing active ids {missing}")
        previous_set = set(map(int, previous_reference_ids))
        if any(stable_id not in previous_set for stable_id in extras):
            raise ValueError(
                f"Frame {frame}: extra PhysX ids are not one-frame expired stragglers: {extras}"
            )

        if extras:
            keep = np.fromiter((int(value) in expected_set for value in ids), bool, len(ids))
            write_motion_state(
                output_path,
                arrays["position"][keep],
                arrays["velocity"][keep],
                ids[keep],
            )
            corrections.append(
                {
                    "frame": frame,
                    "removed_expired_ids": extras,
                    "source_particles": len(ids),
                    "aligned_particles": int(np.count_nonzero(keep)),
                }
            )
        else:
            shutil.copy2(source_path, output_path)

        _, written = read_bgeo(output_path)
        order = np.argsort(written["id"], kind="stable")
        expected_order = np.argsort(expected_ids, kind="stable")
        if not np.array_equal(written["id"][order], expected_ids[expected_order]):
            raise ValueError(f"Frame {frame}: aligned id set verification failed")
        source_index = {int(stable_id): index for index, stable_id in enumerate(ids)}
        selected = np.fromiter(
            (source_index[int(stable_id)] for stable_id in written["id"]),
            np.int64,
            len(written["id"]),
        )
        if not np.array_equal(written["position"], arrays["position"][selected]):
            raise ValueError(f"Frame {frame}: position changed during alignment")
        if not np.array_equal(written["velocity"], arrays["velocity"][selected]):
            raise ValueError(f"Frame {frame}: velocity changed during alignment")
        total_particles += len(written["id"])
        previous_reference_ids = expected_ids

    report = {
        "schema": "physx-external-motion-lifecycle-alignment/v1",
        "valid": True,
        "frame_range": [args.start_frame, args.end_frame],
        "source_directory": str(source),
        "reference_directory": str(reference),
        "output_directory": str(output),
        "position_authority": "bit-exact retained PhysX native readback",
        "velocity_authority": "bit-exact retained PhysX native readback",
        "lifecycle_authority": "FoamGenerator single-precision chronological lifetime",
        "frames_corrected": len(corrections),
        "particles_removed": sum(len(item["removed_expired_ids"]) for item in corrections),
        "particles_retained": total_particles,
        "corrections": corrections,
    }
    write_json_atomic(output / "alignment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
