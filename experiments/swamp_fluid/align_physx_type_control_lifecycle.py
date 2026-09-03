#!/usr/bin/env python3
"""Remove expired one-frame stragglers from cached CUDA type controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from foam_bgeo_io import read_bgeo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--external-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    return parser.parse_args()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    with temporary.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
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
    source = args.control_directory.resolve()
    external = args.external_directory.resolve()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)

    corrections: list[dict] = []
    total_states = 0
    for frame in range(args.start_frame, args.end_frame + 1):
        source_path = source / f"control_{frame:06d}.npz"
        output_path = output / source_path.name
        with np.load(source_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        ids = np.asarray(arrays["id"], dtype=np.int64)
        _, motion = read_bgeo(external / f"external_{frame:06d}.bgeo")
        expected_ids = np.asarray(motion["id"], dtype=np.int64)
        if len(np.unique(ids)) != len(ids) or len(np.unique(expected_ids)) != len(expected_ids):
            raise ValueError(f"Frame {frame}: duplicate stable ids")
        expected_set = set(map(int, expected_ids))
        source_set = set(map(int, ids))
        missing = sorted(expected_set - source_set)
        extras = sorted(source_set - expected_set)
        if missing:
            raise ValueError(f"Frame {frame}: control lacks PhysX ids {missing}")
        keep = np.fromiter((int(value) in expected_set for value in ids), bool, len(ids))
        filtered: dict[str, np.ndarray] = {}
        for name, values in arrays.items():
            if values.ndim > 0 and values.shape[0] == len(ids):
                filtered[name] = values[keep]
            else:
                filtered[name] = values
        atomic_npz(output_path, filtered)

        with np.load(output_path, allow_pickle=False) as archive:
            written = {name: np.asarray(archive[name]) for name in archive.files}
            if not np.array_equal(written["id"], ids[keep]):
                raise ValueError(f"Frame {frame}: written control ids changed")
            for name, expected in filtered.items():
                if not np.array_equal(written[name], expected):
                    raise ValueError(f"Frame {frame}: control attribute {name} changed")
        if extras:
            corrections.append(
                {
                    "frame": frame,
                    "removed_expired_ids": extras,
                    "source_states": len(ids),
                    "aligned_states": int(np.count_nonzero(keep)),
                }
            )
        total_states += int(np.count_nonzero(keep))

    report = {
        "schema": "physx-type-control-lifecycle-alignment/v1",
        "valid": True,
        "frame_range": [args.start_frame, args.end_frame],
        "source_directory": str(source),
        "external_motion_authority": str(external),
        "output_directory": str(output),
        "frames_corrected": len(corrections),
        "states_removed": sum(len(item["removed_expired_ids"]) for item in corrections),
        "states_retained": total_states,
        "all_retained_control_attributes_bit_exact": True,
        "corrections": corrections,
    }
    atomic_json(output / "alignment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
