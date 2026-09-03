#!/usr/bin/env python3
"""Verify that external motion remains authoritative through FoamGenerator replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from audit_foamgenerator_birth_events import read_bgeo, require_attribute


KINDS = ("foam", "spray", "bubbles")
TYPE_BY_KIND = {"foam": 0, "spray": 1, "bubbles": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def read_split(directory: Path, frame: int) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for kind in KINDS:
        path = directory / f"secondary_{frame:06d}_{kind}.bgeo"
        if not path.exists():
            continue
        attributes, arrays = read_bgeo(path)
        for name, attribute_type, width in (
            ("position", 5, 3),
            ("velocity", 5, 3),
            ("id", 1, 1),
            ("remaining_lifetime", 0, 1),
            ("birth_frame", 1, 1),
            ("particle_type", 1, 1),
            ("source_particle_index", 1, 1),
        ):
            require_attribute(path, attributes, name, attribute_type, width)
        if np.any(arrays["particle_type"] != TYPE_BY_KIND[kind]):
            raise ValueError(f"{path}: particle_type does not match the split filename")
        parts.append(arrays)
    if not parts:
        raise FileNotFoundError(f"Frame {frame}: no split lifecycle output")
    combined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(combined["id"], kind="stable")
    combined = {key: value[order] for key, value in combined.items()}
    if len(np.unique(combined["id"])) != len(combined["id"]):
        raise ValueError(f"Frame {frame}: duplicate ids across split output")
    return combined


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


def main() -> int:
    args = parse_args()
    if args.end_frame < args.start_frame:
        raise ValueError("Invalid frame range")
    external_directory = args.external_directory.resolve()
    output_directory = args.output_directory.resolve()
    reference_directory = args.reference_directory.resolve()
    records: list[dict] = []
    total_particles = 0
    changed_classifications = 0

    for frame in range(args.start_frame, args.end_frame + 1):
        external_path = external_directory / f"external_{frame:06d}.bgeo"
        external_attributes, external = read_bgeo(external_path)
        for name, attribute_type, width in (
            ("position", 5, 3),
            ("velocity", 5, 3),
            ("id", 1, 1),
        ):
            require_attribute(external_path, external_attributes, name, attribute_type, width)
        external_order = np.argsort(external["id"], kind="stable")
        external = {key: value[external_order] for key, value in external.items()}

        output = read_split(output_directory, frame)
        reference = read_split(reference_directory, frame)
        if not np.array_equal(output["id"], external["id"]):
            raise ValueError(f"Frame {frame}: output id set differs from external motion")
        if not np.array_equal(output["id"], reference["id"]):
            raise ValueError(f"Frame {frame}: birth replay id set differs from the authority pass")
        if not np.array_equal(output["position"], external["position"]):
            raise ValueError(f"Frame {frame}: FoamGenerator changed PhysX-authored positions")
        if not np.array_equal(output["velocity"], external["velocity"]):
            raise ValueError(f"Frame {frame}: FoamGenerator changed PhysX-authored velocities")
        for name in ("remaining_lifetime", "birth_frame", "source_particle_index"):
            if not np.array_equal(output[name], reference[name]):
                raise ValueError(f"Frame {frame}: replayed {name} differs from the authority pass")
        if np.any(output["particle_type"] < 0) or np.any(output["particle_type"] > 2):
            raise ValueError(f"Frame {frame}: output contains an invalid dynamic type")

        classification_changes = int(
            np.count_nonzero(output["particle_type"] != reference["particle_type"])
        )
        changed_classifications += classification_changes
        total_particles += len(output["id"])
        records.append(
            {
                "frame": frame,
                "particle_count": len(output["id"]),
                "classification_changes_from_internal_advection_reference": classification_changes,
                "positions_bit_exact": True,
                "velocities_bit_exact": True,
                "birth_and_lifecycle_authority_preserved": True,
            }
        )

    report = {
        "schema": "foamgenerator-external-motion-authority-audit/v1",
        "valid": True,
        "frame_range": [args.start_frame, args.end_frame],
        "external_directory": str(external_directory),
        "output_directory": str(output_directory),
        "reference_directory": str(reference_directory),
        "particles_audited": total_particles,
        "positions_bit_exact": True,
        "velocities_bit_exact": True,
        "foamgenerator_motion_or_collision_applied": False,
        "foamgenerator_classification_applied": True,
        "foamgenerator_chronological_lifetime_applied": True,
        "authoritative_birth_events_replayed": True,
        "classification_changes_from_internal_advection_reference": changed_classifications,
        "frames": records,
    }
    atomic_json(args.output_report.resolve(), report)
    print(
        json.dumps(
            {
                "valid": True,
                "particles_audited": total_particles,
                "positions_bit_exact": True,
                "velocities_bit_exact": True,
                "classification_changes_from_internal_advection_reference": changed_classifications,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
