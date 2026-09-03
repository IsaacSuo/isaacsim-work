#!/usr/bin/env python3
"""Audit time alignment and CUDA reproducibility of PhysX type controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
try:
    import warp  # noqa: F401
except ImportError:
    candidates = sorted(
        Path(r"Y:\isaacsim\extscache").glob("omni.warp.core-*"), reverse=True
    )
    if not candidates:
        raise RuntimeError("Isaac Sim Warp extension was not found")
    sys.path.insert(0, str(candidates[0]))

from audit_foamgenerator_external_motion import read_split  # noqa: E402
from foam_bgeo_io import read_bgeo, read_birth_events  # noqa: E402
from foamgenerator_warp_dynamics import (  # noqa: E402
    BUBBLES,
    FOAM,
    SPRAY,
    FoamGeneratorWarpNeighborhood,
)


TYPE_NAMES = {FOAM: "foam", SPRAY: "spray", BUBBLES: "bubbles"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--primary-pattern", required=True)
    parser.add_argument("--external-directory", type=Path, required=True)
    parser.add_argument("--classification-directory", type=Path, required=True)
    parser.add_argument("--birth-pattern", required=True)
    parser.add_argument("--baseline-external-directory", type=Path)
    parser.add_argument("--start-output-frame", type=int, required=True)
    parser.add_argument("--end-output-frame", type=int, required=True)
    parser.add_argument("--particle-radius", type=float, default=0.004)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def frame_path(pattern: str, frame: int) -> Path:
    start = pattern.find("#")
    if start < 0:
        raise ValueError("Frame pattern requires a # marker")
    end = start
    while end < len(pattern) and pattern[end] == "#":
        end += 1
    return Path(pattern[:start] + f"{frame:0{end-start}d}" + pattern[end:])


def sorted_motion(path: Path) -> dict[str, np.ndarray]:
    _, arrays = read_bgeo(path)
    required = ("id", "position", "velocity")
    if any(name not in arrays for name in required):
        raise ValueError(f"{path}: motion state lacks {required}")
    order = np.argsort(arrays["id"], kind="stable")
    return {name: np.asarray(arrays[name])[order] for name in required}


def atomic_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10.0)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
    }


def main() -> int:
    args = parse_args()
    if args.end_output_frame < args.start_output_frame:
        raise ValueError("Invalid output-frame range")
    controller = FoamGeneratorWarpNeighborhood(args.particle_radius, args.device)
    transition = np.zeros((3, 3), dtype=np.int64)
    vertical_displacements = {name: [] for name in TYPE_NAMES.values()}
    baseline_vertical_offsets = {name: [] for name in TYPE_NAMES.values()}
    records = []
    total_states = 0
    total_existing = 0
    total_newborn = 0

    for frame in range(args.start_output_frame, args.end_output_frame + 1):
        source_frame = frame - 1
        control_path = args.control_directory / f"control_{frame:06d}.npz"
        with np.load(control_path, allow_pickle=False) as archive:
            control = {name: np.asarray(archive[name]).copy() for name in archive.files}
        ids = np.asarray(control["id"], dtype=np.int64)
        query_positions = np.asarray(control["query_position"], dtype=np.float32)
        query_velocities = np.asarray(control["query_velocity"], dtype=np.float32)
        particle_types = np.asarray(control["particle_type"], dtype=np.int32)
        neighbor_counts = np.asarray(control["neighbor_count"], dtype=np.int32)
        local_velocities = np.asarray(
            control["local_fluid_velocity"], dtype=np.float32
        )
        weight_sums = np.asarray(control["kernel_weight_sum"], dtype=np.float32)
        count = len(ids)
        if (
            len(np.unique(ids)) != count
            or query_positions.shape != (count, 3)
            or query_velocities.shape != (count, 3)
            or particle_types.shape != (count,)
            or neighbor_counts.shape != (count,)
            or local_velocities.shape != (count, 3)
            or weight_sums.shape != (count,)
            or int(control["source_state_frame"]) != source_frame
            or int(control["target_output_frame"]) != frame
        ):
            raise ValueError(f"{control_path}: invalid control schema")

        primary_path = frame_path(args.primary_pattern, source_frame)
        with np.load(primary_path, allow_pickle=False) as archive:
            primary_positions = np.asarray(archive["positions"], dtype=np.float32)
            primary_velocities = np.asarray(archive["velocities"], dtype=np.float32)
        reproduced = controller.query(
            primary_positions, primary_velocities, query_positions
        )
        if not np.array_equal(reproduced.particle_types, particle_types):
            raise ValueError(f"Frame {frame}: CUDA particle types are not reproducible")
        if not np.array_equal(reproduced.neighbor_counts, neighbor_counts):
            raise ValueError(f"Frame {frame}: CUDA neighbor counts are not reproducible")
        if not np.array_equal(reproduced.fluid_velocities, local_velocities):
            raise ValueError(f"Frame {frame}: CUDA fluid velocities are not bit-exact")
        if not np.array_equal(reproduced.weight_sums, weight_sums):
            raise ValueError(f"Frame {frame}: CUDA kernel sums are not bit-exact")

        try:
            previous = read_split(args.classification_directory, source_frame)
        except FileNotFoundError:
            if source_frame != args.start_output_frame - 1:
                raise
            # FoamGenerator's asynchronous split writer does not materialize
            # files for the leading frame when all three type sets are empty.
            previous = {
                "id": np.empty(0, dtype=np.int32),
                "position": np.empty((0, 3), dtype=np.float32),
                "velocity": np.empty((0, 3), dtype=np.float32),
                "particle_type": np.empty(0, dtype=np.int32),
                "remaining_lifetime": np.empty(0, dtype=np.float32),
            }
        # Split output is written after the lifetime decrement.  Particles that
        # reached zero are still present in that source-frame file, but are
        # removed before the next PhysX control query.
        survivor_mask = previous["remaining_lifetime"] > 0.0
        previous = {name: values[survivor_mask] for name, values in previous.items()}
        birth_path = frame_path(args.birth_pattern, source_frame)
        birth_ids = (
            read_birth_events(birth_path)["id"].astype(np.int64)
            if birth_path.exists()
            else np.empty(0, dtype=np.int64)
        )
        expected_ids = np.concatenate((previous["id"].astype(np.int64), birth_ids))
        if not np.array_equal(ids, expected_ids):
            raise ValueError(f"Frame {frame}: control IDs do not equal survivors plus births")
        previous_count = len(previous["id"])
        if not np.array_equal(
            particle_types[:previous_count], previous["particle_type"].astype(np.int32)
        ):
            raise ValueError(
                f"Frame {frame}: existing control types differ from prior FoamGenerator output"
            )
        if not np.array_equal(
            query_positions[:previous_count], previous["position"].astype(np.float32)
        ):
            raise ValueError(f"Frame {frame}: prior positions do not match control query")

        motion = sorted_motion(args.external_directory / f"external_{frame:06d}.bgeo")
        classified = read_split(args.classification_directory, frame)
        if not (
            np.array_equal(ids, motion["id"].astype(np.int64))
            and np.array_equal(ids, classified["id"].astype(np.int64))
            and np.array_equal(motion["position"], classified["position"])
            and np.array_equal(motion["velocity"], classified["velocity"])
        ):
            raise ValueError(f"Frame {frame}: PhysX/FoamGenerator state handoff differs")

        post_types = classified["particle_type"].astype(np.int32)
        np.add.at(transition, (particle_types, post_types), 1)
        vertical = motion["position"][:, 1] - query_positions[:, 1]
        for type_value, name in TYPE_NAMES.items():
            vertical_displacements[name].extend(vertical[particle_types == type_value])

        if args.baseline_external_directory is not None:
            baseline = sorted_motion(
                args.baseline_external_directory / f"external_{frame:06d}.bgeo"
            )
            if not np.array_equal(ids, baseline["id"].astype(np.int64)):
                raise ValueError(f"Frame {frame}: ballistic baseline ID set differs")
            vertical_offset = motion["position"][:, 1] - baseline["position"][:, 1]
            for type_value, name in TYPE_NAMES.items():
                baseline_vertical_offsets[name].extend(
                    vertical_offset[particle_types == type_value]
                )

        total_states += count
        total_existing += previous_count
        total_newborn += len(birth_ids)
        records.append(
            {
                "frame": frame,
                "source_state_frame": source_frame,
                "particle_count": count,
                "existing_ids_checked_against_prior_classification": previous_count,
                "newborn_ids": len(birth_ids),
                "cuda_control_bit_exact": True,
                "prior_type_alignment_bit_exact": True,
                "physx_foamgenerator_handoff_bit_exact": True,
                "type_counts_before_step": {
                    name: int(np.count_nonzero(particle_types == type_value))
                    for type_value, name in TYPE_NAMES.items()
                },
                "type_counts_after_step": {
                    name: int(np.count_nonzero(post_types == type_value))
                    for type_value, name in TYPE_NAMES.items()
                },
            }
        )

    report = {
        "schema": "physx-foamgenerator-type-dynamics-control-audit/v1",
        "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frame_range": [args.start_output_frame, args.end_output_frame],
        "particle_states_audited": total_states,
        "existing_states_aligned_to_prior_foamgenerator_type": total_existing,
        "newborn_states_classified_on_cuda": total_newborn,
        "cuda_reproduction_bit_exact": True,
        "prior_foamgenerator_type_alignment_bit_exact": True,
        "physx_foamgenerator_handoff_bit_exact": True,
        "transition_matrix_before_to_after_step": transition.tolist(),
        "vertical_displacement_m_by_control_type": {
            name: distribution(np.asarray(values))
            for name, values in vertical_displacements.items()
        },
        "vertical_offset_from_ballistic_baseline_m_by_control_type": (
            {
                name: distribution(np.asarray(values))
                for name, values in baseline_vertical_offsets.items()
            }
            if args.baseline_external_directory is not None
            else None
        ),
        "finite_difference_velocity_used": False,
        "records": records,
    }
    atomic_json(args.output_report.resolve(), report)
    print(
        json.dumps(
            {
                "valid": True,
                "particle_states_audited": total_states,
                "vertical_displacement_m_by_control_type": report[
                    "vertical_displacement_m_by_control_type"
                ],
                "vertical_offset_from_ballistic_baseline_m_by_control_type": report[
                    "vertical_offset_from_ballistic_baseline_m_by_control_type"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
