#!/usr/bin/env python3
"""Audit a CUDA Warp port of FoamGenerator's secondary neighborhood query."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from audit_foamgenerator_external_motion import read_split  # noqa: E402


def load_warp():
    try:
        import warp as wp
    except ImportError:
        candidates = sorted(
            Path(r"Y:\isaacsim\extscache").glob("omni.warp.core-*"),
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("Isaac Sim Warp extension was not found")
        sys.path.insert(0, str(candidates[0]))
        import warp as wp
    return wp


wp = load_warp()


@wp.kernel
def query_foam_neighborhood(
    grid: wp.uint64,
    primary_positions: wp.array(dtype=wp.vec3),
    primary_velocities: wp.array(dtype=wp.vec3),
    secondary_positions: wp.array(dtype=wp.vec3),
    support_radius: float,
    neighbor_counts: wp.array(dtype=wp.int32),
    particle_types: wp.array(dtype=wp.int32),
    fluid_velocities: wp.array(dtype=wp.vec3),
    weight_sums: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    position = secondary_positions[index]
    radius_squared = support_radius * support_radius
    count = int(0)
    velocity_sum = wp.vec3(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    query = wp.hash_grid_query(grid, position, support_radius)
    for primary_index in query:
        delta = position - primary_positions[primary_index]
        distance_squared = wp.dot(delta, delta)
        if distance_squared <= radius_squared:
            count += 1
            q = wp.sqrt(distance_squared) / support_radius
            weight = float(0.0)
            if q <= 0.5:
                q_squared = q * q
                weight = 6.0 * q_squared * q - 6.0 * q_squared + 1.0
            elif q <= 1.0:
                one_minus_q = 1.0 - q
                weight = 2.0 * one_minus_q * one_minus_q * one_minus_q
            velocity_sum += primary_velocities[primary_index] * weight
            weight_sum += weight

    particle_type = int(0)
    if count < 6:
        particle_type = 1
    elif count > 20:
        particle_type = 2
    neighbor_counts[index] = count
    particle_types[index] = particle_type
    weight_sums[index] = weight_sum
    if weight_sum > 0.0:
        fluid_velocities[index] = velocity_sum / weight_sum
    else:
        fluid_velocities[index] = wp.vec3(0.0, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-pattern", required=True)
    parser.add_argument("--classification-directory", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--particle-radius", type=float, default=0.004)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def frame_path(pattern: str, frame: int) -> Path:
    marker = "#" * pattern.count("#")
    if not marker:
        raise ValueError("Primary pattern must contain a contiguous # frame marker")
    return Path(pattern.replace(marker, f"{frame:0{len(marker)}d}"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.particle_radius <= 0.0:
        raise ValueError("Particle radius must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")

    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError(f"CUDA device required, got {device}")
    support_radius = np.float32(4.0 * args.particle_radius)
    records = []
    total_particles = 0
    total_mismatches = 0

    for frame in sorted(set(args.frames)):
        primary_path = frame_path(args.primary_pattern, frame).resolve()
        with np.load(primary_path, allow_pickle=False) as archive:
            primary_positions_np = np.ascontiguousarray(
                archive["positions"], dtype=np.float32
            )
            primary_velocities_np = np.ascontiguousarray(
                archive["velocities"], dtype=np.float32
            )
        if (
            primary_positions_np.shape != primary_velocities_np.shape
            or primary_positions_np.ndim != 2
            or primary_positions_np.shape[1] != 3
        ):
            raise ValueError(f"{primary_path}: invalid primary arrays")
        classified = read_split(args.classification_directory.resolve(), frame)
        secondary_positions_np = np.ascontiguousarray(
            classified["position"], dtype=np.float32
        )
        expected_types = np.ascontiguousarray(
            classified["particle_type"], dtype=np.int32
        )

        primary_positions = wp.array(
            primary_positions_np, dtype=wp.vec3, device=device
        )
        primary_velocities = wp.array(
            primary_velocities_np, dtype=wp.vec3, device=device
        )
        secondary_positions = wp.array(
            secondary_positions_np, dtype=wp.vec3, device=device
        )
        neighbor_counts = wp.zeros(len(expected_types), dtype=wp.int32, device=device)
        particle_types = wp.zeros(len(expected_types), dtype=wp.int32, device=device)
        fluid_velocities = wp.zeros(len(expected_types), dtype=wp.vec3, device=device)
        weight_sums = wp.zeros(len(expected_types), dtype=wp.float32, device=device)
        grid = wp.HashGrid(128, 128, 128, device=device, dtype=wp.float32)
        grid.build(primary_positions, float(support_radius))
        wp.launch(
            query_foam_neighborhood,
            dim=len(expected_types),
            inputs=[
                grid.id,
                primary_positions,
                primary_velocities,
                secondary_positions,
                float(support_radius),
            ],
            outputs=[neighbor_counts, particle_types, fluid_velocities, weight_sums],
            device=device,
        )
        wp.synchronize_device(device)
        actual_types = particle_types.numpy()
        counts_np = neighbor_counts.numpy()
        fluid_velocities_np = fluid_velocities.numpy()
        weights_np = weight_sums.numpy()
        mismatch = actual_types != expected_types
        mismatch_count = int(np.count_nonzero(mismatch))
        total_mismatches += mismatch_count
        total_particles += len(expected_types)
        records.append(
            {
                "frame": frame,
                "primary_particles": len(primary_positions_np),
                "secondary_particles": len(expected_types),
                "classification_mismatches": mismatch_count,
                "classification_bit_exact": mismatch_count == 0,
                "neighbor_count_minimum": int(counts_np.min()) if len(counts_np) else None,
                "neighbor_count_maximum": int(counts_np.max()) if len(counts_np) else None,
                "zero_weight_non_spray_particles": int(
                    np.count_nonzero((weights_np <= 0.0) & (actual_types != 1))
                ),
                "maximum_interpolated_fluid_speed": (
                    float(np.linalg.norm(fluid_velocities_np, axis=1).max())
                    if len(fluid_velocities_np)
                    else 0.0
                ),
                "primary_file": str(primary_path),
                "primary_sha256": sha256_file(primary_path),
            }
        )

    valid = total_mismatches == 0 and all(
        record["zero_weight_non_spray_particles"] == 0 for record in records
    )
    report = {
        "schema": "foamgenerator-warp-neighborhood-audit/v1",
        "valid": valid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "warp_version": wp.__version__,
        "particle_radius_m": args.particle_radius,
        "support_radius_m": float(support_radius),
        "classification_thresholds": {"spray_below": 6, "bubbles_above": 20},
        "kernel": "SPlisHSPlasH CubicKernel without normalization (cancels in weighted mean)",
        "classification_directory": str(args.classification_directory.resolve()),
        "frames": records,
        "particle_states_audited": total_particles,
        "classification_mismatches": total_mismatches,
        "classification_bit_exact": total_mismatches == 0,
        "finite_difference_velocity_used": False,
        "local_fluid_velocity_source": "GPU kernel-weighted native primary velocities",
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "audit_report.json", report)
    if not valid:
        obsolete = {
            "schema": "obsolete-artifact/v1",
            "reason": "CUDA Warp neighborhood gate did not reproduce FoamGenerator exactly",
            "replacement": None,
            "audit_report": str((output / "audit_report.json").resolve()),
        }
        (output / "OBSOLETE.json").write_text(
            json.dumps(obsolete, indent=2) + "\n", encoding="utf-8"
        )
        return 2
    print(json.dumps({"valid": True, "particle_states_audited": total_particles}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
