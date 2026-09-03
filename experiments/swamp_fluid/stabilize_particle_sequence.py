"""Apply conservative adaptive temporal stabilization to identity-stable PBD particles."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_binary_ply(path):
    with path.open("rb") as stream:
        count = None
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Invalid PLY header: {path}")
            if line.startswith(b"element vertex "):
                count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        points = np.fromfile(stream, dtype="<f4")
    if count is None or points.size != count * 3:
        raise RuntimeError(f"Invalid PLY payload: {path}")
    return points.reshape(count, 3)


def write_binary_ply(path, points):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment adaptively stabilized PhysX PBD particles; visual use only\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        np.asarray(points, dtype="<f4").tofile(stream)
    temporary.replace(path)


def metadata_digest(paths):
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        stat = path.stat()
        total_bytes += stat.st_size
        digest.update(
            f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return digest.hexdigest(), total_bytes


def descending_smooth_weight(values, full_below, off_above):
    if not 0 <= full_below < off_above:
        raise ValueError("Falloff thresholds must satisfy 0 <= full < off")
    t = np.clip((values - full_below) / (off_above - full_below), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def ascending_smooth_weight(values, zero_below, full_above):
    if not 0 <= zero_below < full_above:
        raise ValueError("Impact thresholds must satisfy 0 <= radius < full distance")
    t = np.clip((values - zero_below) / (full_above - zero_below), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--water-level", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=0.008)
    parser.add_argument("--impact-x", type=float, default=-0.73)
    parser.add_argument("--impact-z", type=float, default=0.78)
    parser.add_argument(
        "--impact-radius",
        type=float,
        default=0.45,
        help="No particle smoothing at or inside this X/Z distance.",
    )
    parser.add_argument(
        "--impact-full-distance",
        type=float,
        default=0.65,
        help="Impact-distance weight reaches one at this distance.",
    )
    parser.add_argument("--speed-full", type=float, default=0.02)
    parser.add_argument("--speed-off", type=float, default=0.12)
    parser.add_argument("--residual-full", type=float, default=0.0003)
    parser.add_argument("--residual-off", type=float, default=0.0010)
    parser.add_argument(
        "--max-correction",
        type=float,
        default=0.00075,
        help="Hard per-frame displacement limit in metres.",
    )
    parser.add_argument(
        "--spray-clearance",
        type=float,
        help="Disable filtering above water level plus this value (default: 0.5 spacing).",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0 or args.spacing <= 0 or args.max_correction <= 0:
        raise ValueError("FPS, spacing, and maximum correction must be positive")

    paths = sorted(args.input_directory.glob("particles_*.ply"))
    if args.frames is not None:
        if args.frames < 1:
            raise ValueError("--frames must be positive")
        paths = paths[: args.frames]
    if not paths:
        raise RuntimeError("No particle PLY sequence found")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    spray_clearance = (
        args.spray_clearance
        if args.spray_clearance is not None
        else 0.5 * args.spacing
    )
    input_digest, total_bytes = metadata_digest(paths)
    configuration = {
        "schema": 2,
        "algorithm": "adaptive-symmetric-binomial",
        "input_directory": str(args.input_directory.resolve()),
        "input_count": len(paths),
        "input_metadata_sha256": input_digest,
        "input_total_bytes": total_bytes,
        "fps": args.fps,
        "water_level": args.water_level,
        "spacing": args.spacing,
        "impact": [
            args.impact_x,
            args.impact_z,
            args.impact_radius,
            args.impact_full_distance,
        ],
        "speed_full": args.speed_full,
        "speed_off": args.speed_off,
        "residual_full": args.residual_full,
        "residual_off": args.residual_off,
        "max_correction": args.max_correction,
        "spray_clearance": spray_clearance,
        "endpoints": "preserved",
    }
    manifest_path = args.output_directory / "stabilization_manifest.json"
    existing_outputs = list(args.output_directory.glob("particles_*.ply"))
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("configuration") != configuration and not args.force:
            raise RuntimeError(
                f"Existing stabilization manifest does not match: {manifest_path}. "
                "Use a new output directory or pass --force."
            )
    elif existing_outputs and not args.force:
        raise RuntimeError(
            f"{args.output_directory} contains PLY files without a matching manifest. "
            "Use a new output directory or pass --force after verifying the target."
        )

    previous = read_binary_ply(paths[0])
    current = previous
    summary = []
    for index, path in enumerate(paths):
        following = (
            read_binary_ply(paths[index + 1]) if index + 1 < len(paths) else current
        )
        if previous.shape != current.shape or current.shape != following.shape:
            raise RuntimeError("Particle identity/count changed across frames")

        target = 0.25 * previous + 0.50 * current + 0.25 * following
        correction = target - current
        correction_norm = np.linalg.norm(correction, axis=1)
        speed = np.linalg.norm(following - previous, axis=1) * (0.5 * args.fps)
        impact_distance = np.linalg.norm(
            current[:, (0, 2)]
            - np.asarray((args.impact_x, args.impact_z), dtype=np.float32),
            axis=1,
        )

        weight = descending_smooth_weight(
            speed, args.speed_full, args.speed_off
        )
        weight *= descending_smooth_weight(
            correction_norm, args.residual_full, args.residual_off
        )
        weight *= ascending_smooth_weight(
            impact_distance, args.impact_radius, args.impact_full_distance
        )
        weight[current[:, 1] > args.water_level + spray_clearance] = 0.0
        if index == 0 or index == len(paths) - 1:
            weight.fill(0.0)

        applied = correction * weight[:, None]
        applied_norm = np.linalg.norm(applied, axis=1)
        over_limit = applied_norm > args.max_correction
        if np.any(over_limit):
            applied[over_limit] *= (
                args.max_correction / applied_norm[over_limit]
            )[:, None]
            applied_norm[over_limit] = args.max_correction
        stabilized = current + applied

        output = args.output_directory / path.name
        if args.force or not output.is_file():
            write_binary_ply(output, stabilized)
        diagnostics = {
            "frame": index,
            "adjusted_fraction": float(np.mean(weight > 0.01)),
            "full_weight_fraction": float(np.mean(weight >= 0.99)),
            "capped_particles": int(np.count_nonzero(over_limit)),
            "median_correction_m": float(np.median(applied_norm)),
            "p99_correction_m": float(np.quantile(applied_norm, 0.99)),
            "maximum_correction_m": float(np.max(applied_norm)),
        }
        summary.append(diagnostics)
        print(
            f"[particle-stabilize] {index + 1:03d}/{len(paths):03d} {output.name} "
            f"adjusted={diagnostics['adjusted_fraction']:.2%} "
            f"full={diagnostics['full_weight_fraction']:.2%} "
            f"p99={1000.0 * diagnostics['p99_correction_m']:.3f}mm "
            f"max={1000.0 * diagnostics['maximum_correction_m']:.3f}mm "
            f"capped={diagnostics['capped_particles']}",
            flush=True,
        )
        previous, current = current, following

    payload = {
        "configuration": configuration,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "frames": summary,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(
        f"STABILIZED_PARTICLES={args.output_directory} frames={len(paths)} "
        "visual_only=true"
    )


if __name__ == "__main__":
    main()
