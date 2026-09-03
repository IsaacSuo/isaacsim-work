"""Audit stable secondary-particle caches before Blender production rendering."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LAYERS = ("spray", "foam", "bubbles")
REQUIRED_ARRAYS = ("id", "position", "velocity", "radius", "age", "lifetime", "opacity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("secondary_directory", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--physics-report", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sphere-clearance", type=float, default=0.0005)
    parser.add_argument("--maximum-frame-displacement", type=float, default=0.30)
    return parser.parse_args()


def frame_number(path):
    return int(path.stem.split("_")[-1])


def load_cache(path):
    with np.load(path) as cache:
        missing = [name for name in REQUIRED_ARRAYS if name not in cache]
        if missing:
            raise RuntimeError(f"{path} is missing arrays: {missing}")
        payload = {name: np.asarray(cache[name]).copy() for name in REQUIRED_ARRAYS}
        payload["shape"] = (
            np.asarray(cache["shape"]).copy()
            if "shape" in cache
            else np.ones((len(payload["id"]), 3), dtype=np.float32)
        )
    return payload


def main():
    args = parse_args()
    if args.fps <= 0 or args.sphere_clearance < 0:
        raise ValueError("FPS must be positive and sphere clearance cannot be negative")

    manifest_path = args.secondary_directory / "secondary_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = {
        layer: sorted((args.secondary_directory / layer).glob("frame_*.npz"))
        for layer in LAYERS
    }
    frame_sets = {layer: {frame_number(path) for path in paths} for layer, paths in frame_paths.items()}
    if not frame_sets["spray"] or any(
        frames != frame_sets["spray"] for frames in frame_sets.values()
    ):
        raise RuntimeError(f"Secondary layers do not contain identical frame sets: {frame_sets}")

    physics_metrics = {}
    sphere_radius = float(manifest["configuration"].get("sphere_radius", 0.08))
    if args.physics_report is not None:
        physics = json.loads(args.physics_report.read_text(encoding="utf-8"))
        physics_metrics = {
            int(row["output_frame"]): row for row in physics.get("metrics", [])
        }
        sphere_radius = float(physics.get("impactor", {}).get("radius", sphere_radius))

    issues = []
    statistics = {}
    previous = {layer: None for layer in LAYERS}
    for layer in LAYERS:
        counts = []
        radii = []
        maximum_displacement = 0.0
        minimum_sphere_clearance = float("inf")
        maximum_shape_volume_error = 0.0
        for path in frame_paths[layer]:
            frame = frame_number(path)
            cache = load_cache(path)
            count = len(cache["id"])
            counts.append(count)
            radii.append(cache["radius"])
            expected_shapes = {
                "id": (count,),
                "position": (count, 3),
                "velocity": (count, 3),
                "radius": (count,),
                "age": (count,),
                "lifetime": (count,),
                "opacity": (count,),
                "shape": (count, 3),
            }
            for name, expected in expected_shapes.items():
                if cache[name].shape != expected:
                    issues.append(
                        f"{path}: {name} shape {cache[name].shape}, expected {expected}"
                    )
            if count:
                if np.any(np.diff(cache["id"].astype(np.int64)) <= 0):
                    issues.append(f"{path}: IDs are not strictly increasing and unique")
                for name in REQUIRED_ARRAYS[1:] + ("shape",):
                    if not np.all(np.isfinite(cache[name])):
                        issues.append(f"{path}: {name} contains non-finite values")
                if np.any(cache["radius"] <= 0) or np.any(cache["shape"] <= 0):
                    issues.append(f"{path}: non-positive radius or shape scale")
                if np.any(cache["age"] < 0) or np.any(cache["age"] >= cache["lifetime"]):
                    issues.append(f"{path}: age lies outside [0, lifetime)")
                if np.any((cache["opacity"] < 0) | (cache["opacity"] > 1)):
                    issues.append(f"{path}: opacity lies outside [0, 1]")
                shape_volume_error = np.max(
                    np.abs(np.prod(cache["shape"], axis=1) - 1.0)
                )
                maximum_shape_volume_error = max(
                    maximum_shape_volume_error, float(shape_volume_error)
                )

            old = previous[layer]
            if old is not None and count and len(old["id"]):
                shared, old_indices, new_indices = np.intersect1d(
                    old["id"], cache["id"], assume_unique=True, return_indices=True
                )
                if len(shared):
                    displacement = np.linalg.norm(
                        cache["position"][new_indices] - old["position"][old_indices],
                        axis=1,
                    )
                    maximum_displacement = max(
                        maximum_displacement, float(displacement.max())
                    )
                    if np.any(displacement > args.maximum_frame_displacement):
                        issues.append(
                            f"{path}: {layer} contains a shared-ID displacement above "
                            f"{args.maximum_frame_displacement:.3f} m/frame"
                        )

            if layer == "bubbles" and count and frame in physics_metrics:
                center = np.asarray(
                    physics_metrics[frame]["sphere_center"], dtype=np.float32
                )
                clearance = (
                    np.linalg.norm(cache["position"] - center[None, :], axis=1)
                    - sphere_radius
                    - cache["radius"] * np.max(cache["shape"], axis=1)
                )
                minimum_sphere_clearance = min(
                    minimum_sphere_clearance, float(clearance.min())
                )
                if np.any(clearance < args.sphere_clearance - 0.0003):
                    issues.append(f"{path}: one or more bubbles overlap the impactor")
            previous[layer] = cache

        all_radii = np.concatenate(radii) if radii else np.empty(0)
        statistics[layer] = {
            "minimum_count": int(min(counts)),
            "maximum_count": int(max(counts)),
            "first_nonempty_frame": next(
                (frame_number(path) for path, count in zip(frame_paths[layer], counts) if count),
                None,
            ),
            "radius_quantiles_m": (
                np.quantile(all_radii, [0.0, 0.5, 0.95, 0.99, 1.0]).tolist()
                if len(all_radii)
                else [0.0] * 5
            ),
            "maximum_shared_id_displacement_m_per_frame": maximum_displacement,
            "minimum_impactor_clearance_m": (
                minimum_sphere_clearance
                if np.isfinite(minimum_sphere_clearance)
                else None
            ),
            "maximum_shape_volume_error": maximum_shape_volume_error,
        }

    report = {
        "valid": not issues,
        "secondary_directory": str(args.secondary_directory.resolve()),
        "manifest_schema": manifest.get("schema"),
        "frames": len(frame_sets["spray"]),
        "statistics": statistics,
        "issues": issues,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output_report)
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
