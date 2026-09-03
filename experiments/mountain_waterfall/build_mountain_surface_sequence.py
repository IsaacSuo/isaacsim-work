"""Build a resumable Mountain Splashsurf primary-surface sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("particle_directory", type=Path)
parser.add_argument("output_directory", type=Path)
parser.add_argument("--parent-report", type=Path, required=True)
parser.add_argument("--wall-contact-audit", type=Path, required=True)
parser.add_argument("--stream-connectivity-audit", type=Path, required=True)
parser.add_argument("--pysplashsurf", type=Path, required=True)
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--end-frame", type=int, required=True)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--threads-per-worker", type=int, default=5)
parser.add_argument("--particle-radius", type=float, required=True)
parser.add_argument("--smoothing-length", type=float, default=1.612)
parser.add_argument("--cube-size", type=float, default=0.6045)
parser.add_argument("--surface-threshold", type=float, default=0.6)
parser.add_argument("--mesh-smoothing-iterations", type=int, default=8)
parser.add_argument("--normal-smoothing-iterations", type=int, default=8)
parser.add_argument("--minimum-y", type=float, default=-0.38)
parser.add_argument("--minimum-component-triangles", type=int, default=100)
parser.add_argument("--domain-min", nargs=3, type=float, default=(2.25, -0.5, 0.65))
parser.add_argument("--domain-max", nargs=3, type=float, default=(5.1, 2.0, 2.75))
parser.add_argument("--keep-small-primary-components", action="store_true")
args = parser.parse_args()

if args.start_frame < 0 or args.end_frame < args.start_frame:
    raise ValueError("Invalid frame range")
if args.workers < 1 or args.threads_per_worker < 1:
    raise ValueError("workers and threads-per-worker must be positive")
for path in (
    args.parent_report,
    args.wall_contact_audit,
    args.stream_connectivity_audit,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("valid") is not True:
        raise RuntimeError(f"Refusing to build from an invalid parent gate: {path}")
if not args.pysplashsurf.is_file():
    raise FileNotFoundError(args.pysplashsurf)

script_directory = Path(__file__).resolve().parent
clip_script = script_directory / "clip_waterfall_pool_surface.py"
prune_script = script_directory / "prune_detached_waterfall_components.py"
directories = {
    name: args.output_directory / name
    for name in ("raw", "clipped", "final_primary", "clip_reports", "prune_reports")
}
for directory in directories.values():
    directory.mkdir(parents=True, exist_ok=True)
manifest_path = args.output_directory / "build_manifest.json"


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command):
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(map(str, command))}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )


def build_frame(frame):
    stem = f"surface_{frame:04d}"
    particle = args.particle_directory / f"particles_{frame:04d}.ply"
    raw = directories["raw"] / f"{stem}.obj"
    clipped = directories["clipped"] / f"{stem}.obj"
    final = directories["final_primary"] / f"{stem}.obj"
    clip_report = directories["clip_reports"] / f"{stem}.json"
    prune_report = directories["prune_reports"] / f"{stem}.json"
    if not particle.is_file():
        raise FileNotFoundError(particle)
    if not raw.is_file():
        run(
            [
                str(args.pysplashsurf),
                "reconstruct",
                str(particle),
                "--particle-radius",
                str(args.particle_radius),
                "--smoothing-length",
                str(args.smoothing_length),
                "--cube-size",
                str(args.cube_size),
                "--surface-threshold",
                str(args.surface_threshold),
                "--mesh-smoothing-iters",
                str(args.mesh_smoothing_iterations),
                "--normals=on",
                "--normals-smoothing-iters",
                str(args.normal_smoothing_iterations),
                "--mesh-cleanup=on",
                "--check-mesh=off",
                "--num-threads",
                str(args.threads_per_worker),
                "--output-file",
                str(raw),
            ]
        )
    if not clipped.is_file() or not clip_report.is_file():
        if clipped.exists() or clip_report.exists():
            raise RuntimeError(f"Incomplete clip pair requires manual review: {stem}")
        run(
            [
                sys.executable,
                str(clip_script),
                str(raw),
                str(clipped),
                "--minimum-y",
                str(args.minimum_y),
                "--report",
                str(clip_report),
            ]
        )
    if not final.is_file() or not prune_report.is_file():
        if final.exists() or prune_report.exists():
            raise RuntimeError(f"Incomplete prune pair requires manual review: {stem}")
        command = [
            sys.executable,
            str(prune_script),
            str(clipped),
            str(final),
            "--report",
            str(prune_report),
            "--minimum-component-triangles",
            str(args.minimum_component_triangles),
            "--domain-min",
            *map(str, args.domain_min),
            "--domain-max",
            *map(str, args.domain_max),
        ]
        if not args.keep_small_primary_components:
            command.append("--remove-small-components-everywhere")
        run(command)
    return {
        "frame": frame,
        "particle": str(particle.resolve()),
        "final_primary": str(final.resolve()),
        "clip_report": str(clip_report.resolve()),
        "prune_report": str(prune_report.resolve()),
        "status": "complete",
    }


frames = list(range(args.start_frame, args.end_frame + 1))
manifest = {
    "schema": 1,
    "product": "mountain_splashsurf_sequence_build",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "complete": False,
    "frame_range": [args.start_frame, args.end_frame],
    "parameters": {
        "particle_radius_m": args.particle_radius,
        "smoothing_length": args.smoothing_length,
        "cube_size": args.cube_size,
        "surface_threshold": args.surface_threshold,
        "mesh_smoothing_iterations": args.mesh_smoothing_iterations,
        "normal_smoothing_iterations": args.normal_smoothing_iterations,
        "minimum_y_m": args.minimum_y,
        "minimum_component_triangles": args.minimum_component_triangles,
        "small_components_are_secondary_droplets": not args.keep_small_primary_components,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
    },
    "parents": {
        "physx": str(args.parent_report.resolve()),
        "wall_contact": str(args.wall_contact_audit.resolve()),
        "stream_connectivity": str(args.stream_connectivity_audit.resolve()),
    },
    "frames": [],
    "errors": [],
}
atomic_json(manifest_path, manifest)
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(build_frame, frame): frame for frame in frames}
    for future in as_completed(futures):
        frame = futures[future]
        try:
            manifest["frames"].append(future.result())
            print(f"[mountain-surface] frame={frame:04d} complete", flush=True)
        except Exception as error:
            manifest["errors"].append({"frame": frame, "error": str(error)})
            print(f"[mountain-surface] frame={frame:04d} failed: {error}", flush=True)
        manifest["frames"].sort(key=lambda item: item["frame"])
        atomic_json(manifest_path, manifest)

manifest["complete"] = not manifest["errors"] and len(manifest["frames"]) == len(frames)
manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
atomic_json(manifest_path, manifest)
if not manifest["complete"]:
    raise RuntimeError(f"Surface sequence is incomplete; see {manifest_path}")
print(f"MOUNTAIN_SURFACE_SEQUENCE={args.output_directory.resolve()}")
