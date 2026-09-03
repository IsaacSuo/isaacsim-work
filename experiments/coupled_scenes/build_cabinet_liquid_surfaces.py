"""Reconstruct selected coupled-scene particle frames as renderable water surfaces."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np


PRODUCT = "coupled_scene_liquid_surface_sequence"
SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("particle_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument(
        "--spacing",
        type=float,
        help="Override source-cache spacing; default is read from its manifest.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--surface-threshold", type=float, default=0.60)
    parser.add_argument("--smoothing-length", type=float, default=2.0)
    parser.add_argument("--cube-size", type=float, default=1.0)
    parser.add_argument("--mesh-smoothing-iters", type=int, default=25)
    parser.add_argument(
        "--mesh-smoothing-weights",
        choices=("on", "off"),
        default="on",
        help="Use Splashsurf feature-preserving mesh-smoothing weights.",
    )
    parser.add_argument("--normal-smoothing-iters", type=int, default=10)
    parser.add_argument(
        "--allow-invalid-source-preview",
        action="store_true",
        help=(
            "Allow an artist preview from a complete cache that failed its "
            "physics gates; the override is recorded in surface provenance."
        ),
    )
    parser.add_argument(
        "--splashsurf",
        type=Path,
        default=Path("/mnt/y/tools/pysplashsurf/.venv/Scripts/pysplashsurf.exe"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def windows_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        return str(resolved)
    text = str(resolved)
    if text.startswith("/mnt/") and len(text) > 7:
        drive = text[5].upper()
        relative = text[7:].replace("/", "\\")
        return str(PureWindowsPath(f"{drive}:\\{relative}"))
    raise ValueError(f"Windows executable cannot consume this path: {resolved}")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_binary_ply(path: Path, positions: np.ndarray) -> None:
    positions = np.ascontiguousarray(positions, dtype="<f4")
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("Particle positions must have shape (n, 3)")
    if not len(positions) or not np.all(np.isfinite(positions)):
        raise ValueError("Particle positions must be non-empty and finite")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment coupled-scene primary liquid particles\n"
        f"element vertex {len(positions)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        stream.write(positions.tobytes(order="C"))
    temporary.replace(path)


def count_obj(path: Path) -> tuple[int, int]:
    vertices = 0
    faces = 0
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices += 1
            elif line.startswith("f "):
                faces += 1
    return vertices, faces


def main():
    args = parse_args()
    particle_directory = args.particle_directory.resolve()
    output_directory = args.output_directory.resolve()
    source_manifest_path = particle_directory / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    state = source_manifest.get("state") or {}
    if not state.get("complete"):
        raise RuntimeError("Primary-fluid cache is not complete")
    if not state.get("valid") and not args.allow_invalid_source_preview:
        raise RuntimeError("Primary-fluid cache is not valid")
    source_spacing = float(source_manifest["source"]["spacing"])
    spacing = float(args.spacing) if args.spacing is not None else source_spacing
    if min(
        spacing,
        args.smoothing_length,
        args.cube_size,
        args.workers,
        args.threads_per_worker,
    ) <= 0:
        raise ValueError("Spacing, workers, and threads-per-worker must be positive")
    if len(set(args.frames)) != len(args.frames):
        raise ValueError("Requested frames must be unique")

    source_samples = {
        int(sample["frame"]): sample for sample in source_manifest.get("samples", [])
    }
    missing = sorted(set(args.frames) - set(source_samples))
    if missing:
        raise RuntimeError(f"Requested particle frames do not exist: {missing}")
    empty = [
        frame for frame in args.frames if int(source_samples[frame]["particle_count"]) == 0
    ]
    if empty:
        raise RuntimeError(f"Splashsurf cannot reconstruct empty frames: {empty}")
    if not args.splashsurf.is_file():
        raise FileNotFoundError(args.splashsurf)

    output_directory.mkdir(parents=True, exist_ok=True)
    particle_ply_directory = output_directory / "particles"
    surface_directory = output_directory / "surface"
    particle_ply_directory.mkdir(parents=True, exist_ok=True)
    surface_directory.mkdir(parents=True, exist_ok=True)
    equal_volume_radius = math.pow(3.0 / (4.0 * math.pi), 1.0 / 3.0) * spacing
    smoothing_length = args.smoothing_length
    cube_size = args.cube_size
    configuration = {
        "product": PRODUCT,
        "schema": SCHEMA,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_cache_valid": bool(state.get("valid")),
        "invalid_source_preview_override": bool(
            args.allow_invalid_source_preview and not state.get("valid")
        ),
        "selected_frames": sorted(args.frames),
        "spacing_m": spacing,
        "source_spacing_m": source_spacing,
        "particle_radius_m": equal_volume_radius,
        "smoothing_length": smoothing_length,
        "cube_size": cube_size,
        "voxel_size_m": equal_volume_radius * cube_size,
        "surface_threshold": args.surface_threshold,
        "mesh_smoothing_iters": args.mesh_smoothing_iters,
        "mesh_smoothing_weights": args.mesh_smoothing_weights,
        "normal_smoothing_iters": args.normal_smoothing_iters,
        "splashsurf_executable": str(args.splashsurf.resolve()),
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
    }
    manifest_path = output_directory / "surface_manifest.json"
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration") != configuration:
            raise RuntimeError(
                f"Existing surface cache has different provenance: {manifest_path}"
            )
        if (existing.get("state") or {}).get("complete"):
            print(f"[surface-reuse] {manifest_path}")
            return 0

    atomic_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "product": PRODUCT,
            "configuration": configuration,
            "state": {"complete": False, "valid": None, "completed_frames": 0},
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    def build_frame(frame: int) -> dict:
        source_path = particle_directory / source_samples[frame]["file"]
        ply_path = particle_ply_directory / f"particles_{frame:04d}.ply"
        surface_path = surface_directory / f"surface_{frame:04d}.obj"
        if surface_path.is_file() and surface_path.stat().st_size > 1_000 and not args.force:
            vertices, faces = count_obj(surface_path)
            return {
                "frame": frame,
                "particle_count": int(source_samples[frame]["particle_count"]),
                "surface": surface_path.name,
                "vertices": vertices,
                "triangles": faces,
                "bytes": surface_path.stat().st_size,
                "sha256": sha256_file(surface_path),
                "status": "reused",
            }
        positions = np.load(source_path, allow_pickle=False)["positions"]
        atomic_binary_ply(ply_path, positions)
        temporary_surface = surface_directory / f"surface_{frame:04d}.tmp.obj"
        command = [
            str(args.splashsurf.resolve()),
            "reconstruct",
            windows_path(ply_path),
            "-r", str(equal_volume_radius),
            "-l", str(smoothing_length),
            "-c", str(cube_size),
            "-t", str(args.surface_threshold),
            f"--mesh-smoothing-weights={args.mesh_smoothing_weights}",
            "--mesh-smoothing-iters", str(args.mesh_smoothing_iters),
            "--mesh-cleanup=on",
            "--normals=on",
            "--normals-smoothing-iters", str(args.normal_smoothing_iters),
            "--check-mesh=off",
            "--num-threads", str(args.threads_per_worker),
            "-o", windows_path(temporary_surface),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"Splashsurf failed for frame {frame}:\n{completed.stdout}\n{completed.stderr}"
            )
        if not temporary_surface.is_file() or temporary_surface.stat().st_size <= 1_000:
            raise RuntimeError(f"Splashsurf wrote no usable surface for frame {frame}")
        temporary_surface.replace(surface_path)
        vertices, faces = count_obj(surface_path)
        if not vertices or not faces:
            raise RuntimeError(f"Surface frame {frame} contains no mesh")
        return {
            "frame": frame,
            "particle_count": int(len(positions)),
            "surface": surface_path.name,
            "vertices": vertices,
            "triangles": faces,
            "bytes": surface_path.stat().st_size,
            "sha256": sha256_file(surface_path),
            "status": "rebuilt",
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_frame, frame): frame for frame in args.frames}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[surface] frame={result['frame']:04d} "
                f"particles={result['particle_count']} vertices={result['vertices']} "
                f"triangles={result['triangles']}",
                flush=True,
            )

    results.sort(key=lambda row: row["frame"])
    valid = len(results) == len(args.frames) and all(
        row["vertices"] > 0 and row["triangles"] > 0 for row in results
    )
    atomic_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "product": PRODUCT,
            "configuration": configuration,
            "state": {
                "complete": True,
                "valid": valid,
                "completed_frames": len(results),
                "frames": results,
            },
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    if not valid:
        raise RuntimeError("Liquid surface sequence validation failed")
    print(f"[complete] valid=True manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
