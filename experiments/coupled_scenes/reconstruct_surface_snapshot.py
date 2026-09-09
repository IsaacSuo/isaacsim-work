"""Reconstruct a diagnostic NPZ snapshot without inventing a valid fluid cache."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from build_cabinet_liquid_surfaces import atomic_binary_ply, sha256_file, windows_path, count_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing", type=float, default=.004)
    parser.add_argument("--mesh-smoothing-iters", type=int, default=25)
    args = parser.parse_args()
    if not math.isfinite(args.spacing) or args.spacing <= 0 or args.mesh_smoothing_iters < 0:
        raise ValueError("Spacing must be positive and smoothing iterations nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.snapshot, allow_pickle=False) as data:
        positions = data["positions"]
        seconds = float(data["simulated_seconds"])
    ply = args.output/"particles.ply"
    mesh = args.output/"water.obj"
    atomic_binary_ply(ply, positions)
    radius = (3/(4*math.pi))**(1/3)*args.spacing
    command = ["/mnt/y/tools/pysplashsurf/.venv/Scripts/pysplashsurf.exe", "reconstruct",
               windows_path(ply), "-r", str(radius), "-l", "2.0", "-c", "1.0", "-t", "0.60",
               "--mesh-smoothing-iters", str(args.mesh_smoothing_iters),
               "--mesh-smoothing-weights=on", "--mesh-cleanup=on", "--normals=on",
               "--normals-smoothing-iters", "10", "--check-mesh=off",
               "--num-threads", "8", "-o", windows_path(mesh)]
    report = dict(source=str(args.snapshot.resolve()),source_sha256=sha256_file(args.snapshot),
                  particle_count=len(positions),simulated_seconds=seconds,physics_gate_passed=False,
                  purpose="visual inspection of final state, not certified still water",
                  mesh_smoothing_iters=args.mesh_smoothing_iters,mesh_smoothing_weights="on",
                  normal_smoothing_iters=10,particle_radius_m=radius,
                  smoothing_length=2.,cube_size=1.,surface_threshold=.6,command=command,complete=False)
    manifest = args.output/"snapshot_surface.json"
    manifest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (args.output/"reconstruct.log").open("w",encoding="utf-8") as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Reconstruction failed: {result.returncode}; see reconstruct.log")
    vertices, triangles = count_obj(mesh)
    assert vertices and triangles
    report.update(complete=True,vertices=vertices,triangles=triangles,mesh_sha256=sha256_file(mesh))
    manifest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
