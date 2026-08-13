"""Reconstruct one PhysX particle frame with SplashSurf recommended settings."""

import argparse
import time

import numpy as np
import pysplashsurf


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--smoothing-length", type=float, default=2.0)
parser.add_argument("--cube-size", type=float, default=0.5)
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--mesh-smoothing-iters", type=int, default=25)
parser.add_argument("--normals-smoothing-iters", type=int, default=10)
args = parser.parse_args()

source = np.load(args.input)
positions = np.asarray(source["positions"], dtype=np.float32)
particle_radius = float(source["particle_radius"])

started = time.perf_counter()
mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
    positions,
    particle_radius=particle_radius,
    smoothing_length=args.smoothing_length,
    cube_size=args.cube_size,
    iso_surface_threshold=args.threshold,
    mesh_cleanup=True,
    compute_normals=True,
    normals_smoothing_iters=args.normals_smoothing_iters,
    mesh_smoothing_iters=args.mesh_smoothing_iters,
    mesh_smoothing_weights=True,
)
elapsed = time.perf_counter() - started

vertices = np.asarray(mesh_with_data.mesh.vertices, dtype=np.float32)
triangles = np.asarray(mesh_with_data.mesh.triangles, dtype=np.int32)
normals = np.asarray(
    mesh_with_data.point_attributes["normals"], dtype=np.float32
)
if not np.isfinite(normals).all():
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    normals = np.zeros_like(vertices)
    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > 1e-12
    normals[valid_normals] /= normal_lengths[valid_normals, None]
    normals[~valid_normals] = (0.0, 1.0, 0.0)
    print("[splashsurf] replaced non-finite normals with geometric normals")

np.savez_compressed(
    args.output,
    vertices=vertices,
    triangles=triangles,
    normals=normals,
)
print(
    f"[splashsurf] particles={len(positions)}, "
    f"vertices={mesh_with_data.nvertices}, triangles={mesh_with_data.ncells}, "
    f"seconds={elapsed:.3f}, output={args.output}"
)
