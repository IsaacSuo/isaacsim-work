"""Sparse point collision queries for markers outside the carrier grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .collision_fields import resolve_motion_matrix
from .scene_contract import ColliderSpec, SceneContract
from .terrain_fields import load_open_terrain_mesh, query_open_terrain_points


def _normalized(vectors, fallback=(0.0, 1.0, 0.0)):
    vectors = np.asarray(vectors, dtype=np.float64).copy()
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1.0e-12)
    vectors[valid] /= lengths[valid, None]
    vectors[~valid] = fallback
    return vectors


def interpolate_rigid_pose(motion, snapshot0, snapshot1, alpha, length_scale):
    """Interpolate translation and rotation while preserving a rigid pose."""

    row0, translation0 = resolve_motion_matrix(
        motion, snapshot0, length_scale=length_scale
    )
    if snapshot1 is None or motion.kind == "static":
        return row0, translation0
    row1, translation1 = resolve_motion_matrix(
        motion, snapshot1, length_scale=length_scale
    )
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rotations = Rotation.from_matrix(np.stack((row0.T, row1.T), axis=0))
    interpolated = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0].T
    translation = (1.0 - alpha) * translation0 + alpha * translation1
    return interpolated, translation


def _load_mesh_collider(collider):
    if collider.shape != "mesh":
        return None
    import trimesh

    mesh = trimesh.load(
        collider.parameters["path"], force="mesh", process=False, validate=False
    )
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Collider {collider.identifier!r} is not a triangle mesh")
    if collider.parameters["require_watertight"] and not mesh.is_watertight:
        raise ValueError(f"Collider {collider.identifier!r} is not watertight")
    if not mesh.is_winding_consistent or mesh.volume <= 0.0:
        raise ValueError(f"Collider {collider.identifier!r} has invalid winding")
    return mesh


@dataclass
class SparsePointCollisionScene:
    """Loaded scene contract for low-count, arbitrary-position SDF queries."""

    contract: SceneContract

    def __post_init__(self):
        self.terrain = (
            load_open_terrain_mesh(
                self.contract.terrain,
                length_scale=self.contract.metres_per_unit,
            )
            if self.contract.terrain is not None
            else None
        )
        self.collider_meshes = {
            collider.identifier: _load_mesh_collider(collider)
            for collider in self.contract.colliders
            if "solid" in collider.roles
        }

    def metadata(self):
        return {
            "mode": "sparse_arbitrary_point_query",
            "terrain": None if self.terrain is None else self.terrain.metadata,
            "colliders": [
                {
                    "id": collider.identifier,
                    "shape": collider.shape,
                    "motion": collider.motion.kind,
                }
                for collider in self.contract.colliders
                if "solid" in collider.roles
            ],
            "open_boundary_policy": "invalid query; never extrude an open edge",
            "union_policy": "minimum finite negative-inside signed distance",
            "pose_interpolation": "translation lerp plus quaternion slerp",
        }

    def _query_collider(self, collider, points, snapshot0, snapshot1, alpha):
        row_rotation, translation = interpolate_rigid_pose(
            collider.motion,
            snapshot0,
            snapshot1,
            alpha,
            self.contract.metres_per_unit,
        )
        scale = self.contract.metres_per_unit
        local = (points - translation) @ row_rotation.T
        if collider.shape == "sphere":
            radius = scale * float(collider.parameters["radius"])
            length = np.linalg.norm(local, axis=1)
            normal_local = _normalized(local)
            distance = length - radius
        elif collider.shape == "box":
            half = scale * np.asarray(collider.parameters["half_extents"], np.float64)
            q = np.abs(local) - half
            outside_delta = np.sign(local) * np.maximum(q, 0.0)
            outside_length = np.linalg.norm(outside_delta, axis=1)
            outside = outside_length > 0.0
            distance = outside_length + np.minimum(np.max(q, axis=1), 0.0)
            normal_local = np.zeros_like(local)
            normal_local[outside] = _normalized(outside_delta[outside])
            inside_ids = np.flatnonzero(~outside)
            if len(inside_ids):
                axes = np.argmax(q[inside_ids], axis=1)
                signs = np.sign(local[inside_ids, axes])
                signs[signs == 0.0] = 1.0
                normal_local[inside_ids, axes] = signs
        elif collider.shape == "capsule":
            axis = {"x": 0, "y": 1, "z": 2}[collider.parameters["axis"]]
            half = scale * float(collider.parameters["half_length"])
            radius = scale * float(collider.parameters["radius"])
            closest = np.zeros_like(local)
            closest[:, axis] = np.clip(local[:, axis], -half, half)
            delta = local - closest
            length = np.linalg.norm(delta, axis=1)
            fallback = np.zeros(3)
            fallback[(axis + 1) % 3] = 1.0
            normal_local = _normalized(delta, fallback=fallback)
            distance = length - radius
        elif collider.shape == "plane":
            distance = local[:, 1] - scale * float(collider.parameters["offset"])
            normal_local = np.zeros_like(local)
            normal_local[:, 1] = 1.0
        else:
            import trimesh

            mesh = self.collider_meshes[collider.identifier]
            authored_points = local / scale
            # trimesh is positive inside; this system is negative inside.
            distance = -np.asarray(
                trimesh.proximity.signed_distance(mesh, authored_points),
                dtype=np.float64,
            ) * scale
            _, _, triangle_id = trimesh.proximity.closest_point(mesh, authored_points)
            normal_local = np.asarray(mesh.face_normals[triangle_id], dtype=np.float64)
        normal_world = _normalized(normal_local @ row_rotation)
        return np.asarray(distance, np.float64), normal_world

    def query(self, points, snapshot0=None, snapshot1=None, alpha=0.0):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite Nx3 array")
        count = len(points)
        distance = np.full(count, np.inf, dtype=np.float64)
        normal = np.zeros((count, 3), dtype=np.float64)
        source = np.zeros(count, dtype=np.uint16)
        boundary_invalid = np.zeros(count, dtype=bool)
        source_names = {0: "none"}
        next_source = 1

        if self.terrain is not None:
            result = query_open_terrain_points(self.terrain, points)
            take = result["valid"] & (result["distance"] < distance)
            distance[take] = result["distance"][take]
            normal[take] = result["normal"][take]
            source[take] = next_source
            boundary_invalid |= result["touches_boundary"]
            source_names[next_source] = self.contract.terrain.identifier
            next_source += 1

        for collider in self.contract.colliders:
            if "solid" not in collider.roles:
                continue
            collider_distance, collider_normal = self._query_collider(
                collider, points, snapshot0, snapshot1, alpha
            )
            take = np.isfinite(collider_distance) & (collider_distance < distance)
            distance[take] = collider_distance[take]
            normal[take] = collider_normal[take]
            source[take] = next_source
            source_names[next_source] = collider.identifier
            next_source += 1
        valid = np.isfinite(distance) & (source != 0)
        return {
            "distance": distance,
            "normal": normal,
            "valid": valid,
            "source": source,
            "source_names": source_names,
            "open_boundary_invalid": boundary_invalid,
        }
