"""Exact closest-triangle sampling for an oriented, possibly open surface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


class TriangleSurfaceSampler:
    """Return signed normal distance and smooth normals for a triangle mesh.

    The sign is local and remains well-defined for an open free-surface sheet:
    it is the dot product between ``point - closest_point`` and the oriented
    smooth normal.  No watertight-volume assumption or height-field reduction
    is made.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        loaded = trimesh.load_mesh(
            self.path,
            process=False,
            maintain_order=True,
        )
        if isinstance(loaded, trimesh.Scene):
            geometries = tuple(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"Surface scene is empty: {self.path}")
            loaded = trimesh.util.concatenate(geometries)
        if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
            raise ValueError(f"Surface is not a non-empty triangle mesh: {self.path}")
        self.mesh = loaded
        self._cache_positions = None
        self._cache_result = None

    def _query(self, positions):
        positions = np.ascontiguousarray(positions, dtype=np.float64).reshape((-1, 3))
        if (
            self._cache_positions is not None
            and positions.shape == self._cache_positions.shape
            and np.array_equal(positions, self._cache_positions)
        ):
            return self._cache_result
        closest, distance, triangle_ids = trimesh.proximity.closest_point(
            self.mesh, positions
        )
        triangles = self.mesh.triangles[triangle_ids]
        barycentric = trimesh.triangles.points_to_barycentric(triangles, closest)
        vertex_ids = self.mesh.faces[triangle_ids]
        vertex_normals = self.mesh.vertex_normals[vertex_ids]
        normals = np.einsum("ni,nij->nj", barycentric, vertex_normals)
        lengths = np.linalg.norm(normals, axis=1)
        valid = np.isfinite(lengths) & (lengths > 1.0e-12)
        normals[valid] /= lengths[valid, None]
        normals[~valid] = self.mesh.face_normals[triangle_ids[~valid]]
        signed = np.einsum("ij,ij->i", positions - closest, normals)
        result = (
            closest.astype(np.float64, copy=False),
            signed.astype(np.float64, copy=False),
            np.asarray(distance, dtype=np.float64),
            normals.astype(np.float64, copy=False),
            np.asarray(triangle_ids, dtype=np.int64),
        )
        self._cache_positions = positions.copy()
        self._cache_result = result
        return result

    def sample_phi(self, positions):
        return self._query(positions)[1]

    def sample_normal(self, positions):
        return self._query(positions)[3]

    def closest(self, positions):
        return self._query(positions)

    def project(self, positions, offset=0.0):
        closest, _signed, distance, normals, triangle_ids = self._query(positions)
        return closest + float(offset) * normals, distance, normals, triangle_ids
