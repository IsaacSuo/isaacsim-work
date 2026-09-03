"""Audited one-sided distance fields for explicitly selected open terrain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scene_contract import TerrainSpec


@dataclass(frozen=True)
class OpenTerrainMesh:
    mesh: object
    boundary_opposite_vertex: np.ndarray
    metadata: dict


def _resolve_static_terrain_transform(terrain, length_scale):
    """Resolve a schema-validated static terrain transform without solver deps."""

    length_scale = float(length_scale)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    matrix = np.asarray(terrain.motion.transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    if terrain.motion.matrix_layout == "row_translation":
        row_rotation = rotation
        translation = matrix[3, :3]
    else:
        row_rotation = rotation.T
        translation = matrix[:3, 3]
    return row_rotation, translation * length_scale


def load_obj_triangle_arrays(path):
    """Load vertices and triangulated faces from an OBJ without repair."""

    vertices = []
    faces = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"OBJ vertex is incomplete at line {line_number}")
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            indices = []
            for token in fields[1:]:
                vertex_token = token.split("/", 1)[0]
                index = int(vertex_token)
                if index == 0:
                    raise ValueError(f"OBJ face uses index zero at line {line_number}")
                index = index - 1 if index > 0 else len(vertices) + index
                indices.append(index)
            if len(indices) < 3:
                raise ValueError(f"OBJ face is incomplete at line {line_number}")
            for offset in range(1, len(indices) - 1):
                faces.append((indices[0], indices[offset], indices[offset + 1]))
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ValueError(f"OBJ contains no valid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        raise ValueError(f"OBJ contains no valid triangle faces: {path}")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError(f"OBJ contains an out-of-range face index: {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"OBJ contains non-finite vertices: {path}")
    return vertices, faces


def open_terrain_world_arrays(terrain: TerrainSpec, length_scale=1.0):
    """Return selected terrain vertices/faces transformed to solver metres."""

    vertices, faces = load_obj_triangle_arrays(terrain.parameters["path"])
    selection = terrain.parameters["selection"]["selected_mesh"]
    if int(selection.get("vertex_count", -1)) != len(vertices):
        raise ValueError("Terrain selection vertex count does not match the OBJ")
    if int(selection.get("triangle_count", -1)) != len(faces):
        raise ValueError("Terrain selection triangle count does not match the OBJ")
    length_scale = float(length_scale)
    row_rotation, translation = _resolve_static_terrain_transform(
        terrain, length_scale
    )
    world_vertices = length_scale * vertices @ row_rotation + translation
    return world_vertices, faces


def rasterize_open_terrain_heightfield(
    terrain: TerrainSpec,
    x_values,
    z_values,
    maximum_y=None,
    length_scale=1.0,
    barycentric_tolerance=1.0e-8,
):
    """Rasterize the highest selected support surface below ``maximum_y``.

    The source is already the audited fluid-facing component. Multiple hits in
    a gravity column are resolved by choosing the highest eligible support,
    never by taking the first hit from the complete authored scene. Vertical
    faces deliberately do not create floor samples.
    """

    x_values = np.asarray(x_values, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.float64)
    if (
        x_values.ndim != 1
        or z_values.ndim != 1
        or len(x_values) < 2
        or len(z_values) < 2
        or not np.isfinite(x_values).all()
        or not np.isfinite(z_values).all()
        or np.any(np.diff(x_values) <= 0.0)
        or np.any(np.diff(z_values) <= 0.0)
    ):
        raise ValueError("Terrain raster axes must be finite and strictly increasing")
    maximum_y = np.inf if maximum_y is None else float(maximum_y)
    if not np.isfinite(maximum_y) and maximum_y != np.inf:
        raise ValueError("maximum_y must be finite or None")
    tolerance = float(barycentric_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("barycentric_tolerance must be non-negative")
    vertices, faces = open_terrain_world_arrays(terrain, length_scale=length_scale)
    terrain_y = np.full((len(x_values), len(z_values)), np.nan, dtype=np.float64)
    projected_faces = 0
    eligible_samples = 0
    for face in faces:
        triangle = vertices[face]
        planar = triangle[:, (0, 2)]
        a, b, c = planar
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
            a[1] - c[1]
        )
        if abs(denominator) <= 1.0e-14:
            continue
        ix0 = max(0, int(np.searchsorted(x_values, planar[:, 0].min(), side="left") - 1))
        ix1 = min(len(x_values), int(np.searchsorted(x_values, planar[:, 0].max(), side="right") + 1))
        iz0 = max(0, int(np.searchsorted(z_values, planar[:, 1].min(), side="left") - 1))
        iz1 = min(len(z_values), int(np.searchsorted(z_values, planar[:, 1].max(), side="right") + 1))
        if ix0 >= ix1 or iz0 >= iz1:
            continue
        grid_x, grid_z = np.meshgrid(
            x_values[ix0:ix1], z_values[iz0:iz1], indexing="ij"
        )
        wa = (
            (b[1] - c[1]) * (grid_x - c[0])
            + (c[0] - b[0]) * (grid_z - c[1])
        ) / denominator
        wb = (
            (c[1] - a[1]) * (grid_x - c[0])
            + (a[0] - c[0]) * (grid_z - c[1])
        ) / denominator
        wc = 1.0 - wa - wb
        inside = (
            (wa >= -tolerance) & (wb >= -tolerance) & (wc >= -tolerance)
        )
        height = wa * triangle[0, 1] + wb * triangle[1, 1] + wc * triangle[2, 1]
        eligible = inside & (height <= maximum_y + 1.0e-12)
        if not np.any(eligible):
            continue
        target = terrain_y[ix0:ix1, iz0:iz1]
        replace = eligible & (~np.isfinite(target) | (height > target))
        target[replace] = height[replace]
        projected_faces += 1
        eligible_samples += int(np.count_nonzero(eligible))
    metadata = {
        "method": "highest_selected_support_below_cap",
        "global_first_hit_ray_cast": False,
        "maximum_y": None if maximum_y == np.inf else maximum_y,
        "projected_triangle_count": projected_faces,
        "source_triangle_count": int(len(faces)),
        "eligible_triangle_samples": eligible_samples,
        "valid_cells": int(np.count_nonzero(np.isfinite(terrain_y))),
        "total_cells": int(terrain_y.size),
    }
    return terrain_y.astype(np.float32), metadata


def _load_trimesh():
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            "Open terrain meshes require trimesh and its rtree dependency"
        ) from error
    return trimesh


def _edge_incidence(faces):
    """Return edge counts and boundary flags opposite each triangle vertex."""

    faces = np.asarray(faces, dtype=np.int64)
    edge_keys = []
    for face in faces:
        a, b, c = (int(value) for value in face)
        edge_keys.extend(
            (
                tuple(sorted((b, c))),
                tuple(sorted((c, a))),
                tuple(sorted((a, b))),
            )
        )
    counts = {}
    for edge in edge_keys:
        counts[edge] = counts.get(edge, 0) + 1
    boundary = np.asarray([counts[edge] == 1 for edge in edge_keys], dtype=bool)
    return counts, boundary.reshape((-1, 3))


def load_open_terrain_mesh(terrain: TerrainSpec, length_scale=1.0):
    """Load an authored open sheet without silently repairing its topology."""

    if terrain.representation != "open_triangle_mesh":
        raise ValueError(f"Unsupported terrain representation {terrain.representation!r}")
    trimesh = _load_trimesh()
    mesh = trimesh.load(
        terrain.parameters["path"], force="mesh", process=False, validate=False
    )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Terrain {terrain.identifier!r} is not a triangle mesh")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"Terrain {terrain.identifier!r} must be triangulated")
    if terrain.parameters["require_open"] and mesh.is_watertight:
        raise ValueError(
            f"Terrain {terrain.identifier!r} is watertight and must use a solid collider"
        )
    if not mesh.is_winding_consistent:
        raise ValueError(f"Terrain {terrain.identifier!r} has inconsistent winding")
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.isfinite(triangles).all() or np.any(doubled_area <= 1.0e-12):
        raise ValueError(f"Terrain {terrain.identifier!r} has invalid or degenerate faces")
    edge_counts, boundary_opposite_vertex = _edge_incidence(mesh.faces)
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    if nonmanifold_edges:
        raise ValueError(
            f"Terrain {terrain.identifier!r} has {nonmanifold_edges} non-manifold edges"
        )
    selection = terrain.parameters["selection"]
    selected_mesh = selection["selected_mesh"]
    if int(selected_mesh.get("triangle_count", -1)) != len(mesh.faces):
        raise ValueError("Terrain selection triangle count does not match the mesh")
    if int(selected_mesh.get("vertex_count", -1)) != len(mesh.vertices):
        raise ValueError("Terrain selection vertex count does not match the mesh")

    length_scale = float(length_scale)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    row_rotation, translation = _resolve_static_terrain_transform(
        terrain, length_scale
    )
    world_vertices = (
        length_scale * np.asarray(mesh.vertices, dtype=np.float64) @ row_rotation
        + translation
    )
    world_mesh = trimesh.Trimesh(
        vertices=world_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
        validate=False,
    )
    metadata = {
        "id": terrain.identifier,
        "representation": terrain.representation,
        "path": terrain.parameters["path"],
        "sha256": terrain.parameters["sha256"],
        "selection_path": terrain.parameters["selection_path"],
        "selection_sha256": terrain.parameters["selection_sha256"],
        "normal_convention": terrain.parameters["normal_convention"],
        "vertex_count": int(len(world_mesh.vertices)),
        "triangle_count": int(len(world_mesh.faces)),
        "boundary_edge_count": int(sum(count == 1 for count in edge_counts.values())),
        "connected_component_count": int(
            selection.get("selected_mesh", {}).get("connected_component_count", 0)
        ),
        "bounds_minimum": world_vertices.min(axis=0).astype(float).tolist(),
        "bounds_maximum": world_vertices.max(axis=0).astype(float).tolist(),
        "distance_method": "oriented_euclidean_closest_surface",
        "invalidity_rule": "closest point lies on an authored open boundary",
    }
    return OpenTerrainMesh(world_mesh, boundary_opposite_vertex, metadata)


def _barycentric_coordinates(points, triangles):
    """Return barycentric weights for paired points and non-degenerate triangles."""

    a = triangles[:, 0]
    ab = triangles[:, 1] - a
    ac = triangles[:, 2] - a
    ap = points - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denominator = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.stack((1.0 - v - w, v, w), axis=1)


def query_open_terrain_points(
    loaded: OpenTerrainMesh,
    points,
    boundary_barycentric_tolerance=2.0e-7,
):
    """Query an authored open sheet at arbitrary points without edge extrusion.

    The returned distance is positive on the fluid-facing side and negative on
    the solid side. A closest point touching an authored boundary is invalid,
    because an open edge is not permission to invent an infinite wall.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    tolerance = float(boundary_barycentric_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary_barycentric_tolerance must be non-negative")
    if not len(points):
        return {
            "distance": np.empty(0, dtype=np.float64),
            "normal": np.empty((0, 3), dtype=np.float64),
            "valid": np.empty(0, dtype=bool),
            "closest": np.empty((0, 3), dtype=np.float64),
            "triangle_id": np.empty(0, dtype=np.int64),
            "touches_boundary": np.empty(0, dtype=bool),
        }
    trimesh = _load_trimesh()
    mesh = loaded.mesh
    closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, points)
    closest = np.asarray(closest, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    triangle_id = np.asarray(triangle_id, dtype=np.int64)
    finite = (
        np.isfinite(closest).all(axis=1)
        & np.isfinite(distance)
        & (triangle_id >= 0)
        & (triangle_id < len(mesh.faces))
    )
    if not np.all(finite):
        raise RuntimeError("Open terrain closest-point query returned invalid data")
    triangles = np.asarray(mesh.triangles[triangle_id], dtype=np.float64)
    barycentric = _barycentric_coordinates(closest, triangles)
    touches_boundary = np.any(
        (barycentric <= tolerance)
        & loaded.boundary_opposite_vertex[triangle_id],
        axis=1,
    )
    normal = np.asarray(mesh.face_normals[triangle_id], dtype=np.float64)
    normal_side = np.einsum("ij,ij->i", points - closest, normal)
    signed = np.where(normal_side < 0.0, -distance, distance)
    signed[distance <= 1.0e-12] = 0.0
    valid = ~touches_boundary
    return {
        "distance": np.where(valid, signed, np.inf),
        "normal": normal,
        "valid": valid,
        "closest": closest,
        "triangle_id": triangle_id,
        "touches_boundary": touches_boundary,
    }


def open_terrain_collision_sdf(
    spec,
    terrain: TerrainSpec,
    length_scale=1.0,
    chunk_size=250_000,
    boundary_barycentric_tolerance=2.0e-7,
):
    """Build a one-sided SDF; positive is on the authored fluid-facing side.

    Unlike a heightfield, the sign comes from the selected sheet's oriented
    normal.  This supports slopes, vertical walls and overhangs.  Nodes whose
    closest point is on the open asset boundary are deliberately invalid: an
    open edge is not silently extruded into a wall or floor.
    """

    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    tolerance = float(boundary_barycentric_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary_barycentric_tolerance must be non-negative")
    loaded = load_open_terrain_mesh(terrain, length_scale=length_scale)
    mesh = loaded.mesh
    trimesh = _load_trimesh()
    sdf_flat = np.empty(spec.cell_count, dtype=np.float32)
    valid_flat = np.empty(spec.cell_count, dtype=np.uint8)
    axes = spec.axes(dtype=np.float64)
    shape = np.asarray(spec.shape, dtype=np.int64)

    for start in range(0, spec.cell_count, chunk_size):
        stop = min(start + chunk_size, spec.cell_count)
        flat = np.arange(start, stop, dtype=np.int64)
        ix = flat // (shape[1] * shape[2])
        remainder = flat - ix * shape[1] * shape[2]
        iy = remainder // shape[2]
        iz = remainder - iy * shape[2]
        points = np.column_stack((axes[0][ix], axes[1][iy], axes[2][iz]))
        closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, points)
        triangle_id = np.asarray(triangle_id, dtype=np.int64)
        finite = (
            np.isfinite(closest).all(axis=1)
            & np.isfinite(distance)
            & (triangle_id >= 0)
            & (triangle_id < len(mesh.faces))
        )
        if not np.all(finite):
            raise RuntimeError("Open terrain closest-point query returned invalid data")
        triangles = np.asarray(mesh.triangles[triangle_id], dtype=np.float64)
        barycentric = _barycentric_coordinates(closest, triangles)
        touches_boundary = np.any(
            (barycentric <= tolerance)
            & loaded.boundary_opposite_vertex[triangle_id],
            axis=1,
        )
        normal = np.asarray(mesh.face_normals[triangle_id], dtype=np.float64)
        normal_side = np.einsum("ij,ij->i", points - closest, normal)
        sign = np.where(normal_side < 0.0, -1.0, 1.0)
        signed_distance = sign * np.asarray(distance, dtype=np.float64)
        signed_distance[np.asarray(distance) <= 1.0e-12] = 0.0
        valid = ~touches_boundary
        sdf_flat[start:stop] = np.where(valid, signed_distance, np.inf).astype(
            np.float32
        )
        valid_flat[start:stop] = valid.astype(np.uint8)

    return (
        sdf_flat.reshape(spec.shape),
        valid_flat.reshape(spec.shape),
        loaded.metadata,
    )


def _face_components(faces):
    """Connected components by shared topological edges, without geometry repair."""

    faces = np.asarray(faces, dtype=np.int64)
    parent = np.arange(len(faces), dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners = {}
    for face_index, face in enumerate(faces):
        a, b, c = (int(value) for value in face)
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            previous = owners.get(key)
            if previous is None:
                owners[key] = face_index
            else:
                union(previous, face_index)
    labels = np.asarray([find(index) for index in range(len(faces))], dtype=np.int64)
    unique = {value: index for index, value in enumerate(sorted(set(labels.tolist())))}
    return np.asarray([unique[value] for value in labels], dtype=np.int32)


def select_open_terrain_faces(
    vertices,
    faces,
    roi_minimum,
    roi_maximum,
    maximum_surface_y,
    orientation_reference=(0.0, 1.0, 0.0),
    expand_supported_components=False,
):
    """Select an audited local top shell using ROI + level + connectivity.

    This intentionally performs no global first-hit ray cast.  The caller must
    provide a fluid ROI and a maximum surface level; the returned source face
    indices make the selection reproducible and reviewable.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    roi_minimum = np.asarray(roi_minimum, dtype=np.float64)
    roi_maximum = np.asarray(roi_maximum, dtype=np.float64)
    orientation_reference = np.asarray(orientation_reference, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must have shape (n, 3) and be finite")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must contain triangles")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces contain an invalid vertex index")
    if (
        roi_minimum.shape != (3,)
        or roi_maximum.shape != (3,)
        or not np.isfinite(roi_minimum).all()
        or not np.isfinite(roi_maximum).all()
        or np.any(roi_maximum <= roi_minimum)
    ):
        raise ValueError("ROI must contain finite increasing three-vectors")
    maximum_surface_y = float(maximum_surface_y)
    if not np.isfinite(maximum_surface_y):
        raise ValueError("maximum_surface_y must be finite")
    reference_length = np.linalg.norm(orientation_reference)
    if not np.isfinite(reference_length) or reference_length <= 1.0e-12:
        raise ValueError("orientation_reference must be non-zero")
    orientation_reference /= reference_length

    triangles = vertices[faces]
    overlaps_roi = np.all(triangles.max(axis=1) >= roi_minimum, axis=1) & np.all(
        triangles.min(axis=1) <= roi_maximum, axis=1
    )
    support_seed_mask = overlaps_roi & (
        triangles[:, :, 1].min(axis=1) < maximum_surface_y
    )
    support_seed_indices = np.flatnonzero(support_seed_mask)
    if not len(support_seed_indices):
        raise ValueError("No terrain faces match the fluid ROI and surface level")
    if expand_supported_components:
        candidate_indices = np.flatnonzero(overlaps_roi)
        candidate_faces = faces[candidate_indices]
        candidate_components = _face_components(candidate_faces)
        seed_lookup = set(int(value) for value in support_seed_indices)
        supported_components = {
            int(component)
            for source_index, component in zip(candidate_indices, candidate_components)
            if int(source_index) in seed_lookup
        }
        source_face_indices = candidate_indices[
            np.asarray(
                [int(component) in supported_components for component in candidate_components],
                dtype=bool,
            )
        ]
    else:
        source_face_indices = support_seed_indices
    selected_faces_source_vertices = faces[source_face_indices].copy()
    component_labels = _face_components(selected_faces_source_vertices)
    flipped_components = []
    for component in sorted(set(component_labels.tolist())):
        mask = component_labels == component
        component_triangles = vertices[selected_faces_source_vertices[mask]]
        area_vectors = np.cross(
            component_triangles[:, 1] - component_triangles[:, 0],
            component_triangles[:, 2] - component_triangles[:, 0],
        )
        orientation_score = float(np.dot(area_vectors.sum(axis=0), orientation_reference))
        if abs(orientation_score) <= 1.0e-12:
            raise ValueError(
                f"Terrain component {component} has ambiguous fluid-facing orientation"
            )
        if orientation_score < 0.0:
            selected_faces_source_vertices[mask, 1:3] = (
                selected_faces_source_vertices[mask, 2:0:-1]
            )
            flipped_components.append(int(component))

    used_vertices = np.unique(selected_faces_source_vertices.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
    selected_vertices = vertices[used_vertices]
    selected_faces = remap[selected_faces_source_vertices]
    edge_counts, _ = _edge_incidence(selected_faces)
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError("Selected terrain contains a non-manifold edge")
    metadata = {
        "source_face_indices": source_face_indices.astype(int).tolist(),
        "support_seed_face_indices": support_seed_indices.astype(int).tolist(),
        "source_face_count": int(len(faces)),
        "selected_face_count": int(len(selected_faces)),
        "selected_vertex_count": int(len(selected_vertices)),
        "connected_component_count": int(len(set(component_labels.tolist()))),
        "flipped_component_indices": flipped_components,
        "bounds_minimum": selected_vertices.min(axis=0).astype(float).tolist(),
        "bounds_maximum": selected_vertices.max(axis=0).astype(float).tolist(),
        "boundary_edge_count": int(sum(count == 1 for count in edge_counts.values())),
        "selector": {
            "method": "roi_level_connected_components",
            "roi_minimum": roi_minimum.astype(float).tolist(),
            "roi_maximum": roi_maximum.astype(float).tolist(),
            "maximum_surface_y": maximum_surface_y,
            "level_rule": "any_vertex_below",
            "component_policy": (
                "expand_components_connected_to_level_seeds"
                if expand_supported_components
                else "level_seeds_only"
            ),
            "orientation_reference": orientation_reference.astype(float).tolist(),
            "global_first_hit_ray_cast": False,
        },
    }
    return selected_vertices, selected_faces, metadata
