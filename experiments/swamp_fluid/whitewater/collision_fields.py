"""Analytic collider SDFs driven by the versioned scene contract."""

from __future__ import annotations

import numpy as np

from .liquid_fields import GridSpec, union_collision_sdf
from .scene_contract import ColliderSpec, SceneContract


def resolve_motion_matrix(motion, snapshot, length_scale=1.0):
    length_scale = float(length_scale)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    if motion.kind == "static":
        value = motion.transform
    else:
        if snapshot is None or motion.source_field not in snapshot:
            raise ValueError(
                f"Snapshot is missing collider transform field {motion.source_field!r}"
            )
        value = snapshot[motion.source_field]
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Resolved collider transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation @ rotation.T, np.eye(3), rtol=0.0, atol=2.0e-5
    ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=2.0e-5):
        raise ValueError("Resolved collider transform must remain rigid and right-handed")
    if motion.matrix_layout == "row_translation":
        if not np.allclose(matrix[:3, 3], 0.0, rtol=0.0, atol=1.0e-8):
            raise ValueError("Resolved row-translation matrix has a non-zero last column")
        translation = matrix[3, :3]
        row_rotation = rotation
    else:
        if not np.allclose(matrix[3, :3], 0.0, rtol=0.0, atol=1.0e-8):
            raise ValueError("Resolved column-translation matrix has a non-zero last row")
        translation = matrix[:3, 3]
        row_rotation = rotation.T
    if not np.isclose(matrix[3, 3], 1.0, rtol=0.0, atol=1.0e-8):
        raise ValueError("Resolved collider transform has an invalid homogeneous component")
    return row_rotation, translation * length_scale


def _local_coordinates(spec: GridSpec, row_rotation, translation):
    x, y, z = spec.axes(dtype=np.float64)
    dx = x[:, None, None] - translation[0]
    dy = y[None, :, None] - translation[1]
    dz = z[None, None, :] - translation[2]
    local = []
    for component in range(3):
        local.append(
            dx * row_rotation[component, 0]
            + dy * row_rotation[component, 1]
            + dz * row_rotation[component, 2]
        )
    return local


def _mesh_collider_sdf(
    spec, collider, row_rotation, translation, length_scale, chunk_size=250_000
):
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("Mesh colliders require the trimesh package") from error

    # Do not let trimesh repair winding or merge topology before provenance
    # gates run. A collision asset must pass as authored; silent repair would
    # make the generated SDF depend on undocumented geometry mutation.
    mesh = trimesh.load(
        collider.parameters["path"], force="mesh", process=False, validate=False
    )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Collider {collider.identifier!r} did not load as a triangle mesh")
    if collider.parameters["require_watertight"] and not mesh.is_watertight:
        raise ValueError(f"Collider {collider.identifier!r} mesh is not watertight")
    if not mesh.is_winding_consistent:
        raise ValueError(f"Collider {collider.identifier!r} mesh winding is inconsistent")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * length_scale
    vertices = vertices @ row_rotation + translation
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
        validate=False,
    )
    if mesh.volume <= 0.0:
        raise ValueError(
            f"Collider {collider.identifier!r} mesh must have outward winding and positive volume"
        )

    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("Mesh SDF chunk_size must be positive")
    result = np.empty(spec.cell_count, dtype=np.float32)
    shape = np.asarray(spec.shape, dtype=np.int64)
    yz = int(shape[1] * shape[2])
    origin = np.asarray(spec.origin, dtype=np.float64)
    for start in range(0, spec.cell_count, chunk_size):
        stop = min(start + chunk_size, spec.cell_count)
        flat = np.arange(start, stop, dtype=np.int64)
        ix = flat // yz
        remainder = flat - ix * yz
        iy = remainder // shape[2]
        iz = remainder - iy * shape[2]
        points = origin + spec.spacing * np.column_stack((ix, iy, iz))
        # trimesh uses positive-inside signed distance; the whitewater field
        # contract requires negative inside every solid.
        result[start:stop] = -np.asarray(
            trimesh.proximity.signed_distance(mesh, points), dtype=np.float32
        )
    result = result.reshape(spec.shape)
    if not np.isfinite(result).all():
        raise RuntimeError(f"Collider {collider.identifier!r} produced a non-finite mesh SDF")
    return result


def collider_sdf(
    spec: GridSpec, collider: ColliderSpec, snapshot=None, length_scale=1.0
):
    """Return the negative-inside analytic SDF for one collider."""

    length_scale = float(length_scale)
    row_rotation, translation = resolve_motion_matrix(
        collider.motion, snapshot, length_scale=length_scale
    )
    if collider.shape == "mesh":
        return _mesh_collider_sdf(
            spec, collider, row_rotation, translation, length_scale
        )
    local = _local_coordinates(spec, row_rotation, translation)
    if collider.shape == "sphere":
        radius = length_scale * float(collider.parameters["radius"])
        sdf = np.sqrt(local[0] ** 2 + local[1] ** 2 + local[2] ** 2) - radius
    elif collider.shape == "box":
        half_extents = length_scale * np.asarray(
            collider.parameters["half_extents"], dtype=np.float64
        )
        q = [np.abs(local[axis]) - half_extents[axis] for axis in range(3)]
        outside = np.sqrt(sum(np.maximum(value, 0.0) ** 2 for value in q))
        inside = np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0.0)
        sdf = outside + inside
    elif collider.shape == "capsule":
        axis = {"x": 0, "y": 1, "z": 2}[collider.parameters["axis"]]
        half_length = length_scale * float(collider.parameters["half_length"])
        radius = length_scale * float(collider.parameters["radius"])
        axial = np.maximum(np.abs(local[axis]) - half_length, 0.0)
        radial_axes = [value for index, value in enumerate(local) if index != axis]
        sdf = np.sqrt(axial**2 + radial_axes[0] ** 2 + radial_axes[1] ** 2) - radius
    else:
        # The local +Y normal points out of the solid half-space.
        sdf = local[1] - length_scale * float(collider.parameters["offset"])
    result = np.asarray(sdf, dtype=np.float32)
    if result.shape != spec.shape or not np.isfinite(result).all():
        raise RuntimeError(f"Collider {collider.identifier!r} produced an invalid SDF")
    return result


def build_collider_fields(
    spec: GridSpec, contract: SceneContract, snapshot=None, include_motion=None
):
    """Build role-separated union fields without conflating walls and churn sources."""

    include_motion = (
        None if include_motion is None else frozenset(str(value) for value in include_motion)
    )
    individual = {}
    role_fields = {"solid": [], "dynamic": [], "churn_source": []}
    selected_colliders = [
        collider
        for collider in contract.colliders
        if include_motion is None or collider.motion.kind in include_motion
    ]
    for collider in selected_colliders:
        field = collider_sdf(
            spec,
            collider,
            snapshot=snapshot,
            length_scale=contract.metres_per_unit,
        )
        individual[collider.identifier] = field
        for role in collider.roles:
            role_fields[role].append(field)

    outputs = {"individual": individual}
    if role_fields["solid"]:
        outputs["collider_collision_sdf"] = union_collision_sdf(*role_fields["solid"])
    if role_fields["dynamic"]:
        outputs["dynamic_collision_sdf"] = union_collision_sdf(*role_fields["dynamic"])
    if role_fields["churn_source"]:
        outputs["churn_source_sdf"] = union_collision_sdf(*role_fields["churn_source"])
    outputs["metadata"] = {
        "colliders": [
            {
                "id": collider.identifier,
                "shape": collider.shape,
                "roles": list(collider.roles),
                "motion": collider.motion.kind,
            }
            for collider in selected_colliders
        ],
        "field_roles": {
            role: [
                collider.identifier
                for collider in selected_colliders
                if role in collider.roles
            ]
            for role in role_fields
        },
    }
    return outputs
