"""Versioned, scene-independent input contract for the whitewater pipeline."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCENE_CONTRACT_SCHEMA = 2
SUPPORTED_SCENE_CONTRACT_SCHEMAS = frozenset({1, 2})
SCENE_CONTRACT_PRODUCT = "whitewater_scene_contract"
SUPPORTED_COLLIDER_SHAPES = frozenset(
    {"sphere", "box", "capsule", "plane", "mesh"}
)
SUPPORTED_COLLIDER_ROLES = frozenset({"solid", "dynamic", "churn_source"})
SUPPORTED_MOTION_TYPES = frozenset({"static", "source_matrix"})
SUPPORTED_MATRIX_LAYOUTS = frozenset({"row_translation", "column_translation"})
SUPPORTED_TERRAIN_REPRESENTATIONS = frozenset({"open_triangle_mesh"})
TERRAIN_SELECTION_PRODUCT = "whitewater_open_terrain_selection"
TERRAIN_SELECTION_SCHEMA = 1


def _finite_vector(value, length, label):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return tuple(float(item) for item in vector)


def _positive(value, label, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rigid_matrix(value, layout, label):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if layout == "row_translation":
        affine_column = matrix[:3, 3]
        homogeneous = matrix[3, 3]
    else:
        affine_column = matrix[3, :3]
        homogeneous = matrix[3, 3]
    if not np.allclose(affine_column, 0.0, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{label} has values outside the selected matrix layout")
    if not np.isclose(homogeneous, 1.0, rtol=0.0, atol=1.0e-9):
        raise ValueError(f"{label} has an invalid homogeneous component")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation @ rotation.T, np.eye(3), rtol=0.0, atol=2.0e-6
    ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError(f"{label} must contain a rigid, right-handed rotation")
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class MotionSpec:
    kind: str
    matrix_layout: str
    transform: tuple[tuple[float, ...], ...] | None = None
    source_field: str | None = None

    @classmethod
    def from_mapping(cls, payload, label):
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be an object")
        allowed = {"kind", "matrix_layout", "transform", "source_field"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"{label} has unknown keys: {sorted(unknown)}")
        kind = str(payload.get("kind", "")).strip()
        layout = str(payload.get("matrix_layout", "row_translation")).strip()
        if kind not in SUPPORTED_MOTION_TYPES:
            raise ValueError(f"{label}.kind is unsupported: {kind!r}")
        if layout not in SUPPORTED_MATRIX_LAYOUTS:
            raise ValueError(f"{label}.matrix_layout is unsupported: {layout!r}")
        if kind == "static":
            if payload.get("source_field") is not None:
                raise ValueError(f"{label} static motion cannot define source_field")
            transform = payload.get("transform", np.eye(4).tolist())
            return cls(
                kind=kind,
                matrix_layout=layout,
                transform=_rigid_matrix(transform, layout, f"{label}.transform"),
            )
        source_field = payload.get("source_field")
        if not isinstance(source_field, str) or not source_field.strip():
            raise ValueError(f"{label}.source_field must be a non-empty string")
        if payload.get("transform") is not None:
            raise ValueError(f"{label} source_matrix motion cannot define transform")
        return cls(kind=kind, matrix_layout=layout, source_field=source_field.strip())

    def metadata(self):
        payload = {"kind": self.kind, "matrix_layout": self.matrix_layout}
        if self.transform is not None:
            payload["transform"] = [list(row) for row in self.transform]
        if self.source_field is not None:
            payload["source_field"] = self.source_field
        return payload


@dataclass(frozen=True)
class ColliderSpec:
    identifier: str
    shape: str
    roles: tuple[str, ...]
    parameters: dict
    motion: MotionSpec

    @classmethod
    def from_mapping(cls, payload, index):
        label = f"colliders[{index}]"
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be an object")
        allowed = {"id", "shape", "roles", "parameters", "motion"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"{label} has unknown keys: {sorted(unknown)}")
        identifier = payload.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{label}.id must be a non-empty string")
        identifier = identifier.strip()
        shape = str(payload.get("shape", "")).strip()
        if shape not in SUPPORTED_COLLIDER_SHAPES:
            raise ValueError(f"{label}.shape is unsupported: {shape!r}")
        roles_value = payload.get("roles", ["solid"])
        if not isinstance(roles_value, list) or not roles_value:
            raise ValueError(f"{label}.roles must be a non-empty list")
        roles = tuple(str(role).strip() for role in roles_value)
        if len(set(roles)) != len(roles) or set(roles) - SUPPORTED_COLLIDER_ROLES:
            raise ValueError(f"{label}.roles contains duplicates or unsupported roles")
        if "solid" not in roles:
            raise ValueError(f"{label} must include the solid role")
        motion = MotionSpec.from_mapping(payload.get("motion", {}), f"{label}.motion")
        if motion.kind != "static" and "dynamic" not in roles:
            raise ValueError(f"{label} source-driven motion must include the dynamic role")
        if motion.kind == "static" and "dynamic" in roles:
            raise ValueError(f"{label} static motion cannot include the dynamic role")
        parameters = cls._validated_parameters(shape, payload.get("parameters", {}), label)
        return cls(identifier, shape, roles, parameters, motion)

    @staticmethod
    def _validated_parameters(shape, payload, label):
        if not isinstance(payload, dict):
            raise ValueError(f"{label}.parameters must be an object")
        if shape == "sphere":
            expected = {"radius"}
            if set(payload) != expected:
                raise ValueError(f"{label}.parameters must contain exactly radius")
            return {"radius": _positive(payload["radius"], f"{label}.parameters.radius")}
        if shape == "box":
            expected = {"half_extents"}
            if set(payload) != expected:
                raise ValueError(f"{label}.parameters must contain exactly half_extents")
            half_extents = _finite_vector(
                payload["half_extents"], 3, f"{label}.parameters.half_extents"
            )
            if min(half_extents) <= 0.0:
                raise ValueError(f"{label}.parameters.half_extents must be positive")
            return {"half_extents": half_extents}
        if shape == "capsule":
            expected = {"radius", "half_length", "axis"}
            if set(payload) != expected:
                raise ValueError(
                    f"{label}.parameters must contain radius, half_length and axis"
                )
            axis = str(payload["axis"]).strip().lower()
            if axis not in {"x", "y", "z"}:
                raise ValueError(f"{label}.parameters.axis must be x, y or z")
            return {
                "radius": _positive(payload["radius"], f"{label}.parameters.radius"),
                "half_length": _positive(
                    payload["half_length"],
                    f"{label}.parameters.half_length",
                    allow_zero=True,
                ),
                "axis": axis,
            }
        if shape == "mesh":
            expected = {"path", "sha256", "require_watertight"}
            if set(payload) != expected:
                raise ValueError(
                    f"{label}.parameters must contain path, sha256 and require_watertight"
                )
            path = Path(payload["path"]).resolve()
            expected_hash = payload["sha256"]
            if not path.is_file():
                raise ValueError(f"{label}.parameters.path does not exist: {path}")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"{label}.parameters.sha256 must be a SHA-256 hex digest")
            try:
                int(expected_hash, 16)
            except ValueError as error:
                raise ValueError(
                    f"{label}.parameters.sha256 must be a SHA-256 hex digest"
                ) from error
            actual_hash = _sha256_file(path)
            if actual_hash.lower() != expected_hash.lower():
                raise ValueError(f"{label}.parameters mesh hash mismatch: {path}")
            if not isinstance(payload["require_watertight"], bool):
                raise ValueError(f"{label}.parameters.require_watertight must be boolean")
            if payload["require_watertight"] is not True:
                raise ValueError(
                    f"{label} solid mesh colliders must require a watertight surface"
                )
            return {
                "path": str(path),
                "sha256": actual_hash,
                "require_watertight": payload["require_watertight"],
            }
        expected = {"offset"}
        if set(payload) != expected:
            raise ValueError(f"{label}.parameters must contain exactly offset")
        return {
            "offset": float(
                _finite_vector([payload["offset"]], 1, f"{label}.parameters.offset")[0]
            )
        }

    def metadata(self):
        parameters = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.parameters.items()
        }
        return {
            "id": self.identifier,
            "shape": self.shape,
            "roles": list(self.roles),
            "parameters": parameters,
            "motion": self.motion.metadata(),
        }


def _validated_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error
    return value.lower()


@dataclass(frozen=True)
class TerrainSpec:
    """An explicitly selected, fluid-facing open terrain surface."""

    identifier: str
    representation: str
    parameters: dict
    motion: MotionSpec

    @classmethod
    def from_mapping(cls, payload):
        label = "terrain"
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be an object")
        allowed = {"id", "representation", "parameters", "motion"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"{label} has unknown keys: {sorted(unknown)}")
        identifier = payload.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{label}.id must be a non-empty string")
        representation = str(payload.get("representation", "")).strip()
        if representation not in SUPPORTED_TERRAIN_REPRESENTATIONS:
            raise ValueError(
                f"{label}.representation is unsupported: {representation!r}"
            )
        motion = MotionSpec.from_mapping(payload.get("motion", {}), f"{label}.motion")
        if motion.kind != "static":
            raise ValueError("Open terrain must currently use static motion")
        parameters = cls._validated_parameters(payload.get("parameters", {}), label)
        return cls(identifier.strip(), representation, parameters, motion)

    @staticmethod
    def _validated_parameters(payload, label):
        if not isinstance(payload, dict):
            raise ValueError(f"{label}.parameters must be an object")
        expected = {
            "path",
            "sha256",
            "selection_path",
            "selection_sha256",
            "normal_convention",
            "require_open",
        }
        if set(payload) != expected:
            raise ValueError(
                f"{label}.parameters must contain exactly {sorted(expected)}"
            )
        path = Path(payload["path"]).resolve()
        selection_path = Path(payload["selection_path"]).resolve()
        if not path.is_file():
            raise ValueError(f"{label}.parameters.path does not exist: {path}")
        if not selection_path.is_file():
            raise ValueError(
                f"{label}.parameters.selection_path does not exist: {selection_path}"
            )
        expected_hash = _validated_sha256(
            payload["sha256"], f"{label}.parameters.sha256"
        )
        expected_selection_hash = _validated_sha256(
            payload["selection_sha256"], f"{label}.parameters.selection_sha256"
        )
        actual_hash = _sha256_file(path)
        actual_selection_hash = _sha256_file(selection_path)
        if actual_hash != expected_hash:
            raise ValueError(f"{label}.parameters terrain mesh hash mismatch: {path}")
        if actual_selection_hash != expected_selection_hash:
            raise ValueError(
                f"{label}.parameters terrain selection hash mismatch: {selection_path}"
            )
        if payload["normal_convention"] != "toward_fluid":
            raise ValueError(
                f"{label}.parameters.normal_convention must be 'toward_fluid'"
            )
        if payload["require_open"] is not True:
            raise ValueError(f"{label}.parameters.require_open must be true")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not isinstance(selection, dict):
            raise ValueError("Terrain selection record must be an object")
        if (
            selection.get("schema") != TERRAIN_SELECTION_SCHEMA
            or selection.get("product") != TERRAIN_SELECTION_PRODUCT
        ):
            raise ValueError("Terrain selection record has an unsupported contract")
        selected_mesh = selection.get("selected_mesh")
        if not isinstance(selected_mesh, dict):
            raise ValueError("Terrain selection record is missing selected_mesh")
        if selected_mesh.get("sha256", "").lower() != actual_hash:
            raise ValueError("Terrain selection record does not own the selected mesh")
        if int(selected_mesh.get("triangle_count", 0)) <= 0:
            raise ValueError("Terrain selection record has no selected triangles")
        if selection.get("normal_convention") != "toward_fluid":
            raise ValueError("Terrain selection record has the wrong normal convention")
        return {
            "path": str(path),
            "sha256": actual_hash,
            "selection_path": str(selection_path),
            "selection_sha256": actual_selection_hash,
            "normal_convention": "toward_fluid",
            "require_open": True,
            "selection": selection,
        }

    def metadata(self):
        parameters = dict(self.parameters)
        parameters.pop("selection", None)
        return {
            "id": self.identifier,
            "representation": self.representation,
            "parameters": parameters,
            "motion": self.motion.metadata(),
        }


@dataclass(frozen=True)
class SceneContract:
    schema: int
    name: str
    metres_per_unit: float
    gravity: tuple[float, float, float]
    colliders: tuple[ColliderSpec, ...]
    terrain: TerrainSpec | None
    metadata_payload: dict

    @classmethod
    def from_mapping(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("Scene contract must be an object")
        allowed = {
            "schema",
            "product",
            "name",
            "coordinate_system",
            "physics",
            "colliders",
            "terrain",
            "metadata",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Scene contract has unknown keys: {sorted(unknown)}")
        schema = payload.get("schema")
        if schema not in SUPPORTED_SCENE_CONTRACT_SCHEMAS:
            raise ValueError(f"Unsupported scene contract schema: {payload.get('schema')!r}")
        if payload.get("product") != SCENE_CONTRACT_PRODUCT:
            raise ValueError(f"Unexpected scene contract product: {payload.get('product')!r}")
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Scene contract name must be a non-empty string")
        coordinate = payload.get("coordinate_system")
        if not isinstance(coordinate, dict) or set(coordinate) != {
            "axes",
            "handedness",
            "metres_per_unit",
        }:
            raise ValueError("coordinate_system must define axes, handedness and metres_per_unit")
        if coordinate["axes"] != "xyz" or coordinate["handedness"] != "right":
            raise ValueError("Only right-handed xyz scene contracts are supported")
        metres_per_unit = _positive(
            coordinate["metres_per_unit"], "coordinate_system.metres_per_unit"
        )
        physics = payload.get("physics")
        if not isinstance(physics, dict) or set(physics) != {"gravity"}:
            raise ValueError("physics must contain exactly gravity")
        gravity = _finite_vector(physics["gravity"], 3, "physics.gravity")
        if np.linalg.norm(gravity) <= 1.0e-9:
            raise ValueError("physics.gravity must be non-zero")
        collider_payloads = payload.get("colliders", [])
        if not isinstance(collider_payloads, list):
            raise ValueError("colliders must be a list")
        colliders = tuple(
            ColliderSpec.from_mapping(value, index)
            for index, value in enumerate(collider_payloads)
        )
        identifiers = [collider.identifier for collider in colliders]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Collider IDs must be unique")
        terrain_payload = payload.get("terrain")
        if schema == 1 and terrain_payload is not None:
            raise ValueError("Schema 1 scene contracts cannot define terrain")
        if schema == 2 and terrain_payload is None:
            raise ValueError("Schema 2 scene contracts must define terrain")
        terrain = (
            TerrainSpec.from_mapping(terrain_payload)
            if terrain_payload is not None
            else None
        )
        if terrain is not None and terrain.identifier in identifiers:
            raise ValueError("Terrain and collider IDs must be unique")
        metadata_payload = payload.get("metadata", {})
        if not isinstance(metadata_payload, dict):
            raise ValueError("metadata must be an object")
        return cls(
            schema=int(schema),
            name=name.strip(),
            metres_per_unit=metres_per_unit,
            gravity=gravity,
            colliders=colliders,
            terrain=terrain,
            metadata_payload=dict(metadata_payload),
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for collider in payload.get("colliders", []):
            if collider.get("shape") != "mesh":
                continue
            parameters = collider.get("parameters", {})
            mesh_path = Path(parameters.get("path", ""))
            if not mesh_path.is_absolute():
                parameters["path"] = str((path.parent / mesh_path).resolve())
        terrain = payload.get("terrain")
        if isinstance(terrain, dict):
            parameters = terrain.get("parameters", {})
            for key in ("path", "selection_path"):
                terrain_path = Path(parameters.get(key, ""))
                if not terrain_path.is_absolute():
                    parameters[key] = str((path.parent / terrain_path).resolve())
        return cls.from_mapping(payload)

    def metadata(self):
        payload = {
            "schema": self.schema,
            "product": SCENE_CONTRACT_PRODUCT,
            "name": self.name,
            "coordinate_system": {
                "axes": "xyz",
                "handedness": "right",
                "metres_per_unit": self.metres_per_unit,
            },
            "physics": {"gravity": list(self.gravity)},
            "colliders": [collider.metadata() for collider in self.colliders],
            "metadata": self.metadata_payload,
        }
        if self.terrain is not None:
            payload["terrain"] = self.terrain.metadata()
        return payload


def legacy_sphere_impact_contract(run_report):
    """Create the v1 contract represented by the historical PhysX run report."""

    radius = float(run_report["impactor"]["radius"])
    gravity = run_report.get("gravity", [0.0, -9.81, 0.0])
    if isinstance(gravity, (int, float)):
        gravity = [0.0, -abs(float(gravity)), 0.0]
    payload = {
        "schema": 1,
        "product": SCENE_CONTRACT_PRODUCT,
        "name": "legacy_sphere_impact",
        "coordinate_system": {
            "axes": "xyz",
            "handedness": "right",
            "metres_per_unit": 1.0,
        },
        "physics": {"gravity": gravity},
        "colliders": [
            {
                "id": "legacy_impactor",
                "shape": "sphere",
                "roles": ["solid", "dynamic", "churn_source"],
                "parameters": {"radius": radius},
                "motion": {
                    "kind": "source_matrix",
                    "source_field": "sphere_transform",
                    "matrix_layout": "row_translation",
                },
            }
        ],
        "metadata": {"adapter": "legacy_physx_sphere_run_report"},
    }
    return SceneContract.from_mapping(payload)
