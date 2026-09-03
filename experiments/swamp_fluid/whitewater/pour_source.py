"""Versioned, deterministic primary-liquid pour source contract.

This module deliberately contains no Isaac Sim imports.  It turns an artist
authored outlet, flow-rate curve and rigid motion into append-only particle
batches that a PhysX adapter can pre-author and enable at their birth steps.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


POUR_SOURCE_SCHEMA = 2
SUPPORTED_POUR_SOURCE_SCHEMAS = (1, 2)
POUR_SOURCE_PRODUCT = "physx_primary_liquid_pour_source"


def _canonical_json_value(value):
    """Return a stable JSON value after rigid-vector normalization noise."""
    if isinstance(value, dict):
        return {key: _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _strict_keys(payload, expected, label):
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    keys = set(payload)
    expected = set(expected)
    if keys != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - keys)} "
            f"unknown={sorted(keys - expected)}"
        )


def _finite_vector(value, length, label):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return vector


def _positive(value, label, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return value


def _unit(value, label):
    vector = _finite_vector(value, 3, label)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-9:
        raise ValueError(f"{label} must be non-zero")
    return vector / length


def _pose_basis(normal, up_hint, label):
    normal = _unit(normal, f"{label}.normal")
    up_hint = _unit(up_hint, f"{label}.up_hint")
    tangent_u = up_hint - normal * float(np.dot(up_hint, normal))
    tangent_length = float(np.linalg.norm(tangent_u))
    if tangent_length <= 1.0e-5:
        raise ValueError(f"{label}.up_hint must not be parallel to normal")
    tangent_u /= tangent_length
    tangent_v = np.cross(normal, tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    return normal, tangent_u, tangent_v


@dataclass(frozen=True)
class SourcePose:
    centre: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray


@dataclass(frozen=True)
class OutletSpec:
    radii_m: tuple[float, float]
    pattern: str
    stable_warp_fraction: float


@dataclass(frozen=True)
class FlowSpec:
    rate_m3_s: float
    speed_m_s: float
    edge_speed_fraction: float
    profile_exponent: float
    tangent_perturbation_m_s: float
    seed: int


@dataclass(frozen=True)
class EmissionSpec:
    kind: str
    phase_count: int


@dataclass(frozen=True)
class MotionKeyframe:
    time_seconds: float
    centre: np.ndarray
    normal: np.ndarray
    up_hint: np.ndarray


@dataclass(frozen=True)
class ParticleBatch:
    batch_index: int
    birth_step: int
    birth_time_seconds: float
    positions: np.ndarray
    velocities: np.ndarray
    local_coordinates: np.ndarray


@dataclass(frozen=True)
class PourSource:
    schema_version: int
    name: str
    metres_per_unit: float
    outlet: OutletSpec
    flow: FlowSpec
    emission: EmissionSpec
    start_seconds: float
    stop_seconds: float
    keyframes: tuple[MotionKeyframe, ...]

    @classmethod
    def from_mapping(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("pour source must be an object")
        schema_version = payload.get("schema")
        if schema_version not in SUPPORTED_POUR_SOURCE_SCHEMAS:
            raise ValueError(f"Unsupported pour source schema: {schema_version!r}")
        expected_root_keys = [
            "schema",
            "product",
            "name",
            "coordinate_system",
            "outlet",
            "flow",
            "timing",
            "motion",
            "metadata",
        ]
        if schema_version >= 2:
            expected_root_keys.append("emission")
        _strict_keys(
            payload,
            expected_root_keys,
            "pour source",
        )
        if payload["product"] != POUR_SOURCE_PRODUCT:
            raise ValueError(f"Unexpected pour source product: {payload['product']!r}")
        name = payload["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("pour source name must be a non-empty string")

        coordinate = payload["coordinate_system"]
        _strict_keys(
            coordinate,
            ("axes", "handedness", "up_axis", "metres_per_unit"),
            "coordinate_system",
        )
        if (
            coordinate["axes"] != "xyz"
            or coordinate["handedness"] != "right"
            or coordinate["up_axis"] != "y"
        ):
            raise ValueError("Only right-handed xyz, Y-up source contracts are supported")
        metres_per_unit = _positive(
            coordinate["metres_per_unit"], "coordinate_system.metres_per_unit"
        )

        outlet_payload = payload["outlet"]
        _strict_keys(
            outlet_payload,
            ("shape", "radii_m", "pattern", "stable_warp_fraction"),
            "outlet",
        )
        if outlet_payload["shape"] != "ellipse":
            raise ValueError("Only ellipse pour outlets are supported in schema 1")
        radii = _finite_vector(outlet_payload["radii_m"], 2, "outlet.radii_m")
        if np.min(radii) <= 0.0:
            raise ValueError("outlet.radii_m must be positive")
        if outlet_payload["pattern"] != "staggered_hex":
            raise ValueError("outlet.pattern must be staggered_hex")
        stable_warp = _positive(
            outlet_payload["stable_warp_fraction"],
            "outlet.stable_warp_fraction",
            allow_zero=True,
        )
        if stable_warp > 0.25:
            raise ValueError("outlet.stable_warp_fraction must not exceed 0.25")
        outlet = OutletSpec(tuple(map(float, radii)), "staggered_hex", stable_warp)

        flow_payload = payload["flow"]
        _strict_keys(
            flow_payload,
            (
                "rate_m3_s",
                "speed_m_s",
                "velocity_profile",
                "tangent_perturbation_m_s",
                "seed",
            ),
            "flow",
        )
        profile = flow_payload["velocity_profile"]
        _strict_keys(profile, ("kind", "edge_fraction", "exponent"), "flow.velocity_profile")
        if profile["kind"] != "power_law":
            raise ValueError("flow.velocity_profile.kind must be power_law")
        edge_fraction = _positive(
            profile["edge_fraction"], "flow.velocity_profile.edge_fraction"
        )
        if edge_fraction > 1.0:
            raise ValueError("flow.velocity_profile.edge_fraction must not exceed 1")
        seed = flow_payload["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("flow.seed must be a non-negative integer")
        flow = FlowSpec(
            rate_m3_s=_positive(flow_payload["rate_m3_s"], "flow.rate_m3_s"),
            speed_m_s=_positive(flow_payload["speed_m_s"], "flow.speed_m_s"),
            edge_speed_fraction=edge_fraction,
            profile_exponent=_positive(profile["exponent"], "flow.velocity_profile.exponent"),
            tangent_perturbation_m_s=_positive(
                flow_payload["tangent_perturbation_m_s"],
                "flow.tangent_perturbation_m_s",
                allow_zero=True,
            ),
            seed=seed,
        )
        if flow.tangent_perturbation_m_s > 0.25 * flow.speed_m_s:
            raise ValueError("Tangent perturbation must not exceed 25% of source speed")

        if schema_version == 1:
            emission = EmissionSpec("complete_cross_section_layers", 1)
        else:
            emission_payload = payload["emission"]
            _strict_keys(emission_payload, ("kind", "phase_count"), "emission")
            if emission_payload["kind"] != "phased_cross_section":
                raise ValueError("emission.kind must be phased_cross_section in schema 2")
            phase_count = emission_payload["phase_count"]
            if (
                not isinstance(phase_count, int)
                or isinstance(phase_count, bool)
                or not 2 <= phase_count <= 16
            ):
                raise ValueError("emission.phase_count must be an integer from 2 through 16")
            emission = EmissionSpec("phased_cross_section", phase_count)

        timing = payload["timing"]
        _strict_keys(timing, ("start_seconds", "stop_seconds"), "timing")
        start = _positive(timing["start_seconds"], "timing.start_seconds", allow_zero=True)
        stop = _positive(timing["stop_seconds"], "timing.stop_seconds")
        if stop <= start:
            raise ValueError("timing.stop_seconds must be later than start_seconds")

        motion = payload["motion"]
        _strict_keys(motion, ("kind", "interpolation", "keyframes"), "motion")
        if motion["kind"] != "keyframed_rigid":
            raise ValueError("motion.kind must be keyframed_rigid")
        if motion["interpolation"] != "linear_orthonormalized":
            raise ValueError("motion.interpolation must be linear_orthonormalized")
        keyframe_payloads = motion["keyframes"]
        if not isinstance(keyframe_payloads, list) or not keyframe_payloads:
            raise ValueError("motion.keyframes must be a non-empty list")
        keyframes = []
        for index, keyframe in enumerate(keyframe_payloads):
            label = f"motion.keyframes[{index}]"
            _strict_keys(keyframe, ("time_seconds", "centre", "normal", "up_hint"), label)
            keyframe_time = _positive(
                keyframe["time_seconds"], f"{label}.time_seconds", allow_zero=True
            )
            centre = _finite_vector(keyframe["centre"], 3, f"{label}.centre")
            normal, tangent_u, _ = _pose_basis(
                keyframe["normal"], keyframe["up_hint"], label
            )
            keyframes.append(
                MotionKeyframe(keyframe_time, centre, normal, tangent_u)
            )
        times = [item.time_seconds for item in keyframes]
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError("motion keyframe times must be strictly increasing")
        if times[0] > start or times[-1] < stop:
            raise ValueError("motion keyframes must cover the complete pour timing interval")
        if not isinstance(payload["metadata"], dict):
            raise ValueError("metadata must be an object")
        return cls(
            schema_version=schema_version,
            name=name.strip(),
            metres_per_unit=metres_per_unit,
            outlet=outlet,
            flow=flow,
            emission=emission,
            start_seconds=start,
            stop_seconds=stop,
            keyframes=tuple(keyframes),
        )

    @classmethod
    def load(cls, path):
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def pose_at(self, time_seconds):
        time_seconds = float(time_seconds)
        if not np.isfinite(time_seconds):
            raise ValueError("pose time must be finite")
        if time_seconds <= self.keyframes[0].time_seconds:
            first = self.keyframes[0]
            normal, tangent_u, tangent_v = _pose_basis(
                first.normal, first.up_hint, "motion pose"
            )
            return SourcePose(first.centre.copy(), normal, tangent_u, tangent_v)
        if time_seconds >= self.keyframes[-1].time_seconds:
            last = self.keyframes[-1]
            normal, tangent_u, tangent_v = _pose_basis(
                last.normal, last.up_hint, "motion pose"
            )
            return SourcePose(last.centre.copy(), normal, tangent_u, tangent_v)
        for first, second in zip(self.keyframes[:-1], self.keyframes[1:]):
            if first.time_seconds <= time_seconds <= second.time_seconds:
                fraction = (time_seconds - first.time_seconds) / (
                    second.time_seconds - first.time_seconds
                )
                centre = first.centre + fraction * (second.centre - first.centre)
                normal = first.normal + fraction * (second.normal - first.normal)
                up_hint = first.up_hint + fraction * (second.up_hint - first.up_hint)
                normal, tangent_u, tangent_v = _pose_basis(
                    normal, up_hint, "interpolated motion pose"
                )
                return SourcePose(centre, normal, tangent_u, tangent_v)
        raise RuntimeError("Failed to bracket source motion time")

    def _base_local_coordinates(self, spacing):
        spacing = _positive(spacing, "particle spacing")
        radius_u, radius_v = self.outlet.radii_m
        particle_radius = 0.5 * spacing
        usable_u = radius_u - particle_radius
        usable_v = radius_v - particle_radius
        if min(usable_u, usable_v) <= 0.0:
            raise ValueError("Pour outlet is too small for the configured particle spacing")
        row_spacing = math.sqrt(3.0) * 0.5 * spacing
        rows = range(
            int(math.floor(-usable_u / row_spacing)),
            int(math.ceil(usable_u / row_spacing)) + 1,
        )
        points = []
        for row in rows:
            u = row * row_spacing
            phase = 0.5 * spacing if row & 1 else 0.0
            column_min = int(math.floor((-usable_v - phase) / spacing))
            column_max = int(math.ceil((usable_v - phase) / spacing))
            for column in range(column_min, column_max + 1):
                v = column * spacing + phase
                normalized_radius = (u / usable_u) ** 2 + (v / usable_v) ** 2
                if normalized_radius <= 1.0 + 1.0e-12:
                    points.append((u, v))
        if len(points) < 3:
            raise ValueError("Pour outlet produces fewer than three particles per batch")
        return np.asarray(points, dtype=np.float64)

    def local_coordinates(self, spacing, batch_index):
        if not isinstance(batch_index, int) or batch_index < 0:
            raise ValueError("batch_index must be a non-negative integer")
        coordinates = self._base_local_coordinates(spacing).copy()
        layer_index = batch_index // self.emission.phase_count
        if self.outlet.stable_warp_fraction != 0.0:
            usable = np.asarray(self.outlet.radii_m, dtype=np.float64) - 0.5 * spacing
            normalized = coordinates / usable
            radius_squared = np.sum(normalized * normalized, axis=1)
            interior_weight = np.maximum(0.0, 1.0 - radius_squared)
            identifiers = np.arange(len(coordinates), dtype=np.float64)
            seed_phase = (self.flow.seed % 104729) * 0.0000601
            temporal_phase = layer_index * 0.6180339887498949
            amplitude = self.outlet.stable_warp_fraction * spacing * interior_weight
            coordinates[:, 0] += amplitude * np.sin(
                2.399963229728653 * identifiers + temporal_phase + seed_phase
            )
            coordinates[:, 1] += amplitude * np.cos(
                1.618033988749895 * identifiers + temporal_phase + seed_phase
            )
            warped_radius_squared = np.sum((coordinates / usable) ** 2, axis=1)
            if np.max(warped_radius_squared) > 1.0 + 1.0e-9:
                raise RuntimeError("Stable source warp moved a particle outside the outlet")
        if self.emission.phase_count == 1:
            return coordinates
        phase_index = batch_index % self.emission.phase_count
        identifiers = np.arange(len(coordinates), dtype=np.int64)
        return coordinates[identifiers % self.emission.phase_count == phase_index]

    def full_layer_particle_count(self, spacing):
        return int(len(self._base_local_coordinates(spacing)))

    def batch_particle_count(self, spacing, batch_index=0):
        return int(len(self.local_coordinates(spacing, batch_index)))

    def batch_volume_m3(self, spacing, batch_index=0):
        return self.batch_particle_count(spacing, batch_index) * float(spacing) ** 3

    def birth_steps(self, physics_fps, end_seconds=None):
        if not isinstance(physics_fps, int) or physics_fps <= 0:
            raise ValueError("physics_fps must be a positive integer")
        end = self.stop_seconds if end_seconds is None else min(float(end_seconds), self.stop_seconds)
        if end <= self.start_seconds:
            return tuple()
        # This schedule is populated later by schedule_for_spacing because the
        # discrete particle volume is part of the mass-flow contract.
        raise RuntimeError("Use schedule_for_spacing so flow rate includes particle volume")

    def schedule_for_spacing(self, physics_fps, spacing, end_seconds=None):
        if not isinstance(physics_fps, int) or physics_fps <= 0:
            raise ValueError("physics_fps must be a positive integer")
        end = self.stop_seconds if end_seconds is None else min(float(end_seconds), self.stop_seconds)
        if end <= self.start_seconds:
            return tuple()
        first_step = int(math.ceil(self.start_seconds * physics_fps - 1.0e-12))
        last_step = int(math.floor(end * physics_fps + 1.0e-12))
        births = []
        if self.emission.phase_count == 1:
            batch_volume = self.batch_volume_m3(spacing)
            emitted_batches = 0
            for step in range(first_step, last_step + 1):
                elapsed = max(0.0, step / physics_fps - self.start_seconds)
                expected_batches = int(math.floor(
                    elapsed * self.flow.rate_m3_s / batch_volume + 1.0e-12
                ))
                while emitted_batches < expected_batches:
                    births.append(step)
                    emitted_batches += 1
            return tuple(births)

        # Schema 2 emits spatially interleaved subsets of a cross-section.
        # The subset sizes can differ by one particle, so schedule against the
        # cumulative emitted volume rather than pretending every batch is the
        # same size.  This preserves the mass-flow contract at every step.
        emitted_volume = 0.0
        batch_index = 0
        for step in range(first_step, last_step + 1):
            elapsed = max(0.0, step / physics_fps - self.start_seconds)
            target_volume = elapsed * self.flow.rate_m3_s
            while True:
                next_volume = self.batch_volume_m3(spacing, batch_index)
                if emitted_volume + next_volume > target_volume + 1.0e-12:
                    break
                births.append(step)
                emitted_volume += next_volume
                batch_index += 1
        return tuple(births)

    def particle_batch(self, batch_index, birth_step, physics_fps, spacing):
        if not isinstance(birth_step, int) or birth_step < 0:
            raise ValueError("birth_step must be a non-negative integer")
        birth_time = birth_step / float(physics_fps)
        pose = self.pose_at(birth_time)
        local = self.local_coordinates(spacing, batch_index)
        positions = (
            pose.centre[None, :]
            + local[:, 0:1] * pose.tangent_u[None, :]
            + local[:, 1:2] * pose.tangent_v[None, :]
        )
        usable = np.asarray(self.outlet.radii_m, dtype=np.float64) - 0.5 * spacing
        normalized_radius_squared = np.sum((local / usable) ** 2, axis=1)
        profile = self.flow.edge_speed_fraction + (
            1.0 - self.flow.edge_speed_fraction
        ) * np.maximum(0.0, 1.0 - normalized_radius_squared) ** self.flow.profile_exponent
        axial_speed = self.flow.speed_m_s * profile
        identifiers = np.arange(len(local), dtype=np.float64)
        phase = (
            identifiers * 2.399963229728653
            + batch_index * 0.3819660112501051
            + (self.flow.seed % 65537) * 0.000137
        )
        tangent_scale = (
            self.flow.tangent_perturbation_m_s
            * np.maximum(0.0, 1.0 - normalized_radius_squared)
        )
        perturb_u = tangent_scale * np.sin(phase)
        perturb_v = tangent_scale * np.cos(phase)
        # Remove batch-wide drift; perturbation changes the internal velocity
        # profile but cannot steer the complete pour away from its authored aim.
        perturb_u -= np.mean(perturb_u)
        perturb_v -= np.mean(perturb_v)
        velocities = (
            axial_speed[:, None] * pose.normal[None, :]
            + perturb_u[:, None] * pose.tangent_u[None, :]
            + perturb_v[:, None] * pose.tangent_v[None, :]
        )
        return ParticleBatch(
            batch_index=batch_index,
            birth_step=birth_step,
            birth_time_seconds=birth_time,
            positions=np.ascontiguousarray(positions, dtype=np.float32),
            velocities=np.ascontiguousarray(velocities, dtype=np.float32),
            local_coordinates=np.ascontiguousarray(local, dtype=np.float32),
        )

    def kinematic_flow_audit(self, spacing):
        local = np.concatenate(
            [
                self.local_coordinates(spacing, batch_index)
                for batch_index in range(self.emission.phase_count)
            ],
            axis=0,
        )
        usable = np.asarray(self.outlet.radii_m, dtype=np.float64) - 0.5 * spacing
        normalized_radius_squared = np.sum((local / usable) ** 2, axis=1)
        profile = self.flow.edge_speed_fraction + (
            1.0 - self.flow.edge_speed_fraction
        ) * np.maximum(0.0, 1.0 - normalized_radius_squared) ** self.flow.profile_exponent
        axial_speeds = self.flow.speed_m_s * profile
        effective_cross_section = len(local) * float(spacing) ** 2
        implied_rate = float(np.mean(axial_speeds) * effective_cross_section)
        return {
            "effective_discrete_cross_section_m2": effective_cross_section,
            "minimum_axial_speed_m_s": float(np.min(axial_speeds)),
            "mean_axial_speed_m_s": float(np.mean(axial_speeds)),
            "maximum_axial_speed_m_s": float(np.max(axial_speeds)),
            "required_mean_speed_m_s": float(
                self.flow.rate_m3_s / effective_cross_section
            ),
            "implied_rate_m3_s": implied_rate,
            "configured_rate_m3_s": self.flow.rate_m3_s,
            "relative_rate_mismatch": float(
                abs(implied_rate - self.flow.rate_m3_s) / self.flow.rate_m3_s
            ),
        }

    def schedule_audit(self, physics_fps, spacing, end_seconds=None):
        end = self.stop_seconds if end_seconds is None else min(float(end_seconds), self.stop_seconds)
        births = self.schedule_for_spacing(physics_fps, spacing, end)
        duration = max(0.0, end - self.start_seconds)
        batch_counts = [
            self.batch_particle_count(spacing, batch_index)
            for batch_index in range(len(births))
        ]
        batch_volumes = [count * float(spacing) ** 3 for count in batch_counts]
        emitted_volume = float(sum(batch_volumes))
        target_volume = duration * self.flow.rate_m3_s
        relative_error = (
            abs(emitted_volume - target_volume) / target_volume if target_volume > 0.0 else 0.0
        )
        return {
            "physics_fps": int(physics_fps),
            "particle_spacing_m": float(spacing),
            "emission_kind": self.emission.kind,
            "phase_count": self.emission.phase_count,
            "full_layer_particles": self.full_layer_particle_count(spacing),
            "particles_per_batch": (
                batch_counts[0]
                if batch_counts and min(batch_counts) == max(batch_counts)
                else None
            ),
            "minimum_particles_per_batch": min(batch_counts, default=0),
            "maximum_particles_per_batch": max(batch_counts, default=0),
            "particle_volume_m3": float(spacing) ** 3,
            "batch_volume_m3": (
                batch_volumes[0]
                if batch_volumes and min(batch_volumes) == max(batch_volumes)
                else None
            ),
            "maximum_batch_volume_m3": max(batch_volumes, default=0.0),
            "birth_batches": len(births),
            "first_birth_step": births[0] if births else None,
            "last_birth_step": births[-1] if births else None,
            "target_volume_m3": target_volume,
            "emitted_volume_m3": emitted_volume,
            "relative_volume_error": relative_error,
            "kinematic_flow_audit": self.kinematic_flow_audit(spacing),
        }

    def configuration_sha256(self):
        encoded = json.dumps(
            _canonical_json_value(self.metadata()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def metadata(self):
        payload = {
            "schema": self.schema_version,
            "product": POUR_SOURCE_PRODUCT,
            "name": self.name,
            "coordinate_system": {
                "axes": "xyz",
                "handedness": "right",
                "up_axis": "y",
                "metres_per_unit": self.metres_per_unit,
            },
            "outlet": {
                "shape": "ellipse",
                "radii_m": list(self.outlet.radii_m),
                "pattern": self.outlet.pattern,
                "stable_warp_fraction": self.outlet.stable_warp_fraction,
            },
            "flow": {
                "rate_m3_s": self.flow.rate_m3_s,
                "speed_m_s": self.flow.speed_m_s,
                "velocity_profile": {
                    "kind": "power_law",
                    "edge_fraction": self.flow.edge_speed_fraction,
                    "exponent": self.flow.profile_exponent,
                },
                "tangent_perturbation_m_s": self.flow.tangent_perturbation_m_s,
                "seed": self.flow.seed,
            },
            "timing": {
                "start_seconds": self.start_seconds,
                "stop_seconds": self.stop_seconds,
            },
            "motion": {
                "kind": "keyframed_rigid",
                "interpolation": "linear_orthonormalized",
                "keyframes": [
                    {
                        "time_seconds": keyframe.time_seconds,
                        "centre": keyframe.centre.astype(float).tolist(),
                        "normal": keyframe.normal.astype(float).tolist(),
                        "up_hint": keyframe.up_hint.astype(float).tolist(),
                    }
                    for keyframe in self.keyframes
                ],
            },
            "metadata": {},
        }
        if self.schema_version >= 2:
            payload["emission"] = {
                "kind": self.emission.kind,
                "phase_count": self.emission.phase_count,
            }
        return payload
