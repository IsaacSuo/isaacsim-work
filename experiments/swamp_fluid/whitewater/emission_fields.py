"""Scale-aware emission potentials for the v6 offline whitewater solver.

This module does not create secondary particles.  It converts audited liquid
fields into three dimensionless, bounded source channels that can be inspected
and calibrated before any marker birth is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from .flow_profiles import FlowProfile, default_impact_profile


CHANNEL_NAMES = ("spray", "entrained_air", "churn")


def smoothstep(edge0, edge1, value):
    edge0 = float(edge0)
    edge1 = float(edge1)
    if not np.isfinite(edge0) or not np.isfinite(edge1) or edge1 <= edge0:
        raise ValueError("smoothstep edges must be finite and increasing")
    value = np.asarray(value, dtype=np.float64)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _condition(field, support, sigma):
    """Normalized Gaussian conditioning that does not bleed through masks."""

    field = np.asarray(field, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    if field.shape != support.shape:
        raise ValueError("Conditioning field/support shapes differ")
    numerator = gaussian_filter(field * support, sigma=sigma, mode="nearest")
    denominator = gaussian_filter(support, sigma=sigma, mode="nearest")
    output = np.zeros_like(field)
    np.divide(numerator, denominator, out=output, where=denominator > 1.0e-6)
    return output


@dataclass(frozen=True)
class EmissionModel:
    """Dimensionless thresholds expressed through gravity and voxel scale."""

    spacing: float
    gravity: float = 9.81
    conditioning_sigma: float = 1.0
    surface_half_width_cells: float = 1.5
    churn_depth_cells: float = 4.0
    sphere_influence_cells: float = 8.0
    profile: FlowProfile = field(default_factory=default_impact_profile)

    def __post_init__(self):
        for name in (
            "spacing",
            "gravity",
            "conditioning_sigma",
            "surface_half_width_cells",
            "churn_depth_cells",
            "sphere_influence_cells",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def characteristic_time(self):
        return float(np.sqrt(self.characteristic_length / self.gravity))

    @property
    def characteristic_velocity(self):
        return float(np.sqrt(self.gravity * self.characteristic_length))

    @property
    def capillary_length(self):
        return float(
            np.sqrt(
                self.profile.surface_tension_n_m
                / (self.profile.density_kg_m3 * self.gravity)
            )
        )

    @property
    def characteristic_length(self):
        # Fine grids are locked to the physical capillary scale so thresholds do
        # not shrink with arbitrary voxel refinement. Coarser grids use their
        # resolvable voxel scale instead of pretending to resolve finer detail.
        return float(max(self.spacing, self.capillary_length))

    @property
    def bond_number(self):
        length = self.characteristic_length
        return float(
            self.profile.density_kg_m3 * self.gravity * length**2
            / self.profile.surface_tension_n_m
        )

    def metadata(self):
        return {
            "spacing": self.spacing,
            "gravity": self.gravity,
            "capillary_length": self.capillary_length,
            "characteristic_length": self.characteristic_length,
            "characteristic_time": self.characteristic_time,
            "characteristic_velocity": self.characteristic_velocity,
            "bond_number": self.bond_number,
            "scale_rule": "max(voxel_spacing, physical_capillary_length)",
            "conditioning_sigma": self.conditioning_sigma,
            "surface_half_width_cells": self.surface_half_width_cells,
            "churn_depth_cells": self.churn_depth_cells,
            "churn_source_influence_cells": self.sphere_influence_cells,
            "sphere_influence_cells": self.sphere_influence_cells,
            "sphere_influence_cells_status": "legacy parameter name; applies to any tagged churn source",
            "dimensionless_thresholds": {
                key: list(value) for key, value in self.profile.thresholds.items()
            },
            "flow_profile": self.profile.metadata(),
        }


def _required(fields, name, shape=None, vector=False):
    if name not in fields:
        raise KeyError(f"Missing liquid field {name}")
    value = np.asarray(fields[name])
    expected = shape + ((3,) if vector else ()) if shape is not None else None
    if expected is not None and value.shape != expected:
        raise ValueError(f"Unexpected {name} shape {value.shape}, expected {expected}")
    if not np.isfinite(value).all():
        raise ValueError(f"Liquid field {name} contains non-finite values")
    return value.astype(np.float64, copy=False)


def raw_emission_fields(fields, model: EmissionModel):
    """Compute bounded raw potentials without contact or temporal gating."""

    phi = _required(fields, "phi")
    if phi.ndim != 3:
        raise ValueError("phi must be a three-dimensional field")
    shape = phi.shape
    normal = _required(fields, "normal", shape, vector=True)
    velocity = _required(fields, "velocity", shape, vector=True)
    acceleration = _required(fields, "acceleration", shape, vector=True)
    curvature = _required(fields, "curvature", shape)
    divergence = _required(fields, "divergence", shape)
    vorticity = _required(fields, "vorticity", shape, vector=True)
    strain_rate = _required(fields, "strain_rate", shape)
    collision_sdf = _required(fields, "collision_sdf", shape)
    churn_source_name = (
        "churn_source_sdf"
        if "churn_source_sdf" in fields
        else "sphere_collision_sdf" if "sphere_collision_sdf" in fields else None
    )
    churn_source_sdf = (
        _required(fields, churn_source_name, shape)
        if churn_source_name is not None
        else None
    )
    surface_valid = _required(fields, "surface_valid", shape) > 0.5
    velocity_valid = _required(fields, "velocity_valid", shape) > 0.5
    acceleration_valid = _required(fields, "acceleration_valid", shape) > 0.5

    spacing = model.spacing
    surface_width = model.surface_half_width_cells * spacing
    surface_shell = np.exp(-0.5 * (phi / surface_width) ** 2)
    surface_support = surface_valid & velocity_valid & (np.abs(phi) <= 2.5 * surface_width)
    solid_clearance = smoothstep(0.5 * spacing, 2.0 * spacing, collision_sdf)
    surface_confidence = surface_shell * surface_support * solid_clearance

    normal_velocity = np.sum(velocity * normal, axis=-1)
    normal_acceleration = np.sum(acceleration * normal, axis=-1)
    conditioned_curvature = _condition(
        curvature, surface_support, model.conditioning_sigma
    )
    conditioned_divergence = _condition(
        divergence, velocity_valid, model.conditioning_sigma
    )
    vorticity_magnitude = np.linalg.norm(vorticity, axis=-1)
    conditioned_rotation = _condition(
        np.maximum(vorticity_magnitude, strain_rate),
        velocity_valid,
        model.conditioning_sigma,
    )
    conditioned_normal_velocity = _condition(
        normal_velocity, surface_support, model.conditioning_sigma
    )
    conditioned_normal_acceleration = _condition(
        normal_acceleration, surface_support & acceleration_valid, model.conditioning_sigma
    )

    curvature_number = (
        np.maximum(conditioned_curvature, 0.0) * model.characteristic_length
    )
    outward_velocity_number = (
        np.maximum(conditioned_normal_velocity, 0.0)
        / model.characteristic_velocity
    )
    inward_velocity_number = (
        np.maximum(-conditioned_normal_velocity, 0.0)
        / model.characteristic_velocity
    )
    outward_acceleration_number = (
        np.maximum(conditioned_normal_acceleration, 0.0) / model.gravity
    )
    inward_acceleration_number = (
        np.maximum(-conditioned_normal_acceleration, 0.0) / model.gravity
    )
    convergence_number = (
        np.maximum(-conditioned_divergence, 0.0) * model.characteristic_time
    )
    rotation_number = conditioned_rotation * model.characteristic_time

    thresholds = model.profile.thresholds
    crest_geometry = smoothstep(*thresholds["crest_curvature"], curvature_number)
    outward_motion = smoothstep(
        *thresholds["spray_outward_velocity"], outward_velocity_number
    )
    outward_impulse = smoothstep(
        *thresholds["spray_outward_acceleration"], outward_acceleration_number
    )
    gravity_facing = 0.25 + 0.75 * smoothstep(
        *thresholds["gravity_facing_normal"], normal[..., 1]
    )
    spray = (
        surface_confidence
        * gravity_facing
        * crest_geometry
        * np.maximum(
            outward_motion, model.profile.spray_impulse_weight * outward_impulse
        )
    )

    compression = smoothstep(
        *thresholds["entrainment_convergence"], convergence_number
    )
    agitation = smoothstep(*thresholds["entrainment_rotation"], rotation_number)
    inward_motion = smoothstep(
        *thresholds["entrainment_inward_velocity"], inward_velocity_number
    )
    entrained_air = (
        surface_confidence
        * compression
        * np.maximum(
            agitation, model.profile.entrainment_inward_weight * inward_motion
        )
    )

    liquid_depth = np.maximum(-phi, 0.0)
    churn_depth = np.exp(
        -0.5 * (liquid_depth / (model.churn_depth_cells * spacing)) ** 2
    )
    liquid_support = (phi <= surface_width) & velocity_valid & acceleration_valid
    churn_clearance = smoothstep(0.0, 0.5 * spacing, collision_sdf)
    policy = model.profile.churn_source_policy
    if policy == "disabled":
        churn_source_proximity = np.zeros(shape, dtype=np.float64)
    elif churn_source_sdf is None:
        churn_source_proximity = (
            np.zeros(shape, dtype=np.float64)
            if policy == "required_tagged_source"
            else np.ones(shape, dtype=np.float64)
        )
    else:
        churn_source_proximity = 1.0 - smoothstep(
            0.5 * spacing,
            model.sphere_influence_cells * spacing,
            churn_source_sdf,
        )
    impact_pressure = smoothstep(
        *thresholds["churn_inward_acceleration"], inward_acceleration_number
    )
    churn_agitation = smoothstep(
        *thresholds["churn_agitation"], rotation_number
    )
    churn = (
        liquid_support
        * churn_clearance
        * churn_depth
        * churn_source_proximity
        * impact_pressure
        * np.maximum(churn_agitation, compression)
    )

    outputs = {
        "surface_confidence": surface_confidence,
        "curvature_number": curvature_number,
        "outward_velocity_number": outward_velocity_number,
        "inward_velocity_number": inward_velocity_number,
        "outward_acceleration_number": outward_acceleration_number,
        "inward_acceleration_number": inward_acceleration_number,
        "convergence_number": convergence_number,
        "rotation_number": rotation_number,
        "spray_raw": model.profile.channel_gains["spray"] * spray,
        "entrained_air_raw": model.profile.channel_gains["entrained_air"]
        * entrained_air,
        "churn_raw": model.profile.channel_gains["churn"] * churn,
    }
    return {
        name: np.clip(value, 0.0, 1.0).astype(np.float32)
        for name, value in outputs.items()
    }


def estimate_precontact_floors(raw_frames, quantile=0.999, multiplier=1.10):
    """Estimate scalar channel noise floors from strictly pre-contact frames."""

    quantile = float(quantile)
    multiplier = float(multiplier)
    if not 0.90 <= quantile < 1.0 or multiplier < 1.0:
        raise ValueError("Invalid pre-contact floor settings")
    frames = list(raw_frames)
    if not frames:
        raise ValueError("At least one pre-contact raw frame is required")
    floors = {}
    for channel in CHANNEL_NAMES:
        key = channel + "_raw"
        values = [np.asarray(frame[key], dtype=np.float64).ravel() for frame in frames]
        floor = float(np.quantile(np.concatenate(values), quantile))
        floors[channel] = float(np.clip(multiplier * floor, 0.0, 0.95))
    return floors


def suppress_precontact_floor(raw, floor):
    raw = np.asarray(raw, dtype=np.float64)
    floor = float(floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError("Noise floor must lie within 0..1")
    normalized = np.clip((raw - floor) / max(1.0 - floor, 1.0e-8), 0.0, 1.0)
    return smoothstep(0.0, 1.0, normalized).astype(np.float32)


class TemporalEmissionFilter:
    """Deterministic attack/decay filtering with exact contact gating."""

    def __init__(
        self,
        shape,
        *,
        attack_seconds=0.018,
        decay_seconds=0.120,
    ):
        self.shape = tuple(int(value) for value in shape)
        self.attack_seconds = float(attack_seconds)
        self.decay_seconds = float(decay_seconds)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("Temporal filter shape must be a positive 3D shape")
        if self.attack_seconds <= 0.0 or self.decay_seconds <= 0.0:
            raise ValueError("Temporal time constants must be positive")
        self.state = {
            channel: np.zeros(self.shape, dtype=np.float32)
            for channel in CHANNEL_NAMES
        }

    def advance(self, targets, dt, event_enabled):
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("Temporal filter dt must be positive")
        outputs = {}
        for channel in CHANNEL_NAMES:
            target = np.asarray(targets[channel], dtype=np.float32)
            if target.shape != self.shape or not np.isfinite(target).all():
                raise ValueError(f"Invalid temporal target for {channel}")
            target = np.clip(target, 0.0, 1.0)
            if not event_enabled:
                self.state[channel].fill(0.0)
                outputs[channel] = self.state[channel].copy()
                continue
            previous = self.state[channel]
            time_constant = np.where(
                target >= previous, self.attack_seconds, self.decay_seconds
            )
            alpha = np.exp(-dt / time_constant)
            updated = alpha * previous + (1.0 - alpha) * target
            self.state[channel] = updated.astype(np.float32)
            outputs[channel] = self.state[channel].copy()
        return outputs
