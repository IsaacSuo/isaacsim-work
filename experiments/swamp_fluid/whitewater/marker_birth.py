"""Deterministic grid-to-marker birth primitives for the v6 whitewater solver.

The emission fields are dimensionless source potentials.  This module turns
their space-time integral into discrete, reproducible birth events.  It does
not advance markers or make rendering decisions beyond recording an optical
scale class; those responsibilities belong to later solver/render stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .liquid_fields import GridSpec
from .state_machine import splitmix64, uniform01_from_keys


class BirthChannel(IntEnum):
    SPRAY = 1
    ENTRAINED_AIR = 2
    CHURN = 3


class PhaseKind(IntEnum):
    LIQUID = 1
    GAS = 2


class RenderClass(IntEnum):
    MICRO_VOLUME = 0
    INSTANCED = 1
    HERO = 2


CHANNEL_NAMES = {
    BirthChannel.SPRAY: "spray",
    BirthChannel.ENTRAINED_AIR: "entrained_air",
    BirthChannel.CHURN: "churn",
}


BIRTH_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("channel", "u1"),
        ("phase", "u1"),
        ("render_class", "u1"),
        ("source_node_id", "<u4"),
        ("source_emission_index", "<u4"),
        ("source_sample", "<i4"),
        ("birth_time", "<f8"),
        ("position", "<f4", (3,)),
        ("velocity", "<f4", (3,)),
        ("physical_radius", "<f4"),
        ("representative_count", "<f4"),
        ("phase_volume", "<f4"),
        ("random_key", "<u8"),
    ]
)


@dataclass(frozen=True)
class MarkerBirthModel:
    """Physical-scale marker rates and size distributions.

    Rate density has units markers / (m^3 s).  It multiplies the nodal control
    volume, the dimensionless emission potential, and the step duration.
    Physical radius controls the later dynamics.  ``representative_count``
    allows sub-pixel micro markers to stand for a small unresolved population.
    """

    spacing: float
    gravity: float = 9.81
    spray_rate_density: float = 1.20e6
    entrained_air_rate_density: float = 1.60e6
    churn_rate_density: float = 5.00e6
    maximum_position_attempts: int = 4

    def __post_init__(self):
        values = (
            self.spacing,
            self.gravity,
            self.spray_rate_density,
            self.entrained_air_rate_density,
            self.churn_rate_density,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("Birth model scales and rate densities must be positive")
        if not 1 <= int(self.maximum_position_attempts) <= 16:
            raise ValueError("maximum_position_attempts must lie within 1..16")

    @property
    def characteristic_velocity(self):
        return float(np.sqrt(self.gravity * self.spacing))

    def rate_density(self, channel):
        channel = BirthChannel(channel)
        return {
            BirthChannel.SPRAY: self.spray_rate_density,
            BirthChannel.ENTRAINED_AIR: self.entrained_air_rate_density,
            BirthChannel.CHURN: self.churn_rate_density,
        }[channel]

    def metadata(self):
        return {
            "spacing": self.spacing,
            "gravity": self.gravity,
            "characteristic_velocity": self.characteristic_velocity,
            "rate_density_markers_per_m3_s": {
                CHANNEL_NAMES[channel]: self.rate_density(channel)
                for channel in BirthChannel
            },
            "maximum_position_attempts": self.maximum_position_attempts,
            "alternating_projection_iterations": 3,
            "radius_distributions": {
                "spray": {
                    "kind": "clipped_lognormal",
                    "median_m": 0.00045,
                    "log_sigma": 0.55,
                    "range_m": [0.00015, 0.00250],
                },
                "entrained_air": {
                    "kind": "clipped_lognormal",
                    "median_m": 0.00055,
                    "log_sigma": 0.70,
                    "range_m": [0.00012, 0.00400],
                },
                "churn": {
                    "kind": "clipped_lognormal",
                    "median_m": 0.00110,
                    "log_sigma": 0.75,
                    "range_m": [0.00018, 0.00800],
                },
            },
            "render_class_semantics": {
                "0": "unresolved micro population; render as density/cluster",
                "1": "ordinary instanced marker",
                "2": "individually resolved hero marker",
            },
        }


def _channel_key(channel):
    mask = (1 << 64) - 1
    return np.uint64((int(channel) * 0xD6E8FEB86659FD93) & mask)


def _stream_uniform(keys, stream):
    mask = (1 << 64) - 1
    salt = np.uint64((int(stream) * 0x9E3779B97F4A7C15) & mask)
    return uniform01_from_keys(splitmix64(np.asarray(keys, dtype=np.uint64) ^ salt))


def marker_ids(channel, source_node_ids, source_emission_indices):
    """Pack a stable channel/node/counter identity into one uint64."""

    channel = int(BirthChannel(channel))
    nodes = np.asarray(source_node_ids, dtype=np.uint64)
    counters = np.asarray(source_emission_indices, dtype=np.uint64)
    if np.any(nodes >= np.uint64(1 << 30)):
        raise ValueError("source node ID exceeds the 30-bit marker-ID allocation")
    if np.any(counters >= np.uint64(1 << 32)):
        raise ValueError("source emission counter exceeds uint32")
    return (np.uint64(channel) << np.uint64(62)) | (nodes << np.uint64(32)) | counters


class GridEmissionReservoir:
    """Integrate nodal marker budgets with exact within-step crossing times."""

    def __init__(self, node_count, channel, seed=0):
        self.node_count = int(node_count)
        self.channel = BirthChannel(channel)
        self.seed = int(seed)
        if not 1 <= self.node_count < (1 << 30):
            raise ValueError("node_count must fit the marker identity allocation")
        nodes = np.arange(self.node_count, dtype=np.uint64)
        seed_key = np.uint64(self.seed & ((1 << 64) - 1))
        keys = splitmix64(nodes ^ _channel_key(self.channel) ^ seed_key)
        self.accumulator = uniform01_from_keys(keys)
        self.emission_count = np.zeros(self.node_count, dtype=np.uint32)

    def advance(self, expected_births, step_start_time, dt):
        expected = np.asarray(expected_births, dtype=np.float64).reshape(-1)
        if expected.shape != (self.node_count,):
            raise ValueError("Expected-birth field has the wrong node count")
        if not np.isfinite(expected).all() or np.any(expected < 0.0):
            raise ValueError("Expected births must be finite and non-negative")
        dt = float(dt)
        step_start_time = float(step_start_time)
        if dt <= 0.0 or not np.isfinite((dt, step_start_time)).all():
            raise ValueError("Birth step time and duration must be finite")

        active = np.flatnonzero(expected > 0.0)
        if not len(active):
            return _empty_reservoir_result()
        before = self.accumulator[active].copy()
        accumulated = before + expected[active]
        births_per_node = np.floor(accumulated).astype(np.uint64)
        self.accumulator[active] = accumulated - births_per_node
        emitting = births_per_node > 0
        if not np.any(emitting):
            return _empty_reservoir_result()

        sources = active[emitting]
        counts = births_per_node[emitting]
        total = int(counts.sum(dtype=np.uint64))
        if total > np.iinfo(np.int32).max:
            raise OverflowError("A single marker step is unreasonably large")
        repeated_sources = np.repeat(sources, counts.astype(np.int64))
        prefix = np.cumsum(counts, dtype=np.uint64) - counts
        local_offsets = (
            np.arange(total, dtype=np.uint64) - np.repeat(prefix, counts.astype(np.int64))
        )
        old_counts = self.emission_count[sources].astype(np.uint64)
        emission_indices = np.repeat(old_counts, counts.astype(np.int64)) + local_offsets
        if np.any(emission_indices >= np.uint64(1 << 32)):
            raise OverflowError("A source node exceeded its uint32 emission counter")
        self.emission_count[sources] += counts.astype(np.uint32)

        expected_repeated = expected[repeated_sources]
        # ``active`` is sorted, so this recovers the immutable pre-step phase
        # without allocating a second full-grid lookup field.
        before_repeated = before[np.searchsorted(active, repeated_sources)]
        fractions = (
            1.0 - before_repeated + local_offsets.astype(np.float64)
        ) / expected_repeated
        fractions = np.clip(fractions, 0.0, 1.0)

        source_u64 = repeated_sources.astype(np.uint64)
        counter_u64 = emission_indices.astype(np.uint64)
        random_keys = splitmix64(
            source_u64
            ^ _channel_key(self.channel)
            ^ (counter_u64 * np.uint64(0xA0761D6478BD642F))
            ^ np.uint64(self.seed & ((1 << 64) - 1))
        )
        return {
            "source_node_id": repeated_sources.astype(np.uint32),
            "source_emission_index": emission_indices.astype(np.uint32),
            "step_fraction": fractions,
            "birth_time": step_start_time + dt * fractions,
            "random_key": random_keys,
        }


def _empty_reservoir_result():
    return {
        "source_node_id": np.empty(0, dtype=np.uint32),
        "source_emission_index": np.empty(0, dtype=np.uint32),
        "step_fraction": np.empty(0, dtype=np.float64),
        "birth_time": np.empty(0, dtype=np.float64),
        "random_key": np.empty(0, dtype=np.uint64),
    }


def nodal_control_volumes(spec: GridSpec):
    """Trapezoidal control volumes for a node-centred Cartesian grid."""

    axes = []
    for size in spec.shape:
        weights = np.ones(size, dtype=np.float64)
        weights[[0, -1]] = 0.5
        axes.append(weights)
    return (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
        * spec.spacing**3
    )


def expected_marker_budget(strength, control_volumes, rate_density, dt):
    strength = np.asarray(strength, dtype=np.float64)
    volumes = np.asarray(control_volumes, dtype=np.float64)
    if strength.shape != volumes.shape:
        raise ValueError("Emission strength and control-volume shapes differ")
    if not np.isfinite(strength).all() or np.any((strength < 0.0) | (strength > 1.0)):
        raise ValueError("Emission strength must be finite and bounded by 0..1")
    if rate_density <= 0.0 or dt <= 0.0:
        raise ValueError("Rate density and dt must be positive")
    return strength * volumes * float(rate_density) * float(dt)


def node_positions(source_node_ids, spec: GridSpec):
    ids = np.asarray(source_node_ids, dtype=np.uint64)
    yz = np.uint64(spec.shape[1] * spec.shape[2])
    nz = np.uint64(spec.shape[2])
    ix = ids // yz
    remainder = ids - ix * yz
    iy = remainder // nz
    iz = remainder - iy * nz
    indices = np.column_stack((ix, iy, iz)).astype(np.float64)
    return np.asarray(spec.origin, dtype=np.float64) + spec.spacing * indices


def sample_scalar_trilinear(field, positions, spec: GridSpec, outside=np.nan):
    field = np.asarray(field)
    positions = np.asarray(positions, dtype=np.float64)
    if field.shape != spec.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Invalid scalar field or positions for trilinear sampling")
    coordinates = (positions - np.asarray(spec.origin)) / spec.spacing
    finite = np.isfinite(coordinates).all(axis=1)
    base = np.zeros(coordinates.shape, dtype=np.int64)
    base[finite] = np.floor(coordinates[finite]).astype(np.int64)
    fraction = coordinates - base
    valid = finite & np.all(
        (base >= 0) & (base < np.asarray(spec.shape) - 1), axis=1
    )
    result = np.full(len(positions), outside, dtype=np.float64)
    if not np.any(valid):
        return result
    b = base[valid]
    f = fraction[valid]
    values = np.zeros(len(b), dtype=np.float64)
    for ox in (0, 1):
        wx = f[:, 0] if ox else 1.0 - f[:, 0]
        for oy in (0, 1):
            wy = f[:, 1] if oy else 1.0 - f[:, 1]
            for oz in (0, 1):
                wz = f[:, 2] if oz else 1.0 - f[:, 2]
                values += wx * wy * wz * field[
                    b[:, 0] + ox, b[:, 1] + oy, b[:, 2] + oz
                ]
    result[valid] = values
    return result


def scalar_gradient_at_nodes(field, source_node_ids, spec: GridSpec):
    """Sample a finite-difference scalar gradient at selected grid nodes."""

    field = np.asarray(field, dtype=np.float64)
    if field.shape != spec.shape:
        raise ValueError("Scalar gradient field does not match the grid")
    ids = np.asarray(source_node_ids, dtype=np.uint64)
    yz = np.uint64(spec.shape[1] * spec.shape[2])
    nz = np.uint64(spec.shape[2])
    ix = (ids // yz).astype(np.int64)
    remainder = ids - ix.astype(np.uint64) * yz
    iy = (remainder // nz).astype(np.int64)
    iz = (remainder - iy.astype(np.uint64) * nz).astype(np.int64)
    indices = (ix, iy, iz)
    gradient = np.empty((len(ids), 3), dtype=np.float64)
    for axis in range(3):
        lower = list(indices)
        upper = list(indices)
        lower[axis] = np.maximum(indices[axis] - 1, 0)
        upper[axis] = np.minimum(indices[axis] + 1, spec.shape[axis] - 1)
        denominator = (upper[axis] - lower[axis]) * spec.spacing
        gradient[:, axis] = (
            field[tuple(upper)] - field[tuple(lower)]
        ) / denominator
    return gradient


def _normal_tangent_frame(normals, keys, stream):
    u = _stream_uniform(keys, stream)
    v = _stream_uniform(keys, stream + 1)
    y = 2.0 * u - 1.0
    radial = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    angle = 2.0 * np.pi * v
    random_direction = np.column_stack(
        (radial * np.cos(angle), y, radial * np.sin(angle))
    )
    tangent = random_direction - np.sum(random_direction * normals, axis=1)[
        :, None
    ] * normals
    norm = np.linalg.norm(tangent, axis=1)
    degenerate = norm < 1.0e-8
    if np.any(degenerate):
        reference = np.tile((1.0, 0.0, 0.0), (np.count_nonzero(degenerate), 1))
        parallel = np.abs(normals[degenerate, 0]) > 0.9
        reference[parallel] = (0.0, 0.0, 1.0)
        tangent[degenerate] = np.cross(normals[degenerate], reference)
        norm[degenerate] = np.linalg.norm(tangent[degenerate], axis=1)
    tangent /= norm[:, None]
    bitangent = np.cross(normals, tangent)
    return tangent, bitangent


def _radii_and_population(channel, keys):
    channel = BirthChannel(channel)
    u1 = np.clip(_stream_uniform(keys, 20), 1.0e-12, 1.0)
    u2 = _stream_uniform(keys, 21)
    gaussian = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    settings = {
        BirthChannel.SPRAY: (0.00045, 0.55, 0.00015, 0.00250),
        BirthChannel.ENTRAINED_AIR: (0.00055, 0.70, 0.00012, 0.00400),
        BirthChannel.CHURN: (0.00110, 0.75, 0.00018, 0.00800),
    }[channel]
    median, sigma, minimum, maximum = settings
    radius = np.clip(median * np.exp(sigma * gaussian), minimum, maximum)
    if channel == BirthChannel.SPRAY:
        representative = np.ones(len(keys), dtype=np.float64)
        hero_threshold = 0.00120
        micro_threshold = 0.00030
    else:
        representative = np.clip((0.00045 / radius) ** 1.5, 1.0, 12.0)
        hero_threshold = 0.00250 if channel == BirthChannel.ENTRAINED_AIR else 0.00350
        micro_threshold = 0.00030
    render_class = np.full(len(keys), RenderClass.INSTANCED, dtype=np.uint8)
    render_class[radius < micro_threshold] = RenderClass.MICRO_VOLUME
    render_class[radius >= hero_threshold] = RenderClass.HERO
    return radius, representative, render_class


def _candidate_positions(channel, node_position, phi, normal, keys, spacing, attempt):
    tangent, bitangent = _normal_tangent_frame(normal, keys, 30 + 5 * attempt)
    j1 = _stream_uniform(keys, 32 + 5 * attempt) - 0.5
    j2 = _stream_uniform(keys, 33 + 5 * attempt) - 0.5
    channel = BirthChannel(channel)
    if channel == BirthChannel.CHURN:
        direction_u = _stream_uniform(keys, 34 + 5 * attempt)
        direction_v = _stream_uniform(keys, 35 + 5 * attempt)
        vertical = 2.0 * direction_u - 1.0
        radial = np.sqrt(np.maximum(1.0 - vertical * vertical, 0.0))
        angle = 2.0 * np.pi * direction_v
        direction = np.column_stack(
            (radial * np.cos(angle), vertical, radial * np.sin(angle))
        )
        return node_position + direction * (0.22 * spacing)

    surface = node_position - phi[:, None] * normal
    tangential = spacing * 0.55 * (j1[:, None] * tangent + j2[:, None] * bitangent)
    depth_u = _stream_uniform(keys, 34 + 5 * attempt)
    if channel == BirthChannel.SPRAY:
        offset = spacing * (0.15 + 0.35 * depth_u)
    else:
        offset = -spacing * (0.35 + 0.65 * depth_u)
    return surface + tangential + offset[:, None] * normal


def _initial_velocity(channel, carrier_velocity, normal, keys, model):
    tangent, bitangent = _normal_tangent_frame(normal, keys, 70)
    angle = 2.0 * np.pi * _stream_uniform(keys, 72)
    tangent_direction = np.cos(angle)[:, None] * tangent + np.sin(angle)[
        :, None
    ] * bitangent
    magnitude_u = _stream_uniform(keys, 73)
    vc = model.characteristic_velocity
    channel = BirthChannel(channel)
    if channel == BirthChannel.SPRAY:
        outward = np.maximum(np.sum(carrier_velocity * normal, axis=1), 0.0)
        normal_kick = vc * (0.10 + 0.25 * magnitude_u) + 0.35 * outward
        return (
            carrier_velocity
            + normal_kick[:, None] * normal
            + (0.05 * vc * magnitude_u)[:, None] * tangent_direction
        )
    if channel == BirthChannel.ENTRAINED_AIR:
        return (
            carrier_velocity
            - (0.04 * vc * magnitude_u)[:, None] * normal
            + (0.025 * vc * magnitude_u)[:, None] * tangent_direction
        )
    random_y = 2.0 * _stream_uniform(keys, 74) - 1.0
    random_radial = np.sqrt(np.maximum(1.0 - random_y * random_y, 0.0))
    random_angle = 2.0 * np.pi * _stream_uniform(keys, 75)
    random_direction = np.column_stack(
        (
            random_radial * np.cos(random_angle),
            random_y,
            random_radial * np.sin(random_angle),
        )
    )
    return carrier_velocity + (0.10 * vc * magnitude_u)[:, None] * random_direction


def realize_marker_births(
    reservoir_result,
    channel,
    source_sample,
    liquid_fields,
    spec: GridSpec,
    model: MarkerBirthModel,
):
    """Create accepted marker records and a deterministic rejection report."""

    channel = BirthChannel(channel)
    source_nodes = np.asarray(reservoir_result["source_node_id"], dtype=np.uint32)
    count = len(source_nodes)
    if not count:
        return np.empty(0, dtype=BIRTH_DTYPE), {
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "rejected_nonfinite": 0,
            "rejected_phase": 0,
            "rejected_collision": 0,
        }
    shape = spec.shape
    required_scalar = ("phi", "collision_sdf")
    required_vector = ("normal", "velocity")
    for name in required_scalar:
        if name not in liquid_fields or np.asarray(liquid_fields[name]).shape != shape:
            raise ValueError(f"Missing or malformed liquid scalar field {name}")
    for name in required_vector:
        if name not in liquid_fields or np.asarray(liquid_fields[name]).shape != shape + (3,):
            raise ValueError(f"Missing or malformed liquid vector field {name}")

    flat_nodes = source_nodes.astype(np.int64)
    node_position = node_positions(source_nodes, spec)
    phi = np.asarray(liquid_fields["phi"]).reshape(-1)[flat_nodes].astype(np.float64)
    normal = np.asarray(liquid_fields["normal"]).reshape(-1, 3)[flat_nodes].astype(np.float64)
    normal_length = np.linalg.norm(normal, axis=1)
    valid_normal = normal_length > 0.5
    normal[valid_normal] /= normal_length[valid_normal, None]
    normal[~valid_normal] = (0.0, 1.0, 0.0)
    carrier_velocity = np.asarray(liquid_fields["velocity"]).reshape(-1, 3)[
        flat_nodes
    ].astype(np.float64)
    keys = np.asarray(reservoir_result["random_key"], dtype=np.uint64)
    radius, representative, render_class = _radii_and_population(channel, keys)
    collision_normal = scalar_gradient_at_nodes(
        liquid_fields["collision_sdf"], source_nodes, spec
    )
    collision_normal_length = np.linalg.norm(collision_normal, axis=1)
    valid_collision_normal = collision_normal_length > 1.0e-6
    collision_normal[valid_collision_normal] /= collision_normal_length[
        valid_collision_normal, None
    ]
    collision_normal[~valid_collision_normal] = normal[~valid_collision_normal]

    position = np.zeros((count, 3), dtype=np.float64)
    accepted = np.zeros(count, dtype=bool)
    final_phi = np.full(count, np.nan, dtype=np.float64)
    final_collision = np.full(count, np.nan, dtype=np.float64)
    for attempt in range(model.maximum_position_attempts):
        unresolved = ~accepted
        if not np.any(unresolved):
            break
        candidate = _candidate_positions(
            channel,
            node_position[unresolved],
            phi[unresolved],
            normal[unresolved],
            keys[unresolved],
            spec.spacing,
            attempt,
        )
        clearance = radius[unresolved] + 0.08 * spec.spacing
        # Alternate phase and solid projections because satisfying one
        # constraint can slightly violate the other near contact curves.
        for _ in range(3):
            sampled_phi = sample_scalar_trilinear(
                liquid_fields["phi"], candidate, spec
            )
            sampled_collision = sample_scalar_trilinear(
                liquid_fields["collision_sdf"], candidate, spec
            )
            finite = np.isfinite(sampled_phi) & np.isfinite(sampled_collision)
            if channel == BirthChannel.SPRAY:
                phase_correction = np.maximum(
                    0.05 * spec.spacing - sampled_phi, 0.0
                )
                candidate[finite] += (
                    phase_correction[finite, None] * normal[unresolved][finite]
                )
            else:
                phase_correction = np.maximum(
                    sampled_phi + 0.05 * spec.spacing, 0.0
                )
                candidate[finite] -= (
                    phase_correction[finite, None] * normal[unresolved][finite]
                )
            collision_correction = np.maximum(clearance - sampled_collision, 0.0)
            candidate[finite] += (
                collision_correction[finite, None]
                * collision_normal[unresolved][finite]
            )
        sampled_phi = sample_scalar_trilinear(liquid_fields["phi"], candidate, spec)
        sampled_collision = sample_scalar_trilinear(
            liquid_fields["collision_sdf"], candidate, spec
        )
        phase_ok = (
            sampled_phi >= -0.20 * spec.spacing
            if channel == BirthChannel.SPRAY
            else sampled_phi <= 0.20 * spec.spacing
        )
        valid = (
            np.isfinite(sampled_phi)
            & np.isfinite(sampled_collision)
            & phase_ok
            & (sampled_collision >= clearance)
        )
        unresolved_ids = np.flatnonzero(unresolved)
        final_phi[unresolved_ids] = sampled_phi
        final_collision[unresolved_ids] = sampled_collision
        accepted_ids = unresolved_ids[valid]
        position[accepted_ids] = candidate[valid]
        accepted[accepted_ids] = True

    accepted_ids = np.flatnonzero(accepted)
    if not len(accepted_ids):
        finite = np.isfinite(final_phi) & np.isfinite(final_collision)
        phase_failure = (
            final_phi < -0.20 * spec.spacing
            if channel == BirthChannel.SPRAY
            else final_phi > 0.20 * spec.spacing
        )
        collision_failure = final_collision < radius + 0.08 * spec.spacing
        return np.empty(0, dtype=BIRTH_DTYPE), {
            "candidate_count": count,
            "accepted_count": 0,
            "rejected_count": count,
            "rejected_nonfinite": int(np.count_nonzero(~finite)),
            "rejected_phase": int(np.count_nonzero(finite & phase_failure)),
            "rejected_collision": int(
                np.count_nonzero(finite & ~phase_failure & collision_failure)
            ),
        }
    velocity = _initial_velocity(
        channel,
        carrier_velocity[accepted_ids],
        normal[accepted_ids],
        keys[accepted_ids],
        model,
    )
    records = np.zeros(len(accepted_ids), dtype=BIRTH_DTYPE)
    records["channel"] = np.uint8(channel)
    records["phase"] = np.uint8(
        PhaseKind.LIQUID if channel == BirthChannel.SPRAY else PhaseKind.GAS
    )
    records["source_node_id"] = source_nodes[accepted_ids]
    records["source_emission_index"] = np.asarray(
        reservoir_result["source_emission_index"], dtype=np.uint32
    )[accepted_ids]
    records["id"] = marker_ids(
        channel,
        records["source_node_id"],
        records["source_emission_index"],
    )
    records["source_sample"] = int(source_sample)
    records["birth_time"] = np.asarray(
        reservoir_result["birth_time"], dtype=np.float64
    )[accepted_ids]
    records["position"] = position[accepted_ids].astype(np.float32)
    records["velocity"] = velocity.astype(np.float32)
    records["physical_radius"] = radius[accepted_ids].astype(np.float32)
    records["representative_count"] = representative[accepted_ids].astype(np.float32)
    records["phase_volume"] = (
        (4.0 / 3.0)
        * np.pi
        * radius[accepted_ids] ** 3
        * representative[accepted_ids]
    ).astype(np.float32)
    records["render_class"] = render_class[accepted_ids]
    records["random_key"] = keys[accepted_ids]
    order = np.argsort(records["id"])
    rejected = ~accepted
    finite = np.isfinite(final_phi) & np.isfinite(final_collision)
    phase_failure = (
        final_phi < -0.20 * spec.spacing
        if channel == BirthChannel.SPRAY
        else final_phi > 0.20 * spec.spacing
    )
    collision_failure = final_collision < radius + 0.08 * spec.spacing
    return records[order], {
        "candidate_count": count,
        "accepted_count": len(records),
        "rejected_count": count - len(records),
        "rejected_nonfinite": int(np.count_nonzero(rejected & ~finite)),
        "rejected_phase": int(np.count_nonzero(rejected & finite & phase_failure)),
        "rejected_collision": int(
            np.count_nonzero(rejected & finite & ~phase_failure & collision_failure)
        ),
    }
