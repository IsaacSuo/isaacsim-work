"""Convert one-shot PhysX liquid ownership transfers into spray births."""

from __future__ import annotations

import numpy as np

from .marker_birth import (
    BIRTH_DTYPE,
    BirthChannel,
    PhaseKind,
    RenderClass,
    marker_ids,
)
from .state_machine import splitmix64


EXTERNAL_SOURCE_NAMESPACE = np.uint32(1 << 29)
MAX_EXTERNAL_PARTICLE_ID = (1 << 29) - 1


def equal_volume_radius(volume):
    volume = float(volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("Particle phase volume must be finite and positive")
    return float(np.cbrt(3.0 * volume / (4.0 * np.pi)))


def handoff_spray_births(
    source_positions,
    source_velocities,
    source_particle_indices,
    *,
    source_sample,
    birth_time,
    particle_spacing,
    seed=0,
):
    """Create exactly one liquid spray marker for each transferred source ID."""

    positions = np.asarray(source_positions, dtype=np.float32)
    velocities = np.asarray(source_velocities, dtype=np.float32)
    indices = np.ascontiguousarray(source_particle_indices, dtype=np.int64)
    if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Source position/velocity arrays must have shape (n, 3)")
    if (
        indices.ndim != 1
        or len(indices) != len(positions)
        or np.any(indices < 0)
        or np.any(indices > MAX_EXTERNAL_PARTICLE_ID)
        or (len(indices) > 1 and np.any(np.diff(indices) <= 0))
    ):
        raise ValueError("External source particle IDs must be sorted, unique and fit 29 bits")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("External spray inputs contain non-finite values")
    particle_spacing = float(particle_spacing)
    if not np.isfinite(particle_spacing) or particle_spacing <= 0.0:
        raise ValueError("particle_spacing must be finite and positive")
    source_nodes = EXTERNAL_SOURCE_NAMESPACE | indices.astype(np.uint32)
    counters = np.zeros(len(indices), dtype=np.uint32)
    ids = marker_ids(BirthChannel.SPRAY, source_nodes, counters)
    phase_volume = particle_spacing**3
    radius = equal_volume_radius(phase_volume)
    records = np.zeros(len(indices), dtype=BIRTH_DTYPE)
    records["id"] = ids
    records["channel"] = np.uint8(BirthChannel.SPRAY)
    records["phase"] = np.uint8(PhaseKind.LIQUID)
    records["render_class"] = np.uint8(RenderClass.HERO)
    records["source_node_id"] = source_nodes
    records["source_emission_index"] = counters
    records["source_sample"] = np.int32(source_sample)
    records["birth_time"] = np.float64(birth_time)
    records["position"] = positions
    records["velocity"] = velocities
    records["physical_radius"] = np.float32(radius)
    records["representative_count"] = 1.0
    records["phase_volume"] = np.float32(phase_volume)
    records["random_key"] = splitmix64(
        ids ^ np.uint64(int(seed) & ((1 << 64) - 1))
    )
    return records


def external_particle_id(source_node_id):
    source_node_id = np.asarray(source_node_id, dtype=np.uint32)
    if np.any((source_node_id & EXTERNAL_SOURCE_NAMESPACE) == 0):
        raise ValueError("Source node does not belong to the external namespace")
    return (source_node_id & np.uint32(MAX_EXTERNAL_PARTICLE_ID)).astype(np.int64)
