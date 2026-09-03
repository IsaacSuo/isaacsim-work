"""Synthetic gates for deterministic, scale-aware v6 marker birth events."""

from __future__ import annotations

import json

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import (
    BIRTH_DTYPE,
    BirthChannel,
    GridEmissionReservoir,
    MarkerBirthModel,
    PhaseKind,
    expected_marker_budget,
    marker_ids,
    nodal_control_volumes,
    realize_marker_births,
    sample_scalar_trilinear,
)
from whitewater.secondary_handoff import (
    EXTERNAL_SOURCE_NAMESPACE,
    external_particle_id,
    handoff_spray_births,
)


def collect_reservoir(reservoir, increments, durations):
    rows = []
    time = 0.0
    for increment, duration in zip(increments, durations):
        event = reservoir.advance(increment, time, duration)
        rows.extend(
            zip(
                event["source_node_id"].tolist(),
                event["source_emission_index"].tolist(),
                event["birth_time"].tolist(),
                event["random_key"].tolist(),
            )
        )
        time += duration
    return sorted(rows)


# Exact within-step crossing times must not depend on how a constant rate is
# divided into timesteps.
total_expected = np.asarray((0.2, 1.7, 3.4, 0.0, 8.25, 0.999), dtype=np.float64)
single = GridEmissionReservoir(6, BirthChannel.ENTRAINED_AIR, seed=81173)
split = GridEmissionReservoir(6, BirthChannel.ENTRAINED_AIR, seed=81173)
single_rows = collect_reservoir(single, [total_expected], [1.0])
split_rows = collect_reservoir(
    split, [total_expected / 17.0] * 17, [1.0 / 17.0] * 17
)
if len(single_rows) != len(split_rows):
    raise AssertionError("Marker count depends on timestep partition")
for left, right in zip(single_rows, split_rows):
    if left[:2] != right[:2] or left[3] != right[3]:
        raise AssertionError("Marker identity depends on timestep partition")
    if abs(left[2] - right[2]) > 3.0e-15:
        raise AssertionError("Marker crossing time depends on timestep partition")
if not np.allclose(single.accumulator, split.accumulator, atol=3.0e-15):
    raise AssertionError("Reservoir remainder depends on timestep partition")


# Trapezoidal nodal control volumes must integrate the same physical box at
# different resolutions.
coarse = GridSpec((-0.05, -0.05, -0.05), 0.01, (11, 11, 11))
fine = GridSpec((-0.05, -0.05, -0.05), 0.005, (21, 21, 21))
coarse_volume = float(nodal_control_volumes(coarse).sum())
fine_volume = float(nodal_control_volumes(fine).sum())
expected_volume = 0.1**3
if not np.isclose(coarse_volume, expected_volume, rtol=0.0, atol=1.0e-15):
    raise AssertionError("Coarse nodal control volumes do not integrate the box")
if not np.isclose(fine_volume, expected_volume, rtol=0.0, atol=1.0e-15):
    raise AssertionError("Fine nodal control volumes do not integrate the box")
coarse_budget = expected_marker_budget(
    np.ones(coarse.shape), nodal_control_volumes(coarse), 1.25e6, 0.01
).sum()
fine_budget = expected_marker_budget(
    np.ones(fine.shape), nodal_control_volumes(fine), 1.25e6, 0.01
).sum()
if not np.isclose(coarse_budget, fine_budget, rtol=0.0, atol=2.0e-12):
    raise AssertionError("Expected marker budget changes with grid resolution")


# A linear signed-distance plane lets the gate check surface projection and
# phase placement without interpolation ambiguity.
spec = GridSpec((-0.06, -0.06, -0.06), 0.01, (13, 13, 13))
x, y, z = spec.axes()
phi = np.broadcast_to(y[None, :, None], spec.shape).astype(np.float32).copy()
normal = np.zeros(spec.shape + (3,), dtype=np.float32)
normal[..., 1] = 1.0
velocity = np.zeros_like(normal)
velocity[..., 0] = 0.20
velocity[..., 1] = 0.10
collision = np.ones(spec.shape, dtype=np.float32)
liquid = {
    "phi": phi,
    "normal": normal,
    "velocity": velocity,
    "collision_sdf": collision,
}
model = MarkerBirthModel(spec.spacing)


def plane_node_ids(y_index):
    ids = []
    for ix in range(2, 11, 2):
        for iz in range(2, 11, 2):
            ids.append((ix * spec.shape[1] + y_index) * spec.shape[2] + iz)
    return np.asarray(ids, dtype=np.int64)


all_records = []
channel_metrics = {}
for channel, y_index in (
    (BirthChannel.SPRAY, 6),
    (BirthChannel.ENTRAINED_AIR, 6),
    (BirthChannel.CHURN, 4),
):
    expected = np.zeros(spec.cell_count, dtype=np.float64)
    expected[plane_node_ids(y_index)] = 2.25
    reservoir = GridEmissionReservoir(spec.cell_count, channel, seed=9901)
    result = reservoir.advance(expected, 0.25, 0.01)
    records, metrics = realize_marker_births(
        result, channel, 30, liquid, spec, model
    )
    if metrics["accepted_count"] == 0 or metrics["rejected_count"] != 0:
        raise AssertionError(f"Synthetic {channel.name} births were unexpectedly rejected")
    if records.dtype != BIRTH_DTYPE or np.any(np.diff(records["id"]) <= 0):
        raise AssertionError("Birth record contract or ordering is invalid")
    sampled_phi = sample_scalar_trilinear(phi, records["position"], spec)
    if channel == BirthChannel.SPRAY:
        if np.any(records["phase"] != np.uint8(PhaseKind.LIQUID)):
            raise AssertionError("Spray marker did not conserve liquid phase identity")
        if np.any(sampled_phi < -0.20 * spec.spacing - 1.0e-7):
            raise AssertionError("Spray marker was born too far inside the liquid")
    else:
        if np.any(records["phase"] != np.uint8(PhaseKind.GAS)):
            raise AssertionError("Bubble marker did not conserve gas phase identity")
        if np.any(sampled_phi > 0.20 * spec.spacing + 1.0e-7):
            raise AssertionError("Bubble marker was born outside the liquid")
    computed_volume = (
        (4.0 / 3.0)
        * np.pi
        * records["physical_radius"].astype(np.float64) ** 3
        * records["representative_count"].astype(np.float64)
    )
    if not np.allclose(records["phase_volume"], computed_volume, rtol=3.0e-7):
        raise AssertionError("Recorded marker phase volume is inconsistent")
    if np.any(records["birth_time"] < 0.25) or np.any(
        records["birth_time"] > 0.26
    ):
        raise AssertionError("Birth time lies outside its source step")
    expected_ids = marker_ids(
        channel, records["source_node_id"], records["source_emission_index"]
    )
    if not np.array_equal(records["id"], expected_ids):
        raise AssertionError("Packed marker IDs do not match their provenance")
    all_records.append(records)
    channel_metrics[channel.name.lower()] = metrics

combined_ids = np.concatenate([records["id"] for records in all_records])
if len(np.unique(combined_ids)) != len(combined_ids):
    raise AssertionError("Marker identities collide across birth channels")


# Solid overlap must be rejected deterministically after all placement tries.
blocked_liquid = dict(liquid)
blocked_liquid["collision_sdf"] = np.full(spec.shape, -1.0, dtype=np.float32)
blocked_expected = np.zeros(spec.cell_count, dtype=np.float64)
blocked_expected[plane_node_ids(6)] = 1.5
blocked_reservoir = GridEmissionReservoir(
    spec.cell_count, BirthChannel.ENTRAINED_AIR, seed=17
)
blocked_result = blocked_reservoir.advance(blocked_expected, 0.0, 0.01)
blocked_records, blocked_metrics = realize_marker_births(
    blocked_result,
    BirthChannel.ENTRAINED_AIR,
    0,
    blocked_liquid,
    spec,
    model,
)
if len(blocked_records) or blocked_metrics["rejected_count"] == 0:
    raise AssertionError("Collision placement gate accepted an embedded marker")


external_indices = np.asarray((3, 100, 719340), dtype=np.int64)
external_positions = np.asarray(
    ((0.0, 0.02, 0.0), (0.01, 0.03, 0.0), (0.02, 0.04, 0.0)),
    dtype=np.float32,
)
external_velocities = np.asarray(
    ((0.1, 0.2, 0.0), (0.2, 0.3, 0.0), (0.3, 0.4, 0.0)),
    dtype=np.float32,
)
external = handoff_spray_births(
    external_positions,
    external_velocities,
    external_indices,
    source_sample=120,
    birth_time=1.0,
    particle_spacing=0.008,
    seed=17,
)
if not np.array_equal(external_particle_id(external["source_node_id"]), external_indices):
    raise AssertionError("External marker IDs do not recover their PhysX source IDs")
if np.intersect1d(external["id"], combined_ids).size:
    raise AssertionError("External marker namespace collides with grid emission IDs")
external_expected_volume = len(external) * 0.008**3
if not np.isclose(
    external["phase_volume"].sum(dtype=np.float64),
    external_expected_volume,
    rtol=1.0e-7,
):
    raise AssertionError("External marker transfer does not conserve liquid volume")


report = {
    "valid": True,
    "timestep_partition_births": len(single_rows),
    "nodal_box_volume_m3": coarse_volume,
    "resolution_invariant_expected_births": float(coarse_budget),
    "channel_metrics": channel_metrics,
    "unique_marker_ids": len(combined_ids),
    "blocked_candidates_rejected": blocked_metrics["rejected_count"],
    "external_namespace_bit": int(EXTERNAL_SOURCE_NAMESPACE),
    "external_handoff_births": len(external),
    "external_handoff_volume_m3": float(
        external["phase_volume"].sum(dtype=np.float64)
    ),
}
print(json.dumps(report, indent=2))
