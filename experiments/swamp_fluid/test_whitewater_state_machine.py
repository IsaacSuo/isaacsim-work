"""Synthetic gates for deterministic and conservative whitewater states."""

from __future__ import annotations

import json

import numpy as np

from whitewater.state_machine import (
    DeterministicEmissionReservoir,
    EmissionChannel,
    WhitewaterState,
    advance_kinematics,
    bubble_shape,
    bubble_terminal_velocity,
    make_markers,
    sphere_volume,
    transition_state,
)


def collect_emissions(reservoir, increments):
    rows = []
    for increment in increments:
        result = reservoir.advance(increment)
        rows.extend(
            zip(
                result["source_particle_id"].tolist(),
                result["source_emission_index"].tolist(),
                result["random_key"].tolist(),
            )
        )
    return sorted(rows)


particle_count = 6
total_expected = np.asarray((0.2, 1.7, 3.4, 0.0, 8.25, 0.999), dtype=np.float64)
single = DeterministicEmissionReservoir(
    particle_count, EmissionChannel.ENTRAINED_AIR, seed=9817
)
split = DeterministicEmissionReservoir(
    particle_count, EmissionChannel.ENTRAINED_AIR, seed=9817
)
single_rows = collect_emissions(single, [total_expected])
split_rows = collect_emissions(split, [total_expected / 17.0] * 17)
if single_rows != split_rows:
    raise AssertionError("Emission identities depend on timestep partition")
if not np.allclose(single.accumulator, split.accumulator, atol=2.0e-15):
    raise AssertionError("Emission reservoir depends on timestep partition")
if not np.array_equal(single.emission_count, split.emission_count):
    raise AssertionError("Emission counts depend on timestep partition")

radii = np.asarray((0.0003, 0.0006, 0.001, 0.002, 0.004, 0.006))
rise = bubble_terminal_velocity(radii)
if np.any(np.diff(rise) < -1.0e-12):
    raise AssertionError(f"Bubble rise is not monotonic: {rise}")
if rise[0] <= 0.0 or rise[-1] > 0.3500001:
    raise AssertionError(f"Bubble terminal velocity leaves accepted bounds: {rise}")
shapes = bubble_shape(radii, rise)
shape_volume_error = float(np.max(np.abs(np.prod(shapes, axis=1) - 1.0)))
if shape_volume_error > 2.0e-6:
    raise AssertionError(f"Bubble shape does not preserve volume: {shape_volume_error}")
if np.any(shapes[:, 1] > shapes[:, 0] + 1.0e-7):
    raise AssertionError("Bubble model produced a prolate bubble")

count = len(radii)
weights = np.asarray((1.0, 2.0, 4.0, 1.5, 0.75, 0.5), dtype=np.float32)
bubbles = make_markers(
    marker_ids=np.arange(100, 100 + count, dtype=np.uint64),
    state=WhitewaterState.ENTRAINED_BUBBLE,
    source_particle_ids=np.arange(count, dtype=np.int32),
    birth_sample=12,
    birth_time=0.1,
    positions=np.zeros((count, 3), dtype=np.float32),
    velocities=np.zeros((count, 3), dtype=np.float32),
    radii=radii,
    representative_weights=weights,
    random_keys=np.arange(900, 900 + count, dtype=np.uint64),
)
expected_gas = sphere_volume(radii) * weights
if not np.allclose(bubbles["gas_volume"], expected_gas, rtol=2.0e-7, atol=0.0):
    raise AssertionError("Bubble marker gas volume is inconsistent with radius/weight")
if np.any(bubbles["liquid_volume"] != 0.0):
    raise AssertionError("Entrained bubbles incorrectly carry liquid volume")
gas_before = float(bubbles["gas_volume"].sum(dtype=np.float64))
selection = np.asarray((True, False, True, False, True, False))
transition_state(bubbles, selection, WhitewaterState.SURFACE_BUBBLE)
gas_after = float(bubbles["gas_volume"].sum(dtype=np.float64))
if gas_before != gas_after:
    raise AssertionError("Gas volume changed during bubble phase transition")
if not np.all(
    bubbles["state"][selection] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
):
    raise AssertionError("Selected bubble states did not transition")

spray = make_markers(
    marker_ids=np.asarray((1,), dtype=np.uint64),
    state=WhitewaterState.SPRAY,
    source_particle_ids=np.asarray((3,), dtype=np.int32),
    birth_sample=0,
    birth_time=0.0,
    positions=np.asarray(((0.0, 1.0, 0.0),), dtype=np.float32),
    velocities=np.asarray(((1.0, 2.0, 0.0),), dtype=np.float32),
    radii=np.asarray((0.001,), dtype=np.float32),
    representative_weights=np.asarray((1.0,), dtype=np.float32),
    random_keys=np.asarray((7,), dtype=np.uint64),
)
advance_kinematics(spray, 0.01)
expected_velocity = np.asarray((1.0, 2.0 - 9.81 * 0.01, 0.0))
if not np.allclose(spray["velocity"][0], expected_velocity, atol=2.0e-7):
    raise AssertionError("Spray semi-implicit gravity update is incorrect")
expected_position = np.asarray((0.01, 1.0 + expected_velocity[1] * 0.01, 0.0))
if not np.allclose(spray["position"][0], expected_position, atol=2.0e-7):
    raise AssertionError("Spray position did not follow semi-implicit velocity")

test_bubble = bubbles[[1]].copy()
test_bubble["state"] = np.uint8(WhitewaterState.ENTRAINED_BUBBLE)
test_bubble["velocity"] = 0.0
terminal = float(bubble_terminal_velocity(test_bubble["radius"])[0])
previous_velocity = 0.0
for _ in range(200):
    advance_kinematics(test_bubble, 0.001)
    current_velocity = float(test_bubble["velocity"][0, 1])
    if current_velocity + 1.0e-8 < previous_velocity or current_velocity > terminal + 1.0e-6:
        raise AssertionError("Bubble relaxation overshot or reversed")
    previous_velocity = current_velocity

report = {
    "valid": True,
    "emitted_markers": len(single_rows),
    "emission_counts": single.emission_count.tolist(),
    "terminal_velocity_mps": rise.tolist(),
    "shape_volume_maximum_error": shape_volume_error,
    "gas_volume_m3": gas_after,
    "bubble_relaxed_velocity_mps": previous_velocity,
    "bubble_terminal_velocity_mps": terminal,
}
print(json.dumps(report, indent=2))
