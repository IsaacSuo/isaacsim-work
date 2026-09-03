"""Synthetic gates for v6 surface-foam parcels and burst repellents."""

from __future__ import annotations

import json

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_solver import EVENT_DTYPE, TerminalEventKind
from whitewater.surface_foam import (
    FOAM_DTYPE,
    REPELLENT_DTYPE,
    SurfaceFoamModel,
    advance_surface_layer,
    source_surface_foam,
)


def make_event(event_id, kind, radius, representative, speed, position=(0, 0, 0)):
    event = np.zeros(1, dtype=EVENT_DTYPE)
    event["event_id"] = np.uint64(event_id)
    event["marker_id"] = np.uint64(event_id + 1000)
    event["kind"] = np.uint8(kind)
    event["phase"] = np.uint8(
        1 if kind == TerminalEventKind.SPRAY_REENTRY else 2
    )
    event["event_time"] = 0.0
    event["position"] = position
    event["normal"] = (0.0, 1.0, 0.0)
    event["physical_radius"] = radius
    event["representative_count"] = representative
    event["phase_volume"] = (
        (4.0 / 3.0) * np.pi * radius**3 * representative
    )
    event["random_key"] = np.uint64(event_id * 9187)
    event["impact_speed"] = speed
    return event


model = SurfaceFoamModel(0.01)
events = np.concatenate(
    (
        make_event(1, TerminalEventKind.SPRAY_REENTRY, 0.0005, 1.0, 0.05),
        make_event(2, TerminalEventKind.SPRAY_REENTRY, 0.0008, 1.0, 2.0),
        make_event(3, TerminalEventKind.BUBBLE_BURST, 0.0015, 3.0, 0.0),
    )
)
foam, repellents, source_metrics = source_surface_foam(events, model)
if foam.dtype != FOAM_DTYPE or repellents.dtype != REPELLENT_DTYPE:
    raise AssertionError("Surface source arrays have the wrong dtype")
if len(foam) != 2 or len(repellents) != 1:
    raise AssertionError("Weber gate or bubble repellent source count is wrong")
if source_metrics["rejected_low_weber"] != 1:
    raise AssertionError("Low-Weber spray event was not reported")
if np.any(foam["film_area"] <= 0.0) or np.any(
    foam["support_area"] < foam["film_area"]
):
    raise AssertionError("Foam source area/coverage contract is invalid")
foam_repeat, repellent_repeat, _ = source_surface_foam(events, model)
if not np.array_equal(foam, foam_repeat) or not np.array_equal(
    repellents, repellent_repeat
):
    raise AssertionError("Surface source generation is not deterministic")


spec = GridSpec((-0.10, -0.10, -0.10), 0.01, (21, 21, 21))
x, y, z = spec.axes()
phi = np.broadcast_to(y[None, :, None], spec.shape).astype(np.float32).copy()
normal = np.zeros(spec.shape + (3,), dtype=np.float32)
normal[..., 1] = 1.0
velocity = np.zeros_like(normal)
velocity[..., 0] = 0.20
fields = {
    "phi": phi,
    "normal": normal,
    "velocity": velocity,
    "collision_sdf": np.ones(spec.shape, dtype=np.float32),
    "dynamic_collision_sdf": np.ones(spec.shape, dtype=np.float32),
}


# Exact exponential drainage must be invariant to timestep partition.
single_foam = foam[[0]].copy()
single_repellent = np.empty(0, dtype=REPELLENT_DTYPE)
single_foam, _, single_metrics = advance_surface_layer(
    single_foam, single_repellent, 0.0, 0.10, fields, fields, spec, model
)
split_foam = foam[[0]].copy()
split_foam, _, split_metrics_a = advance_surface_layer(
    split_foam, single_repellent, 0.0, 0.05, fields, fields, spec, model
)
split_foam, _, split_metrics_b = advance_surface_layer(
    split_foam, single_repellent, 0.05, 0.05, fields, fields, spec, model
)
if not np.allclose(single_foam["film_area"], split_foam["film_area"], rtol=2.0e-7):
    raise AssertionError("Foam drainage depends on timestep partition")
if not np.allclose(single_foam["position"], split_foam["position"], atol=2.0e-5):
    raise AssertionError("Surface advection does not converge under timestep split")
drained_split = (
    split_metrics_a["drained_film_area_m2"]
    + split_metrics_b["drained_film_area_m2"]
)
initial_film = float(foam["film_area"][0])
if not np.isclose(
    initial_film,
    float(single_foam["film_area"][0]) + single_metrics["drained_film_area_m2"],
    rtol=2.0e-7,
):
    raise AssertionError("Single-step foam film area is not balanced")
if not np.isclose(
    initial_film,
    float(split_foam["film_area"][0]) + drained_split,
    rtol=2.0e-7,
):
    raise AssertionError("Split-step foam film area is not balanced")


# Surface support area follows the tangential divergence, independently of
# film drainage.
expanding_velocity = np.zeros_like(normal)
expanding_velocity[..., 0] = 0.7 * x[:, None, None]
expanding_velocity[..., 2] = 0.3 * z[None, None, :]
expanding_fields = dict(fields)
expanding_fields["velocity"] = expanding_velocity
expanding = foam[[0]].copy()
old_support = float(expanding["support_area"][0])
expanding, _, _ = advance_surface_layer(
    expanding,
    single_repellent,
    0.0,
    0.10,
    expanding_fields,
    expanding_fields,
    spec,
    model,
)
expected_support = old_support * np.exp((0.7 + 0.3) * 0.10)
if not np.isclose(expanding["support_area"][0], expected_support, rtol=3.0e-5):
    raise AssertionError("Foam support area did not follow surface divergence")


# Losing the tracked interface consumes the remaining film into an explicit
# topology sink instead of snapping the parcel to an unrelated surface.
lost_fields = dict(fields)
lost_fields["phi"] = np.full(spec.shape, 0.2, dtype=np.float32)
lost = foam[[0]].copy()
lost, _, lost_metrics = advance_surface_layer(
    lost, single_repellent, 0.0, 0.01, lost_fields, lost_fields, spec, model
)
if len(lost) or lost_metrics["topology_lost_foam"] != 1:
    raise AssertionError("Foam topology loss did not remove the parcel")
if lost_metrics["topology_lost_film_area_m2"] <= 0.0:
    raise AssertionError("Foam topology sink did not record remaining area")


# Repellent lifetime is independent from foam drainage.
short_repellent = repellents.copy()
short_repellent["lifetime"] = 0.005
_, short_repellent, _ = advance_surface_layer(
    np.empty(0, dtype=FOAM_DTYPE),
    short_repellent,
    0.0,
    0.01,
    fields,
    fields,
    spec,
    model,
)
if len(short_repellent):
    raise AssertionError("Expired burst repellent remained active")


report = {
    "valid": True,
    "foam_sources": source_metrics["foam_sources"],
    "repellent_sources": source_metrics["repellent_sources"],
    "low_weber_rejections": source_metrics["rejected_low_weber"],
    "initial_film_area_m2": initial_film,
    "drained_area_m2": single_metrics["drained_film_area_m2"],
    "surface_displacement_m": float(
        np.linalg.norm(single_foam["position"][0] - foam["position"][0])
    ),
    "expanded_support_area_m2": float(expanding["support_area"][0]),
    "topology_lost_area_m2": lost_metrics["topology_lost_film_area_m2"],
}
print(json.dumps(report, indent=2))
