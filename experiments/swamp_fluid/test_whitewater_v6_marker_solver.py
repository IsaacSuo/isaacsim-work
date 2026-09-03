"""Synthetic gates for persistent v6 marker dynamics and terminal events."""

from __future__ import annotations

import json

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import BIRTH_DTYPE, BirthChannel, PhaseKind
from whitewater.marker_solver import (
    EVENT_DTYPE,
    SOLVER_DTYPE,
    SUPPORT_LOSS_DTYPE,
    MarkerSolverModel,
    NonterminalTransitionKind,
    TerminalEventKind,
    advance_marker_interval,
    births_to_solver_markers,
    sample_scalar_trilinear,
)
from whitewater.state_machine import WhitewaterState


spec = GridSpec((-0.10, -0.10, -0.10), 0.01, (21, 21, 21))
x, y, z = spec.axes()
phi = np.broadcast_to(y[None, :, None], spec.shape).astype(np.float32).copy()
normal = np.zeros(spec.shape + (3,), dtype=np.float32)
normal[..., 1] = 1.0
velocity = np.zeros_like(normal)
velocity[..., 0] = 0.30
collision = np.ones(spec.shape, dtype=np.float32)
fields = {
    "phi": phi,
    "normal": normal,
    "velocity": velocity,
    "collision_sdf": collision,
}
model = MarkerSolverModel(spec.spacing)


def make_birth(marker_id, channel, position, marker_velocity, radius=0.001):
    birth = np.zeros(1, dtype=BIRTH_DTYPE)
    birth["id"] = np.uint64(marker_id)
    birth["channel"] = np.uint8(channel)
    birth["phase"] = np.uint8(
        PhaseKind.LIQUID if channel == BirthChannel.SPRAY else PhaseKind.GAS
    )
    birth["position"] = position
    birth["velocity"] = marker_velocity
    birth["physical_radius"] = radius
    birth["representative_count"] = 1.0
    birth["phase_volume"] = (4.0 / 3.0) * np.pi * radius**3
    birth["random_key"] = np.uint64(marker_id * 971)
    return birth


# Birth conversion must preserve provenance and phase volume exactly.
spray_birth = make_birth(
    101, BirthChannel.SPRAY, (0.0, 0.025, 0.0), (0.2, 0.5, 0.0), 0.0005
)
spray = births_to_solver_markers(spray_birth)
if spray.dtype != SOLVER_DTYPE or spray["state"][0] != WhitewaterState.SPRAY:
    raise AssertionError("Spray birth did not enter the spray state")
spray_volume = float(spray["phase_volume"][0])
spray_after, spray_events, _, spray_metrics = advance_marker_interval(
    spray, 0.0, 0.002, fields, fields, spec, model
)
if len(spray_after) != 1 or len(spray_events):
    raise AssertionError("Free spray was terminated unexpectedly")
if spray_after["velocity"][0, 1] >= spray["velocity"][0, 1]:
    raise AssertionError("Spray gravity/drag did not reduce vertical velocity")
if float(spray_after["phase_volume"][0]) != spray_volume:
    raise AssertionError("Spray dynamics changed represented liquid volume")


# An entrained bubble must relax toward carrier flow plus buoyant rise while
# remaining in the liquid when it starts sufficiently deep.
bubble_birth = make_birth(
    202, BirthChannel.ENTRAINED_AIR, (0.0, -0.060, 0.0), (0.0, 0.0, 0.0), 0.001
)
bubble = births_to_solver_markers(bubble_birth)
bubble_after, bubble_events, _, bubble_metrics = advance_marker_interval(
    bubble, 0.0, 0.01, fields, fields, spec, model
)
if len(bubble_after) != 1 or len(bubble_events):
    raise AssertionError("Deep bubble was terminated unexpectedly")
if bubble_after["position"][0, 0] <= 0.0 or bubble_after["position"][0, 1] <= -0.060:
    raise AssertionError("Bubble did not follow carrier flow and buoyancy")
if bubble_after["state"][0] != WhitewaterState.ENTRAINED_BUBBLE:
    raise AssertionError("Deep bubble transitioned to the surface too early")


# Timestep refinement should converge tightly for a uniform carrier field.
single = births_to_solver_markers(bubble_birth)
single, _, _, _ = advance_marker_interval(single, 0.0, 0.01, fields, fields, spec, model)
split = births_to_solver_markers(bubble_birth)
split, _, _, _ = advance_marker_interval(split, 0.0, 0.005, fields, fields, spec, model)
split, _, _, _ = advance_marker_interval(split, 0.005, 0.005, fields, fields, spec, model)
if not np.allclose(single["velocity"], split["velocity"], atol=2.0e-7):
    raise AssertionError("Bubble carrier relaxation is timestep dependent")
if not np.allclose(single["position"], split["position"], atol=2.0e-4):
    raise AssertionError("Bubble advection does not converge under timestep refinement")


# Spray crossing into the liquid must produce one conservative re-entry event.
reentry_birth = make_birth(
    303, BirthChannel.SPRAY, (0.0, 0.008, 0.0), (0.0, -2.0, 0.0), 0.0007
)
reentry = births_to_solver_markers(reentry_birth)
reentry_after, reentry_events, _, _ = advance_marker_interval(
    reentry, 0.0, 0.01, fields, fields, spec, model
)
if len(reentry_after) or len(reentry_events) != 1:
    raise AssertionError("Spray re-entry did not terminate exactly one marker")
if reentry_events.dtype != EVENT_DTYPE or reentry_events["kind"][0] != TerminalEventKind.SPRAY_REENTRY:
    raise AssertionError("Spray re-entry produced the wrong terminal event")
if reentry_events["phase_volume"][0] != reentry_birth["phase_volume"][0]:
    raise AssertionError("Spray re-entry event changed liquid volume")


# A near-surface bubble transitions without losing gas, then bursts into one
# explicit event after its deterministic surface residence time.
surface_birth = make_birth(
    404, BirthChannel.CHURN, (0.0, -0.0002, 0.0), (0.0, 0.1, 0.0), 0.0015
)
surface = births_to_solver_markers(surface_birth)
surface, transition_events, _, transition_metrics = advance_marker_interval(
    surface, 0.0, 0.002, fields, fields, spec, model
)
if len(transition_events) or len(surface) != 1:
    raise AssertionError("Surface transition incorrectly created a terminal event")
if surface["state"][0] != WhitewaterState.SURFACE_BUBBLE:
    raise AssertionError("Near-surface bubble did not enter the surface state")
if surface["phase_volume"][0] != surface_birth["phase_volume"][0]:
    raise AssertionError("Surface transition changed represented gas volume")

# Losing the production render-surface support is an explicit, conservative
# re-entrainment transition rather than a fake burst or marker deletion.
supported_fields = dict(fields)
supported_fields["render_surface_support"] = np.ones(spec.shape, dtype=np.uint8)
unsupported_fields = dict(fields)
unsupported_fields["render_surface_support"] = np.zeros(spec.shape, dtype=np.uint8)
reentrained = births_to_solver_markers(surface_birth)
reentrained, _, _, _ = advance_marker_interval(
    reentrained, 0.0, 0.002, supported_fields, supported_fields, spec, model
)
reentrained, false_bursts, support_losses, support_metrics = advance_marker_interval(
    reentrained, 0.002, 0.001, unsupported_fields, unsupported_fields, spec, model
)
if len(false_bursts) or len(reentrained) != 1:
    raise AssertionError("Support loss deleted a bubble or created a fake burst")
if reentrained["state"][0] != WhitewaterState.ENTRAINED_BUBBLE:
    raise AssertionError("Unsupported surface bubble was not re-entrained")
if support_losses.dtype != SUPPORT_LOSS_DTYPE or len(support_losses) != 1:
    raise AssertionError("Support loss did not produce one auditable transition")
if support_losses["kind"][0] != NonterminalTransitionKind.SURFACE_SUPPORT_LOSS_REENTRAINMENT:
    raise AssertionError("Support loss transition has the wrong kind")
if support_losses["phase_volume"][0] != surface_birth["phase_volume"][0]:
    raise AssertionError("Re-entrainment changed represented gas volume")
if support_metrics["surface_support_loss_count"] != 1:
    raise AssertionError("Support loss metric disagrees with the transition record")

surface["state_age"] = 1.0
surface, burst_events, _, _ = advance_marker_interval(
    surface, 0.002, 0.001, fields, fields, spec, model
)
if len(surface) or len(burst_events) != 1:
    raise AssertionError("Expired surface bubble did not burst exactly once")
if burst_events["kind"][0] != TerminalEventKind.BUBBLE_BURST:
    raise AssertionError("Surface bubble produced the wrong terminal event")
if burst_events["phase_volume"][0] != surface_birth["phase_volume"][0]:
    raise AssertionError("Bubble burst event changed represented gas volume")


# A marker crossing a planar solid must be projected to physical-radius
# clearance and retain a non-inward normal velocity.
wall_collision = np.broadcast_to(
    x[:, None, None] + 0.02, spec.shape
).astype(np.float32).copy()
wall_fields = dict(fields)
wall_fields["collision_sdf"] = wall_collision
wall_birth = make_birth(
    505, BirthChannel.SPRAY, (-0.005, 0.04, 0.0), (-2.0, 0.0, 0.0), 0.001
)
wall = births_to_solver_markers(wall_birth)
wall, wall_events, _, wall_metrics = advance_marker_interval(
    wall, 0.0, 0.01, wall_fields, wall_fields, spec, model
)
if len(wall) != 1 or len(wall_events):
    raise AssertionError("Solid collision incorrectly terminated the marker")
wall_clearance = sample_scalar_trilinear(
    wall_collision, wall["position"], spec
)[0]
required = 0.001 + model.solid_clearance_cells * spec.spacing
if wall_clearance < required - 2.0e-6:
    raise AssertionError("Solid projection did not restore marker clearance")
if wall["velocity"][0, 0] < -1.0e-6:
    raise AssertionError("Solid projection retained inward normal velocity")

# Canonical dynamic-collider projection must remain exactly compatible with
# the historical moving-sphere alias while accepting arbitrary SDF geometry.
canonical_dynamic_fields = dict(wall_fields)
canonical_dynamic_fields["dynamic_collision_sdf"] = wall_collision
legacy_dynamic_fields = dict(wall_fields)
legacy_dynamic_fields["sphere_collision_sdf"] = wall_collision
canonical_dynamic = births_to_solver_markers(wall_birth)
canonical_dynamic, canonical_events, _, canonical_metrics = advance_marker_interval(
    canonical_dynamic,
    0.0,
    0.01,
    canonical_dynamic_fields,
    canonical_dynamic_fields,
    spec,
    model,
)
legacy_dynamic = births_to_solver_markers(wall_birth)
legacy_dynamic, legacy_events, _, legacy_metrics = advance_marker_interval(
    legacy_dynamic,
    0.0,
    0.01,
    legacy_dynamic_fields,
    legacy_dynamic_fields,
    spec,
    model,
)
if len(canonical_events) or len(legacy_events):
    raise AssertionError("Dynamic collision compatibility created terminal events")
if not np.array_equal(canonical_dynamic, legacy_dynamic):
    raise AssertionError("Canonical and legacy dynamic collision projections differ")
if canonical_metrics["solid_projection_count"] != legacy_metrics[
    "solid_projection_count"
]:
    raise AssertionError("Canonical and legacy dynamic collision metrics differ")

# A sparse secondary-scene projection may move spray back into the carrier
# grid. The dense union SDF must be reconciled in the same substep rather than
# leaving a boundary endpoint penetrating another constraint.
boundary_fields = dict(fields)
boundary_fields["phi"] = np.ones(spec.shape, dtype=np.float32)
boundary_fields["collision_sdf"] = np.broadcast_to(
    y[None, :, None] + 0.02, spec.shape
).astype(np.float32).copy()
boundary_birth = make_birth(
    606, BirthChannel.SPRAY, (-0.099, -0.019, 0.0), (-1.0, 0.0, 0.0), 0.001
)


def sparse_return_query(positions, alpha):
    del alpha
    positions = np.asarray(positions)
    normal = np.zeros_like(positions)
    normal[:, 0] = 1.0
    return {
        "distance": positions[:, 0] + 0.095,
        "normal": normal,
        "valid": np.ones(len(positions), dtype=bool),
        "open_boundary_invalid": np.zeros(len(positions), dtype=bool),
    }


boundary_marker = births_to_solver_markers(boundary_birth)
boundary_model = MarkerSolverModel(
    spec.spacing, secondary_domain_padding_cells=2.0
)
boundary_marker, boundary_events, _, boundary_metrics = advance_marker_interval(
    boundary_marker,
    0.0,
    0.005,
    boundary_fields,
    boundary_fields,
    spec,
    boundary_model,
    secondary_collision_query=sparse_return_query,
)
if len(boundary_events) or len(boundary_marker) != 1:
    raise AssertionError("Secondary-to-carrier reconciliation lost the marker")
boundary_clearance = sample_scalar_trilinear(
    boundary_fields["collision_sdf"], boundary_marker["position"], spec
)[0]
if boundary_clearance < required - 2.0e-6:
    raise AssertionError("Secondary-to-carrier endpoint was not reconciled")
if boundary_metrics["secondary_to_carrier_reconciliation_count"] < 1:
    raise AssertionError("Secondary-to-carrier reconciliation was not audited")

# Even an in-grid marker needs the sparse query inside the finite-difference
# guard band, where one side of the dense gradient stencil would be outside.
guard_birth = make_birth(
    607, BirthChannel.SPRAY, (-0.0995, -0.019, 0.0), (0.0, 0.0, 0.0), 0.001
)


def sparse_guard_query(positions, alpha):
    del alpha
    positions = np.asarray(positions)
    normal = np.zeros_like(positions)
    normal[:, 1] = 1.0
    return {
        "distance": positions[:, 1] + 0.02,
        "normal": normal,
        "valid": np.ones(len(positions), dtype=bool),
        "open_boundary_invalid": np.zeros(len(positions), dtype=bool),
    }


guard_marker = births_to_solver_markers(guard_birth)
guard_marker, guard_events, _, guard_metrics = advance_marker_interval(
    guard_marker,
    0.0,
    0.001,
    boundary_fields,
    boundary_fields,
    spec,
    boundary_model,
    secondary_collision_query=sparse_guard_query,
)
guard_clearance = sample_scalar_trilinear(
    boundary_fields["collision_sdf"], guard_marker["position"], spec
)[0]
if len(guard_events) or guard_clearance < required - 2.0e-6:
    raise AssertionError("Gradient-guard sparse collision did not restore clearance")
if guard_metrics["secondary_solid_projection_count"] < 1:
    raise AssertionError("Gradient-guard sparse query was not exercised")


report = {
    "valid": True,
    "spray_substeps": spray_metrics["substeps"],
    "bubble_position_m": bubble_after["position"][0].tolist(),
    "bubble_velocity_mps": bubble_after["velocity"][0].tolist(),
    "spray_reentry_volume_m3": float(reentry_events["phase_volume"][0]),
    "bubble_burst_volume_m3": float(burst_events["phase_volume"][0]),
    "solid_projection_count": wall_metrics["solid_projection_count"],
    "solid_clearance_m": float(wall_clearance),
    "surface_transition_count": transition_metrics["surface_transition_count"],
    "dynamic_collision_legacy_compatible": True,
    "secondary_to_carrier_reconciliation_count": boundary_metrics[
        "secondary_to_carrier_reconciliation_count"
    ],
    "gradient_guard_secondary_projection_count": guard_metrics[
        "secondary_solid_projection_count"
    ],
}
print(json.dumps(report, indent=2))
