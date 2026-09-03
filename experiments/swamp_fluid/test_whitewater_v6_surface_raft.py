"""Synthetic gates for the conservative surface-bubble raft solver."""

from __future__ import annotations

import json

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import PhaseKind
from whitewater.marker_solver import SOLVER_DTYPE
from whitewater.state_machine import WhitewaterState
from whitewater.surface_raft import RAFT_DTYPE, SurfaceRaftModel, build_surface_raft


def plane_fields(spec, support=1.0):
    _, y, _ = spec.axes()
    phi = np.broadcast_to(y[None, :, None], spec.shape).astype(np.float32).copy()
    normal = np.zeros(spec.shape + (3,), dtype=np.float32)
    normal[..., 1] = 1.0
    return {
        "phi": phi,
        "normal": normal,
        "velocity": np.zeros_like(normal),
        "collision_sdf": np.ones(spec.shape, dtype=np.float32),
        "dynamic_collision_sdf": np.ones(spec.shape, dtype=np.float32),
        "render_surface_support": np.full(spec.shape, support, dtype=np.float32),
    }


def markers(rows):
    result = np.zeros(len(rows), dtype=SOLVER_DTYPE)
    for index, (identifier, x, radius, representative) in enumerate(rows):
        result["id"][index] = identifier
        result["state"][index] = np.uint8(WhitewaterState.SURFACE_BUBBLE)
        result["phase"][index] = np.uint8(PhaseKind.GAS)
        result["position"][index] = (x, 0.0, 0.0)
        result["physical_radius"][index] = radius
        result["representative_count"][index] = representative
        result["phase_volume"][index] = (
            representative * (4.0 / 3.0) * np.pi * radius**3
        )
        result["shape"][index] = (1.0, 0.65, 1.0)
        result["random_key"][index] = np.uint64(identifier * 7919)
    return result


spec = GridSpec((-0.05, -0.03, -0.05), 0.002, (51, 31, 51))
fields = plane_fields(spec)
model = SurfaceRaftModel(spec.spacing)

# Two separated bubbles attract, but the real-radius contact constraint keeps
# them from crossing or being replaced by an enlarged visual collision radius.
pair_input = markers(((1, -0.003, 0.0010, 2.0), (2, 0.003, 0.0010, 3.0)))
pair, pair_metrics = build_surface_raft(pair_input, None, 1.0 / 30.0, fields, spec, model)
initial_distance = 0.006
pair_distance = float(np.linalg.norm(pair["raft_position"][1] - pair["raft_position"][0]))
if not 0.002 - 2.0e-6 <= pair_distance < initial_distance:
    raise AssertionError("Two-bubble capillary/contact constraint failed")

# Inverse-radius mobility makes the small bubble move farther than the large
# bubble under the same pair constraint.
unequal_input = markers(((11, -0.003, 0.0005, 1.0), (12, 0.003, 0.0020, 1.0)))
unequal, _ = build_surface_raft(unequal_input, None, 1.0 / 30.0, fields, spec, model)
unequal_motion = np.linalg.norm(
    unequal["raft_position"].astype(np.float64) - unequal["raw_anchor_position"], axis=1
)
if not unequal_motion[0] > unequal_motion[1]:
    raise AssertionError("Small-bubble mobility is not greater than large-bubble mobility")

# Exact source fields and row identity must survive the derived layer bitwise.
if not np.array_equal(pair["marker_id"], pair_input["id"]):
    raise AssertionError("Raft changed marker identity")
for source, derived in (
    ("physical_radius", "physical_radius"),
    ("representative_count", "representative_count"),
    ("phase_volume", "phase_volume"),
    ("shape", "shape"),
    ("random_key", "random_key"),
):
    if not np.array_equal(pair[source], pair_input[derived]):
        raise AssertionError(f"Raft changed conserved field {source}")
if pair_metrics["gas_volume_residual_m3"] != 0.0:
    raise AssertionError("Raft gas volume is not exactly conserved")

# A dense three-bubble row exercises the sequential nonpenetration stage; the
# averaged attraction solve alone is intentionally not trusted for this gate.
dense_input = markers(
    ((5, -0.001, 0.001, 1.0), (6, 0.0, 0.001, 1.0), (7, 0.001, 0.001, 1.0))
)
dense, dense_metrics = build_surface_raft(
    dense_input, None, 1.0 / 30.0, fields, spec, model
)
if dense_metrics["excess_compression_pairs"]:
    raise AssertionError("Sequential contact exceeded the deformable-bubble cap")

# A transported prior offset is inherited by ID, then constrained by the hard
# 12 mm tether even if the previous cache was adversarial.
previous = np.zeros(1, dtype=RAFT_DTYPE)
previous["marker_id"] = 21
previous["raw_anchor_position"] = (0.0, 0.0, 0.0)
previous["raft_position"] = (0.030, 0.0, 0.0)
tether_input = markers(((21, 0.001, 0.001, 1.0),))
tethered, tether_metrics = build_surface_raft(
    tether_input, previous, 1.0 / 30.0, fields, spec, model
)
if float(tethered["anchor_displacement"][0]) > 0.0120001:
    raise AssertionError("Hard anchor tether exceeded 12 mm")
if tether_metrics["inherited"] != 1:
    raise AssertionError("Cross-frame marker-ID inheritance failed")

# Unsupported displaced candidates fall back to their audited anchors rather
# than being deleted. Here only the right half of the plane is supported.
support_fields = plane_fields(spec)
x, _, _ = spec.axes()
support_fields["render_surface_support"] = np.broadcast_to(
    (x[:, None, None] <= 0.002).astype(np.float32), spec.shape
).copy()
previous_bad = np.zeros(1, dtype=RAFT_DTYPE)
previous_bad["marker_id"] = 31
previous_bad["raw_anchor_position"] = (0.0, 0.0, 0.0)
previous_bad["raft_position"] = (0.008, 0.0, 0.0)
fallback_input = markers(((31, 0.0, 0.001, 1.0),))
fallback, fallback_metrics = build_surface_raft(
    fallback_input, previous_bad, 1.0 / 30.0, support_fields, spec, model
)
if len(fallback) != 1 or fallback["support_fallback"][0] != 1:
    raise AssertionError("Unsupported candidate was not retained at its anchor")
if not np.allclose(fallback["raft_position"][0], fallback_input["position"][0]):
    raise AssertionError("Support fallback did not restore the audited anchor")
if fallback_metrics["support_bad"]:
    raise AssertionError("Support fallback left a bad final support sample")

# Exponential per-frame strengths should give a closely converged result under
# timestep partition and additional solver iterations.
stable_input = markers(((41, -0.003, 0.001, 1.0), (42, 0.003, 0.001, 1.0)))
one_step, _ = build_surface_raft(stable_input, None, 1.0 / 30.0, fields, spec, model)
half_step, _ = build_surface_raft(stable_input, None, 1.0 / 60.0, fields, spec, model)
half_step, _ = build_surface_raft(stable_input, half_step, 1.0 / 60.0, fields, spec, model)
if not np.allclose(one_step["raft_position"], half_step["raft_position"], atol=8.0e-5):
    raise AssertionError("Raft result is excessively timestep dependent")
more_iterations = SurfaceRaftModel(spec.spacing, solver_iterations=24)
iterated, _ = build_surface_raft(
    stable_input, None, 1.0 / 30.0, fields, spec, more_iterations
)
if not np.allclose(one_step["raft_position"], iterated["raft_position"], atol=8.0e-5):
    raise AssertionError("Raft result is excessively iteration dependent")

report = {
    "valid": True,
    "pair_initial_distance_m": initial_distance,
    "pair_final_distance_m": pair_distance,
    "small_bubble_motion_m": float(unequal_motion[0]),
    "large_bubble_motion_m": float(unequal_motion[1]),
    "maximum_tether_displacement_m": float(tethered["anchor_displacement"][0]),
    "support_fallbacks": fallback_metrics["support_fallbacks"],
    "gas_volume_residual_m3": pair_metrics["gas_volume_residual_m3"],
    "dense_maximum_contact_compression": dense_metrics[
        "maximum_contact_compression_fraction"
    ],
    "timestep_partition_delta_m": float(
        np.max(np.linalg.norm(one_step["raft_position"] - half_step["raft_position"], axis=1))
    ),
}
print(json.dumps(report, indent=2))
