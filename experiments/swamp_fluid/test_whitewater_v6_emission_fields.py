"""Synthetic validation of scale-aware v6 whitewater emission channels."""

from __future__ import annotations

import json

import numpy as np

from whitewater.emission_fields import (
    CHANNEL_NAMES,
    EmissionModel,
    TemporalEmissionFilter,
    estimate_precontact_floors,
    raw_emission_fields,
    suppress_precontact_floor,
)


shape = (17, 17, 17)
spacing = 0.016
model = EmissionModel(spacing)
axis = (np.arange(shape[0]) - shape[0] // 2) * spacing
x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
phi = y.copy()
normal = np.zeros(shape + (3,), dtype=np.float32)
normal[..., 1] = 1.0


def base_fields():
    return {
        "phi": phi.copy(),
        "normal": normal.copy(),
        "velocity": np.zeros(shape + (3,), dtype=np.float32),
        "acceleration": np.zeros(shape + (3,), dtype=np.float32),
        "curvature": np.zeros(shape, dtype=np.float32),
        "divergence": np.zeros(shape, dtype=np.float32),
        "vorticity": np.zeros(shape + (3,), dtype=np.float32),
        "strain_rate": np.zeros(shape, dtype=np.float32),
        "collision_sdf": np.full(shape, 1.0, dtype=np.float32),
        "sphere_collision_sdf": np.full(shape, 1.0, dtype=np.float32),
        "surface_valid": np.ones(shape, dtype=np.uint8),
        "velocity_valid": np.ones(shape, dtype=np.uint8),
        "acceleration_valid": np.ones(shape, dtype=np.uint8),
    }


surface = np.abs(phi) <= 0.5 * spacing
feature_band = (
    np.abs(phi) <= 2.5 * model.surface_half_width_cells * spacing
)
flat = raw_emission_fields(base_fields(), model)

crest_fields = base_fields()
crest_fields["curvature"][feature_band] = 0.35 / spacing
crest_fields["velocity"][feature_band, 1] = 0.45 * model.characteristic_velocity
crest_fields["acceleration"][feature_band, 1] = 2.5 * model.gravity
crest = raw_emission_fields(crest_fields, model)

entrainment_fields = base_fields()
entrainment_fields["velocity"][feature_band, 1] = -0.35 * model.characteristic_velocity
entrainment_fields["divergence"][feature_band] = -0.16 / model.characteristic_time
entrainment_fields["vorticity"][feature_band, 0] = 0.22 / model.characteristic_time
entrainment = raw_emission_fields(entrainment_fields, model)

churn_fields = base_fields()
churn_fields["acceleration"][..., 1] = -5.0 * model.gravity
churn_fields["divergence"][...] = -0.18 / model.characteristic_time
churn_fields["vorticity"][..., 2] = 0.20 / model.characteristic_time
churn_fields["sphere_collision_sdf"] = np.sqrt(x * x + y * y + z * z) - 0.04
churn = raw_emission_fields(churn_fields, model)
canonical_churn_fields = dict(churn_fields)
canonical_churn_fields["churn_source_sdf"] = canonical_churn_fields.pop(
    "sphere_collision_sdf"
)
canonical_churn = raw_emission_fields(canonical_churn_fields, model)
no_churn_source_fields = dict(churn_fields)
no_churn_source_fields.pop("sphere_collision_sdf")
no_churn_source = raw_emission_fields(no_churn_source_fields, model)

precontact_frames = []
for amplitude in (0.002, 0.004, 0.006):
    noise_fields = base_fields()
    noise_fields["curvature"][surface] = amplitude / spacing
    noise_fields["divergence"][surface] = -amplitude / model.characteristic_time
    precontact_frames.append(raw_emission_fields(noise_fields, model))
floors = estimate_precontact_floors(precontact_frames)
suppressed_noise = {
    channel: suppress_precontact_floor(
        precontact_frames[-1][channel + "_raw"], floors[channel]
    )
    for channel in CHANNEL_NAMES
}


def step_response(dt, duration):
    temporal = TemporalEmissionFilter(shape)
    target = {channel: np.ones(shape, dtype=np.float32) for channel in CHANNEL_NAMES}
    temporal.advance(target, dt, event_enabled=False)
    steps = int(round(duration / dt))
    output = None
    for _ in range(steps):
        output = temporal.advance(target, dt, event_enabled=True)
    return output


response_120 = step_response(1.0 / 120.0, 0.1)
response_240 = step_response(1.0 / 240.0, 0.1)
gate_filter = TemporalEmissionFilter(shape)
ones = {channel: np.ones(shape, dtype=np.float32) for channel in CHANNEL_NAMES}
gated = gate_filter.advance(ones, 1.0 / 120.0, event_enabled=False)

center = tuple(size // 2 for size in shape)
metrics = {
    "flat_channel_maximum": {
        channel: float(np.max(flat[channel + "_raw"])) for channel in CHANNEL_NAMES
    },
    "crest_center": {
        channel: float(crest[channel + "_raw"][center]) for channel in CHANNEL_NAMES
    },
    "entrainment_center": {
        channel: float(entrainment[channel + "_raw"][center])
        for channel in CHANNEL_NAMES
    },
    "churn_center": {
        channel: float(churn[channel + "_raw"][center]) for channel in CHANNEL_NAMES
    },
    "canonical_churn_error_max": float(
        np.max(np.abs(churn["churn_raw"] - canonical_churn["churn_raw"]))
    ),
    "no_churn_source_maximum": float(np.max(no_churn_source["churn_raw"])),
    "precontact_floors": floors,
    "suppressed_noise_maximum": {
        channel: float(np.max(suppressed_noise[channel])) for channel in CHANNEL_NAMES
    },
    "contact_gate_maximum": {
        channel: float(np.max(gated[channel])) for channel in CHANNEL_NAMES
    },
    "timestep_response_difference": {
        channel: float(np.max(np.abs(response_120[channel] - response_240[channel])))
        for channel in CHANNEL_NAMES
    },
}
all_outputs = [
    *flat.values(),
    *crest.values(),
    *entrainment.values(),
    *churn.values(),
    *canonical_churn.values(),
    *no_churn_source.values(),
    *suppressed_noise.values(),
    *response_120.values(),
    *response_240.values(),
]
criteria = {
    "all_outputs_finite_and_bounded": all(
        np.isfinite(value).all() and np.min(value) >= 0.0 and np.max(value) <= 1.0
        for value in all_outputs
    ),
    "static_flat_surface_is_quiet": max(metrics["flat_channel_maximum"].values()) == 0.0,
    "crest_selects_spray": (
        metrics["crest_center"]["spray"] >= 0.10
        and metrics["crest_center"]["entrained_air"] == 0.0
        and metrics["crest_center"]["churn"] == 0.0
    ),
    "compression_selects_entrained_air": (
        metrics["entrainment_center"]["entrained_air"] >= 0.10
        and metrics["entrainment_center"]["spray"] == 0.0
    ),
    "near_tagged_collider_impulse_selects_churn": metrics["churn_center"]["churn"] >= 0.10,
    "canonical_churn_source_matches_legacy_alias": metrics[
        "canonical_churn_error_max"
    ]
    == 0.0,
    "missing_churn_source_does_not_treat_walls_as_impactors": metrics[
        "no_churn_source_maximum"
    ]
    == 0.0,
    "precontact_noise_is_suppressed": max(
        metrics["suppressed_noise_maximum"].values()
    ) <= 1.0e-7,
    "contact_gate_is_exact_zero": max(metrics["contact_gate_maximum"].values()) == 0.0,
    "temporal_filter_is_timestep_invariant": max(
        metrics["timestep_response_difference"].values()
    ) <= 2.0e-6,
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_emission_fields",
    "valid": bool(all(criteria.values())),
    "criteria": criteria,
    "metrics": metrics,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
