"""Validate strict flow profiles and capillary-scale emission normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from whitewater.emission_fields import EmissionModel, raw_emission_fields
from whitewater.flow_profiles import FlowProfile


profile_directory = Path(__file__).parent / "configs" / "flow_profiles"
profiles = {
    path.stem: FlowProfile.load(path)
    for path in sorted(profile_directory.glob("*.json"))
}


def fields_for(model, include_churn_source=False):
    shape = (9, 9, 9)
    spacing = model.spacing
    axis = (np.arange(shape[0]) - shape[0] // 2) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    phi = y.astype(np.float32)
    normal = np.zeros(shape + (3,), dtype=np.float32)
    normal[..., 1] = 1.0
    fields = {
        "phi": phi,
        "normal": normal,
        "velocity": np.zeros(shape + (3,), dtype=np.float32),
        "acceleration": np.zeros(shape + (3,), dtype=np.float32),
        "curvature": np.zeros(shape, dtype=np.float32),
        "divergence": np.zeros(shape, dtype=np.float32),
        "vorticity": np.zeros(shape + (3,), dtype=np.float32),
        "strain_rate": np.zeros(shape, dtype=np.float32),
        "collision_sdf": np.full(shape, 1.0, dtype=np.float32),
        "surface_valid": np.ones(shape, dtype=np.uint8),
        "velocity_valid": np.ones(shape, dtype=np.uint8),
        "acceleration_valid": np.ones(shape, dtype=np.uint8),
    }
    band = np.abs(phi) <= 2.5 * model.surface_half_width_cells * spacing
    fields["curvature"][band] = 180.0
    fields["velocity"][band, 1] = -0.25
    fields["acceleration"][..., 1] = -4.0 * model.gravity
    fields["divergence"][...] = -0.15 / model.characteristic_time
    fields["vorticity"][..., 2] = 0.18 / model.characteristic_time
    if include_churn_source:
        fields["churn_source_sdf"] = np.sqrt(x * x + y * y + z * z) - 0.003
    return fields


impact = profiles["impact"]
fine_model = EmissionModel(
    0.001,
    conditioning_sigma=impact.conditioning_sigma_cells,
    surface_half_width_cells=impact.surface_half_width_cells,
    churn_depth_cells=impact.churn_depth_cells,
    sphere_influence_cells=impact.churn_source_influence_cells,
    profile=impact,
)
finer_model = EmissionModel(
    0.002,
    conditioning_sigma=impact.conditioning_sigma_cells,
    surface_half_width_cells=impact.surface_half_width_cells,
    churn_depth_cells=impact.churn_depth_cells,
    sphere_influence_cells=impact.churn_source_influence_cells,
    profile=impact,
)
fine = raw_emission_fields(fields_for(fine_model, include_churn_source=True), fine_model)
finer = raw_emission_fields(fields_for(finer_model, include_churn_source=True), finer_model)
center = (4, 4, 4)
scale_differences = {
    channel: abs(float(fine[channel + "_raw"][center]) - float(finer[channel + "_raw"][center]))
    for channel in ("spray", "entrained_air", "churn")
}

policy_churn = {}
for name in ("river", "waterfall", "wake"):
    profile = profiles[name]
    model = EmissionModel(
        0.016,
        conditioning_sigma=profile.conditioning_sigma_cells,
        surface_half_width_cells=profile.surface_half_width_cells,
        churn_depth_cells=profile.churn_depth_cells,
        sphere_influence_cells=profile.churn_source_influence_cells,
        profile=profile,
    )
    output = raw_emission_fields(fields_for(model, include_churn_source=False), model)
    policy_churn[name] = float(np.max(output["churn_raw"]))

strict_unknown_rejected = False
invalid = impact.metadata()
invalid["unexpected"] = True
try:
    FlowProfile.from_mapping(invalid)
except ValueError:
    strict_unknown_rejected = True

metrics = {
    "profile_names": sorted(profiles),
    "capillary_length_m": fine_model.capillary_length,
    "fine_characteristic_length_m": fine_model.characteristic_length,
    "finer_characteristic_length_m": finer_model.characteristic_length,
    "subcapillary_scale_differences": scale_differences,
    "missing_source_churn_maximum": policy_churn,
}
criteria = {
    "four_flow_families_are_available": set(profiles) == {"impact", "river", "waterfall", "wake"},
    "profile_schema_is_strict": strict_unknown_rejected,
    "subcapillary_resolution_uses_physical_scale": (
        fine_model.characteristic_length == finer_model.characteristic_length
        and max(scale_differences.values()) <= 1.0e-6
    ),
    "river_disables_churn": policy_churn["river"] == 0.0,
    "waterfall_can_churn_without_tagged_impactor": policy_churn["waterfall"] > 0.0,
    "wake_requires_tagged_hull_source": policy_churn["wake"] == 0.0,
    "profile_birth_rates_are_positive": all(
        min(profile.birth_rate_density.values()) > 0.0 for profile in profiles.values()
    ),
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_flow_profiles",
    "valid": bool(all(criteria.values())),
    "criteria": criteria,
    "metrics": metrics,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
