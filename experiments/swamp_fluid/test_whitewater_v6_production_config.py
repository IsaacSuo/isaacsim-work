"""Strict gates for artist-assisted production configuration."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from whitewater.production_config import ProductionConfig


root = Path(__file__).resolve().parent
formal_path = root / "configs" / "swamp_plateau_artist_setup_v2.production.json"
formal = ProductionConfig.load(formal_path)
formal_metadata_a = formal.resolved_metadata()
formal_metadata_b = formal.resolved_metadata()
formal_command = formal.splashsurf_command(formal.paths["output_root"] / "test_surface")
render_command = formal.render_command()

base_payload = json.loads(formal_path.read_text(encoding="utf-8"))
for name, path in formal.paths.items():
    base_payload["paths"][name] = str(path)

rejected = set()
with tempfile.TemporaryDirectory(prefix="wwv6_production_config_") as temporary:
    temporary = Path(temporary)

    def expect_rejected(name, mutate, exception=(ValueError, FileNotFoundError)):
        payload = copy.deepcopy(base_payload)
        mutate(payload)
        path = temporary / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            ProductionConfig.load(path)
        except exception:
            rejected.add(name)

    expect_rejected(
        "missing_manual_confirmation",
        lambda payload: payload["manual_setup"].__setitem__("shoreline_reviewed", False),
    )
    expect_rejected(
        "particle_spacing_mismatch",
        lambda payload: payload["physical"].__setitem__("particle_spacing_m", 0.01),
    )
    expect_rejected(
        "missing_reference_ply",
        lambda payload: (
            payload["splashsurf"].__setitem__("shoreline_mode", "reference-ply"),
            payload["splashsurf"].__setitem__(
                "shoreline_reference_ply", str(temporary / "missing.ply")
            ),
        ),
    )
    expect_rejected(
        "unknown_field",
        lambda payload: payload.__setitem__("per_frame_artist_fix", True),
    )
    expect_rejected(
        "fractional_worker_count",
        lambda payload: payload["splashsurf"].__setitem__("workers", 1.5),
    )
    expect_rejected(
        "dynamic_shoreline",
        lambda payload: payload["splashsurf"].__setitem__("shoreline_mode", "dynamic"),
    )

configured_seed = tuple(formal.whitewater["water_body_seed_xz"])
impact_x_index = formal_command.index("--impact-x") + 1
impact_z_index = formal_command.index("--impact-z") + 1
source_text = (root / "whitewater" / "production_config.py").read_text(encoding="utf-8")
locked_surface = json.loads(
    (
        root.parent.parent
        / "output"
        / "swamp_fluid_preview"
        / "whitewater_v6_splashsurf_phase6_auto_domain_s120_v3_hysteretic"
        / "splashsurf_manifest.json"
    ).read_text(encoding="utf-8")
)["configuration"]

criteria = {
    "formal_swamp_configuration_resolves": formal.name == "swamp_plateau_artist_setup_v2",
    "manual_confirmation_is_mandatory": "missing_manual_confirmation" in rejected,
    "spacing_mismatch_is_rejected": "particle_spacing_mismatch" in rejected,
    "missing_reference_ply_is_rejected": "missing_reference_ply" in rejected,
    "unknown_fields_are_rejected": "unknown_field" in rejected,
    "integer_fields_do_not_truncate": "fractional_worker_count" in rejected,
    "dynamic_per_frame_shoreline_is_rejected": "dynamic_shoreline" in rejected,
    "resolved_hash_is_stable": (
        formal_metadata_a["configuration_sha256"]
        == formal_metadata_b["configuration_sha256"]
    ),
    "command_uses_configured_seed": (
        float(formal_command[impact_x_index]) == configured_seed[0]
        and float(formal_command[impact_z_index]) == configured_seed[1]
    ),
    "render_command_uses_explicit_camera": (
        render_command[render_command.index("--camera") + 1] == "custom"
        and "--camera-eye" in render_command
        and "--camera-target" in render_command
    ),
    "render_command_uses_source_sample_surfaces": (
        render_command[render_command.index("--surface-index-mode") + 1]
        == "source-sample"
    ),
    "command_builder_has_no_swamp_literals": (
        "swamp" not in source_text.lower()
        and "-0.73" not in source_text
        and "-1.380990" not in source_text
    ),
    "surface_recipe_matches_locked_domain_build": all(
        abs(formal.splashsurf[config_name] - locked_surface[manifest_name]) <= 1.0e-14
        for config_name, manifest_name in (
            ("particle_radius_m", "particle_radius"),
            ("smoothing_length", "smoothing_length"),
            ("cube_size", "cube_size"),
            ("surface_threshold", "surface_threshold"),
        )
    ),
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_production_configuration",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "configuration_sha256": formal_metadata_a["configuration_sha256"],
    "rejected_cases": sorted(rejected),
}
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
