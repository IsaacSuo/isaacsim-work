"""Summarize the five artist decisions and block unsafe production setup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from whitewater.production_config import ProductionConfig


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("configuration", type=Path)
parser.add_argument("output_report", type=Path)
args = parser.parse_args()

configuration = ProductionConfig.load(args.configuration)
scene = configuration.scene_contract
terrain = scene.terrain
selection = terrain.parameters["selection"]
selection_mesh = selection["selected_mesh"]
domain = json.loads(
    configuration.paths["domain_manifest"].read_text(encoding="utf-8")
)
body = next(
    item for item in domain["bodies"]
    if item["body_id"] == configuration.whitewater["water_body_id"]
)
source = json.loads(
    configuration.paths["source_manifest"].read_text(encoding="utf-8")
)
source_rows = {
    int(item["sample_index"]): item for item in source["samples"]
}
selected_samples = configuration.whitewater["source_samples"]
colliders = [
    {
        "id": collider.identifier,
        "shape": collider.shape,
        "roles": list(collider.roles),
        "motion": collider.motion.kind,
    }
    for collider in scene.colliders
]
churn_sources = [
    collider["id"] for collider in colliders if "churn_source" in collider["roles"]
]
sample_times = [float(source_rows[sample]["simulation_time"]) for sample in selected_samples]

steps = {
    "1_terrain": {
        "confirmed": configuration.manual_setup["terrain_selected"],
        "terrain_id": terrain.identifier,
        "representation": terrain.representation,
        "mesh": terrain.parameters["path"],
        "triangle_count": int(selection_mesh["triangle_count"]),
        "connected_components": int(selection_mesh["connected_component_count"]),
        "normal_convention": terrain.parameters["normal_convention"],
        "selection_record": terrain.parameters["selection_path"],
    },
    "2_water_body_domain": {
        "confirmed": configuration.manual_setup["water_body_domain_reviewed"],
        "water_body_id": body["body_id"],
        "reference_bounds_minimum_m": body["reference_bounds_minimum"],
        "reference_bounds_maximum_m": body["reference_bounds_maximum"],
        "domain_origin_m": body["domain"]["origin"],
        "domain_maximum_m": body["domain"]["maximum"],
        "particle_count": int(body["particle_count"]),
        "particle_spacing_m": configuration.physical["particle_spacing_m"],
        "water_level_m": configuration.physical["water_level_m"],
        "selected_source_samples": list(selected_samples),
        "selected_time_range_s": [min(sample_times), max(sample_times)],
    },
    "3_collider_roles": {
        "confirmed": configuration.manual_setup["collider_roles_reviewed"],
        "colliders": colliders,
        "churn_sources": churn_sources,
        "profile_policy": configuration.flow_profile.churn_source_policy,
    },
    "4_shoreline": {
        "confirmed": configuration.manual_setup["shoreline_reviewed"],
        "mode": configuration.splashsurf["shoreline_mode"],
        "reference_ply": configuration.splashsurf["shoreline_reference_ply"],
        "minimum_particle_layers": configuration.splashsurf["minimum_particle_layers"],
        "erosion_cells": configuration.splashsurf["shoreline_erosion_cells"],
        "temporally_fixed": configuration.splashsurf["shoreline_mode"]
        in {"terrain", "reference-ply"},
    },
    "5_camera_and_render": {
        "confirmed": configuration.manual_setup["camera_reviewed"],
        **configuration.render,
        "blender_scene": str(configuration.paths["blender_scene"]),
        "hdri": str(configuration.paths["hdri"]),
    },
}
criteria = {
    "all_five_artist_decisions_are_confirmed": all(
        item["confirmed"] for item in steps.values()
    ),
    "terrain_is_selected_open_surface": (
        terrain.representation == "open_triangle_mesh"
        and int(selection_mesh["triangle_count"]) > 0
    ),
    "water_body_has_particles_and_classified_samples": (
        int(body["particle_count"]) > 0 and len(selected_samples) > 0
    ),
    "required_churn_source_is_tagged": (
        configuration.flow_profile.churn_source_policy != "required_tagged_source"
        or bool(churn_sources)
    ),
    "shoreline_is_temporally_fixed": steps["4_shoreline"]["temporally_fixed"],
    "camera_and_render_are_explicitly_reviewed": steps["5_camera_and_render"]["confirmed"],
}
report = {
    "schema": 1,
    "product": "whitewater_v6_artist_setup_preflight",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": all(criteria.values()),
    "configuration": str(configuration.path),
    "configuration_sha256": sha256_file(configuration.path),
    "resolved_configuration_sha256": configuration.resolved_metadata()[
        "configuration_sha256"
    ],
    "criteria": criteria,
    "artist_setup": steps,
    "notes": configuration.manual_setup["notes"],
    "next_action": (
        "Resolve the configuration and run the fingerprinted DAG."
        if all(criteria.values())
        else "Review the failed artist setup items before any simulation or render."
    ),
}
write_once(args.output_report, report)
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
