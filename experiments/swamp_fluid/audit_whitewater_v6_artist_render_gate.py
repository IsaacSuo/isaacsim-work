"""Audit a representative artist-configured Plateau render frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_dimensions(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def write_once(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("resolved_configuration", type=Path)
parser.add_argument("render_directory", type=Path)
parser.add_argument("surface_manifest", type=Path)
parser.add_argument("output_report", type=Path)
parser.add_argument("--source-sample", type=int, required=True)
args = parser.parse_args()

resolved_path = args.resolved_configuration.resolve()
render_directory = args.render_directory.resolve()
surface_manifest_path = args.surface_manifest.resolve()
resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
render_manifest_path = render_directory / "render_manifest.json"
render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
surface_manifest = json.loads(surface_manifest_path.read_text(encoding="utf-8"))
frames = [
    frame for frame in render_manifest["frames"]
    if int(frame["source_sample_index"]) == args.source_sample
]
if len(frames) != 1:
    raise ValueError("Representative source sample must occur exactly once")
frame = frames[0]
surface_frame = next(
    item for item in surface_manifest["state"]["frames"]
    if int(item["frame_id"]) == args.source_sample
)
png = Path(frame["png"])
water_obj = Path(frame["water_obj"])
plateau_npz = Path(
    resolved["paths"]["plateau_directory"]["path"]
) / next(
    item["file"]
    for item in json.loads(
        (Path(resolved["paths"]["plateau_directory"]["path"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )["samples"]
    if int(item["source_sample_index"]) == args.source_sample
)
with np.load(plateau_npz, allow_pickle=False) as cache:
    gas_volume = float(cache["cells"]["gas_volume"].sum(dtype=np.float64))
lod = frame["lod"]
lod_gas_volume = sum(
    float(lod[name])
    for name in ("micro_gas_volume_m3", "cluster_gas_volume_m3", "hero_gas_volume_m3")
)
configuration = render_manifest["configuration"]
expected = resolved["render"]
criteria = {
    "render_completed": render_manifest.get("complete") is True and len(frames) == 1,
    "explicit_camera_pose_matches_configuration": (
        configuration.get("camera") == "custom"
        and np.allclose(
            configuration.get("camera_eye_isaac_xyz_m"),
            expected["camera_eye_xyz_m"], rtol=0.0, atol=1.0e-12,
        )
        and np.allclose(
            configuration.get("camera_target_isaac_xyz_m"),
            expected["camera_target_xyz_m"], rtol=0.0, atol=1.0e-12,
        )
        and abs(configuration["camera_lens_mm"] - expected["camera_lens_mm"]) <= 1.0e-12
    ),
    "source_sample_surface_indexing_is_used": (
        configuration.get("surface_index_mode") == "source-sample"
        and frame.get("surface_index_mode") == "source-sample"
        and int(frame["splashsurf_render_frame"]) == args.source_sample
        and water_obj.name == f"surface_{args.source_sample:06d}_clipped.obj"
    ),
    "surface_hash_matches_owned_reconstruction": (
        sha256_file(water_obj) == frame["water_obj_sha256"]
        == surface_frame["surface_sha256"]
    ),
    "physical_render_policy_is_preserved": (
        render_manifest.get("actual_scale") is True
        and render_manifest.get("visibility_enlargement") is False
        and render_manifest.get("emission_used") is False
        and render_manifest.get("arbitrary_opacity_used") is False
        and frame.get("actual_scale") is True
        and frame.get("emission_used") is False
        and frame.get("arbitrary_opacity_used") is False
    ),
    "png_is_complete_and_correct_size": (
        png.is_file()
        and sha256_file(png) == frame["png_sha256"]
        and png_dimensions(png) == tuple(configuration["resolution"])
    ),
    "gas_lod_is_conservative": abs(gas_volume - lod_gas_volume) <= 1.0e-18,
    "micro_cap_area_is_conservative": (
        abs(float(lod["micro_density"]["cap_area_residual_m2"])) <= 1.0e-14
    ),
}
report = {
    "schema": 1,
    "product": "whitewater_v6_artist_render_gate_audit",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": {
        "source_sample": args.source_sample,
        "png_dimensions": list(png_dimensions(png)),
        "cells": frame["counts"]["cells"],
        "films": frame["counts"]["films"],
        "borders": frame["counts"]["borders"],
        "nodes": frame["counts"]["nodes"],
        "micro_cluster_hero": [lod["micro_cells"], lod["cluster_cells"], lod["hero_cells"]],
        "gas_lod_residual_m3": abs(gas_volume - lod_gas_volume),
        "micro_cap_area_residual_m2": abs(float(lod["micro_density"]["cap_area_residual_m2"])),
    },
    "provenance": {
        "resolved_configuration": str(resolved_path),
        "resolved_configuration_sha256": sha256_file(resolved_path),
        "render_manifest": str(render_manifest_path),
        "render_manifest_sha256": sha256_file(render_manifest_path),
        "surface_manifest": str(surface_manifest_path),
        "surface_manifest_sha256": sha256_file(surface_manifest_path),
        "png": str(png),
        "png_sha256": sha256_file(png),
    },
}
write_once(args.output_report, report)
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
