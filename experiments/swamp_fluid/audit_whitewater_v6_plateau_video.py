"""Audit a Plateau production render sequence and its encoded short video."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_directory", type=Path)
    parser.add_argument("plateau_directory", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--ffprobe", type=Path, required=True)
    return parser.parse_args()


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
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


args = parse_args()
render_directory = args.render_directory.resolve()
plateau_directory = args.plateau_directory.resolve()
video = args.video.resolve()
report_path = args.output_report.resolve()
render_manifest_path = render_directory / "render_manifest.json"
plateau_manifest_path = plateau_directory / "manifest.json"
render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
plateau_manifest = json.loads(plateau_manifest_path.read_text(encoding="utf-8"))
errors = []
if render_manifest.get("product") != "whitewater_v6_plateau_production_render":
    errors.append("unexpected render product")
if not render_manifest.get("complete"):
    errors.append("render manifest is incomplete")
if not plateau_manifest.get("complete"):
    errors.append("Plateau manifest is incomplete")
if render_manifest.get("actual_scale") is not True:
    errors.append("render is not declared actual scale")
if render_manifest.get("visibility_enlargement") is not False:
    errors.append("visibility enlargement is enabled")
if render_manifest.get("emission_used") is not False:
    errors.append("emission is enabled")
if render_manifest.get("arbitrary_opacity_used") is not False:
    errors.append("arbitrary opacity is enabled")

plateau_by_source = {
    int(sample["source_sample_index"]): sample
    for sample in plateau_manifest["samples"]
}
width, height = render_manifest["configuration"]["resolution"]
maximum = {
    "gas_lod_residual_m3": 0.0,
    "micro_cap_area_residual_m2": 0.0,
    "ledger_gas_residual_m3": 0.0,
    "ledger_film_area_residual_m2": 0.0,
}
frame_rows = []
for expected_sequence, frame in enumerate(render_manifest.get("frames", []), start=1):
    source = int(frame["source_sample_index"])
    if int(frame["sequence_index"]) != expected_sequence:
        errors.append(f"source {source}: non-contiguous sequence index")
    if source not in plateau_by_source:
        errors.append(f"source {source}: missing Plateau input")
        continue
    png = render_directory / f"plateau_{expected_sequence:04d}.png"
    frame_report = png.with_suffix(".json")
    if not png.is_file() or not frame_report.is_file():
        errors.append(f"source {source}: missing PNG or frame report")
        continue
    if png_dimensions(png) != (width, height):
        errors.append(f"source {source}: PNG dimensions mismatch")
    if frame.get("png_sha256") != sha256_file(png):
        errors.append(f"source {source}: PNG hash mismatch")
    if frame.get("actual_scale") is not True or frame.get("emission_used") is not False or frame.get("arbitrary_opacity_used") is not False:
        errors.append(f"source {source}: physical render policy mismatch")
    sample = plateau_by_source[source]
    plateau_path = plateau_directory / sample["file"]
    if frame.get("plateau_npz_sha256") != sha256_file(plateau_path):
        errors.append(f"source {source}: Plateau cache hash mismatch")
    with np.load(plateau_path, allow_pickle=False) as cache:
        cells = np.asarray(cache["cells"])
        films = np.asarray(cache["films"])
        borders = np.asarray(cache["borders"])
        nodes = np.asarray(cache["nodes"])
    source_gas = float(cells["gas_volume"].sum(dtype=np.float64))
    lod = frame["lod"]
    lod_gas = sum(
        float(lod[name])
        for name in (
            "micro_gas_volume_m3",
            "cluster_gas_volume_m3",
            "hero_gas_volume_m3",
        )
    )
    gas_lod_residual = lod_gas - source_gas
    cap_area_residual = float(lod["micro_density"]["cap_area_residual_m2"])
    ledger = frame["ledgers"]
    gas_ledger_residual = float(ledger["plateau_gas_volume_m3"]) - source_gas
    film_area_residual = float(ledger["shared_film_area_m2"]) - float(
        films["area"].sum(dtype=np.float64)
    )
    maximum["gas_lod_residual_m3"] = max(maximum["gas_lod_residual_m3"], abs(gas_lod_residual))
    maximum["micro_cap_area_residual_m2"] = max(maximum["micro_cap_area_residual_m2"], abs(cap_area_residual))
    maximum["ledger_gas_residual_m3"] = max(maximum["ledger_gas_residual_m3"], abs(gas_ledger_residual))
    maximum["ledger_film_area_residual_m2"] = max(maximum["ledger_film_area_residual_m2"], abs(film_area_residual))
    if int(frame["counts"]["cells"]) != len(cells) or int(frame["counts"]["films"]) != len(films) or int(frame["counts"]["borders"]) != len(borders) or int(frame["counts"]["nodes"]) != len(nodes):
        errors.append(f"source {source}: render count ledger mismatch")
    frame_rows.append({
        "sequence_index": expected_sequence,
        "source_sample_index": source,
        "png": str(png),
        "png_sha256": sha256_file(png),
        "lod_counts": {
            "micro": int(lod["micro_cells"]),
            "cluster": int(lod["cluster_cells"]),
            "hero": int(lod["hero_cells"]),
        },
        "gas_lod_residual_m3": gas_lod_residual,
        "micro_cap_area_residual_m2": cap_area_residual,
    })

tolerances = {"volume_m3": 1.0e-15, "area_m2": 1.0e-14}
if max(maximum["gas_lod_residual_m3"], maximum["ledger_gas_residual_m3"]) > tolerances["volume_m3"]:
    errors.append("gas render ledger gate exceeded")
if max(maximum["micro_cap_area_residual_m2"], maximum["ledger_film_area_residual_m2"]) > tolerances["area_m2"]:
    errors.append("render area ledger gate exceeded")

probe_data = None
if not video.is_file():
    errors.append("encoded video is missing")
else:
    process = subprocess.run(
        [
            str(args.ffprobe.resolve()), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-of", "json", str(video),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    probe_data = json.loads(process.stdout)["streams"][0]
    if int(probe_data["width"]) != width or int(probe_data["height"]) != height:
        errors.append("encoded video dimensions mismatch")
    if int(probe_data.get("nb_frames", -1)) != len(frame_rows):
        errors.append("encoded video frame count mismatch")

payload = {
    "schema": 1,
    "product": "whitewater_v6_plateau_production_video_audit",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": not errors,
    "lookdev_approved": False,
    "approval_state": "numeric_and_render_policy_passed_visual_approval_pending",
    "criteria": {
        "render_inputs_are_hash_locked": not any("hash mismatch" in item for item in errors),
        "actual_scale_without_visibility_enlargement": not any("actual scale" in item or "enlargement" in item for item in errors),
        "no_emission_or_arbitrary_opacity": not any("emission" in item or "opacity" in item for item in errors),
        "camera_lod_preserves_gas_volume": maximum["gas_lod_residual_m3"] <= tolerances["volume_m3"],
        "micro_density_preserves_removed_cap_area": maximum["micro_cap_area_residual_m2"] <= tolerances["area_m2"],
        "frame_and_video_counts_match": not any("frame count" in item for item in errors),
    },
    "timing": {
        "fps": render_manifest["configuration"]["fps"],
        "frames": len(frame_rows),
        "duration_seconds": len(frame_rows) / float(render_manifest["configuration"]["fps"]),
        "frame_duplication": False,
        "frame_interpolation": False,
    },
    "maximum_absolute_residuals": maximum,
    "tolerances": tolerances,
    "frames": frame_rows,
    "video": {
        "path": str(video),
        "bytes": video.stat().st_size if video.is_file() else None,
        "sha256": sha256_file(video) if video.is_file() else None,
        "probe": probe_data,
    },
    "errors": errors,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
if report_path.exists():
    raise FileExistsError(f"Refusing to overwrite {report_path}")
report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({"valid": payload["valid"], "criteria": payload["criteria"], "maximum_absolute_residuals": maximum, "errors": errors}, indent=2))
if errors:
    raise SystemExit(1)
