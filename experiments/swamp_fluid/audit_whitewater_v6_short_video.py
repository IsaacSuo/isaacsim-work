"""Audit a diagnostic whitewater-v6 image sequence and encoded video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


STATE_SURFACE_BUBBLE = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_directory", type=Path)
    parser.add_argument("marker_directory", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--source-samples", nargs="+", type=int, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--playback-fps", type=float, required=True)
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


def close_enough(first, second, tolerance=1.0e-12):
    return abs(float(first) - float(second)) <= tolerance


args = parse_args()
frames_directory = args.frames_directory.resolve()
marker_directory = args.marker_directory.resolve()
video_path = args.video.resolve()
output_report = args.output_report.resolve()
expected_sources = list(args.source_samples)

errors = []
frame_rows = []
for sequence_index, expected_source in enumerate(expected_sources, start=1):
    png_path = frames_directory / f"diagnostic_{sequence_index:04d}.png"
    render_report_path = png_path.with_suffix(".json")
    if not png_path.is_file() or not render_report_path.is_file():
        errors.append(f"Missing rendered frame or report at sequence index {sequence_index}")
        continue
    render_report = json.loads(render_report_path.read_text(encoding="utf-8"))
    source_sample = int(render_report["source_sample_index"])
    if source_sample != expected_source:
        errors.append(
            f"Source mapping mismatch at sequence index {sequence_index}: "
            f"expected {expected_source}, got {source_sample}"
        )
    output_frame = int(render_report["output_frame"])
    marker_path = marker_directory / f"marker_state_{output_frame:06d}.npz"
    if not marker_path.is_file():
        errors.append(f"Missing marker cache: {marker_path}")
        continue
    with np.load(marker_path, allow_pickle=False) as cache:
        if int(cache["source_sample_index"]) != source_sample:
            errors.append(f"Marker/report source mismatch at source {source_sample}")
        markers = np.asarray(cache["markers"])
    surface = markers[markers["state"] == STATE_SURFACE_BUBBLE]
    source_count = int(len(surface))
    source_gas = float(surface["phase_volume"].sum(dtype=np.float64))
    rendered_count = int(render_report["counts"]["surface_bubble_anchors"])
    unbound_count = source_count - rendered_count
    if unbound_count < 0:
        errors.append(f"Rendered surface count exceeds source count at source {source_sample}")

    gas_ledger = render_report.get("surface_bubble_gas_volume")
    if gas_ledger is None:
        if unbound_count:
            errors.append(
                f"Frame {source_sample} omits {unbound_count} surface bubbles without a gas ledger"
            )
        unbound_gas = 0.0
    else:
        unbound_ids = np.asarray(gas_ledger["unbound_marker_ids"], dtype=np.uint64)
        unbound_mask = np.isin(surface["id"], unbound_ids)
        unbound_gas = float(surface["phase_volume"][unbound_mask].sum(dtype=np.float64))
        checks = (
            (int(np.count_nonzero(unbound_mask)) == unbound_count, "unbound count"),
            (close_enough(gas_ledger["source_m3"], source_gas), "source gas volume"),
            (close_enough(gas_ledger["unbound_m3"], unbound_gas), "unbound gas volume"),
            (
                close_enough(
                    gas_ledger["render_bound_m3"] + gas_ledger["unbound_m3"],
                    source_gas,
                ),
                "gas partition",
            ),
            (close_enough(gas_ledger["residual_m3"], 0.0), "gas residual"),
        )
        for passed, label in checks:
            if not passed:
                errors.append(f"Invalid {label} at source {source_sample}")

    film = render_report["film_area"]
    contact = render_report["surface_bubble_gas_footprint"]["deformable_contact"]
    dome = render_report["surface_bubble_gas_footprint"]["resolved_dome_volume_proxy"]
    if abs(float(film["residual_m2"])) > 1.0e-12:
        errors.append(f"Film-area residual failed at source {source_sample}")
    if float(contact["maximum_compression"]) > 0.3001:
        errors.append(f"Contact compression failed at source {source_sample}")
    if abs(float(dome["volume_scale_residual"])) > 1.0e-12:
        errors.append(f"Dome volume proxy failed at source {source_sample}")
    width, height = png_dimensions(png_path)
    if (width, height) != (720, 720):
        errors.append(f"Unexpected PNG dimensions at source {source_sample}: {width}x{height}")
    frame_rows.append(
        {
            "sequence_index": sequence_index,
            "source_sample_index": source_sample,
            "output_frame": output_frame,
            "png": str(png_path),
            "png_sha256": sha256_file(png_path),
            "render_report": str(render_report_path),
            "render_report_sha256": sha256_file(render_report_path),
            "surface_bubble_count": source_count,
            "render_bound_surface_bubble_count": rendered_count,
            "unbound_surface_bubble_count": unbound_count,
            "source_surface_gas_volume_m3": source_gas,
            "unbound_surface_gas_volume_m3": unbound_gas,
            "film_area_residual_m2": float(film["residual_m2"]),
            "unbound_film_area_m2": float(film["unbound_m2"]),
            "maximum_contact_compression": float(contact["maximum_compression"]),
            "dome_volume_scale_residual": float(dome["volume_scale_residual"]),
        }
    )

video_probe = None
if not video_path.is_file():
    errors.append(f"Missing encoded video: {video_path}")
else:
    probe = subprocess.run(
        [
            str(args.ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    video_probe = json.loads(probe.stdout)["streams"][0]
    if int(video_probe["width"]) != 720 or int(video_probe["height"]) != 720:
        errors.append("Encoded video dimensions do not match the PNG sequence")
    if int(video_probe.get("nb_frames", -1)) != len(expected_sources):
        errors.append("Encoded video frame count does not match the source sequence")

payload = {
    "schema": 1,
    "product": "whitewater_v6_surface_raft_short_video_diagnostic",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": not errors,
    "diagnostic_only": True,
    "lookdev_approved": False,
    "approval_state": "numeric_and_render_ledger_passed_visual_approval_pending",
    "errors": errors,
    "timing": {
        "source_fps": args.source_fps,
        "playback_fps": args.playback_fps,
        "playback_speed_fraction": args.playback_fps / args.source_fps,
        "source_duration_seconds": len(expected_sources) / args.source_fps,
        "playback_duration_seconds": len(expected_sources) / args.playback_fps,
        "frame_duplication": False,
        "frame_interpolation": False,
    },
    "source_samples": expected_sources,
    "frames": frame_rows,
    "video": {
        "path": str(video_path),
        "bytes": video_path.stat().st_size if video_path.is_file() else None,
        "sha256": sha256_file(video_path) if video_path.is_file() else None,
        "probe": video_probe,
    },
}
output_report.parent.mkdir(parents=True, exist_ok=True)
temporary = output_report.with_suffix(output_report.suffix + ".tmp")
with temporary.open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output_report)
print(json.dumps({"valid": payload["valid"], "errors": errors}, indent=2))
if errors:
    raise SystemExit(1)
