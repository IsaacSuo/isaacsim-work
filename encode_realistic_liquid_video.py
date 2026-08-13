"""Validate, encode, and probe a realistic-liquid PNG sequence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from liquid_video_cache import (
    atomic_write_json,
    file_sha256,
    read_jsonl,
    validate_job,
    validate_png,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_job(job_dir: Path) -> dict[str, Any]:
    with open(job_dir / "job.json", "r", encoding="utf-8-sig") as stream:
        return validate_job(json.load(stream))


def job_path(job_dir: Path, job: dict[str, Any], name: str) -> Path:
    return job_dir / Path(job["paths"][name])


def latest_render_rows(job_dir: Path, job: dict[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = job_path(job_dir, job, "render_manifest")
    if manifest.is_file():
        rows = read_jsonl(manifest)
    else:
        rows = []
        segments_dir = job_path(job_dir, job, "render_segments_dir")
        for accepted_path in sorted(segments_dir.glob("segment_*_accepted.json")):
            if "_rejected_" in accepted_path.name:
                continue
            stem = accepted_path.name.removesuffix("_accepted.json")
            segment_manifest = segments_dir / f"{stem}.jsonl"
            if not segment_manifest.is_file():
                raise RuntimeError(f"Accepted segment lacks manifest: {stem}")
            with open(accepted_path, "r", encoding="utf-8-sig") as stream:
                accepted = json.load(stream)
            if not accepted.get("valid") or str(accepted.get("manifest_sha256")) != file_sha256(
                segment_manifest
            ):
                raise RuntimeError(f"Accepted segment marker is stale: {stem}")
            rows.extend(read_jsonl(segment_manifest))
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["output_index"])
        if index in latest:
            raise RuntimeError(f"Overlapping render manifests contain frame {index}")
        latest[index] = row
    return latest


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("[encode] " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=True,
        capture_output=capture,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be available in WSL PATH")
    job_dir = Path(args.job).resolve()
    job = load_job(job_dir)
    video = job["video"]
    expected_total = int(video["output_frames"])
    fps = int(video["output_fps"])
    width = int(video["width"])
    height = int(video["height"])
    frame_count = args.frame_count if args.frame_count is not None else expected_total
    if args.start < 0 or frame_count <= 0 or args.start + frame_count > expected_total:
        raise ValueError("Requested encode range is outside the job timeline")
    partial = args.start != 0 or frame_count != expected_total
    expected_render_provenance: str | None = None
    if partial and not args.allow_partial:
        raise RuntimeError("Partial encoding requires --allow-partial")

    if not partial:
        render_complete = job_path(job_dir, job, "render_complete")
        if not render_complete.is_file():
            raise RuntimeError(f"Missing render completion marker: {render_complete}")
        with open(render_complete, "r", encoding="utf-8-sig") as stream:
            completion = json.load(stream)
        if not completion.get("valid") or int(completion.get("frame_count", -1)) != expected_total:
            raise RuntimeError("render_complete.json does not validate the complete sequence")
        if str(completion.get("take_id")) != str(job["take_id"]):
            raise RuntimeError("Render completion belongs to another take")
        if str(completion.get("config_hash")) != str(job["config_hash"]):
            raise RuntimeError("Render completion belongs to another config")
        if str(completion.get("simulation_provenance_hash")) != str(
            job["simulation_provenance_hash"]
        ):
            raise RuntimeError("Render completion belongs to another simulation provenance")
        expected_render_provenance = str(
            completion.get("render_provenance_hash", "")
        )
        if not expected_render_provenance:
            raise RuntimeError("Render completion lacks render provenance")

    frames_dir = job_path(job_dir, job, "frames_dir")
    rows = latest_render_rows(job_dir, job)
    selected_rows: list[dict[str, Any]] = []
    duplicate_pairs: list[tuple[int, int]] = []
    previous_hash: str | None = None
    previous_index: int | None = None
    for index in range(args.start, args.start + frame_count):
        row = rows.get(index)
        if row is None:
            raise RuntimeError(f"Render manifest is missing frame {index}")
        if str(row.get("take_id")) != str(job["take_id"]):
            raise RuntimeError(f"Frame {index} belongs to another take")
        if str(row.get("config_hash")) != str(job["config_hash"]):
            raise RuntimeError(f"Frame {index} belongs to another config")
        if str(row.get("simulation_provenance_hash")) != str(
            job["simulation_provenance_hash"]
        ):
            raise RuntimeError(f"Frame {index} belongs to another simulation provenance")
        row_render_provenance = str(row.get("render_provenance_hash", ""))
        if expected_render_provenance is None:
            expected_render_provenance = row_render_provenance
        if not row_render_provenance or row_render_provenance != expected_render_provenance:
            raise RuntimeError(f"Frame {index} belongs to another render provenance")
        png_path = frames_dir / f"rgb_{index:06d}.png"
        validated = validate_png(
            png_path,
            width=width,
            height=height,
            expected_sha256=str(row["png_sha256"]),
        )
        digest = validated["png_sha256"]
        if digest == previous_hash and previous_index is not None:
            duplicate_pairs.append((previous_index, index))
        previous_hash = digest
        previous_index = index
        selected_rows.append(row)

    output = (
        Path(args.output).resolve()
        if args.output
        else job_path(job_dir, job, "video_dir") / "final_h264.mp4"
    )
    if output.suffix.lower() != ".mp4":
        raise ValueError("The H.264 encoder only supports an .mp4 output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.part.mp4")
    if temporary_output.exists():
        temporary_output.unlink()
    pattern = frames_dir / "rgb_%06d.png"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(args.start),
        "-i",
        str(pattern),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        str(temporary_output),
    ]
    run(command)
    if not temporary_output.is_file() or temporary_output.stat().st_size <= 0:
        raise RuntimeError("ffmpeg did not produce a non-empty video")

    probe_result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,nb_read_frames,color_primaries,color_transfer,color_space:format=duration",
            "-of",
            "json",
            str(temporary_output),
        ],
        capture=True,
    )
    probe = json.loads(probe_result.stdout)
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream, found {len(streams)}")
    stream = streams[0]
    expected_duration = frame_count / fps
    actual_duration = float(probe["format"]["duration"])
    time_base_parts = str(stream.get("time_base", "0/0")).split("/")
    valid_time_base = (
        len(time_base_parts) == 2
        and int(time_base_parts[0]) > 0
        and int(time_base_parts[1]) > 0
    )
    checks = {
        "codec": stream.get("codec_name") == "h264",
        "width": int(stream.get("width", -1)) == width,
        "height": int(stream.get("height", -1)) == height,
        "pixel_format": stream.get("pix_fmt") == "yuv420p",
        "frame_rate": stream.get("r_frame_rate") == f"{fps}/1",
        "average_frame_rate": stream.get("avg_frame_rate") == f"{fps}/1",
        "time_base": valid_time_base,
        "color_primaries": stream.get("color_primaries") == "bt709",
        "color_transfer": stream.get("color_transfer") == "bt709",
        "color_space": stream.get("color_space") == "bt709",
        "frame_count": int(stream.get("nb_read_frames", -1)) == frame_count,
        "duration": abs(actual_duration - expected_duration) <= max(0.001, 0.5 / fps),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Encoded video failed ffprobe checks: {checks}; probe={probe}")

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(temporary_output),
            "-f",
            "null",
            "-",
        ]
    )
    os.replace(temporary_output, output)

    reports_dir = job_path(job_dir, job, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    prefix = "pilot" if partial else "final"
    representative_offsets = sorted({0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1})
    representative_files: list[str] = []
    for offset in representative_offsets:
        index = args.start + offset
        destination = reports_dir / f"{prefix}_frame_{index:06d}.png"
        shutil.copy2(frames_dir / f"rgb_{index:06d}.png", destination)
        representative_files.append(destination.name)

    sheet_count = min(20, frame_count)
    if sheet_count == 1:
        selected_offsets = [0]
    else:
        selected_offsets = sorted(
            {
                round(i * (frame_count - 1) / (sheet_count - 1))
                for i in range(sheet_count)
            }
        )
    expression = "+".join(f"eq(n\\,{index})" for index in selected_offsets)
    columns = 5
    rows_count = (len(selected_offsets) + columns - 1) // columns
    contact_sheet = reports_dir / f"{prefix}_contact_sheet.png"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output),
            "-vf",
            f"select='{expression}',scale=256:-2,tile={columns}x{rows_count}",
            "-frames:v",
            "1",
            str(contact_sheet),
        ]
    )
    validate_png(
        contact_sheet,
        width=columns * 256,
        height=(height * 256 // width + (height * 256 // width) % 2) * rows_count,
    )

    result = {
        "schema_version": 1,
        "valid": True,
        "completed_at": utc_now(),
        "partial": partial,
        "start_frame": args.start,
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": actual_duration,
        "width": width,
        "height": height,
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "checks": checks,
        "duplicate_consecutive_frames": duplicate_pairs,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": file_sha256(output),
        "representative_frames": representative_files,
        "contact_sheet": contact_sheet.name,
        "take_id": job["take_id"],
        "config_hash": job["config_hash"],
        "simulation_provenance_hash": job["simulation_provenance_hash"],
        "render_provenance_hash": expected_render_provenance,
        "encoder_sha256": file_sha256(Path(__file__).resolve()),
    }
    marker = reports_dir / ("pilot_encode_complete.json" if partial else "encode_complete.json")
    atomic_write_json(marker, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
