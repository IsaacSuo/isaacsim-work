"""Validate and encode a fixed-topology cached render sequence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fixed_topology_video import job_path, load_job
from liquid_video_cache import atomic_write_json, file_sha256, read_jsonl, validate_png


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("[encode] " + " ".join(command), flush=True)
    return subprocess.run(command, check=True, capture_output=capture, text=True)


def render_rows(job_dir: Path, job: dict[str, Any]) -> dict[int, dict[str, Any]]:
    complete_manifest = job_path(job_dir, job, "render_manifest")
    if complete_manifest.is_file():
        rows = read_jsonl(complete_manifest)
    else:
        rows = []
        segments = job_path(job_dir, job, "render_segments_dir")
        for accepted_path in sorted(segments.glob("segment_*_accepted.json")):
            if "_rejected_" in accepted_path.name:
                continue
            accepted = json.loads(accepted_path.read_text(encoding="utf-8-sig"))
            manifest = segments / accepted_path.name.replace("_accepted.json", ".jsonl")
            if not accepted.get("valid") or accepted.get("manifest_sha256") != file_sha256(manifest):
                raise RuntimeError(f"Stale accepted render segment: {accepted_path}")
            rows.extend(read_jsonl(manifest))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["output_index"])
        if index in result:
            raise RuntimeError(f"Overlapping render manifests at frame {index}")
        result[index] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    args = parser.parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be available in PATH")
    job_dir = Path(args.job).resolve()
    job = load_job(job_dir)
    video = job["video"]
    total = int(video["output_frames"])
    fps, width, height = int(video["output_fps"]), int(video["width"]), int(video["height"])
    count = args.frame_count if args.frame_count is not None else total
    if args.start < 0 or count <= 0 or args.start + count > total:
        raise ValueError("Encode range is outside the job timeline")
    partial = args.start != 0 or count != total
    if partial and not args.allow_partial:
        raise RuntimeError("Partial encoding requires --allow-partial")

    expected_render_provenance: str | None = None
    if not partial:
        completion_path = job_path(job_dir, job, "render_complete")
        if not completion_path.is_file():
            raise RuntimeError(f"Missing render completion marker: {completion_path}")
        completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
        if not completion.get("valid") or int(completion.get("frame_count", -1)) != total:
            raise RuntimeError("Complete render marker is invalid")
        for key in ("take_id", "config_hash", "simulation_provenance_hash"):
            if str(completion.get(key)) != str(job[key]):
                raise RuntimeError(f"Complete render has another {key}")
        expected_render_provenance = str(completion["render_provenance_hash"])

    rows = render_rows(job_dir, job)
    frames_dir = job_path(job_dir, job, "frames_dir")
    previous_hash: str | None = None
    duplicates: list[tuple[int, int]] = []
    for index in range(args.start, args.start + count):
        row = rows.get(index)
        if row is None:
            raise RuntimeError(f"Render manifest lacks frame {index}")
        for key in ("take_id", "config_hash", "simulation_provenance_hash"):
            if str(row.get(key)) != str(job[key]):
                raise RuntimeError(f"Rendered frame {index} has another {key}")
        provenance = str(row.get("render_provenance_hash", ""))
        if expected_render_provenance is None:
            expected_render_provenance = provenance
        if not provenance or provenance != expected_render_provenance:
            raise RuntimeError(f"Rendered frame {index} has another render provenance")
        png = validate_png(
            frames_dir / f"rgb_{index:06d}.png",
            width=width,
            height=height,
            expected_sha256=str(row["png_sha256"]),
        )
        if previous_hash == png["png_sha256"]:
            duplicates.append((index - 1, index))
        previous_hash = png["png_sha256"]

    output = (
        Path(args.output).resolve()
        if args.output
        else job_path(job_dir, job, "video_dir") / ("pilot_h264.mp4" if partial else "final_h264.mp4")
    )
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must be an .mp4 file")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part.mp4")
    if temporary.exists():
        temporary.unlink()
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
            "-framerate", str(fps), "-start_number", str(args.start),
            "-i", str(frames_dir / "rgb_%06d.png"), "-frames:v", str(count),
            "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart",
            str(temporary),
        ]
    )
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError("ffmpeg did not produce a video")
    probe = json.loads(
        run(
            [
                "ffprobe", "-v", "error", "-count_frames", "-show_entries",
                "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,color_primaries,color_transfer,color_space:format=duration",
                "-of", "json", str(temporary),
            ],
            capture=True,
        ).stdout
    )
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("Encoded file must contain exactly one video stream")
    stream = streams[0]
    duration = float(probe["format"]["duration"])
    checks = {
        "codec": stream.get("codec_name") == "h264",
        "width": int(stream.get("width", -1)) == width,
        "height": int(stream.get("height", -1)) == height,
        "pixel_format": stream.get("pix_fmt") == "yuv420p",
        "frame_rate": stream.get("r_frame_rate") == f"{fps}/1",
        "average_frame_rate": stream.get("avg_frame_rate") == f"{fps}/1",
        "frame_count": int(stream.get("nb_read_frames", -1)) == count,
        "duration": abs(duration - count / fps) <= max(0.001, 0.5 / fps),
        "color_primaries": stream.get("color_primaries") == "bt709",
        "color_transfer": stream.get("color_transfer") == "bt709",
        "color_space": stream.get("color_space") == "bt709",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Encoded video failed checks: {checks}")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(temporary), "-f", "null", "-"])
    os.replace(temporary, output)
    reports = job_path(job_dir, job, "reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "valid": True,
        "completed_at": utc_now(),
        "partial": partial,
        "start_frame": args.start,
        "frame_count": count,
        "fps": fps,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "checks": checks,
        "duplicate_consecutive_frames": duplicates,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": file_sha256(output),
        "take_id": job["take_id"],
        "config_hash": job["config_hash"],
        "simulation_provenance_hash": job["simulation_provenance_hash"],
        "render_provenance_hash": expected_render_provenance,
        "encoder_sha256": file_sha256(Path(__file__).resolve()),
    }
    marker = reports / ("pilot_encode_complete.json" if partial else "encode_complete.json")
    atomic_write_json(marker, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
