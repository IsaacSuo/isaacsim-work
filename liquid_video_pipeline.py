"""WSL-side orchestration for the recoverable realistic-liquid video job."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from liquid_video_cache import (
    atomic_write_json,
    atomic_write_jsonl,
    config_hash,
    expected_output_frames,
    file_sha256,
    immutable_job_config,
    load_surface_cache,
    read_jsonl,
    render_provenance_hash,
    require_integer_capture_stride,
    simulation_provenance_payload,
    validate_contiguous_manifest,
    validate_job,
    validate_png,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output" / "long_video"
BATCH_PATH = PROJECT_DIR / "run_realistic_liquid_long_video.bat"
ENCODER_PATH = PROJECT_DIR / "encode_realistic_liquid_video.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0].strip() if lines else None


def wsl_to_windows(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        suffix = "\\".join(parts[3:])
        return f"{drive}:\\{suffix}"
    raise ValueError(
        "Isaac Sim jobs must live on a Windows-mounted WSL path such as /mnt/y; "
        f"UNC-backed WSL paths are not supported reliably: {resolved}"
    )


def load_job(job_dir: Path) -> dict[str, Any]:
    with open(job_dir / "job.json", "r", encoding="utf-8-sig") as stream:
        job = json.load(stream)
    return validate_job(job)


def job_path(job_dir: Path, job: dict[str, Any], name: str) -> Path:
    return job_dir / Path(job["paths"][name])


def ensure_free_space(path: Path, minimum_gib: float) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"Insufficient free space at {path}: {free_gib:.2f} GiB < {minimum_gib:.2f} GiB"
        )
    return {
        "total_gib": usage.total / 1024**3,
        "used_gib": usage.used / 1024**3,
        "free_gib": free_gib,
        "required_free_gib": minimum_gib,
    }


def script_hashes() -> dict[str, str | None]:
    names = [
        "physx_realistic_liquid.py",
        "liquid_video_cache.py",
        "render_realistic_liquid_cache.py",
        "liquid_video_pipeline.py",
        "encode_realistic_liquid_video.py",
        "run_realistic_liquid_long_video.bat",
        "run_realistic_liquid_stage.ps1",
    ]
    result: dict[str, str | None] = {}
    for name in names:
        path = PROJECT_DIR / name
        result[name] = file_sha256(path) if path.is_file() else None
    return result


def initialize_job(args: argparse.Namespace) -> None:
    output_frames = expected_output_frames(args.duration, args.fps)
    capture_stride = require_integer_capture_stride(args.physics_fps, args.fps)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.job_id):
        raise ValueError(
            "job-id must be one safe basename matching "
            "[A-Za-z0-9][A-Za-z0-9._-]*"
        )
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    disk = ensure_free_space(root, args.min_free_gib)
    job_dir = (root / args.job_id).resolve()
    if job_dir.parent != root:
        raise ValueError("job-id escapes the configured output root")
    if job_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing job: {job_dir}")

    job_uuid = str(uuid.uuid4())
    take_id = f"take_{uuid.uuid4().hex}"
    render_profile = f"path_{args.width}x{args.height}_{args.fps}fps_{args.spp}spp"
    config = {
        "job_uuid": job_uuid,
        "take_id": take_id,
        "video": {
            "duration_seconds": args.duration,
            "output_fps": args.fps,
            "output_frames": output_frames,
            "width": args.width,
            "height": args.height,
        },
        "physics": {
            "physics_fps": args.physics_fps,
            "capture_stride": capture_stride,
            "substeps": args.substeps,
            "solver_iterations": args.solver_iterations,
            "source_mode": "emitter",
            "emitter_recycle": True,
            "spacing_m": args.spacing,
        },
        "render": {
            "renderer": "PathTracing",
            "path_spp": args.spp,
            "camera": args.camera,
            "segment_frames": args.segment_frames,
            "profile": render_profile,
        },
    }
    digest = config_hash(config)
    hashes = script_hashes()
    if hashes["physx_realistic_liquid.py"] is None:
        raise RuntimeError("Missing physx_realistic_liquid.py for provenance")
    simulation_digest = config_hash(
        simulation_provenance_payload(
            {
                **config,
                "script_sha256": hashes,
            }
        )
    )
    paths = {
        "take_dir": f"takes/{take_id}",
        "cache_dir": f"takes/{take_id}/cache",
        "simulation_manifest": f"takes/{take_id}/simulation_manifest.jsonl",
        "simulation_complete": f"takes/{take_id}/simulation_complete.json",
        "render_template": f"takes/{take_id}/render_template.usda",
        "render_dir": f"renders/{render_profile}",
        "frames_dir": f"renders/{render_profile}/frames",
        "render_segments_dir": f"renders/{render_profile}/segments",
        "render_manifest": f"renders/{render_profile}/render_manifest.jsonl",
        "render_complete": f"renders/{render_profile}/render_complete.json",
        "video_dir": "video",
        "reports_dir": "reports",
        "logs_dir": "logs",
        "particle_set_prim": "/World/WaterParticles",
        "particle_system_prim": "/World/ParticleSystem",
        "cached_mesh_prim": "/World/CachedLiquid",
        "water_material_prim": "/World/Looks/WaterRender",
        "camera_prim": "/World/RenderCamera",
    }
    job = {
        "schema_version": 2,
        "job_id": args.job_id,
        "job_uuid": job_uuid,
        "created_at": utc_now(),
        "status": "initialized",
        "take_id": take_id,
        "config_hash": digest,
        "simulation_provenance_hash": simulation_digest,
        "video": config["video"],
        "physics": config["physics"],
        "render": config["render"],
        "paths": paths,
        "resource_budget": {
            "minimum_free_gib": args.min_free_gib,
            "hard_vram_limit_mib": args.vram_limit_mib,
            "target_vram_mib": args.vram_target_mib,
            "initial_disk": disk,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "ffmpeg": command_version(["ffmpeg", "-version"]),
            "ffprobe": command_version(["ffprobe", "-version"]),
            "nvidia_smi": command_version(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        },
        "script_sha256": hashes,
        "timeline": {
            "frame_zero": "post_preroll_static_state",
            "sample_times": "output_index/output_fps",
            "last_sample_seconds": (output_frames - 1) / args.fps,
            "container_duration_seconds": output_frames / args.fps,
            "uniform": True,
        },
    }
    for name in (
        "cache_dir",
        "frames_dir",
        "render_segments_dir",
        "video_dir",
        "reports_dir",
        "logs_dir",
    ):
        (job_dir / paths[name]).mkdir(parents=True, exist_ok=True)
    validate_job(job)
    atomic_write_json(job_dir / "job.json", job)
    print(json.dumps({"job": str(job_dir), "config_hash": digest, "frames": output_frames}, indent=2))


def validate_cache(job_dir: Path, *, full: bool) -> dict[str, Any]:
    job = load_job(job_dir)
    complete_path = job_path(job_dir, job, "simulation_complete")
    if not complete_path.is_file():
        raise RuntimeError(f"Missing simulation completion marker: {complete_path}")
    with open(complete_path, "r", encoding="utf-8-sig") as stream:
        complete = json.load(stream)
    if not complete.get("valid"):
        raise RuntimeError("simulation_complete.json is not valid")
    if str(complete.get("take_id")) != str(job["take_id"]):
        raise RuntimeError("Simulation completion marker has another take ID")
    if str(complete.get("config_hash")) != str(job["config_hash"]):
        raise RuntimeError("Simulation completion marker has another config hash")
    if str(complete.get("simulation_provenance_hash")) != str(
        job["simulation_provenance_hash"]
    ):
        raise RuntimeError("Simulation completion marker has another provenance hash")
    if int(complete.get("frame_count", -1)) != int(job["video"]["output_frames"]):
        raise RuntimeError("Simulation completion marker has the wrong frame count")
    template_path = job_path(job_dir, job, "render_template")
    if not template_path.is_file() or file_sha256(template_path) != str(
        complete.get("render_template_sha256")
    ):
        raise RuntimeError("Render template is missing or its hash has changed")

    manifest_path = job_path(job_dir, job, "simulation_manifest")
    rows = validate_contiguous_manifest(
        read_jsonl(manifest_path),
        expected_frames=int(job["video"]["output_frames"]),
        expected_take_id=str(job["take_id"]),
        expected_config_hash=str(job["config_hash"]),
        expected_simulation_provenance_hash=str(
            job["simulation_provenance_hash"]
        ),
    )
    cache_dir = job_path(job_dir, job, "cache_dir")
    total_bytes = 0
    for row in rows:
        path = cache_dir / str(row["cache_file"])
        if not path.is_file() or path.stat().st_size != int(row["cache_bytes"]):
            raise RuntimeError(f"Missing or size-mismatched cache: {path}")
        total_bytes += path.stat().st_size
        if full:
            loaded = load_surface_cache(path, expected_sha256=str(row["cache_sha256"]))
            if int(loaded["metadata"]["output_index"]) != int(row["output_index"]):
                raise RuntimeError(f"Cache output index mismatch: {path}")
        elif file_sha256(path) != str(row["cache_sha256"]):
            raise RuntimeError(f"Cache hash mismatch: {path}")
    result = {
        "schema_version": 1,
        "valid": True,
        "validated_at": utc_now(),
        "validation_mode": "full_arrays" if full else "hash_only",
        "take_id": job["take_id"],
        "config_hash": job["config_hash"],
        "simulation_provenance_hash": job["simulation_provenance_hash"],
        "frame_count": len(rows),
        "cache_bytes": total_bytes,
        "cache_gib": total_bytes / 1024**3,
    }
    atomic_write_json(job_path(job_dir, job, "reports_dir") / "cache_validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_windows_simulation(job_dir: Path) -> None:
    if not BATCH_PATH.is_file():
        raise FileNotFoundError(BATCH_PATH)
    job = load_job(job_dir)
    run_id = uuid.uuid4().hex
    command = [
        "cmd.exe",
        "/d",
        "/s",
        "/c",
        wsl_to_windows(BATCH_PATH),
        "simulate-cache",
        wsl_to_windows(job_dir),
        run_id,
    ]
    print("[pipeline] starting one continuous authoritative simulation take", flush=True)
    subprocess.run(command, check=True)
    accepted_path = job_path(job_dir, job, "take_dir") / "simulation_accepted.json"
    if not accepted_path.is_file():
        raise RuntimeError(f"Simulation did not write accepted marker: {accepted_path}")
    with open(accepted_path, "r", encoding="utf-8-sig") as stream:
        accepted = json.load(stream)
    if not accepted.get("valid") or str(accepted.get("run_id")) != run_id:
        raise RuntimeError("Simulation accepted marker does not belong to this run")
    if str(accepted.get("config_hash")) != str(job["config_hash"]):
        raise RuntimeError("Simulation accepted marker has another config")
    if str(accepted.get("simulation_provenance_hash")) != str(
        job["simulation_provenance_hash"]
    ):
        raise RuntimeError("Simulation accepted marker has another provenance")
    validate_cache(job_dir, full=False)
    print("[pipeline] authoritative simulation take accepted", flush=True)


def current_render_provenance(
    job_dir: Path,
    job: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    completion_path = job_path(job_dir, job, "simulation_complete")
    if not completion_path.is_file():
        raise RuntimeError(f"Missing simulation completion marker: {completion_path}")
    with open(completion_path, "r", encoding="utf-8-sig") as stream:
        completion = json.load(stream)
    if not completion.get("valid"):
        raise RuntimeError("Simulation completion marker is not valid")
    if str(completion.get("take_id")) != str(job["take_id"]):
        raise RuntimeError("Simulation completion marker has another take")
    if str(completion.get("config_hash")) != str(job["config_hash"]):
        raise RuntimeError("Simulation completion marker has another config")
    if str(completion.get("simulation_provenance_hash")) != str(
        job["simulation_provenance_hash"]
    ):
        raise RuntimeError("Simulation completion marker has another provenance")
    template = job_path(job_dir, job, "render_template")
    template_hash = file_sha256(template)
    if template_hash != str(completion.get("render_template_sha256")):
        raise RuntimeError("Render template hash differs from simulation completion")
    renderer_hash = file_sha256(PROJECT_DIR / "render_realistic_liquid_cache.py")
    provenance = render_provenance_hash(
        job,
        template_sha256=template_hash,
        renderer_sha256=renderer_hash,
        isaac_version=completion.get("isaac_version"),
    )
    return provenance, completion


def canonical_render_segments(job: dict[str, Any]) -> list[tuple[int, int]]:
    expected_frames = int(job["video"]["output_frames"])
    segment_size = int(job["render"]["segment_frames"])
    if segment_size <= 0:
        raise ValueError("render.segment_frames must be positive")
    return [
        (start, min(expected_frames - 1, start + segment_size - 1))
        for start in range(0, expected_frames, segment_size)
    ]


def run_windows_render_segment(job_dir: Path, start: int, end: int) -> None:
    if not BATCH_PATH.is_file():
        raise FileNotFoundError(BATCH_PATH)
    job = load_job(job_dir)
    expected_render_provenance, _ = current_render_provenance(job_dir, job)
    if (start, end) not in canonical_render_segments(job):
        raise ValueError(
            f"Render range {start}..{end} is not a canonical non-overlapping segment: "
            f"{canonical_render_segments(job)}"
        )
    run_id = uuid.uuid4().hex
    command = [
        "cmd.exe",
        "/d",
        "/s",
        "/c",
        wsl_to_windows(BATCH_PATH),
        "render-segment",
        wsl_to_windows(job_dir),
        str(start),
        str(end),
        run_id,
    ]
    print(f"[pipeline] rendering frames {start:06d}..{end:06d}", flush=True)
    subprocess.run(command, check=True)
    segments_dir = job_path(job_dir, job, "render_segments_dir")
    complete = segments_dir / f"segment_{start:06d}_{end:06d}_complete.json"
    accepted = segments_dir / f"segment_{start:06d}_{end:06d}_accepted.json"
    manifest = segments_dir / f"segment_{start:06d}_{end:06d}.jsonl"
    if not complete.is_file() or not accepted.is_file() or not manifest.is_file():
        raise RuntimeError(
            "Render segment is missing completion, accepted, or manifest output: "
            f"{complete}, {accepted}, {manifest}"
        )
    with open(complete, "r", encoding="utf-8-sig") as stream:
        complete_data = json.load(stream)
    with open(accepted, "r", encoding="utf-8-sig") as stream:
        accepted_data = json.load(stream)
    if not complete_data.get("valid") or not accepted_data.get("valid"):
        raise RuntimeError("Render segment markers are not valid")
    if str(accepted_data.get("run_id")) != run_id:
        raise RuntimeError("Render accepted marker does not belong to this run")
    if str(accepted_data.get("manifest_sha256")) != file_sha256(manifest):
        raise RuntimeError("Render accepted marker does not bind the current manifest")
    if str(accepted_data.get("config_hash")) != str(job["config_hash"]):
        raise RuntimeError("Render accepted marker has another config hash")
    if str(accepted_data.get("render_provenance_hash")) != expected_render_provenance:
        raise RuntimeError("Render accepted marker has another render provenance")
    print(f"[pipeline] completed frames {start:06d}..{end:06d}", flush=True)


def finalize_render(job_dir: Path) -> dict[str, Any]:
    job = load_job(job_dir)
    segments_dir = job_path(job_dir, job, "render_segments_dir")
    expected_render_provenance, _ = current_render_provenance(job_dir, job)
    collected: dict[int, dict[str, Any]] = {}
    for start, end in canonical_render_segments(job):
        stem = f"segment_{start:06d}_{end:06d}"
        manifest = segments_dir / f"{stem}.jsonl"
        complete_path = segments_dir / f"{stem}_complete.json"
        accepted_path = segments_dir / f"{stem}_accepted.json"
        if not manifest.is_file() or not complete_path.is_file() or not accepted_path.is_file():
            raise RuntimeError(f"Canonical segment {stem} is not fully accepted")
        with open(complete_path, "r", encoding="utf-8-sig") as stream:
            complete = json.load(stream)
        with open(accepted_path, "r", encoding="utf-8-sig") as stream:
            accepted = json.load(stream)
        if not complete.get("valid") or not accepted.get("valid"):
            raise RuntimeError(f"Canonical segment {stem} has an invalid marker")
        if str(accepted.get("manifest_sha256")) != file_sha256(manifest):
            raise RuntimeError(f"Accepted marker does not bind {manifest}")
        if str(accepted.get("render_provenance_hash")) != expected_render_provenance:
            raise RuntimeError(f"Canonical segment {stem} has stale render provenance")
        segment_rows = read_jsonl(manifest)
        segment_indices = [int(row["output_index"]) for row in segment_rows]
        if segment_indices != list(range(start, end + 1)):
            raise RuntimeError(
                f"Canonical segment {stem} has wrong or duplicate indices: "
                f"{segment_indices[:10]}..."
            )
        for row in segment_rows:
            index = int(row["output_index"])
            if index in collected:
                raise RuntimeError(f"Overlapping render manifests contain frame {index}")
            if str(row.get("render_provenance_hash")) != expected_render_provenance:
                raise RuntimeError(f"Frame {index} has stale render provenance")
            collected[index] = row
    rows = validate_contiguous_manifest(
        collected.values(),
        expected_frames=int(job["video"]["output_frames"]),
        expected_take_id=str(job["take_id"]),
        expected_config_hash=str(job["config_hash"]),
        expected_simulation_provenance_hash=str(
            job["simulation_provenance_hash"]
        ),
    )
    frames_dir = job_path(job_dir, job, "frames_dir")
    width = int(job["video"]["width"])
    height = int(job["video"]["height"])
    for row in rows:
        validate_png(
            frames_dir / f"rgb_{int(row['output_index']):06d}.png",
            width=width,
            height=height,
            expected_sha256=str(row["png_sha256"]),
        )
    manifest_path = job_path(job_dir, job, "render_manifest")
    atomic_write_jsonl(manifest_path, rows)
    timings = [float(row["render_seconds"]) for row in rows]
    sorted_timings = sorted(timings)
    p95_index = min(len(sorted_timings) - 1, max(0, int(round(0.95 * len(sorted_timings))) - 1))
    result = {
        "schema_version": 1,
        "valid": True,
        "completed_at": utc_now(),
        "take_id": job["take_id"],
        "config_hash": job["config_hash"],
        "simulation_provenance_hash": job["simulation_provenance_hash"],
        "render_provenance_hash": expected_render_provenance,
        "frame_count": len(rows),
        "width": width,
        "height": height,
        "fps": int(job["video"]["output_fps"]),
        "renderer": job["render"]["renderer"],
        "path_spp": int(job["render"]["path_spp"]),
        "render_seconds_total": sum(timings),
        "render_seconds_median": statistics.median(timings),
        "render_seconds_p95": sorted_timings[p95_index],
    }
    atomic_write_json(job_path(job_dir, job, "render_complete"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_encoder(job_dir: Path, *, pilot: bool) -> None:
    command = [sys.executable, str(ENCODER_PATH), "--job", str(job_dir)]
    if pilot:
        command.extend(
            [
                "--start",
                "0",
                "--frame-count",
                str(int(load_job(job_dir)["video"]["output_fps"])),
                "--allow-partial",
                "--output",
                str(job_path(job_dir, load_job(job_dir), "reports_dir") / "pilot_1s_h264.mp4"),
            ]
        )
    subprocess.run(command, check=True)


def show_status(job_dir: Path) -> None:
    job = load_job(job_dir)
    result: dict[str, Any] = {
        "job": str(job_dir),
        "job_id": job["job_id"],
        "config_hash": job["config_hash"],
        "expected_frames": int(job["video"]["output_frames"]),
    }
    for name in ("simulation_complete", "render_complete"):
        path = job_path(job_dir, job, name)
        result[name] = str(path) if path.is_file() else None
    cache_manifest = job_path(job_dir, job, "simulation_manifest")
    render_manifest = job_path(job_dir, job, "render_manifest")
    result["cached_frames"] = len(read_jsonl(cache_manifest)) if cache_manifest.is_file() else 0
    if render_manifest.is_file():
        rendered_indices = {
            int(row["output_index"]) for row in read_jsonl(render_manifest)
        }
    else:
        rendered_indices = set()
        segments_dir = job_path(job_dir, job, "render_segments_dir")
        for start, end in canonical_render_segments(job):
            stem = f"segment_{start:06d}_{end:06d}"
            segment_manifest = segments_dir / f"{stem}.jsonl"
            accepted_path = segments_dir / f"{stem}_accepted.json"
            if not segment_manifest.is_file() or not accepted_path.is_file():
                continue
            with open(accepted_path, "r", encoding="utf-8-sig") as stream:
                accepted = json.load(stream)
            if (
                accepted.get("valid")
                and str(accepted.get("manifest_sha256"))
                == file_sha256(segment_manifest)
            ):
                rendered_indices.update(
                    int(row["output_index"])
                    for row in read_jsonl(segment_manifest)
                )
    result["rendered_frames"] = len(rendered_indices)
    result["next_missing_render_frame"] = next(
        (
            index
            for index in range(int(job["video"]["output_frames"]))
            if index not in rendered_indices
        ),
        None,
    )
    result["segment_completions"] = len(
        list(job_path(job_dir, job, "render_segments_dir").glob("segment_*_accepted.json"))
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--job-id", required=True)
    init.add_argument("--root", default=str(DEFAULT_OUTPUT_ROOT))
    init.add_argument("--duration", type=float, default=10.0)
    init.add_argument("--fps", type=int, default=30)
    init.add_argument("--physics-fps", type=int, default=60)
    init.add_argument("--width", type=int, default=1280)
    init.add_argument("--height", type=int, default=720)
    init.add_argument("--spp", type=int, default=32)
    init.add_argument("--camera", default="stream-front")
    init.add_argument("--substeps", type=int, default=4)
    init.add_argument("--solver-iterations", type=int, default=8)
    init.add_argument("--spacing", type=float, default=0.0015)
    init.add_argument("--segment-frames", type=int, default=50)
    init.add_argument("--min-free-gib", type=float, default=60.0)
    init.add_argument("--vram-limit-mib", type=int, default=10752)
    init.add_argument("--vram-target-mib", type=int, default=9472)

    for name in ("status", "simulate", "validate-cache", "render-pilot", "render-all", "finalize-render", "encode"):
        command = subparsers.add_parser(name)
        command.add_argument("--job", required=True)
        if name == "validate-cache":
            command.add_argument("--full", action="store_true")
        if name == "render-all":
            command.add_argument("--confirm-production", default="")

    segment = subparsers.add_parser("render-segment")
    segment.add_argument("--job", required=True)
    segment.add_argument("--start", type=int, required=True)
    segment.add_argument("--end", type=int, required=True)

    args = parser.parse_args()
    if args.command == "init":
        initialize_job(args)
        return
    job_dir = Path(args.job).resolve()
    if args.command == "status":
        show_status(job_dir)
    elif args.command == "simulate":
        run_windows_simulation(job_dir)
    elif args.command == "validate-cache":
        validate_cache(job_dir, full=args.full)
    elif args.command == "render-segment":
        run_windows_render_segment(job_dir, args.start, args.end)
    elif args.command == "render-pilot":
        job = load_job(job_dir)
        start, end = canonical_render_segments(job)[0]
        run_windows_render_segment(job_dir, start, end)
        run_encoder(job_dir, pilot=True)
    elif args.command == "render-all":
        job = load_job(job_dir)
        expected = int(job["video"]["output_frames"])
        if args.confirm_production != f"{expected}-frames":
            raise RuntimeError(
                "Full production rendering requires the explicit flag "
                f"--confirm-production {expected}-frames"
            )
        for start, end in canonical_render_segments(job):
            run_windows_render_segment(job_dir, start, end)
        finalize_render(job_dir)
    elif args.command == "finalize-render":
        finalize_render(job_dir)
    elif args.command == "encode":
        run_encoder(job_dir, pilot=False)


if __name__ == "__main__":
    main()
