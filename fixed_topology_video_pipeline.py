"""WSL-side orchestration for cached fixed-topology simulation videos."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fixed_topology_video import (
    JOB_TYPE,
    immutable_job_config,
    job_path,
    load_frame,
    load_job,
    load_topology,
    render_provenance_hash,
    simulation_provenance_payload,
    validate_job,
    validate_manifest,
)
from liquid_video_cache import (
    atomic_write_json,
    atomic_write_jsonl,
    config_hash,
    expected_output_frames,
    file_sha256,
    read_jsonl,
    require_integer_capture_stride,
    validate_png,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output" / "deformable_video"
BATCH_PATH = PROJECT_DIR / "run_fixed_topology_video.bat"
ENCODER_PATH = PROJECT_DIR / "encode_fixed_topology_video.py"
RENDERER_PATH = PROJECT_DIR / "render_fixed_topology_video.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0].strip() if lines else None


def wsl_to_windows(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        return f"{parts[2].upper()}:\\" + "\\".join(parts[3:])
    raise ValueError(f"Expected a Windows-mounted WSL path such as /mnt/y: {resolved}")


def ensure_free_space(path: Path, minimum_gib: float) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    if free_gib < minimum_gib:
        raise RuntimeError(f"Insufficient free space: {free_gib:.2f} GiB < {minimum_gib:.2f} GiB")
    return {
        "total_gib": usage.total / 1024**3,
        "used_gib": usage.used / 1024**3,
        "free_gib": free_gib,
        "required_free_gib": minimum_gib,
    }


def script_hashes(simulation_script: str) -> dict[str, str]:
    names = [
        simulation_script,
        "soft_body_bounce_hero.py",
        "fixed_topology_video.py",
        "fixed_topology_video_pipeline.py",
        "render_fixed_topology_video.py",
        "encode_fixed_topology_video.py",
        "run_fixed_topology_video_stage.ps1",
        "run_fixed_topology_video.bat",
    ]
    result: dict[str, str] = {}
    for name in names:
        path = PROJECT_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required pipeline script: {path}")
        result[name] = file_sha256(path)
    return result


def initialize_soft_body_job(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.job_id):
        raise ValueError("job-id must be one safe basename")
    output_frames = expected_output_frames(args.duration, args.fps)
    capture_stride = require_integer_capture_stride(args.physics_fps, args.fps)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    disk = ensure_free_space(root, args.min_free_gib)
    job_dir = (root / args.job_id).resolve()
    if job_dir.parent != root or job_dir.exists():
        raise FileExistsError(f"Refusing to overwrite or escape output root: {job_dir}")
    model = Path(args.model).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    model_windows = wsl_to_windows(model)
    take_id = f"take_{uuid.uuid4().hex}"
    render_profile = f"path_{args.width}x{args.height}_{args.fps}fps_{args.spp}spp"
    paths = {
        "take_dir": f"takes/{take_id}",
        "cache_dir": f"takes/{take_id}/cache",
        "topology": f"takes/{take_id}/topology.npz",
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
    }
    simulation = {
        "script": "soft_body_fixed_topology_adapter.py",
        "cache_schema": "fixed_topology_points_v1",
        "physics_fps": args.physics_fps,
        "capture_stride": capture_stride,
        "substeps": args.substeps,
        "physics_steps": (output_frames - 1) * capture_stride,
        "dynamic_root_prim": "/World/SoftBall",
        "visual_mesh_prim": "/World/SoftBall/Visual",
        "camera_prim": "/World/HeroCamera",
        "hidden_prims": [
            "/World/SoftBall/SimulationMesh",
            "/World/SoftBall/CollisionMesh",
        ],
        "parameters": {
            "model": model_windows,
            "model_sha256": file_sha256(model),
            "model_height": args.model_height,
            "model_yaw": args.model_yaw,
            "drop_height": args.drop_height,
            "youngs_modulus": args.youngs_modulus,
            "linear_damping": args.linear_damping,
            "poissons_ratio": args.poissons_ratio,
            "density": args.density,
            "deformable_resolution": args.deformable_resolution,
            "self_collision_filter_distance": args.self_collision_filter_distance,
        },
    }
    video = {
        "duration_seconds": args.duration,
        "output_fps": args.fps,
        "output_frames": output_frames,
        "width": args.width,
        "height": args.height,
    }
    render = {
        "renderer": "PathTracing",
        "path_spp": args.spp,
        "segment_frames": args.segment_frames,
        "profile": render_profile,
        "camera": "template-camera",
    }
    base = {
        "job_type": JOB_TYPE,
        "adapter": "soft_body_bounce",
        "take_id": take_id,
        "video": video,
        "simulation": simulation,
        "render": render,
        "paths": paths,
    }
    digest = config_hash(base)
    hashes = script_hashes(simulation["script"])
    job: dict[str, Any] = {
        "schema_version": 1,
        "job_type": JOB_TYPE,
        "adapter": "soft_body_bounce",
        "job_id": args.job_id,
        "job_uuid": str(uuid.uuid4()),
        "created_at": utc_now(),
        "status": "initialized",
        "take_id": take_id,
        "config_hash": digest,
        "simulation_provenance_hash": "pending",
        "video": video,
        "simulation": simulation,
        "render": render,
        "paths": paths,
        "resource_budget": {
            "minimum_free_gib": args.min_free_gib,
            "hard_vram_limit_mib": args.vram_limit_mib,
            "initial_disk": disk,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "ffmpeg": command_version(["ffmpeg", "-version"]),
            "ffprobe": command_version(["ffprobe", "-version"]),
            "nvidia_smi": command_version(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
            ),
        },
        "script_sha256": hashes,
        "timeline": {
            "frame_zero": "initial_pre_simulation_state",
            "sample_times": "output_index/output_fps",
            "last_sample_seconds": (output_frames - 1) / args.fps,
            "container_duration_seconds": output_frames / args.fps,
            "uniform": True,
        },
    }
    if config_hash(immutable_job_config(job)) != digest:
        raise RuntimeError("Internal job config hashing error")
    job["simulation_provenance_hash"] = config_hash(simulation_provenance_payload(job))
    for name in ("cache_dir", "frames_dir", "render_segments_dir", "video_dir", "reports_dir", "logs_dir"):
        (job_dir / paths[name]).mkdir(parents=True, exist_ok=True)
    validate_job(job)
    atomic_write_json(job_dir / "job.json", job)
    print(json.dumps({"job": str(job_dir), "frames": output_frames, "config_hash": digest}, indent=2))


def validate_cache(job_dir: Path, *, full: bool) -> dict[str, Any]:
    job = load_job(job_dir)
    complete_path = job_path(job_dir, job, "simulation_complete")
    if not complete_path.is_file():
        raise RuntimeError(f"Missing simulation completion marker: {complete_path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8-sig"))
    for key in ("take_id", "config_hash", "simulation_provenance_hash"):
        if str(complete.get(key)) != str(job[key]):
            raise RuntimeError(f"Simulation completion has another {key}")
    if not complete.get("valid") or int(complete.get("frame_count", -1)) != int(job["video"]["output_frames"]):
        raise RuntimeError("Simulation completion is invalid or incomplete")
    template = job_path(job_dir, job, "render_template")
    topology_path = job_path(job_dir, job, "topology")
    if file_sha256(template) != str(complete.get("render_template_sha256")):
        raise RuntimeError("Render template hash changed")
    topology = load_topology(topology_path, expected_sha256=str(complete["topology_sha256"]))
    rows = validate_manifest(read_jsonl(job_path(job_dir, job, "simulation_manifest")), job=job)
    total_bytes = topology_path.stat().st_size
    cache_dir = job_path(job_dir, job, "cache_dir")
    for row in rows:
        frame = cache_dir / str(row["cache_file"])
        if not frame.is_file() or frame.stat().st_size != int(row["cache_bytes"]):
            raise RuntimeError(f"Missing or size-mismatched frame cache: {frame}")
        total_bytes += frame.stat().st_size
        if full:
            load_frame(
                frame,
                expected_sha256=str(row["cache_sha256"]),
                expected_vertex_count=int(topology["metadata"]["vertex_count"]),
            )
        elif file_sha256(frame) != str(row["cache_sha256"]):
            raise RuntimeError(f"Frame cache hash mismatch: {frame}")
    result = {
        "schema_version": 1,
        "valid": True,
        "validated_at": utc_now(),
        "validation_mode": "full_arrays" if full else "hash_only",
        "frame_count": len(rows),
        "vertex_count": int(topology["metadata"]["vertex_count"]),
        "cache_bytes": total_bytes,
        "cache_gib": total_bytes / 1024**3,
        "take_id": job["take_id"],
        "config_hash": job["config_hash"],
        "simulation_provenance_hash": job["simulation_provenance_hash"],
    }
    atomic_write_json(job_path(job_dir, job, "reports_dir") / "cache_validation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def canonical_segments(job: dict[str, Any]) -> list[tuple[int, int]]:
    count = int(job["video"]["output_frames"])
    size = int(job["render"]["segment_frames"])
    return [(start, min(count - 1, start + size - 1)) for start in range(0, count, size)]


def run_stage(job_dir: Path, stage: str, *, start: int = -1, end: int = -1) -> None:
    job = load_job(job_dir)
    run_id = uuid.uuid4().hex
    command = ["cmd.exe", "/d", "/s", "/c", wsl_to_windows(BATCH_PATH), stage, wsl_to_windows(job_dir)]
    if stage == "render-segment":
        command.extend([str(start), str(end), run_id])
    else:
        command.append(run_id)
    subprocess.run(command, check=True)
    if stage == "simulate-cache":
        accepted = job_path(job_dir, job, "take_dir") / "simulation_accepted.json"
    else:
        accepted = job_path(job_dir, job, "render_segments_dir") / f"segment_{start:06d}_{end:06d}_accepted.json"
    if not accepted.is_file():
        raise RuntimeError(f"Stage did not write its accepted marker: {accepted}")
    data = json.loads(accepted.read_text(encoding="utf-8-sig"))
    if not data.get("valid") or str(data.get("run_id")) != run_id:
        raise RuntimeError(f"Stage accepted marker is invalid: {accepted}")


def current_render_provenance(job_dir: Path, job: dict[str, Any]) -> str:
    complete = json.loads(job_path(job_dir, job, "simulation_complete").read_text(encoding="utf-8-sig"))
    return render_provenance_hash(
        job,
        template_sha256=file_sha256(job_path(job_dir, job, "render_template")),
        topology_sha256=file_sha256(job_path(job_dir, job, "topology")),
        renderer_sha256=file_sha256(RENDERER_PATH),
        isaac_version=complete.get("isaac_version"),
    )


def finalize_render(job_dir: Path) -> dict[str, Any]:
    job = load_job(job_dir)
    expected_provenance = current_render_provenance(job_dir, job)
    segments_dir = job_path(job_dir, job, "render_segments_dir")
    rows: list[dict[str, Any]] = []
    for start, end in canonical_segments(job):
        stem = f"segment_{start:06d}_{end:06d}"
        manifest = segments_dir / f"{stem}.jsonl"
        accepted_path = segments_dir / f"{stem}_accepted.json"
        if not manifest.is_file() or not accepted_path.is_file():
            raise RuntimeError(f"Missing accepted render segment: {stem}")
        accepted = json.loads(accepted_path.read_text(encoding="utf-8-sig"))
        if not accepted.get("valid") or accepted.get("manifest_sha256") != file_sha256(manifest):
            raise RuntimeError(f"Stale accepted render segment: {stem}")
        if accepted.get("render_provenance_hash") != expected_provenance:
            raise RuntimeError(f"Render segment has another provenance: {stem}")
        rows.extend(read_jsonl(manifest))
    count = int(job["video"]["output_frames"])
    by_index = {int(row["output_index"]): row for row in rows}
    if sorted(by_index) != list(range(count)) or len(rows) != count:
        raise RuntimeError("Render segments overlap or do not cover the complete timeline")
    width, height = int(job["video"]["width"]), int(job["video"]["height"])
    for index in range(count):
        validate_png(
            job_path(job_dir, job, "frames_dir") / f"rgb_{index:06d}.png",
            width=width,
            height=height,
            expected_sha256=str(by_index[index]["png_sha256"]),
        )
    manifest_path = job_path(job_dir, job, "render_manifest")
    atomic_write_jsonl(manifest_path, [by_index[index] for index in range(count)])
    result = {
        "schema_version": 1,
        "valid": True,
        "frame_count": count,
        "take_id": job["take_id"],
        "config_hash": job["config_hash"],
        "simulation_provenance_hash": job["simulation_provenance_hash"],
        "render_provenance_hash": expected_provenance,
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_json(job_path(job_dir, job, "render_complete"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def show_status(job_dir: Path) -> None:
    job = load_job(job_dir)
    manifest = job_path(job_dir, job, "simulation_manifest")
    cached = len(read_jsonl(manifest)) if manifest.is_file() else 0
    rendered: set[int] = set()
    segments_dir = job_path(job_dir, job, "render_segments_dir")
    for path in segments_dir.glob("segment_*_accepted.json"):
        if "_rejected_" in path.name:
            continue
        manifest_path = segments_dir / path.name.replace("_accepted.json", ".jsonl")
        if manifest_path.is_file():
            rendered.update(int(row["output_index"]) for row in read_jsonl(manifest_path))
    result = {
        "job": str(job_dir),
        "job_id": job["job_id"],
        "adapter": job["adapter"],
        "expected_frames": job["video"]["output_frames"],
        "cached_frames": cached,
        "rendered_frames": len(rendered),
        "next_missing_render_frame": next(
            (i for i in range(int(job["video"]["output_frames"])) if i not in rendered), None
        ),
        "simulation_complete": job_path(job_dir, job, "simulation_complete").is_file(),
        "render_complete": job_path(job_dir, job, "render_complete").is_file(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-soft-body")
    init.add_argument("--job-id", required=True)
    init.add_argument("--root", default=str(DEFAULT_OUTPUT_ROOT))
    init.add_argument("--model", default=str(PROJECT_DIR / "assets" / "soft_body_elephant.stl"))
    init.add_argument("--duration", type=float, default=2.5)
    init.add_argument("--fps", type=int, default=60)
    init.add_argument("--physics-fps", type=int, default=60)
    init.add_argument("--width", type=int, default=960)
    init.add_argument("--height", type=int, default=960)
    init.add_argument("--spp", type=int, default=32)
    init.add_argument("--segment-frames", type=int, default=50)
    init.add_argument("--substeps", type=int, default=4)
    init.add_argument("--model-height", type=float, default=1.45)
    init.add_argument("--model-yaw", type=float, default=-56.0)
    init.add_argument("--drop-height", type=float, default=3.0)
    init.add_argument("--youngs-modulus", type=float, default=110000.0)
    init.add_argument("--linear-damping", type=float, default=1.35)
    init.add_argument("--poissons-ratio", type=float, default=0.45)
    init.add_argument("--density", type=float, default=1050.0)
    init.add_argument("--deformable-resolution", type=int, default=24)
    init.add_argument("--self-collision-filter-distance", type=float, default=0.05)
    init.add_argument("--min-free-gib", type=float, default=10.0)
    init.add_argument("--vram-limit-mib", type=int, default=10752)
    for name in ("status", "simulate", "validate-cache", "render-pilot", "render-all", "finalize-render", "encode"):
        command = sub.add_parser(name)
        command.add_argument("--job", required=True)
        if name == "validate-cache":
            command.add_argument("--full", action="store_true")
        if name == "render-all":
            command.add_argument("--confirm-production", default="")
    segment = sub.add_parser("render-segment")
    segment.add_argument("--job", required=True)
    segment.add_argument("--start", type=int, required=True)
    segment.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    if args.command == "init-soft-body":
        initialize_soft_body_job(args)
        return
    job_dir = Path(args.job).resolve()
    if args.command == "status":
        show_status(job_dir)
    elif args.command == "simulate":
        run_stage(job_dir, "simulate-cache")
        validate_cache(job_dir, full=False)
    elif args.command == "validate-cache":
        validate_cache(job_dir, full=args.full)
    elif args.command == "render-segment":
        job = load_job(job_dir)
        if (args.start, args.end) not in canonical_segments(job):
            raise ValueError("Requested range is not a canonical render segment")
        run_stage(job_dir, "render-segment", start=args.start, end=args.end)
    elif args.command == "render-pilot":
        job = load_job(job_dir)
        start, end = canonical_segments(job)[0]
        run_stage(job_dir, "render-segment", start=start, end=end)
        pilot_count = end - start + 1
        output_frames = int(job["video"]["output_frames"])
        encoder_command = [
            sys.executable,
            str(ENCODER_PATH),
            "--job", str(job_dir),
            "--start", str(start),
            "--frame-count", str(pilot_count),
        ]
        if pilot_count == output_frames:
            finalize_render(job_dir)
        else:
            encoder_command.append("--allow-partial")
        subprocess.run(
            encoder_command,
            check=True,
        )
    elif args.command == "render-all":
        job = load_job(job_dir)
        count = int(job["video"]["output_frames"])
        if args.confirm_production != f"{count}-frames":
            raise RuntimeError(f"Full rendering requires --confirm-production {count}-frames")
        for start, end in canonical_segments(job):
            run_stage(job_dir, "render-segment", start=start, end=end)
        finalize_render(job_dir)
    elif args.command == "finalize-render":
        finalize_render(job_dir)
    elif args.command == "encode":
        subprocess.run([sys.executable, str(ENCODER_PATH), "--job", str(job_dir)], check=True)


if __name__ == "__main__":
    main()
