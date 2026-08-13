"""Render fixed-topology cached mesh frames without running PhysX."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from fixed_topology_video import (
    job_path,
    load_frame,
    load_job,
    load_topology,
    render_provenance_hash,
)
from liquid_video_cache import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    read_jsonl,
    validate_png,
)


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--job", required=True)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--capture-timeout-updates", type=int, default=900)
parser.add_argument("--mesh-settle-updates", type=int, default=4)
parser.add_argument("--no-resume", action="store_true")
args = parser.parse_args()

job_dir = Path(args.job).resolve()
job = load_job(job_dir)
video, simulation, render = job["video"], job["simulation"], job["render"]
expected_frames = int(video["output_frames"])
segment_size = int(render["segment_frames"])
expected_end = min(expected_frames - 1, args.start + segment_size - 1)
if args.start < 0 or args.start % segment_size or args.end != expected_end:
    raise ValueError(f"Non-canonical render segment {args.start}..{args.end}")

width, height = int(video["width"]), int(video["height"])
path_spp = int(render["path_spp"])
take_id, config_hash = str(job["take_id"]), str(job["config_hash"])
cache_dir = job_path(job_dir, job, "cache_dir")
template_path = job_path(job_dir, job, "render_template")
topology_path = job_path(job_dir, job, "topology")
manifest_path = job_path(job_dir, job, "simulation_manifest")
completion_path = job_path(job_dir, job, "simulation_complete")
frames_dir = job_path(job_dir, job, "frames_dir")
segments_dir = job_path(job_dir, job, "render_segments_dir")
frames_dir.mkdir(parents=True, exist_ok=True)
segments_dir.mkdir(parents=True, exist_ok=True)
segment_name = f"segment_{args.start:06d}_{args.end:06d}"
segment_manifest = segments_dir / f"{segment_name}.jsonl"
segment_complete = segments_dir / f"{segment_name}_complete.json"
segment_accepted = segments_dir / f"{segment_name}_accepted.json"

completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
if not completion.get("valid") or int(completion.get("frame_count", -1)) != expected_frames:
    raise RuntimeError("Simulation completion marker is invalid")
for key in ("take_id", "config_hash", "simulation_provenance_hash"):
    if str(completion.get(key)) != str(job[key]):
        raise RuntimeError(f"Simulation completion has another {key}")
template_hash = file_sha256(template_path)
topology_hash = file_sha256(topology_path)
if template_hash != completion.get("render_template_sha256"):
    raise RuntimeError("Render template hash changed")
if topology_hash != completion.get("topology_sha256"):
    raise RuntimeError("Static topology hash changed")
topology = load_topology(topology_path, expected_sha256=topology_hash)
vertex_count = int(topology["metadata"]["vertex_count"])
rows = read_jsonl(manifest_path)
cache_rows = {int(row["output_index"]): row for row in rows}
for index in range(args.start, args.end + 1):
    row = cache_rows.get(index)
    if row is None:
        raise RuntimeError(f"Simulation manifest lacks frame {index}")
    for key in ("take_id", "config_hash", "simulation_provenance_hash"):
        if str(row.get(key)) != str(job[key]):
            raise RuntimeError(f"Frame {index} has another {key}")

current_provenance = render_provenance_hash(
    job,
    template_sha256=template_hash,
    topology_sha256=topology_hash,
    renderer_sha256=file_sha256(Path(__file__).resolve()),
    isaac_version=completion.get("isaac_version"),
)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "renderer": "PathTracing", "width": width, "height": height}
)

import carb
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, Vt


def update(count: int) -> None:
    for _ in range(max(0, count)):
        simulation_app.update()


def capture(viewport, path: Path) -> None:
    if path.exists():
        path.unlink()
    request = capture_viewport_to_file(viewport, file_path=str(path))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(args.capture_timeout_updates):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        task.cancel()
        raise RuntimeError(f"Timed out waiting for capture: {path}")
    if not task.result():
        raise RuntimeError(f"Capture returned no AOVs: {path}")
    for _ in range(args.capture_timeout_updates):
        if path.is_file() and path.stat().st_size > 0:
            return
        simulation_app.update()
    raise RuntimeError(f"Capture did not produce a non-empty file: {path}")


context = omni.usd.get_context()
if not context.open_stage(str(template_path)):
    raise RuntimeError(f"Failed to open render template: {template_path}")
update(4)
stage = context.get_stage()
if stage is None:
    raise RuntimeError("Render template did not produce a stage")

visual = UsdGeom.Mesh.Get(stage, Sdf.Path(simulation["visual_mesh_prim"]))
if not visual:
    raise RuntimeError(f"Template lacks visual mesh {simulation['visual_mesh_prim']}")
visual.GetFaceVertexCountsAttr().Set(
    Vt.IntArray.FromNumpy(topology["face_vertex_counts"])
)
visual.GetFaceVertexIndicesAttr().Set(
    Vt.IntArray.FromNumpy(topology["face_vertex_indices"])
)
root_prim = stage.GetPrimAtPath(Sdf.Path(simulation["dynamic_root_prim"]))
if not root_prim:
    raise RuntimeError(f"Template lacks dynamic root {simulation['dynamic_root_prim']}")
for prim_path in simulation.get("hidden_prims", []):
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if prim:
        UsdGeom.Imageable(prim).MakeInvisible()

camera = UsdGeom.Camera.Get(stage, Sdf.Path(simulation["camera_prim"]))
if not camera:
    raise RuntimeError(f"Template lacks camera {simulation['camera_prim']}")
viewport = get_active_viewport()
viewport.camera_path = camera.GetPath()
viewport.set_texture_resolution((width, height))

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", "PathTracing")
settings.set("/persistent/app/viewport/displayOptions", 0)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/indirectDiffuse/enabled", True)
settings.set("/rtx/pathtracing/spp", path_spp)
settings.set("/rtx/pathtracing/totalSpp", path_spp)
settings.set("/rtx/pathtracing/maxBounces", 8)
update(3)

existing_rows: list[dict] = []
accepted_resume = False
if segment_manifest.is_file() and segment_accepted.is_file():
    prior = json.loads(segment_accepted.read_text(encoding="utf-8-sig"))
    accepted_resume = bool(
        prior.get("valid")
        and prior.get("manifest_sha256") == file_sha256(segment_manifest)
        and prior.get("render_provenance_hash") == current_provenance
    )
if accepted_resume:
    existing_rows = read_jsonl(segment_manifest)
else:
    for path in (segment_manifest, segment_complete, segment_accepted):
        if path.exists():
            os.replace(path, path.with_name(f"{path.stem}_rejected_{args.run_id}{path.suffix}"))
existing_by_index = {int(row["output_index"]): row for row in existing_rows}
rendered, skipped = 0, 0

try:
    for index in range(args.start, args.end + 1):
        output_path = frames_dir / f"rgb_{index:06d}.png"
        existing = existing_by_index.get(index)
        if not args.no_resume and existing is not None:
            try:
                if (
                    existing.get("render_provenance_hash") == current_provenance
                    and existing.get("take_id") == take_id
                    and existing.get("config_hash") == config_hash
                ):
                    validate_png(
                        output_path,
                        width=width,
                        height=height,
                        expected_sha256=str(existing["png_sha256"]),
                    )
                    skipped += 1
                    continue
            except (KeyError, ValueError):
                pass
        cache_row = cache_rows[index]
        cached = load_frame(
            cache_dir / str(cache_row["cache_file"]),
            expected_sha256=str(cache_row["cache_sha256"]),
            expected_vertex_count=vertex_count,
        )
        if int(cached["metadata"]["output_index"]) != index:
            raise RuntimeError(f"Cache has the wrong output index: {index}")
        visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(cached["points"]))
        visual.GetExtentAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(*(float(v) for v in cached["bounds_min"])),
                    Gf.Vec3f(*(float(v) for v in cached["bounds_max"])),
                ]
            )
        )
        translate_attr = root_prim.GetAttribute("xformOp:translate")
        if not translate_attr:
            translate_attr = UsdGeom.Xformable(root_prim).AddTranslateOp().GetAttr()
        translate_attr.Set(Gf.Vec3d(*(float(v) for v in cached["translate"])))
        started = time.perf_counter()
        update(args.mesh_settle_updates)
        temporary = frames_dir / f".rgb_{index:06d}.part.png"
        capture(viewport, temporary)
        validate_png(temporary, width=width, height=height)
        os.replace(temporary, output_path)
        png = validate_png(output_path, width=width, height=height)
        row = {
            "schema_version": 1,
            "status": "rendered",
            "output_index": index,
            "sim_step": int(cache_row["sim_step"]),
            "sim_time_seconds": float(cache_row["sim_time_seconds"]),
            "take_id": take_id,
            "config_hash": config_hash,
            "simulation_provenance_hash": job["simulation_provenance_hash"],
            "render_provenance_hash": current_provenance,
            "run_id": args.run_id,
            "cache_file": cache_row["cache_file"],
            "cache_sha256": cache_row["cache_sha256"],
            **png,
            "renderer": "PathTracing",
            "path_spp": path_spp,
            "render_seconds": time.perf_counter() - started,
        }
        existing_by_index[index] = row
        atomic_write_jsonl(segment_manifest, [existing_by_index[i] for i in sorted(existing_by_index)])
        rendered += 1

    final_rows = {int(row["output_index"]): row for row in read_jsonl(segment_manifest)}
    for index in range(args.start, args.end + 1):
        row = final_rows.get(index)
        if row is None:
            raise RuntimeError(f"Segment manifest lacks frame {index}")
        validate_png(
            frames_dir / f"rgb_{index:06d}.png",
            width=width,
            height=height,
            expected_sha256=str(row["png_sha256"]),
        )
    atomic_write_json(
        segment_complete,
        {
            "schema_version": 1,
            "valid": True,
            "segment": segment_name,
            "start": args.start,
            "end": args.end,
            "frame_count": args.end - args.start + 1,
            "rendered_this_run": rendered,
            "skipped_verified": skipped,
            "take_id": take_id,
            "config_hash": config_hash,
            "simulation_provenance_hash": job["simulation_provenance_hash"],
            "render_provenance_hash": current_provenance,
            "run_id": args.run_id,
            "manifest_sha256": file_sha256(segment_manifest),
        },
    )
finally:
    simulation_app.close()
