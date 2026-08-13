"""Render validated liquid surface caches as independent PathTracing frames."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np

from liquid_video_cache import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    load_surface_cache,
    read_jsonl,
    render_provenance_hash,
    validate_job,
    validate_png,
)


os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True, help="Inclusive output frame")
parser.add_argument("--run-id", required=True)
parser.add_argument("--capture-timeout-updates", type=int, default=900)
parser.add_argument("--mesh-settle-updates", type=int, default=4)
parser.add_argument("--no-resume", action="store_true")
args = parser.parse_args()

job_dir = Path(args.job).resolve()
job_path = job_dir / "job.json"
with open(job_path, "r", encoding="utf-8-sig") as stream:
    job = validate_job(json.load(stream))

video = job["video"]
render = job["render"]
paths = job["paths"]
expected_frames = int(video["output_frames"])
segment_frames = int(render["segment_frames"])
expected_segment_end = min(expected_frames - 1, args.start + segment_frames - 1)
if (
    args.start < 0
    or args.start % segment_frames != 0
    or args.end != expected_segment_end
):
    raise ValueError(
        f"Invalid non-canonical frame range {args.start}..{args.end}; "
        f"segment size is {segment_frames}"
    )

width = int(video["width"])
height = int(video["height"])
path_spp = int(render["path_spp"])
config_hash = str(job["config_hash"])
take_id = str(job["take_id"])

def job_relative(name: str) -> Path:
    return job_dir / Path(paths[name])

cache_dir = job_relative("cache_dir")
simulation_manifest_path = job_relative("simulation_manifest")
simulation_complete_path = job_relative("simulation_complete")
template_path = job_relative("render_template")
frames_dir = job_relative("frames_dir")
segments_dir = job_relative("render_segments_dir")
frames_dir.mkdir(parents=True, exist_ok=True)
segments_dir.mkdir(parents=True, exist_ok=True)
segment_name = f"segment_{args.start:06d}_{args.end:06d}"
segment_manifest_path = segments_dir / f"{segment_name}.jsonl"
segment_complete_path = segments_dir / f"{segment_name}_complete.json"
segment_accepted_path = segments_dir / f"{segment_name}_accepted.json"

if not simulation_complete_path.is_file():
    raise RuntimeError(f"Missing simulation completion marker: {simulation_complete_path}")
if not template_path.is_file():
    raise RuntimeError(f"Missing render template: {template_path}")
with open(simulation_complete_path, "r", encoding="utf-8-sig") as stream:
    simulation_complete = json.load(stream)
if not simulation_complete.get("valid"):
    raise RuntimeError("simulation_complete.json is not valid")
if str(simulation_complete.get("take_id")) != take_id:
    raise RuntimeError("simulation_complete.json has another take ID")
if str(simulation_complete.get("config_hash")) != config_hash:
    raise RuntimeError("simulation_complete.json has another config hash")
if str(simulation_complete.get("simulation_provenance_hash")) != str(
    job["simulation_provenance_hash"]
):
    raise RuntimeError("simulation_complete.json has another provenance hash")
if int(simulation_complete.get("frame_count", -1)) != expected_frames:
    raise RuntimeError("simulation_complete.json has the wrong frame count")
template_sha256 = file_sha256(template_path)
if template_sha256 != str(simulation_complete.get("render_template_sha256")):
    raise RuntimeError("Render template hash differs from simulation completion")
renderer_sha256 = file_sha256(Path(__file__).resolve())
current_render_provenance = render_provenance_hash(
    job,
    template_sha256=template_sha256,
    renderer_sha256=renderer_sha256,
    isaac_version=simulation_complete.get("isaac_version"),
)

simulation_rows = read_jsonl(simulation_manifest_path)
cache_rows = {int(row["output_index"]): row for row in simulation_rows}
for output_index in range(args.start, args.end + 1):
    if output_index not in cache_rows:
        raise RuntimeError(f"Simulation manifest is missing frame {output_index}")
    row = cache_rows[output_index]
    if str(row.get("take_id")) != take_id:
        raise RuntimeError(f"Frame {output_index} belongs to another take")
    if str(row.get("config_hash")) != config_hash:
        raise RuntimeError(f"Frame {output_index} has another config hash")
    if str(row.get("simulation_provenance_hash")) != str(
        job["simulation_provenance_hash"]
    ):
        raise RuntimeError(f"Frame {output_index} has another simulation provenance")

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "PathTracing",
        "width": width,
        "height": height,
    }
)

import carb
import omni.kit.commands
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def set_camera(camera_prim, eye, target):
    transform = Gf.Matrix4d().SetLookAt(
        eye,
        target,
        Gf.Vec3d(0.0, 1.0, 0.0),
    ).GetInverse()
    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(transform)


def render_updates(count):
    for _ in range(max(0, count)):
        simulation_app.update()


def capture_viewport(viewport, output_path):
    output_path = str(output_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    capture = capture_viewport_to_file(viewport, file_path=output_path)
    task = asyncio.ensure_future(capture.wait_for_result(completion_frames=2))
    for _ in range(args.capture_timeout_updates):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        task.cancel()
        raise RuntimeError(f"Timed out waiting for capture: {output_path}")
    if not task.result():
        raise RuntimeError(f"Capture returned no AOVs: {output_path}")
    for _ in range(args.capture_timeout_updates):
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return
        simulation_app.update()
    raise RuntimeError(f"Capture did not produce a non-empty file: {output_path}")


context = omni.usd.get_context()
print(f"[cache-render] opening template {template_path}", flush=True)
if not context.open_stage(str(template_path)):
    print(f"[cache-render] failed to open template {template_path}", flush=True)
    raise RuntimeError(f"Failed to open render template: {template_path}")
render_updates(4)
stage = context.get_stage()
if stage is None:
    print("[cache-render] USD context returned no stage", flush=True)
    raise RuntimeError("Render template did not produce a USD stage")
print("[cache-render] render template opened", flush=True)

for prim_path in (
    paths.get("particle_set_prim", "/World/WaterParticles"),
    paths.get("particle_system_prim", "/World/ParticleSystem"),
):
    prim = stage.GetPrimAtPath(prim_path)
    if prim:
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
        else:
            prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set(
                UsdGeom.Tokens.invisible
            )
print("[cache-render] source particle prims hidden", flush=True)

cached_mesh_path = Sdf.Path(paths.get("cached_mesh_prim", "/World/CachedLiquid"))
existing_cached_mesh = stage.GetPrimAtPath(cached_mesh_path)
if existing_cached_mesh:
    stage.RemovePrim(cached_mesh_path)
mesh = UsdGeom.Mesh.Define(stage, cached_mesh_path)
mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
mesh.CreateDoubleSidedAttr().Set(False)
print(f"[cache-render] cached mesh defined at {cached_mesh_path}", flush=True)
water_material_path = Sdf.Path(
    paths.get("water_material_prim", "/World/Looks/WaterRender")
)
water_material = UsdShade.Material.Get(stage, water_material_path)
if not water_material:
    raise RuntimeError(f"Render template lacks water material {water_material_path}")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(water_material)
print(f"[cache-render] water material bound from {water_material_path}", flush=True)

camera_path = Sdf.Path(paths.get("camera_prim", "/World/RenderCamera"))
camera = UsdGeom.Camera.Get(stage, camera_path)
if not camera:
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(26.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    set_camera(
        camera.GetPrim(),
        Gf.Vec3d(0.02, 0.68, 2.15),
        Gf.Vec3d(-0.03, 0.34, 0.0),
    )

viewport = get_active_viewport()
viewport.camera_path = camera.GetPath()
viewport.set_texture_resolution((width, height))
print(f"[cache-render] camera ready at {camera.GetPath()}", flush=True)
settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", "PathTracing")
settings.set("/persistent/app/viewport/displayOptions", 0)
settings.set("/rtx/translucency/maxRefractionBounces", 12)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/indirectDiffuse/enabled", True)
settings.set("/rtx/pathtracing/fractionalCutoutOpacity", True)
settings.set("/rtx/pathtracing/spp", path_spp)
settings.set("/rtx/pathtracing/totalSpp", path_spp)
settings.set("/rtx/pathtracing/maxBounces", 12)
render_updates(3)
print("[cache-render] PathTracing settings ready", flush=True)

existing_rows = []
resume_is_accepted = False
if segment_manifest_path.is_file() and segment_accepted_path.is_file():
    with open(segment_accepted_path, "r", encoding="utf-8-sig") as stream:
        prior_accepted = json.load(stream)
    resume_is_accepted = bool(
        prior_accepted.get("valid")
        and str(prior_accepted.get("manifest_sha256"))
        == file_sha256(segment_manifest_path)
        and str(prior_accepted.get("render_provenance_hash"))
        == current_render_provenance
    )
if resume_is_accepted:
    existing_rows = read_jsonl(segment_manifest_path)
else:
    if segment_manifest_path.exists():
        os.replace(
            segment_manifest_path,
            segments_dir / f"{segment_name}_rejected_{args.run_id}.jsonl",
        )
    if segment_complete_path.exists():
        os.replace(
            segment_complete_path,
            segments_dir / f"{segment_name}_rejected_{args.run_id}_complete.json",
        )
    if segment_accepted_path.exists():
        os.replace(
            segment_accepted_path,
            segments_dir / f"{segment_name}_rejected_{args.run_id}_accepted.json",
        )
existing_by_index = {int(row["output_index"]): row for row in existing_rows}
rendered_count = 0
skipped_count = 0

try:
    for output_index in range(args.start, args.end + 1):
        print(f"[cache-render] preparing frame {output_index}", flush=True)
        output_path = frames_dir / f"rgb_{output_index:06d}.png"
        existing = existing_by_index.get(output_index)
        if not args.no_resume and existing is not None:
            try:
                if (
                    str(existing.get("config_hash")) == config_hash
                    and str(existing.get("take_id")) == take_id
                    and str(existing.get("render_provenance_hash"))
                    == current_render_provenance
                ):
                    validate_png(
                        output_path,
                        width=width,
                        height=height,
                        expected_sha256=str(existing["png_sha256"]),
                    )
                    skipped_count += 1
                    print(f"[cache-render] frame={output_index} already verified")
                    continue
            except (KeyError, ValueError):
                pass

        cache_row = cache_rows[output_index]
        cache_path = cache_dir / str(cache_row["cache_file"])
        loaded = load_surface_cache(
            cache_path,
            expected_sha256=str(cache_row["cache_sha256"]),
        )
        print(f"[cache-render] loaded cache {cache_path}", flush=True)
        metadata = loaded["metadata"]
        if int(metadata["output_index"]) != output_index:
            raise RuntimeError(f"Cache {cache_path} has the wrong output index")
        if str(metadata["take_id"]) != take_id or str(metadata["config_hash"]) != config_hash:
            raise RuntimeError(f"Cache {cache_path} belongs to another take/config")
        if str(metadata.get("simulation_provenance_hash")) != str(
            job["simulation_provenance_hash"]
        ):
            raise RuntimeError(f"Cache {cache_path} has another simulation provenance")

        points = loaded["points"]
        indices = loaded["face_vertex_indices"]
        counts = loaded["face_vertex_counts"]
        if len(counts) == 0:
            counts = np.full(len(indices) // 3, 3, dtype=np.int32)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
        mesh.GetExtentAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(*(float(value) for value in loaded["bounds_min"])),
                    Gf.Vec3f(*(float(value) for value in loaded["bounds_max"])),
                ]
            )
        )
        normals = loaded["normals"]
        if len(normals):
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
            mesh.SetNormalsInterpolation(
                UsdGeom.Tokens.vertex
                if len(normals) == len(points)
                else UsdGeom.Tokens.uniform
            )
        else:
            mesh.GetNormalsAttr().Clear()

        start_time = time.perf_counter()
        render_updates(args.mesh_settle_updates)
        temporary_output = frames_dir / f".rgb_{output_index:06d}.part.png"
        capture_viewport(viewport, temporary_output)
        png = validate_png(temporary_output, width=width, height=height)
        os.replace(temporary_output, output_path)
        png = validate_png(output_path, width=width, height=height)
        elapsed = time.perf_counter() - start_time
        render_row = {
            "schema_version": 1,
            "status": "rendered",
            "output_index": output_index,
            "sim_step": int(cache_row["sim_step"]),
            "sim_time_seconds": float(cache_row["sim_time_seconds"]),
            "take_id": take_id,
            "config_hash": config_hash,
            "simulation_provenance_hash": job["simulation_provenance_hash"],
            "render_provenance_hash": current_render_provenance,
            "run_id": args.run_id,
            "cache_file": str(cache_row["cache_file"]),
            "cache_sha256": str(cache_row["cache_sha256"]),
            **png,
            "renderer": "PathTracing",
            "path_spp": path_spp,
            "camera": str(render["camera"]),
            "render_seconds": elapsed,
        }
        existing_by_index[output_index] = render_row
        atomic_write_jsonl(
            segment_manifest_path,
            [existing_by_index[index] for index in sorted(existing_by_index)],
        )
        rendered_count += 1
        print(
            f"[cache-render] frame={output_index}, vertices={len(points)}, "
            f"faces={len(counts)}, seconds={elapsed:.3f}, path={output_path}"
        )

    final_rows = {
        int(row["output_index"]): row
        for row in read_jsonl(segment_manifest_path)
        if args.start <= int(row["output_index"]) <= args.end
    }
    for output_index in range(args.start, args.end + 1):
        row = final_rows.get(output_index)
        if row is None:
            raise RuntimeError(f"Segment manifest lacks rendered frame {output_index}")
        validate_png(
            frames_dir / f"rgb_{output_index:06d}.png",
            width=width,
            height=height,
            expected_sha256=str(row["png_sha256"]),
        )
    atomic_write_json(
        segment_complete_path,
        {
            "schema_version": 1,
            "segment": segment_name,
            "start": args.start,
            "end": args.end,
            "frame_count": args.end - args.start + 1,
            "rendered_this_run": rendered_count,
            "skipped_verified": skipped_count,
            "take_id": take_id,
            "config_hash": config_hash,
            "simulation_provenance_hash": job["simulation_provenance_hash"],
            "render_provenance_hash": current_render_provenance,
            "run_id": args.run_id,
            "manifest_sha256": file_sha256(segment_manifest_path),
            "valid": True,
        },
    )
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
