"""Render PhysX SDF debug geometry beside the source mesh for visual QA."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--model-height", type=float, required=True)
parser.add_argument("--model-yaw", type=float, default=0.0)
parser.add_argument("--sdf-resolution", type=int, default=512)
parser.add_argument("--sdf-remeshing", action="store_true")
parser.add_argument("--size", type=int, default=768)
parser.add_argument("--output", required=True)
args = parser.parse_args()

model_path = Path(args.model).resolve()
output_dir = Path(args.output).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": args.size,
        "height": args.size,
    }
)

import carb
import numpy as np
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.physx import (
    get_physx_interface,
    get_physx_scene_query_interface,
    get_physx_visualization_interface,
)
from omni.physx.bindings._physx import SETTING_VISUALIZATION_COLLISION_MESH
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

extension_manager = omni.kit.app.get_app().get_extension_manager()
extension_manager.set_extension_enabled_immediate("omni.physx.ui", True)
from omni.physxui import get_physxui_interface


def load_binary_stl(path: Path, height: float, yaw_degrees: float):
    with path.open("rb") as stream:
        stream.read(80)
        triangle_count = int.from_bytes(stream.read(4), "little")
        records = np.fromfile(
            stream,
            dtype=np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            ),
            count=triangle_count,
        )
    if len(records) != triangle_count:
        raise ValueError(f"Incomplete binary STL: {path}")
    vertices, inverse = np.unique(records["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    faces = faces[
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    ]
    points = vertices.astype(np.float64)
    yaw = math.radians(yaw_degrees)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    x, z = points[:, 0].copy(), points[:, 2].copy()
    points[:, 0] = cosine * x + sine * z
    points[:, 2] = -sine * x + cosine * z
    extents = points.max(axis=0) - points.min(axis=0)
    points *= height / float(np.max(extents))
    points[:, 0] -= 0.5 * (points[:, 0].min() + points[:, 0].max())
    points[:, 2] -= 0.5 * (points[:, 2].min() + points[:, 2].max())
    points[:, 1] -= points[:, 1].min()
    return points.astype(np.float32), faces


def set_look_at(prim, eye, target):
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 1.0, 0.0)
    ).GetInverse()
    xformable = UsdGeom.Xformable(prim)
    transform_ops = [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform
    ]
    (transform_ops[0] if transform_ops else xformable.AddTransformOp()).Set(matrix)


def make_material(stage, path, color, opacity=1.0):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.48)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if opacity < 1.0:
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def capture(path: Path):
    if path.exists():
        path.unlink()
    request = capture_viewport_to_file(viewport, file_path=str(path))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(900):
        app.update()
        if task.done():
            break
    if not task.done() or not task.result():
        raise RuntimeError(f"Viewport capture failed: {path}")


simulation_started = False
try:
    print("[phase] creating isolated SDF stage", flush=True)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, -1.0, 0.0))
    # Keep the isolated asset fixed while PhysX is stepped to emit debug geometry.
    # This preview has no collision floor and is not a dynamics experiment.
    physics_scene.CreateGravityMagnitudeAttr(0.0)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr().Set(60)
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

    points, triangles = load_binary_stl(model_path, args.model_height, args.model_yaw)
    comparison_separation = args.model_height * 1.25
    root = UsdGeom.Xform.Define(stage, "/World/Tree")
    UsdGeom.Xformable(root).AddTranslateOp().Set(
        Gf.Vec3d(comparison_separation * 0.5, 0.0, 0.0)
    )
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateDensityAttr().Set(2700.0)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Tree/Visual")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
    sdf.CreateSdfResolutionAttr().Set(args.sdf_resolution)
    sdf.CreateSdfEnableRemeshingAttr().Set(args.sdf_remeshing)
    PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr().Set(0.05)
    PhysxSchema.PhysxCollisionAPI(mesh.GetPrim()).CreateRestOffsetAttr().Set(0.01)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        make_material(stage, "/World/Looks/ColliderTransparent", (0.72, 0.10, 0.72), 0.0)
    )

    reference_root = UsdGeom.Xform.Define(stage, "/World/Reference")
    UsdGeom.Xformable(reference_root).AddTranslateOp().Set(
        Gf.Vec3d(-comparison_separation * 0.5, 0.0, 0.0)
    )
    reference = UsdGeom.Mesh.Define(stage, "/World/Reference/Visual")
    reference.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    reference.CreateFaceVertexCountsAttr([3] * len(triangles))
    reference.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
    reference.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdShade.MaterialBindingAPI.Apply(reference.GetPrim()).Bind(
        make_material(stage, "/World/Looks/Source", (0.72, 0.74, 0.78), 1.0)
    )

    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.CreateSizeAttr(2.0)
    floor_xform = UsdGeom.Xformable(floor)
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.03, 0.0))
    floor_xform.AddScaleOp().Set(Gf.Vec3f(1.2, 0.03, 1.2))
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
        make_material(stage, "/World/Looks/Floor", (0.09, 0.10, 0.12))
    )

    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(650.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr(1300.0)
    key.CreateAngleAttr(4.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, -35.0, 0.0))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(58.0)
    camera.CreateHorizontalApertureAttr(36.0)
    viewport = get_active_viewport()
    viewport.camera_path = camera.GetPath()
    viewport.set_texture_resolution((args.size, args.size))
    carb.settings.get_settings().set("/rtx/post/tonemap/exposure", 0.5)

    print("[phase] attaching PhysX and cooking SDF", flush=True)
    physx = get_physx_interface()
    physx.start_simulation()
    simulation_started = True
    physx.update_simulation(1.0 / 60.0, 1.0 / 60.0)
    physx.update_transformations(False, True, True)

    print("[phase] sampling cooked SDF surface with PhysX raycasts", flush=True)
    local_min = points.min(axis=0).astype(np.float64)
    local_max = points.max(axis=0).astype(np.float64)
    world_min = local_min + np.array([comparison_separation * 0.5, 0.0, 0.0])
    world_max = local_max + np.array([comparison_separation * 0.5, 0.0, 0.0])
    margin = args.model_height * 0.08
    ray_grid = 112
    query = get_physx_scene_query_interface()
    surface_hits = []
    for axis in range(3):
        transverse = [index for index in range(3) if index != axis]
        transverse_values = [
            np.linspace(world_min[index], world_max[index], ray_grid)
            for index in transverse
        ]
        ray_distance = float(world_max[axis] - world_min[axis] + 2.0 * margin)
        for sign in (1.0, -1.0):
            direction = [0.0, 0.0, 0.0]
            direction[axis] = sign
            start_axis = world_min[axis] - margin if sign > 0.0 else world_max[axis] + margin
            for first in transverse_values[0]:
                for second in transverse_values[1]:
                    origin = [0.0, 0.0, 0.0]
                    origin[axis] = float(start_axis)
                    origin[transverse[0]] = float(first)
                    origin[transverse[1]] = float(second)
                    hit = query.raycast_closest(origin, direction, ray_distance, True)
                    if hit and hit.get("hit") and hit.get("collision") == "/World/Tree/Visual":
                        surface_hits.append(tuple(float(value) for value in hit["position"]))
    if not surface_hits:
        raise RuntimeError("PhysX raycasts returned no SDF surface points")
    surface_hits = np.unique(np.round(np.asarray(surface_hits), decimals=6), axis=0)
    hit_points = UsdGeom.Points.Define(stage, "/World/SDFRaycastSurface")
    hit_points.CreatePointsAttr([Gf.Vec3f(*point) for point in surface_hits])
    hit_points.CreateWidthsAttr([args.model_height * 0.006])
    hit_points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    hit_points.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.02, 0.55)])
    print(f"[raycast] unique surface hits: {len(surface_hits)}", flush=True)

    physx_ui = get_physxui_interface()
    visualizer = get_physx_visualization_interface()
    print("[phase] enabling PhysX SDF debug visualization", flush=True)
    physx_ui.enable_debug_visualization(True)
    physx_ui.set_visualization_distance(10.0)
    visualizer.enable_visualization(True)
    visualizer.set_visualization_scale(1.0)
    for name in ("CollisionShapes", "CollisionEdges"):
        visualizer.set_visualization_parameter(name, True)
    visualizer.set_visualization_parameter("SDFs", False)
    carb.settings.get_settings().set(SETTING_VISUALIZATION_COLLISION_MESH, True)
    for step in range(12):
        current_time = (step + 2) / 60.0
        physx.update_simulation(1.0 / 60.0, current_time)
        physx.update_transformations(False, True, True)
        app.update()
    for _ in range(48):
        app.update()

    target = (0.0, args.model_height * 0.44, 0.0)
    radius = args.model_height * 3.05
    elevation = args.model_height * 1.15
    captures = []
    physx_ui.set_collision_mesh_type("both")
    for mode in ("side_by_side",):
        for view, azimuth in enumerate((30.0, 120.0, 210.0, 300.0)):
            angle = math.radians(azimuth)
            eye = (radius * math.cos(angle), elevation, radius * math.sin(angle))
            set_look_at(camera.GetPrim(), eye, target)
            for _ in range(8):
                app.update()
            path = output_dir / f"sdf_{mode}_view{view}.png"
            capture(path)
            captures.append(path.name)
            print(f"[capture] {path}", flush=True)

    physx_ui.explode_view_distance(0.0)
    for name in ("CollisionShapes", "CollisionEdges", "SDFs"):
        visualizer.set_visualization_parameter(name, False)
    visualizer.enable_visualization(False)
    physx_ui.enable_debug_visualization(False)
    for _ in range(8):
        app.update()
    for view, azimuth in enumerate((30.0, 120.0, 210.0, 300.0)):
        angle = math.radians(azimuth)
        eye = (radius * math.cos(angle), elevation, radius * math.sin(angle))
        set_look_at(camera.GetPrim(), eye, target)
        for _ in range(8):
            app.update()
        raycast_path = output_dir / f"sdf_raycast_surface_view{view}.png"
        capture(raycast_path)
        captures.append(raycast_path.name)
        print(f"[capture] {raycast_path}", flush=True)

    stage.GetRootLayer().Export(str(output_dir / "sdf_preview_stage.usda"))
    report = {
        "model": str(model_path),
        "model_height": args.model_height,
        "model_yaw": args.model_yaw,
        "sdf_resolution": args.sdf_resolution,
        "sdf_remeshing": args.sdf_remeshing,
        "source_vertices": len(points),
        "source_triangles": len(triangles),
        "raycast_surface_hits": len(surface_hits),
        "captures": captures,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
except Exception:
    traceback.print_exc()
    raise
finally:
    if simulation_started:
        get_physx_interface().reset_simulation()
    app.close()
