"""Create, simulate, validate, and render a cloth flag hero scene in Isaac Sim 6.0.

A surface-deformable (cloth) flag hangs from a horizontal kinematic pole. The pole
swings sinusoidally about its long axis, driving continuous billowing and folding.
"""

import argparse
import asyncio
import json
import math
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--renderer",
        choices=("RaytracedLighting", "PathTracing"),
        default="RaytracedLighting",
    )
    parser.add_argument("--path-spp", type=int, default=32)
    parser.add_argument("--render-settle", type=int, default=16)
    parser.add_argument("--capture-timeout-updates", type=int, default=600)
    parser.add_argument(
        "--render-video-frames",
        action="store_true",
        help="Render every simulated physics frame as a PathTracing PNG sequence.",
    )
    parser.add_argument(
        "--video-frames-dir",
        default="video_frames",
        help="Absolute path or output-relative directory for video PNG frames.",
    )
    parser.add_argument(
        "--output",
        default=r"Y:\isaacsim_work\output\cloth_flag_hero",
    )
    # Cloth geometry
    parser.add_argument("--flag-dimx", type=int, default=80)
    parser.add_argument("--flag-dimy", type=int, default=48)
    parser.add_argument("--flag-width", type=float, default=3.0)
    parser.add_argument("--flag-height", type=float, default=2.0)
    parser.add_argument("--pole-height", type=float, default=2.6)
    # Cloth material
    parser.add_argument("--density", type=float, default=700.0)
    parser.add_argument("--stretch-stiffness", type=float, default=1500.0)
    parser.add_argument("--shear-stiffness", type=float, default=200.0)
    parser.add_argument("--bend-stiffness", type=float, default=30.0)
    parser.add_argument("--thickness", type=float, default=0.04)
    parser.add_argument("--linear-damping", type=float, default=0.6)
    # Driving swing
    parser.add_argument("--swing-z-amp", type=float, default=0.4)
    parser.add_argument("--swing-period-s", type=float, default=2.2)
    return parser.parse_args()


ARGS = parse_args()
OUTPUT_DIR = Path(ARGS.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "run.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def fileno(self):
        return self.streams[0].fileno()


log_file = LOG_PATH.open("w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print("[phase 1/6] Starting Isaac Sim")
print(f"[paths] log={LOG_PATH}")
print(f"[paths] output={OUTPUT_DIR}")

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": ARGS.renderer,
        "width": ARGS.width,
        "height": ARGS.height,
    }
)

import carb
import numpy as np
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils

USD_PATH = OUTPUT_DIR / "cloth_flag_hero.usda"
REPORT_PATH = OUTPUT_DIR / "run_complete.json"
PNG_NAMES = (
    "flag_initial.png",
    "flag_swing_a.png",
    "flag_swing_b.png",
    "flag_billow.png",
    "flag_fold.png",
)

FLAG_ROOT = Sdf.Path("/World/ClothFlag")
FLAG_MESH_PATH = Sdf.Path("/World/ClothFlag/FlagMesh")
FLAG_SIM_PATH = Sdf.Path("/World/ClothFlag/FlagSimMesh")
POLE_PATH = Sdf.Path("/World/FlagPole")
FLOOR_TOP = 0.0


def stage_notice(index, text):
    print(f"[phase {index}/6] {text}")


def create_preview_material(stage, path, color, roughness, metallic=0.0, clearcoat=0.0):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.16)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_visual_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def add_cube(stage, path, size, position, material, collision=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)
    xformable = UsdGeom.Xformable(cube)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*position))
    xformable.AddScaleOp().Set(Gf.Vec3f(size[0] * 0.5, size[1] * 0.5, size[2] * 0.5))
    bind_visual_material(cube.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def set_look_at(prim, eye, target, up=Gf.Vec3d(0.0, 1.0, 0.0)):
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()
    UsdGeom.Xformable(prim).AddTransformOp().Set(transform)


def add_rect_light(stage, path, eye, target, color, intensity, width, height):
    light = UsdLux.RectLight.Define(stage, path)
    light.CreateColorAttr(Gf.Vec3f(*color))
    light.CreateIntensityAttr(intensity)
    light.CreateWidthAttr(width)
    light.CreateHeightAttr(height)
    light.CreateNormalizeAttr(True)
    set_look_at(light.GetPrim(), eye, target)
    return light


def create_flag_mesh(stage, path, dimx, dimy, width, height):
    points, indices = deformableUtils.create_triangle_mesh_square(dimx, dimy, scale=1.0)
    scaled = [Gf.Vec3f(p[0] * width, p[1] * height, 0.0) for p in points]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(scaled)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    return mesh


def create_pole(stage, path, length, height, material):
    # Horizontal pole along X at the flag's top edge, kinematically driven.
    pole = UsdGeom.Cylinder.Define(stage, path)
    pole.CreateAxisAttr(UsdGeom.Tokens.x)
    pole.CreateRadiusAttr(0.035)
    pole.CreateHeightAttr(length)
    xformable = UsdGeom.Xformable(pole)
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, height, 0.0))
    xformable.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    bind_visual_material(pole.GetPrim(), material)
    UsdPhysics.CollisionAPI.Apply(pole.GetPrim())
    rigid = UsdPhysics.RigidBodyAPI.Apply(pole.GetPrim())
    rigid.CreateKinematicEnabledAttr().Set(True)
    return pole


def transform_points(points, matrix):
    return np.asarray(
        [tuple(matrix.Transform(Gf.Vec3d(*map(float, point)))) for point in points],
        dtype=np.float64,
    )


def flag_snapshot(stage, flag_mesh, flag_root, pole, frame):
    local_points = np.asarray(flag_mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = UsdGeom.Xformable(flag_root.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_points = transform_points(local_points, matrix)
    minimum = world_points.min(axis=0)
    maximum = world_points.max(axis=0)
    center = world_points.mean(axis=0)
    # Top edge is the highest local-Y row; bottom is the lowest.
    top_band = world_points[world_points[:, 1] >= world_points[:, 1].max() - 1e-3]
    bottom_band = world_points[world_points[:, 1] <= world_points[:, 1].min() + 1e-3]
    return {
        "frame": frame,
        "points": local_points.copy(),
        "translate": flag_root.GetPrim().GetAttribute("xformOp:translate").Get(),
        "orient": flag_root.GetPrim().GetAttribute("xformOp:orient").Get(),
        "pole_translate": pole.GetPrim().GetAttribute("xformOp:translate").Get(),
        "minimum": minimum,
        "maximum": maximum,
        "center": center,
        "top_mean_y": float(top_band[:, 1].mean()),
        "top_mean_z": float(top_band[:, 2].mean()),
        "bottom_mean_y": float(bottom_band[:, 1].mean()),
        "sag": float(top_band[:, 1].mean() - bottom_band[:, 1].mean()),
        "z_swing": float(center[2]),
        "finite": bool(np.isfinite(world_points).all()),
    }


def snapshot_summary(snapshot):
    return {
        "frame": int(snapshot["frame"]),
        "top_mean_y": round(snapshot["top_mean_y"], 6),
        "bottom_mean_y": round(snapshot["bottom_mean_y"], 6),
        "sag": round(snapshot["sag"], 6),
        "z_swing": round(snapshot["z_swing"], 6),
        "finite": bool(snapshot["finite"]),
    }


def set_flag_snapshot(stage, flag_mesh, flag_root, pole, snapshot):
    flag_mesh.GetPointsAttr().Set(
        [Gf.Vec3f(*map(float, point)) for point in snapshot["points"]]
    )
    if snapshot["translate"] is not None:
        flag_root.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*snapshot["translate"]))
    if snapshot["orient"] is not None:
        flag_root.GetPrim().GetAttribute("xformOp:orient").Set(snapshot["orient"])
    if snapshot.get("pole_translate") is not None:
        pole.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*snapshot["pole_translate"]))
    stage.SetEditTarget(stage.GetRootLayer())


def capture_current_viewport(viewport, file_path):
    if file_path.exists():
        file_path.unlink()
    request = capture_viewport_to_file(viewport, file_path=str(file_path))
    task = asyncio.ensure_future(request.wait_for_result(completion_frames=2))
    for _ in range(ARGS.capture_timeout_updates):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        task.cancel()
        raise RuntimeError(f"Timed out waiting for viewport capture: {file_path}")
    if not task.result():
        raise RuntimeError(f"Viewport capture returned no AOVs: {file_path}")
    for _ in range(ARGS.capture_timeout_updates):
        if file_path.is_file() and file_path.stat().st_size > 0:
            print(f"[capture] {file_path.name} bytes={file_path.stat().st_size}")
            return
        simulation_app.update()
    raise RuntimeError(f"Captured file did not become visible: {file_path}")


def build_scene():
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
    physx_scene.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    physx_scene.CreateTimeStepsPerSecondAttr().Set(60 * ARGS.substeps)
    physx_scene.CreateGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1048576)

    cloth_material = create_preview_material(
        stage, "/World/Looks/CoralCloth", (0.90, 0.22, 0.08), 0.55, clearcoat=0.15
    )
    pole_material = create_preview_material(
        stage, "/World/Looks/BrushedBronze", (0.66, 0.35, 0.12), 0.2, metallic=0.92
    )
    floor_material = create_preview_material(
        stage, "/World/Looks/WarmStudio", (0.50, 0.44, 0.37), 0.72
    )
    wall_material = create_preview_material(
        stage, "/World/Looks/WarmBackdrop", (0.60, 0.53, 0.45), 0.76
    )

    add_cube(stage, "/World/StudioFloor", (20.0, 0.18, 20.0), (0.0, -0.09, 0.0), floor_material, True)
    add_cube(stage, "/World/Backdrop", (28.0, 10.0, 0.20), (0.0, 4.0, -3.6), wall_material, False)

    pole = create_pole(stage, POLE_PATH, ARGS.flag_width + 0.35, ARGS.pole_height, pole_material)

    flag_root = UsdGeom.Xform.Define(stage, FLAG_ROOT)
    flag_root.AddTranslateOp().Set(Gf.Vec3d(0.0, ARGS.pole_height - ARGS.flag_height * 0.5, 0.0))
    flag_root.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    flag_mesh = create_flag_mesh(
        stage, FLAG_MESH_PATH, ARGS.flag_dimx, ARGS.flag_dimy, ARGS.flag_width, ARGS.flag_height
    )
    bind_visual_material(flag_mesh.GetPrim(), cloth_material)

    if not deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        FLAG_ROOT,
        FLAG_SIM_PATH,
        FLAG_MESH_PATH,
        False,
        True,
    ):
        raise RuntimeError("create_auto_surface_deformable_hierarchy returned False")

    flag_root.GetPrim().ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    flag_root.GetPrim().GetAttribute("physxDeformableBody:selfCollision").Set(True)
    flag_root.GetPrim().GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
    flag_root.GetPrim().GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(16)
    flag_root.GetPrim().GetAttribute("physxDeformableBody:collisionIterationMultiplier").Set(4)
    flag_root.GetPrim().GetAttribute("physxDeformableBody:linearDamping").Set(ARGS.linear_damping)
    flag_root.GetPrim().GetAttribute("physxDeformableBody:settlingDamping").Set(4.0)

    sim_mesh_prim = stage.GetPrimAtPath(FLAG_SIM_PATH)
    sim_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    PhysxSchema.PhysxCollisionAPI(sim_mesh_prim).GetRestOffsetAttr().Set(0.01)
    PhysxSchema.PhysxCollisionAPI(sim_mesh_prim).GetContactOffsetAttr().Set(0.03)

    deformable_material_path = Sdf.Path("/World/Looks/ClothPhysics")
    UsdShade.Material.Define(stage, deformable_material_path)
    if not deformableUtils.add_surface_deformable_material(
        stage,
        deformable_material_path,
        density=ARGS.density,
        dynamic_friction=0.4,
        static_friction=0.5,
        surface_thickness=ARGS.thickness,
        surface_stretch_stiffness=ARGS.stretch_stiffness,
        surface_shear_stiffness=ARGS.shear_stiffness,
        surface_bend_stiffness=ARGS.bend_stiffness,
    ):
        raise RuntimeError("add_surface_deformable_material returned False")
    physicsUtils.add_physics_material_to_prim(stage, flag_root.GetPrim(), deformable_material_path)
    deformable_material_prim = stage.GetPrimAtPath(deformable_material_path)
    deformable_material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    deformable_material_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.05)
    deformable_material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.05)

    attachment_path = FLAG_ROOT.AppendElementString("attachment")
    if not deformableUtils.create_auto_deformable_attachment(
        stage,
        target_attachment_path=attachment_path,
        attachable0_path=FLAG_ROOT,
        attachable1_path=POLE_PATH,
    ):
        raise RuntimeError("create_auto_deformable_attachment returned False")

    camera = UsdGeom.Camera.Define(stage, "/World/HeroCamera")
    camera.CreateFocalLengthAttr(58.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    camera.CreateFocusDistanceAttr(12.5)
    camera.CreateFStopAttr(0.0)
    set_look_at(camera.GetPrim(), (4.6, 2.6, 5.6), (0.0, ARGS.pole_height - ARGS.flag_height * 0.5, 0.0))

    add_rect_light(stage, "/World/Lights/Key", (-3.8, 6.0, 4.6), (0.0, 1.7, 0.0), (1.0, 0.73, 0.56), 9200.0, 4.0, 4.0)
    add_rect_light(stage, "/World/Lights/Rim", (4.6, 4.2, -0.8), (0.0, 1.8, 0.0), (0.38, 0.62, 1.0), 7600.0, 2.0, 3.0)
    add_rect_light(stage, "/World/Lights/Fill", (-4.0, 2.2, 1.2), (0.0, 1.4, 0.0), (1.0, 0.92, 0.78), 3400.0, 3.0, 3.0)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    dome.CreateColorAttr(Gf.Vec3f(0.32, 0.39, 0.50))
    dome.CreateIntensityAttr(620.0)

    return stage, flag_mesh, flag_root, pole, camera, len(flag_mesh.GetPointsAttr().Get())


def main():
    report = {
        "valid": False,
        "isaac_sim_version": "6.0.1-rc.7+release.42383.32955d8d.gl",
        "renderer": ARGS.renderer,
        "resolution": [ARGS.width, ARGS.height],
        "path_spp": ARGS.path_spp if ARGS.renderer == "PathTracing" else None,
        "physics_frames": ARGS.frames,
        "physics_substeps": ARGS.substeps,
        "output_directory": str(OUTPUT_DIR),
        "usd_path": str(USD_PATH),
    }
    simulation = None
    attached = False
    try:
        stage_notice(2, "Building studio, cloth flag, pole, and surface deformable hierarchy")
        stage, flag_mesh, flag_root, pole, camera, visual_vertex_count = build_scene()
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", ARGS.renderer)
        settings.set("/persistent/app/viewport/displayOptions", 0)
        settings.set("/rtx/post/tonemap/exposure", 0.7)
        settings.set("/rtx/reflections/enabled", True)
        settings.set("/rtx/indirectDiffuse/enabled", True)
        settings.set("/rtx/ambientOcclusion/enabled", True)
        if ARGS.renderer == "PathTracing":
            settings.set("/rtx/pathtracing/spp", ARGS.path_spp)
            settings.set("/rtx/pathtracing/totalSpp", ARGS.path_spp)
            settings.set("/rtx/pathtracing/maxBounces", 8)

        viewport = get_active_viewport()
        viewport.camera_path = camera.GetPath()
        viewport.set_texture_resolution((ARGS.width, ARGS.height))
        for _ in range(4):
            simulation_app.update()

        if not stage.GetRootLayer().Export(str(USD_PATH)):
            raise RuntimeError(f"Failed to export USD: {USD_PATH}")
        print(f"[usd] exported={USD_PATH} bytes={USD_PATH.stat().st_size}")

        stage_notice(3, "Attaching GPU PhysX and driving the swinging pole")
        simulation = get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        simulation.attach_stage(stage_id)
        attached = True

        pole_translate = pole.GetPrim().GetAttribute("xformOp:translate")
        snapshots = []
        dt = 1.0 / 60.0
        for frame in range(ARGS.frames):
            t = frame * dt
            z_offset = ARGS.swing_z_amp * math.sin(2.0 * math.pi * t / ARGS.swing_period_s)
            pole_translate.Set(Gf.Vec3d(0.0, ARGS.pole_height, z_offset))
            simulation.simulate(dt, frame / 60.0)
            simulation.fetch_results()
            simulation_app.update()
            snapshot = flag_snapshot(stage, flag_mesh, flag_root, pole, frame)
            snapshots.append(snapshot)
            if not snapshot["finite"]:
                raise RuntimeError(f"Non-finite cloth coordinates at physics frame {frame}")
            if frame % 20 == 0 or frame == ARGS.frames - 1:
                print(
                    f"[cloth] frame={frame:03d}/{ARGS.frames - 1} "
                    f"top_y={snapshot['top_mean_y']:.4f} bottom_y={snapshot['bottom_mean_y']:.4f} "
                    f"sag={snapshot['sag']:.4f} z_swing={snapshot['z_swing']:.4f}"
                )

        simulation.detach_stage()
        attached = False

        stage_notice(4, "Selecting representative flag states")
        sag_values = [s["sag"] for s in snapshots]
        z_values = [s["z_swing"] for s in snapshots]
        initial = snapshots[0]
        swing_a_index = int(np.argmin(z_values))
        swing_b_index = int(np.argmax(z_values))
        billow_index = int(np.argmax(sag_values))
        # Fold: moment of most lateral velocity (largest |z swing| change) mid-cycle.
        z_delta = np.abs(np.diff(z_values))
        fold_index = int(np.argmax(z_delta)) + 1
        index_map = {
            "initial": 0,
            "swing_a": swing_a_index,
            "swing_b": swing_b_index,
            "billow": billow_index,
            "fold": fold_index,
        }
        for name, index in index_map.items():
            print(f"[keyframe] {name} index={index} {snapshot_summary(snapshots[index])}")

        stage_notice(5, f"Rendering five representative frames with {ARGS.renderer}")
        for (name, index), png_name in zip(index_map.items(), PNG_NAMES):
            set_flag_snapshot(stage, flag_mesh, flag_root, pole, snapshots[index])
            for _ in range(ARGS.render_settle):
                simulation_app.update()
            capture_current_viewport(viewport, OUTPUT_DIR / png_name)
            print(f"[render] state={name} source_frame={snapshots[index]['frame']} file={png_name}")

        video_frames_dir = None
        video_frame_files = []
        if ARGS.render_video_frames:
            if ARGS.renderer != "PathTracing":
                raise ValueError("--render-video-frames requires --renderer PathTracing")
            requested_dir = Path(ARGS.video_frames_dir)
            video_frames_dir = (
                requested_dir if requested_dir.is_absolute() else OUTPUT_DIR / requested_dir
            )
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            for stale_frame in video_frames_dir.glob("frame_*.png"):
                stale_frame.unlink()
            print(
                f"[video] rendering {ARGS.frames} physics frames with "
                f"PathTracing spp={ARGS.path_spp} directory={video_frames_dir}"
            )
            for video_index, snapshot in enumerate(snapshots):
                set_flag_snapshot(stage, flag_mesh, flag_root, pole, snapshot)
                for _ in range(ARGS.render_settle):
                    simulation_app.update()
                frame_path = video_frames_dir / f"frame_{video_index:06d}.png"
                capture_current_viewport(viewport, frame_path)
                video_frame_files.append(frame_path)
                if video_index % 10 == 0 or video_index == ARGS.frames - 1:
                    print(
                        f"[video] frame={video_index:06d}/{ARGS.frames - 1:06d} "
                        f"physics_frame={snapshot['frame']}"
                    )

        stage_notice(6, "Writing validation report and closing Isaac Sim")
        report.update(
            {
                "valid": True,
                "visual_vertex_count": visual_vertex_count,
                "all_points_finite": all(s["finite"] for s in snapshots),
                "top_edge_fixed": bool(
                    max(abs(s["top_mean_y"] - snapshots[0]["top_mean_y"]) for s in snapshots) < 0.15
                ),
                "sag_initial": snapshots[0]["sag"],
                "sag_max": max(sag_values),
                "z_swing_range": float(max(z_values) - min(z_values)),
                "pole_top_z_alignment_max": float(
                    max(abs(s["top_mean_z"] - s["pole_translate"][2]) for s in snapshots)
                ),
                "pole_height": ARGS.pole_height,
                "flag_bottom_above_floor": bool(min(s["bottom_mean_y"] for s in snapshots) > FLOOR_TOP - 0.05),
                "keyframes": {
                    name: {
                        **snapshot_summary(snapshots[index]),
                        "frame": int(snapshots[index]["frame"]),
                    }
                    for name, index in index_map.items()
                },
                "cloth_material": {
                    "density": ARGS.density,
                    "surface_thickness": ARGS.thickness,
                    "surface_stretch_stiffness": ARGS.stretch_stiffness,
                    "surface_shear_stiffness": ARGS.shear_stiffness,
                    "surface_bend_stiffness": ARGS.bend_stiffness,
                    "linear_damping": ARGS.linear_damping,
                },
                "swing": {
                    "z_amp": ARGS.swing_z_amp,
                    "period_s": ARGS.swing_period_s,
                },
                "png_files": [str(OUTPUT_DIR / png_name) for png_name in PNG_NAMES],
                "video_frames": {
                    "enabled": ARGS.render_video_frames,
                    "directory": str(video_frames_dir) if video_frames_dir else None,
                    "count": len(video_frame_files),
                    "expected_count": ARGS.frames if ARGS.render_video_frames else 0,
                    "renderer": ARGS.renderer if ARGS.render_video_frames else None,
                    "path_spp": ARGS.path_spp if ARGS.render_video_frames else None,
                    "valid": bool(
                        not ARGS.render_video_frames
                        or (
                            len(video_frame_files) == ARGS.frames
                            and all(path.is_file() and path.stat().st_size > 0 for path in video_frame_files)
                        )
                    ),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        traceback.print_exc()
        report["valid"] = False
        report["error"] = str(exc)
        raise
    finally:
        if simulation is not None and attached:
            try:
                simulation.detach_stage()
            except Exception:  # noqa: BLE001
                pass
        try:
            with open(REPORT_PATH, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass
        print(f"[complete] valid={report['valid']} report={REPORT_PATH}")
        simulation_app.close()


if __name__ == "__main__":
    main()
