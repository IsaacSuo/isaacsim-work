"""Create, simulate, validate, and render a soft-body bounce hero scene in Isaac Sim 6.0."""

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
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--substeps", type=int, default=4)
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
        default=r"Y:\isaacsim_work\output\soft_body_elephant_hero",
    )
    parser.add_argument("--ball-radius", type=float, default=0.65)
    parser.add_argument(
        "--model",
        default=r"Y:\isaacsim_work\assets\soft_body_elephant.stl",
        help="Closed binary STL used as the deformable visual and cooking mesh.",
    )
    parser.add_argument("--model-height", type=float, default=1.45)
    parser.add_argument("--model-yaw", type=float, default=-56.0)
    parser.add_argument("--drop-height", type=float, default=3.0)
    parser.add_argument("--youngs-modulus", type=float, default=110000.0)
    parser.add_argument("--linear-damping", type=float, default=1.35)
    parser.add_argument("--poissons-ratio", type=float, default=0.45)
    parser.add_argument("--density", type=float, default=1050.0)
    parser.add_argument("--deformable-resolution", type=int, default=24)
    parser.add_argument("--self-collision-filter-distance", type=float, default=0.05)
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


USD_PATH = OUTPUT_DIR / "soft_body_bounce_hero.usda"
REPORT_PATH = OUTPUT_DIR / "run_complete.json"
PEDESTAL_TOP = 0.58
BALL_PATH = Sdf.Path("/World/SoftBall")
VISUAL_PATH = Sdf.Path("/World/SoftBall/Visual")
SIM_PATH = Sdf.Path("/World/SoftBall/SimulationMesh")
COLLISION_PATH = Sdf.Path("/World/SoftBall/CollisionMesh")
PNG_NAMES = (
    "hero_initial.png",
    "hero_impact.png",
    "hero_compression.png",
    "hero_rebound.png",
    "hero_apex.png",
)


def stage_notice(index, text):
    print(f"[phase {index}/6] {text}")


def normalize(vector):
    length = math.sqrt(sum(component * component for component in vector))
    return tuple(component / length for component in vector)


def create_icosphere(radius, subdivisions=3):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = [
        (-1, phi, 0),
        (1, phi, 0),
        (-1, -phi, 0),
        (1, -phi, 0),
        (0, -1, phi),
        (0, 1, phi),
        (0, -1, -phi),
        (0, 1, -phi),
        (phi, 0, -1),
        (phi, 0, 1),
        (-phi, 0, -1),
        (-phi, 0, 1),
    ]
    vertices = [normalize(vertex) for vertex in vertices]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    for _ in range(subdivisions):
        midpoint_cache = {}

        def midpoint(a, b):
            edge = tuple(sorted((a, b)))
            if edge in midpoint_cache:
                return midpoint_cache[edge]
            point = normalize(
                tuple((vertices[a][axis] + vertices[b][axis]) * 0.5 for axis in range(3))
            )
            vertices.append(point)
            index = len(vertices) - 1
            midpoint_cache[edge] = index
            return index

        refined = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined

    points = np.asarray(vertices, dtype=np.float32) * np.float32(radius)
    triangles = np.asarray(faces, dtype=np.int32)
    return points, triangles


def load_binary_stl(path, model_height):
    with open(path, "rb") as stream:
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
        raise ValueError(f"Incomplete binary STL: expected {triangle_count} triangles")
    raw_vertices = records["vertices"].reshape(-1, 3)
    vertices, inverse = np.unique(raw_vertices, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    nondegenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    faces = faces[nondegenerate]

    # The asset is authored Y-up. Rotate it about Y so its elephant profile and
    # trunk read clearly from the fixed three-quarter hero camera.
    oriented = vertices.astype(np.float64)
    yaw = math.radians(ARGS.model_yaw)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x = oriented[:, 0].copy()
    z = oriented[:, 2].copy()
    oriented[:, 0] = cos_yaw * x + sin_yaw * z
    oriented[:, 2] = -sin_yaw * x + cos_yaw * z
    extents = oriented.max(axis=0) - oriented.min(axis=0)
    if not np.isfinite(oriented).all() or extents[1] <= 0.0:
        raise ValueError(f"Invalid STL geometry: {path}")
    oriented *= float(model_height) / float(extents[1])
    oriented[:, 0] -= 0.5 * (oriented[:, 0].min() + oriented[:, 0].max())
    oriented[:, 2] -= 0.5 * (oriented[:, 2].min() + oriented[:, 2].max())
    oriented[:, 1] -= oriented[:, 1].min()
    return oriented.astype(np.float32), faces


def create_torus_mesh(stage, path, major_radius, tube_radius, major_segments=96, tube_segments=12):
    points = []
    normals = []
    triangles = []
    for major in range(major_segments):
        u = 2.0 * math.pi * major / major_segments
        cu, su = math.cos(u), math.sin(u)
        for tube in range(tube_segments):
            v = 2.0 * math.pi * tube / tube_segments
            cv, sv = math.cos(v), math.sin(v)
            points.append(
                (
                    (major_radius + tube_radius * cv) * cu,
                    (major_radius + tube_radius * cv) * su,
                    tube_radius * sv,
                )
            )
            normals.append((cv * cu, cv * su, sv))
    for major in range(major_segments):
        next_major = (major + 1) % major_segments
        for tube in range(tube_segments):
            next_tube = (tube + 1) % tube_segments
            a = major * tube_segments + tube
            b = next_major * tube_segments + tube
            c = next_major * tube_segments + next_tube
            d = major * tube_segments + next_tube
            triangles.extend(((a, b, c), (a, c, d)))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([index for triangle in triangles for index in triangle])
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    return mesh


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


def transform_points(points, matrix):
    return np.asarray(
        [tuple(matrix.Transform(Gf.Vec3d(*map(float, point)))) for point in points],
        dtype=np.float64,
    )


def current_snapshot(stage, visual_mesh, root_prim, frame):
    local_points = np.asarray(visual_mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_points = transform_points(local_points, matrix)
    minimum = world_points.min(axis=0)
    maximum = world_points.max(axis=0)
    center = world_points.mean(axis=0)
    extents = maximum - minimum
    radial = np.sqrt((world_points[:, 0] - center[0]) ** 2 + (world_points[:, 2] - center[2]) ** 2)
    translate_attr = root_prim.GetAttribute("xformOp:translate")
    translate = translate_attr.Get() if translate_attr else None
    return {
        "frame": frame,
        "points": local_points.copy(),
        "translate": tuple(translate) if translate is not None else None,
        "center": center,
        "minimum": minimum,
        "maximum": maximum,
        "height": float(extents[1]),
        "horizontal_radius": float(radial.max()),
        "finite": bool(np.isfinite(world_points).all()),
    }


def snapshot_summary(snapshot):
    return {
        "frame": int(snapshot["frame"]),
        "center": [round(float(value), 6) for value in snapshot["center"]],
        "minimum_y": round(float(snapshot["minimum"][1]), 6),
        "maximum_y": round(float(snapshot["maximum"][1]), 6),
        "vertical_height": round(float(snapshot["height"]), 6),
        "horizontal_radius": round(float(snapshot["horizontal_radius"]), 6),
        "finite": bool(snapshot["finite"]),
    }


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


def choose_keyframes(snapshots, initial_height):
    centers_y = np.asarray([snapshot["center"][1] for snapshot in snapshots])
    bottoms = np.asarray([snapshot["minimum"][1] for snapshot in snapshots])
    heights = np.asarray([snapshot["height"] for snapshot in snapshots])

    contact_candidates = np.flatnonzero(bottoms <= PEDESTAL_TOP + 0.035)
    if len(contact_candidates) == 0:
        contact_index = int(np.argmin(bottoms))
        contact_detected = False
    else:
        contact_index = int(contact_candidates[0])
        contact_detected = True

    first_window_end = min(len(snapshots), contact_index + 42)
    compression_index = contact_index + int(np.argmin(heights[contact_index:first_window_end]))
    low_center_index = contact_index + int(np.argmin(centers_y[contact_index:first_window_end]))

    rebound_index = None
    for index in range(max(compression_index + 1, low_center_index + 1), len(snapshots)):
        if centers_y[index] >= centers_y[low_center_index] + 0.30:
            rebound_index = index
            break
    if rebound_index is None:
        rebound_index = int(np.argmax(centers_y[low_center_index:])) + low_center_index

    actual_apex_index = int(np.argmax(centers_y[rebound_index:])) + rebound_index
    apex_window_start = max(rebound_index, actual_apex_index - 10)
    apex_window_end = min(len(snapshots), actual_apex_index + 11)
    initial_radius = snapshots[0]["horizontal_radius"]
    apex_index = min(
        range(apex_window_start, apex_window_end),
        key=lambda index: (
            abs(heights[index] / initial_height - 1.0)
            + 0.65
            * abs(snapshots[index]["horizontal_radius"] / initial_radius - 1.0)
            + 0.025 * abs(index - actual_apex_index)
        ),
    )
    rising_distance = float(centers_y[actual_apex_index] - centers_y[low_center_index])
    rebound_detected = rising_distance >= 0.30

    return {
        "initial": 0,
        "impact": contact_index,
        "compression": compression_index,
        "rebound": rebound_index,
        "apex": apex_index,
        "actual_apex": actual_apex_index,
        "contact_detected": contact_detected,
        "rebound_detected": rebound_detected,
        "rising_distance": rising_distance,
        "low_center_index": low_center_index,
        "minimum_height": float(heights.min()),
        "maximum_horizontal_radius": float(max(snapshot["horizontal_radius"] for snapshot in snapshots)),
        "compression_ratio": float(1.0 - heights.min() / initial_height),
        "squash_ratio": float(initial_height / max(heights.min(), 1e-9)),
    }


def set_snapshot(stage, visual_mesh, root_prim, snapshot):
    visual_mesh.GetPointsAttr().Set(
        [Gf.Vec3f(*map(float, point)) for point in snapshot["points"]]
    )
    if snapshot["translate"] is not None:
        root_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*snapshot["translate"]))
    stage.SetEditTarget(stage.GetRootLayer())


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

    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    del looks
    ball_material = create_preview_material(
        stage, "/World/Looks/CoralSilicone", (0.96, 0.14, 0.055), 0.21, clearcoat=0.38
    )
    pedestal_material = create_preview_material(
        stage, "/World/Looks/DeepTeal", (0.028, 0.145, 0.15), 0.34, metallic=0.06
    )
    floor_material = create_preview_material(
        stage, "/World/Looks/WarmStudio", (0.50, 0.44, 0.37), 0.72
    )
    wall_material = create_preview_material(
        stage, "/World/Looks/WarmBackdrop", (0.60, 0.53, 0.45), 0.76
    )
    ring_material = create_preview_material(
        stage, "/World/Looks/BrushedBronze", (0.66, 0.35, 0.12), 0.2, metallic=0.92
    )

    add_cube(stage, "/World/StudioFloor", (20.0, 0.18, 20.0), (0.0, -0.09, 0.0), floor_material, True)
    add_cube(stage, "/World/Backdrop", (28.0, 10.0, 0.20), (0.0, 4.0, -3.15), wall_material, False)

    pedestal = UsdGeom.Cylinder.Define(stage, "/World/Pedestal")
    pedestal.CreateAxisAttr(UsdGeom.Tokens.y)
    pedestal.CreateRadiusAttr(2.05)
    pedestal.CreateHeightAttr(PEDESTAL_TOP)
    UsdGeom.Xformable(pedestal).AddTranslateOp().Set(Gf.Vec3d(0.0, PEDESTAL_TOP * 0.5, 0.0))
    bind_visual_material(pedestal.GetPrim(), pedestal_material)
    UsdPhysics.CollisionAPI.Apply(pedestal.GetPrim())
    pedestal_collision = PhysxSchema.PhysxCollisionAPI.Apply(pedestal.GetPrim())
    pedestal_collision.CreateContactOffsetAttr().Set(0.025)
    pedestal_collision.CreateRestOffsetAttr().Set(0.004)

    ring = create_torus_mesh(stage, "/World/HeroRing", 2.38, 0.055)
    UsdGeom.Xformable(ring).AddTranslateOp().Set(Gf.Vec3d(-1.35, 2.45, -2.55))
    bind_visual_material(ring.GetPrim(), ring_material)

    ball_root = UsdGeom.Xform.Define(stage, BALL_PATH)
    ball_root.AddTranslateOp().Set(Gf.Vec3d(0.0, ARGS.drop_height, 0.0))
    model_path = Path(ARGS.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Deformable model not found: {model_path}")
    ball_points, ball_triangles = load_binary_stl(model_path, ARGS.model_height)
    visual = UsdGeom.Mesh.Define(stage, VISUAL_PATH)
    visual.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in ball_points])
    visual.CreateFaceVertexCountsAttr([3] * len(ball_triangles))
    visual.CreateFaceVertexIndicesAttr(ball_triangles.reshape(-1).tolist())
    visual.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    bind_visual_material(visual.GetPrim(), ball_material)

    hierarchy_created = deformableUtils.create_auto_volume_deformable_hierarchy(
        stage,
        BALL_PATH,
        SIM_PATH,
        COLLISION_PATH,
        VISUAL_PATH,
        True,
        False,
        True,
    )
    if not hierarchy_created:
        raise RuntimeError("create_auto_volume_deformable_hierarchy returned False")
    root_prim = ball_root.GetPrim()
    root_prim.GetAttribute("physxDeformableBody:resolution").Set(ARGS.deformable_resolution)
    if not root_prim.ApplyAPI("PhysxBaseDeformableBodyAPI"):
        raise RuntimeError("Failed to apply PhysxBaseDeformableBodyAPI")
    root_prim.GetAttribute("physxDeformableBody:linearDamping").Set(ARGS.linear_damping)
    root_prim.GetAttribute("physxDeformableBody:settlingDamping").Set(18.0)
    root_prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(24)
    root_prim.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
    root_prim.GetAttribute("physxDeformableBody:selfCollision").Set(False)
    root_prim.GetAttribute("physxDeformableBody:selfCollisionFilterDistance").Set(
        ARGS.self_collision_filter_distance
    )

    collision_prim = stage.GetPrimAtPath(COLLISION_PATH)
    if collision_prim and collision_prim.IsValid():
        collision_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
        physx_collision = PhysxSchema.PhysxCollisionAPI(collision_prim)
        physx_collision.CreateContactOffsetAttr().Set(0.03)
        physx_collision.CreateRestOffsetAttr().Set(0.01)

    deformable_material_path = Sdf.Path("/World/Looks/SoftSiliconePhysics")
    UsdShade.Material.Define(stage, deformable_material_path)
    if not deformableUtils.add_deformable_material(
        stage,
        deformable_material_path,
        density=ARGS.density,
        static_friction=0.42,
        dynamic_friction=0.30,
        youngs_modulus=ARGS.youngs_modulus,
        poissons_ratio=ARGS.poissons_ratio,
    ):
        raise RuntimeError("add_deformable_material returned False")
    physicsUtils.add_physics_material_to_prim(stage, root_prim, deformable_material_path)

    camera = UsdGeom.Camera.Define(stage, "/World/HeroCamera")
    camera.CreateFocalLengthAttr(58.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    camera.CreateFocusDistanceAttr(12.5)
    camera.CreateFStopAttr(0.0)
    set_look_at(camera.GetPrim(), (7.3, 4.6, 10.8), (0.0, 2.05, 0.0))

    add_rect_light(
        stage,
        "/World/Lights/Key",
        (-3.8, 6.8, 5.0),
        (0.0, 1.7, 0.0),
        (1.0, 0.73, 0.56),
        9200.0,
        4.0,
        4.0,
    )
    add_rect_light(
        stage,
        "/World/Lights/Rim",
        (4.8, 4.7, -0.4),
        (0.0, 2.1, 0.0),
        (0.38, 0.62, 1.0),
        7600.0,
        2.0,
        3.0,
    )
    add_rect_light(
        stage,
        "/World/Lights/Fill",
        (-4.0, 2.4, 1.0),
        (0.0, 1.4, 0.0),
        (1.0, 0.92, 0.78),
        3400.0,
        3.0,
        3.0,
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    dome.CreateColorAttr(Gf.Vec3f(0.32, 0.39, 0.50))
    dome.CreateIntensityAttr(620.0)

    return stage, visual, root_prim, camera, len(ball_points), len(ball_triangles)


def main():
    report = {
        "valid": False,
        "isaac_sim_version": "6.0.1-rc.7+release.42383.32955d8d.gl",
        "renderer": ARGS.renderer,
        "resolution": [ARGS.width, ARGS.height],
        "path_spp": ARGS.path_spp if ARGS.renderer == "PathTracing" else None,
        "physics_frames": ARGS.frames,
        "physics_substeps": ARGS.substeps,
        "render_video_frames": ARGS.render_video_frames,
        "output_directory": str(OUTPUT_DIR),
        "usd_path": str(USD_PATH),
    }
    simulation = None
    attached = False
    try:
        if ARGS.render_video_frames and ARGS.renderer != "PathTracing":
            raise ValueError("--render-video-frames requires --renderer PathTracing")
        stage_notice(2, "Building studio, closed elephant mesh, and volume deformable hierarchy")
        stage, visual, root_prim, camera, visual_vertex_count, visual_triangle_count = build_scene()
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

        initial = current_snapshot(stage, visual, root_prim, -1)
        snapshots = [initial]
        initial_height = initial["height"]
        initial_center_height = float(initial["center"][1])

        stage_notice(3, "Attaching GPU PhysX and running deformable simulation")
        simulation = get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        simulation.attach_stage(stage_id)
        attached = True

        simulation_points_count = None
        for frame in range(ARGS.frames):
            simulation.simulate(1.0 / 60.0, frame / 60.0)
            simulation.fetch_results()
            simulation_app.update()
            snapshot = current_snapshot(stage, visual, root_prim, frame)
            snapshots.append(snapshot)
            if not snapshot["finite"]:
                raise RuntimeError(f"Non-finite deformable coordinates at physics frame {frame}")
            if simulation_points_count is None:
                sim_points = UsdGeom.TetMesh.Get(stage, SIM_PATH).GetPointsAttr().Get()
                if sim_points is not None:
                    simulation_points_count = len(sim_points)
            if frame % 15 == 0 or frame == ARGS.frames - 1:
                print(
                    f"[physics] frame={frame:03d}/{ARGS.frames - 1} "
                    f"center_y={snapshot['center'][1]:.4f} bottom={snapshot['minimum'][1]:.4f} "
                    f"height={snapshot['height']:.4f} radius_xz={snapshot['horizontal_radius']:.4f}"
                )

        simulation.detach_stage()
        attached = False

        keyframes = choose_keyframes(snapshots, initial_height)
        stage_notice(4, "Selecting keyframes from measured contact, squash, and rebound")
        index_map = {
            "initial": keyframes["initial"],
            "impact": keyframes["impact"],
            "compression": keyframes["compression"],
            "rebound": keyframes["rebound"],
            "apex": keyframes["apex"],
        }
        for name, index in index_map.items():
            print(f"[keyframe] {name} index={index} {snapshot_summary(snapshots[index])}")

        stage_notice(5, f"Rendering five representative frames with {ARGS.renderer}")
        for (name, index), png_name in zip(index_map.items(), PNG_NAMES):
            set_snapshot(stage, visual, root_prim, snapshots[index])
            for _ in range(ARGS.render_settle):
                simulation_app.update()
            capture_current_viewport(viewport, OUTPUT_DIR / png_name)
            print(f"[render] state={name} source_frame={snapshots[index]['frame']} file={png_name}")

        video_frames_dir = None
        video_frame_files = []
        if ARGS.render_video_frames:
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
            for video_index, snapshot in enumerate(snapshots[1:]):
                set_snapshot(stage, visual, root_prim, snapshot)
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

        all_finite = all(snapshot["finite"] for snapshot in snapshots)
        min_center_height = min(float(snapshot["center"][1]) for snapshot in snapshots)
        min_bottom = min(float(snapshot["minimum"][1]) for snapshot in snapshots)
        png_files = [OUTPUT_DIR / name for name in PNG_NAMES]
        verified_pngs = [
            str(path) for path in png_files if path.is_file() and path.stat().st_size > 0
        ]
        usd_valid = USD_PATH.is_file() and USD_PATH.stat().st_size > 0
        video_frames_valid = (
            not ARGS.render_video_frames
            or (
                len(video_frame_files) == ARGS.frames
                and all(path.is_file() and path.stat().st_size > 0 for path in video_frame_files)
            )
        )
        compression_visible = keyframes["compression_ratio"] >= 0.10
        lateral_expansion_visible = (
            keyframes["maximum_horizontal_radius"] >= initial["horizontal_radius"] * 1.05
        )
        penetration_limit = PEDESTAL_TOP - 0.18
        no_obvious_penetration = min_bottom >= penetration_limit

        report.update(
            {
                "visual_vertex_count": visual_vertex_count,
                "visual_triangle_count": visual_triangle_count,
                "simulation_points_count": simulation_points_count,
                "all_points_finite": all_finite,
                "initial_center_height": initial_center_height,
                "minimum_center_height": min_center_height,
                "initial_vertical_height": initial_height,
                "minimum_vertical_height": keyframes["minimum_height"],
                "initial_horizontal_radius": initial["horizontal_radius"],
                "maximum_horizontal_radius": keyframes["maximum_horizontal_radius"],
                "compression_ratio": keyframes["compression_ratio"],
                "squash_ratio": keyframes["squash_ratio"],
                "contact_detected": keyframes["contact_detected"],
                "rebound_detected": keyframes["rebound_detected"],
                "post_contact_rise": keyframes["rising_distance"],
                "minimum_surface_y": min_bottom,
                "pedestal_top_y": PEDESTAL_TOP,
                "no_obvious_pedestal_penetration": no_obvious_penetration,
                "compression_visible": compression_visible,
                "lateral_expansion_visible": lateral_expansion_visible,
                "usd_saved_nonempty": usd_valid,
                "png_files": verified_pngs,
                "video_frames": {
                    "enabled": ARGS.render_video_frames,
                    "directory": str(video_frames_dir) if video_frames_dir else None,
                    "count": len(video_frame_files),
                    "expected_count": ARGS.frames if ARGS.render_video_frames else 0,
                    "renderer": ARGS.renderer if ARGS.render_video_frames else None,
                    "path_spp": ARGS.path_spp if ARGS.render_video_frames else None,
                    "valid": video_frames_valid,
                },
                "keyframes": {
                    name: snapshot_summary(snapshots[index]) for name, index in index_map.items()
                },
                "physics_material": {
                    "density": ARGS.density,
                    "youngs_modulus": ARGS.youngs_modulus,
                    "poissons_ratio": ARGS.poissons_ratio,
                    "dynamic_friction": 0.30,
                    "linear_damping": ARGS.linear_damping,
                    "deformable_resolution": ARGS.deformable_resolution,
                    "model": str(ARGS.model),
                    "model_height": ARGS.model_height,
                    "model_yaw": ARGS.model_yaw,
                },
            }
        )
        report["valid"] = bool(
            all_finite
            and keyframes["contact_detected"]
            and keyframes["rebound_detected"]
            and compression_visible
            and lateral_expansion_visible
            and no_obvious_penetration
            and usd_valid
            and len(verified_pngs) == len(PNG_NAMES)
            and video_frames_valid
        )
        if not report["valid"]:
            raise RuntimeError(
                "Validation thresholds were not met: "
                f"contact={keyframes['contact_detected']} rebound={keyframes['rebound_detected']} "
                f"compression={keyframes['compression_ratio']:.3f} "
                f"lateral={lateral_expansion_visible} penetration_ok={no_obvious_penetration}"
            )

        stage_notice(6, "Writing validation report and closing Isaac Sim")
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[complete] valid=True report={REPORT_PATH}")
        return 0
    except Exception as exc:
        report["valid"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[failed] {report['error']}")
        traceback.print_exc()
        return 1
    finally:
        if attached and simulation is not None:
            try:
                simulation.detach_stage()
            except Exception:
                traceback.print_exc()
        simulation_app.close()
        log_file.flush()


if __name__ == "__main__":
    raise SystemExit(main())
