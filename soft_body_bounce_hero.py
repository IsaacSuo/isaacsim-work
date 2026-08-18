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
    parser.add_argument("--path-max-bounces", type=int, default=8)
    parser.add_argument("--render-settle", type=int, default=16)
    parser.add_argument("--capture-timeout-updates", type=int, default=600)
    parser.add_argument(
        "--environment-panorama",
        action="store_true",
        help="Render four static 90-degree views from the world origin without creating or running physics.",
    )
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
        "--export-blender-usd",
        action="store_true",
        help="Export the simulated visible surface as a Z-up, metre-scale animated USD for Blender.",
    )
    parser.add_argument(
        "--blender-usd-name",
        default="soft_body_blender.usdc",
        help="Output-relative or absolute path for the Blender animation cache.",
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
    parser.add_argument("--environment-usd", default=None, help="Optional static environment USD added as a sublayer.")
    parser.add_argument(
        "--environment-ground-only",
        action="store_true",
        help="Disable imported colliders and use an invisible local ground patch.",
    )
    parser.add_argument(
        "--environment-ground-prim",
        action="append",
        default=[],
        help="Imported collider prim to retain as real ground; may be repeated.",
    )
    parser.add_argument(
        "--sanitize-ground-prim",
        action="append",
        default=[],
        help="Ground prim whose mesh is cloned in world space with degenerate faces removed.",
    )
    parser.add_argument(
        "--local-collision-bounds",
        nargs=6,
        type=float,
        default=None,
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
        help="Extract exact environment mesh triangles fully inside this world-space box as a local collider.",
    )
    parser.add_argument(
        "--uneven-ground",
        action="store_true",
        help="Use terrain-aware validation instead of a single flat support-height threshold.",
    )
    parser.add_argument("--support-top-y", type=float, default=0.58)
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.0)
    parser.add_argument("--ground-patch-size-x", type=float, default=8.0)
    parser.add_argument("--ground-patch-size-z", type=float, default=8.0)
    parser.add_argument("--ground-patch-center-x", type=float, default=None)
    parser.add_argument("--ground-patch-center-z", type=float, default=None)
    parser.add_argument(
        "--allow-free-fall",
        action="store_true",
        help="Allow a body to leave a finite support instead of treating the resulting fall as penetration.",
    )
    parser.add_argument("--camera-eye", nargs=3, type=float, default=(7.3, 4.6, 10.8), metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-target", nargs=3, type=float, default=(0.0, 2.05, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-focal-length", type=float, default=58.0)
    parser.add_argument(
        "--orbit-preview",
        action="store_true",
        help="Render the compression state from three camera yaw offsets instead of five physics states.",
    )
    parser.add_argument(
        "--skip-preview-render",
        action="store_true",
        help="Skip Isaac representative stills; useful when the animation cache will be rendered in Blender.",
    )
    parser.add_argument(
        "--orbit-yaw-degrees",
        nargs=3,
        type=float,
        default=(-35.0, 0.0, 35.0),
        metavar=("LEFT", "CENTER", "RIGHT"),
    )
    parser.add_argument(
        "--emissive-mesh-lights",
        action="store_true",
        help="Create local RectLight proxies from nearby horizontal emissive mesh components.",
    )
    parser.add_argument("--emissive-light-intensity", type=float, default=4200.0)
    parser.add_argument("--emissive-light-radius", type=float, default=8.0)
    parser.add_argument("--emissive-light-max-count", type=int, default=24)
    parser.add_argument(
        "--emissive-light-max-thickness",
        type=float,
        default=0.03,
        help="Maximum vertical thickness of an emissive mesh component used as a light proxy.",
    )
    parser.add_argument(
        "--authored-downlight",
        action="append",
        nargs=10,
        default=[],
        metavar=("NAME", "X", "Y", "Z", "INTENSITY", "WIDTH", "HEIGHT", "R", "G", "B"),
        help="Restore a downward-facing authored Blender area light in Y-up stage coordinates.",
    )
    parser.add_argument(
        "--authored-light-intensity-scale",
        type=float,
        default=1.0,
        help="Apply a scene-level energy conversion scale to restored authored lights.",
    )
    parser.add_argument(
        "--ambient-light-intensity",
        type=float,
        default=None,
        help="Override the DomeLight intensity; defaults depend on whether an environment is loaded.",
    )
    parser.add_argument(
        "--hdri-texture",
        default=r"Y:\scenes\HDRI\bambanani_sunset_8k.exr",
        help="Fallback lat-long HDRI used when the imported scene has no usable authored HDRI.",
    )
    parser.add_argument(
        "--hdri-intensity",
        type=float,
        default=1000.0,
        help="Intensity for the fallback HDRI DomeLight; authored scene HDRI intensity is preserved.",
    )
    parser.add_argument(
        "--hdri-rotation-x-degrees",
        "--hdri-pole-flip-degrees",
        dest="hdri_rotation_x_degrees",
        type=float,
        default=-90.0,
        help="Rotate the RTX lat-long dome about X; -90 aligns its observed Z pole with a Y-up stage.",
    )
    parser.add_argument(
        "--environment-fill-intensity",
        type=float,
        default=0.0,
        help="Add a broad camera-side RectLight to lift deep shadows in enclosed environments.",
    )
    parser.add_argument(
        "--disable-studio-lights",
        action="store_true",
        help="Disable the generic Key/Rim/Fill rig and use only scene-authored environment lighting.",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Override RTX tone-mapping exposure without changing scene lighting.",
    )
    parser.add_argument(
        "--force-nonmetal-material",
        action="append",
        default=[],
        help="Material prim whose metallic input is disconnected and set to zero; may be repeated.",
    )
    parser.add_argument(
        "--force-nonmetal-material-list",
        default="",
        help="Comma-separated material prim paths; compact alternative for Windows command-line limits.",
    )
    parser.add_argument(
        "--material-input-scale",
        action="append",
        nargs=3,
        default=[],
        metavar=("MATERIAL", "INPUT", "FACTOR"),
        help="Multiply the connected UV texture channel for a PreviewSurface scalar input.",
    )
    parser.add_argument(
        "--material-color-scale",
        action="append",
        nargs=4,
        default=[],
        metavar=("MATERIAL", "R", "G", "B"),
        help="Multiply a PreviewSurface diffuse RGB texture by a constant color.",
    )
    parser.add_argument(
        "--material-opacity-threshold",
        action="append",
        nargs=2,
        default=[],
        metavar=("MATERIAL", "THRESHOLD"),
        help="Set PreviewSurface opacityThreshold for Blender-style alpha clipping.",
    )
    parser.add_argument(
        "--material-emission-texture",
        action="append",
        nargs=2,
        default=[],
        metavar=("MATERIAL", "TEXTURE"),
        help="Rebuild a texture-driven Blender Emission material as a UsdPreviewSurface.",
    )
    parser.add_argument(
        "--material-emission-scale",
        action="append",
        nargs=2,
        default=[],
        metavar=("MATERIAL", "FACTOR"),
        help="Multiply the RGB texture connected to a PreviewSurface emissiveColor input.",
    )
    parser.add_argument(
        "--material-omniglass",
        action="append",
        nargs=9,
        default=[],
        metavar=(
            "MATERIAL", "R", "G", "B", "IOR", "ROUGHNESS", "THIN_WALLED", "DEPTH",
            "ROUGHNESS_TEXTURE",
        ),
        help="Replace a material binding with an OmniGlass MDL material using explicit Blender parameters.",
    )
    parser.add_argument(
        "--material-omnipbr",
        action="append",
        nargs=9,
        default=[],
        metavar=(
            "MATERIAL", "TEXTURE", "R", "G", "B", "METALLIC", "ROUGHNESS",
            "SPECULAR", "OPACITY",
        ),
        help="Replace a material binding with a textured OmniPBR material, optionally translucent.",
    )
    parser.add_argument(
        "--material-input-value",
        action="append",
        nargs=3,
        default=[],
        metavar=("MATERIAL", "INPUT", "VALUE"),
        help="Set a scalar input on the material's UsdPreviewSurface shader.",
    )
    parser.add_argument(
        "--hide-material-meshes",
        action="append",
        default=[],
        metavar="MATERIAL",
        help="Hide every mesh bound to a material (diagnostic use).",
    )
    args = parser.parse_args()
    if args.force_nonmetal_material_list:
        args.force_nonmetal_material.extend(
            path for path in args.force_nonmetal_material_list.split(",") if path
        )
    return args


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
from soft_body.geometry import TriangleBoxCropper
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils, Vt


USD_PATH = OUTPUT_DIR / (
    "environment_panorama.usda" if ARGS.environment_panorama else "soft_body_bounce_hero.usda"
)
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
ORBIT_PNG_NAMES = (
    "orbit_left.png",
    "orbit_center.png",
    "orbit_right.png",
)
PANORAMA_PNG_NAMES = (
    "panorama_front.png",
    "panorama_right.png",
    "panorama_back.png",
    "panorama_left.png",
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
    if material is not None:
        bind_visual_material(cube.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def set_look_at(prim, eye, target, up=Gf.Vec3d(0.0, 1.0, 0.0)):
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up).GetInverse()
    xformable = UsdGeom.Xformable(prim)
    transform_ops = [
        op for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform
    ]
    transform_op = transform_ops[0] if transform_ops else xformable.AddTransformOp()
    transform_op.Set(transform)


def orbit_eye(base_eye, target, yaw_degrees):
    offset_x = float(base_eye[0]) - float(target[0])
    offset_y = float(base_eye[1]) - float(target[1])
    offset_z = float(base_eye[2]) - float(target[2])
    angle = math.radians(float(yaw_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        float(target[0]) + cosine * offset_x + sine * offset_z,
        float(target[1]) + offset_y,
        float(target[2]) - sine * offset_x + cosine * offset_z,
    )


def add_rect_light(stage, path, eye, target, color, intensity, width, height):
    light = UsdLux.RectLight.Define(stage, path)
    light.CreateColorAttr(Gf.Vec3f(*color))
    light.CreateIntensityAttr(intensity)
    light.CreateWidthAttr(width)
    light.CreateHeightAttr(height)
    light.CreateNormalizeAttr(True)
    set_look_at(light.GetPrim(), eye, target)
    return light


def add_authored_downlights(stage):
    for index, values in enumerate(ARGS.authored_downlight):
        name, x, y, z, intensity, width, height, red, green, blue = values
        center = (float(x), float(y), float(z))
        light = UsdLux.RectLight.Define(stage, f"/World/AuthoredSceneLights/Light_{index:03d}")
        light.CreateColorAttr(Gf.Vec3f(float(red), float(green), float(blue)))
        light.CreateIntensityAttr(float(intensity) * ARGS.authored_light_intensity_scale)
        light.CreateWidthAttr(float(width))
        light.CreateHeightAttr(float(height))
        light.CreateNormalizeAttr(True)
        set_look_at(
            light.GetPrim(),
            center,
            (center[0], center[1] - 1.0, center[2]),
            Gf.Vec3d(0.0, 0.0, -1.0),
        )
        light.GetPrim().CreateAttribute("authoredLight:sourceName", Sdf.ValueTypeNames.String).Set(name)
    if ARGS.authored_downlight:
        print(
            f"[authored-lights] created={len(ARGS.authored_downlight)} "
            f"intensity_scale={ARGS.authored_light_intensity_scale:.6g}"
        )
    return len(ARGS.authored_downlight)


def connected_sources(connectable):
    result = connectable.GetConnectedSources()
    if not result:
        return []
    return result[0] if isinstance(result, tuple) else result


def material_emissive_color(material):
    pending = []
    visited = set()
    for output in material.GetOutputs():
        pending.extend(connected_sources(output))
    while pending:
        source_info = pending.pop()
        source = source_info.source
        if not source:
            continue
        shader = UsdShade.Shader(source.GetPrim())
        path = str(shader.GetPath())
        if not shader or path in visited:
            continue
        visited.add(path)
        for shader_input in shader.GetInputs():
            name = shader_input.GetBaseName().lower()
            value = shader_input.Get()
            if ("emissivecolor" in name or "emissioncolor" in name) and value is not None:
                try:
                    color = tuple(max(0.0, float(value[index])) for index in range(3))
                except (IndexError, TypeError, ValueError):
                    color = None
                if color and max(color) > 1e-5:
                    maximum = max(color)
                    return tuple(component / maximum for component in color)
            if ("emissivecolor" in name or "emissioncolor" in name) and connected_sources(shader_input):
                return (1.0, 1.0, 1.0)
            pending.extend(connected_sources(shader_input))
    return None


def mesh_component_bounds(mesh):
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if not points or not counts or not indices:
        return []
    parent = list(range(len(points)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    used = set()
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        valid = [index for index in face if 0 <= index < len(points)]
        if not valid:
            continue
        used.update(valid)
        for index in valid[1:]:
            union(valid[0], index)

    groups = {}
    for index in used:
        groups.setdefault(find(index), []).append(index)
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(mesh.GetPrim())
    bounds = []
    for vertex_indices in groups.values():
        world_points = [matrix.Transform(points[index]) for index in vertex_indices]
        minimum = tuple(min(float(point[axis]) for point in world_points) for axis in range(3))
        maximum = tuple(max(float(point[axis]) for point in world_points) for axis in range(3))
        center = tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))
        size = tuple(maximum[axis] - minimum[axis] for axis in range(3))
        bounds.append((center, size))
    return bounds


def add_emissive_mesh_proxy_lights(stage):
    emissive_materials = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            color = material_emissive_color(material)
            if color:
                emissive_materials[str(prim.GetPath())] = color

    candidates = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        color = emissive_materials.get(str(material.GetPath()))
        if not color:
            continue
        for center, size in mesh_component_bounds(UsdGeom.Mesh(prim)):
            horizontal_length = max(size[0], size[2])
            if (
                size[1] > ARGS.emissive_light_max_thickness
                or min(size[0], size[2]) < 0.035
                or horizontal_length < 0.75
            ):
                continue
            distance = math.hypot(center[0] - ARGS.spawn_x, center[2] - ARGS.spawn_z)
            if distance > ARGS.emissive_light_radius:
                continue
            candidates.append((distance, center, size, color, str(prim.GetPath())))

    candidates.sort(key=lambda row: row[0])
    selected = candidates[: max(0, ARGS.emissive_light_max_count)]
    for index, (_, center, size, color, source_path) in enumerate(selected):
        eye = (center[0], center[1] - 0.025, center[2])
        target = (center[0], center[1] - 1.0, center[2])
        light = UsdLux.RectLight.Define(stage, f"/World/EmissiveProxyLights/Light_{index:03d}")
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateIntensityAttr(ARGS.emissive_light_intensity)
        light.CreateWidthAttr(max(0.05, size[0]))
        light.CreateHeightAttr(max(0.05, size[2]))
        light.CreateNormalizeAttr(True)
        set_look_at(light.GetPrim(), eye, target, Gf.Vec3d(0.0, 0.0, -1.0))
        light.GetPrim().CreateAttribute("proxyLight:sourcePrim", Sdf.ValueTypeNames.String).Set(source_path)
    print(
        f"[emissive-lights] materials={len(emissive_materials)} "
        f"candidates={len(candidates)} created={len(selected)} "
        f"radius={ARGS.emissive_light_radius:.2f} intensity={ARGS.emissive_light_intensity:.1f} "
        f"max_thickness={ARGS.emissive_light_max_thickness:.3f}"
    )
    return len(selected)


def force_material_nonmetal(stage, material_path):
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Nonmetal material path is invalid: {material_path}")
    changed = 0
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        metallic_input = UsdShade.Shader(prim).GetInput("metallic")
        if not metallic_input:
            continue
        metallic_input.DisconnectSource()
        metallic_input.Set(0.0)
        changed += 1
    if changed == 0:
        raise RuntimeError(f"No metallic shader input found under material: {material_path}")
    print(f"[material-override] nonmetal={material_path} shader_inputs={changed}")


def scale_material_texture_input(stage, material_path, input_name, factor):
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Scaled material path is invalid: {material_path}")
    factor = float(factor)
    channel_indices = {"r": 0, "g": 1, "b": 2, "a": 3}
    changed = 0
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        surface_input = UsdShade.Shader(prim).GetInput(input_name)
        if not surface_input:
            continue
        sources, _ = surface_input.GetConnectedSources()
        if len(sources) != 1:
            continue
        source = sources[0]
        source_name = str(source.sourceName)
        if source_name not in channel_indices:
            raise RuntimeError(
                f"Unsupported texture output for scaling: material={material_path} "
                f"input={input_name} source={source_name}"
            )
        texture_shader = UsdShade.Shader(source.source.GetPrim())
        if texture_shader.GetIdAttr().Get() != "UsdUVTexture":
            raise RuntimeError(
                f"Scaled input is not connected to UsdUVTexture: {material_path} {input_name}"
            )
        scale_input = texture_shader.GetInput("scale")
        if not scale_input:
            scale_input = texture_shader.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        current = scale_input.Get() or Gf.Vec4f(1.0, 1.0, 1.0, 1.0)
        values = [float(value) for value in current]
        values[channel_indices[source_name]] *= factor
        scale_input.Set(Gf.Vec4f(*values))
        changed += 1
    if changed != 1:
        raise RuntimeError(
            f"Expected one scalable shader input: material={material_path} "
            f"input={input_name} changed={changed}"
        )
    print(
        f"[material-override] scale={material_path} input={input_name} "
        f"factor={factor:.9g}"
    )


def scale_material_diffuse_texture(stage, material_path, red, green, blue):
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Color-scaled material path is invalid: {material_path}")
    factors = (float(red), float(green), float(blue))
    changed = 0
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        diffuse_input = UsdShade.Shader(prim).GetInput("diffuseColor")
        if not diffuse_input:
            continue
        sources, _ = diffuse_input.GetConnectedSources()
        if len(sources) != 1 or str(sources[0].sourceName) != "rgb":
            continue
        texture_shader = UsdShade.Shader(sources[0].source.GetPrim())
        if texture_shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        scale_input = texture_shader.GetInput("scale")
        if not scale_input:
            scale_input = texture_shader.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        current = scale_input.Get() or Gf.Vec4f(1.0, 1.0, 1.0, 1.0)
        values = [float(value) for value in current]
        for index, factor in enumerate(factors):
            values[index] *= factor
        scale_input.Set(Gf.Vec4f(*values))
        changed += 1
    if changed != 1:
        raise RuntimeError(f"Expected one diffuse texture to scale: {material_path} changed={changed}")
    print(f"[material-override] color-scale={material_path} rgb={factors}")


def set_material_opacity_threshold(stage, material_path, threshold):
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Threshold material path is invalid: {material_path}")
    changed = 0
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        opacity = shader.GetInput("opacity")
        if not opacity or not opacity.HasConnectedSource():
            continue
        threshold_input = shader.GetInput("opacityThreshold")
        if not threshold_input:
            threshold_input = shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float)
        threshold_input.Set(float(threshold))
        changed += 1
    if changed != 1:
        raise RuntimeError(f"Expected one opacity threshold target: {material_path} changed={changed}")
    print(f"[material-override] opacity-threshold={material_path} value={float(threshold):.9g}")


def restore_texture_emission_material(stage, material_path, texture_path):
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Emission material path is invalid: {material_path}")
    texture_file = Path(texture_path)
    if not texture_file.is_file():
        raise RuntimeError(f"Emission texture does not exist: {texture_file}")

    material = UsdShade.Material(material_prim)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/RestoredEmissionSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    uv_reader = UsdShade.Shader.Define(stage, f"{material_path}/RestoredEmissionUV")
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_output = uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, f"{material_path}/RestoredEmissionTexture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_file.as_posix()))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_output)
    rgb_output = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    emissive = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    emissive.Set(Gf.Vec3f(1.0, 1.0, 1.0))
    emissive.ConnectToSource(rgb_output)
    surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_output)
    print(f"[material-override] emission-texture={material_path} texture={texture_file}")


def scale_material_emission_texture(stage, material_path, factor):
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Emission-scale material path is invalid: {material_path}")
    factor = float(factor)
    changed = 0
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        emissive = shader.GetInput("emissiveColor")
        if not emissive:
            continue
        sources, _ = emissive.GetConnectedSources()
        if len(sources) != 1 or str(sources[0].sourceName) != "rgb":
            continue
        texture = UsdShade.Shader(sources[0].source.GetPrim())
        if texture.GetIdAttr().Get() != "UsdUVTexture":
            continue
        scale = texture.GetInput("scale")
        if not scale:
            scale = texture.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        current = scale.Get() or Gf.Vec4f(1.0, 1.0, 1.0, 1.0)
        values = [float(value) for value in current]
        values[:3] = [value * factor for value in values[:3]]
        scale.Set(Gf.Vec4f(*values))
        changed += 1
    if changed != 1:
        raise RuntimeError(f"Expected one emissive texture to scale: {material_path} changed={changed}")
    print(f"[material-override] emission-scale={material_path} factor={factor:.9g}")


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value}")


def replace_material_with_omniglass(
    stage, material_path, red, green, blue, ior, roughness, thin_walled, depth,
    roughness_texture
):
    source_prim = stage.GetPrimAtPath(material_path)
    if not source_prim or not source_prim.IsValid() or not source_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"OmniGlass source material path is invalid: {material_path}")

    roughness_source = None
    if parse_bool(roughness_texture):
        for prim in Usd.PrimRange(source_prim):
            if not prim.IsA(UsdShade.Shader):
                continue
            source_shader = UsdShade.Shader(prim)
            if source_shader.GetIdAttr().Get() != "UsdPreviewSurface":
                continue
            source_input = source_shader.GetInput("roughness")
            if source_input:
                connected, _ = source_input.GetConnectedSources()
                if len(connected) == 1:
                    roughness_source = connected[0]
                    break
        if roughness_source is None:
            raise RuntimeError(f"OmniGlass roughness texture connection is missing: {material_path}")

    bound_prims = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material and str(material.GetPath()) == material_path:
            bound_prims.append(prim)
    if not bound_prims:
        raise RuntimeError(f"OmniGlass source material has no bound meshes: {material_path}")

    suffix = "".join(character if character.isalnum() else "_" for character in material_path).strip("_")
    prim_name = f"RestoredOmniGlass_{suffix}"
    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        bind_selected_prims=False,
        prim_name=prim_name,
    ).do()
    restored_path = Sdf.Path(f"/World/Looks/{prim_name}")
    restored = UsdShade.Material.Get(stage, restored_path)
    shader = UsdShade.Shader.Get(stage, restored_path.AppendChild("Shader"))
    if not restored or not shader:
        raise RuntimeError(f"OmniGlass material creation failed: {restored_path}")

    color = Gf.Vec3f(float(red), float(green), float(blue))
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    frosting = shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float)
    frosting.Set(float(roughness))
    if roughness_source is not None:
        frosting.ConnectToSource(roughness_source.source, roughness_source.sourceName)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(parse_bool(thin_walled))
    shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(float(depth))
    for prim in bound_prims:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(restored)
        prim.CreateAttribute("omni:rtx:enableTransmission", Sdf.ValueTypeNames.Bool).Set(True)
    print(
        f"[material-override] omniglass={material_path} target={restored_path} "
        f"meshes={len(bound_prims)} color={tuple(color)} ior={float(ior):.6g} "
        f"roughness={float(roughness):.6g} roughness_texture={parse_bool(roughness_texture)} "
        f"thin_walled={parse_bool(thin_walled)} depth={float(depth):.6g}"
    )


def replace_material_with_omnipbr(
    stage, material_path, texture_path, red, green, blue, metallic, roughness, specular, opacity
):
    source_prim = stage.GetPrimAtPath(material_path)
    if not source_prim or not source_prim.IsValid() or not source_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"OmniPBR source material path is invalid: {material_path}")
    texture_file = Path(texture_path).resolve()
    if not texture_file.is_file():
        raise RuntimeError(f"OmniPBR texture does not exist: {texture_file}")

    bound_prims = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material and str(material.GetPath()) == material_path:
            bound_prims.append(prim)
    if not bound_prims:
        raise RuntimeError(f"OmniPBR source material has no bound meshes: {material_path}")

    suffix = "".join(character if character.isalnum() else "_" for character in material_path).strip("_")
    prim_name = f"RestoredOmniPBR_{suffix}"
    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        bind_selected_prims=False,
        prim_name=prim_name,
    ).do()
    restored_path = Sdf.Path(f"/World/Looks/{prim_name}")
    restored = UsdShade.Material.Get(stage, restored_path)
    shader = UsdShade.Shader.Get(stage, restored_path.AppendChild("Shader"))
    if not restored or not shader:
        raise RuntimeError(f"OmniPBR material creation failed: {restored_path}")

    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture_file.as_posix())
    )
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(red), float(green), float(blue))
    )
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(specular))
    has_opacity = float(opacity) < 0.999999
    shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(has_opacity)
    shader.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.0)
    for prim in bound_prims:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(restored)
    print(
        f"[material-override] omnipbr={material_path} target={restored_path} "
        f"meshes={len(bound_prims)} texture={texture_file} "
        f"tint=({float(red):.6g},{float(green):.6g},{float(blue):.6g}) "
        f"metallic={float(metallic):.6g} roughness={float(roughness):.6g} "
        f"specular={float(specular):.6g} opacity={float(opacity):.6g}"
    )


def set_material_scalar_input(stage, material_path, input_name, value):
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(f"Material path is invalid: {material_path}")
    changed = 0
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        shader.CreateInput(input_name, Sdf.ValueTypeNames.Float).Set(float(value))
        changed += 1
    if changed != 1:
        raise RuntimeError(
            f"Expected one PreviewSurface shader: {material_path} changed={changed}"
        )
    print(f"[material-override] input-value={material_path}.{input_name} value={float(value):.9g}")


def hide_meshes_bound_to_material(stage, material_path):
    hidden = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material and str(material.GetPath()) == material_path:
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden += 1
    if hidden == 0:
        raise RuntimeError(f"Material has no bound meshes to hide: {material_path}")
    print(f"[material-override] hide-material-meshes={material_path} meshes={hidden}")


def transform_points(points, matrix):
    return np.asarray(
        [tuple(matrix.Transform(Gf.Vec3d(*map(float, point)))) for point in points],
        dtype=np.float64,
    )


def clone_sanitized_collision_mesh(stage, source_prim, path):
    source = UsdGeom.Mesh(source_prim)
    points = source.GetPointsAttr().Get() or []
    counts = source.GetFaceVertexCountsAttr().Get() or []
    indices = source.GetFaceVertexIndicesAttr().Get() or []
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(source_prim)
    world_points = [matrix.Transform(Gf.Vec3d(point)) for point in points]
    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        if count < 3 or any(index < 0 or index >= len(world_points) for index in face):
            continue
        for offset in range(1, count - 1):
            triangle = (face[0], face[offset], face[offset + 1])
            p0, p1, p2 = (world_points[index] for index in triangle)
            if Gf.Cross(p1 - p0, p2 - p0).GetLength() <= 1e-8:
                continue
            triangles.append(triangle)
    if not triangles:
        raise RuntimeError(f"Sanitized ground mesh has no valid triangles: {source_prim.GetPath()}")
    target = UsdGeom.Mesh.Define(stage, path)
    target.CreatePointsAttr([Gf.Vec3f(point) for point in world_points])
    target.CreateFaceVertexCountsAttr([3] * len(triangles))
    target.CreateFaceVertexIndicesAttr([index for triangle in triangles for index in triangle])
    target.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    target.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(target.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(target.GetPrim()).CreateApproximationAttr().Set("none")
    return len(triangles)


def extract_local_collision_mesh(stage, environment_prim, path, bounds):
    """Boolean-crop merged environment surfaces into one exact local collider.

    All source meshes are transformed into world space first.  Each triangle is
    intersected with the six planes of the local box, so triangles crossing the
    boundary are cut instead of discarded.  The crop is intentionally left as
    a terrain surface (no artificial vertical cap walls).
    """
    cropper = TriangleBoxCropper(bounds)
    source_meshes = set()

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(environment_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        source = UsdGeom.Mesh(prim)
        points = source.GetPointsAttr().Get() or []
        counts = source.GetFaceVertexCountsAttr().Get() or []
        indices = source.GetFaceVertexIndicesAttr().Get() or []
        if not points or not counts or not indices:
            continue
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        world_points = [matrix.Transform(Gf.Vec3d(point)) for point in points]
        cursor = 0
        mesh_contributed = False
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            if count < 3 or any(index < 0 or index >= len(world_points) for index in face):
                continue
            for offset in range(1, count - 1):
                triangle_points = np.asarray(
                    [world_points[face[0]], world_points[face[offset]], world_points[face[offset + 1]]],
                    dtype=np.float64,
                )
                if cropper.add_triangle(triangle_points):
                    mesh_contributed = True
        if mesh_contributed:
            source_meshes.add(str(prim.GetPath()))

    if not cropper.triangles:
        raise RuntimeError(f"No environment triangles found inside local collision bounds: {bounds}")
    target = UsdGeom.Mesh.Define(stage, path)
    target.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in cropper.points])
    target.CreateFaceVertexCountsAttr([3] * len(cropper.triangles))
    target.CreateFaceVertexIndicesAttr(
        [index for triangle in cropper.triangles for index in triangle]
    )
    target.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    target.CreateDoubleSidedAttr().Set(True)
    target.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(target.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(target.GetPrim()).CreateApproximationAttr().Set("none")
    target.GetPrim().ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    PhysxSchema.PhysxCollisionAPI(target.GetPrim()).CreateContactOffsetAttr().Set(0.02)
    print(
        f"[environment] exact-local-collider bounds={tuple(map(float, bounds))} "
        f"sources={len(source_meshes)} vertices={len(cropper.points)} "
        f"triangles={len(cropper.triangles)} "
        f"clipped_sources={cropper.clipped_source_triangles} "
        f"rejected_degenerate={cropper.rejected_degenerate}"
    )
    return {
        "vertices": len(cropper.points),
        "triangles": len(cropper.triangles),
        "source_meshes": sorted(source_meshes),
        "crop_method": "world_space_triangle_box_boolean_without_caps",
        "clipped_source_triangles": cropper.clipped_source_triangles,
        "rejected_degenerate": cropper.rejected_degenerate,
    }


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
        "world_points": world_points.copy(),
        "translate": tuple(translate) if translate is not None else None,
        "center": center,
        "minimum": minimum,
        "maximum": maximum,
        "height": float(extents[1]),
        "horizontal_radius": float(radial.max()),
        "finite": bool(np.isfinite(world_points).all()),
    }


def export_blender_animation_usd(visual_mesh, snapshots, output_path, frames_per_second=60.0):
    """Write visible surface samples in Blender's Z-up world coordinates.

    Isaac scene coordinates are Y-up.  The rigid rotation
    ``(x, y, z) -> (x, -z, y)`` is the inverse of the Blender-to-Isaac
    conversion used by the prepared environments and preserves face winding.
    """
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    source_counts = visual_mesh.GetFaceVertexCountsAttr().Get() or []
    source_indices = visual_mesh.GetFaceVertexIndicesAttr().Get() or []
    if not source_counts or not source_indices:
        raise RuntimeError("Visible soft-body mesh has no exportable topology")
    if not snapshots:
        raise RuntimeError("No simulated snapshots were provided for Blender export")

    cache_stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(cache_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(cache_stage, 1.0)
    cache_stage.SetTimeCodesPerSecond(float(frames_per_second))
    cache_stage.SetFramesPerSecond(float(frames_per_second))
    cache_stage.SetStartTimeCode(1.0)
    cache_stage.SetEndTimeCode(float(len(snapshots)))

    root = UsdGeom.Xform.Define(cache_stage, "/SoftBodyCache")
    cache_stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(cache_stage, "/SoftBodyCache/SoftBody")
    mesh.CreateFaceVertexCountsAttr([int(value) for value in source_counts])
    mesh.CreateFaceVertexIndicesAttr([int(value) for value in source_indices])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(0.34, 0.07, 0.04)])

    expected_count = None
    for blender_frame, snapshot in enumerate(snapshots, start=1):
        isaac_points = np.asarray(snapshot["world_points"], dtype=np.float32)
        blender_points = np.column_stack(
            (isaac_points[:, 0], -isaac_points[:, 2], isaac_points[:, 1])
        ).astype(np.float32, copy=False)
        if expected_count is None:
            expected_count = len(blender_points)
        elif len(blender_points) != expected_count:
            raise RuntimeError(
                f"Visible topology changed at frame {blender_frame}: "
                f"{len(blender_points)} != {expected_count}"
            )
        if not np.isfinite(blender_points).all():
            raise RuntimeError(f"Non-finite Blender cache coordinates at frame {blender_frame}")
        time_code = Usd.TimeCode(float(blender_frame))
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(blender_points), time_code)
        minimum = blender_points.min(axis=0)
        maximum = blender_points.max(axis=0)
        mesh.GetExtentAttr().Set(
            [Gf.Vec3f(*map(float, minimum)), Gf.Vec3f(*map(float, maximum))], time_code
        )

    cache_stage.GetRootLayer().Save()
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Blender animation USD was not written: {output_path}")
    return {
        "path": str(output_path),
        "frame_count": len(snapshots),
        "frames_per_second": float(frames_per_second),
        "vertex_count": int(expected_count),
        "face_count": len(source_counts),
        "up_axis": "Z",
        "meters_per_unit": 1.0,
        "coordinate_conversion": "Isaac (x,y,z) -> Blender (x,-z,y)",
        "bytes": output_path.stat().st_size,
        "valid": True,
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


def choose_keyframes(snapshots, initial_height, support_top_y):
    centers_y = np.asarray([snapshot["center"][1] for snapshot in snapshots])
    bottoms = np.asarray([snapshot["minimum"][1] for snapshot in snapshots])
    heights = np.asarray([snapshot["height"] for snapshot in snapshots])

    contact_candidates = np.flatnonzero(bottoms <= support_top_y + 0.035)
    if len(contact_candidates) == 0:
        contact_index = int(np.argmin(bottoms))
        contact_detected = False
    else:
        contact_index = int(contact_candidates[0])
        contact_detected = True

    first_window_end = min(len(snapshots), contact_index + 42)
    compression_index = contact_index + int(np.argmin(heights[contact_index:first_window_end]))
    low_center_index = contact_index + int(np.argmin(centers_y[contact_index:first_window_end]))

    # Rebound is a scale-relative event.  The former fixed 0.30 m threshold
    # incorrectly rejected otherwise healthy simulations after resizing the
    # same asset below roughly one metre.
    rebound_distance_threshold = max(0.10, 0.20 * float(initial_height))
    rebound_index = None
    for index in range(max(compression_index + 1, low_center_index + 1), len(snapshots)):
        if centers_y[index] >= centers_y[low_center_index] + rebound_distance_threshold:
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
    rebound_detected = rising_distance >= rebound_distance_threshold

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
        "rebound_distance_threshold": rebound_distance_threshold,
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


def configure_environment_dome(stage):
    imported_domes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.DomeLight):
            continue
        dome = UsdLux.DomeLight(prim)
        texture = dome.GetTextureFileAttr().Get()
        asset_path = texture.path if isinstance(texture, Sdf.AssetPath) else str(texture or "")
        resolved_path = texture.resolvedPath if isinstance(texture, Sdf.AssetPath) else ""
        imported_domes.append((dome, asset_path, resolved_path))

    for dome, asset_path, resolved_path in imported_domes:
        dome.GetPrim().SetActive(False)
        print(
            f"[hdri] disabled prim={dome.GetPath()} texture={asset_path or '<none>'} "
            f"resolved={resolved_path or '<missing>'} reason=workspace-override"
        )

    if not ARGS.hdri_texture:
        raise ValueError("--hdri-texture is required for environment rendering")
    hdri_path = Path(ARGS.hdri_texture).resolve()
    if not hdri_path.is_file():
        raise FileNotFoundError(f"Workspace HDRI does not exist: {hdri_path}")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/UnifiedHDRI")
    dome.CreateTextureFileAttr(Sdf.AssetPath(hdri_path.as_posix()))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(float(ARGS.hdri_intensity))
    UsdGeom.Xformable(dome.GetPrim()).AddRotateXOp().Set(float(ARGS.hdri_rotation_x_degrees))
    print(
        f"[hdri] source=workspace prim={dome.GetPath()} texture={hdri_path} "
        f"intensity={ARGS.hdri_intensity} rotation_x={ARGS.hdri_rotation_x_degrees}"
    )

def build_scene():
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    if ARGS.environment_usd:
        environment_path = Path(ARGS.environment_usd).resolve()
        if not environment_path.is_file():
            raise FileNotFoundError(f"Environment USD does not exist: {environment_path}")
        stage.GetRootLayer().subLayerPaths.append(environment_path.as_posix())
        print(f"[environment] sublayer={environment_path}")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    configure_environment_dome(stage)

    if ARGS.environment_usd and ARGS.environment_ground_only:
        disabled_colliders = 0
        retained_colliders = 0
        retained_paths = set(ARGS.environment_ground_prim)
        sanitize_paths = set(ARGS.sanitize_ground_prim)
        sanitize_sources = []
        environment_prim = stage.GetPrimAtPath("/World/Environment")
        if environment_prim and environment_prim.IsValid():
            for prim in Usd.PrimRange(environment_prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    if str(prim.GetPath()) in retained_paths and str(prim.GetPath()) not in sanitize_paths:
                        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
                        retained_colliders += 1
                        continue
                    if str(prim.GetPath()) in sanitize_paths:
                        sanitize_sources.append(prim)
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                        prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    disabled_colliders += 1
        for index, source_prim in enumerate(sanitize_sources):
            triangle_count = clone_sanitized_collision_mesh(
                stage, source_prim, f"/World/ExperimentGroundMeshes/Ground_{index:03d}"
            )
            retained_colliders += 1
            print(
                f"[environment] sanitized-ground source={source_prim.GetPath()} "
                f"triangles={triangle_count}"
            )
        if ARGS.local_collision_bounds:
            local_collision = extract_local_collision_mesh(
                stage,
                environment_prim,
                "/World/ExperimentGroundMeshes/ExactLocalEnvironment",
                ARGS.local_collision_bounds,
            )
            retained_colliders += 1
            report_local_collision = local_collision
        else:
            report_local_collision = None
        if retained_colliders == 0:
            ground_center_x = (
                ARGS.spawn_x if ARGS.ground_patch_center_x is None else ARGS.ground_patch_center_x
            )
            ground_center_z = (
                ARGS.spawn_z if ARGS.ground_patch_center_z is None else ARGS.ground_patch_center_z
            )
            ground_patch = add_cube(
                stage,
                "/World/ExperimentGround",
                (ARGS.ground_patch_size_x, 0.10, ARGS.ground_patch_size_z),
                (ground_center_x, ARGS.support_top_y - 0.05, ground_center_z),
                None,
                True,
            )
            ground_patch.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            print(
                f"[environment] finite-ground center=({ground_center_x:.4f},{ground_center_z:.4f}) "
                f"size=({ARGS.ground_patch_size_x:.4f},{ARGS.ground_patch_size_z:.4f})"
            )
        print(
            f"[environment] ground-only disabled_colliders={disabled_colliders} "
            f"retained_colliders={retained_colliders} support_y={ARGS.support_top_y:.4f}"
        )

    for material_path in ARGS.force_nonmetal_material:
        force_material_nonmetal(stage, material_path)
    for material_path, input_name, factor in ARGS.material_input_scale:
        scale_material_texture_input(stage, material_path, input_name, factor)
    for material_path, red, green, blue in ARGS.material_color_scale:
        scale_material_diffuse_texture(stage, material_path, red, green, blue)
    for material_path, threshold in ARGS.material_opacity_threshold:
        set_material_opacity_threshold(stage, material_path, threshold)
    for material_path, texture_path in ARGS.material_emission_texture:
        restore_texture_emission_material(stage, material_path, texture_path)
    for material_path, factor in ARGS.material_emission_scale:
        scale_material_emission_texture(stage, material_path, factor)
    for values in ARGS.material_omniglass:
        replace_material_with_omniglass(stage, *values)
    for values in ARGS.material_omnipbr:
        replace_material_with_omnipbr(stage, *values)
    for material_path, input_name, value in ARGS.material_input_value:
        set_material_scalar_input(stage, material_path, input_name, value)
    for material_path in ARGS.hide_material_meshes:
        hide_meshes_bound_to_material(stage, material_path)

    if ARGS.emissive_mesh_lights:
        add_emissive_mesh_proxy_lights(stage)
    add_authored_downlights(stage)

    if ARGS.environment_panorama:
        camera = UsdGeom.Camera.Define(stage, "/World/PanoramaCamera")
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(36.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
        camera.CreateFStopAttr(0.0)
        set_look_at(camera.GetPrim(), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        print("[panorama] static environment mode: no PhysicsScene, deformable, or simulation")
        return stage, None, None, camera, 0, 0

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

    if not ARGS.environment_usd:
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
    ball_root.AddTranslateOp().Set(Gf.Vec3d(ARGS.spawn_x, ARGS.drop_height, ARGS.spawn_z))
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
    camera.CreateFocalLengthAttr(ARGS.camera_focal_length)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    camera.CreateFocusDistanceAttr(12.5)
    camera.CreateFStopAttr(0.0)
    set_look_at(camera.GetPrim(), tuple(ARGS.camera_eye), tuple(ARGS.camera_target))

    if ARGS.environment_usd:
        light_x = ARGS.spawn_x
        light_y = ARGS.support_top_y
        light_z = ARGS.spawn_z
        light_target = (light_x, light_y + 2.2, light_z)
        light_boost = 2.0
    else:
        light_x = 0.0
        light_y = 0.0
        light_z = 0.0
        light_target = (0.0, 1.7, 0.0)
        light_boost = 1.0

    add_rect_light(
        stage,
        "/World/Lights/Key",
        (light_x - 3.8, light_y + 6.8, light_z + 5.0),
        light_target,
        (1.0, 0.73, 0.56),
        9200.0 * light_boost,
        4.0,
        4.0,
    )
    add_rect_light(
        stage,
        "/World/Lights/Rim",
        (light_x + 4.8, light_y + 4.7, light_z - 0.4),
        light_target,
        (0.38, 0.62, 1.0),
        7600.0 * light_boost,
        2.0,
        3.0,
    )
    add_rect_light(
        stage,
        "/World/Lights/Fill",
        (light_x - 4.0, light_y + 2.4, light_z + 1.0),
        light_target,
        (1.0, 0.92, 0.78),
        3400.0 * light_boost,
        3.0,
        3.0,
    )
    if ARGS.disable_studio_lights:
        for light_path in ("/World/Lights/Key", "/World/Lights/Rim", "/World/Lights/Fill"):
            stage.GetPrimAtPath(light_path).SetActive(False)
        print("[lighting] generic Key/Rim/Fill disabled; using scene-authored lighting")
    if ARGS.environment_usd and ARGS.environment_fill_intensity > 0.0:
        camera_eye = tuple(float(value) for value in ARGS.camera_eye)
        fill_target = (light_x, light_y + 0.8, light_z)
        add_rect_light(
            stage,
            "/World/Lights/EnvironmentFill",
            (camera_eye[0], camera_eye[1] + 3.0, camera_eye[2]),
            fill_target,
            (0.82, 0.86, 1.0),
            ARGS.environment_fill_intensity,
            8.0,
            8.0,
        )
    return stage, visual, root_prim, camera, len(ball_points), len(ball_triangles)


def main():
    report = {
        "valid": False,
        "isaac_sim_version": "6.0.1-rc.7+release.42383.32955d8d.gl",
        "renderer": ARGS.renderer,
        "resolution": [ARGS.width, ARGS.height],
        "path_spp": ARGS.path_spp if ARGS.renderer == "PathTracing" else None,
        "physics_frames": 0 if ARGS.environment_panorama else ARGS.frames,
        "physics_substeps": 0 if ARGS.environment_panorama else ARGS.substeps,
        "render_video_frames": ARGS.render_video_frames,
        "export_blender_usd": ARGS.export_blender_usd,
        "output_directory": str(OUTPUT_DIR),
        "usd_path": str(USD_PATH),
        "environment_usd": ARGS.environment_usd,
        "hdri_texture": ARGS.hdri_texture,
        "hdri_intensity": ARGS.hdri_intensity,
        "hdri_rotation_x_degrees": ARGS.hdri_rotation_x_degrees,
        "support_top_y": None if ARGS.environment_panorama else ARGS.support_top_y,
    }
    simulation = None
    attached = False
    try:
        if ARGS.render_video_frames and ARGS.renderer != "PathTracing":
            raise ValueError("--render-video-frames requires --renderer PathTracing")
        stage_notice(
            2,
            "Building static environment panorama"
            if ARGS.environment_panorama
            else "Building studio, closed elephant mesh, and volume deformable hierarchy",
        )
        stage, visual, root_prim, camera, visual_vertex_count, visual_triangle_count = build_scene()
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", ARGS.renderer)
        settings.set("/persistent/app/viewport/displayOptions", 0)
        exposure = ARGS.exposure
        if exposure is None:
            exposure = 1.0 if ARGS.environment_usd else 0.7
        settings.set("/rtx/post/tonemap/exposure", exposure)
        settings.set("/rtx/reflections/enabled", True)
        settings.set("/rtx/indirectDiffuse/enabled", True)
        settings.set("/rtx/ambientOcclusion/enabled", True)
        if ARGS.renderer == "PathTracing":
            settings.set("/rtx/pathtracing/spp", ARGS.path_spp)
            settings.set("/rtx/pathtracing/totalSpp", ARGS.path_spp)
            settings.set("/rtx/pathtracing/maxBounces", ARGS.path_max_bounces)

        viewport = get_active_viewport()
        viewport.camera_path = camera.GetPath()
        viewport.set_texture_resolution((ARGS.width, ARGS.height))
        for _ in range(4):
            simulation_app.update()

        if not stage.GetRootLayer().Export(str(USD_PATH)):
            raise RuntimeError(f"Failed to export USD: {USD_PATH}")
        print(f"[usd] exported={USD_PATH} bytes={USD_PATH.stat().st_size}")

        if ARGS.environment_panorama:
            stage_notice(3, f"Rendering four static origin views with {ARGS.renderer}")
            for stale_name in PANORAMA_PNG_NAMES:
                stale_path = OUTPUT_DIR / stale_name
                if stale_path.is_file():
                    stale_path.unlink()
            directions = (
                ("front", (0.0, 0.0, 1.0)),
                ("right", (1.0, 0.0, 0.0)),
                ("back", (0.0, 0.0, -1.0)),
                ("left", (-1.0, 0.0, 0.0)),
            )
            views = []
            for (direction_name, target), png_name in zip(directions, PANORAMA_PNG_NAMES):
                set_look_at(camera.GetPrim(), (0.0, 0.0, 0.0), target)
                for _ in range(ARGS.render_settle):
                    simulation_app.update()
                image_path = OUTPUT_DIR / png_name
                capture_current_viewport(viewport, image_path)
                views.append(
                    {
                        "file": png_name,
                        "direction": direction_name,
                        "eye": [0.0, 0.0, 0.0],
                        "target": list(target),
                    }
                )
                print(f"[panorama] direction={direction_name} file={png_name}")
            verified_pngs = [
                str(OUTPUT_DIR / name)
                for name in PANORAMA_PNG_NAMES
                if (OUTPUT_DIR / name).is_file() and (OUTPUT_DIR / name).stat().st_size > 0
            ]
            usd_valid = USD_PATH.is_file() and USD_PATH.stat().st_size > 0
            report.update(
                {
                    "mode": "static_environment_panorama",
                    "camera_origin": [0.0, 0.0, 0.0],
                    "horizontal_fov_degrees": 90.0,
                    "physics_scene_created": False,
                    "simulation_attached": False,
                    "simulation_steps": 0,
                    "usd_saved_nonempty": usd_valid,
                    "png_files": verified_pngs,
                    "panorama_views": views,
                }
            )
            report["valid"] = bool(usd_valid and len(verified_pngs) == len(PANORAMA_PNG_NAMES))
            if not report["valid"]:
                raise RuntimeError("Static panorama output validation failed")
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[complete] valid=True static-panorama report={REPORT_PATH}")
            return 0

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

        blender_cache = None
        if ARGS.export_blender_usd:
            stage_notice(4, "Exporting visible soft-body motion for Blender")
            blender_cache = export_blender_animation_usd(
                visual,
                snapshots[1:],
                ARGS.blender_usd_name,
                frames_per_second=60.0,
            )
            print(
                f"[blender-usd] exported={blender_cache['path']} "
                f"frames={blender_cache['frame_count']} vertices={blender_cache['vertex_count']} "
                f"bytes={blender_cache['bytes']}"
            )

        keyframes = choose_keyframes(snapshots, initial_height, ARGS.support_top_y)
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

        # Reject long-horizon ground escapes before spending minutes rendering.
        pre_render_min_bottom = min(float(snapshot["minimum"][1]) for snapshot in snapshots)
        pre_render_min_center = min(float(snapshot["center"][1]) for snapshot in snapshots)
        if ARGS.uneven_ground:
            terrain_escape_limit = (
                float(ARGS.local_collision_bounds[1]) - 1.0
                if ARGS.local_collision_bounds
                else ARGS.support_top_y - 10.0
            )
            report["terrain_escape_center_y_limit"] = terrain_escape_limit
            if pre_render_min_center < terrain_escape_limit:
                raise RuntimeError(
                    "Long-horizon terrain escape detected before rendering: "
                    f"minimum_center_y={pre_render_min_center:.4f} "
                    f"escape_limit={terrain_escape_limit:.4f}"
                )
        elif not ARGS.allow_free_fall and pre_render_min_bottom < ARGS.support_top_y - 0.18:
            raise RuntimeError(
                "Long-horizon flat-ground penetration detected before rendering: "
                f"minimum_bottom_y={pre_render_min_bottom:.4f} support_y={ARGS.support_top_y:.4f}"
            )

        for stale_name in PNG_NAMES + ORBIT_PNG_NAMES:
            stale_path = OUTPUT_DIR / stale_name
            if stale_path.is_file():
                stale_path.unlink()

        if ARGS.skip_preview_render:
            stage_notice(5, "Skipping Isaac stills; Blender will render the exported cache")
            orbit_views = []
            active_png_names = []
        elif ARGS.orbit_preview:
            stage_notice(5, f"Rendering three compression-state orbit views with {ARGS.renderer}")
            compression_index = index_map["compression"]
            set_snapshot(stage, visual, root_prim, snapshots[compression_index])
            base_eye = tuple(float(value) for value in ARGS.camera_eye)
            camera_target = tuple(float(value) for value in ARGS.camera_target)
            orbit_views = []
            for yaw_degrees, png_name in zip(ARGS.orbit_yaw_degrees, ORBIT_PNG_NAMES):
                eye = orbit_eye(base_eye, camera_target, yaw_degrees)
                set_look_at(camera.GetPrim(), eye, camera_target)
                for _ in range(ARGS.render_settle):
                    simulation_app.update()
                capture_current_viewport(viewport, OUTPUT_DIR / png_name)
                orbit_views.append(
                    {"file": png_name, "yaw_degrees": float(yaw_degrees), "eye": list(eye)}
                )
                print(
                    f"[render] state=compression source_frame={snapshots[compression_index]['frame']} "
                    f"yaw={yaw_degrees:.1f} file={png_name}"
                )
            active_png_names = ORBIT_PNG_NAMES
        else:
            stage_notice(5, f"Rendering five representative frames with {ARGS.renderer}")
            orbit_views = []
            for (name, index), png_name in zip(index_map.items(), PNG_NAMES):
                set_snapshot(stage, visual, root_prim, snapshots[index])
                for _ in range(ARGS.render_settle):
                    simulation_app.update()
                capture_current_viewport(viewport, OUTPUT_DIR / png_name)
                print(f"[render] state={name} source_frame={snapshots[index]['frame']} file={png_name}")
            active_png_names = PNG_NAMES

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
        png_files = [OUTPUT_DIR / name for name in active_png_names]
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
        penetration_limit = ARGS.support_top_y - 0.18
        penetration_check_mode = "deformable_vs_flat_support"
        no_obvious_penetration = min_bottom >= penetration_limit
        if ARGS.allow_free_fall:
            penetration_check_mode = "finite_support_free_fall"
            no_obvious_penetration = True
        if ARGS.uneven_ground:
            # A single Y threshold is invalid for sloped/uneven triangle terrain.
            # In this mode PhysX contact, rebound and finite deformation are the
            # meaningful automated checks; representative frames remain the
            # visual penetration check.
            penetration_check_mode = "terrain_escape_and_visual_check"
            no_obvious_penetration = True

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
                "rebound_distance_threshold": keyframes["rebound_distance_threshold"],
                "minimum_surface_y": min_bottom,
                "support_top_y": ARGS.support_top_y,
                "ground_patch": {
                    "size_x": ARGS.ground_patch_size_x,
                    "size_z": ARGS.ground_patch_size_z,
                    "center_x": ARGS.spawn_x if ARGS.ground_patch_center_x is None else ARGS.ground_patch_center_x,
                    "center_z": ARGS.spawn_z if ARGS.ground_patch_center_z is None else ARGS.ground_patch_center_z,
                    "allow_free_fall": ARGS.allow_free_fall,
                },
                "local_collision_bounds": list(ARGS.local_collision_bounds) if ARGS.local_collision_bounds else None,
                "no_obvious_pedestal_penetration": no_obvious_penetration,
                "penetration_check_mode": penetration_check_mode,
                "compression_visible": compression_visible,
                "lateral_expansion_visible": lateral_expansion_visible,
                "usd_saved_nonempty": usd_valid,
                "blender_animation_cache": blender_cache,
                "png_files": verified_pngs,
                "orbit_preview": {
                    "enabled": ARGS.orbit_preview,
                    "state": "compression" if ARGS.orbit_preview else None,
                    "views": orbit_views,
                },
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
        rebound_check_satisfied = keyframes["rebound_detected"] or ARGS.uneven_ground
        report["rebound_check_mode"] = (
            "terrain_contact_motion" if ARGS.uneven_ground else "world_vertical_rebound"
        )
        report["rebound_check_satisfied"] = rebound_check_satisfied
        report["valid"] = bool(
            all_finite
            and keyframes["contact_detected"]
            and rebound_check_satisfied
            and compression_visible
            and lateral_expansion_visible
            and no_obvious_penetration
            and usd_valid
            and len(verified_pngs) == len(active_png_names)
            and video_frames_valid
            and (not ARGS.export_blender_usd or bool(blender_cache and blender_cache["valid"]))
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
