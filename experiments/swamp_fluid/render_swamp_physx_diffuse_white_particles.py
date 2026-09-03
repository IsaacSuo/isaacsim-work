"""Render native PhysX Diffuse particles as one unclassified whitewater family.

This is a visual compatibility gate.  It combines the existing 16 mm native
Diffuse export with an already validated 8 mm Swamp Splashsurf surface and the
audited Swamp camera/HDRI.  It deliberately does not read or recover
spray/foam/bubble labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diffuse_directory", type=Path)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--diffuse-frames", nargs="+", type=int, required=True)
    parser.add_argument("--surface-frame-offset", type=int, default=14)
    parser.add_argument("--source-sample-scale", type=int, default=4)
    parser.add_argument("--particle-radius", type=float, default=0.0032)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=520)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-eye", nargs=3, type=float, default=(-0.373, -0.335, -0.207))
    parser.add_argument("--camera-target", nargs=3, type=float, default=(-0.73, -1.41099, 0.78))
    parser.add_argument("--camera-lens-mm", type=float, default=55.0)
    parser.add_argument("--camera-sensor-width-mm", type=float, default=24.0)
    parser.add_argument("--camera-clip-start-m", type=float, default=0.005)
    parser.add_argument("--impactor-radius", type=float, default=0.08)
    parser.add_argument(
        "--hdri",
        type=Path,
        default=Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr"),
    )
    parser.add_argument("--hdri-strength", type=float, default=0.95)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def isaac_to_blender(values: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (3,):
        return np.asarray((values[0], -values[2], values[1]), dtype=np.float64)
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def read_binary_xyz_ply(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header_lines: list[bytes] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Expected binary little-endian PLY: {path}")
        vertex_lines = [line for line in header.splitlines() if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError(f"Expected one vertex declaration: {path}")
        count = int(vertex_lines[0].split()[-1])
        expected_properties = ("property float x", "property float y", "property float z")
        if not all(property_line in header for property_line in expected_properties):
            raise ValueError(f"Expected float32 xyz properties: {path}")
        payload = stream.read()
    expected_bytes = count * 12
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PLY payload size mismatch for {path}: expected {expected_bytes}, got {len(payload)}"
        )
    return np.frombuffer(payload, dtype="<f4").reshape(count, 3).copy()


def set_input(node, value, *names: str) -> bool:
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    return False


def make_principled(name, colour, roughness, transmission, ior=1.333, coat=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(shader, colour, "Base Color")
    set_input(shader, roughness, "Roughness")
    set_input(shader, transmission, "Transmission Weight", "Transmission")
    set_input(shader, ior, "IOR")
    set_input(shader, coat, "Coat Weight", "Coat")
    set_input(shader, min(roughness, 0.18), "Coat Roughness")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


ICO_VERTICES = np.asarray(
    [
        (-1, 1.61803398875, 0), (1, 1.61803398875, 0),
        (-1, -1.61803398875, 0), (1, -1.61803398875, 0),
        (0, -1, 1.61803398875), (0, 1, 1.61803398875),
        (0, -1, -1.61803398875), (0, 1, -1.61803398875),
        (1.61803398875, 0, -1), (1.61803398875, 0, 1),
        (-1.61803398875, 0, -1), (-1.61803398875, 0, 1),
    ],
    dtype=np.float64,
)
ICO_VERTICES /= np.linalg.norm(ICO_VERTICES[0])
ICO_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


def white_particle_mesh(points_isaac: np.ndarray, radius: float, material, collection):
    if not len(points_isaac):
        return None
    points = isaac_to_blender(points_isaac)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for center in points:
        local = center + ICO_VERTICES * radius
        offset = len(vertices)
        vertices.extend(map(tuple, local))
        faces.extend(tuple(offset + value for value in face) for face in ICO_FACES)
    mesh = bpy.data.meshes.new("PhysXDiffuseWhiteMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("PhysXDiffuseUnifiedWhite", mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def remove_object(obj) -> None:
    if obj is None:
        return
    mesh = obj.data if isinstance(obj.data, bpy.types.Mesh) else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


args = parse_args()
if min(
    args.particle_radius,
    args.camera_lens_mm,
    args.camera_sensor_width_mm,
    args.camera_clip_start_m,
    args.impactor_radius,
) <= 0.0:
    raise ValueError("Particle, camera and impactor parameters must be positive")
if len(set(args.diffuse_frames)) != len(args.diffuse_frames):
    raise ValueError("diffuse-frames must be unique")

diffuse_directory = args.diffuse_directory.resolve()
surface_directory = args.surface_directory.resolve()
source_directory = args.source_directory.resolve()
output_directory = args.output_directory.resolve()
output_directory.mkdir(parents=True, exist_ok=False)
source_manifest = json.loads((source_directory / "manifest.json").read_text(encoding="utf-8"))

frame_contracts = []
for diffuse_frame in args.diffuse_frames:
    surface_frame = diffuse_frame + args.surface_frame_offset
    source_sample = surface_frame * args.source_sample_scale
    diffuse_path = diffuse_directory / f"diffuse_{diffuse_frame:04d}.ply"
    surface_path = surface_directory / f"surface_{surface_frame:04d}_clipped.obj"
    if source_sample >= len(source_manifest["samples"]):
        raise IndexError(f"Source sample {source_sample} is outside the source manifest")
    source_path = source_directory / source_manifest["samples"][source_sample]["file"]
    for path in (diffuse_path, surface_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frame_contracts.append((diffuse_frame, surface_frame, source_sample, diffuse_path, surface_path, source_path))

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.seed = 0
scene.cycles.max_bounces = 12
scene.cycles.transmission_bounces = 12
render_device = "CPU"
try:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type in {"OPTIX", "CUDA"}
    scene.cycles.device = "GPU"
    render_device = "GPU"
except (KeyError, TypeError, RuntimeError):
    scene.cycles.device = "CPU"

scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = False
scene.render.fps = round(args.fps)
scene.view_settings.look = "AgX - Medium High Contrast"

world = scene.world or bpy.data.worlds.new("SwampPhysXDiffuseWorld")
scene.world = world
world.use_nodes = True
world.node_tree.nodes.clear()
world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
background = world.node_tree.nodes.new("ShaderNodeBackground")
background.inputs["Strength"].default_value = args.hdri_strength
if args.hdri.is_file():
    environment = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(args.hdri.resolve()), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])

camera_data = bpy.data.cameras.new("SwampPhysXDiffuseCamera")
camera = bpy.data.objects.new("SwampPhysXDiffuseCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
target = Vector(isaac_to_blender(args.camera_target))
eye = Vector(isaac_to_blender(args.camera_eye))
camera.location = eye
camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
camera_data.lens = args.camera_lens_mm
camera_data.sensor_width = args.camera_sensor_width_mm
camera_data.clip_start = args.camera_clip_start_m

water_material = make_principled(
    "SwampValidatedSplashsurfWater", (0.78, 0.91, 0.95, 1.0), 0.028, 1.0, coat=0.12
)
white_material = make_principled(
    "PhysXDiffuseUnifiedWhite", (0.94, 0.965, 0.97, 1.0), 0.24, 0.08, coat=0.18
)
impactor_material = make_principled(
    "SwampImpactor", (0.95, 0.18, 0.035, 1.0), 0.24, 0.0, ior=1.45
)
collection = bpy.data.collections.new("SwampPhysXDiffuseWhitewater")
scene.collection.children.link(collection)
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=args.impactor_radius)
impactor = bpy.context.active_object
impactor.name = "SwampImpactor"
impactor.data.materials.append(impactor_material)
for polygon in impactor.data.polygons:
    polygon.use_smooth = True

manifest_path = output_directory / "render_manifest.json"
render_manifest = {
    "schema": 1,
    "product": "swamp_physx_diffuse_unclassified_white_render",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "complete": False,
    "classification_used": False,
    "label_recovery_used": False,
    "compatibility_gate": {
        "native_diffuse_spacing_m": 0.016,
        "validated_surface_spacing_m": 0.008,
        "strict_same_run_claim": False,
        "purpose": "visual evaluation of unified white PhysX Diffuse particles in the validated Swamp lookdev",
    },
    "configuration": {
        "diffuse_frames": args.diffuse_frames,
        "surface_frame_offset": args.surface_frame_offset,
        "source_sample_scale": args.source_sample_scale,
        "particle_radius_m": args.particle_radius,
        "samples": args.samples,
        "resolution": [args.resolution, args.resolution],
        "fps": args.fps,
        "render_device": render_device,
        "camera_eye_isaac_xyz_m": list(args.camera_eye),
        "camera_target_isaac_xyz_m": list(args.camera_target),
    },
    "inputs": {
        "diffuse_directory": str(diffuse_directory),
        "surface_directory": str(surface_directory),
        "source_directory": str(source_directory),
        "blender_scene": str(Path(bpy.data.filepath).resolve()),
        "hdri": str(args.hdri.resolve()),
    },
    "frames": [],
}
atomic_json(manifest_path, render_manifest)

dynamic_objects = []
for sequence_index, contract in enumerate(frame_contracts, start=1):
    for obj in dynamic_objects:
        remove_object(obj)
    dynamic_objects = []
    diffuse_frame, surface_frame, source_sample, diffuse_path, surface_path, source_path = contract

    bpy.ops.wm.obj_import(filepath=str(surface_path))
    water = bpy.context.active_object
    water.name = "SplashsurfWaterFrame"
    water.rotation_euler[0] = math.radians(90.0)
    water.data.materials.clear()
    water.data.materials.append(water_material)
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    dynamic_objects.append(water)

    diffuse_points = read_binary_xyz_ply(diffuse_path)
    diffuse_obj = white_particle_mesh(diffuse_points, args.particle_radius, white_material, collection)
    if diffuse_obj is not None:
        dynamic_objects.append(diffuse_obj)

    with np.load(source_path, allow_pickle=False) as cache:
        sphere_center = np.asarray(cache["sphere_transform"], dtype=np.float64)[3, :3]
    impactor.location = isaac_to_blender(sphere_center)

    output_path = output_directory / f"white_{sequence_index:04d}.png"
    scene.frame_set(sequence_index)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    render_manifest["frames"].append(
        {
            "sequence_index": sequence_index,
            "diffuse_frame": diffuse_frame,
            "surface_frame": surface_frame,
            "source_sample": source_sample,
            "active_diffuse_particles": len(diffuse_points),
            "diffuse_ply": str(diffuse_path),
            "diffuse_sha256": sha256_file(diffuse_path),
            "surface_obj": str(surface_path),
            "surface_sha256": sha256_file(surface_path),
            "source_npz": str(source_path),
            "source_sha256": sha256_file(source_path),
            "png": str(output_path),
            "png_sha256": sha256_file(output_path),
        }
    )
    atomic_json(manifest_path, render_manifest)
    print(
        f"[swamp-white] diffuse={diffuse_frame:04d} surface={surface_frame:04d} "
        f"count={len(diffuse_points)} sequence={sequence_index}/{len(frame_contracts)}"
    )

render_manifest["complete"] = True
render_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
atomic_json(manifest_path, render_manifest)
