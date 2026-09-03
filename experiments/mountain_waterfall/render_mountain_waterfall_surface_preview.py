"""Render selected Splashsurf waterfall frames in the authored mountain camera."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("surface_directory", type=Path)
parser.add_argument("output_directory", type=Path)
parser.add_argument("--frames", nargs="+", type=int, default=(0, 6, 12))
parser.add_argument("--engine", choices=("workbench", "cycles"), default="workbench")
parser.add_argument("--samples", type=int, default=24)
parser.add_argument("--resolution", type=int, default=640)
parser.add_argument(
    "--water-only",
    action="store_true",
    help="Hide authored mesh geometry at render time for unobstructed topology diagnosis.",
)
parser.add_argument(
    "--camera-mode",
    choices=("authored", "waterfall-closeup"),
    default="authored",
    help="Keep the scene camera or use the audited mountain waterfall close-up.",
)
parser.add_argument(
    "--camera-location",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    help="Optional Blender-space camera location; requires --camera-target.",
)
parser.add_argument(
    "--camera-target",
    nargs=3,
    type=float,
    metavar=("X", "Y", "Z"),
    help="Optional Blender-space look-at target; requires --camera-location.",
)
parser.add_argument("--camera-lens", type=float, help="Optional focal length in millimetres.")
args = parser.parse_args(argv)

if (args.camera_location is None) != (args.camera_target is None):
    parser.error("--camera-location and --camera-target must be supplied together")

surface_directory = args.surface_directory.resolve()
output_directory = args.output_directory.resolve()
if not surface_directory.is_dir():
    raise FileNotFoundError(surface_directory)
output_directory.mkdir(parents=True, exist_ok=True)

scene = bpy.context.scene
camera = scene.camera or bpy.data.objects.get("摄像机")
if camera is None or camera.type != "CAMERA":
    raise RuntimeError("mountain.blend does not expose its authored camera")
scene.camera = camera
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = False


def aim_camera(location, target, lens):
    camera.location = Vector(location)
    direction = Vector(target) - camera.location
    if direction.length_squared == 0.0:
        raise ValueError("Camera location and target must differ")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = lens


if args.camera_location is not None:
    aim_camera(
        args.camera_location,
        args.camera_target,
        args.camera_lens if args.camera_lens is not None else camera.data.lens,
    )
elif args.camera_mode == "waterfall-closeup":
    # Scene-level, Blender-space framing for the audited waterfall site.  This is
    # deliberately an explicit preset rather than a per-frame camera correction.
    aim_camera((0.15, -0.10, 1.00), (3.72, -1.70, 0.18), args.camera_lens or 55.0)
elif args.camera_lens is not None:
    camera.data.lens = args.camera_lens


def set_principled_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return
    raise RuntimeError(f"Missing Principled BSDF input: {names}")


def water_material():
    material = bpy.data.materials.get("MountainWaterfallGateWater")
    if material is not None:
        return material
    material = bpy.data.materials.new("MountainWaterfallGateWater")
    material.diffuse_color = (0.035, 0.30, 0.48, 0.72)
    material.metallic = 0.0
    material.roughness = 0.08
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(principled, (0.72, 0.90, 0.96, 1.0), "Base Color")
    set_principled_input(principled, 0.035, "Roughness")
    set_principled_input(principled, 1.333, "IOR")
    set_principled_input(principled, 1.0, "Transmission Weight", "Transmission")
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def configure_hdri():
    hdri_path = Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr")
    if not hdri_path.is_file():
        raise FileNotFoundError(hdri_path)
    world = scene.world or bpy.data.worlds.new("MountainWaterfallWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.65
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])


if args.engine == "workbench":
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
else:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 10
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.35
    configure_hdri()
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type in {"OPTIX", "CUDA"}
        scene.cycles.device = "GPU"
    except (KeyError, TypeError, RuntimeError):
        scene.cycles.device = "CPU"

material = water_material()
if args.water_only:
    for scene_object in scene.objects:
        if scene_object.type == "MESH":
            scene_object.hide_render = True
for frame in args.frames:
    source = surface_directory / f"surface_{frame:04d}.obj"
    target = output_directory / f"surface_{frame:04d}_{args.engine}.png"
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    bpy.ops.wm.obj_import(filepath=str(source))
    water = bpy.context.active_object
    water.name = f"MountainWaterfallSurface_{frame:04d}"
    water.hide_render = False
    # Splashsurf is Isaac Y-up; mountain.blend is Blender Z-up.
    water.rotation_euler[0] = math.radians(90.0)
    water.data.materials.clear()
    water.data.materials.append(material)
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    scene.frame_set(frame)
    scene.render.filepath = str(target)
    bpy.ops.render.render(write_still=True)
    print(f"MOUNTAIN_WATERFALL_PREVIEW={target}")
    bpy.data.objects.remove(water, do_unlink=True)
