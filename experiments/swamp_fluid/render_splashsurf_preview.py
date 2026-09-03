"""Render one Splashsurf OBJ inside the authored Swamp Blender scene.

Run from Blender with the Swamp ``.blend`` already opened::

    blender swamp.blend --background --python render_splashsurf_preview.py -- \
        SURFACE_OBJ OUTPUT_PNG [SPHERE_X SPHERE_Y SPHERE_Z]

Splashsurf receives Isaac Y-up particle coordinates.  The imported object and
the impactor are converted to Blender Z-up without modifying the source scene.
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) < 2:
    raise SystemExit("Expected SURFACE_OBJ OUTPUT_PNG [SPHERE_X SPHERE_Y SPHERE_Z]")

surface_path = Path(argv[0]).resolve()
output_path = Path(argv[1]).resolve()
sphere_isaac = tuple(float(value) for value in argv[2:5]) if len(argv) >= 5 else None
output_path.parent.mkdir(parents=True, exist_ok=True)


def isaac_to_blender(values):
    x, y, z = values
    return Vector((float(x), -float(z), float(y)))


def set_principled_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return
    raise RuntimeError(f"Missing Principled BSDF input: {names}")


def configure_cycles(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    if hasattr(scene.cycles, "use_animated_seed"):
        scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type in {"OPTIX", "CUDA"}
        scene.cycles.device = "GPU"
    except (KeyError, TypeError, RuntimeError):
        scene.cycles.device = "CPU"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"


def configure_hdri(scene):
    hdri_path = Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr")
    if not hdri_path.is_file():
        raise FileNotFoundError(hdri_path)
    world = scene.world or bpy.data.worlds.new("SwampSplashsurfWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.95
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])


def make_water_material():
    material = bpy.data.materials.new("SplashsurfWaterOpenSurface")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(principled, (0.82, 0.94, 0.98, 1.0), "Base Color")
    set_principled_input(principled, 0.03, "Roughness")
    set_principled_input(principled, 1.333, "IOR")
    set_principled_input(principled, 1.0, "Transmission Weight", "Transmission")
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    # This OBJ is an open free-surface sheet. Volume absorption would use an
    # undefined path length and can create dark temporal flashes.
    return material


scene = bpy.context.scene
configure_cycles(scene)
configure_hdri(scene)

bpy.ops.wm.obj_import(filepath=str(surface_path))
water = bpy.context.active_object
water.name = "SplashsurfWaterFrame"
water.rotation_euler[0] = math.radians(90.0)
water.data.materials.clear()
water.data.materials.append(make_water_material())
for polygon in water.data.polygons:
    polygon.use_smooth = True
water_bounds = [water.matrix_world @ Vector(corner) for corner in water.bound_box]
print(
    "SPLASHSURF_BOUNDS_BLENDER="
    f"{[(min(point[axis] for point in water_bounds), max(point[axis] for point in water_bounds)) for axis in range(3)]}"
)

if sphere_isaac is not None:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64,
        ring_count=32,
        radius=0.08,
        location=isaac_to_blender(sphere_isaac),
    )
    sphere = bpy.context.active_object
    sphere.name = "PhysXImpactorFrame"
    material = bpy.data.materials.new("PhysXImpactorOrange")
    material.diffuse_color = (0.95, 0.18, 0.035, 1.0)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    principled = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(principled, (0.95, 0.18, 0.035, 1.0), "Base Color")
    set_principled_input(principled, 0.24, "Roughness")
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    sphere.data.materials.append(material)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True

camera_data = bpy.data.cameras.new("SplashsurfPreviewCamera")
camera = bpy.data.objects.new("SplashsurfPreviewCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
eye = isaac_to_blender((0.12, 1.15, -1.57))
target = isaac_to_blender((-0.73, -1.41099, 0.78))
camera.location = eye
camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
camera_data.lens = 40.0
camera_data.sensor_width = 24.0

scene.render.filepath = str(output_path)
bpy.ops.render.render(write_still=True)
print(f"SPLASHSURF_PREVIEW={output_path}")
