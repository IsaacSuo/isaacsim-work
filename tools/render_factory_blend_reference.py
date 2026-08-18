"""Render factory.blend from the Isaac preview cameras without saving the source blend."""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
HDRI = Path(sys.argv[sys.argv.index("--") + 2]).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)

ISAAC_EYE = Vector((6.0, 2.5, 10.0))
ISAAC_TARGET = Vector((1.823781, 0.5, 5.442124))
ORBIT_YAWS = (-35.0, -18.0, 0.0)


def isaac_to_blender(point):
    return Vector((point.x, -point.z, point.y))


def orbit_eye(base_eye, target, yaw_degrees):
    offset = base_eye - target
    angle = math.radians(yaw_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return Vector(
        (
            target.x + cosine * offset.x + sine * offset.z,
            target.y + offset.y,
            target.z - sine * offset.x + cosine * offset.z,
        )
    )


def look_at(camera, eye, target):
    camera.location = isaac_to_blender(eye)
    direction = isaac_to_blender(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def world_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return {
        "minimum": [min(point[axis] for point in corners) for axis in range(3)],
        "maximum": [max(point[axis] for point in corners) for axis in range(3)],
    }


def mesh_report(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return {"name": name, "missing": True}
    mesh = obj.data
    world_z = [float((obj.matrix_world @ vertex.co).z) for vertex in mesh.vertices]
    return {
        "name": name,
        "type": obj.type,
        "location": [float(value) for value in obj.matrix_world.translation],
        "bounds": world_bounds(obj),
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.polygons),
        "world_z_min": min(world_z),
        "world_z_max": max(world_z),
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
    }


scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.cycles.device = "GPU"
try:
    cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
    cycles_preferences.compute_device_type = "OPTIX"
    cycles_preferences.get_devices()
    for device in cycles_preferences.devices:
        device.use = device.type != "CPU"
except Exception as error:
    scene.cycles.device = "CPU"
    print(f"[cycles] GPU setup failed, using CPU: {error}", flush=True)
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.image_settings.color_mode = "RGBA"

world = scene.world or bpy.data.worlds.new("FactoryReferenceWorld")
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
output = nodes.new("ShaderNodeOutputWorld")
background = nodes.new("ShaderNodeBackground")
environment = nodes.new("ShaderNodeTexEnvironment")
environment.image = bpy.data.images.load(str(HDRI), check_existing=True)
background.inputs["Strength"].default_value = 1.0
world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

camera_data = bpy.data.cameras.new("IsaacReferenceCamera")
camera = bpy.data.objects.new("IsaacReferenceCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.lens = 58.0
camera.data.sensor_width = 36.0

rendered = []
for label, yaw in zip(("left", "center", "right"), ORBIT_YAWS):
    eye = orbit_eye(ISAAC_EYE, ISAAC_TARGET, yaw)
    look_at(camera, eye, ISAAC_TARGET)
    path = OUTPUT / f"blend_{label}.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    rendered.append({"view": label, "yaw": yaw, "eye_isaac": list(eye), "file": str(path)})

report = {
    "blend": bpy.data.filepath,
    "engine": scene.render.engine,
    "hdri": str(HDRI),
    "renders": rendered,
    "water": mesh_report("Object_24"),
    "road": mesh_report("Object_40"),
}
(OUTPUT / "reference_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False))
