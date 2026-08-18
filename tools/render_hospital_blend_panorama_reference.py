"""Render hospital.blend from the Isaac origin panorama cameras without saving it."""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)


def isaac_to_blender(point):
    return Vector((point.x, -point.z, point.y))


def look_at(camera, eye, target):
    camera.location = isaac_to_blender(eye)
    direction = isaac_to_blender(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.cycles.device = "GPU"
try:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type != "CPU"
except Exception as error:
    scene.cycles.device = "CPU"
    print(f"[cycles] GPU setup failed, using CPU: {error}", flush=True)

scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False

camera_data = bpy.data.cameras.new("OriginPanoramaReferenceCamera")
camera = bpy.data.objects.new("OriginPanoramaReferenceCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.lens = 18.0
camera.data.sensor_width = 36.0
camera.data.sensor_height = 36.0

eye = Vector((0.0, 0.0, 0.0))
directions = (
    ("front", Vector((0.0, 0.0, 1.0))),
    ("right", Vector((1.0, 0.0, 0.0))),
    ("back", Vector((0.0, 0.0, -1.0))),
    ("left", Vector((-1.0, 0.0, 0.0))),
)
rendered = []
for label, target in directions:
    look_at(camera, eye, target)
    path = OUTPUT / f"blend_{label}.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    rendered.append({"view": label, "eye_isaac": list(eye), "target_isaac": list(target), "file": str(path)})

world = scene.world
world_backgrounds = []
if world and world.use_nodes and world.node_tree:
    for node in world.node_tree.nodes:
        if node.bl_idname == "ShaderNodeBackground":
            world_backgrounds.append(
                {
                    "name": node.name,
                    "color": list(node.inputs["Color"].default_value),
                    "strength": float(node.inputs["Strength"].default_value),
                }
            )

report = {
    "blend": bpy.data.filepath,
    "engine": scene.render.engine,
    "samples": scene.cycles.samples,
    "world_backgrounds": world_backgrounds,
    "rendered_light_count": sum(obj.type == "LIGHT" and not obj.hide_render for obj in bpy.data.objects),
    "renders": rendered,
}
(OUTPUT / "reference_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False))
