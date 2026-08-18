"""Print a compact JSON audit of the current Blender scene's authored lighting."""

import json

import bpy


scene = bpy.context.scene
world = scene.world
background_nodes = []
environment_nodes = []
if world and world.use_nodes and world.node_tree:
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            background_nodes.append(
                {
                    "name": node.name,
                    "color": list(node.inputs["Color"].default_value),
                    "strength": float(node.inputs["Strength"].default_value),
                    "linked_color": bool(node.inputs["Color"].is_linked),
                }
            )
        elif node.type == "TEX_ENVIRONMENT":
            absolute_path = bpy.path.abspath(node.image.filepath) if node.image else None
            environment_nodes.append(
                {
                    "name": node.name,
                    "image": node.image.filepath if node.image else None,
                    "absolute_path": absolute_path,
                    "exists": bool(absolute_path and __import__("pathlib").Path(absolute_path).is_file()),
                    "packed": bool(node.image and node.image.packed_file),
                    "color_output_linked": bool(node.outputs.get("Color") and node.outputs["Color"].is_linked),
                }
            )

lights = []
cameras = []
for obj in bpy.data.objects:
    if obj.type == "LIGHT":
        lights.append(
            {
                "name": obj.name,
                "type": obj.data.type,
                "energy": float(obj.data.energy),
                "location": list(obj.location),
                "hide_render": bool(obj.hide_render),
            }
        )
    elif obj.type == "CAMERA":
        cameras.append(
            {
                "name": obj.name,
                "location": list(obj.location),
                "rotation_euler": list(obj.rotation_euler),
                "lens": float(obj.data.lens),
                "hide_render": bool(obj.hide_render),
            }
        )

print("BLENDER_LIGHT_AUDIT_BEGIN")
print(
    json.dumps(
        {
            "blend": bpy.data.filepath,
            "engine": scene.render.engine,
            "exposure": float(scene.view_settings.look) if False else float(scene.view_settings.exposure),
            "world": world.name if world else None,
            "world_color": list(world.color) if world else None,
            "background_nodes": background_nodes,
            "environment_nodes": environment_nodes,
            "lights": lights,
            "cameras": cameras,
        },
        indent=2,
    )
)
print("BLENDER_LIGHT_AUDIT_END")
