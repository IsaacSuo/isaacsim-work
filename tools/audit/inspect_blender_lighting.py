"""Print Blender scene lighting and material diagnostics without modifying the file."""

import json
from pathlib import Path

import bpy


scene = bpy.context.scene


def socket_value(node, name):
    socket = node.inputs.get(name)
    if socket is None or socket.is_linked:
        return None
    value = socket.default_value
    try:
        return list(value)
    except TypeError:
        return value


lights = []
for obj in bpy.data.objects:
    if obj.type != "LIGHT":
        continue
    data = obj.data
    lights.append(
        {
            "name": obj.name,
            "type": data.type,
            "energy": data.energy,
            "use_shadow": data.use_shadow,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
        }
    )

materials = []
for material in bpy.data.materials:
    if not material.use_nodes or material.node_tree is None:
        continue
    nodes = []
    for node in material.node_tree.nodes:
        if node.type in {"BSDF_PRINCIPLED", "EMISSION", "OUTPUT_MATERIAL"}:
            entry = {"name": node.name, "type": node.type}
            if node.type == "BSDF_PRINCIPLED":
                entry.update(
                    {
                        "base_color": socket_value(node, "Base Color"),
                        "metallic": socket_value(node, "Metallic"),
                        "roughness": socket_value(node, "Roughness"),
                        "emission_color": socket_value(node, "Emission Color"),
                        "emission_strength": socket_value(node, "Emission Strength"),
                    }
                )
            elif node.type == "EMISSION":
                entry.update(
                    {
                        "color": socket_value(node, "Color"),
                        "strength": socket_value(node, "Strength"),
                    }
                )
            nodes.append(entry)
    if nodes:
        materials.append({"name": material.name, "nodes": nodes})

payload = {
    "blend": str(Path(bpy.data.filepath)),
    "engine": scene.render.engine,
    "lights": lights,
    "materials": materials,
    "world": None,
}
if scene.world and scene.world.use_nodes and scene.world.node_tree:
    payload["world"] = {
        "name": scene.world.name,
        "nodes": [
            {
                "name": node.name,
                "type": node.type,
                "image": bpy.path.abspath(node.image.filepath) if node.type == "TEX_ENVIRONMENT" and node.image else None,
                "strength": socket_value(node, "Strength"),
            }
            for node in scene.world.node_tree.nodes
        ],
    }
print("BLENDER_DIAGNOSTICS=" + json.dumps(payload, ensure_ascii=False))
print("BLENDER_WORLD=" + json.dumps(payload["world"], ensure_ascii=False))
