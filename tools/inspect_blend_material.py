"""Dump selected Blender material node graphs without modifying the .blend file."""

import json
import sys
from pathlib import Path

import bpy


def json_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return list(value)
    except TypeError:
        return str(value)


argv = sys.argv[sys.argv.index("--") + 1 :]
output = Path(argv[0]).resolve()
needles = [value.lower() for value in argv[1:]]

rows = []
for material in bpy.data.materials:
    tree = material.node_tree
    searchable = [material.name.lower()]
    if tree:
        for node in tree.nodes:
            searchable.extend((node.name.lower(), node.label.lower()))
            image = getattr(node, "image", None)
            if image:
                searchable.extend((image.name.lower(), image.filepath.lower()))
    if needles and not any(needle in text for needle in needles for text in searchable):
        continue

    nodes = []
    links = []
    if tree:
        for node in tree.nodes:
            item = {
                "name": node.name,
                "label": node.label,
                "type": node.bl_idname,
                "operation": getattr(node, "operation", None),
                "blend_type": getattr(node, "blend_type", None),
                "use_clamp": getattr(node, "use_clamp", None),
                "inputs": {
                    socket.name: json_value(socket.default_value)
                    for socket in node.inputs
                    if hasattr(socket, "default_value")
                },
                "input_sockets": [
                    {
                        "index": index,
                        "name": socket.name,
                        "default": json_value(socket.default_value),
                        "linked": socket.is_linked,
                    }
                    for index, socket in enumerate(node.inputs)
                    if hasattr(socket, "default_value")
                ],
            }
            image = getattr(node, "image", None)
            if image:
                item["image"] = {
                    "name": image.name,
                    "filepath": image.filepath,
                    "filepath_raw": image.filepath_raw,
                    "colorspace": image.colorspace_settings.name,
                }
            nodes.append(item)
        for link in tree.links:
            links.append(
                {
                    "from_node": link.from_node.name,
                    "from_socket": link.from_socket.name,
                    "to_node": link.to_node.name,
                    "to_socket": link.to_socket.name,
                }
            )

    users = []
    for obj in bpy.data.objects:
        for index, slot in enumerate(obj.material_slots):
            if slot.material == material:
                users.append({"object": obj.name, "type": obj.type, "slot": index})

    rows.append(
        {
            "material": material.name,
            "use_nodes": material.use_nodes,
            "surface_render_method": getattr(material, "surface_render_method", None),
            "nodes": nodes,
            "links": links,
            "users": users,
        }
    )

output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"materials={len(rows)} output={output}")
