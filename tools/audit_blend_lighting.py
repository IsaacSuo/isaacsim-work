"""Read Blender scenes and report authored lights, world lighting, and emissive materials."""

import json
import math
import sys
from pathlib import Path

import bpy


def value(input_socket):
    if input_socket is None:
        return None
    if input_socket.is_linked:
        return {
            "linked": True,
            "sources": [
                {
                    "node": link.from_node.name,
                    "node_type": link.from_node.bl_idname,
                    "socket": link.from_socket.name,
                }
                for link in input_socket.links
            ],
        }
    raw = getattr(input_socket, "default_value", None)
    if raw is None:
        return None
    if isinstance(raw, (str, int, float, bool)):
        return raw
    try:
        return [float(item) for item in raw]
    except (TypeError, ValueError):
        return str(raw)


def socket_by_names(node, names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def material_emission(material):
    if not material.use_nodes or material.node_tree is None:
        return []
    rows = []
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeEmission":
            rows.append(
                {
                    "node": node.name,
                    "node_type": node.bl_idname,
                    "color": value(node.inputs.get("Color")),
                    "strength": value(node.inputs.get("Strength")),
                }
            )
        elif node.bl_idname == "ShaderNodeBsdfPrincipled":
            color = socket_by_names(node, ["Emission Color", "Emission"])
            strength = socket_by_names(node, ["Emission Strength"])
            color_value = value(color)
            strength_value = value(strength)
            linked = bool(color and color.is_linked) or bool(strength and strength.is_linked)
            active_constant = False
            if isinstance(strength_value, (int, float)) and strength_value > 0:
                if isinstance(color_value, list):
                    active_constant = max(color_value[:3], default=0.0) > 0
            if linked or active_constant:
                rows.append(
                    {
                        "node": node.name,
                        "node_type": node.bl_idname,
                        "color": color_value,
                        "strength": strength_value,
                    }
                )
    return rows


def object_row(obj):
    rotation = obj.matrix_world.to_quaternion()
    return {
        "name": obj.name,
        "type": obj.type,
        "location": [float(v) for v in obj.matrix_world.translation],
        "dimensions": [float(v) for v in obj.dimensions],
        "rotation_quaternion_wxyz": [
            float(rotation.w),
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
        ],
        "visible_render": not obj.hide_render,
    }


def audit_world():
    world = bpy.context.scene.world
    if world is None:
        return None
    row = {"name": world.name, "use_nodes": world.use_nodes, "nodes": []}
    if not world.use_nodes or world.node_tree is None:
        row["color"] = [float(v) for v in world.color]
        return row
    for node in world.node_tree.nodes:
        if node.bl_idname == "ShaderNodeBackground":
            row["nodes"].append(
                {
                    "node": node.name,
                    "node_type": node.bl_idname,
                    "color": value(node.inputs.get("Color")),
                    "strength": value(node.inputs.get("Strength")),
                }
            )
        elif node.bl_idname == "ShaderNodeTexEnvironment":
            row["nodes"].append(
                {
                    "node": node.name,
                    "node_type": node.bl_idname,
                    "image": bpy.path.abspath(node.image.filepath) if node.image else None,
                    "projection": node.projection,
                }
            )
    return row


def audit_file(path):
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    lights = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        data = obj.data
        row = object_row(obj)
        row.update(
            {
                "light_type": data.type,
                "energy": float(data.energy),
                "color": [float(v) for v in data.color],
                "use_shadow": bool(data.use_shadow),
            }
        )
        for attribute in ("shape", "size", "size_y", "spot_size", "spot_blend"):
            if hasattr(data, attribute):
                raw = getattr(data, attribute)
                row[attribute] = float(raw) if isinstance(raw, (int, float)) else raw
        lights.append(row)

    emissive_materials = []
    for material in bpy.data.materials:
        nodes = material_emission(material)
        if not nodes:
            continue
        users = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                continue
            if any(slot.material == material for slot in obj.material_slots):
                users.append(object_row(obj))
        emissive_materials.append(
            {
                "material": material.name,
                "nodes": nodes,
                "object_count": len(users),
                "objects": users,
            }
        )
    return {
        "scene": path.stem,
        "blend": str(path),
        "render_engine": bpy.context.scene.render.engine,
        "world": audit_world(),
        "light_count": len(lights),
        "lights": lights,
        "emissive_material_count": len(emissive_materials),
        "emissive_materials": emissive_materials,
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("usage: blender -b --python audit_blend_lighting.py -- OUTPUT BLEND...")
    output = Path(argv[0])
    results = []
    for raw in argv[1:]:
        path = Path(raw)
        print(f"[blend-light-audit] {path}", flush=True)
        results.append(audit_file(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"scene_count": len(results), "results": results}, indent=2), encoding="utf-8")
    print(f"[blend-light-audit] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
