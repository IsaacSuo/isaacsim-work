"""Compare Blender Principled material graphs with exported USD PreviewSurface graphs."""

import argparse
import json
import re
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--blender", required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("scenes", nargs="+")
args = parser.parse_args()

BLENDER_TO_USD = {
    "Base Color": "diffuseColor",
    "Metallic": "metallic",
    "Roughness": "roughness",
    "IOR": "ior",
    "Alpha": "opacity",
    "Normal": "normal",
    "Coat Weight": "clearcoat",
    "Coat Roughness": "clearcoatRoughness",
    "Emission Color": "emissiveColor",
}
CRITICAL = set(BLENDER_TO_USD)
UNSUPPORTED_INTERMEDIATE = {
    "ShaderNodeMath",
    "ShaderNodeVectorMath",
    "ShaderNodeValToRGB",
    "ShaderNodeMapRange",
    "ShaderNodeInvert",
    "ShaderNodeMix",
    "ShaderNodeMixRGB",
    "ShaderNodeHueSaturation",
    "ShaderNodeRGBCurve",
    "ShaderNodeVectorCurve",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def usd_identifier(name):
    value = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if value and value[0].isdigit():
        value = "_" + value
    return value


def trace_upstream(material, start_node):
    nodes = {node["name"]: node for node in material.get("nodes", [])}
    incoming = {}
    for link in material.get("links", []):
        incoming.setdefault(link["to_node"], []).append(link)
    visited = set()
    result = []

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        node = nodes.get(name)
        if node:
            result.append(node)
        for link in incoming.get(name, []):
            visit(link["from_node"])

    visit(start_node)
    return result


def usd_surface(material):
    for shader in material.get("shaders", []):
        if shader.get("id") == "UsdPreviewSurface":
            return shader
    return None


def usd_input(surface, name):
    if not surface:
        return None
    for item in surface.get("inputs", []):
        if item.get("name") == name:
            return item
    return None


def is_zero_multiply(node):
    if node.get("type") != "ShaderNodeMath" or node.get("operation") != "MULTIPLY":
        return False
    sockets = node.get("input_sockets", [])
    return any(
        not socket.get("linked")
        and isinstance(socket.get("default"), (int, float))
        and float(socket["default"]) == 0.0
        for socket in sockets[:2]
    )


all_results = []
for scene in args.scenes:
    blend_rows = load(Path(args.blender) / f"{scene}.json")
    usd_rows = load(Path(args.usd) / f"{scene}.json")
    usd_by_name = {row["material"]: row for row in usd_rows}
    issues = []
    matched = 0

    for blend_material in blend_rows:
        name = blend_material["material"]
        usd_material = usd_by_name.get(name) or usd_by_name.get(usd_identifier(name))
        if usd_material is None:
            issues.append({"material": name, "kind": "missing_in_usd"})
            continue
        matched += 1
        surface = usd_surface(usd_material)
        principled_nodes = [
            node
            for node in blend_material.get("nodes", [])
            if node.get("type") == "ShaderNodeBsdfPrincipled"
        ]
        if not principled_nodes or surface is None:
            issues.append(
                {
                    "material": name,
                    "kind": "surface_model_mismatch",
                    "blender_principled": bool(principled_nodes),
                    "usd_preview_surface": surface is not None,
                }
            )
            continue

        for principled in principled_nodes:
            incoming = [
                link
                for link in blend_material.get("links", [])
                if link["to_node"] == principled["name"] and link["to_socket"] in CRITICAL
            ]
            incoming_by_socket = {link["to_socket"]: link for link in incoming}
            for blender_socket, usd_name in BLENDER_TO_USD.items():
                link = incoming_by_socket.get(blender_socket)
                target = usd_input(surface, usd_name)
                usd_connected = bool(target and target.get("connections"))
                if link:
                    chain = trace_upstream(blend_material, link["from_node"])
                    unsupported = [
                        {
                            "name": node["name"],
                            "type": node["type"],
                            "operation": node.get("operation"),
                            "zero_multiply": is_zero_multiply(node),
                        }
                        for node in chain
                        if node.get("type") in UNSUPPORTED_INTERMEDIATE
                    ]
                    if unsupported:
                        issues.append(
                            {
                                "material": name,
                                "kind": "intermediate_nodes_lost_or_baked",
                                "blender_socket": blender_socket,
                                "usd_input": usd_name,
                                "blender_source": link,
                                "unsupported_nodes": unsupported,
                                "usd_value": None if target is None else target.get("value"),
                                "usd_connections": [] if target is None else target.get("connections", []),
                            }
                        )
                    elif not usd_connected:
                        issues.append(
                            {
                                "material": name,
                                "kind": "linked_input_missing_in_usd",
                                "blender_socket": blender_socket,
                                "usd_input": usd_name,
                                "blender_source": link,
                                "usd_value": None if target is None else target.get("value"),
                            }
                        )
                elif usd_connected:
                    issues.append(
                        {
                            "material": name,
                            "kind": "unexpected_usd_connection",
                            "blender_socket": blender_socket,
                            "usd_input": usd_name,
                            "blender_default": principled.get("inputs", {}).get(blender_socket),
                            "usd_connections": target.get("connections", []),
                        }
                    )

            for unsupported_socket in (
                "Transmission Weight",
                "Subsurface Weight",
                "Anisotropic IOR Level",
                "Sheen Weight",
            ):
                default = principled.get("inputs", {}).get(unsupported_socket)
                linked = any(
                    link["to_node"] == principled["name"] and link["to_socket"] == unsupported_socket
                    for link in blend_material.get("links", [])
                )
                if linked or (isinstance(default, (int, float)) and abs(float(default)) > 1e-6):
                    issues.append(
                        {
                            "material": name,
                            "kind": "unsupported_principled_feature",
                            "blender_socket": unsupported_socket,
                            "linked": linked,
                            "default": default,
                        }
                    )

    result = {
        "scene": scene,
        "blender_materials": len(blend_rows),
        "usd_materials": len(usd_rows),
        "matched_materials": matched,
        "issue_count": len(issues),
        "issues": issues,
    }
    all_results.append(result)
    print(
        f"[compare] scene={scene} blend={len(blend_rows)} usd={len(usd_rows)} "
        f"matched={matched} issues={len(issues)}"
    )

summary = {
    "valid": all(result["issue_count"] == 0 for result in all_results),
    "scene_count": len(all_results),
    "total_issues": sum(result["issue_count"] for result in all_results),
    "results": all_results,
}
Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
