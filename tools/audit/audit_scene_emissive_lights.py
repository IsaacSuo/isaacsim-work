"""Audit emissive PreviewSurface inputs, textures, and proxy-light geometry in every static scene."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageStat

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", default=r"Y:\scenes\static_scene_manifest.json")
parser.add_argument("--output", required=True)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade


def texture_metrics(path):
    if not path or not Path(path).is_file():
        return {"valid": False, "path": path}
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.BILINEAR)
        luminance = image.convert("L")
        histogram = luminance.histogram()
        pixels = luminance.width * luminance.height
        return {
            "valid": True,
            "path": path,
            "mean_luma": ImageStat.Stat(luminance).mean[0] / 255.0,
            "max_luma": max(index for index, count in enumerate(histogram) if count) / 255.0,
            "bright_pct": 100.0 * sum(histogram[26:]) / pixels,
            "very_bright_pct": 100.0 * sum(histogram[128:]) / pixels,
        }


def component_bounds(mesh):
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if not points or not counts or not indices:
        return []
    parent = list(range(len(points)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    used = set()
    cursor = 0
    for count in counts:
        face = [index for index in indices[cursor : cursor + count] if 0 <= index < len(points)]
        cursor += count
        if not face:
            continue
        used.update(face)
        for index in face[1:]:
            union(face[0], index)
    groups = {}
    for index in used:
        groups.setdefault(find(index), []).append(index)
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(mesh.GetPrim())
    rows = []
    for vertex_indices in groups.values():
        world = [matrix.Transform(points[index]) for index in vertex_indices]
        minimum = [min(float(point[axis]) for point in world) for axis in range(3)]
        maximum = [max(float(point[axis]) for point in world) for axis in range(3)]
        center = [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)]
        size = [maximum[axis] - minimum[axis] for axis in range(3)]
        horizontal_length = max(size[0], size[2])
        proxy_candidate = size[1] <= 0.03 and min(size[0], size[2]) >= 0.035 and horizontal_length >= 0.75
        rows.append({"center": center, "size": size, "proxy_candidate": proxy_candidate})
    return rows


def emissive_input(material_prim):
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        shader_input = shader.GetInput("emissiveColor")
        if shader_input:
            return shader_input
    return None


try:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = []
    for entry in manifest["results"]:
        if not entry.get("valid"):
            continue
        scene = entry["scene"]
        stage = Usd.Stage.Open(entry["sim_usd"])
        if stage is None:
            raise RuntimeError(f"Could not open {entry['sim_usd']}")
        materials = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Material):
                continue
            shader_input = emissive_input(prim)
            if not shader_input:
                continue
            value = shader_input.Get()
            connected, _ = shader_input.GetConnectedSources()
            row = {
                "material": str(prim.GetPath()),
                "value": [float(value[index]) for index in range(3)] if value is not None else None,
                "texture": None,
                "geometry": [],
            }
            if len(connected) == 1:
                source = UsdShade.Shader(connected[0].source.GetPrim())
                asset = source.GetInput("file").Get() if source else None
                resolved = asset.resolvedPath if asset and getattr(asset, "resolvedPath", None) else None
                row["texture"] = texture_metrics(resolved)
            materials[row["material"]] = row
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            path = str(material.GetPath()) if material else None
            if path in materials:
                materials[path]["geometry"].append(
                    {"prim": str(prim.GetPath()), "components": component_bounds(UsdGeom.Mesh(prim))}
                )
        material_rows = []
        for row in materials.values():
            constant_max = max(row["value"]) if row["value"] else 0.0
            texture_mean = row["texture"].get("mean_luma", 0.0) if row["texture"] else 0.0
            components = [component for geometry in row["geometry"] for component in geometry["components"]]
            candidate_count = sum(component["proxy_candidate"] for component in components)
            row["active_emission"] = constant_max > 1e-4 or texture_mean > 1e-4
            row["component_count"] = len(components)
            row["proxy_candidate_count"] = candidate_count
            material_rows.append(row)
        active = [row for row in material_rows if row["active_emission"]]
        results.append(
            {
                "scene": scene,
                "usd": entry["sim_usd"],
                "emissive_material_count": len(material_rows),
                "active_material_count": len(active),
                "proxy_candidate_count": sum(row["proxy_candidate_count"] for row in active),
                "materials": material_rows,
            }
        )
        print(
            f"[emissive-audit] scene={scene} active={len(active)} "
            f"candidates={sum(row['proxy_candidate_count'] for row in active)}",
            flush=True,
        )
        stage = None
    payload = {"scene_count": len(results), "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
finally:
    app.close()
