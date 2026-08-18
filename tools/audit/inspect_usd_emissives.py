"""Inspect emissive material networks and the geometry bound to them in a USD stage."""

import argparse
import json
import os
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--all-materials", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade


def serializable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return list(value)
    except TypeError:
        return str(value)


def material_shaders(material):
    shaders = []
    seen = set()
    pending = []
    for output in material.GetOutputs():
        connected, _ = output.GetConnectedSources()
        pending.extend(connected)
    while pending:
        source_info = pending.pop()
        prim = source_info.source.GetPrim()
        path = str(prim.GetPath())
        if path in seen:
            continue
        seen.add(path)
        shader = UsdShade.Shader(prim)
        if shader:
            shaders.append(shader)
            for shader_input in shader.GetInputs():
                connected, _ = shader_input.GetConnectedSources()
                pending.extend(connected)
    return shaders


def mesh_components(mesh):
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
        face = indices[cursor : cursor + count]
        cursor += count
        if not face:
            continue
        valid = [index for index in face if 0 <= index < len(points)]
        used.update(valid)
        for index in valid[1:]:
            union(valid[0], index)

    groups = {}
    for index in used:
        groups.setdefault(find(index), []).append(index)
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(mesh.GetPrim())
    components = []
    for vertex_indices in groups.values():
        world_points = [transform.Transform(points[index]) for index in vertex_indices]
        minimum = [min(float(point[axis]) for point in world_points) for axis in range(3)]
        maximum = [max(float(point[axis]) for point in world_points) for axis in range(3)]
        components.append(
            {
                "min": minimum,
                "max": maximum,
                "center": [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)],
                "size": [maximum[axis] - minimum[axis] for axis in range(3)],
                "vertex_count": len(vertex_indices),
            }
        )
    components.sort(key=lambda row: (row["center"][2], row["center"][0], row["center"][1]))
    return components


def main():
    args = ARGS
    print("[inspect] opening stage", flush=True)
    stage = Usd.Stage.Open(args.usd)
    if not stage:
        raise RuntimeError(f"Could not open USD: {args.usd}")
    print("[inspect] stage opened", flush=True)

    materials = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        material = UsdShade.Material(prim)
        emission_inputs = []
        all_inputs = []
        for shader in material_shaders(material):
            shader_id = shader.GetIdAttr().Get()
            for shader_input in shader.GetInputs():
                name = shader_input.GetBaseName()
                row = {
                    "shader": str(shader.GetPath()),
                    "shader_id": shader_id,
                    "name": name,
                    "value": serializable(shader_input.Get()),
                    "connected": bool(shader_input.HasConnectedSource()),
                    "sources": [
                        {
                            "shader": str(source_info.source.GetPath()),
                            "output": str(source_info.sourceName),
                        }
                        for source_info in shader_input.GetConnectedSources()[0]
                    ],
                }
                all_inputs.append(row)
                lowered = name.lower()
                if "emiss" in lowered or "emission" in lowered:
                    emission_inputs.append(row)
        if emission_inputs or args.all_materials:
            materials[str(prim.GetPath())] = {
                "material": str(prim.GetPath()),
                "emission_inputs": emission_inputs,
                "all_inputs": all_inputs,
                "geometry": [],
            }
    print(f"[inspect] emissive materials={len(materials)}", flush=True)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        key = str(material.GetPath())
        if key not in materials:
            continue
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        row = {
            "prim": str(prim.GetPath()),
            "min": serializable(bounds.GetMin()),
            "max": serializable(bounds.GetMax()),
        }
        if prim.IsA(UsdGeom.Mesh):
            row["components"] = mesh_components(UsdGeom.Mesh(prim))
        materials[key]["geometry"].append(row)
    print("[inspect] geometry bindings resolved", flush=True)

    payload = {"usd": args.usd, "materials": list(materials.values())}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")


try:
    print(f"[inspect] usd={ARGS.usd} json={ARGS.json_path}", flush=True)
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
