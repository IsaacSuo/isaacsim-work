"""Validate configured material override paths against composed scene USD stages."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=r"Y:\isaacsim_work\scene_experiments.json")
parser.add_argument("--scenes-root", default=r"Y:\scenes")
parser.add_argument("--output", required=True)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade


def material_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    return prim if prim and prim.IsValid() and prim.IsA(UsdShade.Material) else None


def preview_input(prim, name):
    if prim is None:
        return None
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(child)
        if shader.GetIdAttr().Get() == "UsdPreviewSurface":
            shader_input = shader.GetInput(name)
            if shader_input:
                return shader_input
    return None


def connected_uv_texture(shader_input):
    if not shader_input:
        return None
    connected, _ = shader_input.GetConnectedSources()
    if len(connected) != 1:
        return None
    shader = UsdShade.Shader(connected[0].source.GetPrim())
    return shader if shader and shader.GetIdAttr().Get() == "UsdUVTexture" else None

try:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.scenes_root)
    results = []
    for scene, experiment in config.items():
        configured_count = sum(
            len(experiment.get(key, []))
            for key in (
                "force_nonmetal_materials",
                "material_input_scales",
                "material_color_scales",
                "material_opacity_thresholds",
                "material_emission_textures",
                "material_emission_scales",
                "material_omniglass",
            )
        )
        if configured_count == 0:
            continue
        stage_path = root / scene / f"{scene}_sim.usda"
        stage = Usd.Stage.Open(str(stage_path))
        rows = []
        bound_materials = set()
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                bound_materials.add(str(material.GetPath()))

        def add(kind, path, valid, detail=None):
            rows.append({"kind": kind, "path": path, "valid": bool(valid), "detail": detail})

        for path in experiment.get("force_nonmetal_materials", []):
            add("force_nonmetal", path, preview_input(material_prim(stage, path), "metallic") is not None)
        for override in experiment.get("material_input_scales", []):
            path = override["material"]
            shader_input = preview_input(material_prim(stage, path), override["input"])
            add("input_scale", path, connected_uv_texture(shader_input) is not None, override["input"])
        for override in experiment.get("material_color_scales", []):
            path = override["material"]
            shader_input = preview_input(material_prim(stage, path), "diffuseColor")
            add("color_scale", path, connected_uv_texture(shader_input) is not None, "diffuseColor")
        for override in experiment.get("material_opacity_thresholds", []):
            path = override["material"]
            shader_input = preview_input(material_prim(stage, path), "opacity")
            add("opacity_threshold", path, connected_uv_texture(shader_input) is not None, "opacity")
        for override in experiment.get("material_emission_scales", []):
            path = override["material"]
            shader_input = preview_input(material_prim(stage, path), "emissiveColor")
            add("emission_scale", path, connected_uv_texture(shader_input) is not None, "emissiveColor")
        for override in experiment.get("material_emission_textures", []):
            path = override["material"]
            texture = Path(override["texture"])
            valid = material_prim(stage, path) is not None and texture.is_file()
            add("emission_texture", path, valid, str(texture))
        for override in experiment.get("material_omniglass", []):
            path = override["material"]
            valid = material_prim(stage, path) is not None and path in bound_materials
            add("omniglass", path, valid, "bound_source_material")
        result = {
            "scene": scene,
            "count": len(rows),
            "valid": all(row["valid"] for row in rows),
            "overrides": rows,
        }
        results.append(result)
        print(f"[override-paths] scene={scene} count={len(rows)} valid={result['valid']}", flush=True)
        stage = None
    payload = {"valid": all(row["valid"] for row in results), "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
finally:
    app.close()
