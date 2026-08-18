"""Dump authored shader inputs and bound mesh paths from a USD asset."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd")
parser.add_argument("--output", required=True)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade

try:
    stage = Usd.Stage.Open(str(Path(args.usd).resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open {args.usd}")

    bindings = defaultdict(list)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = aligned.GetMin()
            maximum = aligned.GetMax()
            bindings[str(material.GetPath())].append(
                {
                    "path": str(prim.GetPath()),
                    "world_min": [float(value) for value in minimum],
                    "world_max": [float(value) for value in maximum],
                }
            )

    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        inputs = {}
        shaders = []
        for child in prim.GetChildren():
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            shader_inputs = []
            for shader_input in shader.GetInputs():
                value = shader_input.Get()
                if value is not None:
                    inputs[shader_input.GetBaseName()] = str(value)
                connected, _ = shader_input.GetConnectedSources()
                shader_inputs.append(
                    {
                        "name": shader_input.GetBaseName(),
                        "value": None if value is None else str(value),
                        "connections": [
                            {
                                "source": str(info.source.GetPath()),
                                "source_name": str(info.sourceName),
                            }
                            for info in connected
                        ],
                    }
                )
            shaders.append(
                {
                    "path": str(child.GetPath()),
                    "id": str(shader.GetIdAttr().Get()),
                    "inputs": shader_inputs,
                }
            )
        path = str(prim.GetPath())
        rows.append(
            {
                "material": path,
                "inputs": inputs,
                "shaders": shaders,
                "bound_mesh_count": len(bindings[path]),
                "bound_meshes": bindings[path],
            }
        )

    Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"materials={len(rows)} output={args.output}")
finally:
    app.close()
