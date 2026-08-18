"""Dump material shader graphs from all exported scene USD files in one Isaac Sim process."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenes-root", default=r"Y:\scenes")
parser.add_argument("--output", required=True)
parser.add_argument("scenes", nargs="+")
args = parser.parse_args()

output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdShade


def shader_graph(material_prim):
    shaders = []
    for child in material_prim.GetChildren():
        if not child.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(child)
        inputs = []
        for shader_input in shader.GetInputs():
            value = shader_input.Get()
            connected, _ = shader_input.GetConnectedSources()
            inputs.append(
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
                "name": child.GetName(),
                "path": str(child.GetPath()),
                "id": str(shader.GetIdAttr().Get()),
                "inputs": inputs,
            }
        )
    return shaders


manifest = []
try:
    root = Path(args.scenes_root).resolve()
    for scene in args.scenes:
        usd_path = root / scene / f"{scene}.usdc"
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"Could not open {usd_path}")
        rows = []
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Material):
                rows.append(
                    {
                        "material": prim.GetName(),
                        "path": str(prim.GetPath()),
                        "shaders": shader_graph(prim),
                    }
                )
        scene_output = output / f"{scene}.json"
        scene_output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        manifest.append(
            {
                "scene": scene,
                "usd": str(usd_path),
                "materials": len(rows),
                "output": str(scene_output),
            }
        )
        print(f"[usd-materials] scene={scene} materials={len(rows)}", flush=True)
        stage = None
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
finally:
    app.close()
