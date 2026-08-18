"""List authored OmniPBR MDL shader inputs in Isaac Sim."""

import os
import json
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from pxr import UsdGeom, UsdShade

try:
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        bind_selected_prims=False,
        prim_name="ProbeOmniPBR",
    ).do()
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        rows.append(
            {
                "shader": str(prim.GetPath()),
                "inputs": [
                    {
                        "name": shader_input.GetBaseName(),
                        "type": str(shader_input.GetTypeName()),
                        "value": str(shader_input.Get()),
                    }
                    for shader_input in shader.GetInputs()
                ],
            }
        )
    Path(r"Y:\isaacsim_work\output\material_audit\omnipbr_inputs.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
finally:
    app.close()
