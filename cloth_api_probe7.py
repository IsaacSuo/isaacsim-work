"""Seventh probe: full property dump of the correct surface-deformable API names to hunt wind/external-force."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom

context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.Xform.Define(stage, "/World")

apis = [
    "PhysxSurfaceDeformableBodyAPI",
    "PhysxSurfaceDeformableMaterialAPI",
    "OmniPhysicsSurfaceDeformableSimAPI",
    "OmniPhysicsDeformableBodyAPI",
    "OmniPhysicsSurfaceDeformableMaterialAPI",
]

for name in apis:
    prim = UsdGeom.Xform.Define(stage, "/World/T_" + name.replace(":", "_")).GetPrim()
    try:
        ok = prim.ApplyAPI(name)
    except Exception as exc:  # noqa: BLE001
        print("APPLY ERR", name, repr(exc))
        continue
    props = sorted(prim.GetPropertyNames())
    print("=====", name, "ok=", ok, "total", len(props))
    for p in props:
        print("  ", p)

app.close()
