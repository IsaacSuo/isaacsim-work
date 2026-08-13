"""Sixth probe: full property dump for the real deformable API names (OmniPhysics namespace)."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom

context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.Xform.Define(stage, "/World")

apis = [
    "PhysxBaseDeformableBodyAPI",
    "OmniPhysicsDeformableBodyAPI",
    "OmniPhysicsSurfaceDeformableSimAPI",
    "OmniPhysicsVolumeDeformableSimAPI",
    "PhysxDeformableBodySimAPI",
]

for name in apis:
    prim = UsdGeom.Xform.Define(stage, "/World/T_" + name).GetPrim()
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
