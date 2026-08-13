"""Fourth probe: dump full property names of scene + deformable body APIs to locate wind/force/fluid controls."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from pxr import PhysxSchema, UsdGeom, UsdPhysics

context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.Xform.Define(stage, "/World")


def dump(api, prim_path, scene=False):
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        prim = UsdPhysics.Scene.Define(stage, prim_path).GetPrim() if scene else UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    cls = getattr(PhysxSchema, api)
    try:
        cls.Apply(prim)
    except Exception as exc:  # noqa: BLE001
        print("APPLY ERR", api, repr(exc))
        return
    props = sorted(prim.GetPropertyNames())
    print("=====", api, "total", len(props))
    for p in props:
        print("  ", p)


dump("PhysxSceneAPI", "/World/PhysxScene", scene=True)
dump("PhysxBaseDeformableBodyAPI", "/World/Test")
dump("PhysxDeformableBodyAPI", "/World/Test")

print("===== UsdPhysics wind/fluid/force names")
print([n for n in dir(UsdPhysics) if any(k in n for k in ("Wind", "wind", "Fluid", "fluid", "Force", "force"))])

app.close()
