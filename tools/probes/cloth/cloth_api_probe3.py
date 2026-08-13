"""Third probe: flag-mesh builder, attachment helper, and wind/force schema attributes."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import inspect

import omni.usd
from omni.physx.scripts import deformableUtils
from pxr import PhysxSchema, UsdGeom

for fn_name in ["create_triangle_mesh_square", "create_auto_deformable_attachment"]:
    fn = getattr(deformableUtils, fn_name)
    print("=" * 70)
    print("FUNC", fn_name)
    try:
        print("SIG", inspect.signature(fn))
    except Exception as exc:  # noqa: BLE001
        print("SIG ERR", repr(exc))
    doc = inspect.getdoc(fn)
    if doc:
        print("DOC:", doc)
    try:
        print("SRC:")
        print(inspect.getsource(fn))
    except Exception as exc:  # noqa: BLE001
        print("SRC ERR", repr(exc))

context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Test")

for api in ("PhysxBaseDeformableBodyAPI", "PhysxDeformableBodyAPI", "PhysxSceneAPI"):
    prim_path = "/World/Test" if api != "PhysxSceneAPI" else "/World/PhysxScene"
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    cls = getattr(PhysxSchema, api)
    try:
        cls.Apply(prim)
    except Exception as exc:  # noqa: BLE001
        print(api, "apply err", repr(exc))
    print("=" * 70)
    props = prim.GetPropertyNames()
    print("PROPS", api, "total", len(props))
    for p in sorted(props):
        if any(
            k in p.lower()
            for k in ("wind", "force", "gravity", "lift", "drag", "pressure", "air", "fluid", "external")
        ):
            print("   ", p, prim.GetAttribute(p).GetTypeName())

app.close()
