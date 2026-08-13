"""Fifth probe: locate wind/force controls on deformable body APIs via string ApplyAPI."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom, UsdPhysics

context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.Xform.Define(stage, "/World")

print("===== UsdPhysics deformable/surface/cloth/wind names")
print([n for n in dir(UsdPhysics) if any(k in n for k in ("Deformable", "deformable", "Surface", "surface", "Cloth", "cloth", "Wind", "wind"))])

apis = [
    "PhysxBaseDeformableBodyAPI",
    "PhysxDeformableBodyAPI",
    "UsdPhysicsDeformableBodyAPI",
    "PhysxDeformableSurfaceAPI",
    "UsdPhysicsSurfaceDeformableMaterialAPI",
]

for name in apis:
    prim = UsdGeom.Xform.Define(stage, "/World/Test_" + name.replace(":", "_")).GetPrim()
    try:
        ok = prim.ApplyAPI(name)
    except Exception as exc:  # noqa: BLE001
        print("APPLY ERR", name, repr(exc))
        continue
    props = sorted(prim.GetPropertyNames())
    hit = [
        p
        for p in props
        if any(
            k in p.lower()
            for k in ("wind", "force", "gravity", "air", "drag", "lift", "pressure", "stretch", "bend", "shear", "thick", "tether", "damp")
        )
    ]
    print("=====", name, "ok=", ok, "total", len(props))
    for p in hit:
        print("  ", p)

app.close()
