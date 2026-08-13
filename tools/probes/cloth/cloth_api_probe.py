"""Probe Isaac Sim 6.0 cloth (surface deformable) API surface without guessing."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import inspect

import omni.usd
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import PhysxSchema, UsdGeom

print("=== deformableUtils public names ===")
print(sorted(n for n in dir(deformableUtils) if not n.startswith("_")))

print("=== physicsUtils public names ===")
print(sorted(n for n in dir(physicsUtils) if not n.startswith("_")))

print("=== add_deformable_material signature ===")
try:
    print(inspect.signature(deformableUtils.add_deformable_material))
except Exception as exc:  # noqa: BLE001
    print("sig err", repr(exc))

print("=== PhysxSchema names (Deformable/Cloth/Soft/Wind/Fabric/Surface) ===")
candidates = [
    n
    for n in sorted(dir(PhysxSchema))
    if any(k in n for k in ("Deformable", "Cloth", "Soft", "Wind", "Fabric", "Surface"))
]
print(candidates)

print("=== candidate API property dump ===")
context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.Xform.Define(stage, "/World")
for name in candidates:
    if not name.endswith("API"):
        continue
    cls = getattr(PhysxSchema, name)
    prim = UsdGeom.Xform.Define(stage, "/World/Test_" + name).GetPrim()
    try:
        applied = cls.Apply(prim) if hasattr(cls, "Apply") else None
        props = [
            p
            for p in prim.GetPropertyNames()
            if any(
                k in p.lower()
                for k in (
                    "stretch",
                    "bend",
                    "shear",
                    "damp",
                    "wind",
                    "tether",
                    "cloth",
                    "mass",
                    "self",
                    "ccd",
                    "thick",
                    "drag",
                    "lift",
                    "pressure",
                    "youngs",
                    "poisson",
                    "aniso",
                    "stiff",
                    "iteration",
                    "gravity",
                )
            )
        ]
        print("---", name, "applied=", applied)
        for p in sorted(props):
            attr = prim.GetAttribute(p)
            print("   ", p, attr.GetTypeName())
    except Exception as exc:  # noqa: BLE001
        print("---", name, "ERR", repr(exc))

app.close()
