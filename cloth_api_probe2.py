"""Second probe: exact signatures, docstrings, and source for the surface deformable (cloth) helpers."""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import inspect

from omni.physx.scripts import deformableUtils

for fn_name in [
    "create_auto_surface_deformable_hierarchy",
    "set_physics_surface_deformable_body",
    "add_surface_deformable_material",
    "add_deformable_material",
    "create_auto_deformable_attachment",
    "add_auto_deformable_attachment",
    "create_triangle_mesh_square",
    "triangulate_mesh",
]:
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

print("=" * 70)
print("SOURCE add_surface_deformable_material:")
try:
    print(inspect.getsource(deformableUtils.add_surface_deformable_material))
except Exception as exc:  # noqa: BLE001
    print("SRC ERR", repr(exc))

print("=" * 70)
print("SOURCE set_physics_surface_deformable_body:")
try:
    print(inspect.getsource(deformableUtils.set_physics_surface_deformable_body))
except Exception as exc:  # noqa: BLE001
    print("SRC ERR", repr(exc))

app.close()
