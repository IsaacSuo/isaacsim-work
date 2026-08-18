"""Inspect Cycles ray visibility on an imported USD mesh."""

import json
import sys

import bpy


cache = sys.argv[sys.argv.index("--") + 1]
before = set(bpy.data.objects)
bpy.ops.wm.usd_import(
    filepath=cache,
    import_cameras=False,
    import_curves=False,
    import_lights=False,
    import_materials=False,
    import_volumes=False,
)
objects = [obj for obj in bpy.data.objects if obj not in before]
rows = []
for obj in objects:
    rows.append(
        {
            "name": obj.name,
            "type": obj.type,
            "visible_camera": getattr(obj, "visible_camera", None),
            "visible_shadow": getattr(obj, "visible_shadow", None),
            "visible_diffuse": getattr(obj, "visible_diffuse", None),
            "visible_glossy": getattr(obj, "visible_glossy", None),
            "visible_transmission": getattr(obj, "visible_transmission", None),
            "visible_volume_scatter": getattr(obj, "visible_volume_scatter", None),
            "is_shadow_catcher": getattr(obj, "is_shadow_catcher", None),
            "display_type": obj.display_type,
            "hide_render": obj.hide_render,
        }
    )
print("USD_VISIBILITY=" + json.dumps(rows))
