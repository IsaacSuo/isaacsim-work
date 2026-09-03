"""Render non-physical site markers for the selected mountain waterfall layout."""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--lip", nargs=3, type=float, default=(3.08, 1.55, 1.12))
parser.add_argument("--impact", nargs=3, type=float, default=(2.72, 1.55, -0.61))
parser.add_argument("--pool-centre", nargs=3, type=float, default=(2.55, 1.55, -0.60))
parser.add_argument("--pool-radius", type=float, default=0.70)
args = parser.parse_args(argv)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"


def material(name, colour):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*colour, 1.0)
    return value


water = material("Layout_Water", (0.02, 0.55, 1.0))
source_material = material("Layout_Source", (0.0, 1.0, 0.2))
pool_material = material("Layout_Pool", (0.0, 0.75, 0.95))

# Blender XYZ is Z-up.  The lip lies on the high side of the measured cliff.
lip = Vector(args.lip)
impact = Vector(args.impact)

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.16, location=lip)
bpy.context.object.name = "Layout_Water_Source"
bpy.context.object.data.materials.append(source_material)

midpoint = 0.5 * (lip + impact)
direction = impact - lip
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.075, depth=direction.length, location=midpoint)
fall = bpy.context.object
fall.name = "Layout_Waterfall_Centreline"
fall.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
fall.data.materials.append(water)

bpy.ops.mesh.primitive_cylinder_add(
    vertices=96, radius=args.pool_radius, depth=0.025, location=args.pool_centre
)
pool = bpy.context.object
pool.name = "Layout_Catch_Pool"
pool.data.materials.append(pool_material)

bpy.ops.mesh.primitive_cone_add(
    vertices=32, radius1=0.13, radius2=0.0, depth=0.34,
    location=lip + Vector((-0.15, 0.0, 0.0)),
    rotation=(0.0, math.pi / 2.0, 0.0),
)
arrow = bpy.context.object
arrow.name = "Layout_Flow_Direction_Minus_X"
arrow.data.materials.append(source_material)

args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
scene.render.filepath = str(args.output.resolve())
bpy.ops.render.render(write_still=True)
print(f"MOUNTAIN_LAYOUT={args.output.resolve()}")
