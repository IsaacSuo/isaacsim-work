"""Sample the authored Swamp terrain into an Isaac-coordinate height field.

Run with ``swamp.blend`` open in Blender.  Bounds are Isaac X/Z coordinates.
"""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) not in {6, 7, 8}:
    raise SystemExit(
        "Expected OUTPUT_NPZ X_MIN X_MAX Z_MIN Z_MAX SPACING "
        "[OBJECT] [MAX_TERRAIN_Y]"
    )

output = Path(argv[0]).resolve()
x_min, x_max, z_min, z_max, spacing = (float(value) for value in argv[1:6])
terrain_name = argv[6] if len(argv) >= 7 else "Object_12"
if len(argv) >= 7:
    terrain_name = argv[6]
maximum_terrain_y = float(argv[7]) if len(argv) == 8 else -1.30
x_values = np.arange(x_min, x_max + 0.5 * spacing, spacing, dtype=np.float64)
z_values = np.arange(z_min, z_max + 0.5 * spacing, spacing, dtype=np.float64)

terrain = bpy.data.objects.get(terrain_name)
if terrain is None or terrain.type != "MESH":
    raise RuntimeError(f"Authored Swamp terrain {terrain_name} is missing")

mesh = terrain.data
mesh.calc_loop_triangles()
world_points = [terrain.matrix_world @ vertex.co for vertex in mesh.vertices]
selected_triangles = []
for loop_triangle in mesh.loop_triangles:
    indices = tuple(loop_triangle.vertices)
    triangle = [world_points[index] for index in indices]
    # Match the PhysX collision builder: only the authored low terrain inside
    # this local basin may define the water floor.  Object_12 also contains
    # unrelated terrain elsewhere and must not be treated as stacked floor.
    if min(point.z for point in triangle) >= maximum_terrain_y:
        continue
    if max(point.x for point in triangle) < x_min or min(point.x for point in triangle) > x_max:
        continue
    blender_y_min, blender_y_max = -z_max, -z_min
    if max(point.y for point in triangle) < blender_y_min or min(point.y for point in triangle) > blender_y_max:
        continue
    selected_triangles.append(indices)
if not selected_triangles:
    raise RuntimeError("No authored low terrain triangles overlap the requested basin")
bvh = BVHTree.FromPolygons(world_points, selected_triangles, all_triangles=True)
heights = np.full((len(x_values), len(z_values)), np.nan, dtype=np.float32)
for ix, x in enumerate(x_values):
    for iz, z in enumerate(z_values):
        # Isaac (x, y, z) maps to Blender (x, -z, y).
        hit, _normal, _face, _distance = bvh.ray_cast(
            Vector((float(x), float(-z), 10.0)), Vector((0.0, 0.0, -1.0)), 30.0
        )
        if hit is not None:
            heights[ix, iz] = hit.z

output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output,
    x_values=x_values.astype(np.float32),
    z_values=z_values.astype(np.float32),
    terrain_y=heights,
    spacing=np.float32(spacing),
    source=np.asarray(str(Path(bpy.data.filepath).resolve())),
    terrain_object=np.asarray(terrain.name),
    maximum_terrain_y=np.float32(maximum_terrain_y),
)
print(
    f"SWAMP_HEIGHTFIELD={output} shape={heights.shape} "
    f"valid={int(np.count_nonzero(np.isfinite(heights)))} "
    f"terrain_triangles={len(selected_triangles)}"
)
