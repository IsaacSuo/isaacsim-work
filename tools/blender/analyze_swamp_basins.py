"""Find connected terrain regions below the former authored water elevation."""

import json

import bpy


WATER_Z = -1.3259903192520142
terrain = bpy.data.objects.get("Object_12")
if terrain is None or terrain.type != "MESH":
    raise RuntimeError("Terrain mesh Object_12 was not found")

mesh = terrain.data
world = terrain.matrix_world
points = [world @ vertex.co for vertex in mesh.vertices]
eligible = {index for index, point in enumerate(points) if point.z < WATER_Z}
neighbors = {index: set() for index in eligible}
for edge in mesh.edges:
    a, b = edge.vertices
    if a in eligible and b in eligible:
        neighbors[a].add(b)
        neighbors[b].add(a)

components = []
remaining = set(eligible)
while remaining:
    seed = remaining.pop()
    component = {seed}
    stack = [seed]
    while stack:
        current = stack.pop()
        additions = neighbors[current] & remaining
        remaining.difference_update(additions)
        component.update(additions)
        stack.extend(additions)
    if len(component) < 3:
        continue
    selected = [points[index] for index in component]
    components.append(
        {
            "vertices": len(component),
            "center_xy": [
                sum(point.x for point in selected) / len(selected),
                sum(point.y for point in selected) / len(selected),
            ],
            "minimum": [min(point[axis] for point in selected) for axis in range(3)],
            "maximum": [max(point[axis] for point in selected) for axis in range(3)],
        }
    )

components.sort(key=lambda item: item["vertices"], reverse=True)
print(json.dumps(components, indent=2))
