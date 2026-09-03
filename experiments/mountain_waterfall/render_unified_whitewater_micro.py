"""Render a low-load A/B gate for a unified white PhysX Diffuse look.

The left basin uses equal opaque white points.  The right basin uses the same
unclassified PhysX Diffuse particles with only camera-facing rendering cues:
local-density radius, a softer white material, and no foam/spray/bubble labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-step", type=int, default=120)
    parser.add_argument("--physics-stride", type=int, default=8)
    parser.add_argument("--preview-frame", type=int, default=0)
    return parser.parse_args(argv)


ARGS = parse_args()
SAMPLE_DIRECTORY = ARGS.cache / "native_diffuse_samples"
FRAME_FILES = [
    path
    for path in sorted(SAMPLE_DIRECTORY.glob("diffuse_native_*.npz"))
    if int(path.stem.rsplit("_", 1)[1]) % ARGS.physics_stride == 0
    and int(path.stem.rsplit("_", 1)[1]) <= ARGS.maximum_step
]
if not FRAME_FILES:
    raise RuntimeError(f"No regular native Diffuse frames found in {SAMPLE_DIRECTORY}")

ARGS.output.mkdir(parents=True, exist_ok=True)
FRAME_DIRECTORY = ARGS.output / "frames"
FRAME_DIRECTORY.mkdir(exist_ok=True)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        # Keep datablocks that Blender still considers in use; the factory scene
        # normally leaves none after deleting its objects.
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def principled_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    metallic: float = 0.0,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if "IOR" in shader.inputs:
        shader.inputs["IOR"].default_value = 1.333
    if "Emission Color" in shader.inputs:
        shader.inputs["Emission Color"].default_value = color
    elif "Emission" in shader.inputs:
        shader.inputs["Emission"].default_value = color
    if "Emission Strength" in shader.inputs:
        shader.inputs["Emission Strength"].default_value = emission_strength
    return material


def point_instance_group(name: str, material: bpy.types.Material) -> bpy.types.NodeTree:
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = group.nodes
    links = group.links

    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    named_radius = nodes.new("GeometryNodeInputNamedAttribute")
    named_radius.data_type = "FLOAT"
    named_radius.inputs["Name"].default_value = "radius"
    combine = nodes.new("ShaderNodeCombineXYZ")
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = 1
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    instance = nodes.new("GeometryNodeInstanceOnPoints")

    links.new(group_input.outputs["Geometry"], instance.inputs["Points"])
    links.new(named_radius.outputs["Attribute"], combine.inputs["X"])
    links.new(named_radius.outputs["Attribute"], combine.inputs["Y"])
    links.new(named_radius.outputs["Attribute"], combine.inputs["Z"])
    links.new(combine.outputs["Vector"], instance.inputs["Scale"])
    links.new(ico.outputs["Mesh"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], group_output.inputs["Geometry"])
    return group


def physx_to_blender(points: np.ndarray, x_offset: float) -> np.ndarray:
    converted = np.empty_like(points, dtype=np.float32)
    converted[:, 0] = points[:, 0] + x_offset
    converted[:, 1] = -points[:, 2]
    converted[:, 2] = points[:, 1]
    return converted


def local_density_radius(points: np.ndarray) -> np.ndarray:
    """Return a restrained radius driven only by local Diffuse concentration."""
    cell_width = 0.055
    cells = np.floor(points / cell_width).astype(np.int32)
    occupancy: dict[tuple[int, int, int], int] = {}
    for cell in map(tuple, cells):
        occupancy[cell] = occupancy.get(cell, 0) + 1

    counts = np.empty(len(points), dtype=np.float32)
    offsets = tuple(
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    )
    for index, cell in enumerate(cells):
        cx, cy, cz = map(int, cell)
        counts[index] = sum(
            occupancy.get((cx + dx, cy + dy, cz + dz), 0)
            for dx, dy, dz in offsets
        )

    low, high = np.percentile(counts, (10.0, 90.0))
    normalized = np.clip((counts - low) / max(float(high - low), 1.0), 0.0, 1.0)
    return np.asarray(0.0048 + 0.0052 * np.sqrt(normalized), dtype=np.float32)


def make_point_object(
    name: str,
    points: np.ndarray,
    radii: np.ndarray,
    node_group: bpy.types.NodeTree,
    visible_frame: int,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()
    radius_attribute = mesh.attributes.new("radius", "FLOAT", "POINT")
    radius_attribute.data.foreach_set("value", np.asarray(radii, dtype=np.float32))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    modifier = obj.modifiers.new("PointInstances", "NODES")
    modifier.node_group = node_group
    driver = obj.driver_add("hide_render").driver
    driver.type = "SCRIPTED"
    driver.expression = f"frame != {visible_frame}"
    return obj


def add_box(name: str, location: tuple[float, float, float], scale: tuple[float, float, float], material) -> None:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)


def add_basin(prefix: str, x_offset: float, material) -> None:
    add_box(prefix + "Bottom", (x_offset, 0.0, -0.08), (0.55, 0.48, 0.05), material)
    add_box(prefix + "Left", (x_offset - 0.52, 0.0, 0.18), (0.03, 0.48, 0.31), material)
    add_box(prefix + "Right", (x_offset + 0.52, 0.0, 0.18), (0.03, 0.48, 0.31), material)
    add_box(prefix + "Back", (x_offset, 0.45, 0.18), (0.55, 0.03, 0.31), material)
    # Keep a low front lip for spatial context without hiding the whitewater.
    add_box(prefix + "Front", (x_offset, -0.45, 0.015), (0.55, 0.03, 0.085), material)


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_label(text: str, location: tuple[float, float, float], camera: bpy.types.Object, material) -> None:
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = 0.075
    obj.data.extrude = 0.001
    obj.data.materials.append(material)
    obj.rotation_euler = (camera.location - obj.location).to_track_quat("Z", "Y").to_euler()


clear_scene()
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = len(FRAME_FILES)
scene.render.film_transparent = False
scene.render.filepath = str(FRAME_DIRECTORY / "frame_")
scene.world.color = (0.012, 0.018, 0.026)
scene.view_settings.look = "AgX - Medium High Contrast"

water_material = principled_material("PrimaryWater", (0.025, 0.16, 0.22, 1.0), 0.22, metallic=0.05)
diagnostic_material = principled_material("EqualWhite", (0.88, 0.90, 0.91, 1.0), 0.52)
unified_material = principled_material("UnifiedWhitewater", (0.96, 0.975, 0.98, 1.0), 0.28, emission_strength=0.035)
basin_material = principled_material("Basin", (0.035, 0.045, 0.052, 1.0), 0.62, metallic=0.08)
label_material = principled_material("Label", (0.78, 0.84, 0.88, 1.0), 0.4, emission_strength=0.3)

water_group = point_instance_group("PrimaryWaterInstances", water_material)
diagnostic_group = point_instance_group("EqualWhiteInstances", diagnostic_material)
unified_group = point_instance_group("UnifiedWhitewaterInstances", unified_material)

left_offset = -0.68
right_offset = 0.68
add_basin("Left", left_offset, basin_material)
add_basin("Right", right_offset, basin_material)

for frame_index, frame_path in enumerate(FRAME_FILES, start=1):
    with np.load(frame_path) as data:
        primary = np.asarray(data["primary_position_inv_mass"][:, :3], dtype=np.float32)
        diffuse = np.asarray(data["position_lifetime"][:, :3], dtype=np.float32)

    primary_radius = np.full(len(primary), 0.0125, dtype=np.float32)
    equal_radius = np.full(len(diffuse), 0.0080, dtype=np.float32)
    unified_radius = local_density_radius(diffuse)

    make_point_object(
        f"LeftPrimary_{frame_index:04d}",
        physx_to_blender(primary, left_offset),
        primary_radius,
        water_group,
        frame_index,
    )
    make_point_object(
        f"RightPrimary_{frame_index:04d}",
        physx_to_blender(primary, right_offset),
        primary_radius,
        water_group,
        frame_index,
    )
    make_point_object(
        f"EqualDiffuse_{frame_index:04d}",
        physx_to_blender(diffuse, left_offset),
        equal_radius,
        diagnostic_group,
        frame_index,
    )
    make_point_object(
        f"UnifiedDiffuse_{frame_index:04d}",
        physx_to_blender(diffuse, right_offset),
        unified_radius,
        unified_group,
        frame_index,
    )

bpy.ops.object.camera_add(location=(0.0, -4.25, 2.15))
camera = bpy.context.object
camera.data.lens = 54
camera.data.sensor_width = 36
look_at(camera, (0.0, 0.0, 0.25))
scene.camera = camera

add_label("EQUAL WHITE POINTS", (left_offset, -0.52, 1.05), camera, label_material)
add_label("UNIFIED WHITEWATER", (right_offset, -0.52, 1.05), camera, label_material)

bpy.ops.object.light_add(type="AREA", location=(-1.6, -2.2, 3.2))
key = bpy.context.object
key.data.energy = 850
key.data.shape = "DISK"
key.data.size = 3.0
look_at(key, (0.0, 0.0, 0.25))
bpy.ops.object.light_add(type="AREA", location=(2.2, -0.8, 1.8))
fill = bpy.context.object
fill.data.energy = 500
fill.data.color = (0.55, 0.72, 1.0)
fill.data.size = 2.5
look_at(fill, (0.4, 0.0, 0.25))
bpy.ops.object.light_add(type="AREA", location=(0.0, 1.8, 2.5))
rim = bpy.context.object
rim.data.energy = 700
rim.data.color = (0.68, 0.82, 1.0)
rim.data.size = 2.0
look_at(rim, (0.0, 0.0, 0.35))

scene.frame_set(max(1, min(ARGS.preview_frame or 1, scene.frame_end)))
blend_path = ARGS.output / "unified_whitewater_micro.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

manifest = {
    "schema": "unified_whitewater_micro/1",
    "cache": str(ARGS.cache),
    "frame_count": len(FRAME_FILES),
    "fps": scene.render.fps,
    "physics_stride": ARGS.physics_stride,
    "maximum_step": ARGS.maximum_step,
    "left": "equal opaque white points",
    "right": "same unclassified particles; local-density radius and softer white material",
    "classification_used": False,
    "frame_files": [str(path) for path in FRAME_FILES],
}
(ARGS.output / "render_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

if ARGS.preview_frame:
    scene.render.filepath = str(ARGS.output / f"preview_{scene.frame_current:04d}.png")
    bpy.ops.render.render(write_still=True)
else:
    scene.frame_set(scene.frame_start)
    bpy.ops.render.render(animation=True)
