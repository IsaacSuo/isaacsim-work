"""Render audited Plateau cells with physical materials and camera-driven LOD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plateau_directory", type=Path)
    parser.add_argument("marker_directory", type=Path)
    parser.add_argument("pending_proxy_directory", type=Path)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=520)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--source-samples", nargs="+", type=int)
    parser.add_argument("--camera", choices=("closeup", "overview", "custom"), default="closeup")
    parser.add_argument("--camera-eye", nargs=3, type=float)
    parser.add_argument("--camera-target", nargs=3, type=float)
    parser.add_argument("--camera-lens-mm", type=float, default=55.0)
    parser.add_argument("--camera-sensor-width-mm", type=float, default=24.0)
    parser.add_argument("--camera-clip-start-m", type=float, default=0.005)
    parser.add_argument(
        "--surface-index-mode",
        choices=("legacy-30fps", "source-sample"),
        default="legacy-30fps",
        help="Use legacy 4-digit 30 FPS frames or 6-digit source sample IDs.",
    )
    parser.add_argument("--impactor-radius", type=float, default=0.08)
    parser.add_argument("--micro-pixel-radius", type=float, default=0.40)
    parser.add_argument("--hero-pixel-radius", type=float, default=1.35)
    parser.add_argument("--hdri", type=Path, default=Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr"))
    parser.add_argument("--hdri-strength", type=float, default=0.95)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def isaac_to_blender(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (3,):
        return np.asarray((values[0], -values[2], values[1]), dtype=np.float64)
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def set_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    return False


def make_principled(name, colour, roughness, transmission, ior=1.333, coat=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(shader, colour, "Base Color")
    set_input(shader, roughness, "Roughness")
    set_input(shader, transmission, "Transmission Weight", "Transmission")
    set_input(shader, ior, "IOR")
    set_input(shader, coat, "Coat Weight", "Coat")
    set_input(shader, min(roughness, 0.18), "Coat Roughness")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material, shader


def make_film_material(name, mode):
    if mode == "micro":
        material, shader = make_principled(
            name, (0.78, 0.84, 0.75, 1.0), 0.28, 0.18, coat=0.22
        )
    elif mode == "cluster":
        material, shader = make_principled(
            name, (0.88, 0.94, 0.88, 1.0), 0.12, 0.58, coat=0.32
        )
    else:
        material, shader = make_principled(
            name, (0.96, 0.99, 0.97, 1.0), 0.045, 0.88, coat=0.38
        )
    thickness_socket = shader.inputs.get("Thin Film Thickness")
    if thickness_socket is not None:
        attribute = material.node_tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_type = "GEOMETRY"
        attribute.attribute_name = "ww_thickness_nm"
        material.node_tree.links.new(attribute.outputs["Fac"], thickness_socket)
        set_input(shader, 1.333, "Thin Film IOR")
    wetness = material.node_tree.nodes.new("ShaderNodeAttribute")
    wetness.attribute_type = "GEOMETRY"
    wetness.attribute_name = "ww_wetness"
    mapping = material.node_tree.nodes.new("ShaderNodeMapRange")
    mapping.clamp = True
    mapping.inputs["From Min"].default_value = 0.0
    mapping.inputs["From Max"].default_value = 1.0
    mapping.inputs["To Min"].default_value = 0.24 if mode == "micro" else 0.13
    mapping.inputs["To Max"].default_value = 0.10 if mode == "micro" else 0.025
    material.node_tree.links.new(wetness.outputs["Fac"], mapping.inputs["Value"])
    material.node_tree.links.new(mapping.outputs["Result"], shader.inputs["Roughness"])
    return material


def make_gas_boundary_material():
    material = bpy.data.materials.new("WWV6_GasBoundary")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (0.97, 0.995, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.018
    glass.inputs["IOR"].default_value = 1.333
    material.node_tree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_materials():
    micro = make_film_material("WWV6_MicroFilm_PhysicalArea", "micro")
    cluster = make_film_material("WWV6_ClusterFilm", "cluster")
    hero = make_film_material("WWV6_HeroThinFilm", "hero")
    gas = make_gas_boundary_material()
    shared = make_film_material("WWV6_SharedMembrane", "hero")
    border, _ = make_principled(
        "WWV6_PlateauBorderWater", (0.78, 0.90, 0.84, 1.0), 0.055, 0.82, coat=0.28
    )
    node, _ = make_principled(
        "WWV6_PlateauNodeWater", (0.74, 0.88, 0.82, 1.0), 0.075, 0.72, coat=0.25
    )
    water, _ = make_principled(
        "SplashsurfWaterOpenSurface", (0.78, 0.91, 0.95, 1.0), 0.028, 1.0, coat=0.12
    )
    spray, _ = make_principled(
        "WWV6_SprayDropletWater", (0.90, 0.97, 1.0, 1.0), 0.035, 1.0, coat=0.08
    )
    pending, _ = make_principled(
        "WWV6_PendingLiquidProxy", (0.84, 0.95, 1.0, 1.0), 0.04, 1.0, coat=0.10
    )
    impactor, _ = make_principled(
        "WWV6_Impactor", (0.95, 0.18, 0.035, 1.0), 0.24, 0.0, ior=1.45
    )
    return {
        "plateau": (micro, cluster, hero, gas, shared, border, node),
        "water": water,
        "spray": spray,
        "bubble": gas,
        "pending": pending,
        "impactor": impactor,
    }


ICO_VERTICES = np.asarray(
    [
        (-1, 1.61803398875, 0), (1, 1.61803398875, 0),
        (-1, -1.61803398875, 0), (1, -1.61803398875, 0),
        (0, -1, 1.61803398875), (0, 1, 1.61803398875),
        (0, -1, -1.61803398875), (0, 1, -1.61803398875),
        (1.61803398875, 0, -1), (1.61803398875, 0, 1),
        (-1.61803398875, 0, -1), (-1.61803398875, 0, 1),
    ],
    dtype=np.float64,
)
ICO_VERTICES /= np.linalg.norm(ICO_VERTICES[0])
ICO_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


def link_mesh(name, vertices, faces, material, collection, reverse=False):
    mesh = bpy.data.meshes.new(name + "Mesh")
    if reverse:
        faces = [tuple(reversed(face)) for face in faces]
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def marker_mesh(name, records, material, collection, gas=False):
    if not len(records):
        return None
    vertices, faces = [], []
    for record in records:
        center = record["position"].astype(np.float64)
        radius = float(record["physical_radius"])
        shape = record["shape"].astype(np.float64)
        local = center + ICO_VERTICES * (radius * shape)[None, :]
        local = isaac_to_blender(local)
        offset = len(vertices)
        vertices.extend(map(tuple, local))
        if gas:
            faces.extend(tuple(offset + value for value in reversed(face)) for face in ICO_FACES)
        else:
            faces.extend(tuple(offset + value for value in face) for face in ICO_FACES)
    return link_mesh(name, vertices, faces, material, collection)


def pending_mesh(records, material, collection):
    if not len(records):
        return None
    vertices, faces = [], []
    for record in records:
        direction = record["orientation"].astype(np.float64)
        direction /= max(np.linalg.norm(direction), 1.0e-12)
        reference = np.array((0.0, 1.0, 0.0))
        if abs(float(np.dot(reference, direction))) > 0.90:
            reference = np.array((1.0, 0.0, 0.0))
        second = np.cross(direction, reference)
        second /= max(np.linalg.norm(second), 1.0e-12)
        third = np.cross(direction, second)
        axes = record["shape_semiaxes"].astype(np.float64)
        local = (
            record["position"].astype(np.float64)
            + ICO_VERTICES[:, 0, None] * axes[0] * direction
            + ICO_VERTICES[:, 1, None] * axes[1] * second
            + ICO_VERTICES[:, 2, None] * axes[2] * third
        )
        local = isaac_to_blender(local)
        offset = len(vertices)
        vertices.extend(map(tuple, local))
        faces.extend(tuple(offset + value for value in face) for face in ICO_FACES)
    return link_mesh("WWV6_PendingLiquid", vertices, faces, material, collection)


def projected_lod(cells, camera_position, resolution, lens, sensor_width, micro_px, hero_px):
    centers = isaac_to_blender(cells["position"])
    distance = np.linalg.norm(centers - camera_position[None, :], axis=1)
    focal_pixels = resolution * lens / sensor_width
    pixels = focal_pixels * cells["physical_radius"].astype(np.float64) / np.maximum(distance, 1.0e-6)
    lod = np.where(pixels < micro_px, 0, np.where(pixels < hero_px, 1, 2)).astype(np.uint8)
    return lod, pixels


def micro_density_mesh(cells, lod, material, collection, bin_size=0.006, sides=16):
    """Replace sub-pixel cap points by equal-area local optical sheets.

    This is a render LOD only.  Every patch area is derived from the exact
    power-cell footprint, and the summed polygon area closes to the removed cap
    area.  No alpha, emission, radius floor or visibility enlargement is used.
    """

    selection = np.flatnonzero(lod == 0)
    if not len(selection):
        return None, {
            "micro_density_patches": 0,
            "source_cap_area_m2": 0.0,
            "rendered_cap_area_m2": 0.0,
            "cap_area_residual_m2": 0.0,
        }
    groups = {}
    for index in selection:
        position = cells["position"][index].astype(np.float64)
        key = tuple(np.floor(position[[0, 2]] / bin_size).astype(np.int64))
        groups.setdefault(key, []).append(int(index))
    vertices, faces = [], []
    polygon_factor = 0.5 * sides * math.sin(2.0 * math.pi / sides)
    maximum_patch_area = polygon_factor * (0.45 * bin_size) ** 2
    rendered_area = 0.0
    patch_count = 0
    for key, members in sorted(groups.items()):
        members = np.asarray(members, dtype=np.int64)
        areas = cells["footprint_area"][members].astype(np.float64)
        total_area = float(areas.sum(dtype=np.float64))
        if total_area <= 0.0:
            continue
        weights = areas / total_area
        center = np.sum(
            cells["position"][members].astype(np.float64) * weights[:, None],
            axis=0,
        )
        normal = np.sum(
            cells["normal"][members].astype(np.float64) * weights[:, None],
            axis=0,
        )
        normal /= max(np.linalg.norm(normal), 1.0e-12)
        height = float(
            np.sum(cells["base_height"][members].astype(np.float64) * weights)
        )
        center += height * normal
        reference = np.array((1.0, 0.0, 0.0))
        if abs(float(np.dot(reference, normal))) > 0.85:
            reference = np.array((0.0, 0.0, 1.0))
        tangent = reference - np.dot(reference, normal) * normal
        tangent /= max(np.linalg.norm(tangent), 1.0e-12)
        bitangent = np.cross(normal, tangent)
        pieces = max(1, int(math.ceil(total_area / maximum_patch_area)))
        piece_area = total_area / pieces
        radius = math.sqrt(piece_area / polygon_factor)
        for piece in range(pieces):
            if pieces == 1:
                patch_center = center
            else:
                angle = 2.0 * math.pi * piece / pieces
                offset = min(0.22 * bin_size, 0.55 * radius) * (
                    math.cos(angle) * tangent + math.sin(angle) * bitangent
                )
                patch_center = center + offset
            base = len(vertices)
            vertices.append(tuple(isaac_to_blender(patch_center)))
            for segment in range(sides):
                angle = 2.0 * math.pi * segment / sides
                point = patch_center + radius * (
                    math.cos(angle) * tangent + math.sin(angle) * bitangent
                )
                vertices.append(tuple(isaac_to_blender(point)))
            for segment in range(sides):
                faces.append(
                    (
                        base,
                        base + 1 + segment,
                        base + 1 + (segment + 1) % sides,
                    )
                )
            rendered_area += polygon_factor * radius * radius
            patch_count += 1
    obj = link_mesh(
        "WWV6_MicroDensityEqualArea",
        vertices,
        faces,
        material,
        collection,
    )
    source_area = float(cells["footprint_area"][selection].sum(dtype=np.float64))
    return obj, {
        "micro_density_patches": patch_count,
        "source_cap_area_m2": source_area,
        "rendered_cap_area_m2": rendered_area,
        "cap_area_residual_m2": rendered_area - source_area,
        "bin_size_m": bin_size,
        "polygon_sides": sides,
    }


def plateau_mesh(cache, materials, collection, camera, resolution, micro_px, hero_px):
    cells = np.asarray(cache["cells"])
    films = np.asarray(cache["films"])
    borders = np.asarray(cache["borders"])
    nodes = np.asarray(cache["nodes"])
    vertices = isaac_to_blender(np.asarray(cache["vertices"]))
    triangles = np.asarray(cache["triangles"])
    source_material = np.asarray(cache["triangle_material"])
    owners = np.asarray(cache["triangle_owner_id"])
    lod, projected_pixels = projected_lod(
        cells,
        np.asarray(camera.location, dtype=np.float64),
        resolution,
        camera.data.lens,
        camera.data.sensor_width,
        micro_px,
        hero_px,
    )
    cell_index = {int(value): index for index, value in enumerate(cells["id"])}
    film_by_id = {int(row["id"]): row for row in films}
    border_by_id = {int(row["id"]): row for row in borders}
    node_by_id = {int(row["id"]): row for row in nodes}
    keep = np.ones(len(triangles), dtype=bool)
    material_index = np.empty(len(triangles), dtype=np.int32)
    thickness_nm = np.zeros(len(triangles), dtype=np.float32)
    wetness = np.zeros(len(triangles), dtype=np.float32)
    initial_thickness_nm = 5000.0
    for index, (kind, owner) in enumerate(zip(source_material, owners)):
        kind, owner = int(kind), int(owner)
        if kind == 0:
            row = cells[cell_index[owner]]
            owner_lod = int(lod[cell_index[owner]])
            if owner_lod == 0:
                keep[index] = False
            material_index[index] = owner_lod
            thickness_nm[index] = initial_thickness_nm
            wetness[index] = row["wetness"]
        elif kind == 1:
            row = cells[cell_index[owner]]
            material_index[index] = 3
            wetness[index] = row["wetness"]
        elif kind == 2:
            row = film_by_id[owner]
            material_index[index] = 4
            thickness_nm[index] = float(row["thickness"]) * 1.0e9
            wetness[index] = row["wetness"]
        elif kind == 3:
            row = border_by_id[owner]
            material_index[index] = 5
            wetness[index] = row["wetness"]
        else:
            row = node_by_id[owner]
            material_index[index] = 6
            wetness[index] = row["wetness"]
    triangles = triangles[keep]
    material_index = material_index[keep]
    thickness_nm = thickness_nm[keep]
    wetness = wetness[keep]
    mesh = bpy.data.meshes.new("WWV6_PlateauCellsMesh")
    mesh.from_pydata(vertices, [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new("WWV6_PlateauCells", mesh)
    collection.objects.link(obj)
    for material in materials:
        mesh.materials.append(material)
    for polygon, index in zip(mesh.polygons, material_index):
        polygon.material_index = int(index)
        polygon.use_smooth = index not in (4,)
    thickness_attribute = mesh.attributes.new("ww_thickness_nm", "FLOAT", "FACE")
    wetness_attribute = mesh.attributes.new("ww_wetness", "FLOAT", "FACE")
    thickness_attribute.data.foreach_set("value", thickness_nm)
    wetness_attribute.data.foreach_set("value", wetness)
    micro_obj, micro_report = micro_density_mesh(
        cells, lod, materials[0], collection
    )
    return (obj, micro_obj), {
        "micro_cells": int(np.count_nonzero(lod == 0)),
        "cluster_cells": int(np.count_nonzero(lod == 1)),
        "hero_cells": int(np.count_nonzero(lod == 2)),
        "minimum_projected_radius_px": float(projected_pixels.min(initial=0.0)),
        "maximum_projected_radius_px": float(projected_pixels.max(initial=0.0)),
        "micro_gas_volume_m3": float(cells["gas_volume"][lod == 0].sum(dtype=np.float64)),
        "cluster_gas_volume_m3": float(cells["gas_volume"][lod == 1].sum(dtype=np.float64)),
        "hero_gas_volume_m3": float(cells["gas_volume"][lod == 2].sum(dtype=np.float64)),
        "micro_density": micro_report,
    }


def remove_object(obj):
    if obj is None:
        return
    mesh = obj.data if isinstance(obj.data, bpy.types.Mesh) else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


args = parse_args()
if not (0.0 < args.micro_pixel_radius < args.hero_pixel_radius):
    raise ValueError("Pixel LOD thresholds must be positive and ordered")
if args.camera == "custom" and (args.camera_eye is None or args.camera_target is None):
    raise ValueError("Custom camera requires --camera-eye and --camera-target")
if min(
    args.camera_lens_mm, args.camera_sensor_width_mm,
    args.camera_clip_start_m, args.impactor_radius,
) <= 0.0:
    raise ValueError("Camera optics and impactor radius must be positive")
directories = {
    "plateau": args.plateau_directory.resolve(),
    "markers": args.marker_directory.resolve(),
    "pending": args.pending_proxy_directory.resolve(),
    "source": args.source_directory.resolve(),
    "surface": args.surface_directory.resolve(),
}
manifests = {
    name: json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, directory in directories.items()
    if name not in {"surface"}
}
if not manifests["plateau"].get("complete"):
    raise RuntimeError("Plateau manifest is incomplete")
plateau_samples = manifests["plateau"]["samples"]
selected_sources = (
    set(args.source_samples)
    if args.source_samples
    else set(int(item["source_sample_index"]) for item in plateau_samples)
)
selected = [
    (index, item)
    for index, item in enumerate(plateau_samples)
    if int(item["source_sample_index"]) in selected_sources
]
if set(int(item["source_sample_index"]) for _, item in selected) != selected_sources:
    raise ValueError("One or more requested source samples are absent")
output_directory = args.output_directory.resolve()
output_directory.mkdir(parents=True, exist_ok=True)
manifest_path = output_directory / "render_manifest.json"
if manifest_path.exists():
    raise FileExistsError(f"Refusing to overwrite {manifest_path}")

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.seed = 0
scene.cycles.max_bounces = 12
scene.cycles.transmission_bounces = 12
render_device = "CPU"
try:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type in {"OPTIX", "CUDA"}
    scene.cycles.device = "GPU"
    render_device = "GPU"
except (KeyError, TypeError, RuntimeError):
    scene.cycles.device = "CPU"
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = False
scene.render.fps = round(args.fps)
scene.view_settings.look = "AgX - Medium High Contrast"
world = scene.world or bpy.data.worlds.new("WWV6_PlateauWorld")
scene.world = world
world.use_nodes = True
world.node_tree.nodes.clear()
world_output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
background = world.node_tree.nodes.new("ShaderNodeBackground")
background.inputs["Strength"].default_value = args.hdri_strength
if args.hdri.is_file():
    environment = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(args.hdri.resolve()), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])

camera_data = bpy.data.cameras.new("WWV6_PlateauCamera")
camera = bpy.data.objects.new("WWV6_PlateauCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
if args.camera == "custom":
    target = Vector(isaac_to_blender(args.camera_target))
    eye = Vector(isaac_to_blender(args.camera_eye))
    camera_data.lens = args.camera_lens_mm
elif args.camera == "closeup":
    target = Vector(isaac_to_blender((-0.73, -1.41099, 0.78)))
    eye = Vector(isaac_to_blender((-0.373, -0.335, -0.207)))
    camera_data.lens = 55.0
else:
    target = Vector(isaac_to_blender((-0.73, -1.41099, 0.78)))
    eye = Vector(isaac_to_blender((0.12, 1.15, -1.57)))
    camera_data.lens = 40.0
camera.location = eye
camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
camera_data.sensor_width = args.camera_sensor_width_mm
camera_data.clip_start = args.camera_clip_start_m

materials = create_materials()
collection = bpy.data.collections.new("WWV6_PlateauProduction")
scene.collection.children.link(collection)
bpy.ops.mesh.primitive_uv_sphere_add(
    segments=64, ring_count=32, radius=args.impactor_radius
)
impactor = bpy.context.active_object
impactor.name = "WWV6_Impactor"
impactor.data.materials.append(materials["impactor"])
for polygon in impactor.data.polygons:
    polygon.use_smooth = True

render_manifest = {
    "schema": 1,
    "product": "whitewater_v6_plateau_production_render",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "complete": False,
    "actual_scale": True,
    "visibility_enlargement": False,
    "emission_used": False,
    "arbitrary_opacity_used": False,
    "configuration": {
        "samples": args.samples,
        "resolution": [args.resolution, args.resolution],
        "fps": args.fps,
        "camera": args.camera,
        "camera_eye_isaac_xyz_m": list(args.camera_eye) if args.camera_eye else None,
        "camera_target_isaac_xyz_m": list(args.camera_target) if args.camera_target else None,
        "camera_lens_mm": camera_data.lens,
        "camera_sensor_width_mm": camera_data.sensor_width,
        "camera_clip_start_m": camera_data.clip_start,
        "surface_index_mode": args.surface_index_mode,
        "impactor_radius_m": args.impactor_radius,
        "micro_pixel_radius": args.micro_pixel_radius,
        "hero_pixel_radius": args.hero_pixel_radius,
        "render_device": render_device,
        "lod_policy": "projected physical radius in pixels; geometry remains volume/area conservative",
        "material_policy": "physical transmission, Fresnel/IOR and cache-driven film thickness/wetness; no emission or arbitrary alpha",
    },
    "inputs": {
        name + "_manifest": str(directory / "manifest.json")
        for name, directory in directories.items()
        if name != "surface"
    },
    "frames": [],
}
atomic_json(manifest_path, render_manifest)

dynamic_objects = []
for sequence_index, (sample_index, plateau_sample) in enumerate(selected, start=1):
    for obj in dynamic_objects:
        remove_object(obj)
    dynamic_objects = []
    source_sample = int(plateau_sample["source_sample_index"])
    output_frame = int(plateau_sample["output_frame"])
    if args.surface_index_mode == "source-sample":
        surface_frame = source_sample
        surface_name = f"surface_{surface_frame:06d}_clipped.obj"
    else:
        surface_frame = source_sample // 4
        surface_name = f"surface_{surface_frame:04d}_clipped.obj"
    surface_path = directories["surface"] / surface_name
    if not surface_path.is_file():
        raise FileNotFoundError(surface_path)
    bpy.ops.wm.obj_import(filepath=str(surface_path))
    water = bpy.context.active_object
    water.name = "SplashsurfWaterFrame"
    water.rotation_euler[0] = math.radians(90.0)
    water.data.materials.clear()
    water.data.materials.append(materials["water"])
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    dynamic_objects.append(water)

    plateau_path = directories["plateau"] / plateau_sample["file"]
    with np.load(plateau_path, allow_pickle=False) as plateau_cache:
        plateau_objects, lod_report = plateau_mesh(
            plateau_cache,
            materials["plateau"],
            collection,
            camera,
            args.resolution,
            args.micro_pixel_radius,
            args.hero_pixel_radius,
        )
        cells = np.asarray(plateau_cache["cells"])
        films = np.asarray(plateau_cache["films"])
        borders = np.asarray(plateau_cache["borders"])
        nodes = np.asarray(plateau_cache["nodes"])
    dynamic_objects.extend(obj for obj in plateau_objects if obj is not None)

    marker_sample = manifests["markers"]["samples"][sample_index]
    with np.load(directories["markers"] / marker_sample["file"], allow_pickle=False) as cache:
        markers = np.asarray(cache["markers"])
    spray_rows = markers[markers["state"] == 1]
    bubble_rows = markers[markers["state"] == 3]
    spray_obj = marker_mesh("WWV6_Spray", spray_rows, materials["spray"], collection)
    bubble_obj = marker_mesh("WWV6_EntrainedBubbles", bubble_rows, materials["bubble"], collection, gas=True)
    dynamic_objects.extend(obj for obj in (spray_obj, bubble_obj) if obj is not None)

    pending_sample = manifests["pending"]["samples"][sample_index]
    with np.load(directories["pending"] / pending_sample["file"], allow_pickle=False) as cache:
        proxies = np.asarray(cache["proxies"])
    proxy_obj = pending_mesh(proxies, materials["pending"], collection)
    if proxy_obj is not None:
        dynamic_objects.append(proxy_obj)

    source_entry = manifests["source"]["samples"][source_sample]
    with np.load(directories["source"] / source_entry["file"], allow_pickle=False) as cache:
        sphere_center = np.asarray(cache["sphere_transform"], dtype=np.float64)[3, :3]
    impactor.location = isaac_to_blender(sphere_center)
    output_path = output_directory / f"plateau_{sequence_index:04d}.png"
    report_path = output_path.with_suffix(".json")
    scene.frame_set(sequence_index)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    frame_report = {
        "schema": 1,
        "product": "whitewater_v6_plateau_render_frame",
        "sequence_index": sequence_index,
        "source_sample_index": source_sample,
        "output_frame": output_frame,
        "splashsurf_render_frame": surface_frame,
        "surface_index_mode": args.surface_index_mode,
        "counts": {
            "cells": len(cells), "films": len(films), "borders": len(borders),
            "nodes": len(nodes), "spray": len(spray_rows),
            "entrained_bubbles": len(bubble_rows), "pending_liquid_proxies": len(proxies),
        },
        "lod": lod_report,
        "ledgers": {
            "plateau_gas_volume_m3": float(cells["gas_volume"].sum(dtype=np.float64)),
            "shared_film_area_m2": float(films["area"].sum(dtype=np.float64)),
            "film_liquid_volume_m3": float(films["liquid_volume"].sum(dtype=np.float64)),
            "border_liquid_volume_m3": float(borders["liquid_volume"].sum(dtype=np.float64)),
            "node_liquid_volume_m3": float(nodes["liquid_volume"].sum(dtype=np.float64)),
            "pending_liquid_volume_m3": float(proxies["liquid_volume"].sum(dtype=np.float64)),
        },
        "png": str(output_path),
        "png_sha256": sha256_file(output_path),
        "water_obj": str(surface_path),
        "water_obj_sha256": sha256_file(surface_path),
        "plateau_npz_sha256": sha256_file(plateau_path),
        "actual_scale": True,
        "emission_used": False,
        "arbitrary_opacity_used": False,
    }
    atomic_json(report_path, frame_report)
    render_manifest["frames"].append(frame_report)
    atomic_json(manifest_path, render_manifest)
    print(
        f"[plateau-render] source={source_sample:04d} item={sequence_index}/{len(selected)} "
        f"lod={lod_report['micro_cells']}/{lod_report['cluster_cells']}/{lod_report['hero_cells']}",
        flush=True,
    )

render_manifest["complete"] = True
render_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
atomic_json(manifest_path, render_manifest)
print(f"PLATEAU_RENDER={output_directory} frames={len(selected)}")
