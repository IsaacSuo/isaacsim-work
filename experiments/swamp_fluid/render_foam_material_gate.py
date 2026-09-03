"""Render an audited foam atlas on one matching Splashsurf surface in Blender."""

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
    parser.add_argument("surface_obj", type=Path)
    parser.add_argument("atlas_directory", type=Path)
    parser.add_argument("source_sample", type=int)
    parser.add_argument("physics_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--cameras", nargs="+", choices=("overview", "macro", "waterline"),
        default=("overview", "macro", "waterline")
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument(
        "--hdri", type=Path,
        default=Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr")
    )
    parser.add_argument("--hdri-strength", type=float, default=0.95)
    parser.add_argument("--diagnostic-atlas", action="store_true")
    parser.add_argument("--diagnostic-normal", action="store_true")
    parser.add_argument("--diagnostic-effective", action="store_true")
    parser.add_argument("--diagnostic-mix-emission", action="store_true")
    parser.add_argument("--diagnostic-cells", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isaac_to_blender(values):
    x, y, z = values
    return Vector((float(x), -float(z), float(y)))


def set_principled_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return socket
    raise RuntimeError(f"Missing Principled input: {names}")


def configure_cycles(scene, samples, resolution):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 240812
    if hasattr(scene.cycles, "use_animated_seed"):
        scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type in {"OPTIX", "CUDA"}
        scene.cycles.device = "GPU"
    except (KeyError, TypeError, RuntimeError):
        scene.cycles.device = "CPU"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"


def configure_world(scene, hdri_path, strength):
    if not hdri_path.is_file():
        raise FileNotFoundError(hdri_path)
    world = scene.world or bpy.data.worlds.new("SwampFoamGateWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])


def float_image(name, rgba):
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("Packed atlas image must have four channels")
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    image.pixels.foreach_set(np.ascontiguousarray(rgba, dtype=np.float32).reshape(-1))
    image.update()
    image.pack()
    return image


def math_node(nodes, operation, label=None):
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    if label:
        node.label = label
    return node


def map_range(nodes, links, value, source_min, source_max, target_min, target_max, label):
    node = nodes.new("ShaderNodeMapRange")
    node.label = label
    node.clamp = True
    if hasattr(node, "interpolation_type"):
        node.interpolation_type = "SMOOTHERSTEP"
    node.inputs["From Min"].default_value = source_min
    node.inputs["From Max"].default_value = source_max
    node.inputs["To Min"].default_value = target_min
    node.inputs["To Max"].default_value = target_max
    links.new(value, node.inputs["Value"])
    return node.outputs["Result"]


def make_atlas_material(atlas_rgba, orientation_rgba, x_values, z_values, diagnostic_atlas=False, diagnostic_normal=False, diagnostic_effective=False, diagnostic_mix_emission=False, diagnostic_cells=False):
    material = bpy.data.materials.new("AuditedWhitewaterSurfaceLOD")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")

    atlas_image = float_image("FoamAtlasPacked", atlas_rgba)
    orientation_image = float_image("FoamOrientationPacked", orientation_rgba)
    texture_coordinate = nodes.new("ShaderNodeTexCoord")
    local_position = nodes.new("ShaderNodeSeparateXYZ")
    links.new(texture_coordinate.outputs["Object"], local_position.inputs["Vector"])
    normalize_x = map_range(
        nodes, links, local_position.outputs["X"],
        float(x_values[0]), float(x_values[-1]), 0.0, 1.0, "Atlas U"
    )
    normalize_z = map_range(
        nodes, links, local_position.outputs["Z"],
        float(z_values[0]), float(z_values[-1]), 0.0, 1.0, "Atlas V"
    )
    atlas_vector = nodes.new("ShaderNodeCombineXYZ")
    links.new(normalize_x, atlas_vector.inputs["X"])
    links.new(normalize_z, atlas_vector.inputs["Y"])

    atlas_texture = nodes.new("ShaderNodeTexImage")
    atlas_texture.label = "R foam coverage | G age | B bubble coverage | A bubble radius"
    atlas_texture.image = atlas_image
    atlas_texture.interpolation = "Linear"
    atlas_texture.extension = "CLIP"
    links.new(atlas_vector.outputs["Vector"], atlas_texture.inputs["Vector"])
    atlas_channels = nodes.new("ShaderNodeSeparateColor")
    links.new(atlas_texture.outputs["Color"], atlas_channels.inputs["Color"])
    foam_coverage = atlas_channels.outputs["Red"]
    foam_age = atlas_channels.outputs["Green"]
    bubble_coverage = atlas_channels.outputs["Blue"]
    bubble_radius = atlas_texture.outputs["Alpha"]

    if diagnostic_atlas:
        diagnostic_color = nodes.new("ShaderNodeCombineColor")
        diagnostic_color.mode = "RGB"
        links.new(foam_coverage, diagnostic_color.inputs["Red"])
        links.new(bubble_coverage, diagnostic_color.inputs["Blue"])
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 3.0
        links.new(diagnostic_color.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material

    orientation_texture = nodes.new("ShaderNodeTexImage")
    orientation_texture.label = "R xx | G xz+0.5 | B zz"
    orientation_texture.image = orientation_image
    orientation_texture.interpolation = "Linear"
    orientation_texture.extension = "CLIP"
    links.new(atlas_vector.outputs["Vector"], orientation_texture.inputs["Vector"])
    orientation = nodes.new("ShaderNodeSeparateColor")
    links.new(orientation_texture.outputs["Color"], orientation.inputs["Color"])
    xx = orientation.outputs["Red"]
    zz = orientation.outputs["Blue"]
    xz_decode = math_node(nodes, "SUBTRACT", "Decode xz")
    xz_decode.inputs[1].default_value = 0.5
    links.new(orientation.outputs["Green"], xz_decode.inputs[0])
    two_xz = math_node(nodes, "MULTIPLY")
    two_xz.inputs[1].default_value = 2.0
    links.new(xz_decode.outputs[0], two_xz.inputs[0])
    tensor_difference = math_node(nodes, "SUBTRACT")
    links.new(xx, tensor_difference.inputs[0])
    links.new(zz, tensor_difference.inputs[1])
    theta = math_node(nodes, "ARCTAN2", "Principal flow angle")
    links.new(two_xz.outputs[0], theta.inputs[0])
    links.new(tensor_difference.outputs[0], theta.inputs[1])
    half_theta = math_node(nodes, "MULTIPLY")
    half_theta.inputs[1].default_value = 0.5
    links.new(theta.outputs[0], half_theta.inputs[0])
    cosine = math_node(nodes, "COSINE")
    sine = math_node(nodes, "SINE")
    links.new(half_theta.outputs[0], cosine.inputs[0])
    links.new(half_theta.outputs[0], sine.inputs[0])

    x_cos = math_node(nodes, "MULTIPLY")
    z_sin = math_node(nodes, "MULTIPLY")
    links.new(local_position.outputs["X"], x_cos.inputs[0])
    links.new(cosine.outputs[0], x_cos.inputs[1])
    links.new(local_position.outputs["Z"], z_sin.inputs[0])
    links.new(sine.outputs[0], z_sin.inputs[1])
    rotated_x = math_node(nodes, "ADD")
    links.new(x_cos.outputs[0], rotated_x.inputs[0])
    links.new(z_sin.outputs[0], rotated_x.inputs[1])
    minus_x_sin = math_node(nodes, "MULTIPLY")
    minus_x_sin.inputs[1].default_value = -1.0
    links.new(local_position.outputs["X"], minus_x_sin.inputs[0])
    minus_x_sin2 = math_node(nodes, "MULTIPLY")
    links.new(minus_x_sin.outputs[0], minus_x_sin2.inputs[0])
    links.new(sine.outputs[0], minus_x_sin2.inputs[1])
    z_cos = math_node(nodes, "MULTIPLY")
    links.new(local_position.outputs["Z"], z_cos.inputs[0])
    links.new(cosine.outputs[0], z_cos.inputs[1])
    rotated_z = math_node(nodes, "ADD")
    links.new(minus_x_sin2.outputs[0], rotated_z.inputs[0])
    links.new(z_cos.outputs[0], rotated_z.inputs[1])

    diff_square = math_node(nodes, "MULTIPLY")
    links.new(tensor_difference.outputs[0], diff_square.inputs[0])
    links.new(tensor_difference.outputs[0], diff_square.inputs[1])
    xz_square = math_node(nodes, "MULTIPLY")
    links.new(two_xz.outputs[0], xz_square.inputs[0])
    links.new(two_xz.outputs[0], xz_square.inputs[1])
    anisotropy_sum = math_node(nodes, "ADD")
    links.new(diff_square.outputs[0], anisotropy_sum.inputs[0])
    links.new(xz_square.outputs[0], anisotropy_sum.inputs[1])
    anisotropy_strength = math_node(nodes, "SQRT")
    links.new(anisotropy_sum.outputs[0], anisotropy_strength.inputs[0])
    anisotropy_scale = math_node(nodes, "MULTIPLY_ADD")
    anisotropy_scale.inputs[1].default_value = 1.5
    anisotropy_scale.inputs[2].default_value = 1.0
    links.new(anisotropy_strength.outputs[0], anisotropy_scale.inputs[0])
    stretched_z = math_node(nodes, "MULTIPLY")
    links.new(rotated_z.outputs[0], stretched_z.inputs[0])
    links.new(anisotropy_scale.outputs[0], stretched_z.inputs[1])
    cell_vector = nodes.new("ShaderNodeCombineXYZ")
    links.new(rotated_x.outputs[0], cell_vector.inputs["X"])
    links.new(stretched_z.outputs[0], cell_vector.inputs["Y"])

    bubble_scale = map_range(
        nodes, links, bubble_radius, 0.0, 1.0, 430.0, 140.0, "Bubble cell frequency"
    )
    inverse_bubble = math_node(nodes, "SUBTRACT")
    inverse_bubble.inputs[0].default_value = 1.0
    links.new(bubble_coverage, inverse_bubble.inputs[1])
    foam_frequency = math_node(nodes, "MULTIPLY")
    foam_frequency.inputs[1].default_value = 230.0
    links.new(inverse_bubble.outputs[0], foam_frequency.inputs[0])
    bubble_frequency = math_node(nodes, "MULTIPLY")
    links.new(bubble_scale, bubble_frequency.inputs[0])
    links.new(bubble_coverage, bubble_frequency.inputs[1])
    primary_frequency = math_node(nodes, "ADD")
    links.new(foam_frequency.outputs[0], primary_frequency.inputs[0])
    links.new(bubble_frequency.outputs[0], primary_frequency.inputs[1])

    voronoi_coarse = nodes.new("ShaderNodeTexVoronoi")
    voronoi_coarse.voronoi_dimensions = "2D"
    voronoi_coarse.feature = "DISTANCE_TO_EDGE"
    links.new(cell_vector.outputs["Vector"], voronoi_coarse.inputs["Vector"])
    links.new(primary_frequency.outputs[0], voronoi_coarse.inputs["Scale"])
    fine_frequency = math_node(nodes, "MULTIPLY")
    fine_frequency.inputs[1].default_value = 2.35
    links.new(primary_frequency.outputs[0], fine_frequency.inputs[0])
    voronoi_fine = nodes.new("ShaderNodeTexVoronoi")
    voronoi_fine.voronoi_dimensions = "2D"
    voronoi_fine.feature = "DISTANCE_TO_EDGE"
    links.new(cell_vector.outputs["Vector"], voronoi_fine.inputs["Vector"])
    links.new(fine_frequency.outputs[0], voronoi_fine.inputs["Scale"])
    coarse_wall = map_range(
        nodes, links, voronoi_coarse.outputs["Distance"], 0.012, 0.075, 1.0, 0.0, "Coarse film walls"
    )
    fine_wall = map_range(
        nodes, links, voronoi_fine.outputs["Distance"], 0.010, 0.050, 0.72, 0.0, "Fine film walls"
    )
    cell_walls = math_node(nodes, "MAXIMUM")
    links.new(coarse_wall, cell_walls.inputs[0])
    links.new(fine_wall, cell_walls.inputs[1])

    bubble_contribution = math_node(nodes, "MULTIPLY")
    bubble_contribution.inputs[1].default_value = 0.45
    links.new(bubble_coverage, bubble_contribution.inputs[0])
    density = math_node(nodes, "MAXIMUM", "Independent foam/bubble density")
    links.new(foam_coverage, density.inputs[0])
    links.new(bubble_contribution.outputs[0], density.inputs[1])
    density_gate = map_range(nodes, links, density.outputs[0], 0.008, 0.34, 0.0, 1.0, "Sparse coverage gate")
    dense_region = map_range(nodes, links, density.outputs[0], 0.32, 0.74, 0.0, 1.0, "Continuous film gate")
    inverse_walls = math_node(nodes, "SUBTRACT")
    inverse_walls.inputs[0].default_value = 1.0
    links.new(cell_walls.outputs[0], inverse_walls.inputs[1])
    dense_fill_control = nodes.new("ShaderNodeValue")
    dense_fill_control.name = "FoamLODDenseFill"
    dense_fill_control.label = "Camera LOD optically dense cell-interior fill"
    dense_fill_control.outputs[0].default_value = 0.50
    dense_region_lod = math_node(nodes, "MULTIPLY")
    links.new(dense_region, dense_region_lod.inputs[0])
    links.new(dense_fill_control.outputs[0], dense_region_lod.inputs[1])
    fill_between_walls = math_node(nodes, "MULTIPLY")
    links.new(dense_region_lod.outputs[0], fill_between_walls.inputs[0])
    links.new(inverse_walls.outputs[0], fill_between_walls.inputs[1])
    structure = math_node(nodes, "ADD")
    links.new(cell_walls.outputs[0], structure.inputs[0])
    links.new(fill_between_walls.outputs[0], structure.inputs[1])
    lod_control = nodes.new("ShaderNodeValue")
    lod_control.name = "FoamLODContinuity"
    lod_control.label = "Camera LOD continuous-film contribution"
    lod_control.outputs[0].default_value = 0.30
    inverse_structure = math_node(nodes, "SUBTRACT")
    inverse_structure.inputs[0].default_value = 1.0
    links.new(structure.outputs[0], inverse_structure.inputs[1])
    lod_fill = math_node(nodes, "MULTIPLY")
    links.new(lod_control.outputs[0], lod_fill.inputs[0])
    links.new(inverse_structure.outputs[0], lod_fill.inputs[1])
    lod_structure = math_node(nodes, "ADD", "LOD blend: cells to continuous film")
    links.new(structure.outputs[0], lod_structure.inputs[0])
    links.new(lod_fill.outputs[0], lod_structure.inputs[1])
    if diagnostic_cells:
        cell_color = nodes.new("ShaderNodeCombineColor")
        cell_color.mode = "RGB"
        links.new(cell_walls.outputs[0], cell_color.inputs["Red"])
        links.new(cell_walls.outputs[0], cell_color.inputs["Green"])
        links.new(cell_walls.outputs[0], cell_color.inputs["Blue"])
        emission = nodes.new("ShaderNodeEmission")
        links.new(cell_color.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material

    geometry = nodes.new("ShaderNodeNewGeometry")
    normal = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Normal"], normal.inputs["Vector"])
    absolute_normal_z = math_node(nodes, "ABSOLUTE", "Open-sheet normal orientation independent")
    links.new(normal.outputs["Z"], absolute_normal_z.inputs[0])
    normal_hemisphere_sign = math_node(nodes, "SIGN", "Orient open-sheet normal upward")
    links.new(normal.outputs["Z"], normal_hemisphere_sign.inputs[0])
    upward_normal = nodes.new("ShaderNodeVectorMath")
    upward_normal.operation = "SCALE"
    upward_normal.label = "Upward shading normal without changing OBJ"
    links.new(geometry.outputs["Normal"], upward_normal.inputs[0])
    links.new(normal_hemisphere_sign.outputs[0], upward_normal.inputs[3])
    top_surface = map_range(
        nodes, links, absolute_normal_z.outputs[0], 0.28, 0.78, 0.0, 1.0, "Near-horizontal surface only"
    )
    if diagnostic_normal:
        normal_color = nodes.new("ShaderNodeCombineColor")
        normal_color.mode = "RGB"
        links.new(absolute_normal_z.outputs[0], normal_color.inputs["Red"])
        links.new(absolute_normal_z.outputs[0], normal_color.inputs["Green"])
        links.new(absolute_normal_z.outputs[0], normal_color.inputs["Blue"])
        emission = nodes.new("ShaderNodeEmission")
        links.new(normal_color.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material
    coverage_structure = math_node(nodes, "MULTIPLY")
    links.new(density_gate, coverage_structure.inputs[0])
    links.new(lod_structure.outputs[0], coverage_structure.inputs[1])
    effective_coverage = math_node(nodes, "MULTIPLY", "Final foam closure weight")
    links.new(coverage_structure.outputs[0], effective_coverage.inputs[0])
    links.new(top_surface, effective_coverage.inputs[1])
    if diagnostic_effective:
        effective_color = nodes.new("ShaderNodeCombineColor")
        effective_color.mode = "RGB"
        links.new(effective_coverage.outputs[0], effective_color.inputs["Red"])
        links.new(effective_coverage.outputs[0], effective_color.inputs["Green"])
        links.new(effective_coverage.outputs[0], effective_color.inputs["Blue"])
        emission = nodes.new("ShaderNodeEmission")
        links.new(effective_color.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material

    water = nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(water, (0.50, 0.72, 0.66, 1.0), "Base Color")
    set_principled_input(water, 1.333, "IOR")
    set_principled_input(water, 1.0, "Transmission Weight", "Transmission")
    water_roughness = math_node(nodes, "MULTIPLY_ADD")
    water_roughness.inputs[1].default_value = 0.11
    water_roughness.inputs[2].default_value = 0.025
    links.new(bubble_coverage, water_roughness.inputs[0])
    links.new(water_roughness.outputs[0], water.inputs["Roughness"])

    foam = nodes.new("ShaderNodeBsdfDiffuse")
    age_color = nodes.new("ShaderNodeMixRGB")
    age_color.blend_type = "MIX"
    age_color.inputs[1].default_value = (0.86, 0.91, 0.80, 1.0)
    age_color.inputs[2].default_value = (0.66, 0.57, 0.39, 1.0)
    links.new(foam_age, age_color.inputs[0])
    links.new(age_color.outputs["Color"], foam.inputs["Color"])
    foam_roughness = math_node(nodes, "MULTIPLY_ADD")
    foam_roughness.inputs[1].default_value = 0.30
    foam_roughness.inputs[2].default_value = 0.28
    links.new(foam_age, foam_roughness.inputs[0])
    links.new(foam_roughness.outputs[0], foam.inputs["Roughness"])
    bump_height = math_node(nodes, "MULTIPLY")
    links.new(cell_walls.outputs[0], bump_height.inputs[0])
    links.new(density_gate, bump_height.inputs[1])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.0012
    links.new(bump_height.outputs[0], bump.inputs["Height"])
    links.new(upward_normal.outputs["Vector"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], foam.inputs["Normal"])

    translucent = nodes.new("ShaderNodeBsdfTranslucent")
    links.new(age_color.outputs["Color"], translucent.inputs["Color"])
    links.new(bump.outputs["Normal"], translucent.inputs["Normal"])
    foam_scatter = nodes.new("ShaderNodeMixShader")
    foam_scatter.label = "Diffuse multiple scattering plus backlit transmission"
    foam_scatter.inputs[0].default_value = 0.28
    links.new(foam.outputs["BSDF"], foam_scatter.inputs[1])
    links.new(translucent.outputs["BSDF"], foam_scatter.inputs[2])

    mix = nodes.new("ShaderNodeMixShader")
    foam_output = foam_scatter.outputs["Shader"]
    if diagnostic_mix_emission:
        mix_emission = nodes.new("ShaderNodeEmission")
        mix_emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        mix_emission.inputs["Strength"].default_value = 2.0
        foam_output = mix_emission.outputs["Emission"]
    links.new(effective_coverage.outputs[0], mix.inputs[0])
    links.new(water.outputs["BSDF"], mix.inputs[1])
    links.new(foam_output, mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


args = parse_args()
surface_path = args.surface_obj.resolve()
atlas_directory = args.atlas_directory.resolve()
physics_report_path = args.physics_report.resolve()
output_directory = args.output_directory.resolve()
if output_directory.exists() and any(output_directory.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
output_directory.mkdir(parents=True, exist_ok=True)

atlas_manifest_path = atlas_directory / "manifest.json"
atlas_audit_path = atlas_directory / "audit_report.json"
atlas_manifest = json.loads(atlas_manifest_path.read_text(encoding="utf-8"))
atlas_audit = json.loads(atlas_audit_path.read_text(encoding="utf-8"))
if atlas_audit.get("valid") is not True or atlas_audit.get("manifest_sha256") != sha256_file(atlas_manifest_path):
    raise ValueError("Foam atlas is not covered by a current valid audit")
item_by_sample = {int(item["source_sample_index"]): item for item in atlas_manifest["samples"]}
if args.source_sample not in item_by_sample:
    raise ValueError("Requested source sample is absent from the foam atlas")
atlas_item = item_by_sample[args.source_sample]
atlas_path = atlas_directory / atlas_item["file"]
if sha256_file(atlas_path) != atlas_item["sha256"]:
    raise ValueError("Foam atlas frame hash mismatch")
coordinates_path = atlas_directory / atlas_manifest["coordinates"]["file"]
if sha256_file(coordinates_path) != atlas_manifest["coordinates"]["sha256"]:
    raise ValueError("Foam atlas coordinate hash mismatch")
with np.load(coordinates_path, allow_pickle=False) as coordinates:
    x_values = np.asarray(coordinates["x_values"], dtype=np.float64)
    z_values = np.asarray(coordinates["z_values"], dtype=np.float64)
with np.load(atlas_path, allow_pickle=False) as atlas:
    foam_coverage = np.asarray(atlas["foam_coverage"], dtype=np.float32)
    foam_age = np.asarray(atlas["foam_mean_age"], dtype=np.float32)
    bubble_coverage = np.asarray(atlas["bubble_coverage"], dtype=np.float32)
    bubble_radius = np.asarray(atlas["bubble_mean_radius"], dtype=np.float32)
    orientation_xx = np.asarray(atlas["foam_orientation_xx"], dtype=np.float32)
    orientation_xz = np.asarray(atlas["foam_orientation_xz"], dtype=np.float32)
    orientation_zz = np.asarray(atlas["foam_orientation_zz"], dtype=np.float32)

packed = np.stack(
    (
        foam_coverage.T,
        np.clip(foam_age.T / 1.2, 0.0, 1.0),
        bubble_coverage.T,
        np.clip(bubble_radius.T / 0.003, 0.0, 1.0),
    ), axis=-1
).astype(np.float32)
packed_orientation = np.stack(
    (
        orientation_xx.T,
        np.clip(orientation_xz.T + 0.5, 0.0, 1.0),
        orientation_zz.T,
        np.ones_like(orientation_xx.T),
    ), axis=-1
).astype(np.float32)

scene = bpy.context.scene
configure_cycles(scene, args.samples, args.resolution)
configure_world(scene, args.hdri.resolve(), args.hdri_strength)
bpy.ops.wm.obj_import(filepath=str(surface_path))
water = bpy.context.active_object
water.name = "SplashsurfWaterFoamGate"
water.rotation_euler[0] = math.radians(90.0)
water.data.materials.clear()
water.data.materials.append(
    make_atlas_material(packed, packed_orientation, x_values, z_values, args.diagnostic_atlas, args.diagnostic_normal, args.diagnostic_effective, args.diagnostic_mix_emission, args.diagnostic_cells)
)
for polygon in water.data.polygons:
    polygon.use_smooth = True

physics_report = json.loads(physics_report_path.read_text(encoding="utf-8"))
render_frame = args.source_sample // atlas_manifest["configuration"]["sample_stride"]
state_directory = Path(atlas_manifest["source"]["state_directory"])
state_manifest = json.loads((state_directory / "manifest.json").read_text(encoding="utf-8"))
source_directory = Path(state_manifest["source"]["directory"])
source_manifest_path = source_directory / "manifest.json"
source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
source_item = next(
    item for item in source_manifest["samples"]
    if int(item["sample_index"]) == args.source_sample
)
source_path = source_directory / source_item["file"]
if sha256_file(source_path) != source_item["sha256"]:
    raise ValueError("Synchronized source frame hash mismatch")
with np.load(source_path, allow_pickle=False) as source_cache:
    sphere_transform = np.asarray(source_cache["sphere_transform"], dtype=np.float64)
if sphere_transform.shape != (4, 4) or not np.isfinite(sphere_transform).all():
    raise ValueError("Invalid synchronized sphere transform")
sphere_center = sphere_transform[3, :3]
bpy.ops.mesh.primitive_uv_sphere_add(
    segments=64, ring_count=32, radius=float(physics_report["impactor"]["radius"]),
    location=isaac_to_blender(sphere_center)
)
sphere = bpy.context.active_object
sphere.name = "PhysXImpactorFoamGate"
sphere_material = bpy.data.materials.new("PhysXImpactorOrange")
sphere_material.use_nodes = True
sphere_nodes = sphere_material.node_tree.nodes
sphere_links = sphere_material.node_tree.links
sphere_nodes.clear()
sphere_output = sphere_nodes.new("ShaderNodeOutputMaterial")
sphere_shader = sphere_nodes.new("ShaderNodeBsdfPrincipled")
set_principled_input(sphere_shader, (0.95, 0.18, 0.035, 1.0), "Base Color")
set_principled_input(sphere_shader, 0.24, "Roughness")
sphere_links.new(sphere_shader.outputs["BSDF"], sphere_output.inputs["Surface"])
sphere.data.materials.append(sphere_material)
for polygon in sphere.data.polygons:
    polygon.use_smooth = True

camera_settings = {
    "overview": {"location": (0.12, 1.15, -1.57), "target": (-0.73, -1.405, 0.78), "lens": 40.0},
    "macro": {"location": (-0.42, -1.10, 0.52), "target": (-0.76, -1.405, 1.00), "lens": 72.0},
    "waterline": {"location": (-0.02, -1.335, -0.05), "target": (-0.73, -1.392, 0.78), "lens": 52.0},
}
camera_data = bpy.data.cameras.new("FoamMaterialGateCamera")
camera = bpy.data.objects.new("FoamMaterialGateCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera

outputs = []
camera_lod_continuity = {
    "overview": 0.78,
    "macro": 0.08,
    "waterline": 0.35,
}
camera_lod_dense_fill = {
    "overview": 1.00,
    "macro": 0.08,
    "waterline": 0.45,
}
for camera_name in args.cameras:
    lod_node = water.active_material.node_tree.nodes.get("FoamLODContinuity")
    if lod_node is not None:
        lod_node.outputs[0].default_value = camera_lod_continuity[camera_name]
    dense_fill_node = water.active_material.node_tree.nodes.get("FoamLODDenseFill")
    if dense_fill_node is not None:
        dense_fill_node.outputs[0].default_value = camera_lod_dense_fill[camera_name]
    settings = camera_settings[camera_name]
    camera.location = isaac_to_blender(settings["location"])
    target = isaac_to_blender(settings["target"])
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = settings["lens"]
    output_path = output_directory / f"foam_gate_sample_{args.source_sample:03d}_{camera_name}.png"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    outputs.append({"camera": camera_name, "file": output_path.name, "bytes": output_path.stat().st_size, "sha256": sha256_file(output_path)})
    print(f"FOAM_GATE_RENDER={output_path}")

manifest = {
    "schema": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sample": args.source_sample,
    "render_frame": render_frame,
    "inputs": {
        "surface_obj": str(surface_path), "surface_sha256": sha256_file(surface_path),
        "atlas_directory": str(atlas_directory), "atlas_manifest_sha256": sha256_file(atlas_manifest_path),
        "atlas_audit_sha256": sha256_file(atlas_audit_path),
        "physics_report": str(physics_report_path), "physics_report_sha256": sha256_file(physics_report_path),
        "synchronized_source_frame": str(source_path),
        "synchronized_source_frame_sha256": sha256_file(source_path),
        "hdri": str(args.hdri.resolve()), "hdri_sha256": sha256_file(args.hdri.resolve()),
    },
    "render": {"engine": "CYCLES", "samples": args.samples, "resolution": args.resolution, "cameras": list(args.cameras), "diagnostic_atlas": args.diagnostic_atlas, "diagnostic_normal": args.diagnostic_normal, "diagnostic_effective": args.diagnostic_effective, "diagnostic_mix_emission": args.diagnostic_mix_emission, "diagnostic_cells": args.diagnostic_cells},
    "material": {
        "representation": "audited optical-depth atlas plus two-scale anisotropic Voronoi film",
        "atlas_channels": ["foam_coverage", "foam_mean_age", "bubble_coverage", "bubble_mean_radius"],
        "orientation_tensor": ["xx", "xz", "zz"],
        "surface_normal_gate": "absolute world normal dot up",
        "surface_normal_gate_range": [0.28, 0.78],
        "foam_shading_normal": "geometry normal oriented to the upward hemisphere before bump",
        "foam_frequency_per_metre": 230.0,
        "bubble_frequency_per_metre": [430.0, 140.0],
        "secondary_frequency_multiplier": 2.35,
        "continuous_film_coverage_range": [0.32, 0.74],
        "camera_lod_continuity": camera_lod_continuity,
        "camera_lod_dense_fill": camera_lod_dense_fill,
        "foam_birth_color": [0.86, 0.91, 0.80, 1.0],
        "foam_mature_color": [0.66, 0.57, 0.39, 1.0],
    },
    "outputs": outputs,
}
(output_directory / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
