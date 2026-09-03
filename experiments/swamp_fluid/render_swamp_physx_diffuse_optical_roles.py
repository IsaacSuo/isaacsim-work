"""Render recovered PhysX Diffuse roles with role-specific optical models.

The recovered labels are not artist heuristics: they reproduce the PhysX 5.9
primary-neighbour thresholds.  This renderer only maps those physical roles to
camera-visible geometry and materials:

* spray  -> velocity-aligned transmissive droplets;
* foam   -> overlapping, soft-edged density splats (not solid spheres);
* bubble -> transparent air cavities with a Fresnel-dominant rim.

The label frame, Splashsurf surface frame and impactor transform are required
to have the same output frame number.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


LABEL_SPRAY = 0
LABEL_FOAM = 1
LABEL_BUBBLE = 2


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_directory", type=Path)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("run_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument(
        "--render-roles",
        nargs="+",
        choices=("spray", "foam", "bubble_shell", "microbubble"),
        default=("spray", "foam", "bubble_shell", "microbubble"),
        help="Optical layers to render; physical label accounting is always preserved in the manifest.",
    )
    parser.add_argument("--water-only", action="store_true")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=520)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--spray-radius", type=float, default=0.00125)
    parser.add_argument("--spray-shutter-fraction", type=float, default=0.42)
    parser.add_argument("--spray-maximum-half-length", type=float, default=0.008)
    parser.add_argument("--foam-radius", type=float, default=0.0080)
    parser.add_argument("--foam-surface-influence-radius", type=float, default=0.012)
    parser.add_argument(
        "--foam-render-mode", choices=("surface_plateau", "splat"), default="surface_plateau"
    )
    parser.add_argument(
        "--material-profile",
        choices=("role_optical", "nvidia_diffuse", "nvidia_screen_space"),
        default="role_optical",
        help="Use role-specific optics or an NVIDIA FleX-style white scattering Diffuse renderer.",
    )
    parser.add_argument("--nvidia-diffuse-scale", type=float, default=0.5)
    parser.add_argument("--nvidia-diffuse-inscatter", type=float, default=0.8)
    parser.add_argument("--nvidia-diffuse-outscatter", type=float, default=0.53)
    parser.add_argument("--screen-extinction-scale", type=float, default=0.65)
    parser.add_argument("--screen-thickness-blur-px", type=float, default=1.45)
    parser.add_argument("--screen-spray-weight", type=float, default=0.38)
    parser.add_argument("--screen-foam-weight", type=float, default=1.0)
    parser.add_argument("--screen-bubble-weight", type=float, default=0.24)
    parser.add_argument("--screen-spray-blur-scale", type=float, default=0.55)
    parser.add_argument("--screen-foam-blur-scale", type=float, default=1.15)
    parser.add_argument("--screen-bubble-blur-scale", type=float, default=1.75)
    parser.add_argument("--screen-foam-broad-blur-scale", type=float, default=2.20)
    parser.add_argument("--screen-foam-broad-fraction", type=float, default=0.30)
    parser.add_argument("--screen-foam-detail-retention", type=float, default=0.55)
    parser.add_argument("--screen-foam-detail-sigma-px", type=float, default=0.85)
    parser.add_argument("--screen-bubble-attenuation-depth-m", type=float, default=0.025)
    parser.add_argument("--screen-water-surface-epsilon-m", type=float, default=0.0015)
    parser.add_argument("--bubble-radius", type=float, default=0.00175)
    parser.add_argument("--camera-eye", nargs=3, type=float, default=(-0.373, -0.335, -0.207))
    parser.add_argument("--camera-target", nargs=3, type=float, default=(-0.73, -1.41099, 0.78))
    parser.add_argument("--camera-lens-mm", type=float, default=55.0)
    parser.add_argument("--camera-sensor-width-mm", type=float, default=24.0)
    parser.add_argument("--camera-clip-start-m", type=float, default=0.005)
    parser.add_argument("--impactor-radius", type=float, default=0.08)
    parser.add_argument(
        "--hdri",
        type=Path,
        default=Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr"),
    )
    parser.add_argument("--hdri-strength", type=float, default=0.95)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def isaac_to_blender(values: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (3,):
        return np.asarray((values[0], -values[2], values[1]), dtype=np.float64)
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def set_input(node, value, *names: str) -> bool:
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
    return material


def make_water_with_surface_foam_material():
    """Water BSDF plus a surface-bound foam film with procedural Plateau borders."""
    material = bpy.data.materials.new("SwampWaterWithPhysXSurfaceFoam")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    water = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(water, (0.74, 0.88, 0.88, 1.0), "Base Color")
    set_input(water, 0.032, "Roughness")
    set_input(water, 1.0, "Transmission Weight", "Transmission")
    set_input(water, 1.333, "IOR")
    set_input(water, 0.10, "Coat Weight", "Coat")
    foam = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(foam, (0.89, 0.925, 0.855, 1.0), "Base Color")
    set_input(foam, 0.50, "Roughness")
    set_input(foam, 0.025, "Transmission Weight", "Transmission")
    set_input(foam, 1.333, "IOR")
    set_input(foam, 0.12, "Coat Weight", "Coat")
    set_input(foam, (0.87, 0.91, 0.84, 1.0), "Emission Color", "Emission")
    set_input(foam, 0.025, "Emission Strength")

    density_attribute = nodes.new("ShaderNodeAttribute")
    density_attribute.attribute_name = "foam_density"
    density_response = nodes.new("ShaderNodeMapRange")
    density_response.clamp = True
    density_response.inputs["From Min"].default_value = 0.12
    density_response.inputs["From Max"].default_value = 0.58
    density_response.inputs["To Min"].default_value = 0.0
    density_response.inputs["To Max"].default_value = 1.0
    geometry = nodes.new("ShaderNodeNewGeometry")
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.voronoi_dimensions = "3D"
    voronoi.feature = "DISTANCE_TO_EDGE"
    voronoi.distance = "EUCLIDEAN"
    voronoi.inputs["Scale"].default_value = 220.0
    border_map = nodes.new("ShaderNodeMapRange")
    border_map.clamp = True
    border_map.inputs["From Min"].default_value = 0.0
    border_map.inputs["From Max"].default_value = 0.065
    border_map.inputs["To Min"].default_value = 1.0
    border_map.inputs["To Max"].default_value = 0.0
    border_floor = nodes.new("ShaderNodeMath")
    border_floor.operation = "MULTIPLY_ADD"
    border_floor.inputs[1].default_value = 0.78
    border_floor.inputs[2].default_value = 0.22
    film_factor = nodes.new("ShaderNodeMath")
    film_factor.operation = "MULTIPLY"
    mix = nodes.new("ShaderNodeMixShader")
    links.new(geometry.outputs["Position"], voronoi.inputs["Vector"])
    links.new(voronoi.outputs["Distance"], border_map.inputs["Value"])
    links.new(border_map.outputs["Result"], border_floor.inputs[0])
    links.new(density_attribute.outputs["Fac"], density_response.inputs["Value"])
    links.new(density_response.outputs["Result"], film_factor.inputs[0])
    links.new(border_floor.outputs[0], film_factor.inputs[1])
    links.new(film_factor.outputs[0], mix.inputs[0])
    links.new(water.outputs["BSDF"], mix.inputs[1])
    links.new(foam.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_spray_material():
    material = bpy.data.materials.new("PhysXDiffuseSprayWaterDroplet")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.92, 0.98, 1.0, 1.0)
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.distribution = "BECKMANN"
    glass.inputs["Color"].default_value = (0.91, 0.975, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.025
    glass.inputs["IOR"].default_value = 1.333
    # Millimetre droplets should not cast the near-black, solid-looking shadows of
    # centimetre glass beads. Preserve refraction/specular at grazing angles and
    # let the centre remain mostly optically clear.
    layer_weight = nodes.new("ShaderNodeLayerWeight")
    layer_weight.inputs["Blend"].default_value = 0.16
    invert = nodes.new("ShaderNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1.0
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY_ADD"
    scale.inputs[1].default_value = 0.52
    scale.inputs[2].default_value = 0.055
    mix = nodes.new("ShaderNodeMixShader")
    links.new(layer_weight.outputs["Facing"], invert.inputs[1])
    links.new(invert.outputs[0], scale.inputs[0])
    links.new(scale.outputs[0], mix.inputs[0])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(glass.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_nvidia_diffuse_material(inscatter: float, outscatter: float):
    """Approximate the NVIDIA FleX Diffuse point renderer in Cycles.

    FleX uses white soft particles, density accumulation, in/out scattering and
    disables Diffuse shadows.  This is deliberately not a glass material.
    """
    material = bpy.data.materials.new("NvidiaStyleDiffuseWhitewater")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.94, 0.97, 0.95, 1.0)
    scatter = nodes.new("ShaderNodeBsdfPrincipled")
    # Inscatter controls the bright, multiply-scattered component. Outscatter
    # reduces contrast without turning each sample into an opaque white bead.
    brightness = float(np.clip(0.90 + 0.10 * inscatter - 0.03 * outscatter, 0.0, 1.0))
    set_input(scatter, (brightness * 0.97, brightness, brightness * 0.96, 1.0), "Base Color")
    set_input(scatter, 0.68, "Roughness")
    set_input(scatter, 0.0, "Transmission Weight", "Transmission")
    set_input(scatter, 0.22, "IOR Level", "Specular IOR Level")
    set_input(scatter, (brightness * 0.94, brightness, brightness * 0.95, 1.0), "Emission Color", "Emission")
    set_input(scatter, 0.06 * inscatter, "Emission Strength")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "diffuse_alpha"
    mix = nodes.new("ShaderNodeMixShader")
    links.new(attribute.outputs["Fac"], mix.inputs[0])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(scatter.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_foam_material():
    """Create a soft density material driven by a point-domain foam_alpha attribute."""
    material = bpy.data.materials.new("PhysXDiffuseFoamDensitySplat")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.84, 0.88, 0.80, 1.0)
    foam = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(foam, (0.88, 0.92, 0.84, 1.0), "Base Color")
    set_input(foam, 0.48, "Roughness")
    set_input(foam, 0.035, "Transmission Weight", "Transmission")
    set_input(foam, 1.333, "IOR")
    set_input(foam, 0.10, "Coat Weight", "Coat")
    set_input(foam, (0.88, 0.92, 0.84, 1.0), "Emission Color", "Emission")
    set_input(foam, 0.035, "Emission Strength")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "foam_alpha"
    mix = nodes.new("ShaderNodeMixShader")
    links.new(attribute.outputs["Fac"], mix.inputs[0])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(foam.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_microbubble_material():
    """Milky multiple-scattering proxy for unresolved, densely packed micro-bubbles."""
    material = bpy.data.materials.new("PhysXDiffuseMicrobubbleDensity")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.86, 0.91, 0.86, 1.0)
    cloud = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(cloud, (0.91, 0.94, 0.88, 1.0), "Base Color")
    set_input(cloud, 0.62, "Roughness")
    set_input(cloud, 0.0, "Transmission Weight", "Transmission")
    set_input(cloud, 1.20, "IOR")
    set_input(cloud, (0.88, 0.93, 0.86, 1.0), "Emission Color", "Emission")
    set_input(cloud, 0.055, "Emission Strength")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "microbubble_alpha"
    mix = nodes.new("ShaderNodeMixShader")
    links.new(attribute.outputs["Fac"], mix.inputs[0])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(cloud.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_bubble_material():
    """Approximate an air cavity: clear centre and reflective/refractive grazing rim."""
    material = bpy.data.materials.new("PhysXDiffuseBubbleAirCavity")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    links = material.node_tree.links
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.82, 0.92, 0.89, 1.0)
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.distribution = "BECKMANN"
    glass.inputs["Color"].default_value = (0.73, 0.88, 0.84, 1.0)
    glass.inputs["Roughness"].default_value = 0.045
    # Air viewed from water has a relative IOR of approximately 1 / 1.333.
    glass.inputs["IOR"].default_value = 0.7502
    layer_weight = nodes.new("ShaderNodeLayerWeight")
    layer_weight.inputs["Blend"].default_value = 0.18
    invert = nodes.new("ShaderNodeMath")
    invert.operation = "SUBTRACT"
    invert.inputs[0].default_value = 1.0
    power = nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.inputs[1].default_value = 1.7
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY_ADD"
    scale.inputs[1].default_value = 0.48
    scale.inputs[2].default_value = 0.018
    mix = nodes.new("ShaderNodeMixShader")
    links.new(layer_weight.outputs["Facing"], invert.inputs[1])
    links.new(invert.outputs[0], power.inputs[0])
    links.new(power.outputs[0], scale.inputs[0])
    links.new(scale.outputs[0], mix.inputs[0])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(glass.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


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


def create_ico_family(name, centers_isaac, basis_u, basis_v, basis_w, radii, half_lengths, material, collection):
    count = len(centers_isaac)
    if count == 0:
        return None
    centers = isaac_to_blender(centers_isaac)
    basis_u = isaac_to_blender(basis_u)
    basis_v = isaac_to_blender(basis_v)
    basis_w = isaac_to_blender(basis_w)
    vertices = np.empty((count, len(ICO_VERTICES), 3), dtype=np.float64)
    for local_index, local in enumerate(ICO_VERTICES):
        vertices[:, local_index, :] = (
            centers
            + basis_u * (local[0] * radii)[:, None]
            + basis_v * (local[1] * radii)[:, None]
            + basis_w * (local[2] * half_lengths)[:, None]
        )
    vertices = vertices.reshape(-1, 3)
    faces = np.asarray(ICO_FACES, dtype=np.int64)[None, :, :] + (
        np.arange(count, dtype=np.int64) * len(ICO_VERTICES)
    )[:, None, None]
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.reshape(-1, 3).tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def orthonormal_frames(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    w = np.zeros_like(vectors, dtype=np.float64)
    moving = norms > 1.0e-8
    w[moving] = vectors[moving] / norms[moving, None]
    w[~moving] = (0.0, 1.0, 0.0)
    helper = np.tile((0.0, 1.0, 0.0), (len(w), 1))
    nearly_parallel = np.abs(w[:, 1]) > 0.92
    helper[nearly_parallel] = (1.0, 0.0, 0.0)
    u = np.cross(w, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1)[:, None], 1.0e-12)
    v = np.cross(w, u)
    return u, v, w


def create_spray(points, velocities, radius, fps, shutter_fraction, maximum_half_length, material, collection):
    if len(points) == 0:
        return None, np.empty(0)
    speed = np.linalg.norm(velocities, axis=1)
    u, v, w = orthonormal_frames(velocities)
    # A shutter-integrated trajectory is represented as a compact velocity-aligned ellipsoid.
    half_length = radius + 0.5 * speed * (shutter_fraction / fps)
    half_length = np.minimum(half_length, maximum_half_length)
    radius_variation = radius * (0.82 + 0.28 * np.minimum(speed / 3.0, 1.0))
    obj = create_ico_family(
        "PhysXDiffuseSpray",
        points,
        u,
        v,
        w,
        radius_variation,
        half_length,
        material,
        collection,
    )
    return obj, half_length


def create_bubbles(points, neighbor_count, base_radius, material, collection):
    if len(points) == 0:
        return None, np.empty(0)
    # Dense micro-bubbles become slightly smaller; deterministic spatial modulation prevents clones.
    phase = np.sin(points[:, 0] * 173.3 + points[:, 1] * 91.7 + points[:, 2] * 137.9)
    density_scale = 1.0 - 0.18 * np.clip((neighbor_count.astype(np.float64) - 8.0) / 8.0, 0.0, 1.0)
    radii = base_radius * density_scale * (0.78 + 0.22 * (phase * 0.5 + 0.5))
    u = np.tile((1.0, 0.0, 0.0), (len(points), 1))
    v = np.tile((0.0, 1.0, 0.0), (len(points), 1))
    w = np.tile((0.0, 0.0, 1.0), (len(points), 1))
    obj = create_ico_family(
        "PhysXDiffuseBubbles",
        points,
        u,
        v,
        w,
        radii,
        radii,
        material,
        collection,
    )
    return obj, radii


def create_foam_splats(points_isaac, neighbor_count, base_radius, camera, material, collection):
    count = len(points_isaac)
    if count == 0:
        return None, np.empty(0), np.empty(0)
    points = isaac_to_blender(points_isaac)
    camera_right = np.asarray(camera.matrix_world.to_quaternion() @ Vector((1.0, 0.0, 0.0)))
    camera_up = np.asarray(camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0)))
    density = np.clip((neighbor_count.astype(np.float64) - 3.0) / 4.0, 0.25, 1.0)
    phase = np.sin(points_isaac[:, 0] * 211.1 + points_isaac[:, 2] * 157.7)
    radii = base_radius * (0.82 + 0.36 * density) * (0.90 + 0.10 * phase)
    centre_alpha = 0.12 + 0.28 * density
    sides = 7
    vertices = np.empty((count * (sides + 1), 3), dtype=np.float64)
    alpha = np.zeros(count * (sides + 1), dtype=np.float32)
    faces = np.empty((count * sides, 3), dtype=np.int64)
    for particle_index in range(count):
        vertex_offset = particle_index * (sides + 1)
        face_offset = particle_index * sides
        vertices[vertex_offset] = points[particle_index]
        alpha[vertex_offset] = centre_alpha[particle_index]
        angular_jitter = phase[particle_index] * 0.23
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides + angular_jitter
            radial_jitter = 0.84 + 0.16 * math.sin((side + 1) * 4.17 + phase[particle_index] * 3.0)
            vertices[vertex_offset + side + 1] = points[particle_index] + radii[particle_index] * radial_jitter * (
                math.cos(angle) * camera_right + math.sin(angle) * camera_up
            )
            next_side = (side + 1) % sides
            faces[face_offset + side] = (
                vertex_offset,
                vertex_offset + side + 1,
                vertex_offset + next_side + 1,
            )
    mesh = bpy.data.meshes.new("PhysXDiffuseFoamMesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    attribute = mesh.attributes.new("foam_alpha", "FLOAT", "POINT")
    attribute.data.foreach_set("value", alpha)
    obj = bpy.data.objects.new("PhysXDiffuseFoamDensity", mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    return obj, radii, centre_alpha


def create_microbubble_splats(points_isaac, neighbor_count, base_radius, camera, material, collection):
    """Represent unresolved dense bubbles as a soft turbidity field, not discrete dark shells."""
    count = len(points_isaac)
    if count == 0:
        return None, np.empty(0), np.empty(0)
    points = isaac_to_blender(points_isaac)
    camera_right = np.asarray(camera.matrix_world.to_quaternion() @ Vector((1.0, 0.0, 0.0)))
    camera_up = np.asarray(camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0)))
    density = np.clip((neighbor_count.astype(np.float64) - 10.0) / 6.0, 1.0 / 6.0, 1.0)
    phase = np.sin(points_isaac[:, 0] * 149.3 + points_isaac[:, 1] * 83.9 + points_isaac[:, 2] * 193.1)
    radii = base_radius * (0.82 + 0.38 * density) * (0.91 + 0.09 * phase)
    centre_alpha = 0.028 + 0.080 * density
    sides = 6
    vertices = np.empty((count * (sides + 1), 3), dtype=np.float64)
    alpha = np.zeros(count * (sides + 1), dtype=np.float32)
    faces = np.empty((count * sides, 3), dtype=np.int64)
    for particle_index in range(count):
        vertex_offset = particle_index * (sides + 1)
        face_offset = particle_index * sides
        vertices[vertex_offset] = points[particle_index]
        alpha[vertex_offset] = centre_alpha[particle_index]
        angular_jitter = phase[particle_index] * 0.19
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides + angular_jitter
            radial_jitter = 0.87 + 0.13 * math.sin((side + 1) * 3.73 + phase[particle_index] * 2.7)
            vertices[vertex_offset + side + 1] = points[particle_index] + radii[particle_index] * radial_jitter * (
                math.cos(angle) * camera_right + math.sin(angle) * camera_up
            )
            next_side = (side + 1) % sides
            faces[face_offset + side] = (
                vertex_offset,
                vertex_offset + side + 1,
                vertex_offset + next_side + 1,
            )
    mesh = bpy.data.meshes.new("PhysXDiffuseMicrobubbleDensityMesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    attribute = mesh.attributes.new("microbubble_alpha", "FLOAT", "POINT")
    attribute.data.foreach_set("value", alpha)
    obj = bpy.data.objects.new("PhysXDiffuseMicrobubbleDensity", mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    return obj, radii, centre_alpha


def create_nvidia_diffuse_splats(
    points_isaac,
    velocities_isaac,
    remaining_lifetime,
    labels,
    neighbor_count,
    neighbor_radius,
    diffuse_scale,
    fps,
    shutter_fraction,
    camera,
    material,
    collection,
):
    """Create camera-facing soft particles following FleX Diffuse conventions."""
    count = len(points_isaac)
    if count == 0:
        return None, {}
    points = isaac_to_blender(points_isaac)
    velocities = isaac_to_blender(velocities_isaac)
    camera_right = np.asarray(camera.matrix_world.to_quaternion() @ Vector((1.0, 0.0, 0.0)))
    camera_up = np.asarray(camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0)))
    projected_x = velocities @ camera_right
    projected_y = velocities @ camera_up
    projected_speed = np.sqrt(projected_x * projected_x + projected_y * projected_y)
    direction_x = np.divide(
        projected_x, projected_speed, out=np.ones_like(projected_x), where=projected_speed > 1.0e-8
    )
    direction_y = np.divide(
        projected_y, projected_speed, out=np.zeros_like(projected_y), where=projected_speed > 1.0e-8
    )
    major_axis = direction_x[:, None] * camera_right + direction_y[:, None] * camera_up
    minor_axis = -direction_y[:, None] * camera_right + direction_x[:, None] * camera_up

    # PhysX classifies roles; the renderer only changes optical footprint.  The
    # FleX g_diffuseScale default is applied to particleContactOffset, which is
    # half of the exported neighbour radius.
    base_radius = 0.5 * float(neighbor_radius) * diffuse_scale
    role_radius = np.choose(labels, (0.58, 1.0, 0.78)).astype(np.float64)
    role_optical_depth = np.choose(labels, (0.15, 0.60, 0.08)).astype(np.float64)
    lifetime_weight = np.power(np.clip(remaining_lifetime, 0.0, 1.0), 0.72)
    density_weight = np.ones(count, dtype=np.float64)
    foam_or_bubble = labels != LABEL_SPRAY
    density_weight[foam_or_bubble] = 0.72 + 0.28 * np.clip(
        neighbor_count[foam_or_bubble].astype(np.float64) / 16.0, 0.0, 1.0
    )
    radii = base_radius * role_radius * (0.72 + 0.28 * lifetime_weight)
    optical_depth = role_optical_depth * lifetime_weight * density_weight
    # Convert optical thickness to opacity.  This is the critical distinction
    # between translucent whitewater and directly drawing opaque white points.
    centre_alpha = 1.0 - np.exp(-optical_depth)
    half_major = radii.copy()
    spray = labels == LABEL_SPRAY
    half_major[spray] += 0.5 * projected_speed[spray] * (shutter_fraction / fps)
    half_major[spray] = np.minimum(half_major[spray], base_radius * 2.8)

    sides = 7
    vertices = np.empty((count * (sides + 1), 3), dtype=np.float64)
    alpha = np.zeros(count * (sides + 1), dtype=np.float32)
    faces = np.empty((count * sides, 3), dtype=np.int64)
    for particle_index in range(count):
        vertex_offset = particle_index * (sides + 1)
        face_offset = particle_index * sides
        vertices[vertex_offset] = points[particle_index]
        alpha[vertex_offset] = centre_alpha[particle_index]
        phase = math.sin(
            points_isaac[particle_index, 0] * 157.1
            + points_isaac[particle_index, 1] * 89.3
            + points_isaac[particle_index, 2] * 211.7
        )
        angular_jitter = 0.16 * phase
        for side in range(sides):
            angle = 2.0 * math.pi * side / sides + angular_jitter
            radial_jitter = 0.90 + 0.10 * math.sin((side + 1) * 4.13 + phase * 2.5)
            vertices[vertex_offset + side + 1] = points[particle_index] + radial_jitter * (
                math.cos(angle) * half_major[particle_index] * major_axis[particle_index]
                + math.sin(angle) * radii[particle_index] * minor_axis[particle_index]
            )
            next_side = (side + 1) % sides
            faces[face_offset + side] = (
                vertex_offset,
                vertex_offset + side + 1,
                vertex_offset + next_side + 1,
            )
    mesh = bpy.data.meshes.new("NvidiaDiffuseSoftParticleMesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    attribute = mesh.attributes.new("diffuse_alpha", "FLOAT", "POINT")
    attribute.data.foreach_set("value", alpha)
    obj = bpy.data.objects.new("NvidiaDiffuseSoftParticles", mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    return obj, {
        "base_radius_m": float(base_radius),
        "radius_range_m": [float(np.min(radii)), float(np.max(radii))],
        "half_major_range_m": [float(np.min(half_major)), float(np.max(half_major))],
        "centre_alpha_range": [float(np.min(centre_alpha)), float(np.max(centre_alpha))],
        "hard_shadow_enabled": bool(getattr(obj, "visible_shadow", False)),
    }


def gaussian_blur_scalar(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return image.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    horizontal_source = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.zeros_like(image, dtype=np.float64)
    for index, weight in enumerate(kernel):
        horizontal += weight * horizontal_source[:, index : index + image.shape[1]]
    vertical_source = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    vertical = np.zeros_like(horizontal)
    for index, weight in enumerate(kernel):
        vertical += weight * vertical_source[index : index + image.shape[0], :]
    return np.asarray(vertical, dtype=np.float32)


def build_world_bvh(obj) -> BVHTree:
    """Build a world-space BVH for deterministic camera-to-water depth queries."""
    bpy.context.view_layer.update()
    world_vertices = [tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices]
    polygons = [tuple(polygon.vertices) for polygon in obj.data.polygons]
    if not world_vertices or not polygons:
        raise ValueError("Cannot build a water-surface BVH from an empty mesh")
    return BVHTree.FromPolygons(world_vertices, polygons, all_triangles=False)


def project_diffuse_optical_depth(
    points_isaac,
    velocities_isaac,
    remaining_lifetime,
    labels,
    neighbor_count,
    neighbor_radius,
    diffuse_scale,
    fps,
    shutter_fraction,
    camera,
    resolution_x,
    resolution_y,
    impactor_center_isaac,
    impactor_radius,
    extinction_scale,
    blur_sigma,
    role_weights,
    role_blur_scales,
    foam_broad_blur_scale,
    foam_broad_fraction,
    water_bvh,
    bubble_attenuation_depth,
    water_surface_epsilon,
):
    """Accumulate independent spray, foam and bubble optical-depth buffers.

    Keeping the PhysX roles separate is important: spray must retain fast,
    sparse structure, surface foam needs a connected mesoscopic coverage
    field, and submerged bubbles should only contribute a broad turbidity
    layer.  Combining them before filtering turns every role into the same
    salt-like screen-space particle.
    """
    points = isaac_to_blender(points_isaac)
    velocities = isaac_to_blender(velocities_isaac)
    camera_matrix_inverse = np.asarray(camera.matrix_world.inverted(), dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    camera_points = homogeneous @ camera_matrix_inverse.T
    camera_velocities = velocities @ camera_matrix_inverse[:3, :3].T
    depth = -camera_points[:, 2]
    focal_pixels = camera.data.lens / camera.data.sensor_width * resolution_x
    pixel_x = resolution_x * 0.5 + focal_pixels * camera_points[:, 0] / np.maximum(depth, 1.0e-8)
    pixel_y = resolution_y * 0.5 + focal_pixels * camera_points[:, 1] / np.maximum(depth, 1.0e-8)

    camera_location = np.asarray(camera.location, dtype=np.float64)
    sphere_center = isaac_to_blender(impactor_center_isaac)
    rays = points - camera_location
    particle_distance = np.linalg.norm(rays, axis=1)
    ray_direction = rays / np.maximum(particle_distance[:, None], 1.0e-12)
    camera_to_sphere = sphere_center - camera_location
    along_ray = ray_direction @ camera_to_sphere
    perpendicular_squared = np.dot(camera_to_sphere, camera_to_sphere) - along_ray * along_ray
    intersects_sphere = perpendicular_squared < impactor_radius * impactor_radius
    front_intersection = along_ray - np.sqrt(
        np.maximum(impactor_radius * impactor_radius - perpendicular_squared, 0.0)
    )
    occluded_by_impactor = (
        intersects_sphere & (front_intersection > 0.0) & (front_intersection < particle_distance)
    )

    base_radius = 0.5 * float(neighbor_radius) * diffuse_scale
    role_radius = np.choose(labels, (0.58, 1.0, 0.78)).astype(np.float64)
    lifetime_weight = np.power(np.clip(remaining_lifetime, 0.0, 1.0), 0.72)
    radii_m = base_radius * role_radius * (0.72 + 0.28 * lifetime_weight)
    radius_px = focal_pixels * radii_m / np.maximum(depth, 1.0e-8)

    velocity_x_px = focal_pixels * camera_velocities[:, 0] / np.maximum(depth, 1.0e-8)
    velocity_y_px = focal_pixels * camera_velocities[:, 1] / np.maximum(depth, 1.0e-8)
    velocity_px = np.sqrt(velocity_x_px * velocity_x_px + velocity_y_px * velocity_y_px)
    direction_x = np.divide(
        velocity_x_px, velocity_px, out=np.ones_like(velocity_px), where=velocity_px > 1.0e-8
    )
    direction_y = np.divide(
        velocity_y_px, velocity_px, out=np.zeros_like(velocity_px), where=velocity_px > 1.0e-8
    )
    half_major_px = radius_px.copy()
    spray = labels == LABEL_SPRAY
    half_major_px[spray] += 0.5 * velocity_px[spray] * (shutter_fraction / fps)
    half_major_px[spray] = np.minimum(half_major_px[spray], radius_px[spray] * 2.8)

    role_weight = np.choose(labels, role_weights).astype(np.float64)
    density_weight = 0.72 + 0.28 * np.clip(neighbor_count.astype(np.float64) / 16.0, 0.0, 1.0)
    sample_weight = extinction_scale * role_weight * lifetime_weight * density_weight
    # A bubble behind the reconstructed free surface must not be pasted onto
    # the front of the image.  Trace the camera ray against the same-frame
    # Splashsurf mesh and apply Beer-Lambert attenuation by submerged path
    # length. Foam and spray keep their original three-dimensional positions.
    water_intersection = np.zeros(len(points), dtype=bool)
    submerged_path_m = np.zeros(len(points), dtype=np.float64)
    bubble_water_transmittance = np.ones(len(points), dtype=np.float64)
    bubble_candidates = np.flatnonzero(
        (labels == LABEL_BUBBLE) & (depth > camera.data.clip_start) & (~occluded_by_impactor)
    )
    ray_origin = Vector(camera_location)
    for particle_index in bubble_candidates:
        hit_location, _hit_normal, _face_index, hit_distance = water_bvh.ray_cast(
            ray_origin,
            Vector(ray_direction[particle_index]),
            float(particle_distance[particle_index]),
        )
        if hit_location is None or hit_distance is None:
            continue
        submerged_distance = float(particle_distance[particle_index] - hit_distance)
        if submerged_distance <= water_surface_epsilon:
            continue
        water_intersection[particle_index] = True
        submerged_path_m[particle_index] = submerged_distance
        bubble_water_transmittance[particle_index] = math.exp(
            -submerged_distance / bubble_attenuation_depth
        )
    sample_weight *= bubble_water_transmittance
    valid = (
        (depth > camera.data.clip_start)
        & (~occluded_by_impactor)
        & (pixel_x + half_major_px >= 0.0)
        & (pixel_x - half_major_px < resolution_x)
        & (pixel_y + half_major_px >= 0.0)
        & (pixel_y - half_major_px < resolution_y)
    )
    role_optical_depth = np.zeros((3, resolution_y, resolution_x), dtype=np.float32)
    valid_indices = np.flatnonzero(valid)
    for particle_index in valid_indices:
        major = max(float(half_major_px[particle_index]), 0.55)
        minor = max(float(radius_px[particle_index]), 0.55)
        extent = int(math.ceil(max(major, minor)))
        centre_x = float(pixel_x[particle_index])
        centre_y = float(pixel_y[particle_index])
        x_min = max(0, int(math.floor(centre_x - extent)))
        x_max = min(resolution_x - 1, int(math.ceil(centre_x + extent)))
        y_min = max(0, int(math.floor(centre_y - extent)))
        y_max = min(resolution_y - 1, int(math.ceil(centre_y + extent)))
        if x_min > x_max or y_min > y_max:
            continue
        grid_x = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5 - centre_x
        grid_y = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5 - centre_y
        delta_x, delta_y = np.meshgrid(grid_x, grid_y)
        cosine = float(direction_x[particle_index])
        sine = float(direction_y[particle_index])
        local_major = cosine * delta_x + sine * delta_y
        local_minor = -sine * delta_x + cosine * delta_y
        normalized_squared = (local_major / major) ** 2 + (local_minor / minor) ** 2
        inside = normalized_squared < 1.0
        if not np.any(inside):
            continue
        kernel = np.zeros_like(normalized_squared, dtype=np.float32)
        # Compact Gaussian-like kernel: zero at the splat boundary, smooth in the centre.
        kernel[inside] = np.exp(-2.25 * normalized_squared[inside]).astype(np.float32)
        role_index = int(labels[particle_index])
        role_optical_depth[role_index, y_min : y_max + 1, x_min : x_max + 1] += (
            float(sample_weight[particle_index]) * kernel
        )
    for role_index, blur_scale in enumerate(role_blur_scales):
        fine_depth = gaussian_blur_scalar(
            role_optical_depth[role_index], blur_sigma * float(blur_scale)
        )
        if role_index == LABEL_FOAM and foam_broad_fraction > 0.0:
            broad_depth = gaussian_blur_scalar(
                role_optical_depth[role_index], blur_sigma * float(foam_broad_blur_scale)
            )
            role_optical_depth[role_index] = (
                fine_depth * (1.0 - foam_broad_fraction)
                + broad_depth * foam_broad_fraction
            )
        else:
            role_optical_depth[role_index] = fine_depth
    optical_depth = np.sum(role_optical_depth, axis=0)
    positive = optical_depth[optical_depth > 1.0e-8]
    role_names = ("spray", "foam", "bubble")
    role_stats = {}
    for role_index, role_name in enumerate(role_names):
        role_depth = role_optical_depth[role_index]
        role_positive = role_depth[role_depth > 1.0e-8]
        role_stats[role_name] = {
            "weight": float(role_weights[role_index]),
            "blur_sigma_px": float(blur_sigma * role_blur_scales[role_index]),
            "optical_depth_maximum": float(np.max(role_depth)),
            "optical_depth_p95_positive": (
                float(np.quantile(role_positive, 0.95)) if len(role_positive) else 0.0
            ),
            "covered_pixels": int(len(role_positive)),
        }
    stats = {
        "input_samples": int(len(points)),
        "projected_samples": int(len(valid_indices)),
        "impactor_occluded_samples": int(np.count_nonzero(occluded_by_impactor)),
        "bubble_samples_behind_water_surface": int(np.count_nonzero(water_intersection)),
        "bubble_submerged_path_mean_m": (
            float(np.mean(submerged_path_m[water_intersection]))
            if np.any(water_intersection)
            else 0.0
        ),
        "bubble_submerged_path_maximum_m": float(np.max(submerged_path_m)),
        "bubble_water_transmittance_minimum": float(np.min(bubble_water_transmittance)),
        "bubble_attenuation_depth_m": float(bubble_attenuation_depth),
        "water_surface_epsilon_m": float(water_surface_epsilon),
        "foam_multiscale": {
            "fine_blur_sigma_px": float(blur_sigma * role_blur_scales[LABEL_FOAM]),
            "broad_blur_sigma_px": float(blur_sigma * foam_broad_blur_scale),
            "broad_fraction": float(foam_broad_fraction),
            "optical_depth_mass_preserving_blend": True,
        },
        "base_radius_m": float(base_radius),
        "extinction_scale": float(extinction_scale),
        "blur_sigma_px": float(blur_sigma),
        "optical_depth_maximum": float(np.max(optical_depth)),
        "optical_depth_p50_positive": float(np.quantile(positive, 0.50)) if len(positive) else 0.0,
        "optical_depth_p95_positive": float(np.quantile(positive, 0.95)) if len(positive) else 0.0,
        "optical_depth_p99_positive": float(np.quantile(positive, 0.99)) if len(positive) else 0.0,
        "roles": role_stats,
    }
    return role_optical_depth, stats


def save_linear_rgba(path: Path, rgba: np.ndarray, scene, name: str) -> None:
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("Expected an RGBA image")
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=True)
    image.pixels.foreach_set(np.asarray(rgba, dtype=np.float32).reshape(-1))
    image.file_format = "PNG"
    image.save_render(str(path), scene=scene)
    bpy.data.images.remove(image)


def bind_foam_density_to_surface(
    mesh, foam_points_isaac, neighbor_count, remaining_lifetime, influence_radius
):
    """Reconstruct a smooth Foam density directly on the imported Splashsurf vertices.

    The OBJ is still in Isaac XYZ local coordinates when this function runs.  A
    spatial hash limits the Gaussian accumulation to nearby surface vertices.
    """
    vertex_positions = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
    density_accumulator = np.zeros(len(vertex_positions), dtype=np.float64)
    if len(foam_points_isaac):
        inverse_cell = 1.0 / influence_radius
        vertex_cells = np.floor(vertex_positions * inverse_cell).astype(np.int64)
        spatial_hash: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for vertex_index, cell in enumerate(vertex_cells):
            spatial_hash[(int(cell[0]), int(cell[1]), int(cell[2]))].append(vertex_index)
        sigma = influence_radius * 0.46
        inverse_two_sigma_squared = 0.5 / (sigma * sigma)
        weights = 0.72 + 0.28 * np.clip(
            (neighbor_count.astype(np.float64) - 4.0) / 3.0, 0.0, 1.0
        )
        lifetime_weight = np.clip(remaining_lifetime.astype(np.float64) / 0.70, 0.0, 1.0)
        weights *= np.power(lifetime_weight, 1.35)
        offsets = tuple(
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        )
        radius_squared = influence_radius * influence_radius
        for point, weight in zip(foam_points_isaac, weights, strict=True):
            cell = np.floor(point * inverse_cell).astype(np.int64)
            for dx, dy, dz in offsets:
                indices = spatial_hash.get(
                    (int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz))
                )
                if not indices:
                    continue
                index_array = np.asarray(indices, dtype=np.int64)
                delta = vertex_positions[index_array] - point
                distance_squared = np.einsum("ij,ij->i", delta, delta)
                inside = distance_squared < radius_squared
                if not np.any(inside):
                    continue
                selected = index_array[inside]
                density_accumulator[selected] += weight * np.exp(
                    -distance_squared[inside] * inverse_two_sigma_squared
                )
    density = 1.0 - np.exp(-0.20 * density_accumulator)
    density = np.asarray(np.clip(density, 0.0, 1.0), dtype=np.float32)
    attribute = mesh.attributes.get("foam_density")
    if attribute is None:
        attribute = mesh.attributes.new("foam_density", "FLOAT", "POINT")
    attribute.data.foreach_set("value", density)
    return density


def remove_object(obj) -> None:
    if obj is None:
        return
    mesh = obj.data if isinstance(obj.data, bpy.types.Mesh) else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


args = parse_args()
positive_values = (
    args.samples,
    args.resolution,
    args.fps,
    args.spray_radius,
    args.spray_shutter_fraction,
    args.spray_maximum_half_length,
    args.foam_radius,
    args.foam_surface_influence_radius,
    args.bubble_radius,
    args.nvidia_diffuse_scale,
    args.camera_lens_mm,
    args.camera_sensor_width_mm,
    args.camera_clip_start_m,
    args.impactor_radius,
    args.screen_extinction_scale,
    args.screen_spray_weight,
    args.screen_foam_weight,
    args.screen_bubble_weight,
    args.screen_spray_blur_scale,
    args.screen_foam_blur_scale,
    args.screen_bubble_blur_scale,
    args.screen_foam_broad_blur_scale,
    args.screen_foam_detail_sigma_px,
    args.screen_bubble_attenuation_depth_m,
    args.screen_water_surface_epsilon_m,
)
if min(positive_values) <= 0:
    raise ValueError("All size, timing, camera and quality parameters must be positive")
if len(set(args.frames)) != len(args.frames):
    raise ValueError("frames must be unique")
if len(set(args.render_roles)) != len(args.render_roles):
    raise ValueError("render-roles must be unique")
if not (0.0 <= args.nvidia_diffuse_inscatter <= 1.0):
    raise ValueError("nvidia-diffuse-inscatter must be in [0, 1]")
if not (0.0 <= args.nvidia_diffuse_outscatter <= 1.0):
    raise ValueError("nvidia-diffuse-outscatter must be in [0, 1]")
if args.screen_thickness_blur_px < 0.0:
    raise ValueError("screen-thickness-blur-px must be non-negative")
if not (0.0 <= args.screen_foam_broad_fraction <= 1.0):
    raise ValueError("screen-foam-broad-fraction must be in [0, 1]")
if not (0.0 <= args.screen_foam_detail_retention <= 1.0):
    raise ValueError("screen-foam-detail-retention must be in [0, 1]")
if args.water_only:
    args.render_roles = []

labels_directory = args.labels_directory.resolve()
surface_directory = args.surface_directory.resolve()
run_report_path = args.run_report.resolve()
output_directory = args.output_directory.resolve()
output_directory.mkdir(parents=True, exist_ok=False)
labels_manifest_path = labels_directory / "manifest.json"
surface_manifest_path = surface_directory.parent / "splashsurf_manifest.json"
for required_path in (labels_manifest_path, surface_manifest_path, run_report_path):
    if not required_path.is_file():
        raise FileNotFoundError(required_path)
labels_manifest = json.loads(labels_manifest_path.read_text(encoding="utf-8"))
surface_manifest = json.loads(surface_manifest_path.read_text(encoding="utf-8"))
run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
if not labels_manifest.get("state", {}).get("complete"):
    raise ValueError("Label manifest is incomplete")
if not surface_manifest.get("state", {}).get("complete"):
    raise ValueError("Splashsurf manifest is incomplete")
if not run_report.get("valid"):
    raise ValueError("PhysX run report is not valid")

surface_by_frame = {
    int(frame["frame_id"]): frame for frame in surface_manifest["state"]["frames"]
}
metric_by_frame = {int(metric["output_frame"]): metric for metric in run_report["metrics"]}
contracts = []
for frame in args.frames:
    labels_path = labels_directory / f"diffuse_labels_{frame:04d}.npz"
    surface_record = surface_by_frame.get(frame)
    metric = metric_by_frame.get(frame)
    if surface_record is None or metric is None:
        raise KeyError(f"Missing same-frame surface or PhysX metric for frame {frame}")
    surface_path = surface_directory / surface_record["surface_file"]
    for path in (labels_path, surface_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contracts.append((frame, labels_path, surface_path, surface_record, metric))

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

world = scene.world or bpy.data.worlds.new("SwampPhysXDiffuseOpticalWorld")
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

camera_data = bpy.data.cameras.new("SwampPhysXDiffuseOpticalCamera")
camera = bpy.data.objects.new("SwampPhysXDiffuseOpticalCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
target = Vector(isaac_to_blender(args.camera_target))
eye = Vector(isaac_to_blender(args.camera_eye))
camera.location = eye
camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
camera_data.lens = args.camera_lens_mm
camera_data.sensor_width = args.camera_sensor_width_mm
camera_data.clip_start = args.camera_clip_start_m
bpy.context.view_layer.update()

water_material = make_water_with_surface_foam_material()
spray_material = make_spray_material()
foam_material = make_foam_material()
bubble_material = make_bubble_material()
microbubble_material = make_microbubble_material()
nvidia_diffuse_material = make_nvidia_diffuse_material(
    args.nvidia_diffuse_inscatter, args.nvidia_diffuse_outscatter
)
impactor_material = make_principled(
    "SwampImpactor", (0.95, 0.18, 0.035, 1.0), 0.24, 0.0, ior=1.45
)
collection = bpy.data.collections.new("SwampPhysXDiffuseOpticalRoles")
scene.collection.children.link(collection)
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=args.impactor_radius)
impactor = bpy.context.active_object
impactor.name = "SwampImpactor"
impactor.data.materials.append(impactor_material)
for polygon in impactor.data.polygons:
    polygon.use_smooth = True

manifest_path = output_directory / "render_manifest.json"
render_manifest = {
    "schema": 1,
    "product": "swamp_physx_diffuse_optical_roles_render",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "complete": False,
    "classification": {
        "authority": "PhysX 5.9 primary-neighbour thresholds recovered on GPU",
        "spray": "primary neighbours < 4",
        "foam": "primary neighbours 4..7",
        "bubble": "primary neighbours >= 8",
        "renderer_does_not_reclassify": True,
    },
    "optical_models": {
        "spray": "velocity-aligned transmissive water droplet; IOR 1.333",
        "foam": "PhysX Foam density reconstructed on the Splashsurf mesh with procedural Plateau borders",
        "bubble": "multi-scale: sparse 8..10-neighbour air-cavity shells plus 11..16-neighbour micro-bubble turbidity",
    },
    "configuration": {
        "frames": args.frames,
        "same_frame_mapping": True,
        "render_roles": list(args.render_roles),
        "material_profile": args.material_profile,
        "samples": args.samples,
        "resolution": [args.resolution, args.resolution],
        "fps": args.fps,
        "render_device": render_device,
        "spray_radius_m": args.spray_radius,
        "spray_shutter_fraction": args.spray_shutter_fraction,
        "spray_maximum_half_length_m": args.spray_maximum_half_length,
        "foam_base_radius_m": args.foam_radius,
        "foam_render_mode": args.foam_render_mode,
        "foam_surface_influence_radius_m": args.foam_surface_influence_radius,
        "bubble_base_radius_m": args.bubble_radius,
        "nvidia_diffuse": {
            "scale": args.nvidia_diffuse_scale,
            "motion_scale": 1.0,
            "hard_shadow": False,
            "inscatter": args.nvidia_diffuse_inscatter,
            "outscatter": args.nvidia_diffuse_outscatter,
            "source_defaults": "NVIDIA FleX demo-derived renderer",
        },
        "screen_space_thickness": {
            "extinction_scale": args.screen_extinction_scale,
            "blur_sigma_px": args.screen_thickness_blur_px,
            "role_weights": {
                "spray": args.screen_spray_weight,
                "foam": args.screen_foam_weight,
                "bubble": args.screen_bubble_weight,
            },
            "role_blur_scales": {
                "spray": args.screen_spray_blur_scale,
                "foam": args.screen_foam_blur_scale,
                "bubble": args.screen_bubble_blur_scale,
            },
            "bubble_attenuation_depth_m": args.screen_bubble_attenuation_depth_m,
            "water_surface_epsilon_m": args.screen_water_surface_epsilon_m,
            "bubble_depth_test": "camera ray against same-frame Splashsurf mesh",
            "foam_multiscale": {
                "broad_blur_scale": args.screen_foam_broad_blur_scale,
                "broad_fraction": args.screen_foam_broad_fraction,
                "mass_preserving": True,
                "water_detail_retention": args.screen_foam_detail_retention,
                "water_detail_sigma_px": args.screen_foam_detail_sigma_px,
            },
            "composite_order": ["bubble", "foam", "spray"],
            "transmittance": "exp(-optical_depth)",
        },
        "camera_eye_isaac_xyz_m": list(args.camera_eye),
        "camera_target_isaac_xyz_m": list(args.camera_target),
    },
    "inputs": {
        "labels_directory": str(labels_directory),
        "labels_manifest": str(labels_manifest_path),
        "labels_manifest_sha256": sha256_file(labels_manifest_path),
        "surface_directory": str(surface_directory),
        "surface_manifest": str(surface_manifest_path),
        "surface_manifest_sha256": sha256_file(surface_manifest_path),
        "run_report": str(run_report_path),
        "run_report_sha256": sha256_file(run_report_path),
        "blender_scene": str(Path(bpy.data.filepath).resolve()),
        "hdri": str(args.hdri.resolve()),
    },
    "frames": [],
}
atomic_json(manifest_path, render_manifest)

dynamic_objects = []
for sequence_index, contract in enumerate(contracts, start=1):
    for obj in dynamic_objects:
        remove_object(obj)
    dynamic_objects = []
    frame, labels_path, surface_path, surface_record, metric = contract

    bpy.ops.wm.obj_import(filepath=str(surface_path))
    water = bpy.context.active_object
    water.name = "SplashsurfWaterFrame"
    water.rotation_euler[0] = math.radians(90.0)
    water.data.materials.clear()
    water.data.materials.append(water_material)
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    dynamic_objects.append(water)

    with np.load(labels_path, allow_pickle=False) as cache:
        position_lifetime = np.asarray(cache["diffuse_position_lifetime"], dtype=np.float64)
        positions = position_lifetime[:, :3]
        remaining_lifetime = position_lifetime[:, 3]
        velocities = np.asarray(cache["diffuse_velocity"][:, :3], dtype=np.float64)
        labels = np.asarray(cache["recovered_label"], dtype=np.uint8)
        neighbor_count = np.asarray(cache["primary_neighbor_count"], dtype=np.uint8)
        diffuse_neighbor_radius = float(cache["diffuse_neighbor_radius"])
        cached_output_frame = int(cache["output_frame"])
    if cached_output_frame != frame:
        raise ValueError(f"Label payload frame {cached_output_frame} does not match requested frame {frame}")
    if not (len(positions) == len(velocities) == len(labels) == len(neighbor_count)):
        raise ValueError(f"Frame {frame}: Diffuse arrays have different lengths")
    unknown = np.count_nonzero((labels < LABEL_SPRAY) | (labels > LABEL_BUBBLE))
    if unknown:
        raise ValueError(f"Frame {frame}: {unknown} unknown recovered labels")

    spray_mask = labels == LABEL_SPRAY
    foam_mask = labels == LABEL_FOAM
    bubble_mask = labels == LABEL_BUBBLE
    bubble_shell_mask = bubble_mask & (neighbor_count <= 10)
    microbubble_mask = bubble_mask & (neighbor_count >= 11)
    counts = {
        "spray": int(np.count_nonzero(spray_mask)),
        "foam": int(np.count_nonzero(foam_mask)),
        "bubble": int(np.count_nonzero(bubble_mask)),
    }
    if sum(counts.values()) != len(labels):
        raise ValueError(f"Frame {frame}: label conservation failed")

    spray_render_mask = spray_mask if "spray" in args.render_roles else np.zeros_like(spray_mask)
    foam_render_mask = foam_mask if "foam" in args.render_roles else np.zeros_like(foam_mask)
    bubble_shell_render_mask = (
        bubble_shell_mask if "bubble_shell" in args.render_roles else np.zeros_like(bubble_shell_mask)
    )
    microbubble_render_mask = (
        microbubble_mask if "microbubble" in args.render_roles else np.zeros_like(microbubble_mask)
    )
    nvidia_render_mask = (
        spray_render_mask | foam_render_mask | bubble_shell_render_mask | microbubble_render_mask
    )
    screen_projection_mask = nvidia_render_mask.copy()
    nvidia_diffuse_stats = None
    nvidia_diffuse_obj = None
    screen_space_stats = None
    screen_space_base_path = None
    screen_space_base_exr_path = None
    screen_space_thickness_path = None
    screen_space_role_thickness_path = None
    if args.material_profile in {"nvidia_diffuse", "nvidia_screen_space"}:
        # NVIDIA-style profiles keep the original three-dimensional Diffuse
        # distribution. The water material receives no procedural Voronoi
        # foam; Foam is reconstructed optically alongside Spray and Bubble.
        foam_surface_density = bind_foam_density_to_surface(
            water.data,
            positions[:0],
            neighbor_count[:0],
            remaining_lifetime[:0],
            args.foam_surface_influence_radius,
        )
        if args.material_profile == "nvidia_diffuse":
            nvidia_diffuse_obj, nvidia_diffuse_stats = create_nvidia_diffuse_splats(
                positions[nvidia_render_mask],
                velocities[nvidia_render_mask],
                remaining_lifetime[nvidia_render_mask],
                labels[nvidia_render_mask],
                neighbor_count[nvidia_render_mask],
                diffuse_neighbor_radius,
                args.nvidia_diffuse_scale,
                args.fps,
                args.spray_shutter_fraction,
                camera,
                nvidia_diffuse_material,
                collection,
            )
        spray_obj = None
        foam_obj = None
        bubble_obj = None
        microbubble_obj = None
        spray_half_lengths = np.empty(0)
        foam_radii = np.empty(0)
        foam_alpha = np.empty(0)
        bubble_radii = np.empty(0)
        microbubble_radii = np.empty(0)
        microbubble_alpha = np.empty(0)
    else:
        spray_obj, spray_half_lengths = create_spray(
            positions[spray_render_mask], velocities[spray_render_mask], args.spray_radius, args.fps,
            args.spray_shutter_fraction, args.spray_maximum_half_length, spray_material, collection,
        )
        if args.foam_render_mode == "surface_plateau":
            foam_surface_density = bind_foam_density_to_surface(
                water.data,
                positions[foam_render_mask],
                neighbor_count[foam_render_mask],
                remaining_lifetime[foam_render_mask],
                args.foam_surface_influence_radius,
            )
            foam_obj = None
            foam_radii = np.empty(0)
            foam_alpha = np.empty(0)
        else:
            foam_surface_density = np.zeros(len(water.data.vertices), dtype=np.float32)
            foam_obj, foam_radii, foam_alpha = create_foam_splats(
                positions[foam_render_mask], neighbor_count[foam_render_mask], args.foam_radius,
                camera, foam_material, collection,
            )
        bubble_obj, bubble_radii = create_bubbles(
            positions[bubble_shell_render_mask], neighbor_count[bubble_shell_render_mask],
            args.bubble_radius, bubble_material, collection,
        )
        microbubble_obj, microbubble_radii, microbubble_alpha = create_microbubble_splats(
            positions[microbubble_render_mask], neighbor_count[microbubble_render_mask],
            args.foam_radius * 0.35, camera, microbubble_material, collection,
        )
    dynamic_objects.extend(
        obj
        for obj in (spray_obj, foam_obj, bubble_obj, microbubble_obj, nvidia_diffuse_obj)
        if obj is not None
    )

    impactor.location = isaac_to_blender(metric["sphere_center"])
    output_path = output_directory / f"optical_roles_{sequence_index:04d}_physx_{frame:04d}.png"
    scene.frame_set(sequence_index)
    if args.material_profile == "nvidia_screen_space":
        screen_space_base_path = output_directory / f"base_{sequence_index:04d}_physx_{frame:04d}.png"
        screen_space_base_exr_path = (
            output_directory / f"base_linear_{sequence_index:04d}_physx_{frame:04d}.exr"
        )
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = str(screen_space_base_exr_path)
        bpy.ops.render.render(write_still=True)
        base_image = bpy.data.images.load(str(screen_space_base_exr_path), check_existing=False)
        width = scene.render.resolution_x
        height = scene.render.resolution_y
        base_rgba = np.asarray(base_image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        bpy.data.images.remove(base_image)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        save_linear_rgba(screen_space_base_path, base_rgba, scene, f"ScreenDiffuseBase{frame:04d}")
        water_bvh = build_world_bvh(water)
        role_optical_depth, screen_space_stats = project_diffuse_optical_depth(
            positions[screen_projection_mask],
            velocities[screen_projection_mask],
            remaining_lifetime[screen_projection_mask],
            labels[screen_projection_mask],
            neighbor_count[screen_projection_mask],
            diffuse_neighbor_radius,
            args.nvidia_diffuse_scale,
            args.fps,
            args.spray_shutter_fraction,
            camera,
            width,
            height,
            metric["sphere_center"],
            args.impactor_radius,
            args.screen_extinction_scale,
            args.screen_thickness_blur_px,
            (
                args.screen_spray_weight,
                args.screen_foam_weight,
                args.screen_bubble_weight,
            ),
            (
                args.screen_spray_blur_scale,
                args.screen_foam_blur_scale,
                args.screen_bubble_blur_scale,
            ),
            args.screen_foam_broad_blur_scale,
            args.screen_foam_broad_fraction,
            water_bvh,
            args.screen_bubble_attenuation_depth_m,
            args.screen_water_surface_epsilon_m,
        )
        optical_depth = np.sum(role_optical_depth, axis=0)
        transmittance = np.exp(-optical_depth).astype(np.float32)
        composite = base_rgba.copy()
        # Broad submerged turbidity first, connected surface foam second, and
        # sparse ballistic spray last.  This is still a camera-space optical
        # approximation, but no longer assigns the same material response to
        # three physically different PhysX states.
        role_composites = (
            (LABEL_BUBBLE, (0.70, 0.80, 0.74), 0.45),
            (LABEL_FOAM, (0.93, 0.965, 0.94), 1.05),
            (LABEL_SPRAY, (0.84, 0.91, 0.89), 0.75),
        )
        for role_index, colour, inscatter_scale in role_composites:
            role_transmittance = np.exp(-role_optical_depth[role_index]).astype(np.float32)
            scatter_colour = np.asarray(colour, dtype=np.float32)
            role_inscatter = min(1.0, args.nvidia_diffuse_inscatter * inscatter_scale)
            role_input = composite[:, :, :3].copy()
            role_output = (
                role_input * role_transmittance[:, :, None]
                + scatter_colour[None, None, :]
                * (1.0 - role_transmittance[:, :, None])
                * role_inscatter
            )
            if role_index == LABEL_FOAM and args.screen_foam_detail_retention > 0.0:
                low_frequency = np.empty_like(role_input)
                for channel in range(3):
                    low_frequency[:, :, channel] = gaussian_blur_scalar(
                        role_input[:, :, channel], args.screen_foam_detail_sigma_px
                    )
                water_high_frequency = role_input - low_frequency
                foam_coverage = 1.0 - role_transmittance
                role_output += (
                    water_high_frequency
                    * foam_coverage[:, :, None]
                    * args.screen_foam_detail_retention
                )
            composite[:, :, :3] = np.maximum(role_output, 0.0)
        save_linear_rgba(output_path, composite, scene, f"ScreenDiffuseComposite{frame:04d}")
        thickness_visible = 1.0 - transmittance
        thickness_rgba = np.empty((height, width, 4), dtype=np.float32)
        thickness_rgba[:, :, :3] = thickness_visible[:, :, None]
        thickness_rgba[:, :, 3] = 1.0
        screen_space_thickness_path = (
            output_directory / f"optical_thickness_{sequence_index:04d}_physx_{frame:04d}.png"
        )
        save_linear_rgba(
            screen_space_thickness_path,
            thickness_rgba,
            scene,
            f"ScreenDiffuseThickness{frame:04d}",
        )
        role_thickness_rgba = np.empty((height, width, 4), dtype=np.float32)
        role_thickness_rgba[:, :, 0] = 1.0 - np.exp(-role_optical_depth[LABEL_SPRAY])
        role_thickness_rgba[:, :, 1] = 1.0 - np.exp(-role_optical_depth[LABEL_FOAM])
        role_thickness_rgba[:, :, 2] = 1.0 - np.exp(-role_optical_depth[LABEL_BUBBLE])
        role_thickness_rgba[:, :, 3] = 1.0
        screen_space_role_thickness_path = (
            output_directory
            / f"optical_thickness_roles_{sequence_index:04d}_physx_{frame:04d}.png"
        )
        save_linear_rgba(
            screen_space_role_thickness_path,
            role_thickness_rgba,
            scene,
            f"ScreenDiffuseRoleThickness{frame:04d}",
        )
        screen_space_stats["transmittance_minimum"] = float(np.min(transmittance))
        screen_space_stats["transmittance_p01"] = float(np.quantile(transmittance, 0.01))
        screen_space_stats["transmittance_p05"] = float(np.quantile(transmittance, 0.05))
    else:
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
    frame_record = {
        "sequence_index": sequence_index,
        "physx_frame": frame,
        "active_diffuse_particles": len(labels),
        "label_histogram": counts,
        "label_conservation": sum(counts.values()) == len(labels),
        "nvidia_diffuse_renderer": nvidia_diffuse_stats,
        "screen_space_renderer": screen_space_stats,
        "screen_space_base_png": str(screen_space_base_path) if screen_space_base_path else None,
        "screen_space_base_sha256": (
            sha256_file(screen_space_base_path) if screen_space_base_path else None
        ),
        "screen_space_base_linear_exr": (
            str(screen_space_base_exr_path) if screen_space_base_exr_path else None
        ),
        "screen_space_base_linear_exr_sha256": (
            sha256_file(screen_space_base_exr_path) if screen_space_base_exr_path else None
        ),
        "screen_space_thickness_png": (
            str(screen_space_thickness_path) if screen_space_thickness_path else None
        ),
        "screen_space_thickness_sha256": (
            sha256_file(screen_space_thickness_path) if screen_space_thickness_path else None
        ),
        "screen_space_role_thickness_png": (
            str(screen_space_role_thickness_path) if screen_space_role_thickness_path else None
        ),
        "screen_space_role_thickness_sha256": (
            sha256_file(screen_space_role_thickness_path)
            if screen_space_role_thickness_path
            else None
        ),
        "rendered_optical_samples": {
            "spray": int(np.count_nonzero(spray_render_mask)),
            "foam": int(np.count_nonzero(foam_render_mask)),
            "bubble_shell": int(np.count_nonzero(bubble_shell_render_mask)),
            "microbubble": int(np.count_nonzero(microbubble_render_mask)),
        },
        "bubble_optical_partition": {
            "explicit_air_cavity_shells": int(np.count_nonzero(bubble_shell_mask)),
            "microbubble_density_samples": int(np.count_nonzero(microbubble_mask)),
            "conservation": int(np.count_nonzero(bubble_shell_mask))
            + int(np.count_nonzero(microbubble_mask)) == counts["bubble"],
        },
        "labels_npz": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "surface_obj": str(surface_path),
        "surface_sha256": sha256_file(surface_path),
        "surface_manifest_sha256": surface_record["surface_sha256"],
        "sphere_center_isaac_xyz_m": metric["sphere_center"],
        "spray_half_length_range_m": (
            [float(np.min(spray_half_lengths)), float(np.max(spray_half_lengths))]
            if len(spray_half_lengths) else None
        ),
        "foam_radius_range_m": (
            [float(np.min(foam_radii)), float(np.max(foam_radii))] if len(foam_radii) else None
        ),
        "foam_centre_alpha_range": (
            [float(np.min(foam_alpha)), float(np.max(foam_alpha))] if len(foam_alpha) else None
        ),
        "foam_surface_density": {
            "active_vertices": int(np.count_nonzero(foam_surface_density > 1.0e-4)),
            "maximum": float(np.max(foam_surface_density)) if len(foam_surface_density) else 0.0,
            "mean_on_active_vertices": (
                float(np.mean(foam_surface_density[foam_surface_density > 1.0e-4]))
                if np.any(foam_surface_density > 1.0e-4) else 0.0
            ),
        },
        "bubble_radius_range_m": (
            [float(np.min(bubble_radii)), float(np.max(bubble_radii))] if len(bubble_radii) else None
        ),
        "microbubble_radius_range_m": (
            [float(np.min(microbubble_radii)), float(np.max(microbubble_radii))]
            if len(microbubble_radii) else None
        ),
        "microbubble_centre_alpha_range": (
            [float(np.min(microbubble_alpha)), float(np.max(microbubble_alpha))]
            if len(microbubble_alpha) else None
        ),
        "png": str(output_path),
        "png_sha256": sha256_file(output_path),
    }
    render_manifest["frames"].append(frame_record)
    atomic_json(manifest_path, render_manifest)
    print(
        f"[physx-optical] frame={frame:04d} total={len(labels)} "
        f"spray={counts['spray']} foam={counts['foam']} bubble={counts['bubble']} "
        f"sequence={sequence_index}/{len(contracts)}"
    )

render_manifest["complete"] = True
render_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
atomic_json(manifest_path, render_manifest)
