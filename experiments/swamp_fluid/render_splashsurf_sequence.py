"""Render a terrain-clipped Splashsurf OBJ sequence in Swamp with Cycles."""

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


FOAM_DENSITY_SUPPORT_M = 0.006
FOAM_DENSITY_ONSET_NEIGHBORS = 8.0
FOAM_DENSITY_FULL_NEIGHBORS = 17.0
FOAM_THICKNESS_ONSET_M = 0.00245
FOAM_THICKNESS_FULL_M = 0.00285
FOAM_MAX_SCATTER_MIX = 0.60


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("physics_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    # Preserve the original positional interface while providing named options.
    parser.add_argument("legacy_samples", nargs="?", type=int)
    parser.add_argument("legacy_resolution", nargs="?", type=int)
    parser.add_argument("legacy_frames", nargs="?", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--frame-list", nargs="+", type=int)
    parser.add_argument("--secondary-directory", type=Path)
    parser.add_argument(
        "--plateau-directory",
        type=Path,
        help="Optional audited PhysX/FoamGenerator Plateau-cell cache.",
    )
    parser.add_argument("--plateau-micro-pixel-radius", type=float, default=0.40)
    parser.add_argument("--plateau-hero-pixel-radius", type=float, default=1.35)
    parser.add_argument("--plateau-micro-bin-size", type=float, default=0.006)
    parser.add_argument(
        "--plateau-components",
        choices=("all", "borders-nodes"),
        default="all",
    )
    parser.add_argument(
        "--foam-payload-directory",
        type=Path,
        help="Optional conservative FoamGenerator surface-film payload.",
    )
    parser.add_argument(
        "--foam-membrane-directory",
        type=Path,
        help="Deprecated connected-membrane cache; retained only for explicit A/B regression.",
    )
    parser.add_argument(
        "--allow-obsolete-foam-membrane",
        action="store_true",
        help="Explicitly allow an OBSOLETE connected membrane for regression rendering.",
    )
    parser.add_argument(
        "--foam-coverage-field-directory",
        type=Path,
        help=(
            "Optional audited FoamGenerator coverage field. It writes scalar "
            "attributes onto the complete liquid surface and renders unbound foam "
            "through its exclusive free-particle route."
        ),
    )
    parser.add_argument(
        "--foam-membrane-shader",
        choices=("wet-film", "cellular"),
        default="wet-film",
        help=(
            "Wet-film is the production default for ordinary shots; cellular is an "
            "explicit hero-close-up diagnostic and is never enabled implicitly."
        ),
    )
    parser.add_argument(
        "--secondary-layers",
        nargs="+",
        choices=("spray", "foam", "bubbles"),
        default=("spray", "foam", "bubbles"),
        help=(
            "Secondary layers to render. This is a render-only visibility "
            "filter; it does not change cached classifications."
        ),
    )
    parser.add_argument(
        "--foam-particle-shader",
        choices=("wet-film", "opaque"),
        default="wet-film",
        help=(
            "Render native FoamGenerator foam parcels as a translucent wet-film "
            "proxy or the legacy opaque diagnostic material."
        ),
    )
    parser.add_argument(
        "--foam-surface-representation",
        choices=("marker-spheres", "microbubble-patches", "microbubble-patches-hero"),
        default="marker-spheres",
        help=(
            "Interpret surface-bound FoamGenerator markers as legacy spheres or "
            "stable-ID microbubble cluster patches, optionally with sparse domes."
        ),
    )
    parser.add_argument(
        "--foam-binding-directory",
        type=Path,
        help="Audited per-frame surface binding required by microbubble patch modes.",
    )
    parser.add_argument("--microbubble-debug-opaque", action="store_true")
    parser.add_argument("--microbubble-debug-mask", action="store_true")
    parser.add_argument("--microbubble-debug-alpha", action="store_true")
    parser.add_argument("--volume-directory", type=Path)
    parser.add_argument("--volume-absorption-density", type=float, default=0.45)
    parser.add_argument("--volume-scatter-density", type=float, default=0.025)
    parser.add_argument("--volume-sphere-clearance", type=float, default=0.001)
    parser.add_argument("--volume-sphere-falloff", type=float, default=0.0015)
    parser.add_argument(
        "--camera",
        choices=("overview", "waterline", "underwater"),
        default="overview",
    )
    parser.add_argument("--camera-location", nargs=3, type=float)
    parser.add_argument("--camera-target", nargs=3, type=float)
    parser.add_argument("--camera-lens", type=float)
    parser.add_argument(
        "--motion-blur-shutter",
        type=float,
        default=0.5,
        help="Cycles shutter duration in output-frame units; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hdri",
        type=Path,
        default=Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr"),
    )
    parser.add_argument("--hdri-strength", type=float, default=0.95)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--render-device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Require an actual Cycles GPU device or explicitly opt into CPU rendering.",
    )
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(path, configuration, state):
    payload = {
        "schema": 2,
        "configuration": configuration,
        "state": state,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def isaac_to_blender(values):
    x, y, z = values
    return Vector((float(x), -float(z), float(y)))


def set_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return
    raise RuntimeError(f"Missing Principled input {names}")


def manifest_configuration_hash(path):
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    configuration = payload.get("configuration")
    return sha256_json(configuration) if configuration is not None else None


def make_principled_material(name, base_color, roughness, transmission=0.0, ior=1.45):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(shader, base_color, "Base Color")
    set_input(shader, roughness, "Roughness")
    set_input(shader, ior, "IOR")
    set_input(shader, transmission, "Transmission Weight", "Transmission")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_volume_material(
    absorption_density,
    scatter_density,
    sphere_radius,
    sphere_clearance,
    sphere_falloff,
):
    material = bpy.data.materials.new("SwampWaterClosedMedium")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = (0.32, 0.58, 0.48, 1.0)
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (0.72, 0.86, 0.78, 1.0)
    if scatter.inputs.get("Anisotropy") is not None:
        scatter.inputs["Anisotropy"].default_value = 0.15

    # A transparent Surface closure is unnecessary for a pure volume boundary
    # and consumes another closure on every nested ray.  Mask the density near
    # the opaque impactor instead of applying a topology-sensitive Boolean to
    # every 180k-triangle volume mesh.
    geometry = nodes.new("ShaderNodeNewGeometry")
    sphere_center = nodes.new("ShaderNodeCombineXYZ")
    distance = nodes.new("ShaderNodeVectorMath")
    distance.operation = "DISTANCE"
    material.node_tree.links.new(geometry.outputs["Position"], distance.inputs[0])
    material.node_tree.links.new(sphere_center.outputs["Vector"], distance.inputs[1])
    density_mask = nodes.new("ShaderNodeMapRange")
    density_mask.clamp = True
    density_mask.inputs["From Min"].default_value = sphere_radius + sphere_clearance
    density_mask.inputs["From Max"].default_value = (
        sphere_radius + sphere_clearance + sphere_falloff
    )
    density_mask.inputs["To Min"].default_value = 0.0
    density_mask.inputs["To Max"].default_value = 1.0
    material.node_tree.links.new(distance.outputs["Value"], density_mask.inputs["Value"])
    absorption_scale = nodes.new("ShaderNodeMath")
    absorption_scale.operation = "MULTIPLY"
    absorption_scale.inputs[1].default_value = absorption_density
    scatter_scale = nodes.new("ShaderNodeMath")
    scatter_scale.operation = "MULTIPLY"
    scatter_scale.inputs[1].default_value = scatter_density
    material.node_tree.links.new(
        density_mask.outputs["Result"], absorption_scale.inputs[0]
    )
    material.node_tree.links.new(density_mask.outputs["Result"], scatter_scale.inputs[0])
    material.node_tree.links.new(absorption_scale.outputs[0], absorption.inputs["Density"])
    material.node_tree.links.new(scatter_scale.outputs[0], scatter.inputs["Density"])
    add_volume = nodes.new("ShaderNodeAddShader")
    material.node_tree.links.new(absorption.outputs["Volume"], add_volume.inputs[0])
    material.node_tree.links.new(scatter.outputs["Volume"], add_volume.inputs[1])
    material.node_tree.links.new(add_volume.outputs[0], output.inputs["Volume"])
    return material, sphere_center


def make_bubble_material():
    material = bpy.data.materials.new("EntrainedAirBubble")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (0.96, 0.99, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.015
    glass.inputs["IOR"].default_value = 1.333
    material.node_tree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_plateau_film_material(name, roughness, transmission):
    material = make_principled_material(
        name,
        (0.94, 0.985, 0.96, 1.0),
        roughness,
        transmission=transmission,
        ior=1.333,
    )
    nodes = material.node_tree.nodes
    shader = next(node for node in nodes if node.bl_idname == "ShaderNodeBsdfPrincipled")
    thickness_socket = shader.inputs.get("Thin Film Thickness")
    if thickness_socket is not None:
        attribute = nodes.new("ShaderNodeAttribute")
        attribute.attribute_type = "GEOMETRY"
        attribute.attribute_name = "ww_thickness_nm"
        material.node_tree.links.new(attribute.outputs["Fac"], thickness_socket)
        set_input(shader, 1.333, "Thin Film IOR")
    wetness = nodes.new("ShaderNodeAttribute")
    wetness.attribute_type = "GEOMETRY"
    wetness.attribute_name = "ww_wetness"
    mapping = nodes.new("ShaderNodeMapRange")
    mapping.clamp = True
    mapping.inputs["From Min"].default_value = 0.0
    mapping.inputs["From Max"].default_value = 1.0
    mapping.inputs["To Min"].default_value = max(roughness, 0.12)
    mapping.inputs["To Max"].default_value = min(roughness, 0.025)
    material.node_tree.links.new(wetness.outputs["Fac"], mapping.inputs["Value"])
    material.node_tree.links.new(mapping.outputs["Result"], shader.inputs["Roughness"])
    return material


def make_wet_foam_membrane_material(name):
    """Coverage-aware transmissive wet film without alpha or cellular blur."""

    material = make_principled_material(
        name,
        (0.84, 0.89, 0.81, 1.0),
        0.12,
        transmission=0.68,
        ior=1.333,
    )
    shader = next(
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    set_input(shader, 0.16, "Coat Weight", "Coat")
    set_input(shader, 0.10, "Coat Roughness")
    coverage = material.node_tree.nodes.new("ShaderNodeAttribute")
    coverage.attribute_type = "GEOMETRY"
    coverage.attribute_name = "foam_coverage"
    transmission = material.node_tree.nodes.new("ShaderNodeMapRange")
    transmission.clamp = True
    transmission.inputs["From Min"].default_value = 0.0
    transmission.inputs["From Max"].default_value = 1.0
    transmission.inputs["To Min"].default_value = 0.94
    transmission.inputs["To Max"].default_value = 0.42
    roughness = material.node_tree.nodes.new("ShaderNodeMapRange")
    roughness.clamp = True
    roughness.inputs["From Min"].default_value = 0.0
    roughness.inputs["From Max"].default_value = 1.0
    roughness.inputs["To Min"].default_value = 0.055
    roughness.inputs["To Max"].default_value = 0.20
    material.node_tree.links.new(coverage.outputs["Fac"], transmission.inputs["Value"])
    material.node_tree.links.new(coverage.outputs["Fac"], roughness.inputs["Value"])
    transmission_socket = shader.inputs.get("Transmission Weight") or shader.inputs.get("Transmission")
    if transmission_socket is not None:
        material.node_tree.links.new(transmission.outputs["Result"], transmission_socket)
    material.node_tree.links.new(roughness.outputs["Result"], shader.inputs["Roughness"])
    thickness_socket = shader.inputs.get("Thin Film Thickness")
    if thickness_socket is not None:
        attribute = material.node_tree.nodes.new("ShaderNodeAttribute")
        attribute.attribute_type = "GEOMETRY"
        attribute.attribute_name = "ww_thickness_nm"
        material.node_tree.links.new(attribute.outputs["Fac"], thickness_socket)
        set_input(shader, 1.33, "Thin Film IOR")
    return material


def make_wet_foam_particle_material(name):
    """Wet-film base with sparse density- and edge-driven microfoam scatter."""

    material = make_principled_material(
        name,
        (0.84, 0.89, 0.81, 1.0),
        0.12,
        transmission=0.68,
        ior=1.333,
    )
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = next(
        node for node in nodes if node.bl_idname == "ShaderNodeOutputMaterial"
    )
    shader = next(
        node
        for node in nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    set_input(shader, 0.16, "Coat Weight", "Coat")
    set_input(shader, 0.10, "Coat Roughness")

    scatter = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(scatter, (0.72, 0.75, 0.70, 1.0), "Base Color")
    set_input(scatter, 0.38, "Roughness")
    set_input(scatter, 1.333, "IOR")
    set_input(scatter, 0.08, "Transmission Weight", "Transmission")
    set_input(scatter, 0.10, "Coat Weight", "Coat")
    set_input(scatter, 0.18, "Coat Roughness")

    density = nodes.new("ShaderNodeAttribute")
    density.attribute_type = "GEOMETRY"
    density.attribute_name = "foam_local_density"
    thickness = nodes.new("ShaderNodeAttribute")
    thickness.attribute_type = "GEOMETRY"
    thickness.attribute_name = "foam_particle_thickness"

    geometry = nodes.new("ShaderNodeNewGeometry")
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.distance = "EUCLIDEAN"
    voronoi.feature = "DISTANCE_TO_EDGE"
    voronoi.inputs["Scale"].default_value = 280.0
    links.new(geometry.outputs["Position"], voronoi.inputs["Vector"])
    cell_edges = nodes.new("ShaderNodeMapRange")
    cell_edges.clamp = True
    cell_edges.inputs["From Min"].default_value = 0.0
    cell_edges.inputs["From Max"].default_value = 0.11
    cell_edges.inputs["To Min"].default_value = 1.0
    cell_edges.inputs["To Max"].default_value = 0.0
    links.new(voronoi.outputs["Distance"], cell_edges.inputs["Value"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = 360.0
    noise.inputs["Detail"].default_value = 2.2
    noise.inputs["Roughness"].default_value = 0.58
    links.new(geometry.outputs["Position"], noise.inputs["Vector"])
    noise_gate = nodes.new("ShaderNodeMapRange")
    noise_gate.clamp = True
    noise_gate.inputs["From Min"].default_value = 0.44
    noise_gate.inputs["From Max"].default_value = 0.70
    noise_gate.inputs["To Min"].default_value = 0.18
    noise_gate.inputs["To Max"].default_value = 1.0
    links.new(noise.outputs["Fac"], noise_gate.inputs["Value"])
    breakup = nodes.new("ShaderNodeMath")
    breakup.operation = "MULTIPLY"
    links.new(cell_edges.outputs["Result"], breakup.inputs[0])
    links.new(noise_gate.outputs["Result"], breakup.inputs[1])

    layer_weight = nodes.new("ShaderNodeLayerWeight")
    rim = nodes.new("ShaderNodeMath")
    rim.operation = "SUBTRACT"
    rim.inputs[0].default_value = 1.0
    links.new(layer_weight.outputs["Facing"], rim.inputs[1])
    weighted_rim = nodes.new("ShaderNodeMath")
    weighted_rim.operation = "MULTIPLY"
    weighted_rim.inputs[1].default_value = 0.25
    links.new(rim.outputs[0], weighted_rim.inputs[0])
    weighted_thickness = nodes.new("ShaderNodeMath")
    weighted_thickness.operation = "MULTIPLY"
    weighted_thickness.inputs[1].default_value = 0.10
    links.new(thickness.outputs["Fac"], weighted_thickness.inputs[0])
    weighted_breakup = nodes.new("ShaderNodeMath")
    weighted_breakup.operation = "MULTIPLY"
    weighted_breakup.inputs[1].default_value = 0.55
    links.new(breakup.outputs[0], weighted_breakup.inputs[0])
    structure = nodes.new("ShaderNodeMath")
    structure.operation = "ADD"
    structure.inputs[0].default_value = 0.10
    links.new(weighted_thickness.outputs[0], structure.inputs[1])
    add_rim = nodes.new("ShaderNodeMath")
    add_rim.operation = "ADD"
    links.new(structure.outputs[0], add_rim.inputs[0])
    links.new(weighted_rim.outputs[0], add_rim.inputs[1])
    add_breakup = nodes.new("ShaderNodeMath")
    add_breakup.operation = "ADD"
    add_breakup.use_clamp = True
    links.new(add_rim.outputs[0], add_breakup.inputs[0])
    links.new(weighted_breakup.outputs[0], add_breakup.inputs[1])

    scatter_mask = nodes.new("ShaderNodeMath")
    scatter_mask.operation = "MULTIPLY"
    links.new(density.outputs["Fac"], scatter_mask.inputs[0])
    links.new(add_breakup.outputs[0], scatter_mask.inputs[1])
    strength = nodes.new("ShaderNodeMath")
    strength.operation = "MULTIPLY"
    strength.inputs[1].default_value = FOAM_MAX_SCATTER_MIX
    strength.use_clamp = True
    links.new(scatter_mask.outputs[0], strength.inputs[0])

    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    mix = nodes.new("ShaderNodeMixShader")
    links.new(strength.outputs[0], mix.inputs[0])
    links.new(shader.outputs["BSDF"], mix.inputs[1])
    links.new(scatter.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    return material


def make_microbubble_patch_material(name):
    """Stable-ID cellular wet patch for one marker representing a microbubble group."""

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    # A surface patch is a render overlay, not a second closed refractive
    # volume.  A glossy lobe over transparency preserves wet highlights without
    # creating the dark double-interface discs caused by a thin glass cylinder.
    wet = nodes.new("ShaderNodeBsdfGlossy")
    wet.inputs["Color"].default_value = (0.84, 0.89, 0.81, 1.0)
    wet.inputs["Roughness"].default_value = 0.07
    scatter = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(scatter, (0.79, 0.82, 0.77, 1.0), "Base Color")
    set_input(scatter, 0.38, "Roughness")
    set_input(scatter, 1.333, "IOR")
    set_input(scatter, 0.08, "Transmission Weight", "Transmission")

    uv = nodes.new("ShaderNodeAttribute")
    uv.attribute_type = "GEOMETRY"
    uv.attribute_name = "foam_patch_uv"
    seed = nodes.new("ShaderNodeAttribute")
    seed.attribute_type = "GEOMETRY"
    seed.attribute_name = "foam_marker_seed"
    density = nodes.new("ShaderNodeAttribute")
    density.attribute_type = "GEOMETRY"
    density.attribute_name = "foam_local_density"
    cell_scale = nodes.new("ShaderNodeAttribute")
    cell_scale.attribute_type = "GEOMETRY"
    cell_scale.attribute_name = "foam_cell_scale"
    opacity = nodes.new("ShaderNodeAttribute")
    opacity.attribute_type = "GEOMETRY"
    opacity.attribute_name = "foam_marker_opacity"

    seed_x = nodes.new("ShaderNodeMath")
    seed_x.operation = "MULTIPLY"
    seed_x.inputs[1].default_value = 19.19
    links.new(seed.outputs["Fac"], seed_x.inputs[0])
    seed_y = nodes.new("ShaderNodeMath")
    seed_y.operation = "MULTIPLY"
    seed_y.inputs[1].default_value = 47.77
    links.new(seed.outputs["Fac"], seed_y.inputs[0])
    seed_vector = nodes.new("ShaderNodeCombineXYZ")
    links.new(seed_x.outputs[0], seed_vector.inputs["X"])
    links.new(seed_y.outputs[0], seed_vector.inputs["Y"])
    stable_uv = nodes.new("ShaderNodeVectorMath")
    stable_uv.operation = "ADD"
    links.new(uv.outputs["Vector"], stable_uv.inputs[0])
    links.new(seed_vector.outputs["Vector"], stable_uv.inputs[1])

    cells = nodes.new("ShaderNodeTexVoronoi")
    cells.distance = "EUCLIDEAN"
    cells.feature = "DISTANCE_TO_EDGE"
    links.new(stable_uv.outputs["Vector"], cells.inputs["Vector"])
    links.new(cell_scale.outputs["Fac"], cells.inputs["Scale"])
    cell_rings = nodes.new("ShaderNodeMapRange")
    cell_rings.clamp = True
    cell_rings.inputs["From Min"].default_value = 0.0
    cell_rings.inputs["From Max"].default_value = 0.32
    cell_rings.inputs["To Min"].default_value = 1.0
    cell_rings.inputs["To Max"].default_value = 0.0
    links.new(cells.outputs["Distance"], cell_rings.inputs["Value"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = 2.6
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.62
    links.new(stable_uv.outputs["Vector"], noise.inputs["Vector"])
    noise_gate = nodes.new("ShaderNodeMapRange")
    noise_gate.clamp = True
    noise_gate.inputs["From Min"].default_value = 0.30
    noise_gate.inputs["From Max"].default_value = 0.72
    noise_gate.inputs["To Min"].default_value = 0.42
    noise_gate.inputs["To Max"].default_value = 1.0
    links.new(noise.outputs["Fac"], noise_gate.inputs["Value"])

    ring_density = nodes.new("ShaderNodeMapRange")
    ring_density.clamp = True
    ring_density.inputs["From Min"].default_value = 0.0
    ring_density.inputs["From Max"].default_value = 1.0
    ring_density.inputs["To Min"].default_value = 0.06
    ring_density.inputs["To Max"].default_value = 0.68
    links.new(density.outputs["Fac"], ring_density.inputs["Value"])
    rings = nodes.new("ShaderNodeMath")
    rings.operation = "MULTIPLY"
    links.new(cell_rings.outputs["Result"], rings.inputs[0])
    links.new(noise_gate.outputs["Result"], rings.inputs[1])
    weighted_rings = nodes.new("ShaderNodeMath")
    weighted_rings.operation = "MULTIPLY"
    links.new(rings.outputs[0], weighted_rings.inputs[0])
    links.new(ring_density.outputs["Result"], weighted_rings.inputs[1])
    density_squared = nodes.new("ShaderNodeMath")
    density_squared.operation = "MULTIPLY"
    links.new(density.outputs["Fac"], density_squared.inputs[0])
    links.new(density.outputs["Fac"], density_squared.inputs[1])
    interior_milk = nodes.new("ShaderNodeMath")
    interior_milk.operation = "MULTIPLY"
    interior_milk.inputs[1].default_value = 0.12
    links.new(density_squared.outputs[0], interior_milk.inputs[0])
    scatter_mix = nodes.new("ShaderNodeMath")
    scatter_mix.operation = "ADD"
    scatter_mix.use_clamp = True
    links.new(weighted_rings.outputs[0], scatter_mix.inputs[0])
    links.new(interior_milk.outputs[0], scatter_mix.inputs[1])
    body = nodes.new("ShaderNodeMixShader")
    links.new(scatter_mix.outputs[0], body.inputs[0])
    links.new(wet.outputs["BSDF"], body.inputs[1])
    links.new(scatter.outputs["BSDF"], body.inputs[2])

    radius = nodes.new("ShaderNodeVectorMath")
    radius.operation = "LENGTH"
    links.new(uv.outputs["Vector"], radius.inputs[0])
    boundary_radius = nodes.new("ShaderNodeMapRange")
    boundary_radius.clamp = True
    boundary_radius.inputs["From Min"].default_value = 0.0
    boundary_radius.inputs["From Max"].default_value = 1.0
    boundary_radius.inputs["To Min"].default_value = 0.80
    boundary_radius.inputs["To Max"].default_value = 0.98
    links.new(noise.outputs["Fac"], boundary_radius.inputs["Value"])
    normalized_radius = nodes.new("ShaderNodeMath")
    normalized_radius.operation = "DIVIDE"
    links.new(radius.outputs["Value"], normalized_radius.inputs[0])
    links.new(boundary_radius.outputs["Result"], normalized_radius.inputs[1])
    silhouette = nodes.new("ShaderNodeMapRange")
    silhouette.clamp = True
    silhouette.inputs["From Min"].default_value = 0.86
    silhouette.inputs["From Max"].default_value = 1.02
    silhouette.inputs["To Min"].default_value = 1.0
    silhouette.inputs["To Max"].default_value = 0.0
    links.new(normalized_radius.outputs[0], silhouette.inputs["Value"])

    structured_tau = nodes.new("ShaderNodeMath")
    structured_tau.operation = "MULTIPLY"
    structured_tau.inputs[1].default_value = 3.20
    links.new(scatter_mix.outputs[0], structured_tau.inputs[0])
    base_tau = nodes.new("ShaderNodeMath")
    base_tau.operation = "ADD"
    base_tau.inputs[1].default_value = 0.02
    links.new(structured_tau.outputs[0], base_tau.inputs[0])
    tau = nodes.new("ShaderNodeMath")
    tau.operation = "MULTIPLY"
    links.new(opacity.outputs["Fac"], tau.inputs[0])
    links.new(base_tau.outputs[0], tau.inputs[1])
    negative_tau = nodes.new("ShaderNodeMath")
    negative_tau.operation = "MULTIPLY"
    negative_tau.inputs[1].default_value = -1.0
    links.new(tau.outputs[0], negative_tau.inputs[0])
    transmittance = nodes.new("ShaderNodeMath")
    transmittance.operation = "EXPONENT"
    links.new(negative_tau.outputs[0], transmittance.inputs[0])
    coverage = nodes.new("ShaderNodeMath")
    coverage.operation = "SUBTRACT"
    coverage.inputs[0].default_value = 1.0
    links.new(transmittance.outputs[0], coverage.inputs[1])
    patch_alpha = nodes.new("ShaderNodeMath")
    patch_alpha.operation = "MULTIPLY"
    links.new(coverage.outputs[0], patch_alpha.inputs[0])
    links.new(silhouette.outputs["Result"], patch_alpha.inputs[1])
    final_mix = nodes.new("ShaderNodeMixShader")
    links.new(patch_alpha.outputs[0], final_mix.inputs[0])
    links.new(transparent.outputs["BSDF"], final_mix.inputs[1])
    links.new(body.outputs[0], final_mix.inputs[2])
    links.new(final_mix.outputs[0], output.inputs["Surface"])
    if args.microbubble_debug_opaque:
        debug = nodes.new("ShaderNodeEmission")
        debug.inputs["Color"].default_value = (1.0, 0.02, 0.02, 1.0)
        debug.inputs["Strength"].default_value = 1.0
        links.new(debug.outputs[0], output.inputs["Surface"])
    elif args.microbubble_debug_mask:
        channels = nodes.new("ShaderNodeCombineColor")
        links.new(density.outputs["Fac"], channels.inputs["Red"])
        links.new(rings.outputs[0], channels.inputs["Green"])
        links.new(silhouette.outputs["Result"], channels.inputs["Blue"])
        debug = nodes.new("ShaderNodeEmission")
        links.new(channels.outputs[0], debug.inputs["Color"])
        debug.inputs["Strength"].default_value = 1.0
        links.new(debug.outputs[0], output.inputs["Surface"])
    elif args.microbubble_debug_alpha:
        debug = nodes.new("ShaderNodeEmission")
        links.new(patch_alpha.outputs[0], debug.inputs["Color"])
        debug.inputs["Strength"].default_value = 1.0
        links.new(debug.outputs[0], output.inputs["Surface"])
    return material


def make_microbubble_dome_material(name):
    material = make_principled_material(
        name,
        (0.84, 0.90, 0.86, 1.0),
        0.075,
        transmission=0.84,
        ior=1.333,
    )
    shader = next(
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    set_input(shader, 0.22, "Coat Weight", "Coat")
    set_input(shader, 0.07, "Coat Roughness")
    return material


def make_water_coverage_material(name):
    """Continuous mid/far-shot water shader driven by a surface coverage mask."""

    material = make_principled_material(
        name,
        (0.82, 0.94, 0.98, 1.0),
        0.03,
        transmission=1.0,
        ior=1.333,
    )
    nodes = material.node_tree.nodes
    shader = next(
        node for node in nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    coverage = nodes.new("ShaderNodeAttribute")
    coverage.attribute_type = "GEOMETRY"
    coverage.attribute_name = "foam_coverage"
    color = nodes.new("ShaderNodeValToRGB")
    color.color_ramp.elements[0].position = 0.0
    color.color_ramp.elements[0].color = (0.82, 0.94, 0.98, 1.0)
    color.color_ramp.elements[1].position = 1.0
    color.color_ramp.elements[1].color = (0.76, 0.82, 0.75, 1.0)
    roughness = nodes.new("ShaderNodeMapRange")
    roughness.clamp = True
    roughness.inputs["From Min"].default_value = 0.0
    roughness.inputs["From Max"].default_value = 1.0
    roughness.inputs["To Min"].default_value = 0.03
    roughness.inputs["To Max"].default_value = 0.16
    transmission = nodes.new("ShaderNodeMapRange")
    transmission.clamp = True
    transmission.inputs["From Min"].default_value = 0.0
    transmission.inputs["From Max"].default_value = 1.0
    transmission.inputs["To Min"].default_value = 1.0
    transmission.inputs["To Max"].default_value = 0.68
    material.node_tree.links.new(coverage.outputs["Fac"], color.inputs["Fac"])
    material.node_tree.links.new(coverage.outputs["Fac"], roughness.inputs["Value"])
    material.node_tree.links.new(coverage.outputs["Fac"], transmission.inputs["Value"])
    material.node_tree.links.new(color.outputs["Color"], shader.inputs["Base Color"])
    material.node_tree.links.new(roughness.outputs["Result"], shader.inputs["Roughness"])
    transmission_socket = shader.inputs.get("Transmission Weight") or shader.inputs.get(
        "Transmission"
    )
    if transmission_socket is not None:
        material.node_tree.links.new(
            transmission.outputs["Result"], transmission_socket
        )
    set_input(shader, 0.12, "Coat Weight", "Coat")
    set_input(shader, 0.08, "Coat Roughness")
    return material


def apply_foam_coverage_to_water(water, cache_path):
    """Attach face and smooth point fields without creating foam geometry."""

    with np.load(cache_path, allow_pickle=False) as cache:
        if (
            str(np.asarray(cache["schema"]))
            != "physx-foamgenerator-water-surface-mask-frame/v2"
        ):
            raise RuntimeError(f"Unsupported water-surface foam mask frame: {cache_path}")
        expected_vertices = int(np.asarray(cache["surface_vertex_count"]))
        expected_faces = int(np.asarray(cache["surface_face_count"]))
        active_faces = np.asarray(cache["active_face_indices"], dtype=np.int64)
        active_coverage = np.asarray(cache["active_face_coverage"], dtype=np.float64)
        active_tau = np.asarray(cache["active_face_optical_depth"], dtype=np.float64)
        surface_sources = len(cache["coverage_sources"])
        free_foam = len(cache["free_foam_id"])
    mesh = water.data
    if len(mesh.vertices) != expected_vertices or len(mesh.polygons) != expected_faces:
        raise RuntimeError(
            f"Coverage field does not match imported surface: {cache_path}"
        )
    loop_totals = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_totals)
    if np.any(loop_totals != 3):
        raise RuntimeError("Coverage field requires an unchanged triangle OBJ")
    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    triangles = loop_vertices.reshape((-1, 3))
    face_area = np.empty(len(mesh.polygons), dtype=np.float64)
    mesh.polygons.foreach_get("area", face_area)
    face_coverage = np.zeros(len(mesh.polygons), dtype=np.float32)
    face_tau = np.zeros(len(mesh.polygons), dtype=np.float32)
    face_coverage[active_faces] = active_coverage.astype(np.float32)
    face_tau[active_faces] = active_tau.astype(np.float32)
    for name, values in (
        ("foam_coverage_face", face_coverage),
        ("foam_optical_depth_face", face_tau),
    ):
        attribute = mesh.attributes.new(name, "FLOAT", "FACE")
        attribute.data.foreach_set("value", values)

    repeated_vertices = triangles.reshape(-1)
    vertex_dual_area = np.bincount(
        repeated_vertices,
        weights=np.repeat(face_area / 3.0, 3),
        minlength=len(mesh.vertices),
    )
    active_vertices = triangles[active_faces].reshape(-1)
    coverage_numerator = np.bincount(
        active_vertices,
        weights=np.repeat(
            face_area[active_faces] * active_coverage / 3.0, 3
        ),
        minlength=len(mesh.vertices),
    )
    tau_numerator = np.bincount(
        active_vertices,
        weights=np.repeat(face_area[active_faces] * active_tau / 3.0, 3),
        minlength=len(mesh.vertices),
    )
    vertex_coverage = np.divide(
        coverage_numerator,
        vertex_dual_area,
        out=np.zeros(len(mesh.vertices), dtype=np.float64),
        where=vertex_dual_area > 0.0,
    ).astype(np.float32)
    vertex_tau = np.divide(
        tau_numerator,
        vertex_dual_area,
        out=np.zeros(len(mesh.vertices), dtype=np.float64),
        where=vertex_dual_area > 0.0,
    ).astype(np.float32)
    for name, values in (
        ("foam_coverage", vertex_coverage),
        ("foam_optical_depth", vertex_tau),
    ):
        attribute = mesh.attributes.new(name, "FLOAT", "POINT")
        attribute.data.foreach_set("value", values)
    mesh.update()
    return {
        "surface_sources": surface_sources,
        "free_foam": free_foam,
        "active_faces": int(len(active_faces)),
        "separate_membrane_geometry": False,
    }


def make_cellular_foam_membrane_material(name, cell_size=0.005):
    """Accepted v10 cellular scattering, generalized to 3D world coordinates."""

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    foam = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(foam, (0.90, 0.94, 0.88, 1.0), "Base Color")
    set_input(foam, 0.38, "Roughness")
    set_input(foam, 1.333, "IOR")
    set_input(foam, 0.10, "Coat Weight", "Coat")
    coverage = nodes.new("ShaderNodeAttribute")
    coverage.attribute_type = "GEOMETRY"
    coverage.attribute_name = "foam_coverage"
    geometry = nodes.new("ShaderNodeNewGeometry")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = 1.0 / cell_size
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.feature = "DISTANCE_TO_EDGE"
    voronoi.voronoi_dimensions = "3D"
    rim = nodes.new("ShaderNodeValToRGB")
    rim.color_ramp.elements[0].position = 0.0
    rim.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    rim.color_ramp.elements[1].position = 0.16
    rim.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    rim_density = nodes.new("ShaderNodeMath")
    rim_density.operation = "MULTIPLY"
    base_density = nodes.new("ShaderNodeMath")
    base_density.operation = "MULTIPLY"
    base_density.inputs[1].default_value = 0.12
    combined = nodes.new("ShaderNodeMath")
    combined.operation = "ADD"
    material.node_tree.links.new(geometry.outputs["Position"], scale.inputs[0])
    material.node_tree.links.new(scale.outputs[0], voronoi.inputs["Vector"])
    material.node_tree.links.new(voronoi.outputs["Distance"], rim.inputs["Fac"])
    material.node_tree.links.new(coverage.outputs["Fac"], rim_density.inputs[0])
    material.node_tree.links.new(rim.outputs["Color"], rim_density.inputs[1])
    material.node_tree.links.new(coverage.outputs["Fac"], base_density.inputs[0])
    material.node_tree.links.new(rim_density.outputs[0], combined.inputs[0])
    material.node_tree.links.new(base_density.outputs[0], combined.inputs[1])
    mix = nodes.new("ShaderNodeMixShader")
    material.node_tree.links.new(combined.outputs[0], mix.inputs[0])
    material.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    material.node_tree.links.new(foam.outputs["BSDF"], mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_plateau_materials():
    micro = make_plateau_film_material("PhysXFoamMicroDensityFilm", 0.28, 0.18)
    cluster = make_plateau_film_material("PhysXFoamClusterFilm", 0.12, 0.58)
    hero = make_plateau_film_material("PhysXFoamHeroThinFilm", 0.045, 0.88)
    gas = make_bubble_material()
    gas.name = "PhysXFoamCellGasBoundary"
    shared = make_plateau_film_material("PhysXFoamSharedMembrane", 0.035, 0.94)
    border = make_principled_material(
        "PhysXFoamPlateauBorder",
        (0.80, 0.91, 0.84, 1.0),
        0.06,
        transmission=0.78,
        ior=1.333,
    )
    node = make_principled_material(
        "PhysXFoamPlateauNode",
        (0.76, 0.89, 0.82, 1.0),
        0.08,
        transmission=0.68,
        ior=1.333,
    )
    return (micro, cluster, hero, gas, shared, border, node)


def raw_to_blender_array(values):
    values = np.asarray(values, dtype=np.float64)
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def create_micro_density_object(cells, lod, material, bin_size):
    selection = np.flatnonzero(lod == 0)
    if not len(selection):
        return None, {
            "micro_density_patches": 0,
            "source_cap_area_m2": 0.0,
            "rendered_cap_area_m2": 0.0,
            "cap_area_residual_m2": 0.0,
        }
    sides = 16
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
    for members in groups.values():
        members = np.asarray(members, dtype=np.int64)
        areas = cells["footprint_area"][members].astype(np.float64)
        total_area = float(areas.sum(dtype=np.float64))
        if total_area <= 0.0:
            continue
        weights = areas / total_area
        center = np.sum(cells["position"][members].astype(np.float64) * weights[:, None], axis=0)
        normal = np.sum(cells["normal"][members].astype(np.float64) * weights[:, None], axis=0)
        normal /= max(np.linalg.norm(normal), 1.0e-12)
        center += float(np.sum(cells["base_height"][members] * weights)) * normal
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
                patch_center = center + min(0.22 * bin_size, 0.55 * radius) * (
                    math.cos(angle) * tangent + math.sin(angle) * bitangent
                )
            base = len(vertices)
            vertices.append(tuple(patch_center))
            for segment in range(sides):
                angle = 2.0 * math.pi * segment / sides
                vertices.append(
                    tuple(
                        patch_center
                        + radius * (
                            math.cos(angle) * tangent + math.sin(angle) * bitangent
                        )
                    )
                )
            for segment in range(sides):
                faces.append((base, base + 1 + segment, base + 1 + (segment + 1) % sides))
            rendered_area += polygon_factor * radius * radius
            patch_count += 1
    mesh = bpy.data.meshes.new("PhysXFoamMicroDensityMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("PhysXFoamMicroDensity", mesh)
    scene.collection.objects.link(obj)
    obj.rotation_euler[0] = math.radians(90.0)
    mesh.materials.append(material)
    source_area = float(cells["footprint_area"][selection].sum(dtype=np.float64))
    return obj, {
        "micro_density_patches": patch_count,
        "source_cap_area_m2": source_area,
        "rendered_cap_area_m2": rendered_area,
        "cap_area_residual_m2": rendered_area - source_area,
        "bin_size_m": bin_size,
        "polygon_sides": sides,
    }


def create_plateau_objects(
    cache_path,
    materials,
    initial_thickness_nm,
    camera,
    resolution,
    micro_pixel_radius,
    hero_pixel_radius,
    micro_bin_size,
    components,
):
    with np.load(cache_path, allow_pickle=False) as cache:
        vertices = np.asarray(cache["vertices"], dtype=np.float32)
        triangles = np.asarray(cache["triangles"], dtype=np.int32)
        source_material = np.asarray(cache["triangle_material"], dtype=np.uint8)
        owners = np.asarray(cache["triangle_owner_id"], dtype=np.uint64)
        cells = np.asarray(cache["cells"])
        films = np.asarray(cache["films"])
        borders = np.asarray(cache["borders"])
        nodes = np.asarray(cache["nodes"])
    cell_by_id = {int(row["id"]): row for row in cells}
    film_by_id = {int(row["id"]): row for row in films}
    border_by_id = {int(row["id"]): row for row in borders}
    node_by_id = {int(row["id"]): row for row in nodes}
    centers = raw_to_blender_array(cells["position"])
    distances = np.linalg.norm(
        centers - np.asarray(camera.location, dtype=np.float64)[None, :], axis=1
    )
    projected_pixels = (
        resolution
        * camera.data.lens
        / camera.data.sensor_width
        * cells["physical_radius"].astype(np.float64)
        / np.maximum(distances, 1.0e-6)
    )
    lod = np.where(
        projected_pixels < micro_pixel_radius,
        0,
        np.where(projected_pixels < hero_pixel_radius, 1, 2),
    ).astype(np.uint8)
    cell_index = {int(value): index for index, value in enumerate(cells["id"])}
    keep = np.ones(len(triangles), dtype=bool)
    if components == "borders-nodes":
        keep &= source_material >= 3
    material_index = np.empty(len(triangles), dtype=np.int32)
    thickness_nm = np.zeros(len(triangles), dtype=np.float32)
    wetness = np.zeros(len(triangles), dtype=np.float32)
    for triangle_index, (kind, owner) in enumerate(zip(source_material, owners)):
        kind, owner = int(kind), int(owner)
        if kind == 0:
            row = cell_by_id[owner]
            wetness[triangle_index] = row["wetness"]
            owner_lod = int(lod[cell_index[owner]])
            if owner_lod == 0:
                keep[triangle_index] = False
            material_index[triangle_index] = owner_lod
            thickness_nm[triangle_index] = initial_thickness_nm
        elif kind == 1:
            row = cell_by_id[owner]
            wetness[triangle_index] = row["wetness"]
            # These are surface-foam outer wet films in an open-air render,
            # not submerged air boundaries.  Reuse the cell's camera LOD film
            # material; the gas-boundary slot is reserved for a later closed
            # water-medium render.
            material_index[triangle_index] = int(lod[cell_index[owner]])
        elif kind == 2:
            row = film_by_id[owner]
            wetness[triangle_index] = row["wetness"]
            thickness_nm[triangle_index] = float(row["thickness"]) * 1.0e9
            material_index[triangle_index] = 4
        elif kind == 3:
            wetness[triangle_index] = border_by_id[owner]["wetness"]
            material_index[triangle_index] = 5
        elif kind == 4:
            wetness[triangle_index] = node_by_id[owner]["wetness"]
            material_index[triangle_index] = 6
        else:
            raise RuntimeError(f"Unknown Plateau triangle material {kind}")
    triangles = triangles[keep]
    material_index = material_index[keep]
    thickness_nm = thickness_nm[keep]
    wetness = wetness[keep]
    mesh = bpy.data.meshes.new("PhysXFoamPlateauMesh")
    mesh.from_pydata(vertices.tolist(), [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new("PhysXFoamPlateau", mesh)
    scene.collection.objects.link(obj)
    obj.rotation_euler[0] = math.radians(90.0)
    for material in materials:
        mesh.materials.append(material)
    for polygon, index in zip(mesh.polygons, material_index):
        polygon.material_index = int(index)
        polygon.use_smooth = int(index) not in (4,)
    thickness_attribute = mesh.attributes.new("ww_thickness_nm", "FLOAT", "FACE")
    wetness_attribute = mesh.attributes.new("ww_wetness", "FLOAT", "FACE")
    if len(thickness_nm):
        thickness_attribute.data.foreach_set("value", thickness_nm)
        wetness_attribute.data.foreach_set("value", wetness)
    micro_object, micro_report = (
        create_micro_density_object(cells, lod, materials[0], micro_bin_size)
        if components == "all"
        else (
            None,
            {
                "micro_density_patches": 0,
                "source_cap_area_m2": 0.0,
                "rendered_cap_area_m2": 0.0,
                "cap_area_residual_m2": 0.0,
            },
        )
    )
    return (obj, micro_object), {
        "cells": len(cells),
        "shared_films": len(films),
        "plateau_borders": len(borders),
        "plateau_nodes": len(nodes),
        "triangles": len(triangles),
        "micro_cells": int(np.count_nonzero(lod == 0)),
        "cluster_cells": int(np.count_nonzero(lod == 1)),
        "hero_cells": int(np.count_nonzero(lod == 2)),
        "minimum_projected_radius_px": float(projected_pixels.min(initial=0.0)),
        "maximum_projected_radius_px": float(projected_pixels.max(initial=0.0)),
        "micro_density": micro_report,
        "components": components,
    }


def create_surface_payload_object(
    cache_path,
    materials,
    patch_key="patches",
    object_name="PhysXFoamSurfacePayload",
):
    with np.load(cache_path, allow_pickle=False) as cache:
        patches = np.asarray(cache[patch_key])
        source_foam = np.asarray(cache["source_foam"])
    source_by_id = {int(row["id"]): row for row in source_foam}
    sides = 16
    vertices, faces, material_indices = [], [], []
    thickness_nm, wetness = [], []
    rendered_area = 0.0
    for patch in patches:
        support = float(patch["support_area"])
        film = float(patch["film_area"])
        if support <= 0.0 or film <= 0.0:
            continue
        polygon_factor = 0.5 * sides * math.sin(2.0 * math.pi / sides)
        coverage_scale = math.sqrt(
            min(film / support, 1.0) * math.pi / polygon_factor
        )
        major = float(patch["major_radius"]) * coverage_scale
        minor = float(patch["minor_radius"]) * coverage_scale
        normal = patch["normal"].astype(np.float64)
        normal /= max(np.linalg.norm(normal), 1.0e-12)
        tangent = patch["principal_direction"].astype(np.float64)
        tangent -= np.dot(tangent, normal) * normal
        tangent /= max(np.linalg.norm(tangent), 1.0e-12)
        bitangent = np.cross(normal, tangent)
        center = patch["position"].astype(np.float64) + 0.00025 * normal
        base = len(vertices)
        vertices.append(tuple(center))
        for segment in range(sides):
            angle = 2.0 * math.pi * segment / sides
            vertices.append(
                tuple(
                    center
                    + major * math.cos(angle) * tangent
                    + minor * math.sin(angle) * bitangent
                )
            )
        source = source_by_id[int(patch["source_foam_id"])]
        drained_thickness_nm = max(
            80.0,
            1200.0
            * math.exp(
                -float(source["age"]) / max(float(source["drainage_time"]), 1.0e-6)
            ),
        )
        for segment in range(sides):
            faces.append((base, base + 1 + segment, base + 1 + (segment + 1) % sides))
            material_indices.append(int(patch["render_class"]))
            thickness_nm.append(drained_thickness_nm)
            wetness.append(float(patch["coverage"]))
        rendered_area += 0.5 * sides * math.sin(2.0 * math.pi / sides) * major * minor
    mesh = bpy.data.meshes.new(f"{object_name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(object_name, mesh)
    scene.collection.objects.link(obj)
    obj.rotation_euler[0] = math.radians(90.0)
    for material in materials:
        mesh.materials.append(material)
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.material_index = int(material_index)
        polygon.use_smooth = False
    thickness_attribute = mesh.attributes.new("ww_thickness_nm", "FLOAT", "FACE")
    if thickness_nm:
        thickness_attribute.data.foreach_set("value", thickness_nm)
    wetness_attribute = mesh.attributes.new("ww_wetness", "FLOAT", "FACE")
    if thickness_nm:
        wetness_attribute.data.foreach_set("value", wetness)
    coverage_attribute = mesh.attributes.new("foam_coverage", "FLOAT", "FACE")
    if thickness_nm:
        coverage_attribute.data.foreach_set("value", wetness)
    source_area = float(patches["film_area"].sum(dtype=np.float64))
    return obj, {
        "patches": len(patches),
        "triangles": len(faces),
        "source_film_area_m2": source_area,
        "rendered_polygon_area_m2": rendered_area,
        "polygon_area_residual_m2": rendered_area - source_area,
        "render_class_counts": {
            "micro": int(np.count_nonzero(patches["render_class"] == 0)),
            "cluster": int(np.count_nonzero(patches["render_class"] == 1)),
            "macro": int(np.count_nonzero(patches["render_class"] == 2)),
        },
    }


def create_foam_membrane_object(cache_path, materials, normal_offset=0.0002):
    with np.load(cache_path, allow_pickle=False) as cache:
        vertices = np.asarray(cache["vertices"], dtype=np.float64).copy()
        triangles = np.asarray(cache["triangles"], dtype=np.int32)
        triangle_normals = np.asarray(cache["triangle_normals"], dtype=np.float64)
        nearest_rows = np.asarray(
            cache["nearest_eligible_patch_rows"], dtype=np.int32
        )
        patches = np.asarray(cache["eligible_patches"])
        source_foam = np.asarray(cache["source_foam"])
        triangle_coverage = (
            np.asarray(cache["triangle_coverage"], dtype=np.float32)
            if "triangle_coverage" in cache.files
            else np.ones(len(triangles), dtype=np.float32)
        )
    source_by_id = {int(row["id"]): row for row in source_foam}
    # The exported membrane is exploded: each triangle owns three vertices, so
    # the audited normal offset cannot drag neighboring unselected water faces.
    for triangle, normal in zip(triangles, triangle_normals):
        vertices[triangle] += normal_offset * normal
    material_indices = np.zeros(len(triangles), dtype=np.int32)
    thickness_nm = np.zeros(len(triangles), dtype=np.float32)
    wetness = np.zeros(len(triangles), dtype=np.float32)
    for index, patch_row in enumerate(nearest_rows):
        patch = patches[int(patch_row)]
        material_indices[index] = min(int(patch["render_class"]), 1)
        wetness[index] = float(patch["coverage"])
        source = source_by_id[int(patch["source_foam_id"])]
        thickness_nm[index] = max(
            80.0,
            1200.0
            * math.exp(
                -float(source["age"])
                / max(float(source["drainage_time"]), 1.0e-6)
            ),
        )
    mesh = bpy.data.meshes.new("PhysXFoamConnectedMembraneMesh")
    mesh.from_pydata(vertices.tolist(), [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new("PhysXFoamConnectedMembrane", mesh)
    scene.collection.objects.link(obj)
    obj.rotation_euler[0] = math.radians(90.0)
    for material in materials:
        mesh.materials.append(material)
    for polygon, material_index in zip(mesh.polygons, material_indices):
        polygon.material_index = int(material_index)
        polygon.use_smooth = True
    thickness_attribute = mesh.attributes.new("ww_thickness_nm", "FLOAT", "FACE")
    if len(triangles):
        thickness_attribute.data.foreach_set("value", thickness_nm)
    wetness_attribute = mesh.attributes.new("ww_wetness", "FLOAT", "FACE")
    if len(triangles):
        wetness_attribute.data.foreach_set("value", wetness)
    coverage_attribute = mesh.attributes.new("foam_coverage", "FLOAT", "FACE")
    if len(triangles):
        coverage_attribute.data.foreach_set("value", triangle_coverage)
    points = vertices[triangles]
    area = float(
        0.5
        * np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            axis=1,
        ).sum(dtype=np.float64)
    )
    return obj, {
        "triangles": len(triangles),
        "represented_area_m2": area,
        "normal_offset_m": normal_offset,
        "coverage": {
            "minimum": float(triangle_coverage.min(initial=0.0)),
            "median": float(np.median(triangle_coverage)) if len(triangle_coverage) else 0.0,
            "maximum": float(triangle_coverage.max(initial=0.0)),
        },
        "material_counts": {
            "micro": int(np.count_nonzero(material_indices == 0)),
            "cluster": int(np.count_nonzero(material_indices == 1)),
        },
    }


def create_point_instancer(
    name, material, subdivisions=1, flip_faces=False, realize_point_attributes=False
):
    mesh = bpy.data.meshes.new(name + "Points")
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.rotation_euler[0] = math.radians(90.0)
    modifier = obj.modifiers.new(name="PointInstances", type="NODES")
    tree = bpy.data.node_groups.new(name + "Geometry", "GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    nodes = tree.nodes
    links = tree.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    position_attributes = {}
    for suffix in ("previous", "current", "next"):
        attribute = nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = "FLOAT_VECTOR"
        attribute.inputs["Name"].default_value = f"position_{suffix}"
        position_attributes[suffix] = attribute
    scale_attributes = {}
    for suffix in ("previous", "current", "next"):
        attribute = nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = "FLOAT_VECTOR"
        attribute.inputs["Name"].default_value = f"scale_{suffix}"
        scale_attributes[suffix] = attribute

    scene_time = nodes.new("GeometryNodeInputSceneTime")
    frame_center = nodes.new("ShaderNodeValue")
    frame_center.name = "MotionBlurFrameCenter"
    frame_delta = nodes.new("ShaderNodeMath")
    frame_delta.operation = "SUBTRACT"
    links.new(scene_time.outputs["Frame"], frame_delta.inputs[0])
    links.new(frame_center.outputs["Value"], frame_delta.inputs[1])
    previous_factor = nodes.new("ShaderNodeMath")
    previous_factor.operation = "MULTIPLY"
    previous_factor.inputs[1].default_value = -1.0
    previous_factor.use_clamp = True
    links.new(frame_delta.outputs[0], previous_factor.inputs[0])
    next_factor = nodes.new("ShaderNodeMath")
    next_factor.operation = "MULTIPLY"
    next_factor.inputs[1].default_value = 1.0
    next_factor.use_clamp = True
    links.new(frame_delta.outputs[0], next_factor.inputs[0])

    def temporal_value(attributes, vector):
        subtract_previous = nodes.new(
            "ShaderNodeVectorMath" if vector else "ShaderNodeMath"
        )
        subtract_previous.operation = "SUBTRACT"
        subtract_next = nodes.new("ShaderNodeVectorMath" if vector else "ShaderNodeMath")
        subtract_next.operation = "SUBTRACT"
        links.new(attributes["previous"].outputs["Attribute"], subtract_previous.inputs[0])
        links.new(attributes["current"].outputs["Attribute"], subtract_previous.inputs[1])
        links.new(attributes["next"].outputs["Attribute"], subtract_next.inputs[0])
        links.new(attributes["current"].outputs["Attribute"], subtract_next.inputs[1])
        scale_previous = nodes.new(
            "ShaderNodeVectorMath" if vector else "ShaderNodeMath"
        )
        scale_previous.operation = "SCALE" if vector else "MULTIPLY"
        scale_next = nodes.new("ShaderNodeVectorMath" if vector else "ShaderNodeMath")
        scale_next.operation = "SCALE" if vector else "MULTIPLY"
        links.new(subtract_previous.outputs[0], scale_previous.inputs[0])
        links.new(previous_factor.outputs[0], scale_previous.inputs[3 if vector else 1])
        links.new(subtract_next.outputs[0], scale_next.inputs[0])
        links.new(next_factor.outputs[0], scale_next.inputs[3 if vector else 1])
        add_previous = nodes.new("ShaderNodeVectorMath" if vector else "ShaderNodeMath")
        add_previous.operation = "ADD"
        links.new(attributes["current"].outputs["Attribute"], add_previous.inputs[0])
        links.new(scale_previous.outputs[0], add_previous.inputs[1])
        add_next = nodes.new("ShaderNodeVectorMath" if vector else "ShaderNodeMath")
        add_next.operation = "ADD"
        links.new(add_previous.outputs[0], add_next.inputs[0])
        links.new(scale_next.outputs[0], add_next.inputs[1])
        return add_next.outputs[0]

    temporal_position = temporal_value(position_attributes, vector=True)
    temporal_scale = temporal_value(scale_attributes, vector=True)
    set_position = nodes.new("GeometryNodeSetPosition")
    links.new(group_input.outputs["Geometry"], set_position.inputs["Geometry"])
    links.new(temporal_position, set_position.inputs["Position"])
    ico_sphere = nodes.new("GeometryNodeMeshIcoSphere")
    ico_sphere.inputs["Radius"].default_value = 1.0
    ico_sphere.inputs["Subdivisions"].default_value = subdivisions
    instance_geometry = ico_sphere.outputs["Mesh"]
    if flip_faces:
        flip = nodes.new("GeometryNodeFlipFaces")
        links.new(instance_geometry, flip.inputs["Mesh"])
        instance_geometry = flip.outputs["Mesh"]
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    # Instance directly on mesh vertices. Mesh-to-Points creates its own
    # built-in radius attribute and would overwrite our millimetre-scale cache.
    links.new(set_position.outputs["Geometry"], instance.inputs["Points"])
    links.new(temporal_scale, instance.inputs["Scale"])
    links.new(instance_geometry, set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], instance.inputs["Instance"])
    if realize_point_attributes:
        realize = nodes.new("GeometryNodeRealizeInstances")
        links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
        links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])
    else:
        links.new(instance.outputs["Instances"], group_output.inputs["Geometry"])
    obj["foam_density_attributes_enabled"] = bool(realize_point_attributes)
    return obj


def create_microbubble_patch_instancer(material):
    """Batch one tangent patch per bound marker and preserve marker attributes."""

    mesh = bpy.data.meshes.new("SurfaceMicrobubblePatchPoints")
    obj = bpy.data.objects.new("SurfaceMicrobubblePatches", mesh)
    scene.collection.objects.link(obj)
    obj.rotation_euler[0] = math.radians(90.0)
    modifier = obj.modifiers.new(name="MicrobubblePatchInstances", type="NODES")
    tree = bpy.data.node_groups.new("SurfaceMicrobubblePatchGeometry", "GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = tree.nodes
    links = tree.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    normal = nodes.new("GeometryNodeInputNamedAttribute")
    normal.data_type = "FLOAT_VECTOR"
    normal.inputs["Name"].default_value = "foam_surface_normal"
    tangent = nodes.new("GeometryNodeInputNamedAttribute")
    tangent.data_type = "FLOAT_VECTOR"
    tangent.inputs["Name"].default_value = "foam_surface_tangent"
    radius = nodes.new("GeometryNodeInputNamedAttribute")
    radius.data_type = "FLOAT"
    radius.inputs["Name"].default_value = "foam_cluster_radius"
    align_normal = nodes.new("FunctionNodeAlignRotationToVector")
    align_normal.axis = "Z"
    align_normal.inputs["Factor"].default_value = 1.0
    links.new(normal.outputs["Attribute"], align_normal.inputs["Vector"])
    align_tangent = nodes.new("FunctionNodeAlignRotationToVector")
    align_tangent.axis = "X"
    align_tangent.pivot_axis = "Z"
    align_tangent.inputs["Factor"].default_value = 1.0
    links.new(align_normal.outputs["Rotation"], align_tangent.inputs["Rotation"])
    links.new(tangent.outputs["Attribute"], align_tangent.inputs["Vector"])
    scale = nodes.new("ShaderNodeCombineXYZ")
    links.new(radius.outputs["Attribute"], scale.inputs["X"])
    links.new(radius.outputs["Attribute"], scale.inputs["Y"])
    links.new(radius.outputs["Attribute"], scale.inputs["Z"])
    patch = nodes.new("GeometryNodeMeshCircle")
    patch.fill_type = "NGON"
    patch.inputs["Vertices"].default_value = 16
    patch.inputs["Radius"].default_value = 1.0
    store_uv = nodes.new("GeometryNodeStoreNamedAttribute")
    store_uv.data_type = "FLOAT_VECTOR"
    store_uv.domain = "CORNER"
    store_uv.inputs["Name"].default_value = "foam_patch_uv"
    patch_position = nodes.new("GeometryNodeInputPosition")
    links.new(patch.outputs["Mesh"], store_uv.inputs["Geometry"])
    links.new(patch_position.outputs["Position"], store_uv.inputs["Value"])
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    links.new(store_uv.outputs["Geometry"], set_material.inputs["Geometry"])
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(group_input.outputs["Geometry"], instance.inputs["Points"])
    links.new(set_material.outputs["Geometry"], instance.inputs["Instance"])
    links.new(align_tangent.outputs["Rotation"], instance.inputs["Rotation"])
    links.new(scale.outputs["Vector"], instance.inputs["Scale"])
    realize = nodes.new("GeometryNodeRealizeInstances")
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])
    return obj


def create_microbubble_dome_instancer(material):
    mesh = bpy.data.meshes.new("SurfaceMicrobubbleDomePoints")
    obj = bpy.data.objects.new("SurfaceMicrobubbleDomes", mesh)
    scene.collection.objects.link(obj)
    obj.rotation_euler[0] = math.radians(90.0)
    modifier = obj.modifiers.new(name="MicrobubbleDomeInstances", type="NODES")
    tree = bpy.data.node_groups.new("SurfaceMicrobubbleDomeGeometry", "GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = tree.nodes
    links = tree.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    normal = nodes.new("GeometryNodeInputNamedAttribute")
    normal.data_type = "FLOAT_VECTOR"
    normal.inputs["Name"].default_value = "foam_surface_normal"
    scale = nodes.new("GeometryNodeInputNamedAttribute")
    scale.data_type = "FLOAT_VECTOR"
    scale.inputs["Name"].default_value = "foam_dome_scale"
    align = nodes.new("FunctionNodeAlignRotationToVector")
    align.axis = "Z"
    align.inputs["Factor"].default_value = 1.0
    links.new(normal.outputs["Attribute"], align.inputs["Vector"])
    sphere = nodes.new("GeometryNodeMeshIcoSphere")
    sphere.inputs["Radius"].default_value = 1.0
    sphere.inputs["Subdivisions"].default_value = 2
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = material
    links.new(sphere.outputs["Mesh"], set_material.inputs["Geometry"])
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(group_input.outputs["Geometry"], instance.inputs["Points"])
    links.new(set_material.outputs["Geometry"], instance.inputs["Instance"])
    links.new(align.outputs["Rotation"], instance.inputs["Rotation"])
    links.new(scale.outputs["Attribute"], instance.inputs["Scale"])
    links.new(instance.outputs["Instances"], group_output.inputs["Geometry"])
    return obj


def read_secondary_cache(cache_path):
    with np.load(cache_path) as cache:
        payload = {
            name: np.asarray(cache[name]).copy()
            for name in ("id", "position", "velocity", "radius", "opacity")
        }
        payload["shape"] = (
            np.asarray(cache["shape"], dtype=np.float32).copy()
            if "shape" in cache
            else np.ones((len(payload["id"]), 3), dtype=np.float32)
        )
        return payload


def read_routed_free_foam_cache(cache_path):
    with np.load(cache_path, allow_pickle=False) as cache:
        return {
            "id": np.asarray(cache["free_foam_id"], dtype=np.int32).copy(),
            "position": np.asarray(
                cache["free_foam_position"], dtype=np.float32
            ).copy(),
            "velocity": np.asarray(
                cache["free_foam_velocity"], dtype=np.float32
            ).copy(),
            "radius": np.asarray(cache["free_foam_radius"], dtype=np.float32).copy(),
            "opacity": np.asarray(
                cache["free_foam_opacity"], dtype=np.float32
            ).copy(),
            "shape": np.asarray(cache["free_foam_shape"], dtype=np.float32).copy(),
        }


def secondary_cache_frames(layer_directory):
    frames = []
    for path in layer_directory.glob("frame_*.npz"):
        try:
            frames.append(int(path.stem.removeprefix("frame_")))
        except ValueError:
            continue
    frames = sorted(set(frames))
    if not frames:
        raise FileNotFoundError(f"No secondary caches in {layer_directory}")
    return frames


def secondary_cache_triplet(secondary_directory, layer, index, available_frames):
    available = set(available_frames)
    if index not in available:
        raise FileNotFoundError(
            secondary_directory / layer / f"frame_{index:04d}.npz"
        )
    minimum = available_frames[0]
    maximum = available_frames[-1]
    previous = index - 1
    following = index + 1
    if previous not in available:
        if index != minimum:
            raise FileNotFoundError(
                secondary_directory / layer / f"frame_{previous:04d}.npz"
            )
        previous = index
    if following not in available:
        if index != maximum:
            raise FileNotFoundError(
                secondary_directory / layer / f"frame_{following:04d}.npz"
            )
        following = index
    return tuple(
        secondary_directory / layer / f"frame_{frame:04d}.npz"
        for frame in (previous, index, following)
    )


def coverage_field_cache_triplet(directory, index, available_frames):
    available = set(available_frames)
    if index not in available:
        raise FileNotFoundError(directory / f"coverage_field_{index:04d}.npz")
    minimum = available_frames[0]
    maximum = available_frames[-1]
    previous = index - 1 if index - 1 in available else index
    following = index + 1 if index + 1 in available else index
    if index != minimum and previous == index:
        raise FileNotFoundError(directory / f"coverage_field_{index - 1:04d}.npz")
    if index != maximum and following == index:
        raise FileNotFoundError(directory / f"coverage_field_{index + 1:04d}.npz")
    return tuple(
        directory / f"coverage_field_{frame:04d}.npz"
        for frame in (previous, index, following)
    )


def sample_secondary(cache, union_ids):
    indices = np.searchsorted(cache["id"], union_ids)
    present = indices < len(cache["id"])
    present_indices = np.flatnonzero(present)
    present[present_indices] &= cache["id"][indices[present_indices]] == union_ids[present_indices]
    positions = np.zeros((len(union_ids), 3), dtype=np.float32)
    velocities = np.zeros_like(positions)
    scales = np.zeros((len(union_ids), 3), dtype=np.float32)
    positions[present] = cache["position"][indices[present]]
    velocities[present] = cache["velocity"][indices[present]]
    effective = cache["radius"] * np.sqrt(np.clip(cache["opacity"], 0.0, 1.0))
    scales[present] = (
        effective[indices[present], None] * cache["shape"][indices[present]]
    )
    return positions, velocities, scales, present


def foam_local_render_fields(cache):
    """Estimate render-only crowding and thickness without changing physics."""

    positions = np.asarray(cache["position"], dtype=np.float64)
    count = len(positions)
    neighbors = np.zeros(count, dtype=np.float64)
    support_squared = FOAM_DENSITY_SUPPORT_M * FOAM_DENSITY_SUPPORT_M
    block_size = 384
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        delta = positions[start:stop, None, :] - positions[None, :, :]
        distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
        neighbors[start:stop] = np.count_nonzero(
            distance_squared <= support_squared, axis=1
        ) - 1
    density = np.clip(
        (neighbors - FOAM_DENSITY_ONSET_NEIGHBORS)
        / (FOAM_DENSITY_FULL_NEIGHBORS - FOAM_DENSITY_ONSET_NEIGHBORS),
        0.0,
        1.0,
    )
    density = density * density * (3.0 - 2.0 * density)
    effective_radius = np.asarray(cache["radius"], dtype=np.float64) * np.sqrt(
        np.clip(np.asarray(cache["opacity"], dtype=np.float64), 0.0, 1.0)
    )
    thickness = np.clip(
        (effective_radius - FOAM_THICKNESS_ONSET_M)
        / (FOAM_THICKNESS_FULL_M - FOAM_THICKNESS_ONSET_M),
        0.0,
        1.0,
    )
    return density.astype(np.float32), thickness.astype(np.float32)


def stable_marker_seed(ids):
    values = np.asarray(ids, dtype=np.uint64).copy()
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return ((values & np.uint64(0x00FFFFFF)).astype(np.float64) / 16777216.0).astype(
        np.float32
    )


def write_point_attribute(mesh, name, data_type, values, field):
    attribute = mesh.attributes.new(name, data_type, "POINT")
    values = np.asarray(values)
    if len(values):
        attribute.data.foreach_set(field, values.reshape(-1))


def update_microbubble_patch_objects(patch_obj, dome_obj, cache, binding):
    rows = np.asarray(binding["source_row"], dtype=np.int64)
    ids = np.asarray(cache["id"], dtype=np.int64)[rows]
    if not np.array_equal(ids, np.asarray(binding["id"], dtype=np.int64)):
        raise RuntimeError("Foam binding IDs do not match native cache rows")
    positions = np.asarray(binding["position"], dtype=np.float64)
    normals = np.asarray(binding["normal"], dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1.0e-12)[:, None]
    velocity = np.asarray(cache["velocity"], dtype=np.float32)[rows]
    opacity = np.clip(
        np.asarray(binding["surface_opacity"], dtype=np.float64), 0.0, 1.0
    ).astype(np.float32)
    density = np.asarray(binding["density"], dtype=np.float32)
    cluster_radius = np.asarray(binding["cluster_radius"], dtype=np.float32)
    cell_scale = np.asarray(binding["cell_scale"], dtype=np.float32)
    tangents = np.asarray(binding["tangent"], dtype=np.float64)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1), 1.0e-12)[:, None]
    seed = stable_marker_seed(ids)
    patch_positions = positions + normals * 0.00025
    mesh = patch_obj.data
    mesh.clear_geometry()
    mesh.from_pydata(patch_positions.tolist(), [], [])
    write_point_attribute(mesh, "foam_surface_normal", "FLOAT_VECTOR", normals, "vector")
    write_point_attribute(mesh, "foam_surface_tangent", "FLOAT_VECTOR", tangents, "vector")
    write_point_attribute(mesh, "foam_native_velocity", "FLOAT_VECTOR", velocity, "vector")
    write_point_attribute(mesh, "foam_cluster_radius", "FLOAT", cluster_radius, "value")
    write_point_attribute(mesh, "foam_local_density", "FLOAT", density, "value")
    write_point_attribute(mesh, "foam_cell_scale", "FLOAT", cell_scale, "value")
    write_point_attribute(mesh, "foam_marker_opacity", "FLOAT", opacity, "value")
    write_point_attribute(mesh, "foam_marker_seed", "FLOAT", seed, "value")
    write_point_attribute(mesh, "foam_stable_id", "INT", ids.astype(np.int32), "value")
    mesh.update()

    hero_count = 0
    if dome_obj is not None:
        hero = (density >= 0.45) & (seed < 0.10) & (opacity > 0.05)
        hero_radius = cluster_radius[hero] * 0.32 * np.sqrt(opacity[hero])
        hero_positions = (
            positions[hero] + normals[hero] * (0.18 * hero_radius)[:, None]
        )
        hero_scale = np.column_stack(
            (hero_radius, hero_radius, 0.42 * hero_radius)
        ).astype(np.float32)
        dome_mesh = dome_obj.data
        dome_mesh.clear_geometry()
        dome_mesh.from_pydata(hero_positions.tolist(), [], [])
        write_point_attribute(
            dome_mesh,
            "foam_surface_normal",
            "FLOAT_VECTOR",
            normals[hero],
            "vector",
        )
        write_point_attribute(
            dome_mesh, "foam_dome_scale", "FLOAT_VECTOR", hero_scale, "vector"
        )
        write_point_attribute(
            dome_mesh,
            "foam_stable_id",
            "INT",
            ids[hero].astype(np.int32),
            "value",
        )
        dome_mesh.update()
        hero_count = int(np.count_nonzero(hero))
    return {
        "bound_marker_clusters": int(len(rows)),
        "hero_domes": hero_count,
        "density_nonzero": int(np.count_nonzero(density > 0.0)),
        "density_high": int(np.count_nonzero(density >= 0.75)),
        "cluster_radius_m": {
            "minimum": float(cluster_radius.min()) if len(cluster_radius) else 0.0,
            "median": float(np.median(cluster_radius)) if len(cluster_radius) else 0.0,
            "maximum": float(cluster_radius.max(initial=0.0)),
        },
        "stable_id_seeded": True,
        "native_velocity_consumed_by_binding": True,
        "density_source": "audited temporally filtered continuous Gaussian kernel",
        "normal_offset_m": 0.00025,
        "patch_polygon_vertices": 16,
    }


def select_secondary_rows(cache, rows):
    rows = np.asarray(rows, dtype=np.int64)
    return {name: np.asarray(values)[rows].copy() for name, values in cache.items()}


def evaluated_float_attribute_range(obj, name):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        attribute = mesh.attributes.get(name)
        if attribute is None:
            return {"present": False, "count": 0, "minimum": 0.0, "maximum": 0.0}
        values = np.empty(len(attribute.data), dtype=np.float32)
        attribute.data.foreach_get("value", values)
        return {
            "present": True,
            "count": int(len(values)),
            "minimum": float(values.min()) if len(values) else 0.0,
            "maximum": float(values.max()) if len(values) else 0.0,
        }
    finally:
        evaluated.to_mesh_clear()


def evaluated_vector_attribute_range(obj, name):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        attribute = mesh.attributes.get(name)
        if attribute is None:
            return {"present": False, "count": 0}
        values = np.empty((len(attribute.data), 3), dtype=np.float32)
        attribute.data.foreach_get("vector", values.reshape(-1))
        return {
            "present": True,
            "count": int(len(values)),
            "minimum": values.min(axis=0).tolist() if len(values) else [0.0] * 3,
            "maximum": values.max(axis=0).tolist() if len(values) else [0.0] * 3,
        }
    finally:
        evaluated.to_mesh_clear()


def update_point_instancer_from_caches(obj, caches, frame_index, fps):
    union_ids = np.unique(np.concatenate([cache["id"] for cache in caches]))
    sampled = [sample_secondary(cache, union_ids) for cache in caches]
    positions = [item[0] for item in sampled]
    velocities = [item[1] for item in sampled]
    scales = [item[2] for item in sampled]
    present = [item[3] for item in sampled]
    # Give births and deaths a continuous subframe trajectory. Radius remains
    # zero outside the active interval so topology is stable during the shutter.
    missing_current = ~present[1]
    from_previous = missing_current & present[0]
    positions[1][from_previous] = (
        positions[0][from_previous] + velocities[0][from_previous] / fps
    )
    from_next = missing_current & ~present[0] & present[2]
    positions[1][from_next] = positions[2][from_next] - velocities[2][from_next] / fps
    positions[0][~present[0]] = positions[1][~present[0]] - velocities[1][~present[0]] / fps
    positions[2][~present[2]] = positions[1][~present[2]] + velocities[1][~present[2]] / fps

    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(positions[1].tolist(), [], [])
    for suffix, values in zip(("previous", "current", "next"), positions):
        attribute = mesh.attributes.new(f"position_{suffix}", "FLOAT_VECTOR", "POINT")
        if len(values):
            attribute.data.foreach_set("vector", values.reshape(-1))
    for suffix, values in zip(("previous", "current", "next"), scales):
        attribute = mesh.attributes.new(f"scale_{suffix}", "FLOAT_VECTOR", "POINT")
        if len(values):
            attribute.data.foreach_set("vector", values.reshape(-1))
    if bool(obj.get("foam_density_attributes_enabled", False)):
        density_cache, thickness_cache = foam_local_render_fields(caches[1])
        current_indices = np.searchsorted(caches[1]["id"], union_ids)
        current_present = current_indices < len(caches[1]["id"])
        present_rows = np.flatnonzero(current_present)
        current_present[present_rows] &= (
            caches[1]["id"][current_indices[present_rows]] == union_ids[present_rows]
        )
        density = np.zeros(len(union_ids), dtype=np.float32)
        thickness = np.zeros(len(union_ids), dtype=np.float32)
        density[current_present] = density_cache[current_indices[current_present]]
        thickness[current_present] = thickness_cache[current_indices[current_present]]
        for name, values in (
            ("foam_local_density", density),
            ("foam_particle_thickness", thickness),
        ):
            attribute = mesh.attributes.new(name, "FLOAT", "POINT")
            if len(values):
                attribute.data.foreach_set("value", values)
    mesh.update()
    obj.modifiers["PointInstances"].node_group.nodes[
        "MotionBlurFrameCenter"
    ].outputs["Value"].default_value = frame_index
    return int(np.count_nonzero(present[1]))


def update_point_instancer(obj, cache_paths, frame_index, fps):
    return update_point_instancer_from_caches(
        obj,
        [read_secondary_cache(path) for path in cache_paths],
        frame_index,
        fps,
    )


args = parse_args()
surface_directory = args.surface_directory.resolve()
physics_report_path = args.physics_report.resolve()
physics_report = json.loads(physics_report_path.read_text(encoding="utf-8"))
output_directory = args.output_directory.resolve()
samples = args.samples or args.legacy_samples or 128
resolution = args.resolution or args.legacy_resolution or 520
requested_frames = (
    args.frames if args.frames is not None else args.legacy_frames
)
secondary_directory = (
    args.secondary_directory.resolve() if args.secondary_directory else None
)
foam_binding_directory = (
    args.foam_binding_directory.resolve() if args.foam_binding_directory else None
)
plateau_directory = (
    args.plateau_directory.resolve() if args.plateau_directory else None
)
foam_payload_directory = (
    args.foam_payload_directory.resolve() if args.foam_payload_directory else None
)
foam_membrane_directory = (
    args.foam_membrane_directory.resolve() if args.foam_membrane_directory else None
)
foam_coverage_field_directory = (
    args.foam_coverage_field_directory.resolve()
    if args.foam_coverage_field_directory
    else None
)
volume_directory = args.volume_directory.resolve() if args.volume_directory else None
output_fps = float(physics_report.get("output_fps", 30.0))
sphere_radius = float(physics_report.get("impactor", {}).get("radius", 0.08))
if samples < 1 or resolution < 1 or output_fps <= 0:
    raise ValueError("Samples and resolution must be positive")
if min(args.volume_sphere_clearance, args.volume_sphere_falloff) < 0:
    raise ValueError("Volume sphere clearance and falloff cannot be negative")
if args.motion_blur_shutter < 0:
    raise ValueError("Motion-blur shutter cannot be negative")
if not (
    0.0 < args.plateau_micro_pixel_radius < args.plateau_hero_pixel_radius
    and args.plateau_micro_bin_size > 0.0
):
    raise ValueError("Plateau LOD thresholds and bin size are invalid")
if (args.camera_location is None) != (args.camera_target is None):
    raise ValueError("Custom camera location and target must be supplied together")
if args.foam_surface_representation != "marker-spheres":
    if secondary_directory is None or foam_binding_directory is None:
        raise ValueError(
            "Microbubble patch modes require both secondary and foam binding directories"
        )
    if foam_coverage_field_directory is not None or foam_membrane_directory is not None:
        raise ValueError(
            "Microbubble patches are mutually exclusive with coverage and connected membrane"
        )
output_directory.mkdir(parents=True, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.cycles.seed = args.seed
if hasattr(scene.cycles, "use_animated_seed"):
    scene.cycles.use_animated_seed = False
scene.cycles.max_bounces = 12
scene.cycles.transmission_bounces = 12
scene.render.use_motion_blur = args.motion_blur_shutter > 0
scene.render.motion_blur_shutter = args.motion_blur_shutter
render_device = {"request": args.render_device, "backend": "CPU", "devices": []}
if args.render_device == "cpu":
    scene.cycles.device = "CPU"
else:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    selected_backend = None
    selected_devices = []
    errors = []
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            candidates = [
                device for device in preferences.devices if device.type == backend
            ]
            if not candidates:
                errors.append(f"{backend}: no device of the requested backend type")
                continue
            for device in preferences.devices:
                device.use = device in candidates
            selected_backend = backend
            selected_devices = [device.name for device in candidates]
            break
        except (KeyError, TypeError, RuntimeError) as error:
            errors.append(f"{backend}: {error}")
    if selected_backend is None:
        raise RuntimeError(
            "GPU rendering was required, but Cycles exposed no usable OPTIX/CUDA device: "
            + "; ".join(errors)
        )
    scene.cycles.device = "GPU"
    render_device = {
        "request": args.render_device,
        "backend": selected_backend,
        "devices": selected_devices,
    }
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.view_settings.look = "AgX - Medium High Contrast"

world = scene.world or bpy.data.worlds.new("SwampSplashsurfWorld")
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
world_output = nodes.new("ShaderNodeOutputWorld")
background = nodes.new("ShaderNodeBackground")
background.inputs["Strength"].default_value = args.hdri_strength
environment = nodes.new("ShaderNodeTexEnvironment")
environment.image = bpy.data.images.load(str(args.hdri), check_existing=True)
world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])

foam_coverage_manifest_path = (
    foam_coverage_field_directory / "manifest.json"
    if foam_coverage_field_directory
    else None
)
foam_coverage_audit_path = (
    foam_coverage_field_directory / "audit_report.json"
    if foam_coverage_field_directory
    else None
)
foam_coverage_manifest = (
    json.loads(foam_coverage_manifest_path.read_text(encoding="utf-8"))
    if foam_coverage_manifest_path is not None
    and foam_coverage_manifest_path.is_file()
    else None
)
foam_coverage_audit = (
    json.loads(foam_coverage_audit_path.read_text(encoding="utf-8"))
    if foam_coverage_audit_path is not None and foam_coverage_audit_path.is_file()
    else None
)
if foam_coverage_field_directory is not None:
    if (foam_coverage_field_directory / "OBSOLETE.json").is_file():
        raise RuntimeError("Coverage-field directory is marked obsolete")
    if (
        foam_coverage_manifest is None
        or foam_coverage_manifest.get("complete") is not True
        or foam_coverage_manifest.get("schema")
        != "physx-foamgenerator-water-surface-mask-prototype/v2"
        or foam_coverage_manifest.get("product")
        != "physx_foamgenerator_water_surface_mask_prototype"
    ):
        raise RuntimeError("Coverage-field directory is missing a complete manifest")
    coverage_semantics = foam_coverage_manifest.get("semantics", {})
    coverage_transport = foam_coverage_manifest.get("transport", {})
    if (
        coverage_semantics.get("maturity")
        != "prototype; not a mature flow-following foam system"
        or coverage_semantics.get("coverage_integral_is_physical_area") is not False
        or coverage_transport.get("finite_difference_velocity_used") is not False
        or not coverage_transport.get("source_normalization", "").startswith(
            "kernel times actual face area"
        )
    ):
        raise RuntimeError("Coverage-field semantics do not match the audited prototype")
    if foam_coverage_audit is None or foam_coverage_audit.get("valid") is not True:
        raise RuntimeError("Coverage-field directory has not passed its independent audit")
    if foam_coverage_audit.get("manifest_sha256") != sha256_file(
        foam_coverage_manifest_path
    ):
        raise RuntimeError("Coverage-field audit does not match its manifest")

foam_binding_manifest_path = (
    foam_binding_directory / "manifest.json" if foam_binding_directory else None
)
foam_binding_manifest = (
    json.loads(foam_binding_manifest_path.read_text(encoding="utf-8"))
    if foam_binding_manifest_path is not None and foam_binding_manifest_path.is_file()
    else None
)
foam_binding_audit_path = (
    foam_binding_directory / "audit_report.json" if foam_binding_directory else None
)
foam_binding_audit = (
    json.loads(foam_binding_audit_path.read_text(encoding="utf-8"))
    if foam_binding_audit_path is not None and foam_binding_audit_path.is_file()
    else None
)
foam_binding_samples = (
    {
        int(sample["output_frame"]): sample
        for sample in foam_binding_manifest.get("samples", [])
    }
    if foam_binding_manifest is not None
    else {}
)
if args.foam_surface_representation != "marker-spheres":
    if (
        foam_binding_manifest is None
        or foam_binding_manifest.get("complete") is not True
        or foam_binding_manifest.get("product")
        != "physx_foamgenerator_temporal_surface_binding"
    ):
        raise RuntimeError("Microbubble patches require a completed v2 temporal surface binding")
    if (
        foam_binding_audit is None
        or foam_binding_audit.get("valid") is not True
        or foam_binding_audit.get("manifest_sha256")
        != sha256_file(foam_binding_manifest_path)
    ):
        raise RuntimeError("Microbubble patches require a passing matching temporal-binding audit")

water_material = (
    make_water_coverage_material("SplashsurfWaterWithFoamCoverage")
    if foam_coverage_field_directory is not None
    else make_principled_material(
        "SplashsurfWaterOpenSurface",
        (0.82, 0.94, 0.98, 1.0),
        0.03,
        transmission=1.0,
        ior=1.333,
    )
)
plateau_manifest_path = plateau_directory / "manifest.json" if plateau_directory else None
plateau_manifest = (
    json.loads(plateau_manifest_path.read_text(encoding="utf-8"))
    if plateau_manifest_path is not None and plateau_manifest_path.is_file()
    else None
)
if plateau_directory is not None:
    if plateau_manifest is None or plateau_manifest.get("complete") is not True:
        raise RuntimeError("Plateau directory is missing a complete manifest")
    plateau_materials = make_plateau_materials()
    initial_plateau_thickness_nm = (
        float(
            plateau_manifest["configuration"]["plateau_model"][
                "initial_film_thickness_m"
            ]
        )
        * 1.0e9
    )
else:
    plateau_materials = None
    initial_plateau_thickness_nm = None
foam_payload_manifest_path = (
    foam_payload_directory / "manifest.json" if foam_payload_directory else None
)
foam_payload_manifest = (
    json.loads(foam_payload_manifest_path.read_text(encoding="utf-8"))
    if foam_payload_manifest_path is not None and foam_payload_manifest_path.is_file()
    else None
)
if foam_payload_directory is not None:
    if foam_payload_manifest is None or foam_payload_manifest.get("complete") is not True:
        raise RuntimeError("Surface-payload directory is missing a complete manifest")
    foam_payload_materials = (
        make_plateau_film_material("PhysXFoamPayloadMicro", 0.30, 0.12),
        make_plateau_film_material("PhysXFoamPayloadCluster", 0.16, 0.42),
        make_plateau_film_material("PhysXFoamPayloadMacro", 0.055, 0.82),
    )
else:
    foam_payload_materials = None
foam_membrane_manifest_path = (
    foam_membrane_directory / "manifest.json" if foam_membrane_directory else None
)
foam_membrane_manifest = (
    json.loads(foam_membrane_manifest_path.read_text(encoding="utf-8"))
    if foam_membrane_manifest_path is not None
    and foam_membrane_manifest_path.is_file()
    else None
)
surface_foam_inputs = tuple(
    directory
    for directory in (
        foam_payload_directory,
        foam_membrane_directory,
        foam_coverage_field_directory,
    )
    if directory is not None
)
if len(surface_foam_inputs) > 1:
    raise RuntimeError(
        "Surface payload, connected membrane, and coverage field are mutually exclusive"
    )
if foam_membrane_directory is not None:
    obsolete_membrane = foam_membrane_directory / "OBSOLETE.json"
    if obsolete_membrane.is_file() and not args.allow_obsolete_foam_membrane:
        raise RuntimeError(
            "Connected membrane is marked OBSOLETE. Use the audited coverage field, "
            "or pass --allow-obsolete-foam-membrane only for an explicit A/B regression."
        )
    if foam_membrane_manifest is None or foam_membrane_manifest.get("complete") is not True:
        raise RuntimeError("Foam-membrane directory is missing a complete manifest")
    if args.foam_membrane_shader == "wet-film":
        foam_membrane_materials = (
            make_wet_foam_membrane_material("PhysXFoamMembraneWetMicro"),
            make_wet_foam_membrane_material("PhysXFoamMembraneWetCluster"),
        )
        foam_membrane_residual_materials = (
            make_wet_foam_membrane_material("PhysXFoamResidualWetMicro"),
            make_wet_foam_membrane_material("PhysXFoamResidualWetCluster"),
            make_wet_foam_membrane_material("PhysXFoamResidualWetMacro"),
        )
    else:
        foam_membrane_materials = (
            make_cellular_foam_membrane_material("PhysXFoamMembraneCellularMicro"),
            make_cellular_foam_membrane_material("PhysXFoamMembraneCellularCluster"),
        )
        foam_membrane_residual_materials = (
            make_plateau_film_material("PhysXFoamResidualMicro", 0.30, 0.12),
            make_plateau_film_material("PhysXFoamResidualCluster", 0.16, 0.42),
            make_plateau_film_material("PhysXFoamResidualMacro", 0.055, 0.82),
        )
else:
    foam_membrane_materials = None
    foam_membrane_residual_materials = None
# The terrain clip produces an open free-surface sheet, not a watertight water
# volume. Volume Absorption on this geometry gives undefined path lengths and
# causes dark temporal flashes, so absorption is intentionally omitted.

sphere_material = bpy.data.materials.new("PhysXImpactorOrange")
sphere_material.use_nodes = True
nodes = sphere_material.node_tree.nodes
nodes.clear()
material_output = nodes.new("ShaderNodeOutputMaterial")
principled = nodes.new("ShaderNodeBsdfPrincipled")
set_input(principled, (0.95, 0.18, 0.035, 1.0), "Base Color")
set_input(principled, 0.24, "Roughness")
sphere_material.node_tree.links.new(
    principled.outputs["BSDF"], material_output.inputs["Surface"]
)

volume_material_result = (
    make_volume_material(
        args.volume_absorption_density,
        args.volume_scatter_density,
        sphere_radius,
        args.volume_sphere_clearance,
        args.volume_sphere_falloff,
    )
    if volume_directory is not None
    else None
)
volume_material = volume_material_result[0] if volume_material_result else None
volume_sphere_center = volume_material_result[1] if volume_material_result else None
secondary_objects = {}
foam_material = None
microbubble_patch_object = None
microbubble_dome_object = None
microbubble_free_object = None
if secondary_directory is not None:
    spray_material = make_principled_material(
        "SecondarySprayWater",
        (0.88, 0.96, 1.0, 1.0),
        0.06,
        transmission=1.0,
        ior=1.333,
    )
    foam_material = (
        make_wet_foam_particle_material("SecondaryFoamWetFilm")
        if args.foam_particle_shader == "wet-film"
        else make_principled_material(
            "SecondaryFoamOpaqueDiagnostic",
            (0.92, 0.96, 0.93, 1.0),
            0.62,
            transmission=0.0,
            ior=1.333,
        )
    )
    bubble_material = make_bubble_material()
    all_secondary_objects = {
        "spray": create_point_instancer(
            "SecondarySpray", spray_material, subdivisions=1
        ),
        "foam": create_point_instancer(
            "SecondaryFoam",
            foam_material,
            subdivisions=1,
            realize_point_attributes=args.foam_particle_shader == "wet-film",
        ),
        # Inward faces represent an air boundary viewed from the surrounding
        # water medium while retaining the water IOR on the glass closure.
        "bubbles": create_point_instancer(
            "SecondaryBubbles",
            bubble_material,
            subdivisions=3 if args.camera == "underwater" else 2,
            flip_faces=True,
        ),
    }
    requested_secondary_layers = set(args.secondary_layers)
    secondary_objects = {
        layer: obj
        for layer, obj in all_secondary_objects.items()
        if layer in requested_secondary_layers
    }
    for layer, obj in all_secondary_objects.items():
        if layer not in requested_secondary_layers:
            bpy.data.objects.remove(obj, do_unlink=True)
    if args.foam_surface_representation != "marker-spheres":
        original_foam_object = secondary_objects.pop("foam", None)
        if original_foam_object is not None:
            bpy.data.objects.remove(original_foam_object, do_unlink=True)
        if "foam" in requested_secondary_layers:
            microbubble_patch_object = create_microbubble_patch_instancer(
                make_microbubble_patch_material("SurfaceMicrobubbleClusterPatch")
            )
            if args.foam_surface_representation == "microbubble-patches-hero":
                microbubble_dome_object = create_microbubble_dome_instancer(
                    make_microbubble_dome_material("SurfaceMicrobubbleSparseDomes")
                )
            microbubble_free_object = create_point_instancer(
                "FreeFoamWetBubbles", foam_material, subdivisions=2
            )
routed_free_foam_object = None
if foam_coverage_field_directory is not None:
    # The original foam layer contains both bound and unbound parcels. Replace
    # it with the coverage product's exclusive free route to prevent double
    # counting while preserving spray and bubbles from the native cache.
    original_foam_object = secondary_objects.pop("foam", None)
    if original_foam_object is not None:
        bpy.data.objects.remove(original_foam_object, do_unlink=True)
    if "foam" in set(args.secondary_layers):
        if foam_material is None:
            foam_material = make_principled_material(
                "SecondaryFoam",
                (0.92, 0.96, 0.93, 1.0),
                0.62,
                transmission=0.0,
                ior=1.333,
            )
        routed_free_foam_object = create_point_instancer(
            "RoutedFreeFoam",
            foam_material,
            subdivisions=1,
            realize_point_attributes=args.foam_particle_shader == "wet-film",
        )
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.08)
sphere = bpy.context.active_object
sphere.name = "PhysXImpactor"
sphere.data.materials.append(sphere_material)
for polygon in sphere.data.polygons:
    polygon.use_smooth = True

camera_data = bpy.data.cameras.new("SplashsurfSequenceCamera")
camera = bpy.data.objects.new("SplashsurfSequenceCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_presets = {
    "overview": {
        "location": (0.12, 1.15, -1.57),
        "target": (-0.73, -1.41099, 0.78),
        "lens": 40.0,
    },
    "waterline": {
        "location": (-0.02, -1.335, -0.05),
        "target": (-0.73, -1.392, 0.78),
        "lens": 52.0,
    },
    "underwater": {
        "location": (-0.32, -1.445, 0.25),
        "target": (-0.73, -1.410, 0.78),
        "lens": 55.0,
    },
}
camera_settings = camera_presets[args.camera].copy()
if args.camera_location is not None:
    camera_settings["location"] = tuple(args.camera_location)
    camera_settings["target"] = tuple(args.camera_target)
if args.camera_lens is not None:
    camera_settings["lens"] = args.camera_lens
camera.location = isaac_to_blender(camera_settings["location"])
target = isaac_to_blender(camera_settings["target"])
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
camera_data.lens = camera_settings["lens"]
camera_data.sensor_width = 24.0
camera_data.clip_start = 0.005

metrics = {int(row["output_frame"]): row for row in physics_report["metrics"]}
frame_count = int(physics_report["frames"])
if args.frame_list is not None:
    frame_indices = sorted(set(args.frame_list))
    invalid_frames = [index for index in frame_indices if not 0 <= index < frame_count]
    if invalid_frames:
        raise ValueError(f"Frame indices outside the physics report: {invalid_frames}")
elif requested_frames is not None:
    frame_indices = list(range(min(frame_count, requested_frames)))
else:
    frame_indices = list(range(frame_count))

secondary_available_frames = {
    layer: secondary_cache_frames(secondary_directory / layer)
    for layer in secondary_objects
}
coverage_available_frames = (
    sorted(int(frame) for frame in foam_coverage_manifest["frames"])
    if foam_coverage_manifest is not None
    else []
)
if foam_coverage_field_directory is not None:
    missing_coverage_frames = sorted(set(frame_indices) - set(coverage_available_frames))
    if missing_coverage_frames:
        raise FileNotFoundError(
            f"Coverage field is missing requested frames: {missing_coverage_frames}"
        )
if args.foam_surface_representation != "marker-spheres":
    missing_binding_frames = sorted(set(frame_indices) - set(foam_binding_samples))
    if missing_binding_frames:
        raise FileNotFoundError(
            f"Foam surface binding is missing requested frames: {missing_binding_frames}"
        )

surface_manifest_path = surface_directory.parent / "splashsurf_manifest.json"
surface_manifest_configuration = (
    json.loads(surface_manifest_path.read_text(encoding="utf-8")).get("configuration")
    if surface_manifest_path.is_file()
    else None
)
secondary_manifest_path = (
    secondary_directory / "secondary_manifest.json"
    if secondary_directory is not None
    else None
)
volume_manifest_path = (
    volume_directory / "water_volume_manifest.json"
    if volume_directory is not None
    else None
)
configuration = {
    "surface_directory": str(surface_directory),
    # Hash only geometry-affecting configuration. The batch builder updates its
    # completion state while a concurrent renderer is running.
    "surface_configuration_sha256": (
        sha256_json(surface_manifest_configuration)
        if surface_manifest_configuration is not None
        else None
    ),
    "physics_report": str(physics_report_path),
    "physics_report_sha256": sha256_file(physics_report_path),
    "render_script_sha256": sha256_file(Path(__file__)),
    "blender_version": bpy.app.version_string,
    "cycles_device": render_device,
    "samples": samples,
    "resolution": resolution,
    "frame_indices": frame_indices,
    "seed": args.seed,
    "max_bounces": 12,
    "transmission_bounces": 12,
    "denoising": True,
    "motion_blur": {
        "enabled": scene.render.use_motion_blur,
        "shutter_frames": args.motion_blur_shutter,
        "stable_id_temporal_sampling": secondary_directory is not None,
    },
    "camera": {
        "preset": args.camera,
        "location_isaac": list(camera_settings["location"]),
        "target_isaac": list(camera_settings["target"]),
        "lens_mm": camera_settings["lens"],
        "sensor_width_mm": camera_data.sensor_width,
    },
    "view_transform_look": "AgX - Medium High Contrast",
    "hdri": str(args.hdri.resolve()),
    "hdri_sha256": sha256_file(args.hdri),
    "hdri_strength": args.hdri_strength,
    "water_material": {
        "geometry": "open-free-surface",
        "base_color": [0.82, 0.94, 0.98, 1.0],
        "roughness": 0.03,
        "ior": 1.333,
        "transmission": 1.0,
        "volume_absorption": False,
        "foam_coverage_attribute": (
            "foam_coverage" if foam_coverage_field_directory is not None else None
        ),
        "foam_coverage_policy": (
            "continuous point-domain scalar on the complete water mesh; no separate membrane"
            if foam_coverage_field_directory is not None
            else None
        ),
    },
    "closed_medium": {
        "directory": str(volume_directory) if volume_directory else None,
        "configuration_sha256": manifest_configuration_hash(volume_manifest_path),
        "absorption_density": args.volume_absorption_density,
        "scatter_density": args.volume_scatter_density,
        "surface_closure": None,
        "sphere_density_exclusion": {
            "radius": sphere_radius,
            "clearance": args.volume_sphere_clearance,
            "falloff": args.volume_sphere_falloff,
        },
    },
    "secondary_particles": {
        "directory": str(secondary_directory) if secondary_directory else None,
        "configuration_sha256": manifest_configuration_hash(
            secondary_manifest_path
        ),
        "layers": sorted(secondary_objects),
        "visibility_filter_only": secondary_directory is not None,
        "cache_frame_ranges": {
            layer: [frames[0], frames[-1]]
            for layer, frames in secondary_available_frames.items()
        },
        "temporal_boundary_mode": "duplicate-first-or-last-real-cache",
        "native_foam_layer_replaced_by_exclusive_free_route": (
            foam_coverage_field_directory is not None
        ),
        "foam_particle_shader": args.foam_particle_shader,
        "foam_particle_semantics": (
            "render-only translucent wet-film proxy on native FoamGenerator parcels; "
            "not resolved soap-film geometry"
            if args.foam_particle_shader == "wet-film"
            else "legacy opaque diagnostic proxy"
        ),
        "foam_scatter_calibration": {
            "active": args.foam_surface_representation == "marker-spheres",
            "role": "render-only local microfoam scattering; not a physical density",
            "support_radius_m": FOAM_DENSITY_SUPPORT_M,
            "onset_neighbors": FOAM_DENSITY_ONSET_NEIGHBORS,
            "full_neighbors": FOAM_DENSITY_FULL_NEIGHBORS,
            "thickness_onset_m": FOAM_THICKNESS_ONSET_M,
            "thickness_full_m": FOAM_THICKNESS_FULL_M,
            "maximum_scatter_mix": FOAM_MAX_SCATTER_MIX,
            "white_region_policy": (
                "density gated, grazing-angle enhanced, world-space Voronoi-edge "
                "and 3D-noise breakup"
            ),
        },
        "foam_marker_clusters": {
            "representation": args.foam_surface_representation,
            "binding_directory": (
                str(foam_binding_directory) if foam_binding_directory else None
            ),
            "binding_manifest_sha256": (
                sha256_file(foam_binding_manifest_path)
                if foam_binding_manifest_path
                else None
            ),
            "binding_audit_sha256": (
                sha256_file(foam_binding_audit_path)
                if foam_binding_audit_path
                else None
            ),
            "marker_semantics": "one native marker carries one visual microbubble group",
            "stable_layout_seed": "64-bit stable marker ID hashed to a unit float",
            "surface_binding_temporally_stable": foam_binding_audit is not None,
            "binding_sequence_audit_enforced": foam_binding_audit is not None,
            "native_velocity_stored": True,
            "native_velocity_consumed": True,
            "native_velocity_render_motion_blur_consumed": False,
            "anchor_stable": True,
            "normal_stable": True,
            "tangent_orientation_stable": True,
            "route_weights_complementary": True,
            "surface_geometry": "16-sided single-layer tangent NGON patches",
            "surface_conformance": "temporally predicted exact-triangle anchor with audited distance and continuity gates",
            "normal_offset_m": 0.00025,
            "normal_offset_role": "render-only z-fighting clearance",
            "instances_realized": True,
            "instance_policy": "Geometry Nodes batches patches, then realizes geometry so point attributes reach the material",
            "microbubble_layout": (
                "marker-local Voronoi edge network; stable-ID millimetre cell scale independent of instantaneous density"
            ),
            "neighbor_search": "cKDTree Gaussian pair kernel in temporal-binding build stage",
            "density_temporally_filtered": True,
            "cluster_radius_temporally_filtered": False,
            "cluster_radius_stable_by_id": True,
            "overlap_composition": (
                "each patch uses 1-exp(-tau) only for its own alpha; final reflected/scattered "
                "pixel radiance is determined by Cycles and is not claimed to equal a unified tau field"
            ),
            "cluster_radius_policy": (
                "render-only radius fixed from first active marker radius and the stable ID's "
                "worst offline-sequence triangle-curvature LOD; constant thereafter"
            ),
            "curvature_policy": (
                "high-curvature IDs receive a smaller sequence-constant planar patch; "
                "no instantaneous radius breathing"
            ),
            "hero_dome_policy": (
                "density>=0.45 and stable seed<0.10; flattened sphere lower half remains submerged"
                if args.foam_surface_representation == "microbubble-patches-hero"
                else None
            ),
            "python_individual_objects_created": False,
            "evaluated_mesh_diagnostics_enabled": bool(
                args.microbubble_debug_opaque
                or args.microbubble_debug_mask
                or args.microbubble_debug_alpha
            ),
            "physical_microbubble_geometry_claimed": False,
            "physical_bubble_radius_or_gas_volume_claimed": False,
        },
    },
    "foam_coverage_field": {
        "directory": (
            str(foam_coverage_field_directory)
            if foam_coverage_field_directory
            else None
        ),
        "manifest_sha256": (
            sha256_file(foam_coverage_manifest_path)
            if foam_coverage_manifest_path
            else None
        ),
        "audit_sha256": (
            sha256_file(foam_coverage_audit_path)
            if foam_coverage_audit_path
            else None
        ),
        "quantity": "dimensionless render coverage/optical depth",
        "maturity": "water-surface foam mask prototype",
        "actual_face_area_normalization_used": (
            foam_coverage_field_directory is not None
        ),
        "surface_field_history_advected": (
            foam_coverage_field_directory is not None
        ),
        "physical_area_claimed": False,
        "physical_film_area_or_gas_volume_claimed": False,
        "surface_geometry": "unchanged complete Splashsurf mesh",
        "free_foam_policy": "exclusive routed particles; no bound/free double counting",
    },
    "surface_plateau": {
        "directory": str(plateau_directory) if plateau_directory else None,
        "manifest_sha256": (
            sha256_file(plateau_manifest_path) if plateau_manifest_path else None
        ),
        "source": "audited PhysX/FoamGenerator stable parcels",
        "geometry": "volume-conservative Laguerre cells, shared films, Plateau borders and nodes",
        "camera_lod": {
            "micro_pixel_radius": args.plateau_micro_pixel_radius,
            "hero_pixel_radius": args.plateau_hero_pixel_radius,
            "micro_bin_size_m": args.plateau_micro_bin_size,
            "micro_policy": "equal-area optical patches; no alpha or radius floor",
        },
        "components": args.plateau_components,
        "open_surface_optical_role": (
            "cell tops and outer walls use wet-film LOD materials; the gas-boundary "
            "material is not used without a closed surrounding water medium"
        ),
        "motion_blur": "per-frame topology gate; deformation blur not yet synthesized",
    },
    "surface_foam_payload": {
        "directory": str(foam_payload_directory) if foam_payload_directory else None,
        "manifest_sha256": (
            sha256_file(foam_payload_manifest_path)
            if foam_payload_manifest_path
            else None
        ),
        "geometry": "oriented equal-film-area ellipses on exact triangle-surface anchors",
        "radius_policy": "major/minor support axes scaled by sqrt(film_area/support_area)",
        "thickness_policy": "render-only exponential film drainage, clamped at 80 nm",
        "double_counting": "payload and Plateau should be gated separately until hero masking is implemented",
    },
    "connected_foam_membrane": {
        "directory": str(foam_membrane_directory) if foam_membrane_directory else None,
        "manifest_sha256": (
            sha256_file(foam_membrane_manifest_path)
            if foam_membrane_manifest_path
            else None
        ),
        "geometry": "exact-area connected subset of real Splashsurf triangles",
        "shader_mode": args.foam_membrane_shader,
        "shading": (
            "coverage-aware high-transmission Principled thin wet film without alpha or "
            "cellular scattering; connected and bound-residual geometry share one optical family"
            if args.foam_membrane_shader == "wet-film"
            else "5 mm 3D Voronoi distance-to-edge scattering modulated by "
            "coverage=1-exp(-local optical depth); explicit hero-close-up diagnostic"
        ),
        "default_optical_policy": (
            "wet-film for ordinary shots; cellular requires explicit opt-in"
        ),
        "normal_offset_m": 0.0002,
        "bound_residual_geometry": (
            "exact-film-area oriented ellipses for the mutually exclusive non-top-facing, "
            "macro, or explicit-hole-host residual patch set"
        ),
        "unbound_residual_policy": (
            "preserved in surface-binding cache; not forced onto the free surface and not "
            "rendered without a closed-water entrained-bubble optical path"
        ),
        "payload_double_counting_forbidden": True,
        "deprecated": True,
        "obsolete_override_used": bool(args.allow_obsolete_foam_membrane),
    },
}
manifest_path = output_directory / "render_manifest.json"
existing_frames = list(output_directory.glob("frame_*.png"))
if manifest_path.is_file():
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing_manifest.get("configuration") != configuration and not args.force:
        raise RuntimeError(
            f"Existing render cache does not match {manifest_path}. "
            "Use a new output directory or pass --force."
        )
elif existing_frames and not args.force:
    raise RuntimeError(
        f"{output_directory} contains PNG files without a render manifest. "
        "Use a new output directory or pass --force after verifying the target."
    )
write_manifest(
    manifest_path,
    configuration,
    {"complete": False, "completed_frames": 0, "total_frames": len(frame_indices)},
)

water = None
water_volume = None
plateau_objects = ()
foam_payload_object = None
foam_membrane_object = None
foam_membrane_residual_object = None
completed_frames = 0
for sequence_position, index in enumerate(frame_indices):
    surface_path = surface_directory / f"surface_{index:04d}_clipped.obj"
    wait_started = time.monotonic()
    while not surface_path.is_file():
        if time.monotonic() - wait_started > 4 * 60 * 60:
            raise TimeoutError(f"Timed out waiting for {surface_path}")
        time.sleep(2.0)
    output_path = output_directory / f"frame_{index:04d}.png"
    if (
        output_path.is_file()
        and output_path.stat().st_size > 10_000
        and not args.force
    ):
        completed_frames += 1
        print(
            f"[splashsurf-render] cached frame={index:03d} "
            f"item={sequence_position + 1}/{len(frame_indices)}",
            flush=True,
        )
        continue
    if water is not None:
        old_mesh = water.data
        bpy.data.objects.remove(water, do_unlink=True)
        bpy.data.meshes.remove(old_mesh)
    bpy.ops.wm.obj_import(filepath=str(surface_path))
    water = bpy.context.active_object
    water.name = "SplashsurfWaterFrame"
    water.rotation_euler[0] = math.radians(90.0)
    water.data.materials.clear()
    water.data.materials.append(water_material)
    # The OBJ contains explicit v//vn references from Splashsurf. The clipped
    # topology is audited before rendering, so do not replace those smooth
    # normals with per-frame geometric normal recalculation.
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    foam_coverage_counts = None
    if foam_coverage_field_directory is not None:
        coverage_path = (
            foam_coverage_field_directory / f"coverage_field_{index:04d}.npz"
        )
        if not coverage_path.is_file():
            raise FileNotFoundError(coverage_path)
        foam_coverage_counts = apply_foam_coverage_to_water(water, coverage_path)
    if volume_directory is not None:
        volume_path = volume_directory / f"volume_{index:04d}.obj"
        if not volume_path.is_file():
            raise FileNotFoundError(volume_path)
        if water_volume is not None:
            old_volume_mesh = water_volume.data
            bpy.data.objects.remove(water_volume, do_unlink=True)
            bpy.data.meshes.remove(old_volume_mesh)
        bpy.ops.wm.obj_import(filepath=str(volume_path))
        water_volume = bpy.context.active_object
        water_volume.name = "ClosedWaterMediumFrame"
        water_volume.rotation_euler[0] = math.radians(90.0)
        water_volume.data.materials.clear()
        water_volume.data.materials.append(volume_material)
    plateau_counts = None
    if plateau_directory is not None:
        plateau_path = plateau_directory / f"plateau_{index:04d}.npz"
        if not plateau_path.is_file():
            raise FileNotFoundError(plateau_path)
        for plateau_object in plateau_objects:
            if plateau_object is not None:
                old_plateau_mesh = plateau_object.data
                bpy.data.objects.remove(plateau_object, do_unlink=True)
                bpy.data.meshes.remove(old_plateau_mesh)
        plateau_objects, plateau_counts = create_plateau_objects(
            plateau_path,
            plateau_materials,
            initial_plateau_thickness_nm,
            camera,
            resolution,
            args.plateau_micro_pixel_radius,
            args.plateau_hero_pixel_radius,
            args.plateau_micro_bin_size,
            args.plateau_components,
        )
    foam_payload_counts = None
    if foam_payload_directory is not None:
        payload_path = foam_payload_directory / f"surface_payload_{index:04d}.npz"
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)
        if foam_payload_object is not None:
            old_payload_mesh = foam_payload_object.data
            bpy.data.objects.remove(foam_payload_object, do_unlink=True)
            bpy.data.meshes.remove(old_payload_mesh)
        foam_payload_object, foam_payload_counts = create_surface_payload_object(
            payload_path, foam_payload_materials
        )
    foam_membrane_counts = None
    if foam_membrane_directory is not None:
        membrane_path = foam_membrane_directory / f"membrane_{index:04d}.npz"
        if not membrane_path.is_file():
            raise FileNotFoundError(membrane_path)
        if foam_membrane_object is not None:
            old_membrane_mesh = foam_membrane_object.data
            bpy.data.objects.remove(foam_membrane_object, do_unlink=True)
            bpy.data.meshes.remove(old_membrane_mesh)
        if foam_membrane_residual_object is not None:
            old_residual_mesh = foam_membrane_residual_object.data
            bpy.data.objects.remove(foam_membrane_residual_object, do_unlink=True)
            bpy.data.meshes.remove(old_residual_mesh)
        foam_membrane_object, foam_membrane_counts = create_foam_membrane_object(
            membrane_path, foam_membrane_materials
        )
        foam_membrane_residual_object, residual_counts = create_surface_payload_object(
            membrane_path,
            foam_membrane_residual_materials,
            patch_key="residual_patches",
            object_name="PhysXFoamMembraneBoundResidual",
        )
        foam_membrane_counts = {
            "connected": foam_membrane_counts,
            "bound_residual": residual_counts,
            "double_counting": False,
        }
    secondary_counts = {}
    for layer, obj in secondary_objects.items():
        cache_paths = secondary_cache_triplet(
            secondary_directory,
            layer,
            index,
            secondary_available_frames[layer],
        )
        secondary_counts[layer] = update_point_instancer(
            obj, cache_paths, index, output_fps
        )
    microbubble_counts = None
    if microbubble_patch_object is not None:
        foam_cache_path = secondary_directory / "foam" / f"frame_{index:04d}.npz"
        binding_sample = foam_binding_samples[index]
        binding_path = foam_binding_directory / binding_sample["file"]
        if sha256_file(foam_cache_path) != binding_sample["source_foam_cache_sha256"]:
            raise RuntimeError(f"Foam cache hash mismatch for binding frame {index}")
        if sha256_file(binding_path) != binding_sample["sha256"]:
            raise RuntimeError(f"Foam binding hash mismatch for frame {index}")
        foam_cache = read_secondary_cache(foam_cache_path)
        with np.load(binding_path, allow_pickle=False) as loaded:
            foam_binding = {
                name: np.asarray(loaded[name]).copy() for name in loaded.files
            }
        microbubble_counts = update_microbubble_patch_objects(
            microbubble_patch_object,
            microbubble_dome_object,
            foam_cache,
            foam_binding,
        )
        if (
            args.microbubble_debug_opaque
            or args.microbubble_debug_mask
            or args.microbubble_debug_alpha
        ):
            microbubble_counts["evaluated_density_attribute"] = (
                evaluated_float_attribute_range(
                    microbubble_patch_object, "foam_local_density"
                )
            )
            microbubble_counts["evaluated_seed_attribute"] = (
                evaluated_float_attribute_range(
                    microbubble_patch_object, "foam_marker_seed"
                )
            )
            microbubble_counts["evaluated_opacity_attribute"] = (
                evaluated_float_attribute_range(
                    microbubble_patch_object, "foam_marker_opacity"
                )
            )
            microbubble_counts["evaluated_patch_uv_attribute"] = (
                evaluated_vector_attribute_range(
                    microbubble_patch_object, "foam_patch_uv"
                )
            )
        free_cache = select_secondary_rows(
            foam_cache, np.asarray(foam_binding["unbound_source_row"], dtype=np.int64)
        )
        free_cache["opacity"] = (
            np.asarray(free_cache["opacity"], dtype=np.float32)
            * np.asarray(foam_binding["unbound_weight"], dtype=np.float32)
        )
        secondary_counts["foam_free_bubbles"] = update_point_instancer_from_caches(
            microbubble_free_object,
            [free_cache, free_cache, free_cache],
            index,
            output_fps,
        )
        secondary_counts["foam_surface_markers"] = microbubble_counts[
            "bound_marker_clusters"
        ]
    if routed_free_foam_object is not None:
        routed_paths = coverage_field_cache_triplet(
            foam_coverage_field_directory,
            index,
            coverage_available_frames,
        )
        secondary_counts["foam_free_route"] = update_point_instancer_from_caches(
            routed_free_foam_object,
            [read_routed_free_foam_cache(path) for path in routed_paths],
            index,
            output_fps,
        )
    sphere.location = isaac_to_blender(metrics[index]["sphere_center"])
    if volume_sphere_center is not None:
        for component, value in zip(("X", "Y", "Z"), sphere.location):
            volume_sphere_center.inputs[component].default_value = value
    scene.frame_set(index)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    completed_frames += 1
    print(
        f"[splashsurf-render] frame={index:03d} "
        f"item={sequence_position + 1}/{len(frame_indices)} "
        f"secondary={secondary_counts} plateau={plateau_counts} "
        f"surface_payload={foam_payload_counts} membrane={foam_membrane_counts} "
        f"coverage={foam_coverage_counts} microbubble={microbubble_counts}",
        flush=True,
    )

write_manifest(
    manifest_path,
    configuration,
    {
        "complete": True,
        "completed_frames": completed_frames,
        "total_frames": len(frame_indices),
    },
)
print(f"SPLASHSURF_RENDER={output_directory} frames={len(frame_indices)}")
