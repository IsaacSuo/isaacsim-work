"""Render one actual-scale Cycles lookdev gate for audited whitewater-v6 data.

Unlike the false-colour diagnostic, this renderer never enlarges sub-pixel
primitives.  Foam film area is represented geometrically: patch area, summed
cluster bubble projected area and visible ring area equal their cache values.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) < 10:
    raise SystemExit(
        "Expected SURFACE_OBJ MARKER_NPZ PAYLOAD_NPZ LIQUID_MANIFEST "
        "LIQUID_NPZ OUTPUT_PNG SPHERE_X SPHERE_Y SPHERE_Z SAMPLES"
    )
surface_path = Path(argv[0]).resolve()
marker_path = Path(argv[1]).resolve()
payload_path = Path(argv[2]).resolve()
liquid_manifest_path = Path(argv[3]).resolve()
liquid_path = Path(argv[4]).resolve()
output_path = Path(argv[5]).resolve()
sphere_isaac = tuple(float(value) for value in argv[6:9])
samples = int(argv[9])
view_mode = argv[10].strip().lower() if len(argv) >= 11 else "overview"
layer_mode = argv[11].strip().lower() if len(argv) >= 12 else "combined"
membrane_path = Path(argv[12]).resolve() if len(argv) >= 13 else None
if membrane_path is not None and str(argv[12]).strip() == "-":
    membrane_path = None
raft_path = Path(argv[13]).resolve() if len(argv) >= 14 else None
if samples < 1:
    raise ValueError("Samples must be positive")
if view_mode not in {"overview", "closeup"}:
    raise ValueError("View mode must be overview or closeup")
if layer_mode not in {"combined", "foam_only", "foam_isolation", "markers_only"}:
    raise ValueError("Layer mode must be combined, foam_only, foam_isolation or markers_only")
output_path.parent.mkdir(parents=True, exist_ok=True)
report_path = output_path.with_suffix(".json")
if output_path.exists() or report_path.exists():
    raise FileExistsError(f"Refusing to overwrite lookdev gate: {output_path}")

STATE_SPRAY = 1
STATE_ENTRAINED_BUBBLE = 3
STATE_SURFACE_BUBBLE = 4
CLASS_MICRO = 0
CLASS_CLUSTER = 1
CLASS_MACRO = 2
SURFACE_CAP_RESOLUTION_M = 0.00120
SURFACE_DENSITY_SPACING_M = 0.0040
SURFACE_DENSITY_MINIMUM_SIGMA_M = 0.0040


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isaac_to_blender(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (3,):
        return np.asarray((values[0], -values[2], values[1]), dtype=np.float64)
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def blender_to_isaac(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (3,):
        return np.asarray((values[0], values[2], -values[1]), dtype=np.float64)
    return np.column_stack((values[:, 0], values[:, 2], -values[:, 1]))


def set_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    return False


def make_principled(name, colour, roughness, transmission=0.0, ior=1.45):
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
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_bubble_material(name, surface=False):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (
        (0.98, 0.995, 1.0, 1.0) if not surface else (0.94, 0.98, 1.0, 1.0)
    )
    glass.inputs["Roughness"].default_value = 0.012 if not surface else 0.025
    # Faces are reversed for gas boundaries viewed from the water side.
    glass.inputs["IOR"].default_value = 1.333
    material.node_tree.links.new(glass.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_surface_foam_cap_material(name):
    """Open water-film cap: clear face-on and reflective at grazing angles."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.98, 0.995, 1.0, 1.0)
    film = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(film, (0.94, 0.98, 0.95, 1.0), "Base Color")
    set_input(film, 0.16, "Roughness")
    set_input(film, 1.333, "IOR")
    set_input(film, 0.30, "Coat Weight", "Coat")
    set_input(film, 0.12, "Coat Roughness")
    set_input(film, 380.0, "Thin Film Thickness")
    set_input(film, 1.33, "Thin Film IOR")
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.18
    mix = nodes.new("ShaderNodeMixShader")
    material.node_tree.links.new(fresnel.outputs["Fac"], mix.inputs[0])
    material.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    material.node_tree.links.new(film.outputs["BSDF"], mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_wet_foam_membrane_material(name):
    """Non-emissive wet foam film; physical transmission replaces white paint."""
    material = make_principled(
        name,
        (0.84, 0.89, 0.81, 1.0),
        0.31,
        transmission=0.14,
        ior=1.333,
    )
    shader = next(
        node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    set_input(shader, 0.18, "Coat Weight", "Coat")
    set_input(shader, 0.22, "Coat Roughness")
    set_input(shader, 420.0, "Thin Film Thickness")
    set_input(shader, 1.33, "Thin Film IOR")
    return material


def build_surface_bubble_density_atlas(records, grid_origin, grid_maximum):
    """Deposit unresolved gas-cap footprint as conservative optical depth."""
    minimum_x = float(grid_origin[0])
    minimum_z = float(grid_origin[2])
    maximum_x = float(grid_maximum[0])
    maximum_z = float(grid_maximum[2])
    spacing = SURFACE_DENSITY_SPACING_M
    nx = int(math.ceil((maximum_x - minimum_x) / spacing))
    nz = int(math.ceil((maximum_z - minimum_z) / spacing))
    tau = np.zeros((nx, nz), dtype=np.float64)
    expected_area = 0.0
    for record in records:
        radius = float(record["physical_radius"])
        shape = np.asarray(record["shape"], dtype=np.float64)
        lateral = radius * 0.5 * (shape[0] + shape[2])
        projected_area = math.pi * lateral * lateral
        expected_area += projected_area
        sigma = max(SURFACE_DENSITY_MINIMUM_SIGMA_M, 2.0 * lateral)
        support = 3.0 * sigma
        px, pz = float(record["position"][0]), float(record["position"][2])
        ix0 = max(0, int(math.floor((px - support - minimum_x) / spacing)))
        ix1 = min(nx, int(math.ceil((px + support - minimum_x) / spacing)) + 1)
        iz0 = max(0, int(math.floor((pz - support - minimum_z) / spacing)))
        iz1 = min(nz, int(math.ceil((pz + support - minimum_z) / spacing)) + 1)
        if ix0 >= ix1 or iz0 >= iz1:
            continue
        x = minimum_x + (np.arange(ix0, ix1) + 0.5) * spacing
        z = minimum_z + (np.arange(iz0, iz1) + 0.5) * spacing
        q2 = (
            ((x[:, None] - px) / sigma) ** 2
            + ((z[None, :] - pz) / sigma) ** 2
        )
        weight = np.exp(-0.5 * q2)
        weight[q2 > 9.0] = 0.0
        discrete_integral = float(weight.sum(dtype=np.float64) * spacing * spacing)
        if discrete_integral > 0.0:
            tau[ix0:ix1, iz0:iz1] += projected_area * weight / discrete_integral
    coverage = 1.0 - np.exp(-tau)
    integrated_tau = float(tau.sum(dtype=np.float64) * spacing * spacing)
    coverage_area = float(coverage.sum(dtype=np.float64) * spacing * spacing)
    return coverage.astype(np.float32), {
        "minimum_x": minimum_x,
        "minimum_z": minimum_z,
        "maximum_x": minimum_x + nx * spacing,
        "maximum_z": minimum_z + nz * spacing,
        "spacing": spacing,
        "shape": [nx, nz],
        "expected_projected_area_m2": expected_area,
        "integrated_tau_m2": integrated_tau,
        "tau_integral_residual_m2": expected_area - integrated_tau,
        "coverage_area_m2": coverage_area,
        "maximum_coverage": float(coverage.max()) if coverage.size else 0.0,
    }


def make_density_overlay_material(name, coverage, atlas):
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
    image = bpy.data.images.new(
        name + "Image",
        width=coverage.shape[0],
        height=coverage.shape[1],
        alpha=True,
        float_buffer=True,
    )
    pixels = np.ones((coverage.shape[1], coverage.shape[0], 4), dtype=np.float32)
    pixels[:, :, :3] = coverage.T[:, :, None]
    pixels[:, :, 3] = 1.0
    image.pixels.foreach_set(pixels.ravel())
    image.pack()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "CLIP"
    coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    cell_size = 0.0050
    mapping.inputs["Scale"].default_value = (
        (atlas["maximum_x"] - atlas["minimum_x"]) / cell_size,
        (atlas["maximum_z"] - atlas["minimum_z"]) / cell_size,
        1.0,
    )
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.feature = "DISTANCE_TO_EDGE"
    voronoi.voronoi_dimensions = "2D"
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
    combined_density = nodes.new("ShaderNodeMath")
    combined_density.operation = "ADD"
    material.node_tree.links.new(coordinates.outputs["UV"], texture.inputs["Vector"])
    material.node_tree.links.new(coordinates.outputs["UV"], mapping.inputs["Vector"])
    material.node_tree.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
    material.node_tree.links.new(voronoi.outputs["Distance"], rim.inputs["Fac"])
    material.node_tree.links.new(texture.outputs["Color"], rim_density.inputs[0])
    material.node_tree.links.new(rim.outputs["Color"], rim_density.inputs[1])
    material.node_tree.links.new(texture.outputs["Color"], base_density.inputs[0])
    material.node_tree.links.new(rim_density.outputs[0], combined_density.inputs[0])
    material.node_tree.links.new(base_density.outputs[0], combined_density.inputs[1])
    mix = nodes.new("ShaderNodeMixShader")
    material.node_tree.links.new(combined_density.outputs[0], mix.inputs[0])
    material.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    material.node_tree.links.new(foam.outputs["BSDF"], mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def make_surface_density_overlay(water, coverage, atlas, material, collection):
    """Copy only top-facing surface faces touched by the smooth density atlas."""
    vertices, faces, uv = [], [], []
    width = atlas["maximum_x"] - atlas["minimum_x"]
    height = atlas["maximum_z"] - atlas["minimum_z"]
    normal_matrix = water.matrix_world.to_3x3()
    for polygon in water.data.polygons:
        local_normal = np.asarray(polygon.normal, dtype=np.float64)
        if abs(local_normal[1]) < 0.50:
            continue
        local_points = [water.data.vertices[index].co.copy() for index in polygon.vertices]
        samples = []
        local_uv = []
        for point in local_points:
            u = (point.x - atlas["minimum_x"]) / width
            v = (point.z - atlas["minimum_z"]) / height
            local_uv.append((u, v))
            ix = int(np.clip(math.floor(u * coverage.shape[0]), 0, coverage.shape[0] - 1))
            iz = int(np.clip(math.floor(v * coverage.shape[1]), 0, coverage.shape[1] - 1))
            samples.append(float(coverage[ix, iz]))
        if max(samples) < 1.0e-3:
            continue
        world_normal = np.asarray(normal_matrix @ polygon.normal, dtype=np.float64)
        if local_normal[1] < 0.0:
            world_normal *= -1.0
        world_normal /= max(np.linalg.norm(world_normal), 1.0e-12)
        start = len(vertices)
        for point in local_points:
            world = np.asarray(water.matrix_world @ point, dtype=np.float64)
            vertices.append(tuple(world + 0.00032 * world_normal))
        faces.append(tuple(range(start, start + len(local_points))))
        uv.extend(local_uv)
    obj = link_mesh(
        "WWV6_UnresolvedSurfaceBubbleDensity",
        vertices,
        faces,
        material,
        collection,
        smooth=True,
    )
    uv_layer = obj.data.uv_layers.new(name="WWV6_SurfaceDensityUV")
    for loop, coordinate in zip(obj.data.loops, uv):
        uv_layer.data[loop.index].uv = coordinate
    return obj


def configure_scene(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
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
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"


def configure_hdri(scene):
    hdri = Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr")
    world = scene.world or bpy.data.worlds.new("WhitewaterV6LookdevWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.95
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return hdri


def link_mesh(name, vertices, faces, material, collection, smooth=True):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=False)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    if smooth:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    return obj


ICO_VERTICES = np.asarray(
    [
        (-1, 1.61803398875, 0), (1, 1.61803398875, 0),
        (-1, -1.61803398875, 0), (1, -1.61803398875, 0),
        (0, -1, 1.61803398875), (0, 1, 1.61803398875),
        (0, -1, -1.61803398875), (0, 1, -1.61803398875),
        (1.61803398875, 0, -1), (1.61803398875, 0, 1),
        (-1.61803398875, 0, -1), (-1.61803398875, 0, 1),
    ], dtype=np.float64,
)
ICO_VERTICES /= np.linalg.norm(ICO_VERTICES[0])
ICO_FACES = (
    (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
    (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
    (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
    (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
)


def make_ico_layer(name, records, material, collection, bubble_shape=False, flip=False):
    vertices, faces = [], []
    centres = isaac_to_blender(records["position"])
    for record, centre in zip(records, centres):
        radius = float(record["physical_radius"])
        scale = np.ones(3)
        if bubble_shape:
            shape = np.asarray(record["shape"], dtype=np.float64)
            scale = shape[[0, 2, 1]]
        offset = len(vertices)
        vertices.extend(map(tuple, centre + ICO_VERTICES * (radius * scale)[None, :]))
        for face in ICO_FACES:
            mapped = tuple(offset + index for index in face)
            faces.append(mapped[::-1] if flip else mapped)
    return link_mesh(name, vertices, faces, material, collection)


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    return vector / length if length > 1.0e-12 else np.asarray((0.0, 0.0, 1.0))


def frame(normal, tangent=None):
    n = normalize(normal)
    if tangent is None:
        tangent = np.asarray((1.0, 0.0, 0.0))
    t = np.asarray(tangent, dtype=np.float64)
    t -= np.dot(t, n) * n
    if np.linalg.norm(t) < 1.0e-8:
        t = np.asarray((0.0, 1.0, 0.0))
        t -= np.dot(t, n) * n
    t = normalize(t)
    b = normalize(np.cross(n, t))
    return n, t, b


def sample_trilinear(field, positions, origin, spacing):
    positions = np.asarray(positions, dtype=np.float64)
    shape = np.asarray(field.shape[:3], dtype=np.int64)
    q = (positions - origin) / spacing
    base = np.floor(q).astype(np.int64)
    fraction = q - base
    base = np.minimum(np.maximum(base, 0), shape - 2)
    result = np.zeros((len(positions),) + field.shape[3:], dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (fraction[:, 0] if dx else 1.0 - fraction[:, 0])
                    * (fraction[:, 1] if dy else 1.0 - fraction[:, 1])
                    * (fraction[:, 2] if dz else 1.0 - fraction[:, 2])
                )
                sampled = field[base[:,0]+dx, base[:,1]+dy, base[:,2]+dz]
                result += sampled * weight.reshape((-1,) + (1,) * (result.ndim - 1))
    return result


def build_world_bvh(obj):
    bpy.context.view_layer.update()
    vertices = [tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices]
    polygons = [tuple(polygon.vertices) for polygon in obj.data.polygons]
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=True)


def bind_to_render_surface(
    bvh,
    positions_isaac,
    normals_isaac,
    maximum_distance=0.024,
    audited_nearest_distance=None,
):
    bound_positions = np.empty_like(np.asarray(positions_isaac, dtype=np.float64))
    bound_normals = np.empty_like(np.asarray(normals_isaac, dtype=np.float64))
    distances = np.full(len(bound_positions), np.inf, dtype=np.float64)
    ray_extent = maximum_distance + 0.003
    fallback_count = 0
    audited_nearest_count = 0
    rejected_count = 0
    for index, (position_isaac, normal_isaac) in enumerate(zip(positions_isaac, normals_isaac)):
        position = isaac_to_blender(position_isaac)
        normal = normalize(isaac_to_blender(normal_isaac))
        candidates = []
        for sign in (-1.0, 1.0):
            origin = position + sign * ray_extent * normal
            direction = -sign * normal
            location, hit_normal, _, _ = bvh.ray_cast(
                Vector(origin), Vector(direction), 2.0 * ray_extent
            )
            if location is None:
                continue
            location = np.asarray(location, dtype=np.float64)
            hit_normal = normalize(np.asarray(hit_normal, dtype=np.float64))
            alignment = float(np.dot(hit_normal, normal))
            if alignment < 0.0:
                hit_normal = -hit_normal
                alignment = -alignment
            distance = float(np.linalg.norm(location - position))
            if alignment >= 0.20 and distance <= maximum_distance:
                candidates.append((distance, -alignment, location, hit_normal))
        if not candidates:
            location, hit_normal, _, distance = bvh.find_nearest(
                Vector(position), maximum_distance
            )
            if location is not None:
                location = np.asarray(location, dtype=np.float64)
                hit_normal = normalize(np.asarray(hit_normal, dtype=np.float64))
                alignment = float(np.dot(hit_normal, normal))
                if alignment < 0.0:
                    hit_normal = -hit_normal
                    alignment = -alignment
                if alignment >= 0.20 and float(distance) <= maximum_distance:
                    candidates.append((float(distance), -alignment, location, hit_normal))
                    fallback_count += 1
        if not candidates and audited_nearest_distance is not None:
            # The trajectory/payload gate has already proved these anchors lie
            # on the retained top shell.  A liquid-field normal can still be
            # nearly orthogonal to the tessellated Splashsurf normal at a
            # sharply curved crest.  Permit a tighter, explicitly counted
            # nearest-face bind for that audited case; never use the wider
            # general render gate or silently drop conserved film area.
            location, hit_normal, _, distance = bvh.find_nearest(
                Vector(position), float(audited_nearest_distance)
            )
            if location is not None and float(distance) <= float(audited_nearest_distance):
                location = np.asarray(location, dtype=np.float64)
                hit_normal = normalize(np.asarray(hit_normal, dtype=np.float64))
                alignment = float(np.dot(hit_normal, normal))
                if alignment < 0.0:
                    hit_normal = -hit_normal
                    alignment = -alignment
                candidates.append((float(distance), -alignment, location, hit_normal))
                audited_nearest_count += 1
        if candidates:
            distance, _, location, hit_normal = min(candidates, key=lambda item: (item[0], item[1]))
            bound_positions[index] = blender_to_isaac(location)
            bound_normals[index] = blender_to_isaac(hit_normal)
            distances[index] = distance
        else:
            # Preserve the audited solver anchor for diagnostics, but count it
            # as a render-binding rejection rather than silently jumping sheet.
            bound_positions[index] = position_isaac
            bound_normals[index] = normal_isaac
            rejected_count += 1
    finite = np.isfinite(distances)
    return bound_positions, bound_normals, finite, {
        "anchor_count": len(bound_positions),
        "bound_count": int(np.count_nonzero(finite)),
        "rejected_count": rejected_count,
        "nearest_fallback_count": fallback_count,
        "audited_top_shell_nearest_count": audited_nearest_count,
        "p50_distance_m": float(np.percentile(distances[finite], 50)) if np.any(finite) else None,
        "p95_distance_m": float(np.percentile(distances[finite], 95)) if np.any(finite) else None,
        "maximum_distance_m": float(np.max(distances[finite])) if np.any(finite) else None,
        "gate_m": maximum_distance,
    }


def bind_raft_clusters_to_render_surface(
    bvh, positions_isaac, normals_isaac, cluster_ids, maximum_distance=0.024
):
    """Bind raft clusters without changing their in-plane packing topology."""
    positions_isaac = np.asarray(positions_isaac, dtype=np.float64)
    normals_isaac = np.asarray(normals_isaac, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids, dtype=np.uint64)
    unique_ids = np.unique(cluster_ids)
    groups = [np.flatnonzero(cluster_ids == identifier) for identifier in unique_ids]
    centroids = np.asarray(
        [positions_isaac[group].mean(axis=0) for group in groups], dtype=np.float64
    )
    centroid_normals = np.asarray(
        [normalize(normals_isaac[group].mean(axis=0)) for group in groups],
        dtype=np.float64,
    )
    bound_centroids, bound_centroid_normals, centroid_bound, centroid_metrics = (
        bind_to_render_surface(
            bvh,
            centroids,
            centroid_normals,
            maximum_distance=maximum_distance,
            audited_nearest_distance=0.016,
        )
    )
    bound_positions = positions_isaac.copy()
    bound_normals = normals_isaac.copy()
    bound = np.zeros(len(positions_isaac), dtype=bool)
    normal_projection_failures = 0
    normal_projection_distances = []
    binding_displacements = []
    ray_extent = 0.008
    for group_index, members in enumerate(groups):
        if not centroid_bound[group_index]:
            continue
        old_centroid_blender = Vector(isaac_to_blender(centroids[group_index]))
        new_centroid_blender = Vector(isaac_to_blender(bound_centroids[group_index]))
        old_normal_blender = Vector(
            normalize(isaac_to_blender(centroid_normals[group_index]))
        )
        new_normal_blender = Vector(
            normalize(isaac_to_blender(bound_centroid_normals[group_index]))
        )
        rotation = old_normal_blender.rotation_difference(new_normal_blender)
        for member in members:
            old_position = Vector(isaac_to_blender(positions_isaac[member]))
            mapped = new_centroid_blender + rotation @ (
                old_position - old_centroid_blender
            )
            candidates = []
            for sign in (-1.0, 1.0):
                ray_origin = mapped + sign * ray_extent * new_normal_blender
                location, hit_normal, _, _ = bvh.ray_cast(
                    ray_origin, -sign * new_normal_blender, 2.0 * ray_extent
                )
                if location is None:
                    continue
                distance = float((location - mapped).length)
                candidates.append((distance, location, hit_normal))
            if candidates:
                distance, location, hit_normal = min(candidates, key=lambda value: value[0])
                hit_normal = Vector(hit_normal)
                if hit_normal.dot(new_normal_blender) < 0.0:
                    hit_normal = -hit_normal
                final_position = location
                final_normal = hit_normal.normalized()
                normal_projection_distances.append(distance)
            else:
                # Keep the coherent tangent map instead of collapsing this row
                # through an unrelated nearest-face fallback. The centroid is
                # already top-shell bound; record the local projection miss.
                final_position = mapped
                final_normal = new_normal_blender
                normal_projection_failures += 1
            bound_positions[member] = blender_to_isaac(final_position)
            bound_normals[member] = blender_to_isaac(final_normal)
            bound[member] = True
            binding_displacements.append(float((final_position - old_position).length))
    return bound_positions, bound_normals, bound, {
        "method": (
            "cluster-centroid production-surface bind, rigid normal-frame map, "
            "then common-normal ray projection preserving in-plane raft coordinates"
        ),
        "anchor_count": len(positions_isaac),
        "bound_count": int(np.count_nonzero(bound)),
        "rejected_count": int(np.count_nonzero(~bound)),
        "cluster_count": len(groups),
        "multirow_clusters": int(sum(len(group) > 1 for group in groups)),
        "normal_projection_failures": normal_projection_failures,
        "normal_projection_p95_m": (
            float(np.percentile(normal_projection_distances, 95))
            if normal_projection_distances else None
        ),
        "binding_displacement_p50_m": (
            float(np.percentile(binding_displacements, 50))
            if binding_displacements else None
        ),
        "binding_displacement_p95_m": (
            float(np.percentile(binding_displacements, 95))
            if binding_displacements else None
        ),
        "binding_displacement_maximum_m": (
            float(np.max(binding_displacements)) if binding_displacements else None
        ),
        "centroid_binding": centroid_metrics,
    }


def bind_raft_smooth_displacement_to_render_surface(
    bvh,
    positions_isaac,
    normals_isaac,
    radii,
    interaction_distance=0.008179818,
    smoothness=12.0,
    iterations=48,
):
    """Fit a smooth production-surface displacement without changing raft edges."""
    positions_isaac = np.asarray(positions_isaac, dtype=np.float64)
    normals_isaac = np.asarray(normals_isaac, dtype=np.float64)
    target_positions, target_normals, target_bound, target_metrics = (
        bind_to_render_surface(
            bvh,
            positions_isaac,
            normals_isaac,
            audited_nearest_distance=0.016,
        )
    )
    if not np.all(target_bound):
        return target_positions, target_normals, target_bound, {
            "method": "smooth displacement aborted because an independent target was unbound",
            "target_binding": target_metrics,
        }
    tree = KDTree(len(positions_isaac))
    for index, position in enumerate(positions_isaac):
        tree.insert(Vector(position), index)
    tree.balance()
    graph_distance = max(float(interaction_distance), 2.0 * float(np.max(radii)))
    pairs = []
    for first, position in enumerate(positions_isaac):
        for _, second, _ in tree.find_range(Vector(position), graph_distance):
            if second > first:
                pairs.append((first, second))
    pairs = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    target_displacement = target_positions - positions_isaac
    parent = np.arange(len(positions_isaac), dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in pairs:
        union(int(first), int(second))
    roots = np.asarray([find(index) for index in range(len(parent))], dtype=np.int64)
    displacement = np.empty_like(target_displacement)
    for root in np.unique(roots):
        members = roots == root
        displacement[members] = target_displacement[members].mean(axis=0)
    coherent = positions_isaac + displacement
    projected = coherent
    projected_normals = target_normals.copy()
    surface_residuals = []
    for index, position_isaac in enumerate(coherent):
        position = Vector(isaac_to_blender(position_isaac))
        location, hit_normal, _, distance = bvh.find_nearest(position, 0.024)
        if location is None:
            surface_residuals.append(0.024)
            continue
        surface_residuals.append(float(distance))
        hit_normal = Vector(hit_normal)
        expected = Vector(normalize(isaac_to_blender(target_normals[index])))
        if hit_normal.dot(expected) < 0.0:
            hit_normal = -hit_normal
        projected_normals[index] = blender_to_isaac(hit_normal.normalized())
    final_displacement = np.linalg.norm(projected - positions_isaac, axis=1)
    smoothing_residual = np.linalg.norm(displacement - target_displacement, axis=1)
    return projected, projected_normals, np.ones(len(projected), dtype=bool), {
        "method": (
            "independent production targets averaged as one rigid translation "
            "per potential-contact connected component; no per-row position projection"
        ),
        "anchor_count": len(projected),
        "bound_count": len(projected),
        "rejected_count": 0,
        "interaction_distance_m": graph_distance,
        "graph_edges": len(pairs),
        "connected_components": int(len(np.unique(roots))),
        "surface_residual_p50_m": float(np.percentile(surface_residuals, 50)),
        "surface_residual_p95_m": float(np.percentile(surface_residuals, 95)),
        "surface_residual_maximum_m": float(np.max(surface_residuals)),
        "binding_displacement_p50_m": float(np.percentile(final_displacement, 50)),
        "binding_displacement_p95_m": float(np.percentile(final_displacement, 95)),
        "binding_displacement_maximum_m": float(final_displacement.max(initial=0.0)),
        "smoothing_residual_p95_m": float(np.percentile(smoothing_residual, 95)),
        "target_binding": target_metrics,
    }


def bind_and_relax_raft_on_render_surface(
    bvh, positions_isaac, normals_isaac, radii, marker_ids, iterations=96
):
    """Solve soft contacts directly on the production mesh after binding."""
    positions, normals, bound, target_metrics = bind_to_render_surface(
        bvh,
        positions_isaac,
        normals_isaac,
        audited_nearest_distance=0.016,
    )
    if not np.all(bound):
        active = np.flatnonzero(bound)
        if not len(active):
            return positions, normals, bound, {
                "method": "production-surface contact solve had no safely bound anchors",
                "target_binding": target_metrics,
            }
        solved_positions, solved_normals, solved_bound, solved_metrics = (
            bind_and_relax_raft_on_render_surface(
                bvh,
                np.asarray(positions_isaac)[active],
                np.asarray(normals_isaac)[active],
                np.asarray(radii)[active],
                np.asarray(marker_ids)[active],
                iterations=iterations,
            )
        )
        if not np.all(solved_bound):
            raise RuntimeError(
                "A production-surface anchor changed from bound to rejected "
                "during the contact-solver subset verification"
            )
        positions[active] = solved_positions
        normals[active] = solved_normals
        solved_metrics = dict(solved_metrics)
        solved_metrics["method"] = (
            "safely bound subset solved on the production surface; rejected "
            "anchors preserved separately for explicit gas-volume accounting"
        )
        solved_metrics["input_anchor_count"] = int(len(bound))
        solved_metrics["solved_anchor_count"] = int(len(active))
        solved_metrics["unbound_anchor_count"] = int(np.count_nonzero(~bound))
        solved_metrics["initial_target_binding"] = target_metrics
        return positions, normals, bound, solved_metrics
    target_positions = positions.copy()
    radii = np.asarray(radii, dtype=np.float64)
    marker_ids = np.asarray(marker_ids, dtype=np.uint64)
    search_distance = 2.0 * float(radii.max(initial=0.0))
    total_corrections = 0
    projection_failures = 0
    normal_ray_projections = 0
    nearest_projection_fallbacks = 0
    iterations_used = 0
    for iteration in range(iterations):
        tree = KDTree(len(positions))
        for index, position in enumerate(positions):
            tree.insert(Vector(position), index)
        tree.balance()
        pairs = []
        maximum_compression = 0.0
        for first, position in enumerate(positions):
            for _, second, distance in tree.find_range(
                Vector(position), float(radii[first] + radii.max())
            ):
                if second <= first:
                    continue
                contact = float(radii[first] + radii[second])
                compression = max(0.0, 1.0 - float(distance) / contact)
                maximum_compression = max(maximum_compression, compression)
                if compression > 0.299:
                    pairs.append((first, second))
        iterations_used = iteration + 1
        if maximum_compression <= 0.3001:
            break
        moved = np.zeros(len(positions), dtype=bool)
        for first, second in pairs:
            delta = positions[second] - positions[first]
            average_normal = normalize(normals[first] + normals[second])
            normal_separation = float(np.dot(delta, average_normal))
            tangent = delta - normal_separation * average_normal
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length <= 1.0e-10:
                reference = np.asarray((1.0, 0.0, 0.0))
                if abs(float(average_normal[0])) > 0.85:
                    reference = np.asarray((0.0, 0.0, 1.0))
                first_tangent = normalize(
                    reference - np.dot(reference, average_normal) * average_normal
                )
                second_tangent = np.cross(average_normal, first_tangent)
                mixed = (
                    int(marker_ids[first]) * 0x9E3779B97F4A7C15
                    ^ int(marker_ids[second]) * 0xD6E8FEB86659FD93
                ) & ((1 << 64) - 1)
                angle = 2.0 * np.pi * ((mixed >> 11) / float(1 << 53))
                direction = np.cos(angle) * first_tangent + np.sin(angle) * second_tangent
            else:
                direction = tangent / tangent_length
            minimum_distance = 0.701 * float(radii[first] + radii[second])
            target_tangent = math.sqrt(
                max(minimum_distance**2 - normal_separation**2, 0.0)
            )
            correction_length = target_tangent - tangent_length
            if correction_length <= 1.0e-10:
                continue
            first_mobility = 1.0 / max(float(radii[first]), 5.0e-5)
            second_mobility = 1.0 / max(float(radii[second]), 5.0e-5)
            mobility_sum = first_mobility + second_mobility
            positions[first] -= direction * correction_length * (
                first_mobility / mobility_sum
            )
            positions[second] += direction * correction_length * (
                second_mobility / mobility_sum
            )
            moved[first] = True
            moved[second] = True
            total_corrections += 1
        offset = positions - target_positions
        offset_length = np.linalg.norm(offset, axis=1)
        outside = offset_length > 0.012
        positions[outside] = target_positions[outside] + offset[outside] * (
            0.012 / offset_length[outside]
        )[:, None]
        for index in np.flatnonzero(moved | outside):
            position = Vector(isaac_to_blender(positions[index]))
            expected = Vector(normalize(isaac_to_blender(normals[index])))
            ray_extent = 0.008
            candidates = []
            for sign in (-1.0, 1.0):
                origin = position + sign * ray_extent * expected
                location, hit_normal, _, _ = bvh.ray_cast(
                    origin, -sign * expected, 2.0 * ray_extent
                )
                if location is None:
                    continue
                hit_normal = Vector(hit_normal)
                alignment = hit_normal.dot(expected)
                if alignment < 0.0:
                    hit_normal = -hit_normal
                    alignment = -alignment
                if alignment < 0.20:
                    continue
                candidates.append(
                    (float((location - position).length), location, hit_normal)
                )
            if candidates:
                _, location, hit_normal = min(candidates, key=lambda value: value[0])
                normal_ray_projections += 1
            else:
                location, hit_normal, _, _ = bvh.find_nearest(position, ray_extent)
                if location is not None:
                    hit_normal = Vector(hit_normal)
                    nearest_projection_fallbacks += 1
            if location is None:
                positions[index] = target_positions[index]
                projection_failures += 1
                continue
            if hit_normal.dot(expected) < 0.0:
                hit_normal = -hit_normal
            positions[index] = blender_to_isaac(location)
            normals[index] = blender_to_isaac(hit_normal.normalized())
    # A sharp open edge can map two valid tangent-separated anchors back onto
    # the same boundary vertex on every closest-surface projection.  Finish
    # those rare cases in the local tangent plane, then audit the resulting
    # sub-millimetre surface residual.  This preserves the contact gate; it is
    # not permission to hide overlap or move a bubble to an unrelated sheet.
    tangent_finish_iterations = 0
    tangent_finish_corrections = 0
    for finish_iteration in range(32):
        tree = KDTree(len(positions))
        for index, position in enumerate(positions):
            tree.insert(Vector(position), index)
        tree.balance()
        pairs = []
        maximum_compression = 0.0
        for first, position in enumerate(positions):
            for _, second, distance in tree.find_range(
                Vector(position), float(radii[first] + radii.max())
            ):
                if second <= first:
                    continue
                contact = float(radii[first] + radii[second])
                compression = max(0.0, 1.0 - float(distance) / contact)
                maximum_compression = max(maximum_compression, compression)
                if compression > 0.299:
                    pairs.append((first, second))
        tangent_finish_iterations = finish_iteration + 1
        if maximum_compression <= 0.3001:
            break
        for first, second in pairs:
            delta = positions[second] - positions[first]
            average_normal = normalize(normals[first] + normals[second])
            normal_separation = float(np.dot(delta, average_normal))
            tangent = delta - normal_separation * average_normal
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length <= 1.0e-10:
                reference = np.asarray((1.0, 0.0, 0.0))
                if abs(float(average_normal[0])) > 0.85:
                    reference = np.asarray((0.0, 0.0, 1.0))
                first_tangent = normalize(
                    reference - np.dot(reference, average_normal) * average_normal
                )
                second_tangent = np.cross(average_normal, first_tangent)
                mixed = (
                    int(marker_ids[first]) * 0x9E3779B97F4A7C15
                    ^ int(marker_ids[second]) * 0xD6E8FEB86659FD93
                ) & ((1 << 64) - 1)
                angle = 2.0 * np.pi * ((mixed >> 11) / float(1 << 53))
                direction = np.cos(angle) * first_tangent + np.sin(angle) * second_tangent
            else:
                direction = tangent / tangent_length
            minimum_distance = 0.701 * float(radii[first] + radii[second])
            target_tangent = math.sqrt(
                max(minimum_distance**2 - normal_separation**2, 0.0)
            )
            correction_length = target_tangent - tangent_length
            if correction_length <= 1.0e-10:
                continue
            first_mobility = 1.0 / max(float(radii[first]), 5.0e-5)
            second_mobility = 1.0 / max(float(radii[second]), 5.0e-5)
            mobility_sum = first_mobility + second_mobility
            positions[first] -= direction * correction_length * (
                first_mobility / mobility_sum
            )
            positions[second] += direction * correction_length * (
                second_mobility / mobility_sum
            )
            tangent_finish_corrections += 1
        offset = positions - target_positions
        offset_length = np.linalg.norm(offset, axis=1)
        outside = offset_length > 0.012
        positions[outside] = target_positions[outside] + offset[outside] * (
            0.012 / offset_length[outside]
        )[:, None]

    surface_residuals = np.zeros(len(positions), dtype=np.float64)
    for index, position_isaac in enumerate(positions):
        _, _, _, distance = bvh.find_nearest(
            Vector(isaac_to_blender(position_isaac)), 0.0012
        )
        surface_residuals[index] = 0.0012 if distance is None else float(distance)
    maximum_surface_residual = float(surface_residuals.max(initial=0.0))
    if maximum_surface_residual > 0.00035:
        raise RuntimeError(
            "Collision-preserving tangent finish exceeded the 0.35 mm "
            f"production-surface residual gate: {maximum_surface_residual}"
        )

    _, _, contact_metrics = build_surface_contacts(positions, radii)
    final_offset = np.linalg.norm(positions - target_positions, axis=1)
    original_binding = np.linalg.norm(
        positions - np.asarray(positions_isaac, dtype=np.float64), axis=1
    )
    return positions, normals, np.ones(len(positions), dtype=bool), {
        "method": (
            "independent exact production-surface targets followed by iterative "
            "soft-contact PBD and closest-face reprojection on that same mesh"
        ),
        "anchor_count": len(positions),
        "bound_count": len(positions),
        "rejected_count": 0,
        "iterations": iterations_used,
        "contact_corrections": total_corrections,
        "projection_failures": projection_failures,
        "normal_ray_projections": normal_ray_projections,
        "nearest_projection_fallbacks": nearest_projection_fallbacks,
        "tangent_finish_iterations": tangent_finish_iterations,
        "tangent_finish_corrections": tangent_finish_corrections,
        "surface_residual_p95_m": float(np.percentile(surface_residuals, 95)),
        "surface_residual_maximum_m": maximum_surface_residual,
        "surface_residual_gate_m": 0.00035,
        "maximum_contact_compression": contact_metrics["maximum_compression"],
        "target_offset_p95_m": float(np.percentile(final_offset, 95)),
        "target_offset_maximum_m": float(final_offset.max(initial=0.0)),
        "binding_displacement_p50_m": float(np.percentile(original_binding, 50)),
        "binding_displacement_p95_m": float(np.percentile(original_binding, 95)),
        "binding_displacement_maximum_m": float(original_binding.max(initial=0.0)),
        "target_binding": target_metrics,
    }


def build_surface_contacts(positions, radii):
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    contacts = [[] for _ in range(len(positions))]
    maximum_by_row = np.zeros(len(positions), dtype=np.float64)
    if not len(positions):
        return contacts, maximum_by_row, {"contact_pairs": 0, "maximum_compression": 0.0}
    tree = KDTree(len(positions))
    for index, position in enumerate(positions):
        tree.insert(Vector(position), index)
    tree.balance()
    maximum_radius = float(radii.max(initial=0.0))
    pair_count = 0
    maximum_compression = 0.0
    maximum_compression_pair = None
    maximum_compression_distance = None
    maximum_compression_contact_distance = None
    for first, position in enumerate(positions):
        for _, second, distance in tree.find_range(
            Vector(position), float(radii[first] + maximum_radius)
        ):
            if second <= first:
                continue
            contact_distance = float(radii[first] + radii[second])
            compression = max(0.0, 1.0 - float(distance) / contact_distance)
            if compression <= 1.0e-7:
                continue
            delta = positions[second] - position
            length = float(np.linalg.norm(delta))
            if length <= 1.0e-10:
                angle = 2.0 * np.pi * (((first + 1) * 2654435761 ^ (second + 1)) & 0xFFFF) / 65536.0
                direction = np.asarray((np.cos(angle), 0.0, np.sin(angle)))
            else:
                direction = delta / length
            contacts[first].append((direction, compression))
            contacts[second].append((-direction, compression))
            maximum_by_row[first] = max(maximum_by_row[first], compression)
            maximum_by_row[second] = max(maximum_by_row[second], compression)
            pair_count += 1
            if compression > maximum_compression:
                maximum_compression = compression
                maximum_compression_pair = [int(first), int(second)]
                maximum_compression_distance = float(distance)
                maximum_compression_contact_distance = contact_distance
    return contacts, maximum_by_row, {
        "contact_pairs": pair_count,
        "maximum_compression": maximum_compression,
        "maximum_compression_pair": maximum_compression_pair,
        "maximum_compression_distance_m": maximum_compression_distance,
        "maximum_compression_contact_distance_m": maximum_compression_contact_distance,
    }


def make_surface_domes(name, records, normals_isaac, contacts, material, collection):
    vertices, faces = [], []
    segments, rings = 12, 4
    area_scale_residual = 0.0
    for record, normal_isaac, record_contacts in zip(records, normals_isaac, contacts):
        centre = isaac_to_blender(record["position"])
        normal = isaac_to_blender(normal_isaac)
        n, t, b = frame(normal)
        radius = float(record["physical_radius"])
        shape = np.asarray(record["shape"], dtype=np.float64)
        lateral = radius * 0.5 * (shape[0] + shape[2])
        height = radius * shape[1]
        radial_factors = np.ones(segments, dtype=np.float64)
        for direction_isaac, compression in record_contacts:
            contact_direction = isaac_to_blender(direction_isaac)
            contact_direction -= np.dot(contact_direction, n) * n
            if np.linalg.norm(contact_direction) <= 1.0e-10:
                continue
            contact_direction = normalize(contact_direction)
            for segment in range(segments):
                angle = 2.0 * math.pi * segment / segments
                radial = math.cos(angle) * t + math.sin(angle) * b
                influence = max(float(np.dot(radial, contact_direction)), 0.0) ** 4
                radial_factors[segment] = min(
                    radial_factors[segment], 1.0 - float(compression) * influence
                )
        footprint_scale = float(np.mean(radial_factors**2))
        height_scale = 1.0 / max(footprint_scale, 1.0e-6)
        area_scale_residual = max(
            area_scale_residual, abs(footprint_scale * height_scale - 1.0)
        )
        start = len(vertices)
        vertices.append(tuple(centre + height * height_scale * n))
        for ring_index in range(1, rings + 1):
            theta = 0.5 * math.pi * ring_index / rings
            for segment in range(segments):
                angle = 2.0 * math.pi * segment / segments
                point = (
                    centre
                    + height * height_scale * math.cos(theta) * n
                    + lateral * radial_factors[segment] * math.sin(theta)
                    * (math.cos(angle) * t + math.sin(angle) * b)
                )
                vertices.append(tuple(point))
        for segment in range(segments):
            faces.append((start, start + 1 + (segment + 1) % segments, start + 1 + segment))
        for ring_index in range(1, rings):
            lower = start + 1 + (ring_index - 1) * segments
            upper = lower + segments
            for segment in range(segments):
                nxt = (segment + 1) % segments
                faces.append((lower+segment, lower+nxt, upper+nxt, upper+segment)[::-1])
    return link_mesh(name, vertices, faces, material, collection), {
        "volume_scale_residual": area_scale_residual,
        "deformed_domes": int(sum(bool(value) for value in contacts)),
    }


def edge_factors(key, segments, amplitude):
    rng = np.random.default_rng(int(key))
    phases = rng.uniform(0.0, 2.0 * np.pi, 3)
    weights = np.asarray((1.0, 0.55, 0.30))
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    noise = sum(w * np.sin((i + 2) * angles + phases[i]) for i, w in enumerate(weights))
    factors = np.maximum(0.35, 1.0 + amplitude * noise / weights.sum())
    # For a radial polygon, area is proportional to mean radius squared.
    return factors / math.sqrt(float(np.mean(factors * factors)))


def make_film_patches(name, records, material, collection, amplitude, segments):
    vertices, faces = [], []
    rendered_area = 0.0
    for record in records:
        film = float(record["film_area"])
        support = float(record["support_area"])
        if film <= 0.0 or support <= 0.0:
            continue
        centre = isaac_to_blender(record["position"])
        normal = isaac_to_blender(record["normal"])
        tangent = isaac_to_blender(record["principal_direction"])
        n, t, b = frame(normal, tangent)
        centre += 0.00035 * n
        coverage_scale = math.sqrt(np.clip(film / support, 0.0, 1.0))
        major = float(record["major_radius"]) * coverage_scale
        minor = float(record["minor_radius"]) * coverage_scale
        factors = edge_factors(record["random_key"], segments, amplitude)
        polygon_area = (
            0.5
            * math.sin(2.0 * math.pi / segments)
            * float(np.sum(factors * np.roll(factors, -1)))
            * major
            * minor
        )
        correction = math.sqrt(film / max(polygon_area, 1.0e-30))
        major *= correction
        minor *= correction
        start = len(vertices)
        vertices.append(tuple(centre))
        for segment, factor in enumerate(factors):
            angle = 2.0 * math.pi * segment / segments
            vertices.append(tuple(
                centre + factor * major * math.cos(angle) * t + factor * minor * math.sin(angle) * b
            ))
        for segment in range(segments):
            faces.append((start, start + 1 + segment, start + 1 + (segment + 1) % segments))
        rendered_area += film
    obj = link_mesh(name, vertices, faces, material, collection, smooth=False)
    obj["represented_film_area_m2"] = rendered_area
    return obj, rendered_area


def make_cluster_cells(records, material, collection):
    vertices, faces = [], []
    cell_count, represented_area = 0, 0.0
    for record in records:
        film = float(record["film_area"])
        if film <= 0.0:
            continue
        rng = np.random.default_rng(int(record["random_key"]))
        count = int(np.clip(math.ceil(film / (math.pi * 0.00030**2)), 2, 24))
        weights = rng.lognormal(mean=0.0, sigma=0.28, size=count)
        radii = weights * math.sqrt(film / (math.pi * float(np.sum(weights * weights))))
        centre = isaac_to_blender(record["position"])
        n, t, b = frame(
            isaac_to_blender(record["normal"]),
            isaac_to_blender(record["principal_direction"]),
        )
        major = float(record["major_radius"])
        minor = float(record["minor_radius"])
        for child, radius in enumerate(radii):
            radial = 0.78 * math.sqrt(float(rng.random()))
            angle = 2.0 * math.pi * float(rng.random())
            child_centre = centre + radial * (
                major * math.cos(angle) * t + minor * math.sin(angle) * b
            ) + 0.00025 * n
            segments, rings = 10, 3
            start = len(vertices)
            vertices.append(tuple(child_centre + 0.48 * radius * n))
            for ring_index in range(1, rings + 1):
                theta = 0.5 * math.pi * ring_index / rings
                for segment in range(segments):
                    phi = 2.0 * math.pi * segment / segments
                    vertices.append(tuple(
                        child_centre
                        + 0.48 * radius * math.cos(theta) * n
                        + radius * math.sin(theta) * (math.cos(phi) * t + math.sin(phi) * b)
                    ))
            for segment in range(segments):
                faces.append((start, start+1+(segment+1)%segments, start+1+segment))
            for ring_index in range(1, rings):
                lower = start + 1 + (ring_index - 1) * segments
                upper = lower + segments
                for segment in range(segments):
                    nxt = (segment + 1) % segments
                    faces.append((lower+segment, lower+nxt, upper+nxt, upper+segment))
            cell_count += 1
        represented_area += film
    obj = link_mesh("WWV6_FoamBubbleCells", vertices, faces, material, collection)
    obj["cell_count"] = cell_count
    obj["represented_film_area_m2"] = represented_area
    return obj, cell_count, represented_area


def make_irregular_rings(records, material, collection):
    vertices, faces = [], []
    represented_area = 0.0
    segments = 48
    for record in records:
        film = float(record["film_area"])
        if film <= 0.0:
            continue
        centre = isaac_to_blender(record["position"])
        n, t, b = frame(isaac_to_blender(record["normal"]))
        centre += 0.00048 * n
        inner = float(record["inner_radius"])
        physical_outer = float(record["outer_radius"])
        annulus = math.pi * max(physical_outer * physical_outer - inner * inner, 1.0e-30)
        coverage = float(np.clip(film / annulus, 0.0, 1.0))
        active_count = int(np.clip(math.ceil(coverage * segments), 1, segments))
        active_fraction = active_count / segments
        # Use only as many angular sectors as the film can occupy, then solve
        # the radial width so the visible geometric area remains exactly the
        # conservative film budget. This produces intermittent Plateau-border
        # arcs instead of a uniformly bright independent circle.
        visible_outer = min(
            physical_outer,
            math.sqrt(inner * inner + film / (math.pi * active_fraction)),
        )
        rng = np.random.default_rng(int(record["random_key"]))
        phase_a, phase_b = rng.uniform(0.0, 2.0 * math.pi, 2)
        angles = 2.0 * math.pi * np.arange(segments) / segments
        sector_score = np.sin(angles + phase_a) + 0.42 * np.sin(3.0 * angles + phase_b)
        active = np.zeros(segments, dtype=bool)
        active[np.argsort(sector_score)[-active_count:]] = True
        factors = edge_factors(record["random_key"], segments, 0.06)
        start = len(vertices)
        for radius in (inner, visible_outer):
            for segment, factor in enumerate(factors):
                angle = 2.0 * math.pi * segment / segments
                vertices.append(tuple(
                    centre + factor * radius * (math.cos(angle) * t + math.sin(angle) * b)
                ))
        for segment in range(segments):
            if not active[segment]:
                continue
            nxt = (segment + 1) % segments
            faces.append((start+segment, start+nxt, start+segments+nxt, start+segments+segment))
        represented_area += film
    obj = link_mesh("WWV6_ConservativeIrregularRings", vertices, faces, material, collection, smooth=False)
    obj["represented_film_area_m2"] = represented_area
    return obj, represented_area


def make_hole_masks(records, material, collection):
    """Place water-surface masks below hosted rims so holes are topological."""
    vertices, faces = [], []
    segments = 40
    for record in records:
        centre = isaac_to_blender(record["position"])
        n, t, b = frame(isaac_to_blender(record["normal"]))
        centre += 0.00043 * n
        radius = 0.96 * float(record["inner_radius"])
        factors = edge_factors(
            int(record["random_key"]) ^ 0x5A17E4D3, segments, 0.055
        )
        start = len(vertices)
        vertices.append(tuple(centre))
        for segment, factor in enumerate(factors):
            angle = 2.0 * math.pi * segment / segments
            vertices.append(
                tuple(
                    centre
                    + factor
                    * radius
                    * (math.cos(angle) * t + math.sin(angle) * b)
                )
            )
        for segment in range(segments):
            faces.append(
                (
                    start,
                    start + 1 + segment,
                    start + 1 + (segment + 1) % segments,
                )
            )
    return link_mesh(
        "WWV6_HostedHoleWaterMasks",
        vertices,
        faces,
        material,
        collection,
        smooth=False,
    )


def make_surface_membrane(vertices_isaac, triangles, normals_isaac, material, collection):
    """Render audited film area as offset copies of real Splashsurf triangles."""
    vertices_isaac = np.asarray(vertices_isaac, dtype=np.float64).copy()
    triangles = np.asarray(triangles, dtype=np.int32)
    normals_isaac = np.asarray(normals_isaac, dtype=np.float64)
    if len(triangles) != len(normals_isaac):
        raise ValueError("Membrane triangle and normal counts differ")
    if len(vertices_isaac) != 3 * len(triangles):
        raise ValueError("Membrane geometry must use exploded triangles")
    for face_index, triangle in enumerate(triangles):
        vertices_isaac[triangle] += 0.00035 * normals_isaac[face_index]
    return link_mesh(
        "WWV6_ConservativeSurfaceMembrane",
        isaac_to_blender(vertices_isaac),
        triangles,
        material,
        collection,
        smooth=False,
    )


with np.load(marker_path, allow_pickle=False) as cache:
    output_frame = int(cache["output_frame"])
    source_sample = int(cache["source_sample_index"])
    markers = np.asarray(cache["markers"])
surface_raft = None
if raft_path is not None:
    with np.load(raft_path, allow_pickle=False) as cache:
        if int(cache["output_frame"]) != output_frame:
            raise RuntimeError("Marker and surface-raft frames differ")
        if int(cache["source_sample_index"]) != source_sample:
            raise RuntimeError("Marker and surface-raft samples differ")
        surface_raft = np.asarray(cache["raft"])
with np.load(payload_path, allow_pickle=False) as cache:
    if int(cache["output_frame"]) != output_frame:
        raise RuntimeError("Marker and foam payload frames differ")
    patches = np.asarray(cache["patches"])
    rings = np.asarray(cache["rings"])
payload_source_film_area = float(
    patches["film_area"].sum(dtype=np.float64)
    + rings["film_area"].sum(dtype=np.float64)
)
membrane = None
membrane_area = 0.0
membrane_source_ids = np.empty(0, dtype=np.uint64)
if membrane_path is not None:
    with np.load(membrane_path, allow_pickle=False) as cache:
        if int(cache["output_frame"]) != output_frame:
            raise RuntimeError("Marker and foam membrane frames differ")
        if int(cache["source_sample_index"]) != source_sample:
            raise RuntimeError("Marker and foam membrane samples differ")
        membrane = {
            "vertices": np.asarray(cache["vertices"]),
            "triangles": np.asarray(cache["triangles"]),
            "triangle_normals": np.asarray(cache["triangle_normals"]),
        }
        membrane_source_ids = np.asarray(
            cache["aggregated_source_foam_ids"], dtype=np.uint64
        )
        membrane_area = float(cache["represented_membrane_area_m2"])
manifest = json.loads(liquid_manifest_path.read_text(encoding="utf-8"))
grid = manifest["grid"]
origin = np.asarray(grid["origin"], dtype=np.float64)
grid_maximum = np.asarray(grid["maximum"], dtype=np.float64)
spacing = float(grid["spacing"])
with np.load(liquid_path, allow_pickle=False) as cache:
    if int(cache["source_sample_index"]) != source_sample:
        raise RuntimeError("Liquid field and marker samples differ")
    normal_field = np.asarray(cache["normal"])

scene = bpy.context.scene
configure_scene(scene)
hdri = configure_hdri(scene)
collection = bpy.data.collections.new("WhitewaterV6_LookdevGate")
scene.collection.children.link(collection)

water_material = make_principled(
    "WWV6_Water", (0.82, 0.94, 0.98, 1.0), 0.03, transmission=1.0, ior=1.333
)
spray_material = make_principled(
    "WWV6_SprayWater", (0.88, 0.96, 1.0, 1.0), 0.045, transmission=1.0, ior=1.333
)
entrained_material = make_bubble_material("WWV6_EntrainedAir")
surface_bubble_material = make_surface_foam_cap_material("WWV6_SurfaceBubbleThinFilmCap")
micro_material = make_principled("WWV6_MicroFoamFilm", (0.93, 0.96, 0.92, 1.0), 0.50)
cluster_material = make_principled("WWV6_FoamCells", (0.96, 0.98, 0.95, 1.0), 0.34)
macro_material = make_principled("WWV6_MacroFoamFilm", (0.91, 0.95, 0.89, 1.0), 0.42)
ring_material = make_principled("WWV6_FoamHoleRims", (0.97, 0.985, 0.96, 1.0), 0.30)
membrane_material = make_wet_foam_membrane_material("WWV6_WetFoamMembrane")
hole_material = make_principled(
    "WWV6_HostedHoleWater", (0.82, 0.94, 0.98, 1.0), 0.03, transmission=1.0, ior=1.333
)
sphere_material = make_principled("WWV6_ImpactorOrange", (0.95, 0.18, 0.035, 1.0), 0.24)

bpy.ops.wm.obj_import(filepath=str(surface_path))
water = bpy.context.active_object
water.name = "WWV6_LiquidSurface"
water.rotation_euler[0] = math.radians(90.0)
water.data.materials.clear()
water.data.materials.append(water_material)
for polygon in water.data.polygons:
    polygon.use_smooth = True
bvh = build_world_bvh(water)

spray = markers[markers["state"] == STATE_SPRAY]
entrained = markers[markers["state"] == STATE_ENTRAINED_BUBBLE]
surface_bubbles = markers[markers["state"] == STATE_SURFACE_BUBBLE]
surface_bubbles = surface_bubbles[np.argsort(surface_bubbles["id"])]
source_surface_bubble_count = len(surface_bubbles)
source_surface_gas_volume = float(
    surface_bubbles["phase_volume"].sum(dtype=np.float64)
)
if surface_raft is not None:
    if not np.array_equal(surface_bubbles["id"], surface_raft["marker_id"]):
        raise RuntimeError("Surface-raft IDs do not match active surface bubbles")
    for marker_name, raft_name in (
        ("physical_radius", "physical_radius"),
        ("representative_count", "representative_count"),
        ("phase_volume", "phase_volume"),
        ("shape", "shape"),
        ("random_key", "random_key"),
    ):
        if not np.array_equal(surface_bubbles[marker_name], surface_raft[raft_name]):
            raise RuntimeError(f"Surface raft changed immutable field {marker_name}")
    surface_bubbles = surface_bubbles.copy()
    surface_bubbles["position"] = surface_raft["raft_position"]
surface_normals = sample_trilinear(
    normal_field, surface_bubbles["position"].astype(np.float64), origin, spacing
)
surface_normals /= np.maximum(np.linalg.norm(surface_normals, axis=1)[:,None], 1.0e-12)
if surface_raft is None:
    bound_surface_positions, bound_surface_normals, surface_bound, surface_binding = bind_to_render_surface(
        bvh,
        surface_bubbles["position"],
        surface_normals,
        audited_nearest_distance=0.016,
    )
else:
    bound_surface_positions, bound_surface_normals, surface_bound, surface_binding = (
        bind_and_relax_raft_on_render_surface(
            bvh,
            surface_bubbles["position"],
            surface_normals,
            surface_bubbles["physical_radius"],
            surface_raft["marker_id"],
        )
    )
surface_bubbles = surface_bubbles.copy()
surface_bubbles["position"] = bound_surface_positions.astype(np.float32)
surface_normals = bound_surface_normals
unbound_surface_bubbles = surface_bubbles[~surface_bound]
unbound_surface_gas_volume = float(
    unbound_surface_bubbles["phase_volume"].sum(dtype=np.float64)
)
if surface_raft is not None:
    surface_raft = surface_raft[surface_bound]
surface_bubbles = surface_bubbles[surface_bound]
surface_normals = surface_normals[surface_bound]
surface_contacts, rendered_contact_compression, rendered_contact_metrics = (
    build_surface_contacts(
        surface_bubbles["position"].astype(np.float64),
        surface_bubbles["physical_radius"].astype(np.float64),
    )
)
if surface_raft is not None:
    cached_contact_compression = surface_raft[
        "maximum_contact_compression"
    ].astype(np.float64)
    rendered_contact_metrics["cached_maximum_compression"] = float(
        cached_contact_compression.max(initial=0.0)
    )
    rendered_contact_metrics["maximum_cache_to_bound_delta"] = float(
        np.max(
            np.abs(rendered_contact_compression - cached_contact_compression),
            initial=0.0,
        )
    )
if rendered_contact_metrics["maximum_compression"] > 0.3001:
    worst_pair = rendered_contact_metrics.get("maximum_compression_pair")
    worst_ids = (
        [int(surface_bubbles[worst_pair[0]]["id"]), int(surface_bubbles[worst_pair[1]]["id"])]
        if worst_pair is not None
        else None
    )
    raise RuntimeError(
        "Production-surface binding pushed a rendered bubble contact above "
        f"the 30% gate: {rendered_contact_metrics['maximum_compression']}; "
        f"rows={worst_pair}; marker_ids={worst_ids}; "
        f"distance_m={rendered_contact_metrics.get('maximum_compression_distance_m')}; "
        "contact_distance_m="
        f"{rendered_contact_metrics.get('maximum_compression_contact_distance_m')}; "
        f"binding_metrics={surface_binding}"
    )

patches = patches.copy()
bound_patch_positions, bound_patch_normals, patch_bound, patch_binding = bind_to_render_surface(
    bvh,
    patches["position"],
    patches["normal"],
    audited_nearest_distance=0.016,
)
patches["position"] = bound_patch_positions.astype(np.float32)
patches["normal"] = bound_patch_normals.astype(np.float32)
unbound_patch_area = float(patches["film_area"][~patch_bound].sum(dtype=np.float64))
patches = patches[patch_bound]
membrane_member = np.isin(patches["source_foam_id"], membrane_source_ids)
if membrane is not None:
    if np.any(patches["hole_fraction"][membrane_member] > 0.0):
        raise RuntimeError("Membrane input absorbed explicit hosted-hole parcels")
    aggregated_patch_area = float(
        patches["film_area"][membrane_member].sum(dtype=np.float64)
    )
    if not math.isclose(aggregated_patch_area, membrane_area, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError(
            "Membrane area does not equal its aggregated payload film area: "
            f"{membrane_area} versus {aggregated_patch_area}"
        )
patches = patches[~membrane_member]
rings = rings.copy()
bound_ring_positions, bound_ring_normals, ring_bound, ring_binding = bind_to_render_surface(
    bvh,
    rings["position"],
    rings["normal"],
    audited_nearest_distance=0.016,
)
rings["position"] = bound_ring_positions.astype(np.float32)
rings["normal"] = bound_ring_normals.astype(np.float32)
unbound_ring_area = float(rings["film_area"][~ring_bound].sum(dtype=np.float64))
rings = rings[ring_bound]

spray_obj = make_ico_layer("WWV6_Spray", spray, spray_material, collection)
entrained_obj = make_ico_layer("WWV6_EntrainedBubbles", entrained, entrained_material, collection, True, True)
surface_bubble_lateral_radius = (
    surface_bubbles["physical_radius"].astype(np.float64)
    * 0.5
    * (
        surface_bubbles["shape"][:, 0].astype(np.float64)
        + surface_bubbles["shape"][:, 2].astype(np.float64)
    )
)
surface_bubble_projected_area = float(
    np.sum(np.pi * surface_bubble_lateral_radius**2, dtype=np.float64)
)
resolved_cap = surface_bubbles["physical_radius"] >= SURFACE_CAP_RESOLUTION_M
resolved_surface_bubbles = surface_bubbles[resolved_cap]
resolved_surface_normals = surface_normals[resolved_cap]
resolved_surface_contacts = [
    surface_contacts[index] for index in np.flatnonzero(resolved_cap)
]
unresolved_surface_bubbles = surface_bubbles[~resolved_cap]
resolved_cap_projected_area = float(
    np.sum(np.pi * surface_bubble_lateral_radius[resolved_cap] ** 2, dtype=np.float64)
)
surface_bubble_obj, surface_dome_metrics = make_surface_domes(
    "WWV6_ResolvedSurfaceBubbleDomes",
    resolved_surface_bubbles,
    resolved_surface_normals,
    resolved_surface_contacts,
    surface_bubble_material,
    collection,
)
surface_density, surface_density_metrics = build_surface_bubble_density_atlas(
    surface_bubbles,
    origin,
    grid_maximum,
)
surface_density_material = make_density_overlay_material(
    "WWV6_UnresolvedSurfaceBubbleOpticalDepth",
    surface_density,
    surface_density_metrics,
)
surface_density_obj = make_surface_density_overlay(
    water,
    surface_density,
    surface_density_metrics,
    surface_density_material,
    collection,
)

micro = patches[patches["render_class"] == CLASS_MICRO]
clusters = patches[patches["render_class"] == CLASS_CLUSTER]
macro = patches[patches["render_class"] == CLASS_MACRO]
micro_obj, micro_area = make_film_patches("WWV6_MicroFoamDensityFilm", micro, micro_material, collection, 0.20, 16)
cluster_obj, cluster_cells, cluster_area = make_cluster_cells(clusters, cluster_material, collection)
macro_obj, macro_area = make_film_patches("WWV6_MacroFoamFilm", macro, macro_material, collection, 0.14, 40)
membrane_obj = None
if membrane is not None:
    membrane_obj = make_surface_membrane(
        membrane["vertices"],
        membrane["triangles"],
        membrane["triangle_normals"],
        membrane_material,
        collection,
    )
ring_obj, ring_area = make_irregular_rings(rings, ring_material, collection)
hole_mask_obj = make_hole_masks(rings, hole_material, collection)
marker_objects = (spray_obj, entrained_obj)
foam_objects = (
    surface_bubble_obj,
    surface_density_obj,
    micro_obj,
    cluster_obj,
    macro_obj,
    membrane_obj,
    ring_obj,
    hole_mask_obj,
)
if layer_mode in {"foam_only", "foam_isolation"}:
    for obj in marker_objects:
        obj.hide_render = True
elif layer_mode == "markers_only":
    for obj in foam_objects:
        if obj is not None:
            obj.hide_render = True

bpy.ops.mesh.primitive_uv_sphere_add(
    segments=64, ring_count=32, radius=0.08, location=tuple(isaac_to_blender(sphere_isaac))
)
sphere = bpy.context.active_object
sphere.name = "WWV6_Impactor"
sphere.data.materials.append(sphere_material)
for polygon in sphere.data.polygons:
    polygon.use_smooth = True
if layer_mode == "foam_isolation":
    water.hide_render = True

camera_data = bpy.data.cameras.new("WWV6_LookdevCamera")
camera = bpy.data.objects.new("WWV6_LookdevCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
target = Vector(isaac_to_blender((-0.73, -1.41099, 0.78)))
if view_mode == "overview":
    eye_isaac = (0.12, 1.15, -1.57)
    camera_data.lens = 40.0
else:
    eye_isaac = (-0.373, -0.335, -0.207)
    camera_data.lens = 55.0
eye = Vector(isaac_to_blender(eye_isaac))
camera.location = eye
camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
camera_data.sensor_width = 24.0

source_film_area = payload_source_film_area
represented_film_area = micro_area + cluster_area + macro_area + membrane_area + ring_area
unbound_film_area = unbound_patch_area + unbound_ring_area
report = {
    "schema": 1,
    "product": "whitewater_v6_actual_scale_lookdev_gate",
    "actual_scale": True,
    "pixel_guard_or_visibility_enlargement": False,
    "output_frame": output_frame,
    "source_sample_index": source_sample,
    "sphere_center_isaac": list(sphere_isaac),
    "counts": {
        "spray": len(spray),
        "entrained_bubbles": len(entrained),
        "surface_bubble_anchors": len(surface_bubbles),
        "source_surface_bubble_anchors": source_surface_bubble_count,
        "unbound_surface_bubble_anchors": len(unbound_surface_bubbles),
        "resolved_surface_bubble_domes": len(resolved_surface_bubbles),
        "surface_bubbles_in_density_base": len(surface_bubbles),
        "micro_film_parcels": len(micro),
        "foam_cluster_parcels": len(clusters),
        "foam_cluster_cells": cluster_cells,
        "macro_film_parcels": len(macro),
        "membrane_source_parcels": len(membrane_source_ids),
        "membrane_triangles": 0 if membrane is None else len(membrane["triangles"]),
        "hole_rings": len(rings),
    },
    "surface_bubble_gas_volume": {
        "source_m3": source_surface_gas_volume,
        "render_bound_m3": source_surface_gas_volume - unbound_surface_gas_volume,
        "unbound_m3": unbound_surface_gas_volume,
        "residual_m3": 0.0,
        "unbound_marker_ids": [
            int(value) for value in unbound_surface_bubbles["id"]
        ],
        "accounting": (
            "phase_volume is copied from immutable marker records; unbound gas "
            "is reported, not reassigned, enlarged, or silently deleted"
        ),
    },
    "film_area": {
        "source_m2": source_film_area,
        "represented_m2": represented_film_area,
        "unbound_m2": unbound_film_area,
        "residual_m2": source_film_area - represented_film_area - unbound_film_area,
    },
    "surface_bubble_gas_footprint": {
        "projected_m2": surface_bubble_projected_area,
        "projected_cm2": 1.0e4 * surface_bubble_projected_area,
        "resolved_cap_projected_m2": resolved_cap_projected_area,
        "ensemble_density_base": surface_density_metrics,
        "accounting": (
            "one gas ensemble with two render frequency bands: conservative "
            "density is the multiple-scattering base and resolved caps add only "
            "thin-film specular detail; their areas are not additive and remain "
            "separate from the conservative liquid-film ledger"
        ),
        "deformable_contact": rendered_contact_metrics,
        "resolved_dome_volume_proxy": surface_dome_metrics,
    },
    "representation": {
        "micro": "irregular tangent-film splats with geometric area equal to film_area",
        "cluster": "deterministic surface bubble cells; summed projected area equals film_area",
        "macro": "anisotropic irregular film patches with geometric area equal to film_area",
        "membrane": (
            "anisotropic density-ranked copies of retained Splashsurf triangles; "
            "geometric area equals the aggregated source parcel film area"
        ),
        "rings": "hosted intermittent Plateau-border arcs with visible area equal to film_area",
        "holes": "water-surface masks below hosted rims remove underlying foam geometry",
        "surface_bubbles": (
            "open, volume-proxy-conserving deformable caps: real-radius contact "
            "compresses the contact-facing footprint and raises cap height; "
            "render binding is rejected above 30 percent linear compression"
        ),
        "surface_bubble_multiscale": (
            "all caps deposit their conservative projected area into a fixed "
            "4 mm world-space optical-depth multiple-scattering base using "
            "coverage=1-exp(-tau); caps at or above 1.2 mm additionally provide "
            "non-additive explicit thin-film specular detail"
        ),
        "surface_bubble_raft_cells": (
            "5 mm world-scale Voronoi distance-to-edge scattering detail "
            "modulated by the conservative optical-depth coverage envelope"
        ),
    },
    "render_surface_binding": {
        "method": (
            "bidirectional local-normal ray cast; aligned nearest fallback; "
            "audited top-shell-only nearest fallback capped at 16 mm"
        ),
        "surface_bubbles": surface_binding,
        "foam_patches": patch_binding,
        "hole_rings": ring_binding,
    },
    "inputs": {
        "surface_obj": str(surface_path), "surface_sha256": sha256_file(surface_path),
        "marker_npz": str(marker_path), "marker_sha256": sha256_file(marker_path),
        "payload_npz": str(payload_path), "payload_sha256": sha256_file(payload_path),
        "liquid_npz": str(liquid_path), "liquid_sha256": sha256_file(liquid_path),
        "membrane_npz": None if membrane_path is None else str(membrane_path),
        "membrane_sha256": None if membrane_path is None else sha256_file(membrane_path),
        "surface_raft_npz": None if raft_path is None else str(raft_path),
        "surface_raft_sha256": None if raft_path is None else sha256_file(raft_path),
    },
    "render": {
        "engine": "CYCLES", "samples": samples, "resolution": [720,720],
        "view_mode": view_mode,
        "layer_mode": layer_mode,
        "hdri": str(hdri), "alpha_dither": False,
    },
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
scene.render.filepath = str(output_path)
bpy.ops.render.render(write_still=True)
print("WHITEWATER_V6_LOOKDEV_GATE=" + str(output_path))
print("WHITEWATER_V6_LOOKDEV_REPORT=" + str(report_path))
