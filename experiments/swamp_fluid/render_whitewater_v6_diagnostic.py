"""Render a layered Blender diagnostic for the audited whitewater-v6 caches.

This is deliberately a diagnostic renderer, not the final whitewater lookdev.
Every simulation/render class is a separate object and uses an unambiguous
false colour.  Sub-pixel primitives can be enlarged for inspection, but the
enlargement policy is written next to the PNG in a JSON report.

Run with Blender::

    blender swamp.blend --background --python render_whitewater_v6_diagnostic.py -- \
        SURFACE_OBJ MARKER_NPZ FOAM_PAYLOAD_NPZ OUTPUT_PNG \
        SPHERE_X SPHERE_Y SPHERE_Z [DISPLAY_SCALE]
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


argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) < 8:
    raise SystemExit(
        "Expected SURFACE_OBJ MARKER_NPZ FOAM_PAYLOAD_NPZ OUTPUT_PNG "
        "SPHERE_X SPHERE_Y SPHERE_Z DISPLAY_SCALE"
    )

surface_path = Path(argv[0]).resolve()
marker_path = Path(argv[1]).resolve()
payload_path = Path(argv[2]).resolve()
output_path = Path(argv[3]).resolve()
sphere_isaac = tuple(float(value) for value in argv[4:7])
display_scale = float(argv[7])
view_mode = argv[8].strip().lower() if len(argv) >= 9 else "context"
if not np.isfinite(display_scale) or display_scale <= 0.0:
    raise SystemExit("DISPLAY_SCALE must be finite and positive")
if view_mode not in {"context", "profile"}:
    raise SystemExit("VIEW_MODE must be 'context' or 'profile'")
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists() or output_path.with_suffix(".json").exists():
    raise RuntimeError(f"Refusing to overwrite existing diagnostic: {output_path}")


STATE_SPRAY = 1
STATE_ENTRAINED_BUBBLE = 3
STATE_SURFACE_BUBBLE = 4
CLASS_MICRO_DENSITY = 0
CLASS_BUBBLE_CLUSTER = 1
CLASS_MACRO_PATCH = 2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isaac_to_blender(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (3,):
        return np.asarray((values[0], -values[2], values[1]), dtype=np.float64)
    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def set_principled_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return True
    return False


def make_material(name, colour, roughness=0.34, metallic=0.0, emission=0.0, alpha=1.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*colour, alpha)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(principled, (*colour, alpha), "Base Color")
    set_principled_input(principled, roughness, "Roughness")
    set_principled_input(principled, metallic, "Metallic")
    set_principled_input(principled, alpha, "Alpha")
    if emission > 0.0:
        set_principled_input(principled, (*colour, 1.0), "Emission Color", "Emission")
        set_principled_input(principled, emission, "Emission Strength")
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    if alpha < 1.0:
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "BLEND"
        if hasattr(material, "use_transparency_overlap"):
            material.use_transparency_overlap = False
    return material


def configure_scene(scene):
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(output_path)
    scene.render.image_settings.color_depth = "8"
    scene.render.engine = scene.render.engine
    scene.render.filepath = str(output_path)
    scene.render.use_file_extension = True
    scene.view_settings.look = "AgX - Medium High Contrast"


def configure_hdri(scene):
    hdri_path = Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr")
    if not hdri_path.is_file():
        return None
    world = scene.world or bpy.data.worlds.new("WhitewaterV6DiagnosticWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 0.65
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return str(hdri_path)


def link_mesh_object(name, vertices, faces, material, collection):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=False)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.data.materials.append(material)
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


def make_marker_layer(name, records, material, collection, minimum_radius, shape_bubbles):
    if not len(records):
        return None
    centres = isaac_to_blender(records["position"])
    radii = np.maximum(
        records["physical_radius"].astype(np.float64) * display_scale,
        minimum_radius,
    )
    shapes = np.ones((len(records), 3), dtype=np.float64)
    if shape_bubbles:
        # Shape is stored in Isaac x/y/z; Blender uses x/-z/y.
        source_shape = records["shape"].astype(np.float64)
        shapes = source_shape[:, (0, 2, 1)]
    vertices = []
    faces = []
    vertex_count = len(ICO_VERTICES)
    for index, (centre, radius, shape) in enumerate(zip(centres, radii, shapes)):
        offset = len(vertices)
        local = centre + ICO_VERTICES * (radius * shape)[None, :]
        vertices.extend(map(tuple, local))
        faces.extend(tuple(offset + value for value in face) for face in ICO_FACES)
    obj = link_mesh_object(name, vertices, faces, material, collection)
    obj["physical_count"] = int(len(records))
    obj["display_scale"] = float(display_scale)
    obj["minimum_display_radius_m"] = float(minimum_radius)
    return obj


def tangent_frame(normal, principal):
    normal = np.asarray(normal, dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1.0e-12)
    tangent = np.asarray(principal, dtype=np.float64)
    tangent -= np.dot(tangent, normal) * normal
    if np.linalg.norm(tangent) < 1.0e-8:
        reference = np.asarray((1.0, 0.0, 0.0))
        if abs(normal[0]) > 0.9:
            reference = np.asarray((0.0, 0.0, 1.0))
        tangent = reference - np.dot(reference, normal) * normal
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)
    return normal, tangent, bitangent


def make_patch_layer(name, patches, material, collection, minimum_radius, normal_offset):
    if not len(patches):
        return None
    vertices = []
    faces = []
    segments = 20
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    for record in patches:
        p = isaac_to_blender(record["position"])
        n = isaac_to_blender(record["normal"])
        t = isaac_to_blender(record["principal_direction"])
        n, t, b = tangent_frame(n, t)
        p = p + normal_offset * n
        major = max(float(record["major_radius"]) * display_scale, minimum_radius)
        minor = max(float(record["minor_radius"]) * display_scale, minimum_radius * 0.70)
        centre_id = len(vertices)
        vertices.append(tuple(p))
        for angle in angles:
            vertices.append(tuple(p + major * math.cos(angle) * t + minor * math.sin(angle) * b))
        for segment in range(segments):
            faces.append(
                (centre_id, centre_id + 1 + segment, centre_id + 1 + (segment + 1) % segments)
            )
    obj = link_mesh_object(name, vertices, faces, material, collection)
    obj["physical_count"] = int(len(patches))
    obj["display_scale"] = float(display_scale)
    obj["minimum_display_radius_m"] = float(minimum_radius)
    return obj


def make_ring_layer(name, rings, material, collection, minimum_width, normal_offset):
    if not len(rings):
        return None
    vertices = []
    faces = []
    segments = 28
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    for record in rings:
        p = isaac_to_blender(record["position"])
        n = isaac_to_blender(record["normal"])
        n, t, b = tangent_frame(n, (1.0, 0.0, 0.0))
        p = p + normal_offset * n
        inner = float(record["inner_radius"]) * display_scale
        physical_width = float(record["outer_radius"] - record["inner_radius"])
        outer = inner + max(physical_width * display_scale, minimum_width)
        start = len(vertices)
        for radius in (inner, outer):
            for angle in angles:
                vertices.append(tuple(p + radius * math.cos(angle) * t + radius * math.sin(angle) * b))
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append(
                (start + segment, start + nxt, start + segments + nxt, start + segments + segment)
            )
    obj = link_mesh_object(name, vertices, faces, material, collection)
    obj["physical_count"] = int(len(rings))
    obj["display_scale"] = float(display_scale)
    obj["minimum_display_width_m"] = float(minimum_width)
    return obj


def add_impactor(collection, material):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=48,
        ring_count=24,
        radius=0.08,
        location=tuple(isaac_to_blender(sphere_isaac)),
    )
    obj = bpy.context.active_object
    obj.name = "WWV6_ImpactSphere"
    # Move from the scene collection into the diagnostic collection.
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


with np.load(marker_path) as cache:
    marker_schema = int(cache["schema"])
    marker_output_frame = int(cache["output_frame"])
    marker_source_sample = int(cache["source_sample_index"])
    markers = np.asarray(cache["markers"])
with np.load(payload_path) as cache:
    payload_schema = int(cache["schema"])
    payload_output_frame = int(cache["output_frame"])
    payload_source_sample = int(cache["source_sample_index"])
    patches = np.asarray(cache["patches"])
    rings = np.asarray(cache["rings"])

if marker_schema != 1 or payload_schema != 1:
    raise RuntimeError("Unsupported marker or payload schema")
if (marker_output_frame, marker_source_sample) != (payload_output_frame, payload_source_sample):
    raise RuntimeError("Marker and foam payload samples do not match")

scene = bpy.context.scene
configure_scene(scene)
hdri = configure_hdri(scene)

diagnostic_collection = bpy.data.collections.new("WhitewaterV6_Diagnostic")
scene.collection.children.link(diagnostic_collection)

materials = {
    "water": make_material("WWV6_WaterContext", (0.035, 0.25, 0.34), 0.16, alpha=0.34),
    "spray": make_material("WWV6_Spray_Gold", (1.0, 0.29, 0.015), 0.22, emission=0.16),
    "entrained": make_material("WWV6_EntrainedBubble_Blue", (0.03, 0.21, 1.0), 0.20, emission=0.14),
    "surface": make_material("WWV6_SurfaceBubble_Cyan", (0.0, 0.95, 0.82), 0.20, emission=0.14),
    "micro": make_material("WWV6_MicroDensity_White", (1.0, 1.0, 1.0), 0.40, emission=0.10),
    "cluster": make_material("WWV6_BubbleCluster_Lime", (0.30, 1.0, 0.03), 0.34, emission=0.12),
    "macro": make_material("WWV6_MacroPatch_Magenta", (1.0, 0.02, 0.58), 0.32, emission=0.14),
    "ring": make_material("WWV6_HoleRing_Red", (0.92, 0.0, 0.025), 0.28, emission=0.20),
    "sphere": make_material("WWV6_Impactor_Charcoal", (0.025, 0.025, 0.025), 0.25, metallic=0.55),
}

bpy.ops.wm.obj_import(filepath=str(surface_path))
water = bpy.context.active_object
water.name = "WWV6_LiquidSurface_Context"
water.rotation_euler[0] = math.radians(90.0)
for owner in list(water.users_collection):
    owner.objects.unlink(water)
diagnostic_collection.objects.link(water)
water.data.materials.clear()
water.data.materials.append(materials["water"])
for polygon in water.data.polygons:
    polygon.use_smooth = True

spray = markers[markers["state"] == STATE_SPRAY]
entrained = markers[markers["state"] == STATE_ENTRAINED_BUBBLE]
surface_bubbles = markers[markers["state"] == STATE_SURFACE_BUBBLE]
micro = patches[patches["render_class"] == CLASS_MICRO_DENSITY]
clusters = patches[patches["render_class"] == CLASS_BUBBLE_CLUSTER]
macro = patches[patches["render_class"] == CLASS_MACRO_PATCH]

# These lower bounds are diagnostic pixel guards, not physical radii.
make_marker_layer("WWV6_Spray", spray, materials["spray"], diagnostic_collection, 0.0010, False)
make_marker_layer("WWV6_EntrainedBubbles", entrained, materials["entrained"], diagnostic_collection, 0.0012, True)
make_marker_layer("WWV6_SurfaceBubbles", surface_bubbles, materials["surface"], diagnostic_collection, 0.0013, True)
make_patch_layer("WWV6_MicroFoamDensity", micro, materials["micro"], diagnostic_collection, 0.0010, 0.00050)
make_patch_layer("WWV6_BubbleClusters", clusters, materials["cluster"], diagnostic_collection, 0.0011, 0.00065)
make_patch_layer("WWV6_MacroFoamPatches", macro, materials["macro"], diagnostic_collection, 0.0012, 0.00080)
make_ring_layer("WWV6_ConservativeHoleRings", rings, materials["ring"], diagnostic_collection, 0.00055, 0.00105)
add_impactor(diagnostic_collection, materials["sphere"])

camera_data = bpy.data.cameras.new("WWV6_DiagnosticCamera")
camera = bpy.data.objects.new("WWV6_DiagnosticCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
if view_mode == "context":
    eye_isaac = (0.12, 1.15, -1.57)
    target_isaac = (-0.73, -1.41099, 0.78)
    camera_data.lens = 48.0
else:
    # Isolate the solver layers from authored terrain so vertical placement,
    # liquid-interface crossings and stray markers remain visible.
    for obj in scene.objects:
        if obj.type == "MESH" and diagnostic_collection not in obj.users_collection:
            obj.hide_render = True
    eye_isaac = (-0.73, -1.24, -1.80)
    target_isaac = (-0.73, -1.37, 0.78)
    camera_data.lens = 40.0
eye = Vector(isaac_to_blender(eye_isaac))
target = Vector(isaac_to_blender(target_isaac))
camera.location = eye
camera.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
camera_data.sensor_width = 24.0

report = {
    "schema": 1,
    "product": "whitewater_v6_layered_diagnostic",
    "diagnostic_only": True,
    "false_colour": True,
    "output_frame": marker_output_frame,
    "source_sample_index": marker_source_sample,
    "sphere_center_isaac": list(sphere_isaac),
    "inputs": {
        "surface_obj": str(surface_path),
        "surface_obj_sha256": sha256_file(surface_path),
        "marker_npz": str(marker_path),
        "marker_npz_sha256": sha256_file(marker_path),
        "foam_payload_npz": str(payload_path),
        "foam_payload_npz_sha256": sha256_file(payload_path),
    },
    "counts": {
        "spray": int(len(spray)),
        "entrained_bubbles": int(len(entrained)),
        "surface_bubbles": int(len(surface_bubbles)),
        "micro_density": int(len(micro)),
        "bubble_clusters": int(len(clusters)),
        "macro_patches": int(len(macro)),
        "hole_rings": int(len(rings)),
    },
    "display_policy": {
        "global_scale": display_scale,
        "marker_minimum_radius_m": {
            "spray": 0.0010,
            "entrained_bubbles": 0.0012,
            "surface_bubbles": 0.0013,
        },
        "patch_minimum_major_radius_m": {
            "micro_density": 0.0010,
            "bubble_clusters": 0.0011,
            "macro_patches": 0.0012,
        },
        "ring_minimum_width_m": 0.00055,
        "note": "Pixel guards enlarge sub-pixel diagnostics; simulation caches are unchanged.",
    },
    "palette": {
        "spray": "gold",
        "entrained_bubbles": "blue",
        "surface_bubbles": "cyan",
        "micro_density": "white",
        "bubble_clusters": "lime",
        "macro_patches": "magenta",
        "hole_rings": "red",
    },
    "render": {
        "engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "hdri": hdri,
        "view_mode": view_mode,
        "camera_lens_mm": camera_data.lens,
    },
}
report_path = output_path.with_suffix(".json")
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

scene.render.filepath = str(output_path)
bpy.ops.render.render(write_still=True)
print("WHITEWATER_V6_DIAGNOSTIC=" + str(output_path))
print("WHITEWATER_V6_DIAGNOSTIC_REPORT=" + str(report_path))
