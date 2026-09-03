"""Render synchronized multi-view observations for a fluid 4DGS episode."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def isaac_to_blender(values):
    x, y, z = values
    return Vector((float(x), float(-z), float(y)))


def blender_normal_to_isaac(array):
    result = np.empty_like(array, dtype=np.float32)
    result[..., 0] = array[..., 0]
    result[..., 1] = array[..., 2]
    result[..., 2] = -array[..., 1]
    return result


def set_principled_input(node, value, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return
    raise RuntimeError(f"Missing Principled input {names}")


def make_water_material():
    material = bpy.data.materials.get("Fluid4DGSWater") or bpy.data.materials.new(
        "Fluid4DGSWater"
    )
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(shader, (0.82, 0.94, 0.98, 1.0), "Base Color")
    set_principled_input(shader, 0.03, "Roughness")
    set_principled_input(shader, 1.333, "IOR")
    set_principled_input(shader, 1.0, "Transmission Weight", "Transmission")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_sphere_material():
    material = bpy.data.materials.get("Fluid4DGSImpactor") or bpy.data.materials.new(
        "Fluid4DGSImpactor"
    )
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(shader, (0.95, 0.18, 0.035, 1.0), "Base Color")
    set_principled_input(shader, 0.24, "Roughness")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_auxiliary_geometry_material():
    """Encode positive camera-axis Z into an opaque emission pass."""

    material = bpy.data.materials.get(
        "Fluid4DGSAuxiliaryGeometry"
    ) or bpy.data.materials.new("Fluid4DGSAuxiliaryGeometry")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    geometry = nodes.new("ShaderNodeNewGeometry")
    camera_offset = nodes.new("ShaderNodeVectorMath")
    camera_offset.name = "DepthCameraOffset"
    camera_offset.operation = "SUBTRACT"
    camera_forward = nodes.new("ShaderNodeVectorMath")
    camera_forward.name = "DepthCameraForward"
    camera_forward.operation = "DOT_PRODUCT"
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(
        geometry.outputs["Position"], camera_offset.inputs[0]
    )
    material.node_tree.links.new(
        camera_offset.outputs["Vector"], camera_forward.inputs[0]
    )
    material.node_tree.links.new(camera_forward.outputs["Value"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def configure_world(scene, hdri_path, strength):
    world = scene.world or bpy.data.worlds.new("Fluid4DGSWorld")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = float(strength)
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])


def configure_renderer(scene, manifest):
    rendering = manifest["rendering"]
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(rendering["resolution"][0])
    scene.render.resolution_y = int(rendering["resolution"][1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = float(rendering["exposure"])
    configure_world(
        scene,
        Path(rendering["lighting"]["hdri"]),
        rendering["lighting"]["strength"],
    )


def create_cameras(scene, cameras_payload):
    cameras = {}
    for item in cameras_payload["cameras"]:
        name = item["camera_name"]
        data = bpy.data.cameras.new(name + "_data")
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        matrix = np.asarray(item["camera_to_world"], dtype=np.float64)
        location = matrix[:3, 3]
        target = location + matrix[:3, 2]
        obj.location = isaac_to_blender(location)
        target_blender = isaac_to_blender(target)
        obj.rotation_euler = (target_blender - obj.location).to_track_quat("-Z", "Y").to_euler()
        data.lens = float(item["intrinsics"]["lens_mm"])
        data.sensor_width = float(item["intrinsics"]["sensor_width_mm"])
        data.sensor_fit = "HORIZONTAL"
        data.shift_x = 0.0
        data.shift_y = 0.0
        data.clip_start = float(item["near"])
        data.clip_end = float(item["far"])
        cameras[name] = obj
    return cameras


def import_water(path, material):
    bpy.ops.wm.obj_import(filepath=str(path))
    water = bpy.context.active_object
    water.name = "Fluid4DGSSurface"
    water.rotation_euler[0] = math.radians(90.0)
    water.data.materials.clear()
    water.data.materials.append(material)
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    return water


def remove_mesh_object(obj):
    if obj is None:
        return
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def isaac_object_to_blender_matrix(matrix):
    basis = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return basis @ np.asarray(matrix, dtype=np.float64) @ basis.T


def read_image(path):
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise RuntimeError(f"OpenImageIO could not open {path}: {oiio.geterror()}")
    try:
        specification = source.spec()
        pixels = source.read_image(oiio.FLOAT)
        array = np.asarray(pixels, dtype=np.float32).reshape(
            specification.height, specification.width, specification.nchannels
        )
        return array, list(specification.channelnames)
    finally:
        source.close()


def write_image(path, array, channel_names, pixel_type):
    array = np.ascontiguousarray(array)
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    specification = oiio.ImageSpec(width, height, channels, pixel_type)
    specification.channelnames = list(channel_names)
    output = oiio.ImageOutput.create(str(path))
    if output is None:
        raise RuntimeError(f"OpenImageIO could not create {path}: {oiio.geterror()}")
    try:
        if not output.open(str(path), specification):
            raise RuntimeError(output.geterror())
        # OpenImageIO's Blender binding interprets a shaped ndarray using
        # x-major strides.  Pass a flat C-contiguous buffer so rows remain
        # top-to-bottom and columns remain left-to-right; otherwise square
        # images are silently transposed.
        if not output.write_image(array.reshape(-1)):
            raise RuntimeError(output.geterror())
    finally:
        output.close()


def configure_auxiliary_outputs(scene, raw_directory):
    old_tree = scene.compositing_node_group
    tree = bpy.data.node_groups.new("Fluid4DGSCompositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    if old_tree is not None and old_tree.users == 0:
        bpy.data.node_groups.remove(old_tree)
    layers = tree.nodes.new("CompositorNodeRLayers")
    tree.interface.new_socket(
        name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
    )
    group_output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(layers.outputs["Image"], group_output.inputs["Image"])

    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.name = "Fluid4DGSDepth"
    depth.directory = str(raw_directory)
    depth.file_name = "depth_####"
    depth.file_output_items.new("RGBA", "Depth")
    depth.format.file_format = "OPEN_EXR_MULTILAYER"
    depth.format.color_depth = "32"
    depth.format.exr_codec = "ZIP"
    tree.links.new(layers.outputs["Image"], depth.inputs["Depth"])

    normal = tree.nodes.new("CompositorNodeOutputFile")
    normal.name = "Fluid4DGSNormal"
    normal.directory = str(raw_directory)
    normal.file_name = "normal_####"
    normal.file_output_items.new("VECTOR", "Normal")
    normal.format.file_format = "OPEN_EXR_MULTILAYER"
    normal.format.color_depth = "32"
    normal.format.exr_codec = "ZIP"
    tree.links.new(layers.outputs["Normal"], normal.inputs["Normal"])


def locate_raw(raw_directory, prefix):
    candidates = sorted(raw_directory.glob(prefix + "*.exr"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {prefix} EXR in {raw_directory}, got {candidates}")
    return candidates[0]


def build_world_bvh(water):
    """Build a BVH in Blender world coordinates from the rendered water mesh."""

    matrix = water.matrix_world
    vertices = [tuple(matrix @ vertex.co) for vertex in water.data.vertices]
    faces = [tuple(polygon.vertices) for polygon in water.data.polygons]
    return BVHTree.FromPolygons(vertices, faces, all_triangles=True, epsilon=0.0)


def blender_vector_from_isaac(value):
    return Vector((float(value[0]), float(-value[2]), float(value[1])))


def isaac_vector_from_blender(value):
    return np.asarray((float(value[0]), float(value[2]), float(-value[1])))


def postprocess_fluid_outputs(
    camera_payload, water_bvh, fluid_rgba_path, output_directory
):
    """Create center-ray geometry labels against the exact rendered mesh.

    Beauty and fluid RGBA remain antialiased renders.  The training labels use
    one unambiguous sample at every pixel center, so transparent shading,
    coverage premultiplication and overlapping thin surfaces cannot corrupt Z.
    """

    rgba, rgba_channels = read_image(fluid_rgba_path)
    if rgba.shape[2] < 4:
        raise RuntimeError(f"Auxiliary render has no alpha channel: {rgba_channels}")
    alpha = rgba[..., 3]
    alpha_mask = alpha > (1.0 / 255.0)
    if not np.any(alpha_mask):
        raise RuntimeError(f"Fluid render produced an empty alpha mask: {fluid_rgba_path}")

    height, width = alpha_mask.shape
    depth = np.zeros((height, width), dtype=np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    matrix = np.asarray(camera_payload["camera_to_world"], dtype=np.float64)
    rotation = matrix[:3, :3]
    center = matrix[:3, 3]
    origin = blender_vector_from_isaac(center)
    intrinsics = camera_payload["intrinsics"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    far = float(camera_payload["far"])

    for row, column in np.argwhere(alpha_mask):
        camera_direction = np.asarray(
            ((column + 0.5 - cx) / fx, (row + 0.5 - cy) / fy, 1.0),
            dtype=np.float64,
        )
        camera_direction /= np.linalg.norm(camera_direction)
        world_direction = rotation @ camera_direction
        hit, hit_normal, _, _ = water_bvh.ray_cast(
            origin, blender_vector_from_isaac(world_direction), far
        )
        if hit is None:
            continue
        hit_isaac = isaac_vector_from_blender(hit)
        optical_z = float(np.dot(hit_isaac - center, rotation[:, 2]))
        if not (float(camera_payload["near"]) < optical_z < far):
            continue
        geometric_normal = isaac_vector_from_blender(hit_normal)
        normal_length = float(np.linalg.norm(geometric_normal))
        if not np.isfinite(normal_length) or normal_length <= 1.0e-8:
            continue
        mask[row, column] = True
        depth[row, column] = optical_z
        normal[row, column] = (geometric_normal / normal_length).astype(np.float32)

    if not np.any(mask):
        raise RuntimeError(f"Every fluid-label center ray missed: {fluid_rgba_path}")

    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    write_image(output_directory / "depth.exr", depth, ["Z"], oiio.FLOAT)
    write_image(output_directory / "mask.png", mask_u8, ["Y"], oiio.UINT8)
    write_image(
        output_directory / "normal.exr",
        normal.astype(np.float32),
        ["Nx", "Ny", "Nz"],
        oiio.FLOAT,
    )
    return {
        "fluid_pixels": int(np.count_nonzero(mask)),
        "fluid_fraction": float(np.mean(mask)),
        "depth_minimum_metres": float(depth[mask].min()),
        "depth_maximum_metres": float(depth[mask].max()),
        "normal_length_minimum": float(np.linalg.norm(normal[mask], axis=1).min()),
        "normal_length_maximum": float(np.linalg.norm(normal[mask], axis=1).max()),
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 1:
        raise SystemExit("Expected EPISODE_DIRECTORY")
    episode = Path(argv[0]).resolve()
    manifest_path = episode / "manifest.json"
    manifest = load_json(manifest_path)
    cameras_payload = load_json(episode / "cameras.json")
    if manifest.get("product") != "fluid_4dgs_episode":
        raise ValueError("Unsupported episode manifest")

    scene = bpy.context.scene
    configure_renderer(scene, manifest)
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    cameras = create_cameras(scene, cameras_payload)
    water_material = make_water_material()
    sphere_material = make_sphere_material()
    initial_state = load_json(episode / manifest["frames"][0]["state"])
    sphere_radius = float(initial_state["rigid_bodies"][0]["radius_metres"])
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=sphere_radius, location=(0.0, 0.0, 0.0)
    )
    sphere = bpy.context.active_object
    sphere.name = "Fluid4DGSImpactor"
    sphere.data.materials.append(sphere_material)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True

    generated_camera_names = {camera.name for camera in cameras.values()}
    base_objects = [
        obj
        for obj in list(scene.objects)
        if obj.name not in generated_camera_names and obj != sphere
    ]
    original_hide = {obj.name: bool(obj.hide_render) for obj in base_objects}
    render_rows = []
    water = None
    frame_states = {}

    for frame_record in manifest["frames"]:
        frame_index = int(frame_record["frame_index"])
        state = load_json(episode / frame_record["state"])
        frame_states[frame_index] = state
        mesh_path = episode / frame_record["mesh"]
        remove_mesh_object(water)
        water = import_water(mesh_path, water_material)
        water_bvh = build_world_bvh(water)
        sphere_matrix = isaac_object_to_blender_matrix(
            state["rigid_bodies"][0]["object_to_world"]
        )
        sphere.matrix_world = Matrix(sphere_matrix.tolist())
        scene.frame_set(frame_index)

        for camera_payload in cameras_payload["cameras"]:
            camera_name = camera_payload["camera_name"]
            camera = cameras[camera_name]
            scene.camera = camera
            output_directory = (
                episode
                / "views"
                / f"frame_{frame_index:04d}"
                / camera_name
            )
            output_directory.mkdir(parents=True, exist_ok=True)

            # Full-scene beauty observation.
            scene.compositing_node_group = None
            scene.render.film_transparent = False
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            for obj in base_objects:
                obj.hide_render = original_hide[obj.name]
            water.hide_render = False
            sphere.hide_render = False
            scene.render.filepath = str(output_directory / "rgb.png")
            bpy.ops.render.render(write_still=True)

            # Fluid-only observation and geometric passes. Keep the HDRI for
            # stable illumination/reflection, but remove every scene object.
            for obj in base_objects:
                obj.hide_render = True
            sphere.hide_render = True
            water.hide_render = False
            scene.render.film_transparent = True
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGBA"
            scene.compositing_node_group = None
            fluid_rgba_path = output_directory / "fluid_rgba.png"
            scene.render.filepath = str(fluid_rgba_path)
            bpy.ops.render.render(write_still=True)

            # Semantic geometry labels are exact center-ray first hits against
            # the same mesh.  This avoids transparent-material Z-pass failures
            # and antialiasing/premultiplication at thin-surface boundaries.
            metrics = postprocess_fluid_outputs(
                camera_payload, water_bvh, fluid_rgba_path, output_directory
            )
            render_rows.append(
                {
                    "frame_index": frame_index,
                    "simulation_time": float(state["simulation_time"]),
                    "camera_name": camera_name,
                    **metrics,
                    "files": {
                        name: {
                            "path": str((output_directory / name).relative_to(episode)),
                            "sha256": sha256_file(output_directory / name),
                            "bytes": (output_directory / name).stat().st_size,
                        }
                        for name in (
                            "rgb.png",
                            "fluid_rgba.png",
                            "depth.exr",
                            "mask.png",
                            "normal.exr",
                        )
                    },
                }
            )
            print(
                f"[fluid-4dgs-render] frame={frame_index:04d} camera={camera_name} "
                f"fluid_pixels={metrics['fluid_pixels']}",
                flush=True,
            )

    # Save a self-contained frame-0 scene entry with the same fixed cameras,
    # world, materials and environment used for every observation.
    remove_mesh_object(water)
    first_frame = manifest["frames"][0]
    water = import_water(episode / first_frame["mesh"], water_material)
    first_state = frame_states[0]
    sphere.matrix_world = Matrix(
        isaac_object_to_blender_matrix(
            first_state["rigid_bodies"][0]["object_to_world"]
        ).tolist()
    )
    for obj in base_objects:
        obj.hide_render = original_hide[obj.name]
    sphere.hide_render = False
    water.hide_render = False
    scene.compositing_node_group = None
    scene.render.film_transparent = False
    scene.camera = cameras["cam_00"]
    scene.frame_set(0)
    scene_path = episode / "scene" / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path), compress=True)

    render_manifest = {
        "schema": 1,
        "product": "fluid_4dgs_render_outputs",
        "complete": len(render_rows) == 4 * 8,
        "blender_version": bpy.app.version_string,
        "engine": scene.render.engine,
        "label_sampling": "one BVH first-hit ray through each OpenCV pixel center with nonzero fluid RGBA coverage",
        "depth_conversion": "dot(first_hit_world - camera_center_world, OpenCV_camera_forward_world), metres",
        "normal_conversion": "first-hit geometric normal converted from Blender world (x,y,z) to Isaac world (x,z,-y)",
        "rows": render_rows,
        "scene": {
            "path": str(scene_path.relative_to(episode)),
            "sha256": sha256_file(scene_path),
            "bytes": scene_path.stat().st_size,
        },
    }
    atomic_json(episode / "render_manifest.json", render_manifest)
    manifest = load_json(manifest_path)
    manifest["state"] = {
        "complete": False,
        "rendered": render_manifest["complete"],
        "audited": False,
    }
    manifest["files"]["render_manifest"] = "render_manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"FLUID_4DGS_RENDER={episode} views={len(render_rows)}")


if __name__ == "__main__":
    main()
