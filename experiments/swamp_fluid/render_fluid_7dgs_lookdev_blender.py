"""Render the high-quality 3x6 fluid 7DGS visual acceptance gate."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_fluid_4dgs_episode_blender import (
    atomic_json,
    blender_vector_from_isaac,
    build_world_bvh,
    create_cameras,
    import_water,
    isaac_object_to_blender_matrix,
    isaac_vector_from_blender,
    load_json,
    read_image,
    remove_mesh_object,
    set_principled_input,
    sha256_file,
    write_image,
)


def configure_cycles(scene, manifest):
    settings = manifest["rendering"]
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(settings["resolution"][0])
    scene.render.resolution_y = int(settings["resolution"][1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGB"
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    if hasattr(scene.cycles, "film_transparent_glass"):
        scene.cycles.film_transparent_glass = False
    scene.cycles.samples = int(settings["cycles_samples"])
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.008
    scene.cycles.seed = 20260831
    if hasattr(scene.cycles, "use_animated_seed"):
        scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.volume_bounces = 2
    scene.cycles.sample_clamp_indirect = 12.0
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    render_device = "CPU"
    enabled_devices = []
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type in {"OPTIX", "CUDA"}
            if device.use:
                enabled_devices.append({"name": device.name, "type": device.type})
        if enabled_devices:
            scene.cycles.device = "GPU"
            render_device = "GPU"
    except (KeyError, TypeError, RuntimeError):
        scene.cycles.device = "CPU"
    return render_device, enabled_devices


def configure_world(scene, hdri_path, strength):
    world = scene.world or bpy.data.worlds.new("Fluid7DGSWorld")
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
    return world


def make_water_material():
    material = bpy.data.materials.get("Fluid7DGSWater") or bpy.data.materials.new(
        "Fluid7DGSWater"
    )
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.name = "PhysicalWater"
    set_principled_input(shader, (0.965, 0.987, 1.0, 1.0), "Base Color")
    set_principled_input(shader, 0.018, "Roughness")
    set_principled_input(shader, 1.333, "IOR")
    set_principled_input(shader, 1.0, "Transmission Weight", "Transmission")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "SubpixelCapillaryNormal"
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = 48.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.58
    bump = nodes.new("ShaderNodeBump")
    bump.name = "SubpixelCapillaryBump"
    bump.inputs["Strength"].default_value = 0.075
    bump.inputs["Distance"].default_value = 0.0015
    material.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    material.node_tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material.diffuse_color = (0.965, 0.987, 1.0, 1.0)
    return material


def make_principled_material(name, color, roughness):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    set_principled_input(shader, color, "Base Color")
    set_principled_input(shader, roughness, "Roughness")
    material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_emission_material(name, value):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (value, value, value, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def save_render_result(scene, path, file_format, color_mode, color_depth):
    path = Path(path)
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = color_mode
    scene.render.image_settings.color_depth = str(color_depth)
    if file_format in {"OPEN_EXR", "OPEN_EXR_MULTILAYER"}:
        scene.render.image_settings.exr_codec = "ZIP"
    bpy.data.images["Render Result"].save_render(filepath=str(path), scene=scene)


def render_linear_and_preview(scene, linear_path, preview_path=None, rgba=False):
    bpy.ops.render.render()
    save_render_result(
        scene, linear_path, "OPEN_EXR", "RGBA" if rgba else "RGB", "16"
    )
    if preview_path is not None:
        save_render_result(scene, preview_path, "PNG", "RGBA" if rgba else "RGB", "8")


def restore_original_visibility(objects, original_hide):
    for obj in objects:
        obj.hide_render = original_hide[obj.name]


def downsample_two(array):
    height, width = array.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("Two-times supersample image has odd dimensions")
    if array.ndim == 2:
        return array.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
    return array.reshape(height // 2, 2, width // 2, 2, array.shape[2]).mean(
        axis=(1, 3)
    )


def render_visible_mask(
    scene,
    water,
    base_objects,
    sphere,
    black_material,
    white_material,
    output_path,
    temporary_path,
    width,
    height,
    production_samples,
):
    del black_material, white_material
    objects = [*base_objects, sphere, water]
    original_pass_indices = {obj.name: int(obj.pass_index) for obj in objects}
    view_layer = scene.view_layers[0]
    original_index_pass = bool(view_layer.use_pass_object_index)
    view_layer.use_pass_object_index = True
    original_compositor = scene.compositing_node_group
    compositor = bpy.data.node_groups.new(
        name="Fluid7DGSVisibleIndexCompositor", type="CompositorNodeTree"
    )
    render_layers = compositor.nodes.new("CompositorNodeRLayers")
    index_output = compositor.nodes.new("CompositorNodeOutputFile")
    index_output.directory = str(Path(temporary_path).parent)
    index_output.file_name = ".visible_index_####"
    index_output.file_output_items.new("FLOAT", "IndexOB")
    index_output.format.file_format = "OPEN_EXR_MULTILAYER"
    index_output.format.color_depth = "32"
    index_output.format.exr_codec = "ZIP"
    compositor.links.new(
        render_layers.outputs["Object Index"], index_output.inputs["IndexOB"]
    )
    try:
        for obj in objects:
            obj.pass_index = 0
        water.pass_index = 1
        view_layer.use_pass_object_index = True
        scene.compositing_node_group = compositor
        scene.render.resolution_x = width * 2
        scene.render.resolution_y = height * 2
        scene.cycles.samples = max(32, min(64, production_samples))
        scene.cycles.use_denoising = False
        scene.render.film_transparent = False
        bpy.ops.render.render()
        candidates = sorted(Path(temporary_path).parent.glob(".visible_index_*.exr"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one compositor IndexOB output, got {candidates}")
        generated_index_path = candidates[0]
        image, channel_names = read_image(generated_index_path)
        if image.shape[2] < 1:
            raise RuntimeError(f"Empty IndexOB output: {channel_names}")
        supersample = np.isclose(image[..., 0], 1.0, atol=0.25)
        coverage = np.clip(downsample_two(supersample.astype(np.float32)), 0.0, 1.0)
        write_image(output_path, np.rint(coverage * 255.0).astype(np.uint8), ["Y"], oiio.UINT8)
        generated_index_path.unlink()
    finally:
        for obj in objects:
            obj.pass_index = original_pass_indices[obj.name]
        view_layer.use_pass_object_index = original_index_pass
        scene.compositing_node_group = original_compositor
        bpy.data.node_groups.remove(compositor)
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.cycles.samples = production_samples
        scene.cycles.use_denoising = True
        if Path(temporary_path).exists():
            Path(temporary_path).unlink()


def render_fluid_geometry_labels(camera, frame_index, water_bvh, output_directory):
    width = int(camera["width"])
    height = int(camera["height"])
    depth = np.zeros((height, width), dtype=np.float32)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    matrix = np.asarray(camera["camera_to_world"], dtype=np.float64)
    rotation = matrix[:3, :3]
    center = matrix[:3, 3]
    origin = blender_vector_from_isaac(center)
    intrinsics = camera["intrinsics"]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    bbox = np.asarray(camera["predicted_mesh_bboxes"][frame_index]["pixels"])
    x0 = max(0, int(math.floor(bbox[0])) - 2)
    y0 = max(0, int(math.floor(bbox[1])) - 2)
    x1 = min(width - 1, int(math.ceil(bbox[2])) + 2)
    y1 = min(height - 1, int(math.ceil(bbox[3])) + 2)
    near = float(camera["near"])
    far = float(camera["far"])
    for row in range(y0, y1 + 1):
        for column in range(x0, x1 + 1):
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
            hit_world = isaac_vector_from_blender(hit)
            optical_z = float(np.dot(hit_world - center, rotation[:, 2]))
            if not near < optical_z < far:
                continue
            hit_normal_world = isaac_vector_from_blender(hit_normal)
            length = float(np.linalg.norm(hit_normal_world))
            if not np.isfinite(length) or length <= 1.0e-8:
                continue
            mask[row, column] = True
            depth[row, column] = optical_z
            normal[row, column] = (hit_normal_world / length).astype(np.float32)
    if not np.any(mask):
        raise RuntimeError(f"Empty fluid geometry label for {camera['camera_name']}")
    write_image(
        output_directory / "fluid_mask.png",
        np.where(mask, 255, 0).astype(np.uint8),
        ["Y"],
        oiio.UINT8,
    )
    write_image(
        output_directory / "fluid_depth.exr", depth, ["Z"], oiio.FLOAT
    )
    write_image(
        output_directory / "fluid_normal.exr",
        normal,
        ["Nx", "Ny", "Nz"],
        oiio.FLOAT,
    )
    rows, columns = np.nonzero(mask)
    return {
        "fluid_pixels": int(np.count_nonzero(mask)),
        "fluid_bbox_pixels": [
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        ],
        "fluid_bbox_width_fraction": float((columns.max() - columns.min() + 1) / width),
        "fluid_bbox_height_fraction": float((rows.max() - rows.min() + 1) / height),
        "depth_minimum_metres": float(depth[mask].min()),
        "depth_maximum_metres": float(depth[mask].max()),
    }


def image_to_float(path):
    image, _ = read_image(path)
    return image


def downsample_three(array):
    height, width = array.shape[:2]
    cropped = array[: height - height % 3, : width - width % 3]
    if array.ndim == 2:
        return cropped.reshape(height // 3, 3, width // 3, 3).mean(axis=(1, 3))
    return cropped.reshape(height // 3, 3, width // 3, 3, array.shape[2]).mean(
        axis=(1, 3)
    )


def resize_nearest(array, output_height, output_width):
    rows = np.minimum(
        (np.arange(output_height) * array.shape[0] / output_height).astype(int),
        array.shape[0] - 1,
    )
    columns = np.minimum(
        (np.arange(output_width) * array.shape[1] / output_width).astype(int),
        array.shape[1] - 1,
    )
    return array[rows[:, None], columns[None, :]]


def make_contact_sheet(episode, render_rows, width, height):
    tile = 256
    columns_count = 6
    rows_count = len(render_rows)
    canvas = np.zeros((rows_count * tile, columns_count * tile, 3), dtype=np.float32)
    all_depth = []
    for row in render_rows:
        directory = episode / row["directory"]
        depth = image_to_float(directory / "fluid_depth.exr")[..., 0]
        mask = depth > 0.0
        all_depth.append(depth[mask])
    depth_values = np.concatenate(all_depth)
    depth_min = float(np.quantile(depth_values, 0.01))
    depth_max = float(np.quantile(depth_values, 0.99))
    for row_index, row in enumerate(render_rows):
        directory = episode / row["directory"]
        preview = image_to_float(directory / "rgb_preview.png")[..., :3]
        background = image_to_float(
            episode / "diagnostics" / "background_previews" / row["background_preview"]
        )[..., :3]
        visible = image_to_float(directory / "visible_fluid_mask.png")[..., 0]
        if visible.max() > 1.0:
            visible /= 255.0
        depth = image_to_float(directory / "fluid_depth.exr")[..., 0]
        normal = image_to_float(directory / "fluid_normal.exr")[..., :3]
        fluid_mask = depth > 0.0
        rows, cols = np.nonzero(fluid_mask)
        pad = 18
        y0 = max(0, int(rows.min()) - pad)
        y1 = min(height, int(rows.max()) + pad + 1)
        x0 = max(0, int(cols.min()) - pad)
        x1 = min(width, int(cols.max()) + pad + 1)
        zoom = resize_nearest(preview[y0:y1, x0:x1], tile, tile)
        depth_viz = np.zeros_like(depth)
        depth_viz[fluid_mask] = np.clip(
            (depth_max - depth[fluid_mask]) / max(depth_max - depth_min, 1.0e-6),
            0.0,
            1.0,
        )
        normal_viz = np.where(fluid_mask[..., None], 0.5 * normal + 0.5, 0.0)
        panels = (
            downsample_three(preview),
            downsample_three(background),
            zoom,
            np.repeat(downsample_three(visible)[..., None], 3, axis=2),
            np.repeat(downsample_three(depth_viz)[..., None], 3, axis=2),
            downsample_three(normal_viz),
        )
        for column_index, panel in enumerate(panels):
            canvas[
                row_index * tile : (row_index + 1) * tile,
                column_index * tile : (column_index + 1) * tile,
            ] = np.clip(panel[..., :3], 0.0, 1.0)
    path = episode / "diagnostics" / "contact_sheet.png"
    write_image(path, np.rint(canvas * 255.0).astype(np.uint8), ["R", "G", "B"], oiio.UINT8)
    atomic_json(
        episode / "diagnostics" / "contact_sheet_layout.json",
        {
            "columns": [
                "rgb_with_fluid_AgX_preview",
                "background_AgX_preview",
                "water_bbox_zoom",
                "visible_fluid_mask",
                "inverse_optical_z_depth_global_scale",
                "world_normal_xyz_mapped_to_rgb",
            ],
            "row_order": [
                {
                    "row": index,
                    "frame_index": row["frame_index"],
                    "camera_name": row["camera_name"],
                }
                for index, row in enumerate(render_rows)
            ],
            "depth_visualization_range_metres": [depth_min, depth_max],
        },
    )
    return path


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 1:
        raise SystemExit("Expected LOOKDEV_DIRECTORY")
    episode = Path(argv[0]).resolve()
    manifest_path = episode / "manifest.json"
    manifest = load_json(manifest_path)
    cameras_payload = load_json(episode / "cameras.json")
    if manifest.get("product") != "fluid_7dgs_visual_acceptance_gate":
        raise ValueError("Unsupported lookdev manifest")
    width, height = map(int, manifest["rendering"]["resolution"])
    production_samples = int(manifest["rendering"]["cycles_samples"])
    scene = bpy.context.scene
    render_device, enabled_devices = configure_cycles(scene, manifest)
    configure_world(
        scene,
        Path(manifest["rendering"]["hdri"]),
        manifest["rendering"]["hdri_strength"],
    )
    cameras = create_cameras(scene, cameras_payload)
    water_material = make_water_material()
    sphere_material = make_principled_material(
        "Fluid7DGSImpactor", (0.95, 0.18, 0.035, 1.0), 0.22
    )
    black_material = make_emission_material("Fluid7DGSMaskBlack", 0.0)
    white_material = make_emission_material("Fluid7DGSMaskWhite", 1.0)
    first_state = load_json(episode / manifest["frames"][0]["state"])
    sphere_radius = float(first_state["rigid_bodies"][0]["radius_metres"])
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=sphere_radius, location=(0.0, 0.0, 0.0)
    )
    sphere = bpy.context.active_object
    sphere.name = "Fluid7DGSImpactor"
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
    background_preview_directory = episode / "diagnostics" / "background_previews"
    background_preview_directory.mkdir(parents=True, exist_ok=True)
    render_rows = []
    water = None
    for frame_record in manifest["frames"]:
        frame_index = int(frame_record["frame_index"])
        state = load_json(episode / frame_record["state"])
        remove_mesh_object(water)
        water = import_water(episode / frame_record["mesh"], water_material)
        water_bvh = build_world_bvh(water)
        sphere.matrix_world = Matrix(
            isaac_object_to_blender_matrix(
                state["rigid_bodies"][0]["object_to_world"]
            ).tolist()
        )
        scene.frame_set(frame_index)
        for camera_payload in cameras_payload["cameras"]:
            camera_name = camera_payload["camera_name"]
            scene.camera = cameras[camera_name]
            output_directory = (
                episode / "views" / f"frame_{frame_index:04d}" / camera_name
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            restore_original_visibility(base_objects, original_hide)
            sphere.hide_render = False
            water.hide_render = False
            scene.render.film_transparent = False
            scene.cycles.samples = production_samples
            scene.cycles.use_denoising = True
            with_fluid_path = output_directory / "rgb_with_fluid_linear.exr"
            preview_path = output_directory / "rgb_preview.png"
            render_linear_and_preview(scene, with_fluid_path, preview_path, rgba=False)

            water.hide_render = True
            background_path = output_directory / "rgb_background_linear.exr"
            background_preview_name = f"frame_{frame_index:04d}_{camera_name}.png"
            render_linear_and_preview(
                scene,
                background_path,
                background_preview_directory / background_preview_name,
                rgba=False,
            )

            for obj in base_objects:
                obj.hide_render = True
            sphere.hide_render = True
            water.hide_render = False
            scene.render.film_transparent = True
            fluid_rgba_path = output_directory / "fluid_rgba_linear.exr"
            render_linear_and_preview(scene, fluid_rgba_path, None, rgba=True)

            restore_original_visibility(base_objects, original_hide)
            sphere.hide_render = False
            water.hide_render = False
            scene.render.film_transparent = False
            visible_mask_path = output_directory / "visible_fluid_mask.png"
            temporary_mask = output_directory / ".visible_mask_supersample.exr"
            render_visible_mask(
                scene,
                water,
                base_objects,
                sphere,
                black_material,
                white_material,
                visible_mask_path,
                temporary_mask,
                width,
                height,
                production_samples,
            )
            label_metrics = render_fluid_geometry_labels(
                camera_payload, frame_index, water_bvh, output_directory
            )
            visible_mask, _ = read_image(visible_mask_path)
            visible = visible_mask[..., 0] > (1.0 / 255.0)
            visible_rows, visible_columns = np.nonzero(visible)
            visible_bbox = [
                int(visible_columns.min()),
                int(visible_rows.min()),
                int(visible_columns.max()),
                int(visible_rows.max()),
            ]
            files = {}
            for name in (
                "rgb_with_fluid_linear.exr",
                "rgb_background_linear.exr",
                "rgb_preview.png",
                "visible_fluid_mask.png",
                "fluid_rgba_linear.exr",
                "fluid_mask.png",
                "fluid_depth.exr",
                "fluid_normal.exr",
            ):
                path = output_directory / name
                files[name] = {
                    "path": str(path.relative_to(episode)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            render_rows.append(
                {
                    "frame_index": frame_index,
                    "simulation_time": float(state["simulation_time"]),
                    "normalized_time_7dgs": float(state["normalized_time_7dgs"]),
                    "camera_name": camera_name,
                    "elevation_tier": camera_payload["elevation_tier"],
                    "directory": str(output_directory.relative_to(episode)),
                    "background_preview": background_preview_name,
                    "visible_fluid_bbox_pixels": visible_bbox,
                    "visible_fluid_pixels_nonzero_coverage": int(np.count_nonzero(visible)),
                    **label_metrics,
                    "files": files,
                }
            )
            print(
                f"[fluid-7dgs-lookdev] frame={frame_index:04d} camera={camera_name} "
                f"fluid={label_metrics['fluid_bbox_width_fraction']:.3f}x"
                f"{label_metrics['fluid_bbox_height_fraction']:.3f}",
                flush=True,
            )
    contact_sheet = make_contact_sheet(episode, render_rows, width, height)
    remove_mesh_object(water)
    first_frame = manifest["frames"][0]
    water = import_water(episode / first_frame["mesh"], water_material)
    first_state = load_json(episode / first_frame["state"])
    sphere.matrix_world = Matrix(
        isaac_object_to_blender_matrix(
            first_state["rigid_bodies"][0]["object_to_world"]
        ).tolist()
    )
    restore_original_visibility(base_objects, original_hide)
    sphere.hide_render = False
    water.hide_render = False
    scene.render.film_transparent = False
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.cycles.samples = production_samples
    scene.camera = cameras["cam_00"]
    scene.frame_set(0)
    scene_path = episode / "scene" / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path), compress=True)
    render_manifest = {
        "schema": 1,
        "product": "fluid_7dgs_lookdev_render_outputs",
        "complete": len(render_rows) == 18,
        "blender_version": bpy.app.version_string,
        "engine": scene.render.engine,
        "cycles_device": render_device,
        "enabled_devices": enabled_devices,
        "cycles_samples": production_samples,
        "adaptive_sampling_threshold": 0.008,
        "denoising": True,
        "motion_blur": False,
        "max_bounces": 12,
        "transmission_bounces": 12,
        "visible_mask_sampling": "2x spatial supersampling and 32-64 Cycles samples; 8-bit coverage",
        "fluid_geometry_sampling": "BVH first hit through every OpenCV pixel center inside projected mesh bounds",
        "linear_exr": {
            "encoding": "scene-linear Rec.709/sRGB primaries",
            "storage": "float16 ZIP OpenEXR",
            "display_transform_applied": False,
        },
        "preview": {
            "view_transform": "AgX",
            "look": "AgX - Medium High Contrast",
            "exposure": 0.0,
        },
        "water_material": {
            "geometry": "open free-surface sheet",
            "base_color_linear_rgba": [0.965, 0.987, 1.0, 1.0],
            "roughness": 0.018,
            "ior": 1.333,
            "transmission_weight": 1.0,
            "micro_normal": {
                "type": "procedural 3D noise bump",
                "scale_inverse_metres": 48.0,
                "strength": 0.075,
                "distance_metres": 0.0015,
                "geometry_labels_unperturbed": True,
            },
        },
        "rows": render_rows,
        "contact_sheet": {
            "path": str(contact_sheet.relative_to(episode)),
            "sha256": sha256_file(contact_sheet),
            "bytes": contact_sheet.stat().st_size,
        },
        "scene": {
            "path": str(scene_path.relative_to(episode)),
            "sha256": sha256_file(scene_path),
            "bytes": scene_path.stat().st_size,
        },
    }
    atomic_json(episode / "render_manifest.json", render_manifest)
    manifest = load_json(manifest_path)
    manifest["state"]["rendered"] = render_manifest["complete"]
    manifest["state"]["audited"] = False
    atomic_json(manifest_path, manifest)
    print(f"FLUID_7DGS_LOOKDEV_RENDER={episode} views={len(render_rows)}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
