"""Independently audit the fluid 7DGS 3x6 visual acceptance gate."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_fluid_4dgs_episode_blender import (  # noqa: E402
    bvh_from_isaac_mesh,
    mask_bbox,
    projection_bbox,
    ray_depth_and_normal_audit,
    read_image,
    read_obj,
)
from render_fluid_4dgs_episode_blender import (  # noqa: E402
    atomic_json,
    load_json,
    sha256_file,
)


def dilate(mask, radius=2):
    result = mask.copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for row_offset in range(3):
            for column_offset in range(3):
                expanded |= padded[
                    row_offset : row_offset + result.shape[0],
                    column_offset : column_offset + result.shape[1],
                ]
        result = expanded
    return result


def bbox_containment_error(predicted, observed):
    return float(
        max(
            0.0,
            predicted[0] - observed[0],
            predicted[1] - observed[1],
            observed[2] - predicted[2],
            observed[3] - predicted[3],
        )
    )


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 1:
        raise SystemExit("Expected LOOKDEV_DIRECTORY")
    episode = Path(argv[0]).resolve()
    manifest_path = episode / "manifest.json"
    manifest = load_json(manifest_path)
    cameras_payload = load_json(episode / "cameras.json")
    timeline = load_json(episode / "timeline.json")
    render_manifest = load_json(episode / "render_manifest.json")
    frames = manifest["frames"]
    cameras = cameras_payload["cameras"]
    errors = []
    criteria = {}
    observations = []
    mesh_rows = []
    mesh_hashes = []
    maximum_depth_error = 0.0
    maximum_depth_p95 = 0.0
    minimum_normal_dot = 1.0
    maximum_fluid_containment_error = 0.0
    maximum_visible_containment_error = 0.0
    maximum_visible_outside_dilated_fraction = 0.0
    minimum_bbox_width_fraction = 1.0
    minimum_bbox_height_fraction = 1.0
    minimum_visible_pixels = 2**63 - 1
    all_formats_valid = True
    all_hashes_valid = True

    criteria["exactly_three_times_and_six_fixed_cameras"] = (
        len(frames) == 3
        and len(cameras) == 6
        and cameras_payload.get("fixed_across_time") is True
    )
    times = np.asarray([row["simulation_time"] for row in timeline["frames"]])
    normalized_times = np.asarray(
        [row["normalized_time_7dgs"] for row in timeline["frames"]]
    )
    criteria["representative_times_span_exactly_half_second"] = bool(
        np.allclose(times, (0.6, 0.85, 1.1), atol=1.0e-12, rtol=0.0)
        and np.allclose(normalized_times, (-1.0, 0.0, 1.0), atol=1.0e-12, rtol=0.0)
        and abs(times[-1] - times[0] - 0.5) <= 1.0e-12
    )
    tiers = [camera["elevation_tier"] for camera in cameras]
    criteria["camera_set_covers_low_medium_and_high_elevations"] = (
        tiers.count("low") == 2
        and tiers.count("medium") == 2
        and tiers.count("high") == 2
    )
    for camera in cameras:
        c2w = np.asarray(camera["camera_to_world"], dtype=np.float64)
        w2c = np.asarray(camera["world_to_camera"], dtype=np.float64)
        intrinsic = np.asarray(camera["intrinsics"]["matrix"], dtype=np.float64)
        camera_valid = (
            c2w.shape == (4, 4)
            and w2c.shape == (4, 4)
            and intrinsic.shape == (3, 3)
            and np.allclose(c2w @ w2c, np.eye(4), atol=1.0e-10)
            and np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1.0e-10)
            and abs(np.linalg.det(c2w[:3, :3]) - 1.0) <= 1.0e-10
            and int(camera["width"]) == 768
            and int(camera["height"]) == 768
        )
        if not camera_valid:
            errors.append(f"Invalid camera matrix: {camera['camera_name']}")
    criteria["camera_intrinsics_and_extrinsics_are_consistent"] = not any(
        error.startswith("Invalid camera matrix") for error in errors
    )

    render_rows = {
        (int(row["frame_index"]), row["camera_name"]): row
        for row in render_manifest["rows"]
    }
    expected_names = (
        "rgb_with_fluid_linear.exr",
        "rgb_background_linear.exr",
        "rgb_preview.png",
        "visible_fluid_mask.png",
        "fluid_rgba_linear.exr",
        "fluid_mask.png",
        "fluid_depth.exr",
        "fluid_normal.exr",
    )
    for frame in frames:
        frame_index = int(frame["frame_index"])
        mesh_path = episode / frame["mesh"]
        state_path = episode / frame["state"]
        if (
            not mesh_path.is_file()
            or sha256_file(mesh_path) != frame["mesh_sha256"]
            or not state_path.is_file()
            or sha256_file(state_path) != frame["state_sha256"]
        ):
            errors.append(f"Frame hash mismatch: {frame_index}")
            all_hashes_valid = False
            continue
        vertices, normals, faces, normal_references = read_obj(mesh_path)
        mesh_hashes.append(frame["mesh_sha256"])
        normal_lengths = np.linalg.norm(normals, axis=1)
        mesh_rows.append(
            {
                "frame_index": frame_index,
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "normal_records": int(len(normals)),
                "normal_face_references": int(normal_references),
                "normal_length_minimum": float(normal_lengths.min()),
                "normal_length_maximum": float(normal_lengths.max()),
                "sha256": frame["mesh_sha256"],
            }
        )
        bvh = bvh_from_isaac_mesh(vertices, faces)
        for camera in cameras:
            camera_name = camera["camera_name"]
            key = (frame_index, camera_name)
            row = render_rows.get(key)
            if row is None:
                errors.append(f"Missing render row: {key}")
                all_hashes_valid = False
                continue
            directory = episode / row["directory"]
            paths = {name: directory / name for name in expected_names}
            if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
                errors.append(f"Missing observation file: {key}")
                all_hashes_valid = False
                continue
            for name, path in paths.items():
                declared = row["files"][name]
                if (
                    declared["sha256"] != sha256_file(path)
                    or int(declared["bytes"]) != path.stat().st_size
                ):
                    errors.append(f"Observation hash mismatch: {key} {name}")
                    all_hashes_valid = False
            with_fluid, with_spec = read_image(paths["rgb_with_fluid_linear.exr"])
            background, background_spec = read_image(paths["rgb_background_linear.exr"])
            preview, preview_spec = read_image(paths["rgb_preview.png"])
            visible_image, visible_spec = read_image(paths["visible_fluid_mask.png"])
            fluid_rgba, fluid_rgba_spec = read_image(paths["fluid_rgba_linear.exr"])
            fluid_mask_image, fluid_mask_spec = read_image(paths["fluid_mask.png"])
            depth_image, depth_spec = read_image(paths["fluid_depth.exr"])
            normal_image, normal_spec = read_image(paths["fluid_normal.exr"])
            depth = depth_image[..., 0]
            normal = normal_image[..., :3]
            fluid_mask = fluid_mask_image[..., 0] > 0.5
            visible_coverage = visible_image[..., 0]
            visible_mask = visible_coverage > (1.0 / 255.0)
            specs = (
                with_spec,
                background_spec,
                preview_spec,
                visible_spec,
                fluid_rgba_spec,
                fluid_mask_spec,
                depth_spec,
                normal_spec,
            )
            dimensions_valid = all(
                spec["width"] == 768 and spec["height"] == 768 for spec in specs
            )
            formats_valid = (
                with_spec["format"] == "half"
                and background_spec["format"] == "half"
                and fluid_rgba_spec["format"] == "half"
                and depth_spec["format"] == "float"
                and normal_spec["format"] == "float"
                and preview_spec["format"] == "uint8"
                and visible_spec["format"] == "uint8"
                and fluid_mask_spec["format"] == "uint8"
                and len(with_spec["channels"]) == 3
                and len(background_spec["channels"]) == 3
                and len(fluid_rgba_spec["channels"]) == 4
                and len(depth_spec["channels"]) == 1
                and len(normal_spec["channels"]) == 3
            )
            depth_valid = (
                np.isfinite(depth).all()
                and np.all(depth[~fluid_mask] == 0.0)
                and np.all(depth[fluid_mask] > float(camera["near"]))
                and np.all(depth[fluid_mask] < float(camera["far"]))
            )
            normal_lengths_image = np.linalg.norm(normal, axis=2)
            normal_valid = (
                np.isfinite(normal).all()
                and np.all(normal[~fluid_mask] == 0.0)
                and np.all(np.abs(normal_lengths_image[fluid_mask] - 1.0) <= 2.0e-5)
            )
            rgb_valid = (
                np.isfinite(with_fluid).all()
                and np.isfinite(background).all()
                and float(np.std(with_fluid[..., :3])) > 0.01
                and float(np.std(background[..., :3])) > 0.01
                and sha256_file(paths["rgb_with_fluid_linear.exr"])
                != sha256_file(paths["rgb_background_linear.exr"])
            )
            all_formats_valid &= bool(
                dimensions_valid
                and formats_valid
                and depth_valid
                and normal_valid
                and rgb_valid
                and np.any(visible_mask)
                and np.any(fluid_mask)
            )
            predicted_bbox = projection_bbox(vertices, camera)
            fluid_bbox = mask_bbox(fluid_mask)
            visible_bbox = mask_bbox(visible_mask)
            fluid_containment = bbox_containment_error(predicted_bbox, fluid_bbox)
            visible_containment = bbox_containment_error(predicted_bbox, visible_bbox)
            maximum_fluid_containment_error = max(
                maximum_fluid_containment_error, fluid_containment
            )
            maximum_visible_containment_error = max(
                maximum_visible_containment_error, visible_containment
            )
            dilated_fluid = dilate(fluid_mask, radius=2)
            visible_outside_fraction = float(
                np.count_nonzero(visible_mask & ~dilated_fluid)
                / max(1, np.count_nonzero(visible_mask))
            )
            maximum_visible_outside_dilated_fraction = max(
                maximum_visible_outside_dilated_fraction, visible_outside_fraction
            )
            bbox_width_fraction = float((fluid_bbox[2] - fluid_bbox[0] + 1.0) / 768.0)
            bbox_height_fraction = float((fluid_bbox[3] - fluid_bbox[1] + 1.0) / 768.0)
            minimum_bbox_width_fraction = min(
                minimum_bbox_width_fraction, bbox_width_fraction
            )
            minimum_bbox_height_fraction = min(
                minimum_bbox_height_fraction, bbox_height_fraction
            )
            minimum_visible_pixels = min(
                minimum_visible_pixels, int(np.count_nonzero(visible_mask))
            )
            ray_metrics = ray_depth_and_normal_audit(
                bvh, camera, depth, normal, fluid_mask, sample_count=64
            )
            maximum_depth_error = max(
                maximum_depth_error, ray_metrics["depth_error_maximum_metres"]
            )
            maximum_depth_p95 = max(
                maximum_depth_p95, ray_metrics["depth_error_p95_metres"]
            )
            minimum_normal_dot = min(
                minimum_normal_dot, ray_metrics["absolute_normal_dot_median"]
            )
            effect = np.abs(with_fluid[..., :3] - background[..., :3]).mean(axis=2)
            effect_inside_visible = float(effect[visible_mask].mean())
            observations.append(
                {
                    "frame_index": frame_index,
                    "camera_name": camera_name,
                    "elevation_tier": camera["elevation_tier"],
                    "formats_and_dimensions_valid": bool(
                        dimensions_valid and formats_valid
                    ),
                    "rgb_pair_valid": bool(rgb_valid),
                    "fluid_bbox_width_fraction": bbox_width_fraction,
                    "fluid_bbox_height_fraction": bbox_height_fraction,
                    "visible_pixels_nonzero_coverage": int(
                        np.count_nonzero(visible_mask)
                    ),
                    "visible_outside_two_pixel_dilated_fluid_fraction": visible_outside_fraction,
                    "fluid_bbox_containment_error_pixels": fluid_containment,
                    "visible_bbox_containment_error_pixels": visible_containment,
                    "mean_linear_rgb_effect_inside_visible_mask": effect_inside_visible,
                    **ray_metrics,
                }
            )

    criteria["all_files_exist_and_declared_hashes_match"] = bool(all_hashes_valid)
    criteria["three_distinct_meshes_keep_world_vertex_normals"] = (
        len(mesh_hashes) == 3
        and len(set(mesh_hashes)) == 3
        and all(
            row["normal_records"] == row["vertices"]
            and row["normal_face_references"] == 3 * row["faces"]
            and 0.95 <= row["normal_length_minimum"] <= row["normal_length_maximum"] <= 1.05
            for row in mesh_rows
        )
    )
    criteria["eighteen_observations_have_required_linear_and_label_formats"] = (
        len(observations) == 18 and all_formats_valid
    )
    criteria["fluid_mesh_occupies_required_image_fraction"] = (
        minimum_bbox_width_fraction >= 0.55
        and minimum_bbox_height_fraction >= 0.45
    )
    criteria["fluid_only_projection_depth_and_normal_are_consistent"] = (
        maximum_fluid_containment_error <= 1.0
        and maximum_depth_p95 <= 5.0e-5
        and maximum_depth_error <= 1.0e-4
        and minimum_normal_dot >= 0.999
    )
    criteria["visible_masks_respect_scene_occlusion_and_mesh_projection"] = (
        minimum_visible_pixels > 0
        and maximum_visible_containment_error <= 2.0
        and maximum_visible_outside_dilated_fraction <= 0.002
    )
    contact_sheet = episode / manifest["files"]["contact_sheet"]
    scene_path = episode / manifest["files"]["scene"]
    criteria["contact_sheet_and_reproducible_scene_are_present"] = (
        contact_sheet.is_file()
        and contact_sheet.stat().st_size > 0
        and scene_path.is_file()
        and render_manifest["scene"]["sha256"] == sha256_file(scene_path)
    )
    temporary_paths = [
        path
        for path in episode.rglob("*")
        if path.name.startswith(".visible_") or path.name == "_raw"
    ]
    criteria["no_temporary_render_outputs_remain"] = not temporary_paths
    valid = all(criteria.values()) and not errors
    manifest["state"]["rendered"] = render_manifest.get("complete") is True
    manifest["state"]["audited"] = True
    manifest["state"]["technical_audit_valid"] = bool(valid)
    atomic_json(manifest_path, manifest)
    report = {
        "schema": 1,
        "product": "fluid_7dgs_lookdev_technical_audit",
        "valid": bool(valid),
        "visual_acceptance_by_user": False,
        "manifest_sha256": sha256_file(manifest_path),
        "criteria": criteria,
        "errors": errors,
        "metrics": {
            "frame_count": len(frames),
            "camera_count": len(cameras),
            "observation_count": len(observations),
            "minimum_fluid_bbox_width_fraction": minimum_bbox_width_fraction,
            "minimum_fluid_bbox_height_fraction": minimum_bbox_height_fraction,
            "minimum_visible_pixels_nonzero_coverage": minimum_visible_pixels,
            "maximum_fluid_bbox_containment_error_pixels": maximum_fluid_containment_error,
            "maximum_visible_bbox_containment_error_pixels": maximum_visible_containment_error,
            "maximum_visible_outside_two_pixel_dilated_fluid_fraction": maximum_visible_outside_dilated_fraction,
            "maximum_depth_error_metres": maximum_depth_error,
            "maximum_depth_p95_error_metres": maximum_depth_p95,
            "minimum_absolute_normal_dot_median": minimum_normal_dot,
            "temporary_path_count": len(temporary_paths),
        },
        "meshes": mesh_rows,
        "observations": observations,
    }
    atomic_json(episode / "audit_report.json", report)
    print(
        __import__("json").dumps(
            {"valid": valid, **report["metrics"]}, indent=2, sort_keys=True
        )
    )
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
