"""Independently audit a synchronized fluid 4DGS episode."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio
from mathutils import Vector
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
        return array, {
            "width": int(specification.width),
            "height": int(specification.height),
            "channels": list(specification.channelnames),
            "format": str(specification.format),
        }
    finally:
        source.close()


def read_obj(path):
    vertices = []
    normals = []
    faces = []
    normal_face_references = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(tuple(float(value) for value in line.split()[1:4]))
            elif line.startswith("vn "):
                normals.append(tuple(float(value) for value in line.split()[1:4]))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) != 3:
                    raise ValueError(f"Non-triangle OBJ face in {path}")
                face = []
                for token in tokens:
                    parts = token.split("/")
                    face.append(int(parts[0]) - 1)
                    if len(parts) >= 3 and parts[2]:
                        normal_face_references += 1
                faces.append(tuple(face))
    vertices = np.asarray(vertices, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or normals.shape != vertices.shape
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
        or not np.isfinite(vertices).all()
        or not np.isfinite(normals).all()
    ):
        raise ValueError(f"Invalid OBJ geometry or normals: {path}")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError(f"OBJ face index is out of range: {path}")
    return vertices, normals, faces, normal_face_references


def blender_vector_from_isaac(value):
    value = np.asarray(value, dtype=np.float64)
    return Vector((float(value[0]), float(-value[2]), float(value[1])))


def isaac_vector_from_blender(value):
    return np.asarray((float(value[0]), float(value[2]), float(-value[1])))


def eroded_mask(mask):
    result = mask.copy()
    result[1:-1, 1:-1] &= (
        mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    result[0] = False
    result[-1] = False
    result[:, 0] = False
    result[:, -1] = False
    return result


def projection_bbox(vertices, camera):
    matrix = np.asarray(camera["camera_to_world"], dtype=np.float64)
    rotation = matrix[:3, :3]
    center = matrix[:3, 3]
    camera_points = (vertices - center) @ rotation
    valid = camera_points[:, 2] > float(camera["near"])
    camera_points = camera_points[valid]
    intrinsics = camera["intrinsics"]
    u = float(intrinsics["fx"]) * camera_points[:, 0] / camera_points[:, 2] + float(
        intrinsics["cx"]
    )
    v = float(intrinsics["fy"]) * camera_points[:, 1] / camera_points[:, 2] + float(
        intrinsics["cy"]
    )
    width = int(camera["width"])
    height = int(camera["height"])
    return np.asarray(
        (
            np.clip(u.min(), 0.0, width - 1.0),
            np.clip(v.min(), 0.0, height - 1.0),
            np.clip(u.max(), 0.0, width - 1.0),
            np.clip(v.max(), 0.0, height - 1.0),
        ),
        dtype=np.float64,
    )


def mask_bbox(mask):
    rows, columns = np.nonzero(mask)
    return np.asarray((columns.min(), rows.min(), columns.max(), rows.max()), dtype=np.float64)


def bvh_from_isaac_mesh(vertices, faces):
    blender_vertices = np.column_stack(
        (vertices[:, 0], -vertices[:, 2], vertices[:, 1])
    )
    return BVHTree.FromPolygons(
        blender_vertices.tolist(), faces.tolist(), all_triangles=True, epsilon=0.0
    )


def ray_depth_and_normal_audit(bvh, camera, depth, normal, mask, sample_count=32):
    interior = eroded_mask(mask)
    coordinates = np.argwhere(interior)
    if len(coordinates) < sample_count:
        raise ValueError("Fluid mask has too few interior pixels for ray audit")
    choice = np.linspace(0, len(coordinates) - 1, sample_count, dtype=np.int64)
    coordinates = coordinates[choice]
    matrix = np.asarray(camera["camera_to_world"], dtype=np.float64)
    rotation = matrix[:3, :3]
    center = matrix[:3, 3]
    intrinsics = camera["intrinsics"]
    depth_errors = []
    normal_dots = []
    missed = 0
    for row, column in coordinates:
        camera_direction = np.asarray(
            (
                (column + 0.5 - float(intrinsics["cx"])) / float(intrinsics["fx"]),
                (row + 0.5 - float(intrinsics["cy"])) / float(intrinsics["fy"]),
                1.0,
            ),
            dtype=np.float64,
        )
        camera_direction /= np.linalg.norm(camera_direction)
        world_direction = rotation @ camera_direction
        hit, hit_normal, _, _ = bvh.ray_cast(
            blender_vector_from_isaac(center),
            blender_vector_from_isaac(world_direction),
            float(camera["far"]),
        )
        if hit is None:
            missed += 1
            continue
        hit_isaac = isaac_vector_from_blender(hit)
        expected_z = float(np.dot(hit_isaac - center, rotation[:, 2]))
        depth_errors.append(abs(expected_z - float(depth[row, column])))
        geometric_normal = isaac_vector_from_blender(hit_normal)
        geometric_normal /= np.linalg.norm(geometric_normal)
        output_normal = np.asarray(normal[row, column], dtype=np.float64)
        normal_dots.append(abs(float(np.dot(geometric_normal, output_normal))))
    if not depth_errors:
        raise ValueError("Every independent mesh ray missed")
    return {
        "sample_count": int(sample_count),
        "ray_hit_count": int(len(depth_errors)),
        "ray_miss_count": int(missed),
        "depth_error_maximum_metres": float(np.max(depth_errors)),
        "depth_error_p95_metres": float(np.quantile(depth_errors, 0.95)),
        "depth_error_median_metres": float(np.median(depth_errors)),
        "absolute_normal_dot_minimum": float(np.min(normal_dots)),
        "absolute_normal_dot_median": float(np.median(normal_dots)),
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 1:
        raise SystemExit("Expected EPISODE_DIRECTORY")
    episode = Path(argv[0]).resolve()
    manifest_path = episode / "manifest.json"
    timeline_path = episode / "timeline.json"
    cameras_path = episode / "cameras.json"
    controls_path = episode / "controls.json"
    render_manifest_path = episode / "render_manifest.json"
    manifest = load_json(manifest_path)
    timeline = load_json(timeline_path)
    cameras_payload = load_json(cameras_path)
    controls = load_json(controls_path)
    render_manifest = load_json(render_manifest_path)
    errors = []
    criteria = {}

    frames = manifest.get("frames", [])
    cameras = cameras_payload.get("cameras", [])
    criteria["exactly_four_frames_and_eight_fixed_cameras"] = (
        len(frames) == 4
        and len(cameras) == 8
        and cameras_payload.get("fixed_across_time") is True
    )
    times = np.asarray(
        [float(item["simulation_time"]) for item in timeline.get("frames", [])],
        dtype=np.float64,
    )
    expected_times = np.asarray((0.8, 97 / 120, 98 / 120, 99 / 120), dtype=np.float64)
    criteria["timeline_is_exact_consecutive_120hz"] = (
        len(times) == 4
        and np.allclose(times, expected_times, rtol=0.0, atol=1.0e-12)
        and np.allclose(np.diff(times), 1.0 / 120.0, rtol=0.0, atol=1.0e-12)
        and [int(item["source_sample_index"]) for item in timeline["frames"]]
        == [96, 97, 98, 99]
    )
    criteria["controls_cover_every_frame_without_invented_internal_forces"] = (
        [int(item["frame_index"]) for item in controls.get("frames", [])]
        == [0, 1, 2, 3]
        and all(
            item.get("simulator_internal_contact_and_hydrodynamic_force") is None
            for item in controls.get("frames", [])
        )
    )

    camera_valid = True
    for camera in cameras:
        c2w = np.asarray(camera["camera_to_world"], dtype=np.float64)
        w2c = np.asarray(camera["world_to_camera"], dtype=np.float64)
        camera_valid &= (
            c2w.shape == (4, 4)
            and np.isfinite(c2w).all()
            and np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1.0e-9)
            and np.isclose(np.linalg.det(c2w[:3, :3]), 1.0, atol=1.0e-9)
            and np.allclose(w2c @ c2w, np.eye(4), atol=1.0e-9)
            and int(camera["width"]) == 512
            and int(camera["height"]) == 512
        )
    criteria["camera_matrices_and_intrinsics_are_valid"] = bool(camera_valid)

    mesh_hashes = []
    mesh_rows = []
    observation_rows = []
    maximum_bbox_error = 0.0
    maximum_bbox_containment_error = 0.0
    maximum_depth_error = 0.0
    maximum_depth_p95 = 0.0
    minimum_normal_dot_median = 1.0
    all_files_valid = True
    all_observation_contracts_valid = True

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
            errors.append(f"Frame provenance mismatch: {frame_index}")
            all_files_valid = False
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
                "bounds_minimum": vertices.min(axis=0).astype(float).tolist(),
                "bounds_maximum": vertices.max(axis=0).astype(float).tolist(),
            }
        )
        bvh = bvh_from_isaac_mesh(vertices, faces)
        for camera in cameras:
            view_directory = (
                episode
                / "views"
                / f"frame_{frame_index:04d}"
                / camera["camera_name"]
            )
            paths = {
                name: view_directory / name
                for name in (
                    "rgb.png",
                    "fluid_rgba.png",
                    "depth.exr",
                    "mask.png",
                    "normal.exr",
                )
            }
            if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
                errors.append(
                    f"Missing observation: frame={frame_index} camera={camera['camera_name']}"
                )
                all_files_valid = False
                continue
            rgb, rgb_spec = read_image(paths["rgb.png"])
            rgba, rgba_spec = read_image(paths["fluid_rgba.png"])
            depth, depth_spec = read_image(paths["depth.exr"])
            mask_image, mask_spec = read_image(paths["mask.png"])
            normal, normal_spec = read_image(paths["normal.exr"])
            depth = depth[..., 0]
            mask = mask_image[..., 0] > 0.5
            normal = normal[..., :3]
            alpha_mask = rgba[..., 3] > (1.0 / 255.0)
            sizes_valid = all(
                spec["width"] == 512 and spec["height"] == 512
                for spec in (rgb_spec, rgba_spec, depth_spec, mask_spec, normal_spec)
            )
            depth_contract = (
                np.isfinite(depth).all()
                and np.all(depth[~mask] == 0.0)
                and np.all(depth[mask] > float(camera["near"]))
                and np.all(depth[mask] < float(camera["far"]))
            )
            normal_lengths = np.linalg.norm(normal, axis=2)
            normal_contract = (
                np.isfinite(normal).all()
                and np.all(normal[~mask] == 0.0)
                and np.all(np.abs(normal_lengths[mask] - 1.0) <= 2.0e-5)
            )
            alpha_only_fraction = float(
                np.count_nonzero(alpha_mask & ~mask) / max(1, np.count_nonzero(alpha_mask))
            )
            mask_contract = (
                np.any(mask)
                and not np.any(mask & ~alpha_mask)
            )
            beauty_contract = np.isfinite(rgb).all() and float(np.std(rgb[..., :3])) > 0.01
            all_observation_contracts_valid &= (
                sizes_valid
                and depth_contract
                and normal_contract
                and mask_contract
                and beauty_contract
            )
            predicted_bbox = projection_bbox(vertices, camera)
            observed_bbox = mask_bbox(mask)
            bbox_error = float(np.max(np.abs(predicted_bbox - observed_bbox)))
            maximum_bbox_error = max(maximum_bbox_error, bbox_error)
            # Raw projected vertex bounds include occluded, back-facing and
            # subpixel triangles, so they may legitimately be wider than a
            # visible center-sampled mask.  A visible mask extending outside
            # the complete mesh projection, however, is a coordinate error.
            bbox_containment_error = float(
                max(
                    0.0,
                    predicted_bbox[0] - observed_bbox[0],
                    predicted_bbox[1] - observed_bbox[1],
                    observed_bbox[2] - predicted_bbox[2],
                    observed_bbox[3] - predicted_bbox[3],
                )
            )
            maximum_bbox_containment_error = max(
                maximum_bbox_containment_error, bbox_containment_error
            )
            ray_metrics = ray_depth_and_normal_audit(
                bvh, camera, depth, normal, mask, sample_count=32
            )
            maximum_depth_error = max(
                maximum_depth_error, ray_metrics["depth_error_maximum_metres"]
            )
            maximum_depth_p95 = max(
                maximum_depth_p95, ray_metrics["depth_error_p95_metres"]
            )
            minimum_normal_dot_median = min(
                minimum_normal_dot_median,
                ray_metrics["absolute_normal_dot_median"],
            )
            observation_rows.append(
                {
                    "frame_index": frame_index,
                    "camera_name": camera["camera_name"],
                    "fluid_pixels": int(np.count_nonzero(mask)),
                    "image_dimensions_valid": bool(sizes_valid),
                    "depth_contract_valid": bool(depth_contract),
                    "normal_contract_valid": bool(normal_contract),
                    "mask_contract_valid": bool(mask_contract),
                    "beauty_contract_valid": bool(beauty_contract),
                    "alpha_only_subpixel_coverage_fraction": alpha_only_fraction,
                    "predicted_mesh_bbox_pixels": predicted_bbox.astype(float).tolist(),
                    "observed_mask_bbox_pixels": observed_bbox.astype(float).tolist(),
                    "bbox_error_maximum_pixels": bbox_error,
                    "bbox_containment_error_maximum_pixels": bbox_containment_error,
                    **ray_metrics,
                }
            )

    criteria["all_declared_files_and_hashes_are_valid"] = all_files_valid
    criteria["meshes_are_distinct_and_keep_world_vertex_normals"] = (
        len(mesh_hashes) == 4
        and len(set(mesh_hashes)) == 4
        and all(
            row["normal_records"] == row["vertices"]
            and row["normal_face_references"] == 3 * row["faces"]
            and 0.95 <= row["normal_length_minimum"] <= row["normal_length_maximum"] <= 1.05
            for row in mesh_rows
        )
    )
    criteria["observation_formats_masks_depth_and_normals_are_valid"] = bool(
        all_observation_contracts_valid and len(observation_rows) == 32
    )
    criteria["mesh_projection_matches_fluid_masks"] = (
        maximum_bbox_containment_error <= 2.0
    )
    criteria["independent_mesh_rays_match_optical_z_depth"] = (
        maximum_depth_p95 <= 0.004 and maximum_depth_error <= 0.012
    )
    criteria["world_normals_agree_with_mesh_geometry"] = minimum_normal_dot_median >= 0.60
    scene_path = episode / manifest["files"]["scene"]
    criteria["scene_entry_is_present_and_hashed"] = (
        scene_path.is_file()
        and render_manifest.get("scene", {}).get("sha256") == sha256_file(scene_path)
    )
    valid = all(criteria.values()) and not errors

    manifest["state"] = {
        "complete": bool(valid),
        "rendered": render_manifest.get("complete") is True,
        "audited": True,
    }
    atomic_json(manifest_path, manifest)
    report = {
        "schema": 1,
        "product": "fluid_4dgs_episode_audit",
        "valid": bool(valid),
        "manifest_sha256": sha256_file(manifest_path),
        "criteria": criteria,
        "errors": errors,
        "metrics": {
            "frame_count": len(frames),
            "camera_count": len(cameras),
            "observation_count": len(observation_rows),
            "maximum_projection_bbox_error_pixels": maximum_bbox_error,
            "maximum_projection_bbox_containment_error_pixels": maximum_bbox_containment_error,
            "maximum_depth_error_metres": maximum_depth_error,
            "maximum_depth_p95_error_metres": maximum_depth_p95,
            "minimum_absolute_normal_dot_median": minimum_normal_dot_median,
        },
        "meshes": mesh_rows,
        "observations": observation_rows,
    }
    atomic_json(episode / "audit_report.json", report)
    print(json.dumps({"valid": valid, **report["metrics"]}, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
