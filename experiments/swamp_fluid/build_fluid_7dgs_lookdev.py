"""Build the 3-time x 6-camera visual gate for the fluid 7DGS episode."""

from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_fluid_4dgs_episode import (
    atomic_json,
    camera_to_world,
    load_json,
    require_exact_sources,
    run_checked,
    sha256_file,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"Y:\isaacsim_work\output\fluid_7dgs\episode_002_lookdev"),
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path(
            r"Y:\isaacsim_work\output\swamp_fluid_preview\whitewater_v2_source_impact_1p5s\whitewater_source"
        ),
    )
    parser.add_argument(
        "--run-report",
        type=Path,
        default=Path(
            r"Y:\isaacsim_work\output\swamp_fluid_preview\whitewater_v2_source_impact_1p5s\run_complete.json"
        ),
    )
    parser.add_argument(
        "--terrain-heightfield",
        type=Path,
        default=Path(
            r"Y:\isaacsim_work\output\swamp_fluid_preview\whitewater_v2_collision_heightfield_8mm.npz"
        ),
    )
    parser.add_argument("--source-samples", type=int, nargs=3, default=(72, 102, 132))
    parser.add_argument("--reference-sample", type=int, default=0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--cycles-samples", type=int, default=128)
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path(r"D:\Program Files (x86)\Blender\blender.exe"),
    )
    parser.add_argument("--scene", type=Path, default=Path(r"Y:\scenes\swamp.blend"))
    parser.add_argument(
        "--hdri",
        type=Path,
        default=Path(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr"),
    )
    parser.add_argument(
        "--pysplashsurf-python",
        type=Path,
        default=Path(r"Y:\tools\pysplashsurf\.venv\Scripts\python.exe"),
    )
    parser.add_argument(
        "--splashsurf",
        type=Path,
        default=Path(r"Y:\tools\pysplashsurf\.venv\Scripts\pysplashsurf.exe"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def read_obj_vertices(path):
    vertices = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(tuple(float(value) for value in line.split()[1:4]))
    result = np.asarray(vertices, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise ValueError(f"OBJ contains no valid vertices: {path}")
    return result


def projection_bbox(vertices, matrix, fx, fy, cx, cy, width, height):
    rotation = matrix[:3, :3]
    center = matrix[:3, 3]
    camera = (vertices - center) @ rotation
    camera = camera[camera[:, 2] > 0.01]
    u = fx * camera[:, 0] / camera[:, 2] + cx
    v = fy * camera[:, 1] / camera[:, 2] + cy
    return np.asarray(
        (
            np.clip(u.min(), 0.0, width - 1.0),
            np.clip(v.min(), 0.0, height - 1.0),
            np.clip(u.max(), 0.0, width - 1.0),
            np.clip(v.max(), 0.0, height - 1.0),
        ),
        dtype=np.float64,
    )


def build_cameras(mesh_vertices, width, height):
    all_vertices = np.concatenate(mesh_vertices, axis=0)
    minimum = all_vertices.min(axis=0)
    maximum = all_vertices.max(axis=0)
    center = 0.5 * (minimum + maximum)
    # Bias toward the persistent shallow sheet rather than the tallest single
    # splash vertex, while still leaving headroom for the late splash crown.
    center[1] = 0.68 * minimum[1] + 0.32 * maximum[1]
    lens_mm = 45.0
    sensor_width_mm = 36.0
    fx = lens_mm / sensor_width_mm * width
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    presets = (
        ("cam_00", "low", 0.0, 27.0, 3.05),
        ("cam_01", "low", 180.0, 30.0, 3.05),
        ("cam_02", "medium", 60.0, 45.0, 3.18),
        ("cam_03", "medium", 240.0, 48.0, 3.18),
        ("cam_04", "high", 120.0, 66.0, 3.34),
        ("cam_05", "high", 300.0, 70.0, 3.34),
    )
    cameras = []
    for name, tier, azimuth_degrees, elevation_degrees, distance in presets:
        azimuth = math.radians(azimuth_degrees)
        elevation = math.radians(elevation_degrees)
        location = center + distance * np.asarray(
            (
                math.cos(elevation) * math.cos(azimuth),
                math.sin(elevation),
                math.cos(elevation) * math.sin(azimuth),
            ),
            dtype=np.float64,
        )
        matrix = camera_to_world(location, center)
        per_frame_bbox = []
        for vertices in mesh_vertices:
            bbox = projection_bbox(vertices, matrix, fx, fy, cx, cy, width, height)
            per_frame_bbox.append(
                {
                    "pixels": bbox.astype(float).tolist(),
                    "width_fraction": float((bbox[2] - bbox[0] + 1.0) / width),
                    "height_fraction": float((bbox[3] - bbox[1] + 1.0) / height),
                }
            )
        cameras.append(
            {
                "camera_name": name,
                "split": "lookdev",
                "elevation_tier": tier,
                "azimuth_degrees": azimuth_degrees,
                "elevation_degrees": elevation_degrees,
                "width": int(width),
                "height": int(height),
                "intrinsics": {
                    "matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "pixel_center_convention": "integer_index_plus_0.5",
                    "lens_mm": lens_mm,
                    "sensor_width_mm": sensor_width_mm,
                    "sensor_fit": "HORIZONTAL",
                    "pixel_aspect": [1.0, 1.0],
                },
                "transform_convention": "camera_to_world",
                "camera_coordinates": "OpenCV right-down-forward (+X right, +Y down, +Z forward)",
                "camera_to_world": matrix.astype(float).tolist(),
                "world_to_camera": np.linalg.inv(matrix).astype(float).tolist(),
                "near": 0.01,
                "far": 100.0,
                "predicted_mesh_bboxes": per_frame_bbox,
            }
        )
    return {
        "schema": 1,
        "product": "fluid_7dgs_lookdev_fixed_cameras",
        "fixed_across_time": True,
        "world_coordinates": "right-handed Isaac/USD XYZ, metres, +Y up",
        "image_coordinates": "u right, v down, origin at top-left image edge",
        "target_world": center.astype(float).tolist(),
        "cameras": cameras,
    }


def main():
    args = parse_args()
    output = args.output.resolve()
    source_directory = args.source_directory.resolve()
    source_samples = tuple(map(int, args.source_samples))
    if source_samples != tuple(sorted(set(source_samples))) or len(source_samples) != 3:
        raise ValueError("source-samples must contain three unique increasing indices")
    if abs((source_samples[-1] - source_samples[0]) / 120.0 - 0.5) > 1.0e-12:
        raise ValueError("The lookdev endpoints must span exactly 0.5 seconds at 120 Hz")
    required = (
        args.run_report.resolve(),
        args.terrain_heightfield.resolve(),
        args.blender.resolve(),
        args.scene.resolve(),
        args.hdri.resolve(),
        args.pysplashsurf_python.resolve(),
        args.splashsurf.resolve(),
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty lookdev gate: {output}")
    for name in ("meshes", "states", "views", "scene", "diagnostics", "_build"):
        (output / name).mkdir(parents=True, exist_ok=True)

    source_manifest, source_audit, source_rows = require_exact_sources(
        source_directory, source_samples
    )
    run_report = load_json(args.run_report)
    if run_report.get("valid") is not True:
        raise ValueError("Parent PhysX run is not valid")

    particle_directory = output / "_build" / "particles"
    particle_manifest_path = particle_directory / "export_manifest.json"
    requested = sorted(set((args.reference_sample,) + source_samples))
    particle_ready = False
    if particle_manifest_path.is_file():
        payload = load_json(particle_manifest_path)
        particle_ready = (
            payload.get("complete") is True
            and payload.get("configuration", {}).get("source_samples") == requested
            and payload.get("configuration", {}).get("sample_stride") == 1
        )
    if not particle_ready:
        if particle_directory.exists() and any(particle_directory.iterdir()):
            raise RuntimeError("Existing particle staging cache does not match lookdev gate")
        run_checked(
            [
                args.pysplashsurf_python,
                SCRIPT_DIRECTORY / "export_whitewater_source_ply.py",
                source_directory,
                particle_directory,
                "--source-samples",
                *requested,
                "--sample-stride",
                1,
            ]
        )

    surface_build = output / "_build" / "splashsurf_120hz"
    surface_directory = surface_build / "surface"
    surface_manifest_path = surface_build / "splashsurf_manifest.json"
    surface_ready = surface_manifest_path.is_file() and all(
        (surface_directory / f"surface_{sample:04d}_clipped.obj").is_file()
        for sample in source_samples
    )
    if not surface_ready:
        run_checked(
            [
                args.pysplashsurf_python,
                SCRIPT_DIRECTORY / "build_splashsurf_sequence.py",
                particle_directory,
                args.terrain_heightfield.resolve(),
                surface_build,
                "--water-level",
                float(run_report["water_level"]["simulated"]),
                "--spacing",
                float(run_report["particle_spacing"]),
                "--workers",
                2,
                "--threads-per-worker",
                8,
                "--particle-radius",
                0.004,
                "--smoothing-length",
                2.0,
                "--cube-size",
                0.75,
                "--surface-threshold",
                0.6,
                "--mesh-smoothing-iters",
                10,
                "--normal-smoothing-iters",
                10,
                "--shoreline-mode",
                "reference-ply",
                "--shoreline-particles-ply",
                particle_directory / f"particles_{args.reference_sample:04d}.ply",
                "--minimum-layers",
                8,
                "--shoreline-erosion-cells",
                4,
                "--frames",
                *source_samples,
                "--splashsurf",
                args.splashsurf.resolve(),
            ]
        )

    frame_records = []
    timeline_frames = []
    mesh_vertices = []
    for frame_index, sample_index in enumerate(source_samples):
        source_row = source_rows[sample_index]
        source_path = source_directory / source_row["file"]
        if sha256_file(source_path) != source_row["sha256"]:
            raise ValueError(f"PhysX source hash mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as cache:
            simulation_time = float(cache["simulation_time"])
            source_transform = np.asarray(cache["sphere_transform"], dtype=np.float64)
            linear_velocity = np.asarray(cache["sphere_linear_velocity"], dtype=np.float64)
            angular_velocity = np.asarray(cache["sphere_angular_velocity"], dtype=np.float64)
        source_mesh = surface_directory / f"surface_{sample_index:04d}_clipped.obj"
        destination_mesh = output / "meshes" / f"frame_{frame_index:04d}.obj"
        if not destination_mesh.is_file() or sha256_file(destination_mesh) != sha256_file(source_mesh):
            shutil.copy2(source_mesh, destination_mesh)
        vertices = read_obj_vertices(destination_mesh)
        mesh_vertices.append(vertices)
        normalized_time = -1.0 + 2.0 * (
            (simulation_time - float(source_rows[source_samples[0]]["simulation_time"])) / 0.5
        )
        state = {
            "schema": 1,
            "product": "fluid_7dgs_lookdev_frame_state",
            "frame_index": frame_index,
            "source_sample_index": sample_index,
            "physics_step": int(source_row["physics_step"]),
            "simulation_time": simulation_time,
            "normalized_time_7dgs": normalized_time,
            "rigid_bodies": [
                {
                    "id": "drop_sphere",
                    "shape": "sphere",
                    "radius_metres": float(run_report["impactor"]["radius"]),
                    "mass_kilograms": float(run_report["impactor"]["mass"]),
                    "object_to_world": source_transform.T.astype(float).tolist(),
                    "linear_velocity_world_metres_per_second": linear_velocity.astype(float).tolist(),
                    "angular_velocity_world_radians_per_second": angular_velocity.astype(float).tolist(),
                }
            ],
            "control": {
                "mode": "unactuated_passive_dynamics",
                "commanded_force_world_newton": [0.0, 0.0, 0.0],
                "commanded_torque_world_newton_metre": [0.0, 0.0, 0.0],
                "target_motion": None,
            },
            "source": {"file": str(source_path), "sha256": source_row["sha256"]},
        }
        state_path = output / "states" / f"frame_{frame_index:04d}.json"
        atomic_json(state_path, state)
        timeline_frames.append(
            {
                "frame_index": frame_index,
                "source_sample_index": sample_index,
                "physics_step": int(source_row["physics_step"]),
                "simulation_time": simulation_time,
                "normalized_time_7dgs": normalized_time,
                "lookdev_role": ("early", "middle", "late")[frame_index],
            }
        )
        frame_records.append(
            {
                "frame_index": frame_index,
                "source_sample_index": sample_index,
                "simulation_time": simulation_time,
                "normalized_time_7dgs": normalized_time,
                "mesh": f"meshes/{destination_mesh.name}",
                "mesh_sha256": sha256_file(destination_mesh),
                "state": f"states/{state_path.name}",
                "state_sha256": sha256_file(state_path),
                "views": f"views/frame_{frame_index:04d}",
            }
        )

    cameras = build_cameras(mesh_vertices, args.width, args.height)
    atomic_json(output / "cameras.json", cameras)
    atomic_json(
        output / "timeline.json",
        {
            "schema": 1,
            "time_unit": "second",
            "gate_sampling": "three representative states spanning the future 61-frame interval",
            "future_training_source_samples_inclusive": [72, 132],
            "future_training_rate_hz": 120,
            "frames": timeline_frames,
        },
    )

    surface_configuration = load_json(surface_manifest_path)["configuration"]
    manifest = {
        "schema": 1,
        "product": "fluid_7dgs_visual_acceptance_gate",
        "episode_id": "episode_002_lookdev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": {"rendered": False, "audited": False, "visual_gate_passed_by_user": False},
        "purpose": "3 times x 6 fixed cameras; no 61-frame production render is authorized yet",
        "coordinate_system": {
            "world": "right-handed Isaac/USD XYZ",
            "up_axis": "+Y",
            "unit": "metre",
            "canonical_matrix_convention": "column vectors; translation in last column",
            "camera": "OpenCV +X right, +Y down, +Z forward",
        },
        "frames": frame_records,
        "files": {
            "timeline": "timeline.json",
            "cameras": "cameras.json",
            "render_manifest": "render_manifest.json",
            "audit": "audit_report.json",
            "contact_sheet": "diagnostics/contact_sheet.png",
            "scene": "scene/scene.blend",
        },
        "surface_reconstruction": {
            "method": "pysplashsurf 0.14.1 plus audited free-surface clipping",
            "configuration": surface_configuration,
            "shoreline_reference_source_sample": int(args.reference_sample),
        },
        "rendering": {
            "engine": "CYCLES",
            "resolution": [args.width, args.height],
            "cycles_samples": args.cycles_samples,
            "motion_blur": False,
            "linear_rgb": "scene-linear Rec.709/sRGB primaries in float16 OpenEXR",
            "preview": "AgX Medium High Contrast, exposure 0, 8-bit PNG",
            "hdri": str(args.hdri.resolve()),
            "hdri_sha256": sha256_file(args.hdri),
            "hdri_strength": 0.95,
            "background_plate_pairing": "same frame and camera; only liquid mesh is hidden",
            "visible_mask_semantics": "scene-visible liquid after terrain, vegetation and rigid-body occlusion",
            "fluid_only_semantics": "non-fluid objects excluded; exact liquid-mesh projection labels",
        },
        "provenance": {
            "physx_source_manifest": str(source_directory / "manifest.json"),
            "physx_source_manifest_sha256": sha256_file(source_directory / "manifest.json"),
            "physx_source_audit": str(source_directory / "audit_report.json"),
            "physx_source_audit_sha256": sha256_file(source_directory / "audit_report.json"),
            "run_report": str(args.run_report.resolve()),
            "run_report_sha256": sha256_file(args.run_report),
            "surface_manifest": str(surface_manifest_path),
            "surface_manifest_sha256": sha256_file(surface_manifest_path),
            "source_scene": str(args.scene.resolve()),
            "source_scene_sha256": sha256_file(args.scene),
        },
    }
    atomic_json(output / "manifest.json", manifest)
    if not args.skip_render:
        run_checked(
            [
                args.blender.resolve(),
                args.scene.resolve(),
                "--background",
                "--python",
                SCRIPT_DIRECTORY / "render_fluid_7dgs_lookdev_blender.py",
                "--",
                output,
            ]
        )
        run_checked(
            [
                args.blender.resolve(),
                (output / "scene" / "scene.blend").resolve(),
                "--background",
                "--python",
                SCRIPT_DIRECTORY / "audit_fluid_7dgs_lookdev_blender.py",
                "--",
                output,
            ]
        )
    print(f"FLUID_7DGS_LOOKDEV={output}")


if __name__ == "__main__":
    main()
