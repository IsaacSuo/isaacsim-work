"""Build a synchronized PhysX + Splashsurf 4DGS interface episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"Y:\isaacsim_work\output\fluid_4dgs\episode_001"),
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
    parser.add_argument("--source-samples", type=int, nargs=4, default=(96, 97, 98, 99))
    parser.add_argument("--reference-sample", type=int, default=0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path(r"D:\Program Files (x86)\Blender\blender.exe"),
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path(r"Y:\scenes\swamp.blend"),
    )
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_checked(command):
    print("[fluid-4dgs] " + " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def camera_to_world(location, target):
    location = np.asarray(location, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - location
    forward /= np.linalg.norm(forward)
    world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = down
    matrix[:3, 2] = forward
    matrix[:3, 3] = location
    return matrix


def build_cameras(count, width, height):
    center = np.asarray((-0.73, -1.405, 0.78), dtype=np.float64)
    radius = 4.2
    # The swamp asset contains tall foreground grass around the basin.  Keep
    # every training view above that canopy so no fixed view becomes an almost
    # fully occluded observation while preserving an oblique surface angle.
    elevation = 3.2
    lens_mm = 45.0
    sensor_width_mm = 36.0
    fx = lens_mm / sensor_width_mm * width
    fy = fx
    cameras = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        location = center + np.asarray(
            (radius * math.cos(angle), elevation, radius * math.sin(angle))
        )
        matrix = camera_to_world(location, center)
        cameras.append(
            {
                "camera_name": f"cam_{index:02d}",
                "width": int(width),
                "height": int(height),
                "intrinsics": {
                    "matrix": [
                        [fx, 0.0, width / 2.0],
                        [0.0, fy, height / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "fx": fx,
                    "fy": fy,
                    "cx": width / 2.0,
                    "cy": height / 2.0,
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
            }
        )
    return {
        "schema": 1,
        "product": "fluid_4dgs_fixed_cameras",
        "fixed_across_time": True,
        "world_coordinates": "right-handed Isaac/USD XYZ, metres, +Y up",
        "image_coordinates": "u right, v down, origin at top-left image edge",
        "cameras": cameras,
    }


def require_exact_sources(source_directory, source_samples):
    manifest_path = source_directory / "manifest.json"
    audit_path = source_directory / "audit_report.json"
    manifest = load_json(manifest_path)
    audit = load_json(audit_path)
    if audit.get("valid") is not True:
        raise ValueError("PhysX source cache has not passed audit")
    if audit.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("PhysX source audit does not own the current manifest")
    rows = {int(row["sample_index"]): row for row in manifest["samples"]}
    if any(index not in rows for index in source_samples):
        raise ValueError("PhysX source cache is missing a requested sample")
    ordered_times = [float(rows[index]["simulation_time"]) for index in source_samples]
    if not np.all(np.diff(ordered_times) > 0.0):
        raise ValueError("Requested PhysX samples are not strictly increasing")
    return manifest, audit, rows


def main():
    args = parse_args()
    output = args.output.resolve()
    source_directory = args.source_directory.resolve()
    run_report_path = args.run_report.resolve()
    terrain_path = args.terrain_heightfield.resolve()
    source_samples = tuple(int(value) for value in args.source_samples)
    if args.views != 8:
        raise ValueError("The phase-1 interface contract requires exactly 8 views")
    if len(set(source_samples)) != 4 or tuple(sorted(source_samples)) != source_samples:
        raise ValueError("source-samples must contain four unique increasing indices")
    required_paths = (
        source_directory / "manifest.json",
        source_directory / "audit_report.json",
        run_report_path,
        terrain_path,
        args.blender.resolve(),
        args.scene.resolve(),
        args.hdri.resolve(),
        args.pysplashsurf_python.resolve(),
        args.splashsurf.resolve(),
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty episode: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("meshes", "states", "views", "scene", "_build"):
        (output / name).mkdir(exist_ok=True)

    source_manifest, source_audit, source_rows = require_exact_sources(
        source_directory, source_samples
    )
    run_report = load_json(run_report_path)
    if run_report.get("valid") is not True:
        raise ValueError("Parent PhysX run is not valid")

    particle_directory = output / "_build" / "particles"
    particle_manifest_path = particle_directory / "export_manifest.json"
    requested_with_reference = sorted(set((args.reference_sample,) + source_samples))
    particle_ready = False
    if particle_manifest_path.is_file():
        particle_manifest = load_json(particle_manifest_path)
        particle_ready = (
            particle_manifest.get("complete") is True
            and particle_manifest.get("configuration", {}).get("source_samples")
            == requested_with_reference
            and particle_manifest.get("configuration", {}).get("sample_stride") == 1
        )
    if not particle_ready:
        if particle_directory.exists() and any(particle_directory.iterdir()):
            raise RuntimeError("Existing particle staging cache does not match this episode")
        run_checked(
            [
                args.pysplashsurf_python,
                SCRIPT_DIRECTORY / "export_whitewater_source_ply.py",
                source_directory,
                particle_directory,
                "--source-samples",
                *requested_with_reference,
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
        water_level = float(run_report["water_level"]["simulated"])
        spacing = float(run_report["particle_spacing"])
        run_checked(
            [
                args.pysplashsurf_python,
                SCRIPT_DIRECTORY / "build_splashsurf_sequence.py",
                particle_directory,
                terrain_path,
                surface_build,
                "--water-level",
                water_level,
                "--spacing",
                spacing,
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
                args.splashsurf,
            ]
        )

    cameras = build_cameras(args.views, args.width, args.height)
    atomic_json(output / "cameras.json", cameras)

    timeline_frames = []
    controls_frames = []
    frame_records = []
    for frame_index, source_sample in enumerate(source_samples):
        source_row = source_rows[source_sample]
        source_path = source_directory / source_row["file"]
        if sha256_file(source_path) != source_row["sha256"]:
            raise ValueError(f"PhysX sample provenance mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as cache:
            simulation_time = float(cache["simulation_time"])
            source_transform = np.asarray(cache["sphere_transform"], dtype=np.float64)
            linear_velocity = np.asarray(cache["sphere_linear_velocity"], dtype=np.float64)
            angular_velocity = np.asarray(cache["sphere_angular_velocity"], dtype=np.float64)
        object_to_world = source_transform.T
        dt_to_next = (
            float(source_rows[source_samples[frame_index + 1]]["simulation_time"])
            - simulation_time
            if frame_index + 1 < len(source_samples)
            else None
        )
        source_mesh = surface_directory / f"surface_{source_sample:04d}_clipped.obj"
        destination_mesh = output / "meshes" / f"frame_{frame_index:04d}.obj"
        if not destination_mesh.is_file() or sha256_file(destination_mesh) != sha256_file(
            source_mesh
        ):
            shutil.copy2(source_mesh, destination_mesh)
        if not any(line.startswith("vn ") for line in destination_mesh.open("r", encoding="utf-8")):
            raise ValueError(f"Packaged mesh is missing vertex normals: {destination_mesh}")
        control = {
            "frame_index": frame_index,
            "simulation_time": simulation_time,
            "control_mode": "unactuated_passive_dynamics",
            "commanded_force_world_newton": [0.0, 0.0, 0.0],
            "commanded_torque_world_newton_metre": [0.0, 0.0, 0.0],
            "target_motion": None,
            "simulator_internal_contact_and_hydrodynamic_force": None,
            "internal_force_availability": "not recorded at the 120 Hz source samples",
        }
        state = {
            "schema": 1,
            "product": "fluid_4dgs_frame_state",
            "frame_index": frame_index,
            "source_sample_index": source_sample,
            "physics_step": int(source_row["physics_step"]),
            "simulation_time": simulation_time,
            "dt_to_next": dt_to_next,
            "mesh": {
                "path": f"../meshes/{destination_mesh.name}",
                "sha256": sha256_file(destination_mesh),
                "coordinates": "Isaac/USD world XYZ in metres, +Y up",
                "vertex_normals": "OBJ vn records in the same world coordinates",
            },
            "rigid_bodies": [
                {
                    "id": "drop_sphere",
                    "shape": "sphere",
                    "radius_metres": float(run_report["impactor"]["radius"]),
                    "mass_kilograms": float(run_report["impactor"]["mass"]),
                    "transform_convention": "object_to_world, column vectors, translation in last column",
                    "object_to_world": object_to_world.astype(float).tolist(),
                    "source_usd_row_matrix": source_transform.astype(float).tolist(),
                    "linear_velocity_world_metres_per_second": linear_velocity.astype(float).tolist(),
                    "angular_velocity_world_radians_per_second": angular_velocity.astype(float).tolist(),
                }
            ],
            "control": control,
            "source": {
                "file": str(source_path),
                "sha256": source_row["sha256"],
            },
        }
        state_path = output / "states" / f"frame_{frame_index:04d}.json"
        atomic_json(state_path, state)
        timeline_frames.append(
            {
                "frame_index": frame_index,
                "source_sample_index": source_sample,
                "physics_step": int(source_row["physics_step"]),
                "simulation_time": simulation_time,
                "dt_to_next": dt_to_next,
            }
        )
        controls_frames.append(control)
        frame_records.append(
            {
                "frame_index": frame_index,
                "simulation_time": simulation_time,
                "source_sample_index": source_sample,
                "mesh": f"meshes/{destination_mesh.name}",
                "mesh_sha256": sha256_file(destination_mesh),
                "state": f"states/{state_path.name}",
                "state_sha256": sha256_file(state_path),
                "views": f"views/frame_{frame_index:04d}",
            }
        )

    atomic_json(
        output / "timeline.json",
        {
            "schema": 1,
            "time_unit": "second",
            "sampling": "four consecutive 120 Hz whitewater source samples",
            "frames": timeline_frames,
        },
    )
    atomic_json(
        output / "controls.json",
        {
            "schema": 1,
            "product": "fluid_4dgs_controls",
            "semantics": (
                "No actuator or target motion was applied. Zero values are commanded "
                "controls, not estimates of gravity, contacts, or hydrodynamic force."
            ),
            "frames": controls_frames,
        },
    )

    surface_configuration = load_json(surface_manifest_path)["configuration"]
    manifest = {
        "schema": 1,
        "product": "fluid_4dgs_episode",
        "episode_id": "episode_001",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": {"complete": False, "rendered": False, "audited": False},
        "coordinate_system": {
            "world": "right-handed Isaac/USD XYZ",
            "unit": "metre",
            "up_axis": "+Y",
            "canonical_matrix_convention": "column vectors; translation in last column",
            "mesh_coordinates": "world-space; no per-frame object transform",
        },
        "observations": {
            "fluid_channel_visibility_scope": (
                "fluid-only; non-fluid scene objects are excluded and therefore do not "
                "occlude fluid_rgba, mask, depth, or normal"
            ),
            "rgb": "8-bit PNG, AgX Medium High Contrast",
            "fluid_rgba": "8-bit PNG on transparent background",
            "fluid_mask": (
                "8-bit PNG; 255 when the OpenCV pixel-center ray first-hits the "
                "fluid mesh and fluid_rgba has nonzero coverage, otherwise 0"
            ),
            "fluid_depth": {
                "format": "float32 OpenEXR",
                "unit": "metre",
                "meaning": "positive camera optical-axis Z-depth (+Z forward in OpenCV camera coordinates)",
                "background_value": 0.0,
                "validity": "mask.png == 255",
                "sampling": "first mesh intersection along the pixel-center ray; no multisample coverage averaging",
            },
            "world_normal": {
                "format": "float32 OpenEXR",
                "meaning": "unit first-hit geometric normal in Isaac world XYZ",
                "background_value": [0.0, 0.0, 0.0],
            },
        },
        "physics": {
            "engine": run_report["physics"],
            "gravity_world_metres_per_second_squared": [0.0, -9.81, 0.0],
            "physics_step_seconds": 1.0 / float(run_report["physics_fps"]),
            "physics_fps": int(run_report["physics_fps"]),
            "source_state_fps": 120,
            "episode_state_fps": 120,
            "solver_iterations": int(run_report["solver_iterations"]),
            "particle_count": int(run_report["particle_count"]),
            "particle_spacing_metres": float(run_report["particle_spacing"]),
            "particle_mass_kilograms": float(run_report["particle_mass"]),
            "fluid_material": {
                "density_kilograms_per_cubic_metre": 1000.0,
                "friction": 0.06,
                **run_report["fluid_material"],
            },
            "rigid_material": {
                "mass_kilograms": float(run_report["impactor"]["mass"]),
                "friction": None,
                "restitution": None,
                "availability": "not recorded in the source run report",
            },
            "rigid_fluid_coupling": run_report["rigid_fluid_coupling"],
        },
        "initial_water": run_report["pool_initialization"],
        "fixed_scene_and_collision": {
            "source_blend": str(args.scene.resolve()),
            "source_blend_sha256": sha256_file(args.scene),
            "terrain_collision": run_report["basin_collision"],
            "terrain_heightfield": str(terrain_path),
            "terrain_heightfield_sha256": sha256_file(terrain_path),
        },
        "surface_reconstruction": {
            "method": "pysplashsurf 0.14.1 plus audited free-surface clipping",
            "configuration": surface_configuration,
            "temporal_rate_hz": 120,
            "shoreline_reference_source_sample": int(args.reference_sample),
        },
        "rendering": {
            "engine": "BLENDER_EEVEE",
            "samples": int(args.samples),
            "resolution": [int(args.width), int(args.height)],
            "view_count": int(args.views),
            "lighting": {
                "hdri": str(args.hdri.resolve()),
                "hdri_sha256": sha256_file(args.hdri),
                "strength": 0.95,
            },
            "exposure": 0.0,
            "view_transform": "AgX",
            "look": "AgX - Medium High Contrast",
            "water_material": {
                "base_color": [0.82, 0.94, 0.98, 1.0],
                "roughness": 0.03,
                "ior": 1.333,
                "transmission": 1.0,
                "geometry": "open free-surface sheet",
            },
            "fixed_across_episode": True,
            "geometry_label_sampling": (
                "BVH first hit at OpenCV integer pixel index + 0.5; RGB/RGBA remain "
                "antialiased Blender renders"
            ),
        },
        "provenance": {
            "physx_source_manifest": str(source_directory / "manifest.json"),
            "physx_source_manifest_sha256": sha256_file(source_directory / "manifest.json"),
            "physx_source_audit": str(source_directory / "audit_report.json"),
            "physx_source_audit_sha256": sha256_file(source_directory / "audit_report.json"),
            "run_report": str(run_report_path),
            "run_report_sha256": sha256_file(run_report_path),
            "surface_manifest": str(surface_manifest_path),
            "surface_manifest_sha256": sha256_file(surface_manifest_path),
        },
        "files": {
            "timeline": "timeline.json",
            "controls": "controls.json",
            "cameras": "cameras.json",
            "scene": "scene/scene.blend",
            "scene_entry": "scene/scene_entry.json",
            "audit": "audit_report.json",
        },
        "frames": frame_records,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "scene" / "scene_entry.json",
        {
            "schema": 1,
            "source_scene": str(args.scene.resolve()),
            "source_scene_sha256": sha256_file(args.scene),
            "episode_scene": "scene.blend",
            "render_script": str(
                (SCRIPT_DIRECTORY / "render_fluid_4dgs_episode_blender.py").resolve()
            ),
            "notes": (
                "The Blender scene uses its native Z-up environment. Episode meshes, "
                "cameras and states remain Isaac Y-up; the render entry applies the "
                "fixed rotation Blender(x,y,z)=Isaac(x,-z,y)."
            ),
        },
    )

    if not args.skip_render:
        run_checked(
            [
                args.blender.resolve(),
                args.scene.resolve(),
                "--background",
                "--python",
                SCRIPT_DIRECTORY / "render_fluid_4dgs_episode_blender.py",
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
                SCRIPT_DIRECTORY / "audit_fluid_4dgs_episode_blender.py",
                "--",
                output,
            ]
        )
    print(f"FLUID_4DGS_EPISODE={output}")


if __name__ == "__main__":
    main()
