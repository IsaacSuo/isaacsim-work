"""Simulate and render mixed-material, multi-object scene videos."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from run_experiments import (
    BASELINE_SCENE_RUNS,
    BLENDER,
    BLENDER_SCRIPT,
    HERO_SCRIPT,
    ISAAC_PYTHON,
    ROOT,
    SCENE_CONFIG_PATH,
    ensure_collision_asset,
    load_scene_context,
    recorded_path_matches,
)
from render_videos import FFMPEG, scaled_camera


DEFAULT_CONFIG = ROOT / "configs" / "multi_object_scene_experiments.json"
PROFILE_CONFIG = ROOT / "configs" / "model_material_experiments.json"
SIMULATION_MODELS = ROOT / "assets" / "simulation_ready_models"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output" / "multi_object_mixed_material_14"
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--stage", choices=("physics", "render", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--debug-deformable-frame",
        type=int,
        default=None,
        help="Export authoritative deformable debug geometry at this one-based physics frame.",
    )
    parser.add_argument(
        "--audit-tet-trajectory",
        action="store_true",
        help="Record per-frame simulation-Tet deformation metrics during physics.",
    )
    parser.add_argument(
        "--deformable-solver-position-iterations",
        type=int,
        default=24,
        help="Position iterations used by deformable bodies (default: 24).",
    )
    return parser.parse_args()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def physics_cache_matches(
    report,
    frame_count,
    body_config,
    collision_asset=None,
    debug_deformable_frame=None,
    audit_tet_trajectory=False,
):
    actual_bodies = report.get("physics_bodies") or []
    expected_bodies = body_config["bodies"]
    if not report.get("valid") or report.get("physics_frames") != frame_count:
        return False
    if report.get("physics_substeps") != body_config["physics_substeps"]:
        return False
    if (
        (report.get("physics_material") or {}).get("solver_position_iterations")
        != body_config["deformable_solver_position_iterations"]
    ):
        return False
    if (
        (report.get("physics_material") or {}).get("deformable_resolution")
        != body_config["deformable_resolution"]
    ):
        return False
    recorded_material = report.get("physics_material") or {}
    collision_policy = body_config["deformable_collision_policy"]
    policy_report_keys = {
        "remeshing_enabled": "collision_remeshing",
        "remeshing_resolution": "remeshing_resolution",
        "target_triangle_count": "target_triangle_count",
        "force_conforming": "force_conforming",
    }
    if any(
        recorded_material.get(report_key) != collision_policy[policy_key]
        for policy_key, report_key in policy_report_keys.items()
    ):
        return False
    if debug_deformable_frame is not None:
        debug_export = report.get("deformable_collision_debug") or {}
        debug_path = Path(debug_export.get("path") or "")
        if (
            debug_export.get("frame") != debug_deformable_frame
            or not debug_path.is_file()
            or debug_path.stat().st_size <= 0
        ):
            return False
    if audit_tet_trajectory:
        trajectory = report.get("tet_deformation_trajectory") or {}
        trajectory_path = Path(trajectory.get("path") or "")
        if (
            not trajectory.get("bodies")
            or not trajectory_path.is_file()
            or trajectory_path.stat().st_size <= 0
        ):
            return False
    if collision_asset is not None:
        if not recorded_path_matches(report.get("prebuilt_collision_usd"), collision_asset):
            return False
    elif report.get("prebuilt_collision_usd"):
        return False
    if len(actual_bodies) != len(expected_bodies):
        return False
    scalar_keys = (
        "model_height",
        "model_yaw",
        "density",
        "youngs_modulus",
        "poissons_ratio",
        "linear_damping",
        "settling_damping",
        "restitution",
        "collision_contact_offset",
        "collision_rest_offset",
    )
    collision_keys = (
        "collision_approximation",
        "sdf_resolution",
        "sdf_enable_remeshing",
    )
    for actual, expected in zip(actual_bodies, expected_bodies):
        if not recorded_path_matches(actual.get("model"), expected["model"]):
            return False
        if actual.get("physics_profile") != expected.get("physics_profile"):
            return False
        if actual.get("material_preset") != expected.get("material_preset"):
            return False
        if actual.get("physics_kind") != expected.get("physics_kind"):
            return False
        if any(actual.get(key) != expected.get(key) for key in collision_keys):
            return False
        if any(
            not math.isclose(
                float(actual.get(key, float("nan"))),
                float(expected[key]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for key in scalar_keys
        ):
            return False
        expected_spawn = (
            expected["spawn_x"],
            expected["drop_height"],
            expected["spawn_z"],
        )
        actual_spawn = actual.get("spawn") or []
        if len(actual_spawn) != 3 or any(
            not math.isclose(float(a), float(e), rel_tol=1e-9, abs_tol=1e-9)
            for a, e in zip(actual_spawn, expected_spawn)
        ):
            return False
    return bool(
        report.get("interbody_contact_required")
        and report.get("interbody_contact_valid")
        and int(report.get("detected_interbody_contact_pairs", 0)) > 0
    )


def validate_config(config, profiles, material_presets, scene_configs):
    experiments = config.get("experiments") or []
    expected_scenes = set(scene_configs)
    actual_scenes = [experiment.get("scene") for experiment in experiments]
    if len(experiments) != 14 or set(actual_scenes) != expected_scenes or len(set(actual_scenes)) != 14:
        raise ValueError("Multi-object config must contain every scene exactly once (14 scenes)")
    ids = [experiment.get("id") for experiment in experiments]
    if len(set(ids)) != len(ids):
        raise ValueError("Multi-object experiment ids must be unique")
    model_pool = set(config.get("simulation_model_pool") or [])
    collision_policies = config.get("rigid_collision_policies") or {}
    if set(collision_policies) != model_pool:
        raise ValueError(
            "Rigid collision policies must cover the complete simulation model pool"
        )
    for model, policy in collision_policies.items():
        approximation = policy.get("approximation")
        if approximation not in {"convexDecomposition", "sdf"}:
            raise ValueError(
                f"Unsupported rigid collision approximation for {model}: {approximation}"
            )
        if approximation == "sdf":
            resolution = int(policy.get("sdf_resolution", 0))
            if resolution <= 1:
                raise ValueError(f"Invalid SDF resolution for {model}: {resolution}")
    deformable_policy = config.get("deformable_collision_policy") or {}
    required_deformable_policy = {
        "remeshing_enabled",
        "remeshing_resolution",
        "target_triangle_count",
        "force_conforming",
    }
    if set(deformable_policy) != required_deformable_policy:
        raise ValueError("Deformable collision policy is incomplete")
    if int(deformable_policy["remeshing_resolution"]) < 0:
        raise ValueError("Deformable remeshing resolution must be non-negative")
    if int(deformable_policy["target_triangle_count"]) < 0:
        raise ValueError("Deformable target triangle count must be non-negative")
    for experiment in experiments:
        bodies = experiment.get("bodies") or []
        if len(bodies) < 3:
            raise ValueError(f"{experiment['id']} must contain at least three bodies")
        if len({body["model"] for body in bodies}) != len(bodies):
            raise ValueError(f"{experiment['id']} repeats a model within one scene")
        if len({body["material_preset"] for body in bodies}) < 3:
            raise ValueError(f"{experiment['id']} must use at least three distinct materials")
        for body in bodies:
            profile = profiles[body["physics_profile"]]
            material = material_presets[body["material_preset"]]
            if profile["expected_behavior"] != material["allowed_behavior"]:
                raise ValueError(
                    f"{experiment['id']} has incompatible physics/material pair: "
                    f"{body['physics_profile']} + {body['material_preset']}"
                )


def build_body_config(
    experiment,
    scene,
    profiles,
    collision_policies,
    deformable_collision_policy,
):
    bodies = []
    for body in experiment["bodies"]:
        model_path = SIMULATION_MODELS / body["model"]
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        profile = profiles[body["physics_profile"]]
        expected_behavior = profile["expected_behavior"]
        physics_kind = "rigid" if expected_behavior == "hard" else "deformable"
        if physics_kind == "rigid":
            collision_policy = collision_policies[body["model"]]
            collision_approximation = body.get(
                "collision_approximation", collision_policy["approximation"]
            )
            if collision_approximation not in {
                "convexDecomposition",
                "sdf",
            }:
                raise ValueError(
                    f"Unsupported body collision approximation: {collision_approximation}"
                )
            sdf_resolution = (
                int(body.get("sdf_resolution", collision_policy.get("sdf_resolution", 384)))
                if collision_approximation == "sdf"
                else None
            )
            sdf_enable_remeshing = (
                bool(
                    body.get(
                        "sdf_enable_remeshing",
                        collision_policy.get("sdf_enable_remeshing", False),
                    )
                )
                if collision_approximation == "sdf"
                else None
            )
        else:
            collision_approximation = "tetrahedral"
            sdf_resolution = None
            sdf_enable_remeshing = None
        bodies.append(
            {
                "model": str(model_path),
                "model_height": body["model_height"],
                "model_yaw": body.get("model_yaw", 0.0),
                "spawn_x": scene["spawn_x"] + body.get("offset_x", 0.0),
                "drop_height": scene["drop_height"] + body.get("drop_offset", 0.0),
                "spawn_z": scene["spawn_z"] + body.get("offset_z", 0.0),
                "density": profile["density"],
                "youngs_modulus": profile["youngs_modulus"],
                "poissons_ratio": profile["poissons_ratio"],
                "linear_damping": profile["linear_damping"],
                "settling_damping": profile["settling_damping"],
                "restitution": profile["restitution"],
                "collision_contact_offset": body.get("collision_contact_offset", 0.03),
                "collision_rest_offset": body.get("collision_rest_offset", 0.01),
                "physics_kind": physics_kind,
                "collision_approximation": collision_approximation,
                "sdf_resolution": sdf_resolution,
                "sdf_enable_remeshing": sdf_enable_remeshing,
                "physics_profile": body["physics_profile"],
                "material_preset": body["material_preset"],
            }
        )
    return {
        "physics_substeps": int(experiment.get("physics_substeps", 4)),
        "deformable_resolution": int(experiment.get("deformable_resolution", 24)),
        "deformable_collision_policy": dict(deformable_collision_policy),
        "bodies": bodies,
    }


def run_physics(
    args,
    experiment,
    scene,
    run_dir,
    profiles,
    collision_policies,
    deformable_collision_policy,
):
    physics_dir = run_dir / "isaac" / scene["name"]
    report_path = physics_dir / "run_complete.json"
    cache_path = physics_dir / "soft_body_blender.usdc"
    body_config_path = physics_dir / "bodies.json"
    body_config = build_body_config(
        experiment,
        scene,
        profiles,
        collision_policies,
        deformable_collision_policy,
    )
    body_config["deformable_solver_position_iterations"] = int(
        args.deformable_solver_position_iterations
    )
    write_json(body_config_path, body_config)
    collision_asset = ensure_collision_asset(args.output, scene) if scene["needs_exact_collision"] else None
    if not args.force and report_path.is_file() and cache_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        exported = report.get("blender_animation_cache") or {}
        if (
            physics_cache_matches(
                report,
                args.frames,
                body_config,
                collision_asset,
                args.debug_deformable_frame,
                args.audit_tet_trajectory,
            )
            and exported.get("valid")
            and exported.get("body_count") == len(body_config["bodies"])
        ):
            print(f"[physics-reuse] {experiment['id']}", flush=True)
            return report, physics_dir

    primary = body_config["bodies"][0]
    command = [
        str(ISAAC_PYTHON), str(HERO_SCRIPT),
        "--frames", str(args.frames),
        "--substeps", str(body_config["physics_substeps"]),
        "--width", "480", "--height", "480", "--renderer", "RaytracedLighting",
        "--output", str(physics_dir),
        "--model", primary["model"],
        "--model-height", str(primary["model_height"]),
        "--model-scale-mode", "max_extent",
        "--model-yaw", str(primary["model_yaw"]),
        "--drop-height", str(primary["drop_height"]),
        "--youngs-modulus", str(primary["youngs_modulus"]),
        "--poissons-ratio", str(primary["poissons_ratio"]),
        "--linear-damping", str(primary["linear_damping"]),
        "--settling-damping", str(primary["settling_damping"]),
        "--restitution", str(primary["restitution"]),
        "--density", str(primary["density"]),
        "--expected-behavior", "soft",
        "--validation-profile", "generic",
        "--deformable-resolution", str(body_config["deformable_resolution"]),
        "--deformable-solver-position-iterations",
        str(body_config["deformable_solver_position_iterations"]),
        "--environment-usd", str(scene["usd"]),
        "--environment-ground-only",
        "--support-top-y", str(scene["support_y"]),
        "--spawn-x", str(scene["spawn_x"]),
        "--spawn-z", str(scene["spawn_z"]),
        "--camera-eye", *[str(value) for value in scene["camera_eye"]],
        "--camera-target", *[str(value) for value in scene["camera_target"]],
        "--body-config", str(body_config_path),
        "--export-blender-usd", "--blender-usd-name", "soft_body_blender.usdc",
        "--skip-preview-render",
        "--require-interbody-contact",
    ]
    deformable_policy = body_config["deformable_collision_policy"]
    if deformable_policy["remeshing_enabled"]:
        command.append("--deformable-collision-remeshing")
    command.extend(
        [
            "--deformable-remeshing-resolution",
            str(deformable_policy["remeshing_resolution"]),
            "--deformable-target-triangle-count",
            str(deformable_policy["target_triangle_count"]),
        ]
    )
    if deformable_policy["force_conforming"]:
        command.append("--deformable-force-conforming")
    if args.debug_deformable_frame is not None:
        command.extend(
            [
                "--debug-deformable-frame",
                str(args.debug_deformable_frame),
                "--debug-deformable-usd-name",
                f"deformable_collision_debug_frame{args.debug_deformable_frame}.usdc",
            ]
        )
    if args.audit_tet_trajectory:
        command.append("--audit-tet-trajectory")
    if collision_asset:
        command.extend(["--prebuilt-collision-usd", str(collision_asset)])
    ground_patch = scene["ground_patch"]
    if ground_patch:
        command.extend(
            [
                "--ground-patch-size-x", str(ground_patch.get("size_x", 8.0)),
                "--ground-patch-size-z", str(ground_patch.get("size_z", 8.0)),
                "--ground-patch-center-x", str(ground_patch.get("center_x", scene["spawn_x"])),
                "--ground-patch-center-z", str(ground_patch.get("center_z", scene["spawn_z"])),
            ]
        )
    if scene["allow_free_fall"]:
        command.append("--allow-free-fall")
    if scene["uneven_ground"]:
        command.append("--uneven-ground")
    print(f"[physics-start] {experiment['id']} bodies={len(body_config['bodies'])}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0 or not report_path.is_file():
        raise RuntimeError(f"Physics failed with code {completed.returncode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise RuntimeError(f"Physics report is invalid: {report.get('error')}")
    return report, physics_dir


def final_state_camera(scene, experiment, physics_report):
    """Frame settled bodies while preserving the scene's authored view direction."""
    eye, target = scaled_camera(scene, experiment)
    if scene["name"] in {"mountain", "hospital"}:
        return eye, target, "fixed_free_fall"
    states = physics_report.get("body_final_states") or []
    if not states:
        return eye, target, "fixed_missing_final_states"

    final_minimum = [min(float(state["minimum"][axis]) for state in states) for axis in range(3)]
    final_maximum = [max(float(state["maximum"][axis]) for state in states) for axis in range(3)]
    final_center = tuple(
        (final_minimum[axis] + final_maximum[axis]) * 0.5 for axis in range(3)
    )
    motion_states = physics_report.get("body_camera_motion_states") or states
    motion_minimum = [
        min(float(state["minimum"][axis]) for state in motion_states) for axis in range(3)
    ]
    motion_maximum = [
        max(float(state["maximum"][axis]) for state in motion_states) for axis in range(3)
    ]
    # The final landing layout owns the composition.  The complete motion
    # bounds below only determine how far the camera must retreat, so an early
    # high drop cannot pull the camera target away from the settled objects.
    framed_target = final_center
    if scene["name"] == "bedroom":
        # Keep the validated clear bedside direction, but derive both the
        # target and the final eye position from this run's measured bounds.
        # The authored negative-Z direction looks through the bed frame.
        view_offset = (-5.2, 2.8, 0.0)
        camera_policy = "final_body_bounds_clear_bedside"
    else:
        view_offset = tuple(float(eye[axis]) - float(target[axis]) for axis in range(3))
        camera_policy = "final_body_bounds"
    current_distance = math.sqrt(sum(value * value for value in view_offset))
    if current_distance <= 1e-6:
        return eye, target, "fixed_degenerate_view"
    view_direction = tuple(value / current_distance for value in view_offset)

    motion_corners = [
        (x, y, z)
        for x in (motion_minimum[0], motion_maximum[0])
        for y in (motion_minimum[1], motion_maximum[1])
        for z in (motion_minimum[2], motion_maximum[2])
    ]
    forward = tuple(-value for value in view_direction)
    world_up = (0.0, 1.0, 0.0)
    right = (
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    )
    right_length = math.sqrt(sum(value * value for value in right))
    if right_length <= 1e-6:
        return eye, target, "fixed_vertical_view"
    right = tuple(value / right_length for value in right)
    camera_up = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    safe_tangent = math.tan(math.radians(17.0))
    required_distance = 0.0
    for corner in motion_corners:
        delta = tuple(corner[axis] - framed_target[axis] for axis in range(3))
        horizontal = abs(sum(delta[axis] * right[axis] for axis in range(3)))
        vertical = abs(sum(delta[axis] * camera_up[axis] for axis in range(3)))
        depth_offset = sum(delta[axis] * forward[axis] for axis in range(3))
        required_distance = max(
            required_distance,
            horizontal / safe_tangent - depth_offset,
            vertical / safe_tangent - depth_offset,
        )
    required_distance += 0.15
    minimum_distance = current_distance * 0.72
    framed_distance = max(minimum_distance, required_distance)
    framed_eye = tuple(
        framed_target[axis] + view_direction[axis] * framed_distance for axis in range(3)
    )
    return framed_eye, framed_target, camera_policy


def run_render(args, experiment, scene, physics_dir, physics_report, frame_count):
    frames_dir = args.output / "video_frames" / experiment["id"]
    video_path = args.output / "videos" / f"{experiment['id']}.mp4"
    report_path = frames_dir / "blender_render_report.json"
    physics_report_path = physics_dir / "run_complete.json"
    cache_path = physics_dir / "soft_body_blender.usdc"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    presets = ",".join(body["material_preset"] for body in experiment["bodies"])
    eye, target, camera_policy = final_state_camera(scene, experiment, physics_report)
    if not args.force and report_path.is_file() and video_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        render_is_newer_than_physics = (
            cache_path.is_file()
            and physics_report_path.is_file()
            and report_path.stat().st_mtime_ns
            >= max(
                cache_path.stat().st_mtime_ns,
                physics_report_path.stat().st_mtime_ns,
            )
            and video_path.stat().st_mtime_ns >= report_path.stat().st_mtime_ns
        )
        if (
            render_is_newer_than_physics
            and report.get("valid")
            and report.get("body_count") == len(experiment["bodies"])
            and report.get("material_presets") == presets.split(",")
            and report.get("rendered_frame_count") == frame_count
            and report.get("camera_eye_isaac") == list(eye)
            and report.get("camera_target_isaac") == list(target)
        ):
            print(f"[video-reuse] {experiment['id']}", flush=True)
            return report, video_path
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()
    command = [
        str(BLENDER), str(scene["blend"]), "--background",
        "--python", str(BLENDER_SCRIPT), "--",
        str(physics_dir / "soft_body_blender.usdc"), str(frames_dir),
        str(frame_count), str(args.samples), str(args.resolution), "all",
        "true" if scene["allow_free_fall"] or scene["uneven_ground"] else "false",
        *[str(value) for value in eye], *[str(value) for value in target],
        "false", "", "", "", "0", "0", "0", presets,
    ]
    print(
        f"[render-start] {experiment['id']} materials={presets} "
        f"camera_policy={camera_policy} eye={eye} target={target}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0 or not report_path.is_file():
        raise RuntimeError(f"Blender render failed with code {completed.returncode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("valid") or report.get("body_count") != len(experiment["bodies"]):
        raise RuntimeError("Blender report did not validate all bodies")
    encode = subprocess.run(
        [
            str(FFMPEG), "-y", "-framerate", "60", "-start_number", "1",
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
        ],
        cwd=ROOT,
        check=False,
    )
    if encode.returncode != 0 or not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"FFmpeg encode failed with code {encode.returncode}")
    return report, video_path


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile_config = json.loads(PROFILE_CONFIG.read_text(encoding="utf-8"))
    profiles = profile_config["physics_profiles"]
    material_presets = profile_config["material_presets"]
    collision_policies = config["rigid_collision_policies"]
    deformable_collision_policy = config["deformable_collision_policy"]
    scene_configs = json.loads(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config, profiles, material_presets, scene_configs)
    selected = set(args.only or [row["id"] for row in config["experiments"]])
    results = []
    for experiment in config["experiments"]:
        if experiment["id"] not in selected:
            continue
        result = {"id": experiment["id"], "scene": experiment["scene"], "valid": False}
        try:
            scene = load_scene_context(experiment["scene"], scene_configs)
            run_dir = args.output / "runs" / experiment["id"]
            physics_dir = run_dir / "isaac" / scene["name"]
            if args.stage in {"physics", "all"}:
                physics_report, physics_dir = run_physics(
                    args,
                    experiment,
                    scene,
                    run_dir,
                    profiles,
                    collision_policies,
                    deformable_collision_policy,
                )
            else:
                report_path = physics_dir / "run_complete.json"
                cache_path = physics_dir / "soft_body_blender.usdc"
                physics_report = json.loads(report_path.read_text(encoding="utf-8"))
                body_config = build_body_config(
                    experiment,
                    scene,
                    profiles,
                    collision_policies,
                    deformable_collision_policy,
                )
                body_config["deformable_solver_position_iterations"] = int(
                    args.deformable_solver_position_iterations
                )
                collision_asset = (
                    ROOT
                    / "output"
                    / "scene_collision_assets"
                    / f"{scene['name']}_exact_collision.usdc"
                    if scene["needs_exact_collision"]
                    else None
                )
                exported = physics_report.get("blender_animation_cache") or {}
                if (
                    not cache_path.is_file()
                    or cache_path.stat().st_size <= 0
                    or not physics_cache_matches(
                        physics_report,
                        args.frames,
                        body_config,
                        collision_asset,
                        args.debug_deformable_frame,
                        args.audit_tet_trajectory,
                    )
                    or not exported.get("valid")
                    or exported.get("body_count") != len(body_config["bodies"])
                ):
                    raise RuntimeError(
                        "Existing physics report/cache is invalid or does not match "
                        "the current production configuration"
                    )
            result["physics_report"] = str(physics_dir / "run_complete.json")
            result["body_count"] = len(experiment["bodies"])
            if args.stage in {"render", "all"}:
                render_report, video_path = run_render(
                    args,
                    experiment,
                    scene,
                    physics_dir,
                    physics_report,
                    int(physics_report["physics_frames"]),
                )
                result.update(
                    {
                        "render_report": str(args.output / "video_frames" / experiment["id"] / "blender_render_report.json"),
                        "video": str(video_path),
                        "rendered_frames": render_report["rendered_frame_count"],
                    }
                )
            result["valid"] = True
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            print(f"[failed] {experiment['id']} {result['error']}", flush=True)
        results.append(result)
        write_json(
            args.output / "summary.json",
            {"valid": False, "seed": config.get("seed"), "results": results},
        )
    valid = bool(results) and all(row["valid"] for row in results)
    write_json(
        args.output / "summary.json",
        {"valid": valid, "seed": config.get("seed"), "results": results},
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
