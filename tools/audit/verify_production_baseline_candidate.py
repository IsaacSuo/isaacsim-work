"""Verify the current 14-scene PhysX-to-Blender baseline candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PureWindowsPath

from experiments.model_material.run_experiments import load_scene_context


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "multi_object_scene_experiments.json"
PROFILE_CONFIG = ROOT / "configs" / "model_material_experiments.json"
SCENE_CONFIG = ROOT / "configs" / "scene_experiments.json"
ENVIRONMENT_CONFIG = ROOT / "configs" / "production_environment.json"
DEFAULT_OUTPUT = ROOT / "output" / "multi_object_mixed_material_14"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=640)
    return parser.parse_args()


def model_name(value):
    return PureWindowsPath(str(value)).name


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILE_CONFIG.read_text(encoding="utf-8"))["physics_profiles"]
    scenes = json.loads(SCENE_CONFIG.read_text(encoding="utf-8"))
    environment = json.loads(ENVIRONMENT_CONFIG.read_text(encoding="utf-8"))
    policies = config["rigid_collision_policies"]
    failures = []
    rows = []

    summary_path = args.output / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    summary_results = {row.get("id"): row for row in summary.get("results", [])}

    for experiment in config["experiments"]:
        experiment_id = experiment["id"]
        scene = experiment["scene"]
        physics_dir = args.output / "runs" / experiment_id / "isaac" / scene
        report_path = physics_dir / "run_complete.json"
        cache_path = physics_dir / "soft_body_blender.usdc"
        render_path = args.output / "video_frames" / experiment_id / "blender_render_report.json"
        video_path = args.output / "videos" / f"{experiment_id}.mp4"
        errors = []

        if not report_path.is_file():
            errors.append("missing physics report")
            report = {}
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        bodies = report.get("physics_bodies") or []
        if not report.get("valid"):
            errors.append("physics report is invalid")
        if report.get("isaac_sim_version") != environment["isaac_sim"]["version"]:
            errors.append("Isaac Sim version mismatch")
        if report.get("physics_frames") != args.frames:
            errors.append("physics frame count mismatch")
        if len(bodies) != len(experiment["bodies"]):
            errors.append("physics body count mismatch")
        else:
            for index, (actual, expected) in enumerate(zip(bodies, experiment["bodies"])):
                profile = profiles[expected["physics_profile"]]
                scene_context = load_scene_context(scene, scenes)
                physics_kind = "rigid" if profile["expected_behavior"] == "hard" else "deformable"
                expected_policy = (
                    policies[expected["model"]]
                    if physics_kind == "rigid"
                    else {"approximation": "tetrahedral"}
                )
                if model_name(actual.get("model")) != expected["model"]:
                    errors.append(f"body {index} model mismatch")
                if actual.get("physics_kind") != physics_kind:
                    errors.append(f"body {index} physics kind mismatch")
                if actual.get("physics_profile") != expected["physics_profile"]:
                    errors.append(f"body {index} physics profile mismatch")
                if actual.get("material_preset") != expected["material_preset"]:
                    errors.append(f"body {index} material mismatch")
                if actual.get("collision_approximation") != expected_policy["approximation"]:
                    errors.append(f"body {index} collision approximation mismatch")
                if actual.get("sdf_resolution") != expected_policy.get("sdf_resolution"):
                    errors.append(f"body {index} SDF resolution mismatch")
                if bool(actual.get("sdf_enable_remeshing", False)) != bool(
                    expected_policy.get("sdf_enable_remeshing", False)
                ):
                    errors.append(f"body {index} SDF remeshing mismatch")
                scalar_expectations = {
                    "model_height": expected["model_height"],
                    "model_yaw": expected.get("model_yaw", 0.0),
                    "density": profile["density"],
                    "youngs_modulus": profile["youngs_modulus"],
                    "poissons_ratio": profile["poissons_ratio"],
                    "linear_damping": profile["linear_damping"],
                    "settling_damping": profile["settling_damping"],
                    "restitution": profile["restitution"],
                    "collision_contact_offset": expected.get(
                        "collision_contact_offset", 0.03
                    ),
                    "collision_rest_offset": expected.get("collision_rest_offset", 0.01),
                }
                for key, expected_value in scalar_expectations.items():
                    try:
                        matches = math.isclose(
                            float(actual.get(key)),
                            float(expected_value),
                            rel_tol=1.0e-9,
                            abs_tol=1.0e-9,
                        )
                    except (TypeError, ValueError):
                        matches = False
                    if not matches:
                        errors.append(f"body {index} {key} mismatch")
                expected_spawn = (
                    float(scene_context["spawn_x"]) + float(expected.get("offset_x", 0.0)),
                    float(scene_context["drop_height"]) + float(expected.get("drop_offset", 0.0)),
                    float(scene_context["spawn_z"]) + float(expected.get("offset_z", 0.0)),
                )
                actual_spawn = actual.get("spawn") or []
                if len(actual_spawn) != 3 or any(
                    not math.isclose(float(value), target, rel_tol=1.0e-9, abs_tol=1.0e-9)
                    for value, target in zip(actual_spawn, expected_spawn)
                ):
                    errors.append(f"body {index} spawn mismatch")

        cache = report.get("blender_animation_cache") or {}
        if not cache_path.is_file() or cache_path.stat().st_size <= 0:
            errors.append("missing/empty Blender USD cache")
        if not cache.get("valid") or cache.get("body_count") != len(experiment["bodies"]):
            errors.append("Blender USD cache metadata is invalid")
        if cache.get("frame_count") != args.frames:
            errors.append("Blender USD cache frame count mismatch")
        if not report.get("interbody_contact_required"):
            errors.append("inter-body contact was not required")
        if not report.get("interbody_contact_valid"):
            errors.append("inter-body contact validation failed")
        if int(report.get("detected_interbody_contact_pairs", 0)) <= 0:
            errors.append("no inter-body contact pair detected")
        if not report.get("all_bodies_penetration_valid"):
            errors.append("ground penetration validation failed")

        if not render_path.is_file():
            errors.append("missing Blender render report")
            render = {}
        else:
            render = json.loads(render_path.read_text(encoding="utf-8"))
        expected_materials = [body["material_preset"] for body in experiment["bodies"]]
        if not render.get("valid"):
            errors.append("Blender render report is invalid")
        if render.get("body_count") != len(experiment["bodies"]):
            errors.append("rendered body count mismatch")
        if render.get("material_presets") != expected_materials:
            errors.append("rendered material order mismatch")
        if render.get("rendered_frame_count") != args.frames:
            errors.append("rendered frame count mismatch")
        if render.get("cycles_samples") != args.samples:
            errors.append("Cycles sample count mismatch")
        if render.get("resolution") != [args.resolution, args.resolution]:
            errors.append("render resolution mismatch")
        if render.get("fps") != environment["timing"]["video_fps"]:
            errors.append("render FPS mismatch")
        if render.get("renderer") != environment["blender"]["render_engine"]:
            errors.append("render engine mismatch")
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            errors.append("missing/empty H.264 video")
        if not summary_results.get(experiment_id, {}).get("valid"):
            errors.append("batch summary does not mark sequence valid")

        rows.append({"id": experiment_id, "scene": scene, "valid": not errors, "errors": errors})
        failures.extend(f"{experiment_id}: {error}" for error in errors)
        print(f"[baseline-candidate] {experiment_id} valid={not errors} errors={len(errors)}")

    if set(summary_results) != {row["id"] for row in config["experiments"]}:
        failures.append("batch summary does not contain exactly the configured experiments")
    if not summary.get("valid"):
        failures.append("batch summary is not globally valid")

    payload = {
        "valid": not failures and len(rows) == 14,
        "config": str(args.config),
        "output": str(args.output),
        "physics_frames": args.frames,
        "results": rows,
        "failures": failures,
    }
    audit_path = args.output / "production_baseline_candidate_audit.json"
    audit_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[baseline-candidate] complete valid={payload['valid']} report={audit_path}")
    for failure in failures:
        print(f"[baseline-candidate] failure={failure}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
