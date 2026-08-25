"""Run reproducible model, stiffness, and Cycles material experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[2]


def configured_path(environment_name, *fallbacks):
    configured = os.environ.get(environment_name)
    if configured:
        return Path(configured).expanduser().resolve()
    for fallback in fallbacks:
        if fallback and Path(fallback).is_file():
            return Path(fallback).resolve()
    for fallback in fallbacks:
        if fallback:
            return Path(fallback).expanduser()
    return Path(environment_name)


def recorded_path_matches(recorded, expected):
    """Compare current paths with reports produced on Windows or Linux."""
    if not recorded:
        return False
    expected = Path(expected)
    candidate = Path(str(recorded))
    if candidate == expected:
        return True
    # A Windows absolute path is a single, literal filename on POSIX.  Reports
    # are portable when the final asset name still identifies the same input.
    return PureWindowsPath(str(recorded)).name == expected.name


bundled_scenes = ROOT / "scenes"
legacy_scenes = Path(r"Y:\scenes")
SCENES = Path(os.environ.get("SCENES_ROOT", bundled_scenes)).expanduser()
if not SCENES.is_dir() and legacy_scenes.is_dir():
    SCENES = legacy_scenes

ISAAC_PYTHON = configured_path(
    "ISAAC_PYTHON",
    ROOT.parent / "isaacsim" / ("python.bat" if os.name == "nt" else "python.sh"),
    sys.executable,
)
BLENDER = configured_path(
    "BLENDER_BIN",
    shutil.which("blender"),
    Path(r"D:\Program Files (x86)\Blender\blender.exe"),
)
HERO_SCRIPT = ROOT / "soft_body_bounce_hero.py"
BLENDER_SCRIPT = ROOT / "tools" / "blender" / "render_blender_soft_body_cache.py"
COLLISION_EXTRACTOR = ROOT / "tools" / "scenes" / "extract_collision_asset.py"
MODELS = ROOT / "assets" / "archieved_models"
HOSPITAL_USD = SCENES / "hospital" / "hospital_sim.usda"
HOSPITAL_BLEND = SCENES / "hospital.blend"
SCENE_CONFIG_PATH = ROOT / "configs" / "scene_experiments.json"
BASELINE_SCENE_RUNS = ROOT / "output" / "blender_scene_videos" / "isaac"
BASELINE_SCENE_RENDERS = ROOT / "output" / "blender_scene_videos" / "cycles_frames"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "model_material_experiments.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "model_material_experiments")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--stage", choices=("collision", "physics", "render", "all"), default="all"
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--deformable-resolution", type=int, default=24)
    parser.add_argument("--collision-contact-offset", type=float, default=None)
    parser.add_argument("--collision-rest-offset", type=float, default=None)
    parser.add_argument(
        "--render-frames",
        default="auto",
        help="Comma-separated Blender frames, or auto for initial/impact/compression/rebound.",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_experiment(experiment, physics_profiles, material_presets):
    model = MODELS / experiment["model"]
    if not model.is_file():
        raise FileNotFoundError(model)
    physics = physics_profiles[experiment["physics_profile"]]
    material = material_presets[experiment["material_preset"]]
    if physics["expected_behavior"] != material["allowed_behavior"]:
        raise ValueError(
            f"Experiment {experiment['id']} pairs {experiment['material_preset']} with "
            f"incompatible behavior {physics['expected_behavior']}"
        )
    return model, physics


def load_scene_context(scene_name, scene_configs):
    if scene_name not in scene_configs:
        raise KeyError(f"Scene is missing from scene_experiments.json: {scene_name}")
    scene_config = scene_configs[scene_name]
    baseline_dir = BASELINE_SCENE_RUNS / scene_name
    baseline_report_path = baseline_dir / "run_complete.json"
    render_report_path = BASELINE_SCENE_RENDERS / scene_name / "blender_render_report.json"
    if scene_name == "hospital" and not baseline_report_path.is_file():
        baseline_dir = ROOT / "output" / "blender_silicone_exact_collision" / "isaac" / "hospital"
        baseline_report_path = baseline_dir / "run_complete.json"
    if not baseline_report_path.is_file():
        raise FileNotFoundError(f"Validated baseline reports are missing for scene: {scene_name}")
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    render_report = (
        json.loads(render_report_path.read_text(encoding="utf-8"))
        if render_report_path.is_file()
        else {}
    )
    ground_patch = baseline.get("ground_patch") or {}
    initial = (baseline.get("keyframes") or {}).get("initial") or {}
    initial_center = initial.get("center") or [0.0, 0.0, 0.0]

    spawn_x = scene_config.get("spawn_x", ground_patch.get("center_x", initial_center[0]))
    spawn_z = scene_config.get("spawn_z", ground_patch.get("center_z", initial_center[2]))
    support_y = scene_config.get("support_top_y", baseline.get("support_top_y"))
    drop_height = scene_config.get("drop_height", initial.get("minimum_y"))
    if support_y is None or drop_height is None:
        raise RuntimeError(f"Scene lacks validated support/drop values: {scene_name}")

    scene_usd = SCENES / scene_name / f"{scene_name}_sim.usda"
    scene_blend = SCENES / f"{scene_name}.blend"
    if not scene_usd.is_file() or not scene_blend.is_file():
        raise FileNotFoundError(f"Scene assets are missing for {scene_name}")

    camera_eye = render_report.get("camera_eye_isaac") or scene_config.get("camera_eye")
    camera_target = render_report.get("camera_target_isaac") or scene_config.get("camera_target")
    if not camera_eye or not camera_target:
        raise RuntimeError(f"Scene lacks a validated camera: {scene_name}")

    return {
        "name": scene_name,
        "config": scene_config,
        "baseline_dir": baseline_dir,
        "usd": scene_usd,
        "blend": scene_blend,
        "support_y": float(support_y),
        "spawn_x": float(spawn_x),
        "spawn_z": float(spawn_z),
        "drop_height": float(drop_height),
        "camera_eye": tuple(camera_eye),
        "camera_target": tuple(camera_target),
        "needs_exact_collision": bool(scene_config.get("local_collision_bounds")),
        "allow_free_fall": bool(scene_config.get("allow_free_fall")),
        "uneven_ground": bool(scene_config.get("uneven_ground")),
        "ground_patch": ground_patch,
    }


def ensure_collision_asset(output_root, scene):
    scene_name = scene["name"]
    # Collision geometry depends only on the source scene and crop bounds, not
    # on a particular experiment output directory. Keep one validated asset per scene.
    collision_asset = ROOT / "output" / "scene_collision_assets" / f"{scene_name}_exact_collision.usdc"
    collision_report = collision_asset.with_suffix(".json")
    bounds = [float(value) for value in scene["config"].get("local_collision_bounds", [])]
    if len(bounds) != 6:
        raise ValueError(f"Scene requires six local collision bounds: {scene_name}")
    source_stage = scene["usd"].resolve()
    environment_prim = scene["config"].get("collision_environment_prim", "/World/Environment")
    if collision_asset.is_file() and collision_asset.stat().st_size > 0 and collision_report.is_file():
        metadata = json.loads(collision_report.read_text(encoding="utf-8"))
        saved_bounds = metadata.get("bounds") or []
        saved_source = str(metadata.get("source", ""))
        saved_source_name = PureWindowsPath(saved_source).name
        if (
            (
                Path(saved_source) == source_stage
                or saved_source_name == source_stage.name
            )
            and len(saved_bounds) == len(bounds)
            and all(math.isclose(float(saved), expected, rel_tol=0.0, abs_tol=1.0e-9)
                    for saved, expected in zip(saved_bounds, bounds))
            and metadata.get("triangles", 0) > 0
            and metadata.get("winding_policy") == "non_vertical_triangles_face_positive_y_v1"
            and metadata.get("environment_prim") == environment_prim
        ):
            return collision_asset
    completed = subprocess.run(
        [
            str(ISAAC_PYTHON), str(COLLISION_EXTRACTOR),
            str(source_stage), str(collision_asset),
            "--bounds", *[str(value) for value in bounds],
            "--environment-prim", environment_prim,
        ],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0 or not collision_asset.is_file() or not collision_report.is_file():
        raise RuntimeError(f"Failed to create reusable collision asset for {scene_name}")
    return collision_asset


def physics_cache_matches(report, args, experiment, model, physics, scene, collision_asset):
    material = report.get("physics_material") or {}
    return all(
        (
            report.get("valid"),
            report.get("physics_frames") == args.frames,
            recorded_path_matches(report.get("environment_usd"), scene["usd"]),
            recorded_path_matches(report.get("prebuilt_collision_usd"), collision_asset)
            if collision_asset is not None else not report.get("prebuilt_collision_usd"),
            recorded_path_matches(material.get("model"), model),
            material.get("model_height") == experiment["model_height"],
            material.get("model_yaw") == experiment.get("model_yaw", 0.0),
            material.get("youngs_modulus") == physics["youngs_modulus"],
            material.get("poissons_ratio") == physics["poissons_ratio"],
            material.get("density") == physics["density"],
            material.get("linear_damping") == physics["linear_damping"],
            material.get("settling_damping") == physics["settling_damping"],
            material.get("restitution") == physics["restitution"],
            material.get("deformable_resolution") == args.deformable_resolution,
            material.get("collision_contact_offset") == physics["collision_contact_offset"],
            material.get("collision_rest_offset") == physics["collision_rest_offset"],
        )
    )


def run_physics(args, experiment, model, physics, scene, physics_dir, collision_asset):
    report_path = physics_dir / "run_complete.json"
    cache_path = physics_dir / "soft_body_blender.usdc"
    if not args.force and report_path.is_file() and cache_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if physics_cache_matches(report, args, experiment, model, physics, scene, collision_asset):
            print(f"[physics-reuse] id={experiment['id']} cache={cache_path}", flush=True)
            return report

    physics_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(ISAAC_PYTHON),
        str(HERO_SCRIPT),
        "--frames", str(args.frames),
        "--substeps", "4",
        "--width", "480",
        "--height", "480",
        "--renderer", "RaytracedLighting",
        "--output", str(physics_dir),
        "--model", str(model),
        "--model-height", str(experiment["model_height"]),
        "--model-scale-mode", "max_extent",
        "--model-yaw", str(experiment.get("model_yaw", 0.0)),
        "--drop-height", str(scene["drop_height"]),
        "--youngs-modulus", str(physics["youngs_modulus"]),
        "--poissons-ratio", str(physics["poissons_ratio"]),
        "--linear-damping", str(physics["linear_damping"]),
        "--settling-damping", str(physics["settling_damping"]),
        "--restitution", str(physics["restitution"]),
        "--density", str(physics["density"]),
        "--expected-behavior", physics["expected_behavior"],
        "--validation-profile", "generic",
        "--deformable-resolution", str(args.deformable_resolution),
        "--collision-contact-offset", str(physics["collision_contact_offset"]),
        "--collision-rest-offset", str(physics["collision_rest_offset"]),
        "--environment-usd", str(scene["usd"]),
        "--environment-ground-only",
        "--support-top-y", str(scene["support_y"]),
        "--spawn-x", str(scene["spawn_x"]),
        "--spawn-z", str(scene["spawn_z"]),
        "--camera-eye", *[str(value) for value in scene["camera_eye"]],
        "--camera-target", *[str(value) for value in scene["camera_target"]],
        "--export-blender-usd",
        "--blender-usd-name", "soft_body_blender.usdc",
        "--skip-preview-render",
    ]
    if collision_asset is not None:
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
    print(
        f"[physics-start] id={experiment['id']} model={model.name} "
        f"youngs={physics['youngs_modulus']}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0 or not report_path.is_file():
        raise RuntimeError(f"Physics failed for {experiment['id']} with code {completed.returncode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise RuntimeError(f"Physics report is invalid for {experiment['id']}: {report.get('error')}")
    return report


def select_render_frames(args, physics_report):
    if args.render_frames.strip().lower() != "auto":
        return args.render_frames
    keyframes = physics_report.get("keyframes") or {}
    selected = [1]
    for name in ("impact", "compression", "rebound"):
        frame = (keyframes.get(name) or {}).get("frame")
        if frame is not None:
            selected.append(max(1, min(args.frames, int(frame) + 1)))
    return ",".join(str(frame) for frame in dict.fromkeys(selected))


def run_render(args, experiment, scene, physics_report, physics_dir, render_dir):
    report_path = render_dir / "blender_render_report.json"
    selected_frames = select_render_frames(args, physics_report)
    if not args.force and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        requested_frames = [int(value) for value in selected_frames.split(",") if value]
        if all(
            (
                report.get("valid"),
                report.get("material_preset") == experiment["material_preset"],
                recorded_path_matches(report.get("source_blend"), scene["blend"]),
                report.get("camera_eye_isaac") == list(scene["camera_eye"]),
                report.get("camera_target_isaac") == list(scene["camera_target"]),
                report.get("selected_frames") == requested_frames,
                report.get("cycles_samples") == args.samples,
                report.get("resolution") == [args.resolution, args.resolution],
            )
        ):
            print(f"[render-reuse] id={experiment['id']} output={render_dir}", flush=True)
            return report

    render_dir.mkdir(parents=True, exist_ok=True)
    for stale in render_dir.glob("preview_*.png"):
        stale.unlink()
    command = [
        str(BLENDER),
        str(scene["blend"]),
        "--background",
        "--python", str(BLENDER_SCRIPT),
        "--",
        str(physics_dir / "soft_body_blender.usdc"),
        str(render_dir),
        str(args.frames),
        str(args.samples),
        str(args.resolution),
        selected_frames,
        "true" if scene["allow_free_fall"] else "false",
        *[str(value) for value in scene["camera_eye"]],
        *[str(value) for value in scene["camera_target"]],
        "false", "", "", "", "0", "0", "0",
        experiment["material_preset"],
    ]
    print(
        f"[render-start] id={experiment['id']} material={experiment['material_preset']}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0 or not report_path.is_file():
        raise RuntimeError(f"Blender render failed for {experiment['id']} with code {completed.returncode}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise RuntimeError(f"Blender report is invalid for {experiment['id']}")
    return report


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene_configs = json.loads(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    selected = set(args.only or [row["id"] for row in config["experiments"]])
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    for experiment in config["experiments"]:
        if experiment["id"] not in selected:
            continue
        scene_name = experiment.get("scene", "hospital")
        scene = load_scene_context(scene_name, scene_configs)
        camera_scale = float(experiment.get("camera_distance_scale", 1.0))
        if camera_scale != 1.0:
            target = scene["camera_target"]
            scene["camera_eye"] = tuple(
                target[index] + camera_scale * (scene["camera_eye"][index] - target[index])
                for index in range(3)
            )
        run_dir = args.output / "runs" / experiment["id"]
        physics_dir = run_dir / "isaac" / scene_name
        render_dir = run_dir / "blender"
        result = {
            "id": experiment["id"],
            "model": experiment["model"],
            "scene": scene_name,
            "physics_profile": experiment["physics_profile"],
            "material_preset": experiment["material_preset"],
            "valid": False,
        }
        try:
            model, physics = validate_experiment(
                experiment, config["physics_profiles"], config["material_presets"]
            )
            physics = dict(physics)
            physics["collision_contact_offset"] = float(
                args.collision_contact_offset
                if args.collision_contact_offset is not None
                else experiment.get("collision_contact_offset", 0.03)
            )
            physics["collision_rest_offset"] = float(
                args.collision_rest_offset
                if args.collision_rest_offset is not None
                else experiment.get("collision_rest_offset", 0.01)
            )
            collision_asset = None
            if args.stage in {"collision", "physics", "all"} and scene["needs_exact_collision"]:
                collision_asset = ensure_collision_asset(args.output, scene)
                result["collision_asset"] = str(collision_asset)
            if args.stage in {"physics", "all"}:
                physics_report = run_physics(
                    args, experiment, model, physics, scene, physics_dir, collision_asset
                )
                result["physics_report"] = str(physics_dir / "run_complete.json")
                result["compression_ratio"] = physics_report.get("compression_ratio")
            if args.stage in {"render", "all"}:
                if args.stage == "render":
                    physics_report = json.loads(
                        (physics_dir / "run_complete.json").read_text(encoding="utf-8")
                    )
                render_report = run_render(
                    args, experiment, scene, physics_report, physics_dir, render_dir
                )
                result["render_report"] = str(render_dir / "blender_render_report.json")
                result["images"] = [str(path) for path in sorted(render_dir.glob("preview_*.png"))]
                result["rendered_material"] = render_report.get("material_preset")
            result["valid"] = True
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            print(f"[experiment-failed] id={experiment['id']} error={result['error']}", flush=True)
        results.append(result)
        (args.output / "summary.json").write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {"valid": bool(results) and all(row["valid"] for row in results), "results": results}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
