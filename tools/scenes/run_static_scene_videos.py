"""Render resumable 5-second PathTracing soft-body videos for prepared static scenes."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=r"Y:\scenes\static_scene_manifest.json")
    parser.add_argument("--config", default=str(WORKSPACE / "configs" / "scene_experiments.json"))
    parser.add_argument("--ground-sites", default=r"Y:\isaacsim_work\output\scene_ground_sites.json")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=("apartment",))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--path-spp", type=int, default=32)
    parser.add_argument("--path-max-bounces", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
ISAAC_PYTHON = Path(r"Y:\isaacsim\python.bat")
HERO_SCRIPT = WORKSPACE / "soft_body_bounce_hero.py"
MODEL = WORKSPACE / "assets" / "soft_body_elephant.stl"
OUTPUT_ROOT = WORKSPACE / "output"
SUMMARY_PATH = OUTPUT_ROOT / "scene_video_summary.json"


def resolved_experiment(row, overrides, ground_sites):
    experiment = dict(row["experiment"])
    ground_candidate = ground_sites.get(row["scene"], {}).get("candidate")
    if ground_candidate:
        shift = float(ground_candidate["y"]) - float(experiment["support_top_y"])
        experiment["support_top_y"] = float(ground_candidate["y"])
        experiment["drop_height"] = float(experiment["drop_height"]) + shift
        experiment["camera_eye"] = [
            experiment["camera_eye"][0], experiment["camera_eye"][1] + shift, experiment["camera_eye"][2]
        ]
        experiment["camera_target"] = [
            experiment["camera_target"][0],
            experiment["camera_target"][1] + shift,
            experiment["camera_target"][2],
        ]
        experiment["ground_prims"] = [ground_candidate["composed_prim"]]
    experiment.update(overrides.get(row["scene"], {}))
    return experiment


def write_summary(results):
    payload = {"valid": bool(results) and all(row["valid"] for row in results), "results": results}
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    if not ISAAC_PYTHON.is_file() or not HERO_SCRIPT.is_file() or not MODEL.is_file():
        raise FileNotFoundError("Isaac Python, hero script, or soft-body model is missing")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is not on PATH")
    manifest = json.loads(Path(ARGS.manifest).read_text(encoding="utf-8"))
    config_path = Path(ARGS.config)
    overrides = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    sites_payload = json.loads(Path(ARGS.ground_sites).read_text(encoding="utf-8"))
    ground_sites = {row["scene"]: row for row in sites_payload["results"]}
    selected = set(ARGS.only or [])
    skipped = set(ARGS.skip or [])
    results = []
    for row in manifest["results"]:
        name = row["scene"]
        if (selected and name not in selected) or name in skipped:
            continue
        output = OUTPUT_ROOT / f"soft_body_{name}_5s"
        frame_dir = output / "video_frames"
        video_path = output / f"soft_body_{name}_5s.mp4"
        report_path = output / "run_complete.json"
        existing_frames = len(list(frame_dir.glob("frame_*.png"))) if frame_dir.is_dir() else 0
        if not ARGS.force and video_path.is_file() and video_path.stat().st_size > 0 and existing_frames == ARGS.frames:
            result = {"scene": name, "valid": True, "status": "existing", "video": str(video_path)}
            results.append(result)
            write_summary(results)
            print(f"[video-skip] {name} existing={video_path}", flush=True)
            continue
        experiment = resolved_experiment(row, overrides, ground_sites)
        output.mkdir(parents=True, exist_ok=True)
        command = [
            str(ISAAC_PYTHON), str(HERO_SCRIPT),
            "--frames", str(ARGS.frames), "--substeps", "4",
            "--model", str(MODEL), "--model-height", "1.45",
            "--drop-height", str(experiment["drop_height"]),
            "--environment-usd", row["sim_usd"], "--environment-ground-only",
            "--support-top-y", str(experiment["support_top_y"]),
            "--spawn-x", str(experiment["spawn_x"]), "--spawn-z", str(experiment["spawn_z"]),
            "--camera-eye", *[str(value) for value in experiment["camera_eye"]],
            "--camera-target", *[str(value) for value in experiment["camera_target"]],
            "--camera-focal-length", "42",
            "--width", str(ARGS.width), "--height", str(ARGS.height),
            "--renderer", "PathTracing", "--path-spp", str(ARGS.path_spp),
            "--path-max-bounces", str(ARGS.path_max_bounces), "--render-settle", "32",
            "--render-video-frames", "--video-frames-dir", "video_frames", "--output", str(output),
        ]
        for ground_prim in experiment.get("ground_prims", []):
            command.extend(("--environment-ground-prim", ground_prim))
        for ground_prim in experiment.get("sanitize_ground_prims", []):
            command.extend(("--sanitize-ground-prim", ground_prim))
        if experiment.get("local_collision_bounds"):
            command.extend(
                (
                    "--local-collision-bounds",
                    *[str(value) for value in experiment["local_collision_bounds"]],
                )
            )
        if experiment.get("uneven_ground"):
            command.append("--uneven-ground")
        if experiment.get("emissive_mesh_lights"):
            command.append("--emissive-mesh-lights")
            command.extend(("--emissive-light-intensity", str(experiment.get("emissive_light_intensity", 4200.0))))
            command.extend(("--emissive-light-radius", str(experiment.get("emissive_light_radius", 100.0))))
            command.extend(("--emissive-light-max-count", str(experiment.get("emissive_light_max_count", 100))))
            command.extend(
                (
                    "--emissive-light-max-thickness",
                    str(experiment.get("emissive_light_max_thickness", 0.03)),
                )
            )
        for light in experiment.get("authored_downlights", []):
            command.extend(
                (
                    "--authored-downlight",
                    light["name"],
                    *[str(value) for value in light["position"]],
                    str(light["intensity"]),
                    str(light["size"][0]),
                    str(light["size"][1]),
                    *[str(value) for value in light["color"]],
                )
            )
        if experiment.get("authored_light_intensity_scale") is not None:
            command.extend(
                (
                    "--authored-light-intensity-scale",
                    str(experiment["authored_light_intensity_scale"]),
                )
            )
        if experiment.get("ambient_light_intensity") is not None:
            command.extend(("--ambient-light-intensity", str(experiment["ambient_light_intensity"])))
        if experiment.get("environment_fill_intensity") is not None:
            command.extend(("--environment-fill-intensity", str(experiment["environment_fill_intensity"])))
        if experiment.get("disable_studio_lights"):
            command.append("--disable-studio-lights")
        if experiment.get("exposure") is not None:
            command.extend(("--exposure", str(experiment["exposure"])))
        if experiment.get("hdri_texture"):
            command.extend(("--hdri-texture", str(experiment["hdri_texture"])))
        if experiment.get("hdri_intensity") is not None:
            command.extend(("--hdri-intensity", str(experiment["hdri_intensity"])))
        nonmetal_materials = experiment.get("force_nonmetal_materials", [])
        if nonmetal_materials:
            command.extend(("--force-nonmetal-material-list", ",".join(nonmetal_materials)))
        for override in experiment.get("material_input_scales", []):
            command.extend(
                (
                    "--material-input-scale",
                    override["material"],
                    override["input"],
                    str(override["factor"]),
                )
            )
        for override in experiment.get("material_color_scales", []):
            command.extend(
                (
                    "--material-color-scale",
                    override["material"],
                    str(override["rgb"][0]),
                    str(override["rgb"][1]),
                    str(override["rgb"][2]),
                )
            )
        for override in experiment.get("material_opacity_thresholds", []):
            command.extend(
                (
                    "--material-opacity-threshold",
                    override["material"],
                    str(override["threshold"]),
                )
            )
        for override in experiment.get("material_emission_textures", []):
            command.extend(
                (
                    "--material-emission-texture",
                    override["material"],
                    override["texture"],
                )
            )
        for override in experiment.get("material_emission_scales", []):
            command.extend(
                (
                    "--material-emission-scale",
                    override["material"],
                    str(override["factor"]),
                )
            )
        for override in experiment.get("material_omniglass", []):
            command.extend(
                (
                    "--material-omniglass",
                    override["material"],
                    str(override["color"][0]),
                    str(override["color"][1]),
                    str(override["color"][2]),
                    str(override["ior"]),
                    str(override["roughness"]),
                    str(bool(override["thin_walled"])),
                    str(override.get("depth", 0.001)),
                    str(bool(override.get("roughness_texture", False))),
                )
            )
        for override in experiment.get("material_omnipbr_overrides", []):
            command.extend(
                (
                    "--material-omnipbr",
                    override["material"],
                    override["texture"],
                    str(override["tint"][0]),
                    str(override["tint"][1]),
                    str(override["tint"][2]),
                    str(override["metallic"]),
                    str(override["roughness"]),
                    str(override.get("specular", 0.5)),
                    str(override["opacity"]),
                )
            )
        for override in experiment.get("material_input_values", []):
            command.extend(
                (
                    "--material-input-value",
                    override["material"],
                    override["input"],
                    str(override["value"]),
                )
            )
        for material_path in experiment.get("hide_material_meshes", []):
            command.extend(("--hide-material-meshes", material_path))
        print(f"[video-start] {name} output={output}", flush=True)
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        frame_count = len(list(frame_dir.glob("frame_*.png"))) if frame_dir.is_dir() else 0
        valid = completed.returncode == 0 and bool(report.get("valid")) and frame_count == ARGS.frames
        error = report.get("error")
        if valid:
            encode = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-framerate", "60", "-start_number", "0",
                 "-i", str(frame_dir / "frame_%06d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)],
                cwd=WORKSPACE, check=False,
            )
            valid = encode.returncode == 0 and video_path.is_file() and video_path.stat().st_size > 0
            if not valid:
                error = f"ffmpeg failed with return code {encode.returncode}"
        result = {
            "scene": name, "valid": valid, "status": "rendered" if valid else "failed",
            "returncode": completed.returncode, "frame_count": frame_count,
            "video": str(video_path), "error": error,
        }
        results.append(result)
        write_summary(results)
        print(f"[video-complete] {name} valid={valid} frames={frame_count}", flush=True)
        if not valid:
            break
    write_summary(results)
    return 0 if results and all(row["valid"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
