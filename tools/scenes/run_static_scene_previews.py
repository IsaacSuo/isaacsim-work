"""Run low-cost soft-body previews for prepared static scene wrappers."""

import argparse
import json
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=r"Y:\scenes\static_scene_manifest.json")
    parser.add_argument("--config", default=str(WORKSPACE / "configs" / "scene_experiments.json"))
    parser.add_argument("--overlay-config", default=None)
    parser.add_argument("--ground-sites", default=r"Y:\isaacsim_work\output\scene_ground_sites.json")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=("apartment",))
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--model-height", type=float, default=1.45)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--renderer",
        choices=("RaytracedLighting", "PathTracing"),
        default=None,
    )
    parser.add_argument("--path-spp", type=int, default=32)
    parser.add_argument("--path-max-bounces", type=int, default=16)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--panorama",
        action="store_true",
        help="Render four static views from world origin without running physics.",
    )
    parser.add_argument(
        "--export-blender-usd",
        action="store_true",
        help="Also export the visible soft-body surface animation for Blender.",
    )
    parser.add_argument(
        "--skip-isaac-render",
        action="store_true",
        help="Skip Isaac preview stills when Blender is the final renderer.",
    )
    return parser.parse_args()


ARGS = parse_args()
ISAAC_PYTHON = Path(r"Y:\isaacsim\python.bat")
HERO_SCRIPT = WORKSPACE / "soft_body_bounce_hero.py"
MODEL = WORKSPACE / "assets" / "soft_body_elephant.stl"
OUTPUT_ROOT = Path(ARGS.output_root) if ARGS.output_root else WORKSPACE / "output" / (
    "scene_panoramas" if ARGS.panorama else "scene_previews"
)
SUMMARY_PATH = OUTPUT_ROOT / ("panorama_summary.json" if ARGS.panorama else "preview_summary.json")


def main():
    manifest = json.loads(Path(ARGS.manifest).read_text(encoding="utf-8"))
    config_path = Path(ARGS.config)
    overrides = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    overlay_path = Path(ARGS.overlay_config) if ARGS.overlay_config else None
    overlay_overrides = (
        json.loads(overlay_path.read_text(encoding="utf-8"))
        if overlay_path and overlay_path.is_file()
        else {}
    )
    ground_sites_path = Path(ARGS.ground_sites)
    ground_sites = {}
    if ground_sites_path.is_file():
        ground_sites = {
            row["scene"]: row for row in json.loads(ground_sites_path.read_text(encoding="utf-8"))["results"]
        }
    selected = set(ARGS.only or [])
    skipped = set(ARGS.skip or [])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    renderer = ARGS.renderer or ("PathTracing" if ARGS.panorama else "RaytracedLighting")
    for row in manifest["results"]:
        name = row["scene"]
        if selected and name not in selected:
            continue
        if name in skipped:
            continue
        if not row.get("valid"):
            results.append({"scene": name, "valid": False, "error": "static audit failed"})
            continue
        experiment = dict(row["experiment"])
        ground_candidate = ground_sites.get(name, {}).get("candidate")
        if ground_candidate:
            y_shift = float(ground_candidate["y"]) - float(experiment["support_top_y"])
            experiment["support_top_y"] = float(ground_candidate["y"])
            experiment["drop_height"] = float(experiment["drop_height"]) + y_shift
            experiment["camera_eye"] = [
                experiment["camera_eye"][0], experiment["camera_eye"][1] + y_shift, experiment["camera_eye"][2]
            ]
            experiment["camera_target"] = [
                experiment["camera_target"][0], experiment["camera_target"][1] + y_shift, experiment["camera_target"][2]
            ]
            experiment["ground_prims"] = [ground_candidate["composed_prim"]]
        experiment.update(overrides.get(name, {}))
        experiment.update(overlay_overrides.get(name, {}))
        output = OUTPUT_ROOT / name
        command = [
            str(ISAAC_PYTHON),
            str(HERO_SCRIPT),
            "--environment-usd",
            row["sim_usd"],
            "--width",
            str(ARGS.width),
            "--height",
            str(ARGS.height),
            "--renderer",
            renderer,
            "--path-spp",
            str(ARGS.path_spp),
            "--path-max-bounces",
            str(ARGS.path_max_bounces),
            "--render-settle",
            "6",
            "--output",
            str(output),
        ]
        if ARGS.panorama:
            command.append("--environment-panorama")
        else:
            command.extend(
                (
                    "--frames", str(ARGS.frames), "--substeps", "4",
                    "--model", str(MODEL), "--model-height", str(ARGS.model_height),
                    "--drop-height", str(experiment["drop_height"]),
                    "--environment-ground-only",
                    "--support-top-y", str(experiment["support_top_y"]),
                    "--spawn-x", str(experiment["spawn_x"]),
                    "--spawn-z", str(experiment["spawn_z"]),
                    "--camera-eye", *[str(value) for value in experiment["camera_eye"]],
                    "--camera-target", *[str(value) for value in experiment["camera_target"]],
                    "--orbit-preview", "--orbit-yaw-degrees",
                    *[str(value) for value in experiment.get("orbit_yaw_degrees", (-35.0, 0.0, 35.0))],
                )
            )
            if experiment.get("ground_patch_size"):
                command.extend(
                    (
                        "--ground-patch-size-x", str(experiment["ground_patch_size"][0]),
                        "--ground-patch-size-z", str(experiment["ground_patch_size"][1]),
                    )
                )
            if experiment.get("ground_patch_center"):
                command.extend(
                    (
                        "--ground-patch-center-x", str(experiment["ground_patch_center"][0]),
                        "--ground-patch-center-z", str(experiment["ground_patch_center"][1]),
                    )
                )
            if experiment.get("allow_free_fall"):
                command.append("--allow-free-fall")
            if experiment.get("local_collision_bounds"):
                command.extend(
                    (
                        "--local-collision-bounds",
                        *[str(value) for value in experiment["local_collision_bounds"]],
                    )
                )
            for ground_prim in experiment.get("ground_prims", []):
                command.extend(("--environment-ground-prim", ground_prim))
            for ground_prim in experiment.get("sanitize_ground_prims", []):
                command.extend(("--sanitize-ground-prim", ground_prim))
            if experiment.get("uneven_ground"):
                command.append("--uneven-ground")
            if ARGS.export_blender_usd:
                command.extend(("--export-blender-usd", "--blender-usd-name", "soft_body_blender.usdc"))
            if ARGS.skip_isaac_render:
                command.append("--skip-preview-render")
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
        print(f"[preview-start] {name} output={output}", flush=True)
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        report_path = output / "run_complete.json"
        report = {}
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        result = {
            "scene": name,
            "returncode": completed.returncode,
            "valid": completed.returncode == 0 and bool(report.get("valid")),
            "output": str(output),
            "error": report.get("error"),
        }
        results.append(result)
        SUMMARY_PATH.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(
            f"[preview-complete] {name} valid={result['valid']} "
            f"returncode={completed.returncode}",
            flush=True,
        )
    summary = {"valid": all(row["valid"] for row in results), "results": results}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[complete] valid={summary['valid']} summary={SUMMARY_PATH}")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
