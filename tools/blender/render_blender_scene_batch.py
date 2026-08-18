"""Render one Cycles validation frame per exported Isaac soft-body scene cache."""

import argparse
import json
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
BLENDER = Path(r"D:\Program Files (x86)\Blender\blender.exe")
RENDER_SCRIPT = Path(__file__).with_name("render_blender_soft_body_cache.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=r"Y:\scenes\static_scene_manifest.json")
    parser.add_argument("--config", default=str(WORKSPACE / "configs" / "scene_experiments.json"))
    parser.add_argument("--ground-sites", default=str(WORKSPACE / "output" / "scene_ground_sites.json"))
    parser.add_argument("--cache-root", default=str(WORKSPACE / "output" / "blender_other_scenes" / "isaac"))
    parser.add_argument("--output-root", default=str(WORKSPACE / "output" / "blender_other_scenes" / "cycles"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=("hospital",))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def effective_experiment(row, overrides, ground_sites):
    experiment = dict(row["experiment"])
    ground_candidate = ground_sites.get(row["scene"], {}).get("candidate")
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
    experiment.update(overrides.get(row["scene"], {}))
    return experiment


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ground_sites_path = Path(args.ground_sites)
    ground_sites = {}
    if ground_sites_path.is_file():
        ground_sites = {
            result["scene"]: result
            for result in json.loads(ground_sites_path.read_text(encoding="utf-8"))["results"]
        }
    cache_root = Path(args.cache_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or [])
    skipped = set(args.skip or [])
    results = []

    for row in manifest["results"]:
        scene_name = row["scene"]
        if selected and scene_name not in selected:
            continue
        if scene_name in skipped:
            continue
        blend_path = Path(r"Y:\scenes") / f"{scene_name}.blend"
        cache_path = cache_root / scene_name / "soft_body_blender.usdc"
        report_path = cache_root / scene_name / "run_complete.json"
        output_dir = output_root / scene_name
        if not (blend_path.is_file() and cache_path.is_file() and report_path.is_file()):
            results.append({"scene": scene_name, "valid": False, "error": "missing blend, cache, or report"})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        experiment = effective_experiment(row, overrides, ground_sites)
        render_frame = int(report["keyframes"]["compression"]["frame"]) + 1
        frame_count = int(report["physics_frames"])
        command = [
            str(BLENDER), str(blend_path), "--background", "--python", str(RENDER_SCRIPT), "--",
            str(cache_path), str(output_dir), str(frame_count), str(args.samples), str(args.resolution),
            str(render_frame), "false",
            *[str(value) for value in experiment["camera_eye"]],
            *[str(value) for value in experiment["camera_target"]], "auto",
        ]
        print(f"[blender-start] scene={scene_name} frame={render_frame}", flush=True)
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        blender_report = output_dir / "blender_render_report.json"
        valid = completed.returncode == 0 and blender_report.is_file()
        results.append(
            {
                "scene": scene_name,
                "valid": valid,
                "returncode": completed.returncode,
                "frame": render_frame,
                "output": str(output_dir / f"preview_{render_frame:04d}_view0.png"),
            }
        )
        (output_root / "batch_render_summary.json").write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
        print(f"[blender-complete] scene={scene_name} valid={valid}", flush=True)

    summary = {"valid": all(result["valid"] for result in results), "results": results}
    (output_root / "batch_render_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
