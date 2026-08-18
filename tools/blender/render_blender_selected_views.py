"""Render the visually selected final Cycles view for every prepared scene."""

import argparse
import json
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
BLENDER = Path(r"D:\Program Files (x86)\Blender\blender.exe")
RENDER_SCRIPT = Path(__file__).with_name("render_blender_soft_body_cache.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", default=str(WORKSPACE / "configs" / "blender_camera_selections.json"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument(
        "--output-root",
        default=str(WORKSPACE / "output" / "blender_other_scenes" / "cycles_selected_final"),
    )
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    selections = json.loads(Path(args.selections).read_text(encoding="utf-8"))
    selected_names = set(args.only or selections)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []

    for scene_name, selection in selections.items():
        if scene_name not in selected_names:
            continue
        blend_path = Path(r"Y:\scenes") / f"{scene_name}.blend"
        if scene_name == "hospital":
            cache_path = WORKSPACE / "output" / "blender_silicone_exact_collision" / "isaac" / "hospital" / "soft_body_blender.usdc"
            frame_count = 300
            frame = int(selection["frame"])
            eye = selection["eye_isaac"]
            target = selection["target_isaac"]
            camera_args = [*[str(value) for value in eye], *[str(value) for value in target]]
        else:
            cache_dir = WORKSPACE / "output" / "blender_other_scenes" / "isaac" / scene_name
            cache_path = cache_dir / "soft_body_blender.usdc"
            physics_report = json.loads((cache_dir / "run_complete.json").read_text(encoding="utf-8"))
            frame_count = int(physics_report["physics_frames"])
            frame = int(physics_report["keyframes"]["compression"]["frame"]) + 1
            camera_args = [
                "0", "0", "0", "0", "0", "0", "auto", "", "",
                str(selection["azimuth_degrees"]),
            ]

        output_dir = output_root / scene_name
        command = [
            str(BLENDER), str(blend_path), "--background", "--python", str(RENDER_SCRIPT), "--",
            str(cache_path), str(output_dir), str(frame_count), str(args.samples), str(args.resolution),
            str(frame), "false", *camera_args,
        ]
        print(f"[selected-start] scene={scene_name} frame={frame}", flush=True)
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        report_path = output_dir / "blender_render_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        rendered = sorted(output_dir.glob("preview_*.png"))
        valid = completed.returncode == 0 and bool(report.get("valid")) and len(rendered) == 1
        results.append(
            {
                "scene": scene_name,
                "valid": valid,
                "frame": frame,
                "azimuth_degrees": selection.get("azimuth_degrees"),
                "reason": selection["reason"],
                "image": str(rendered[0]) if rendered else None,
            }
        )
        (output_root / "selected_render_summary.json").write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[selected-complete] scene={scene_name} valid={valid}", flush=True)

    summary = {"valid": all(row["valid"] for row in results), "results": results}
    (output_root / "selected_render_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
