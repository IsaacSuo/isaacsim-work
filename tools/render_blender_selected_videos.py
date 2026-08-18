"""Render and encode one fixed-camera Cycles video per selected scene."""

import argparse
import json
import subprocess
from pathlib import Path


WORKSPACE = Path(r"Y:\isaacsim_work")
BLENDER = Path(r"D:\Program Files (x86)\Blender\blender.exe")
FFMPEG = Path(r"D:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe")
RENDER_SCRIPT = WORKSPACE / "tools" / "render_blender_soft_body_cache.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", default=str(WORKSPACE / "blender_camera_selections.json"))
    parser.add_argument("--cache-root", default=str(WORKSPACE / "output" / "blender_scene_videos" / "isaac"))
    parser.add_argument("--output-root", default=str(WORKSPACE / "output" / "blender_scene_videos"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--only", nargs="*", default=None)
    return parser.parse_args()


def blender_to_isaac(values):
    x, y, z = (float(value) for value in values)
    return [x, z, -y]


def main():
    args = parse_args()
    selections = json.loads(Path(args.selections).read_text(encoding="utf-8"))
    selected_names = set(args.only or selections)
    cache_root = Path(args.cache_root)
    output_root = Path(args.output_root)
    frames_root = output_root / "cycles_frames"
    videos_root = output_root / "videos"
    reports_root = output_root / "reports"
    for directory in (frames_root, videos_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)
    results = []

    for scene_name, selection in selections.items():
        if scene_name not in selected_names:
            continue
        blend_path = Path(r"Y:\scenes") / f"{scene_name}.blend"
        if scene_name == "hospital":
            cache_path = WORKSPACE / "output" / "blender_silicone_exact_collision" / "isaac" / "hospital" / "soft_body_blender.usdc"
            frame_count = 300
            eye_isaac = selection["eye_isaac"]
            target_isaac = selection["target_isaac"]
            stop_when_out = True
        else:
            cache_dir = cache_root / scene_name
            cache_path = cache_dir / "soft_body_blender.usdc"
            physics_report_path = cache_dir / "run_complete.json"
            if not physics_report_path.is_file():
                print(f"[video-skip] scene={scene_name} missing={physics_report_path}", flush=True)
                results.append({"scene": scene_name, "valid": False, "error": "missing physics report"})
                continue
            physics_report = json.loads(physics_report_path.read_text(encoding="utf-8"))
            frame_count = int(physics_report["physics_frames"])
            if selection.get("manual_camera"):
                eye_isaac = selection["eye_isaac"]
                target_isaac = selection["target_isaac"]
            else:
                camera_report_path = (
                    WORKSPACE / "output" / "blender_other_scenes" / "cycles_selected_final" /
                    scene_name / "blender_render_report.json"
                )
                camera_report = json.loads(camera_report_path.read_text(encoding="utf-8"))
                candidate = camera_report["camera_candidates"][0]
                eye_isaac = blender_to_isaac(candidate["eye_blender"])
                target_isaac = blender_to_isaac(candidate["target_blender"])
            stop_when_out = scene_name == "mountain"

        frames_dir = frames_root / scene_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink()
        command = [
            str(BLENDER), str(blend_path), "--background", "--python", str(RENDER_SCRIPT), "--",
            str(cache_path), str(frames_dir), str(frame_count), str(args.samples), str(args.resolution),
            "all", "true" if stop_when_out else "false",
            *[str(value) for value in eye_isaac], *[str(value) for value in target_isaac],
        ]
        print(
            f"[video-render-start] scene={scene_name} frames={frame_count} "
            f"samples={args.samples} resolution={args.resolution}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=WORKSPACE, check=False)
        render_report_path = frames_dir / "blender_render_report.json"
        if completed.returncode != 0 or not render_report_path.is_file():
            results.append(
                {"scene": scene_name, "valid": False, "error": "Blender render failed", "returncode": completed.returncode}
            )
            (reports_root / "video_render_summary.json").write_text(
                json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            continue
        render_report = json.loads(render_report_path.read_text(encoding="utf-8"))
        rendered_count = int(render_report["rendered_frame_count"])
        video_path = videos_root / f"{scene_name}_soft_body.mp4"
        encode = subprocess.run(
            [
                str(FFMPEG), "-y", "-framerate", "60", "-start_number", "1",
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
            ],
            cwd=WORKSPACE,
            check=False,
        )
        valid = encode.returncode == 0 and video_path.is_file() and video_path.stat().st_size > 0
        results.append(
            {
                "scene": scene_name,
                "valid": valid,
                "physics_frames": frame_count,
                "rendered_frames": rendered_count,
                "duration_seconds": rendered_count / 60.0,
                "video": str(video_path),
                "render_report": str(render_report_path),
            }
        )
        (reports_root / "video_render_summary.json").write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[video-complete] scene={scene_name} valid={valid} "
            f"rendered_frames={rendered_count} video={video_path}",
            flush=True,
        )

    summary = {"valid": all(row["valid"] for row in results), "results": results}
    (reports_root / "video_render_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
