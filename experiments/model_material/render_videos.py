"""Render and encode full videos from completed model/material experiment caches."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from run_experiments import (
    BLENDER,
    BLENDER_SCRIPT,
    ROOT,
    SCENE_CONFIG_PATH,
    configured_path,
    load_scene_context,
    recorded_path_matches,
)


FFMPEG = configured_path(
    "FFMPEG_BIN",
    shutil.which("ffmpeg"),
    Path(r"D:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "model_material_experiments.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "model_material_experiments",
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scaled_camera(scene, experiment):
    target = scene["camera_target"]
    scale = float(experiment.get("camera_distance_scale", 1.0))
    eye = tuple(
        target[index] + scale * (scene["camera_eye"][index] - target[index])
        for index in range(3)
    )
    return eye, target


def video_matches(report_path, video_path, experiment, scene, eye, target, args, frame_count):
    if not report_path.is_file() or not video_path.is_file() or video_path.stat().st_size <= 0:
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return all(
        (
            report.get("valid"),
            report.get("material_preset") == experiment["material_preset"],
            recorded_path_matches(report.get("source_blend"), scene["blend"]),
            report.get("camera_eye_isaac") == list(eye),
            report.get("camera_target_isaac") == list(target),
            report.get("requested_frame_end") == frame_count,
            report.get("cycles_samples") == args.samples,
            report.get("resolution") == [args.resolution, args.resolution],
            report.get("selected_frames") is None,
        )
    )


def write_summary(path, results, complete=False):
    payload = {
        "valid": bool(results) and all(row["valid"] for row in results) if complete else False,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene_configs = json.loads(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    selected = set(args.only or [row["id"] for row in config["experiments"]])
    frames_root = args.output / "video_frames"
    videos_root = args.output / "videos"
    frames_root.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "video_summary.json"
    results = []

    for experiment in config["experiments"]:
        experiment_id = experiment["id"]
        if experiment_id not in selected:
            continue
        scene_name = experiment.get("scene", "hospital")
        scene = load_scene_context(scene_name, scene_configs)
        eye, target = scaled_camera(scene, experiment)
        run_dir = args.output / "runs" / experiment_id
        physics_dir = run_dir / "isaac" / scene_name
        physics_report_path = physics_dir / "run_complete.json"
        cache_path = physics_dir / "soft_body_blender.usdc"
        frames_dir = frames_root / experiment_id
        report_path = frames_dir / "blender_render_report.json"
        video_path = videos_root / f"{experiment_id}.mp4"
        result = {
            "id": experiment_id,
            "scene": scene_name,
            "model": experiment["model"],
            "physics_profile": experiment["physics_profile"],
            "material_preset": experiment["material_preset"],
            "valid": False,
        }
        try:
            if not physics_report_path.is_file() or not cache_path.is_file():
                raise FileNotFoundError(f"Completed physics cache is missing for {experiment_id}")
            physics_report = json.loads(physics_report_path.read_text(encoding="utf-8"))
            if not physics_report.get("valid"):
                raise RuntimeError(f"Physics report is invalid for {experiment_id}")
            frame_count = int(physics_report["physics_frames"])
            if not args.force and video_matches(
                report_path, video_path, experiment, scene, eye, target, args, frame_count
            ):
                render_report = json.loads(report_path.read_text(encoding="utf-8"))
                print(f"[video-reuse] id={experiment_id} video={video_path}", flush=True)
            else:
                frames_dir.mkdir(parents=True, exist_ok=True)
                for stale in frames_dir.glob("frame_*.png"):
                    stale.unlink()
                command = [
                    str(BLENDER),
                    str(scene["blend"]),
                    "--background",
                    "--python", str(BLENDER_SCRIPT),
                    "--",
                    str(cache_path),
                    str(frames_dir),
                    str(frame_count),
                    str(args.samples),
                    str(args.resolution),
                    "all",
                    "true" if scene["allow_free_fall"] or scene["uneven_ground"] else "false",
                    *[str(value) for value in eye],
                    *[str(value) for value in target],
                    "false", "", "", "", "0", "0", "0",
                    experiment["material_preset"],
                ]
                print(
                    f"[video-render-start] id={experiment_id} scene={scene_name} "
                    f"frames={frame_count} material={experiment['material_preset']}",
                    flush=True,
                )
                completed = subprocess.run(command, cwd=ROOT, check=False)
                if completed.returncode != 0 or not report_path.is_file():
                    raise RuntimeError(f"Blender render failed with code {completed.returncode}")
                render_report = json.loads(report_path.read_text(encoding="utf-8"))
                if not render_report.get("valid"):
                    raise RuntimeError("Blender render report is invalid")
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

            rendered_frames = int(render_report["rendered_frame_count"])
            result.update(
                {
                    "valid": True,
                    "physics_frames": frame_count,
                    "rendered_frames": rendered_frames,
                    "duration_seconds": rendered_frames / 60.0,
                    "video": str(video_path),
                    "render_report": str(report_path),
                }
            )
            print(
                f"[video-complete] id={experiment_id} frames={rendered_frames} video={video_path}",
                flush=True,
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            print(f"[video-failed] id={experiment_id} error={result['error']}", flush=True)
        results.append(result)
        write_summary(summary_path, results)

    write_summary(summary_path, results, complete=True)
    return 0 if results and all(row["valid"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
