"""Run the existing soft-body hero and capture its fixed-topology motion.

The hero remains the authoritative scene/physics implementation. This adapter
observes its measured snapshots, writes a cache-only take, and leaves cached
rendering to ``render_fixed_topology_video.py``.
"""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path

from fixed_topology_video import FixedTopologyTakeWriter, job_path, load_job
from liquid_video_cache import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-job", required=True)
    args = parser.parse_args()
    job_dir = Path(args.video_job).resolve()
    job = load_job(job_dir)
    if job["adapter"] != "soft_body_bounce":
        raise ValueError(f"Unsupported adapter job: {job['adapter']}")
    expected_adapter = job["script_sha256"]["soft_body_fixed_topology_adapter.py"]
    if file_sha256(Path(__file__).resolve()) != expected_adapter:
        raise ValueError("Job was initialized for another adapter source version")
    source = Path(__file__).resolve().parent / "soft_body_bounce_hero.py"
    if file_sha256(source) != job["script_sha256"]["soft_body_bounce_hero.py"]:
        raise ValueError("Job was initialized for another soft-body hero source version")

    simulation = job["simulation"]
    params = simulation["parameters"]
    model_path = Path(params["model"])
    if file_sha256(model_path) != params["model_sha256"]:
        raise ValueError(f"Soft-body model hash changed: {model_path}")
    take_dir = job_path(job_dir, job, "take_dir")
    take_dir.mkdir(parents=True, exist_ok=True)
    hero_output = take_dir / "adapter_hero_run"
    physics_steps = int(simulation["physics_steps"])
    hero_argv = [
        str(source),
        "--frames", str(physics_steps),
        "--substeps", str(simulation["substeps"]),
        "--width", "320",
        "--height", "320",
        "--renderer", "RaytracedLighting",
        "--render-settle", "1",
        "--model", str(params["model"]),
        "--model-height", str(params["model_height"]),
        "--model-yaw", str(params["model_yaw"]),
        "--drop-height", str(params["drop_height"]),
        "--youngs-modulus", str(params["youngs_modulus"]),
        "--linear-damping", str(params["linear_damping"]),
        "--poissons-ratio", str(params["poissons_ratio"]),
        "--density", str(params["density"]),
        "--deformable-resolution", str(params["deformable_resolution"]),
        "--self-collision-filter-distance",
        str(params["self_collision_filter_distance"]),
        "--output", str(hero_output),
    ]
    original_argv = sys.argv
    sys.argv = hero_argv
    try:
        namespace = runpy.run_path(str(source), run_name="soft_body_cached_source")
    finally:
        sys.argv = original_argv

    writer = FixedTopologyTakeWriter(job_dir, job)
    captured_source_frames: set[int] = set()
    summaries: list[dict] = []
    stride = int(simulation["capture_stride"])

    def capture_snapshot(snapshot, visual) -> None:
        source_frame = int(snapshot.get("frame", -999))
        if source_frame in captured_source_frames:
            return
        if source_frame == -1:
            output_index, sim_step = 0, 0
        else:
            sim_step = source_frame + 1
            if sim_step % stride:
                return
            output_index = sim_step // stride
        if output_index != len(writer.rows) or output_index >= writer.expected_frames:
            return
        if writer.topology_record is None:
            counts = visual.GetFaceVertexCountsAttr().Get()
            indices = visual.GetFaceVertexIndicesAttr().Get()
            writer.write_topology(counts, indices, len(snapshot["points"]))
        summary = namespace["snapshot_summary"](snapshot)
        writer.write_frame(
            points=snapshot["points"],
            translate=snapshot.get("translate"),
            sim_step=sim_step,
            sim_time_seconds=output_index / int(job["video"]["output_fps"]),
            frame_metadata=summary,
        )
        captured_source_frames.add(source_frame)
        summaries.append(summary)

    original_snapshot = namespace["current_snapshot"]

    def cached_snapshot(stage, visual, root_prim, source_frame):
        snapshot = original_snapshot(stage, visual, root_prim, source_frame)
        capture_snapshot(snapshot, visual)
        return snapshot

    namespace["main"].__globals__["current_snapshot"] = cached_snapshot
    real_simulation_app = namespace["simulation_app"]

    class DeferredClose:
        def __getattr__(self, name):
            return getattr(real_simulation_app, name)

        def close(self):
            print("[fixed-cache] deferring SimulationApp.close()", flush=True)

    namespace["main"].__globals__["simulation_app"] = DeferredClose()
    hero_log = namespace["log_file"]
    try:
        # Behavioral thresholds are experiment metadata here. Keep an expected
        # validation failure in the hero log while independently validating
        # cache structure and finite geometry below.
        sys.stdout = hero_log
        sys.stderr = hero_log
        hero_result = int(namespace["main"]())
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    print(
        f"[fixed-cache] hero_exit={hero_result} captured={len(writer.rows)}/"
        f"{writer.expected_frames}",
        flush=True,
    )

    expected = int(job["video"]["output_frames"])
    if len(writer.rows) != expected:
        raise RuntimeError(f"Adapter captured {len(writer.rows)}/{expected} frames")
    template_source = hero_output / "soft_body_bounce_hero.usda"
    template_target = job_path(job_dir, job, "render_template")
    if not template_source.is_file():
        raise RuntimeError(f"Hero did not export its render template: {template_source}")
    shutil.copy2(template_source, template_target)
    finite = all(bool(row.get("finite")) for row in summaries)
    if not finite:
        raise RuntimeError("Cached soft-body take contains non-finite geometry")
    hero_report_path = hero_output / "run_complete.json"
    hero_report = (
        json.loads(hero_report_path.read_text(encoding="utf-8-sig"))
        if hero_report_path.is_file()
        else {}
    )
    validation = {
        "all_points_finite": finite,
        "hero_exit_code": hero_result,
        "hero_validation_valid": bool(hero_report.get("valid")),
        "hero_validation_error": hero_report.get("error"),
        "minimum_surface_y": min(float(row["minimum_y"]) for row in summaries),
        "maximum_horizontal_radius": max(
            float(row["horizontal_radius"]) for row in summaries
        ),
    }
    completion = writer.finalize(
        render_template=template_target,
        isaac_version=hero_report.get("isaac_sim_version"),
        validation=validation,
    )
    print(
        f"[fixed-cache] complete frames={completion['frame_count']} "
        f"valid={completion['valid']}"
    )
    sys.stdout.flush()
    real_simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
