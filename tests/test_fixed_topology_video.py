from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fixed_topology_video import (
    FixedTopologyTakeWriter,
    immutable_job_config,
    load_frame,
    load_topology,
    simulation_provenance_payload,
    validate_job,
)
from liquid_video_cache import config_hash


def make_job() -> dict:
    take_id = "take_test"
    paths = {
        "take_dir": f"takes/{take_id}",
        "cache_dir": f"takes/{take_id}/cache",
        "topology": f"takes/{take_id}/topology.npz",
        "simulation_manifest": f"takes/{take_id}/simulation_manifest.jsonl",
        "simulation_complete": f"takes/{take_id}/simulation_complete.json",
        "render_template": f"takes/{take_id}/render_template.usda",
        "render_dir": "renders/test",
        "frames_dir": "renders/test/frames",
        "render_segments_dir": "renders/test/segments",
        "render_manifest": "renders/test/render_manifest.jsonl",
        "render_complete": "renders/test/render_complete.json",
        "video_dir": "video",
        "reports_dir": "reports",
        "logs_dir": "logs",
    }
    scripts = {
        name: "a" * 64
        for name in (
            "soft_body_fixed_topology_adapter.py",
            "soft_body_bounce_hero.py",
            "fixed_topology_video.py",
            "fixed_topology_video_pipeline.py",
            "render_fixed_topology_video.py",
            "encode_fixed_topology_video.py",
            "run_fixed_topology_video_stage.ps1",
            "run_fixed_topology_video.bat",
        )
    }
    job = {
        "schema_version": 1,
        "job_type": "fixed_topology_mesh_video",
        "adapter": "soft_body_bounce",
        "job_id": "test",
        "take_id": take_id,
        "config_hash": "pending",
        "simulation_provenance_hash": "pending",
        "video": {
            "duration_seconds": 0.1,
            "output_fps": 30,
            "output_frames": 3,
            "width": 64,
            "height": 64,
        },
        "simulation": {
            "script": "soft_body_fixed_topology_adapter.py",
            "cache_schema": "fixed_topology_points_v1",
            "physics_fps": 60,
            "capture_stride": 2,
            "substeps": 2,
            "physics_steps": 4,
            "dynamic_root_prim": "/World/Body",
            "visual_mesh_prim": "/World/Body/Visual",
            "camera_prim": "/World/Camera",
            "parameters": {},
        },
        "render": {
            "renderer": "PathTracing",
            "path_spp": 1,
            "segment_frames": 2,
            "profile": "test",
            "camera": "template-camera",
        },
        "paths": paths,
        "script_sha256": scripts,
    }
    job["config_hash"] = config_hash(immutable_job_config(job))
    job["simulation_provenance_hash"] = config_hash(simulation_provenance_payload(job))
    return validate_job(job)


class FixedTopologyVideoTests(unittest.TestCase):
    def test_round_trip_writer(self) -> None:
        job = make_job()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = FixedTopologyTakeWriter(root, job)
            writer.write_topology([3], [0, 1, 2], 3)
            base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            for index in range(3):
                writer.write_frame(
                    points=base + index,
                    translate=(index, 0, 0),
                    sim_step=index * 2,
                    sim_time_seconds=index / 30,
                )
            template = root / job["paths"]["render_template"]
            template.parent.mkdir(parents=True, exist_ok=True)
            template.write_text("#usda 1.0\n", encoding="utf-8")
            completion = writer.finalize(
                render_template=template,
                isaac_version="test",
                validation={"all_points_finite": True},
            )
            self.assertTrue(completion["valid"])
            topology = load_topology(root / job["paths"]["topology"])
            self.assertEqual(topology["metadata"]["vertex_count"], 3)
            frame = load_frame(
                root / job["paths"]["cache_dir"] / "frame_000002.npz",
                expected_vertex_count=3,
            )
            np.testing.assert_allclose(frame["points"], base + 2)
            np.testing.assert_allclose(frame["translate"], [2, 0, 0])

    def test_rejects_off_cadence_frame(self) -> None:
        job = make_job()
        with tempfile.TemporaryDirectory() as temporary:
            writer = FixedTopologyTakeWriter(temporary, job)
            writer.write_topology([3], [0, 1, 2], 3)
            points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            writer.write_frame(points=points, translate=None, sim_step=0, sim_time_seconds=0.0)
            with self.assertRaises(ValueError):
                writer.write_frame(points=points, translate=None, sim_step=1, sim_time_seconds=1 / 60)


if __name__ == "__main__":
    unittest.main()
