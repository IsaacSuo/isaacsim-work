from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_formal_scene_entrypoints_exist(self):
        expected = (
            "tools/scenes/run_static_scene_previews.py",
            "tools/scenes/run_static_scene_videos.py",
            "tools/blender/render_blender_soft_body_cache.py",
            "tools/blender/render_blender_selected_videos.py",
        )
        for relative_path in expected:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_all_camera_scenes_have_physics_config(self):
        scenes = json.loads(
            (ROOT / "configs/scene_experiments.json").read_text(encoding="utf-8")
        )
        cameras = json.loads(
            (ROOT / "configs/blender_camera_selections.json").read_text(encoding="utf-8")
        )

        self.assertEqual(set(cameras) - set(scenes), set())

    def test_omniglass_checks_are_not_root_entrypoints(self):
        self.assertEqual(list(ROOT.glob("render_omni*.py")), [])
        self.assertTrue(
            (ROOT / "experiments/omniglass/render_omniglass_preview.py").is_file()
        )

    def test_multi_object_config_covers_all_scenes_with_mixed_materials(self):
        scenes = json.loads(
            (ROOT / "configs/scene_experiments.json").read_text(encoding="utf-8")
        )
        config = json.loads(
            (ROOT / "configs/multi_object_scene_experiments.json").read_text(
                encoding="utf-8"
            )
        )
        experiments = config["experiments"]

        self.assertIsInstance(config.get("seed"), int)
        self.assertEqual(len(experiments), 14)
        self.assertEqual({row["scene"] for row in experiments}, set(scenes))
        self.assertEqual(len({row["scene"] for row in experiments}), len(experiments))
        for experiment in experiments:
            with self.subTest(scene=experiment["scene"]):
                bodies = experiment["bodies"]
                self.assertGreaterEqual(len(bodies), 3)
                self.assertEqual(len({body["model"] for body in bodies}), len(bodies))
                self.assertGreaterEqual(
                    len({body["material_preset"] for body in bodies}), 3
                )
                self.assertEqual(
                    sorted(body["drop_offset"] for body in bodies), [0.0, 0.72, 1.44]
                )


if __name__ == "__main__":
    unittest.main()
