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

    def test_camera_scenes_have_physics_config_or_explicit_exception(self):
        scenes = json.loads(
            (ROOT / "configs/scene_experiments.json").read_text(encoding="utf-8")
        )
        cameras = json.loads(
            (ROOT / "configs/blender_camera_selections.json").read_text(encoding="utf-8")
        )

        self.assertEqual(set(cameras) - set(scenes), {"apartment"})

    def test_omniglass_checks_are_not_root_entrypoints(self):
        self.assertEqual(list(ROOT.glob("render_omni*.py")), [])
        self.assertTrue(
            (ROOT / "experiments/omniglass/render_omniglass_preview.py").is_file()
        )


if __name__ == "__main__":
    unittest.main()
