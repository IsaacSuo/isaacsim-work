from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.model_material.generate_multi_object_config import generate


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_formal_scene_entrypoints_exist(self):
        expected = (
            "tools/scenes/run_static_scene_previews.py",
            "tools/scenes/run_static_scene_videos.py",
            "tools/blender/render_blender_soft_body_cache.py",
            "tools/blender/render_blender_selected_videos.py",
            "tools/blender/audit_simulation_penetration.py",
            "tools/physx/run_penetration_audit.py",
        )
        for relative_path in expected:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_soft_body_core_helpers_are_package_modules(self):
        expected = (
            "soft_body/config.py",
            "soft_body/geometry.py",
            "soft_body/tet_quality.py",
            "soft_body/penetration.py",
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
        collision_policies = config["rigid_collision_policies"]
        deformable_policy = config["deformable_collision_policy"]

        self.assertIsInstance(config.get("seed"), int)
        self.assertEqual(
            set(collision_policies), set(config["simulation_model_pool"])
        )
        self.assertEqual(
            deformable_policy,
            {
                "remeshing_enabled": True,
                "remeshing_resolution": 0,
                "target_triangle_count": 0,
                "force_conforming": True,
            },
        )
        for model, policy in collision_policies.items():
            with self.subTest(collision_model=model):
                self.assertIn(
                    policy["approximation"], {"convexDecomposition", "sdf"}
                )
                if policy["approximation"] == "sdf":
                    self.assertGreater(policy["sdf_resolution"], 1)
        self.assertEqual(len(experiments), 14)
        self.assertEqual({row["scene"] for row in experiments}, set(scenes))
        self.assertEqual(len({row["scene"] for row in experiments}), len(experiments))
        garage = next(row for row in experiments if row["scene"] == "garage")
        self.assertEqual(garage["physics_substeps"], 8)
        self.assertEqual(garage["deformable_resolution"], 12)
        mountain = next(row for row in experiments if row["scene"] == "mountain")
        self.assertEqual(mountain["physics_substeps"], 8)
        self.assertEqual(mountain["deformable_resolution"], 16)
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

    def test_checked_in_multi_object_config_is_reproducible(self):
        config = json.loads(
            (ROOT / "configs/multi_object_scene_experiments.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config, generate(config["seed"]))

    def test_candidate_environment_is_versioned(self):
        environment = json.loads(
            (ROOT / "configs/production_environment.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(environment["baseline_status"], "pre_m0_candidate")
        self.assertTrue(environment["isaac_sim"]["version"].startswith("6.0.1"))
        self.assertEqual(environment["blender"]["version"], "5.0.1")
        self.assertEqual(environment["timing"]["physics_frames"], 300)
        self.assertEqual(environment["timing"]["animation_cache_fps"], 60)

    def test_multi_object_runner_exposes_explicit_deformable_audit_flags(self):
        source = (
            ROOT / "experiments/model_material/run_multi_object_videos.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--debug-deformable-frame"', source)
        self.assertIn('"--audit-tet-trajectory"', source)
        self.assertIn('report.get("deformable_collision_debug")', source)
        self.assertIn('report.get("tet_deformation_trajectory")', source)

    def test_multi_object_runner_exposes_deformable_solver_iterations(self):
        source = (
            ROOT / "experiments/model_material/run_multi_object_videos.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--deformable-solver-position-iterations"', source)
        self.assertIn('report.get("physics_material")', source)
        self.assertIn('get("solver_position_iterations")', source)
        self.assertIn('body_config["deformable_solver_position_iterations"]', source)


if __name__ == "__main__":
    unittest.main()
