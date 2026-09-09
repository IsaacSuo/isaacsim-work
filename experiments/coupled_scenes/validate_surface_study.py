"""Read-only layout checks, requires usd-core (pxr), no SimulationApp."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from coupled_scene.surface_study import motion_state
from pxr import Usd, UsdGeom, UsdPhysics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    specs = sorted(args.input.glob("*/scene_spec.json"))
    assert len(specs) == 5
    for path in specs:
        spec = json.loads(path.read_text(encoding="utf-8"))
        case = spec["case"]
        stage = Usd.Stage.Open(str(path.parent / "colliders.usda"))
        assert stage and UsdGeom.GetStageUpAxis(stage) == "Y"
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1
        colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(colliders) == (6 if case["actuator"] else 5)
        cameras = json.loads((path.parent / "cameras.json").read_text())
        assert len(cameras) == 8
        motion = case["motion"]
        samples = json.loads((path.parent / "motion_samples.json").read_text())
        assert len(samples) == spec["frame_count"]
        assert stage.GetEndTimeCode() == spec["frame_count"]
        for sample in samples:
            position, velocity = motion_state(motion, sample["seconds"])
            assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(position, sample["displacement_isaac_m"]))
            assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(velocity, sample["velocity_isaac_m_s"]))
        if motion:
            target = stage.GetPrimAtPath("/SurfaceStudy/" + motion["target"])
            body = UsdPhysics.RigidBodyAPI(target)
            assert body.GetKinematicEnabledAttr().Get()
            attr = target.GetAttribute("xformOp:translate")
            base = attr.Get(1)
            for sample in samples:
                actual = attr.Get(sample["frame"])
                assert all(math.isclose(actual[i]-base[i], sample["displacement_isaac_m"][i], abs_tol=1e-8) for i in range(3))
            for endpoint in [motion["start_s"], motion["stop_s"]]:
                assert motion_state(motion, endpoint) == ([0.0]*3, [0.0]*3)
            for index in range(1, 100):
                t = motion["start_s"] + index/100*(motion["stop_s"]-motion["start_s"])
                p0, _ = motion_state(motion, t-1e-6)
                p1, _ = motion_state(motion, t+1e-6)
                _, v = motion_state(motion, t)
                assert all(math.isclose((p1[i]-p0[i])/2e-6, v[i], abs_tol=1e-6) for i in range(3))
        print(f'PASS {case["id"]}: USD, colliders, eight cameras, motion samples and analytic velocity', flush=True)
    print('Layout checks only; fluid interaction and Blender animation playback are not validated.')


if __name__ == "__main__":
    main()
