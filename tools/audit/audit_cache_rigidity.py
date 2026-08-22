"""Measure non-rigid deformation in fixed-topology USD point caches.

Translation and rotation are removed with a Kabsch alignment before reporting
RMS vertex displacement as a fraction of the model's initial bounding-box
diagonal. Run this script with Isaac Sim's Python interpreter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--frames", default="22,30,45")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    app = SimulationApp({"headless": True})
    try:
        import numpy as np
        from pxr import Usd, UsdGeom

        for cache_path in args.caches:
            stage = Usd.Stage.Open(str(cache_path.resolve()))
            if stage is None:
                raise RuntimeError(f"Could not open USD cache: {cache_path}")
            meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
            if not meshes:
                raise RuntimeError(f"No meshes in {cache_path}")

            for mesh_prim in meshes:
                points_attr = UsdGeom.Mesh(mesh_prim).GetPointsAttr()
                initial = np.asarray(points_attr.Get(1), dtype=np.float64)
                centered_initial = initial - initial.mean(axis=0)
                diagonal = np.linalg.norm(initial.max(axis=0) - initial.min(axis=0))
                residuals = {}

                for frame in frames:
                    points = np.asarray(points_attr.Get(frame), dtype=np.float64)
                    centered = points - points.mean(axis=0)
                    left, _, right = np.linalg.svd(centered.T @ centered_initial)
                    handedness = np.linalg.det(left @ right)
                    rotation = left @ np.diag([1.0, 1.0, handedness]) @ right
                    aligned = centered @ rotation
                    rms = np.sqrt(
                        np.mean(np.sum((aligned - centered_initial) ** 2, axis=1))
                    )
                    residuals[str(frame)] = round(float(rms / diagonal), 6)

                print(f"{cache_path}:{mesh_prim.GetPath()} {residuals}")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
