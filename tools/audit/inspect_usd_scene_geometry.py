"""Print world-space bounds and transforms for selected USD prims."""

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom


parser = argparse.ArgumentParser()
parser.add_argument("stage")
parser.add_argument("--contains", default="")
parser.add_argument("--limit", type=int, default=20)
args = parser.parse_args()

stage = Usd.Stage.Open(args.stage)
if stage is None:
    raise SystemExit(f"Could not open {args.stage}")
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
count = 0
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    path = str(prim.GetPath())
    if args.contains.lower() not in path.lower():
        continue
    matrix = cache.GetLocalToWorldTransform(prim)
    box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
    print(
        f"{path}\n"
        f"  translation={[round(float(v), 6) for v in matrix.ExtractTranslation()]}\n"
        f"  min={[round(float(v), 6) for v in box.GetMin()]} "
        f"max={[round(float(v), 6) for v in box.GetMax()]}"
    )
    count += 1
    if count >= args.limit:
        break

app.close()
