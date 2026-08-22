"""Audit binary STL geometry before PhysX deformable cooking."""

from __future__ import annotations

import argparse
import gc
import json
import struct
from pathlib import Path

import numpy as np


STL_RECORD = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--topology-max-triangles",
        type=int,
        default=500_000,
        help="Run the more expensive watertightness check only below this face count.",
    )
    return parser.parse_args()


def audit_binary_stl(path: Path, topology_max_triangles: int) -> dict:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    expected_size = 84 + triangle_count * STL_RECORD.itemsize
    if file_size != expected_size:
        return {
            "model": path.stem,
            "path": str(path),
            "valid_binary_stl": False,
            "file_size": file_size,
            "expected_size": expected_size,
            "error": "binary STL size does not match header",
        }

    records = np.memmap(
        path,
        dtype=STL_RECORD,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    triangles = records["vertices"]
    minimum = np.min(triangles, axis=(0, 1)).astype(float)
    maximum = np.max(triangles, axis=(0, 1)).astype(float)
    extent = maximum - minimum
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area_twice = np.linalg.norm(cross, axis=1)
    finite = bool(np.all(np.isfinite(triangles)))
    degenerate_count = int(np.count_nonzero(~np.isfinite(area_twice) | (area_twice <= 1.0e-12)))

    result = {
        "model": path.stem,
        "path": str(path),
        "valid_binary_stl": True,
        "file_size": file_size,
        "triangle_count": triangle_count,
        "bounds_min": minimum.tolist(),
        "bounds_max": maximum.tolist(),
        "extent": extent.tolist(),
        "finite": finite,
        "degenerate_triangles": degenerate_count,
        "topology_checked": False,
        "watertight": None,
        "winding_consistent": None,
        "body_count": None,
        "volume": None,
    }
    del triangles, records, cross, area_twice

    if triangle_count <= topology_max_triangles:
        import trimesh

        mesh = trimesh.load_mesh(path, process=True)
        result.update(
            {
                "topology_checked": True,
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "body_count": int(mesh.body_count),
                "volume": float(mesh.volume),
            }
        )
        del mesh
        gc.collect()
    return result


def main():
    args = parse_args()
    paths = sorted(args.directory.glob("*.stl"))
    if not paths:
        raise SystemExit(f"No STL files found in {args.directory}")
    results = []
    for path in paths:
        result = audit_binary_stl(path, args.topology_max_triangles)
        results.append(result)
        print(
            f"[stl-audit] model={result['model']} triangles={result.get('triangle_count')} "
            f"watertight={result.get('watertight')} degenerates={result.get('degenerate_triangles')}",
            flush=True,
        )
    report = {"valid": all(row.get("valid_binary_stl") for row in results), "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[stl-audit] report={args.output}", flush=True)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
