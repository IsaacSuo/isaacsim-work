"""Audit PhysX/FoamGenerator-to-Plateau binding and conservation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    manifest_path = directory / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    require(manifest.get("schema") == "physx-foamgenerator-plateau/v1", "Wrong schema")
    require(manifest.get("complete") is True, "Manifest is incomplete")
    maximum_gas_residual = 0.0
    maximum_geometry_residual = 0.0
    total_cells = 0
    total_films = 0
    total_unbound = 0
    for sample in manifest["samples"]:
        output_path = directory / sample["file"]
        source_path = Path(sample["source_cache"])
        surface_path = Path(sample["surface"])
        require(output_path.is_file(), f"Missing output: {output_path}")
        require(sha256_file(output_path) == sample["sha256"], f"Output hash mismatch: {output_path}")
        require(source_path.is_file(), f"Missing source: {source_path}")
        require(sha256_file(source_path) == sample["source_cache_sha256"], f"Source hash mismatch: {source_path}")
        require(surface_path.is_file(), f"Missing surface: {surface_path}")
        require(sha256_file(surface_path) == sample["surface_sha256"], f"Surface hash mismatch: {surface_path}")
        with np.load(source_path, allow_pickle=False) as source, np.load(output_path, allow_pickle=False) as result:
            source_ids = np.asarray(source["id"], dtype=np.uint64)
            source_opacity = np.asarray(source["opacity"], dtype=np.float64)
            raft = np.asarray(result["raft"])
            unbound_ids = np.asarray(result["unbound_id"], dtype=np.uint64)
            cells = np.asarray(result["cells"])
            films = np.asarray(result["films"])
            borders = np.asarray(result["borders"])
            nodes = np.asarray(result["nodes"])
            vertices = np.asarray(result["vertices"])
            triangles = np.asarray(result["triangles"])
            materials = np.asarray(result["triangle_material"])
            active_ids = np.sort(source_ids[source_opacity >= manifest["configuration"]["minimum_opacity"]])
            partition_ids = np.sort(np.concatenate((raft["marker_id"], unbound_ids)))
            require(np.array_equal(active_ids, partition_ids), "Bound/unbound IDs do not partition active source IDs")
            require(len(np.unique(raft["marker_id"])) == len(raft), "Duplicate raft stable IDs")
            require(np.all(raft["representative_count"] == 1.0), "Representative count is not one")
            computed_volume = (4.0 / 3.0) * np.pi * raft["physical_radius"].astype(np.float64) ** 3
            require(np.allclose(raft["phase_volume"], computed_volume, rtol=4.0e-7, atol=1.0e-18), "Raft volume/radius mismatch")
            require(np.max(raft["anchor_displacement"], initial=0.0) <= manifest["configuration"]["maximum_bind_distance_m"] + 1.0e-8, "Bound anchor exceeds threshold")
            require(
                len(cells) == len(raft)
                and len(np.unique(cells["parent_marker_id"])) == len(cells)
                and np.array_equal(
                    np.sort(cells["parent_marker_id"]), np.sort(raft["marker_id"])
                ),
                "One-to-one cell/raft contract failed",
            )
            cell_gas = float(cells["gas_volume"].sum(dtype=np.float64))
            raft_gas = float(raft["phase_volume"].sum(dtype=np.float64))
            analytic = float(
                np.sum(
                    cells["footprint_area"]
                    * (cells["base_height"] + (cells["peak_height"] - cells["base_height"]) / 3.0),
                    dtype=np.float64,
                )
            )
            maximum_gas_residual = max(maximum_gas_residual, abs(cell_gas - raft_gas))
            maximum_geometry_residual = max(maximum_geometry_residual, abs(analytic - raft_gas))
            require(np.isfinite(vertices).all(), "Non-finite render vertex")
            require(triangles.ndim == 2 and triangles.shape[1] == 3, "Invalid triangle shape")
            require(np.all((triangles >= 0) & (triangles < len(vertices))), "Triangle index out of range")
            require(np.all(materials <= 4), "Unknown Plateau material class")
            require(len(triangles) == len(materials), "Triangle/material count mismatch")
            require(len(np.unique(films["id"])) == len(films), "Duplicate shared film ID")
            require(np.all(films["cell_a"] < films["cell_b"]), "Shared film pair is not canonical")
            require(np.isfinite(borders["liquid_volume"]).all(), "Non-finite border volume")
            require(np.isfinite(nodes["liquid_volume"]).all(), "Non-finite node volume")
            total_cells += len(cells)
            total_films += len(films)
            total_unbound += len(unbound_ids)
    require(maximum_gas_residual <= 1.0e-14, "Cell gas is not conserved")
    require(maximum_geometry_residual <= 1.0e-12, "Analytic geometry volume is not conserved")
    report = {
        "schema": "physx-foamgenerator-plateau-audit/v1",
        "valid": True,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frames_audited": len(manifest["samples"]),
        "cells_audited": total_cells,
        "shared_films_audited": total_films,
        "unbound_parcels_audited": total_unbound,
        "maximum_gas_volume_residual_m3": maximum_gas_residual,
        "maximum_analytic_geometry_volume_residual_m3": maximum_geometry_residual,
        "finite_difference_velocity_used": False,
        "surface_binding": "exact closest oriented triangle",
    }
    require(not args.output_report.exists(), f"Refusing to overwrite: {args.output_report}")
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(args.output_report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
