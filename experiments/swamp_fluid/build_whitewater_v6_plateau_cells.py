"""Build persistent, volume-conservative Plateau-cell topology and geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.plateau_cells import (
    BORDER_DTYPE,
    CELL_DTYPE,
    EVENT_DTYPE,
    FILM_DTYPE,
    NODE_DTYPE,
    PlateauCellModel,
    build_plateau_cells,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("liquid_field_directory", type=Path)
    parser.add_argument("surface_raft_directory", type=Path)
    parser.add_argument("surface_foam_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main():
    args = parse_args()
    directories = {
        "liquid": args.liquid_field_directory.resolve(),
        "raft": args.surface_raft_directory.resolve(),
        "foam": args.surface_foam_directory.resolve(),
    }
    paths = {name: directory / "manifest.json" for name, directory in directories.items()}
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    expected = {
        "liquid": "whitewater_v6_liquid_fields",
        "raft": "whitewater_v6_surface_raft",
        "foam": "whitewater_v6_surface_foam",
    }
    for name, product in expected.items():
        if manifests[name].get("product") != product:
            raise RuntimeError(f"Unexpected {name} product")
    if not manifests["raft"].get("complete") or not manifests["foam"].get("complete"):
        raise RuntimeError("Plateau inputs are incomplete")
    sample_count = len(manifests["raft"]["samples"])
    if sample_count == 0 or len(manifests["foam"]["samples"]) != sample_count:
        raise RuntimeError("Raft and foam sample counts differ")
    grid = manifests["liquid"]["grid"]
    spec = GridSpec(tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"]))
    model = PlateauCellModel()

    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    script_path = Path(__file__).resolve()
    module_path = Path(__file__).parent / "whitewater" / "plateau_cells.py"
    manifest = {
        "schema": 1,
        "product": "whitewater_v6_plateau_cells",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "grid": spec.metadata(),
        "configuration": {
            "model": model.metadata(),
            "material_classes": {
                "0": "cell_top_membrane",
                "1": "cell_outer_membrane",
                "2": "unique_shared_film",
                "3": "plateau_border",
                "4": "plateau_node",
            },
            "event_kinds": {
                "1": "shared_film_formed",
                "2": "contact_film_separated",
                "3": "film_ruptured_and_cells_coalesced",
                "4": "plateau_node_formed",
                "5": "plateau_node_lost",
            },
        },
        "inputs": {
            name + "_manifest": str(path)
            for name, path in paths.items()
        }
        | {
            name + "_manifest_sha256": sha256_file(path)
            for name, path in paths.items()
        },
        "producer": {
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "module": str(module_path.resolve()),
            "module_sha256": sha256_file(module_path),
        },
        "record_contract": {
            "cell_dtype": CELL_DTYPE.descr,
            "film_dtype": FILM_DTYPE.descr,
            "border_dtype": BORDER_DTYPE.descr,
            "node_dtype": NODE_DTYPE.descr,
            "event_dtype": EVENT_DTYPE.descr,
            "gas_volume": "sum of expanded child cells equals the raft exactly",
            "shared_film": "one row and one geometric membrane per unordered cell pair",
            "film_area": "separate from gas volume and bounded by active surface-foam area",
            "film_liquid": "area times physical thickness; separately debited from carrier liquid",
        },
        "samples": [],
    }
    atomic_json(manifest_path, manifest)

    previous_films = np.empty(0, dtype=FILM_DTYPE)
    previous_nodes = np.empty(0, dtype=NODE_DTYPE)
    previous_ruptured = np.empty(0, dtype=np.uint64)
    cumulative = {
        "maximum_cells": 0,
        "maximum_shared_films": 0,
        "maximum_plateau_borders": 0,
        "maximum_plateau_nodes": 0,
        "film_formations": 0,
        "film_separations": 0,
        "film_ruptures": 0,
        "node_formations": 0,
        "node_losses": 0,
        "maximum_gas_volume_residual_m3": 0.0,
        "maximum_analytic_geometry_volume_residual_m3": 0.0,
        "maximum_parent_packet_volume_residual_m3": 0.0,
        "maximum_film_budget_fraction": 0.0,
        "maximum_cell_aspect": 0.0,
    }
    liquid_samples = manifests["liquid"]["samples"]
    for index, (raft_sample, foam_sample) in enumerate(
        zip(manifests["raft"]["samples"], manifests["foam"]["samples"])
    ):
        if int(raft_sample["source_sample_index"]) != int(
            foam_sample["source_sample_index"]
        ):
            raise RuntimeError("Raft and foam source samples differ")
        raft_path = directories["raft"] / raft_sample["file"]
        foam_path = directories["foam"] / foam_sample["file"]
        with np.load(raft_path, allow_pickle=False) as cache:
            raft = np.asarray(cache["raft"]).copy()
            interval_start = float(cache["interval_start"])
            interval_end = float(cache["interval_end"])
            output_frame = int(cache["output_frame"])
            source_sample = int(cache["source_sample_index"])
        with np.load(foam_path, allow_pickle=False) as cache:
            foam = np.asarray(cache["foam"])
        active_film_area = float(foam["film_area"].sum(dtype=np.float64))
        endpoint_index = min(index + 1, len(liquid_samples) - 1)
        liquid_path = directories["liquid"] / liquid_samples[endpoint_index]["file"]
        with np.load(liquid_path, allow_pickle=False) as cache:
            fields = {
                "phi": np.asarray(cache["phi"]),
                "normal": np.asarray(cache["normal"]),
                "collision_sdf": np.asarray(cache["collision_sdf"]),
            }
            topology, metrics = build_plateau_cells(
                raft,
                fields,
                spec,
                active_surface_film_area_m2=active_film_area,
                event_time=interval_end,
                dt=interval_end - interval_start,
                previous_films=previous_films,
                previous_nodes=previous_nodes,
                previous_ruptured_pairs=previous_ruptured,
                model=model,
            )
        output_path = output_directory / f"plateau_cells_{index:06d}.npz"
        atomic_npz(
            output_path,
            schema=np.int32(1),
            output_frame=np.int32(output_frame),
            source_sample_index=np.int32(source_sample),
            interval_start=np.float64(interval_start),
            interval_end=np.float64(interval_end),
            **topology,
        )
        event_counts = {
            name: int(np.count_nonzero(topology["events"]["kind"] == kind))
            for name, kind in (
                ("film_formations", 1),
                ("film_separations", 2),
                ("film_ruptures", 3),
                ("node_formations", 4),
                ("node_losses", 5),
            )
        }
        gas_residual = metrics["source_gas_volume_m3"] - metrics["cell_gas_volume_m3"]
        analytic_residual = (
            metrics["cell_gas_volume_m3"] - metrics["analytic_geometry_volume_m3"]
        )
        budget_fraction = metrics["shared_film_area_m2"] / max(
            metrics["available_surface_film_area_m2"], 1.0e-30
        )
        for name in (
            "maximum_cells",
            "maximum_shared_films",
            "maximum_plateau_borders",
            "maximum_plateau_nodes",
        ):
            metric_name = {
                "maximum_cells": "expanded_cells",
                "maximum_shared_films": "shared_films",
                "maximum_plateau_borders": "plateau_borders",
                "maximum_plateau_nodes": "plateau_nodes",
            }[name]
            cumulative[name] = max(cumulative[name], metrics[metric_name])
        for name, value in event_counts.items():
            cumulative[name] += value
        cumulative["maximum_gas_volume_residual_m3"] = max(
            cumulative["maximum_gas_volume_residual_m3"], abs(gas_residual)
        )
        cumulative["maximum_analytic_geometry_volume_residual_m3"] = max(
            cumulative["maximum_analytic_geometry_volume_residual_m3"],
            abs(analytic_residual),
        )
        cumulative["maximum_parent_packet_volume_residual_m3"] = max(
            cumulative["maximum_parent_packet_volume_residual_m3"],
            metrics["parent_volume_residual_m3"],
        )
        cumulative["maximum_film_budget_fraction"] = max(
            cumulative["maximum_film_budget_fraction"], budget_fraction
        )
        cumulative["maximum_cell_aspect"] = max(
            cumulative["maximum_cell_aspect"], metrics["maximum_cell_aspect"]
        )
        entry = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "output_frame": output_frame,
            "source_sample_index": source_sample,
            "endpoint_liquid_source_sample": int(
                liquid_samples[endpoint_index]["source_sample_index"]
            ),
            "metrics": metrics,
            "event_counts": event_counts,
            "gas_volume_residual_m3": gas_residual,
            "analytic_geometry_volume_residual_m3": analytic_residual,
            "film_budget_fraction": budget_fraction,
            "geometry": {
                "vertices": len(topology["vertices"]),
                "triangles": len(topology["triangles"]),
            },
        }
        manifest["samples"].append(entry)
        atomic_json(manifest_path, manifest)
        print(
            f"[v6-plateau] source={source_sample:04d} "
            f"cells={metrics['expanded_cells']} contacts={metrics['contact_pairs']} "
            f"films={metrics['shared_films']} borders={metrics['plateau_borders']} "
            f"nodes={metrics['plateau_nodes']} gas_residual={gas_residual:.3e}",
            flush=True,
        )
        previous_films = topology["films"]
        previous_nodes = topology["nodes"]
        previous_ruptured = topology["ruptured_pair_ids"]

    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["cumulative"] = cumulative
    atomic_json(manifest_path, manifest)
    print(json.dumps(cumulative, indent=2))


if __name__ == "__main__":
    main()
