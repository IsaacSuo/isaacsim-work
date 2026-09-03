"""Build persistent v6 surface-foam parcels and burst repellents."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec, load_collision_fields
from whitewater.render_surface_support import (
    build_render_surface_support,
    load_render_surface_support_recipe,
)
from whitewater.surface_foam import (
    FOAM_DTYPE,
    REPELLENT_DTYPE,
    SurfaceFoamModel,
    advance_surface_layer,
    source_surface_foam,
)


SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("liquid_field_directory", type=Path)
    parser.add_argument("trajectory_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_liquid_fields(path, spec, support_recipe):
    with np.load(path) as cache:
        fields = {
            name: np.asarray(cache[name]).copy()
            for name in ("phi", "normal", "velocity")
        }
        load_collision_fields(cache, fields)
        support, support_metrics = build_render_surface_support(
            np.asarray(cache["fluid_mask"]), spec, support_recipe
        )
        fields["render_surface_support"] = support
        return fields, support_metrics


def main():
    args = parse_args()
    liquid_manifest_path = args.liquid_field_directory / "manifest.json"
    trajectory_manifest_path = args.trajectory_directory / "manifest.json"
    liquid_manifest = load_json(liquid_manifest_path)
    trajectory_manifest = load_json(trajectory_manifest_path)
    if liquid_manifest.get("product") != "whitewater_v6_liquid_fields":
        raise RuntimeError("Unexpected liquid-field product")
    if trajectory_manifest.get("product") != "whitewater_v6_marker_trajectories":
        raise RuntimeError("Unexpected marker-trajectory product")
    if not trajectory_manifest.get("complete"):
        raise RuntimeError("Marker trajectories are incomplete")
    if liquid_manifest.get("grid") != trajectory_manifest.get("grid"):
        raise RuntimeError("Liquid and trajectory grids differ")
    liquid_samples = liquid_manifest["samples"]
    trajectory_samples = trajectory_manifest["samples"]
    if len(liquid_samples) != len(trajectory_samples):
        raise RuntimeError("Liquid and trajectory sample counts differ")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty surface-foam directory: {args.output_directory}"
        )

    grid = liquid_manifest["grid"]
    spec = GridSpec(
        tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"])
    )
    model = SurfaceFoamModel(spec.spacing)
    support_recipe = load_render_surface_support_recipe(
        trajectory_manifest["configuration"]["render_surface_support"][
            "recipe_manifest"
        ]
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_directory / "manifest.json"
    module_path = Path(__file__).parent / "whitewater" / "surface_foam.py"
    manifest = {
        "schema": SCHEMA,
        "product": "whitewater_v6_surface_foam",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": spec.metadata(),
        "configuration": {
            "model": model.metadata(),
            "source_events": ["spray_reentry", "bubble_burst"],
            "repellent_source": "bubble_burst",
            "field_time_interpolation": "linear between adjacent snapshots",
            "last_interval_field_policy": "zero_order_hold",
            "render_surface_support": support_recipe.metadata(),
        },
        "inputs": {
            "liquid_manifest": str(liquid_manifest_path.resolve()),
            "liquid_manifest_sha256": sha256_file(liquid_manifest_path),
            "trajectory_manifest": str(trajectory_manifest_path.resolve()),
            "trajectory_manifest_sha256": sha256_file(trajectory_manifest_path),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "surface_foam_module": str(module_path.resolve()),
            "surface_foam_module_sha256": sha256_file(module_path),
        },
        "record_contract": {
            "foam_dtype": FOAM_DTYPE.descr,
            "repellent_dtype": REPELLENT_DTYPE.descr,
            "area_units": "square metres",
            "coverage": "clip(film_area/support_area, 0, 1)",
            "repellent_effective_strength": "strength * clip(1-age/lifetime, 0, 1)",
        },
        "samples": [],
        "complete": False,
    }
    atomic_json(manifest_path, manifest)

    foam = np.empty(0, dtype=FOAM_DTYPE)
    repellents = np.empty(0, dtype=REPELLENT_DTYPE)
    cumulative = {
        "eligible_events": 0,
        "foam_sources": 0,
        "repellent_sources": 0,
        "rejected_low_weber": 0,
        "injected_film_area_m2": 0.0,
        "drained_film_area_m2": 0.0,
        "topology_lost_film_area_m2": 0.0,
        "topology_lost_foam": 0,
        "topology_lost_repellents": 0,
        "expired_foam": 0,
        "expired_repellents": 0,
    }
    current_fields, current_support_metrics = load_liquid_fields(
        args.liquid_field_directory / liquid_samples[0]["file"],
        spec,
        support_recipe,
    )
    for index, (liquid_sample, trajectory_sample) in enumerate(
        zip(liquid_samples, trajectory_samples)
    ):
        trajectory_path = args.trajectory_directory / trajectory_sample["file"]
        with np.load(trajectory_path) as cache:
            events = np.asarray(cache["terminal_events"]).copy()
            interval_start = float(cache["interval_start"])
            interval_end = float(cache["interval_end"])
        dt = interval_end - interval_start
        new_foam, new_repellents, source_metrics = source_surface_foam(events, model)
        if len(new_foam):
            if np.intersect1d(foam["id"], new_foam["id"]).size:
                raise RuntimeError("A foam source ID is already active")
            foam = np.concatenate((foam, new_foam))
        if len(new_repellents):
            if np.intersect1d(repellents["id"], new_repellents["id"]).size:
                raise RuntimeError("A repellent source ID is already active")
            repellents = np.concatenate((repellents, new_repellents))
        if index + 1 < len(liquid_samples):
            next_fields, next_support_metrics = load_liquid_fields(
                args.liquid_field_directory / liquid_samples[index + 1]["file"],
                spec,
                support_recipe,
            )
        else:
            next_fields = None
            next_support_metrics = current_support_metrics
        foam, repellents, advance_metrics = advance_surface_layer(
            foam,
            repellents,
            interval_start,
            dt,
            current_fields,
            next_fields,
            spec,
            model,
        )
        for key in (
            "eligible_events",
            "foam_sources",
            "repellent_sources",
            "rejected_low_weber",
            "injected_film_area_m2",
        ):
            cumulative[key] += source_metrics[key]
        for key in advance_metrics:
            cumulative[key] += advance_metrics[key]
        output_path = args.output_directory / f"surface_foam_state_{index:06d}.npz"
        atomic_npz(
            output_path,
            schema=np.int32(SCHEMA),
            output_frame=np.int32(liquid_sample["output_frame"]),
            source_sample_index=np.int32(liquid_sample["source_sample_index"]),
            interval_start=np.float64(interval_start),
            interval_end=np.float64(interval_end),
            foam=foam,
            repellents=repellents,
        )
        film_area = float(foam["film_area"].sum(dtype=np.float64))
        support_area = float(foam["support_area"].sum(dtype=np.float64))
        coverage = np.divide(
            foam["film_area"].astype(np.float64),
            foam["support_area"].astype(np.float64),
            out=np.zeros(len(foam), dtype=np.float64),
            where=foam["support_area"] > 0.0,
        )
        sample_payload = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "output_frame": int(liquid_sample["output_frame"]),
            "source_sample_index": int(liquid_sample["source_sample_index"]),
            "interval_start": interval_start,
            "interval_end": interval_end,
            "source_metrics": source_metrics,
            "advance_metrics": advance_metrics,
            "render_surface_support_metrics": next_support_metrics,
            "active_foam": len(foam),
            "active_repellents": len(repellents),
            "active_film_area_m2": film_area,
            "active_support_area_m2": support_area,
            "coverage_quantiles": (
                np.quantile(np.clip(coverage, 0.0, 1.0), (0.0, 0.5, 0.9, 0.99, 1.0)).tolist()
                if len(coverage)
                else [0.0] * 5
            ),
        }
        manifest["samples"].append(sample_payload)
        atomic_json(manifest_path, manifest)
        print(
            f"[v6-surface-foam] source={liquid_sample['source_sample_index']:04d} "
            f"events={len(events)} born={len(new_foam)} foam={len(foam)} "
            f"repellents={len(repellents)} film={film_area:.6e}",
            flush=True,
        )
        current_fields = next_fields if next_fields is not None else current_fields
        current_support_metrics = next_support_metrics

    cumulative["active_foam"] = len(foam)
    cumulative["active_repellents"] = len(repellents)
    cumulative["active_film_area_m2"] = float(
        foam["film_area"].sum(dtype=np.float64)
    )
    cumulative["film_area_balance_residual_m2"] = (
        cumulative["injected_film_area_m2"]
        - cumulative["drained_film_area_m2"]
        - cumulative["topology_lost_film_area_m2"]
        - cumulative["active_film_area_m2"]
    )
    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["cumulative"] = cumulative
    atomic_json(manifest_path, manifest)
    print(json.dumps({"complete": True, "samples": len(manifest["samples"]), "cumulative": cumulative}, indent=2))


if __name__ == "__main__":
    main()
