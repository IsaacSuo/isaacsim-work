"""Build an independent, conservative surface-bubble raft cache."""

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
from whitewater.surface_raft import (
    RAFT_DTYPE,
    SurfaceRaftModel,
    build_surface_raft,
)


SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("liquid_field_directory", type=Path)
    parser.add_argument("trajectory_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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


def load_endpoint_fields(path, spec, support_recipe):
    with np.load(path, allow_pickle=False) as cache:
        fields = {
            name: np.asarray(cache[name]).copy()
            for name in ("phi", "normal", "velocity")
        }
        load_collision_fields(cache, fields)
        support, metrics = build_render_surface_support(
            np.asarray(cache["fluid_mask"]), spec, support_recipe
        )
        fields["render_surface_support"] = support
    return fields, metrics


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
    liquid_complete = bool(
        liquid_manifest.get("complete", liquid_manifest.get("state", {}).get("complete"))
    )
    if not liquid_complete or not trajectory_manifest.get("complete"):
        raise RuntimeError("Raft inputs must both be complete")
    if liquid_manifest.get("grid") != trajectory_manifest.get("grid"):
        raise RuntimeError("Liquid and trajectory grids differ")
    liquid_samples = liquid_manifest.get("samples", [])
    trajectory_samples = trajectory_manifest.get("samples", [])
    if not liquid_samples or len(liquid_samples) != len(trajectory_samples):
        raise RuntimeError("Liquid and trajectory sample counts differ")
    for liquid_sample, trajectory_sample in zip(liquid_samples, trajectory_samples):
        if int(liquid_sample["source_sample_index"]) != int(
            trajectory_sample["source_sample_index"]
        ):
            raise RuntimeError("Liquid and trajectory source samples differ")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty raft directory: {args.output_directory}"
        )

    grid = liquid_manifest["grid"]
    spec = GridSpec(tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"]))
    model = SurfaceRaftModel(spec.spacing)
    support_recipe = load_render_surface_support_recipe(
        trajectory_manifest["configuration"]["render_surface_support"][
            "recipe_manifest"
        ]
    )
    args.output_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_directory / "manifest.json"
    script_path = Path(__file__)
    module_path = script_path.parent / "whitewater" / "surface_raft.py"
    manifest = {
        "schema": SCHEMA,
        "product": "whitewater_v6_surface_raft",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": spec.metadata(),
        "configuration": {
            "model": model.metadata(),
            "marker_selection": "state == SURFACE_BUBBLE",
            "temporal_prediction": (
                "previous raft_position + current raw anchor - previous raw anchor, "
                "joined by marker ID"
            ),
            "endpoint_field_alignment": (
                "trajectory row i is at interval end and is constrained against "
                "liquid row min(i+1,last)"
            ),
            "render_surface_support": support_recipe.metadata(),
        },
        "inputs": {
            "liquid_manifest": str(liquid_manifest_path.resolve()),
            "liquid_manifest_sha256": sha256_file(liquid_manifest_path),
            "trajectory_manifest": str(trajectory_manifest_path.resolve()),
            "trajectory_manifest_sha256": sha256_file(trajectory_manifest_path),
        },
        "producer": {
            "script": str(script_path.resolve()),
            "script_sha256": sha256_file(script_path),
            "surface_raft_module": str(module_path.resolve()),
            "surface_raft_module_sha256": sha256_file(module_path),
        },
        "record_contract": {
            "raft_dtype": RAFT_DTYPE.descr,
            "cardinality": "one output row per active surface-bubble marker",
            "ordering": "strictly increasing marker_id",
            "immutable_fields": [
                "marker_id",
                "physical_radius",
                "representative_count",
                "phase_volume",
                "shape",
                "state_age",
                "random_key",
            ],
            "derived_fields": [
                "raft_position",
                "raft_velocity",
                "neighbor_count",
                "contact_count",
                "maximum_contact_compression",
                "packing_fraction",
                "cluster_id",
                "cluster_size",
                "cluster_representative_count",
                "anchor_displacement",
                "support_value",
                "support_fallback",
                "anchor_support_repair",
            ],
        },
        "samples": [],
        "complete": False,
    }
    atomic_json(manifest_path, manifest)

    previous = np.empty(0, dtype=RAFT_DTYPE)
    cumulative = {
        "gas_volume_residual_m3": 0.0,
        "support_fallbacks": 0,
        "anchor_support_repairs": 0,
        "support_bad": 0,
        "hard_tether_hits": 0,
        "maximum_anchor_displacement_m": 0.0,
        "maximum_overlap_m": 0.0,
        "maximum_surface_bubbles": 0,
        "largest_cluster": 0,
    }
    for index, trajectory_sample in enumerate(trajectory_samples):
        trajectory_path = args.trajectory_directory / trajectory_sample["file"]
        with np.load(trajectory_path, allow_pickle=False) as cache:
            markers = np.asarray(cache["markers"]).copy()
            interval_start = float(cache["interval_start"])
            interval_end = float(cache["interval_end"])
            output_frame = int(cache["output_frame"])
            source_sample_index = int(cache["source_sample_index"])
        endpoint_index = min(index + 1, len(liquid_samples) - 1)
        endpoint_sample = liquid_samples[endpoint_index]
        endpoint_path = args.liquid_field_directory / endpoint_sample["file"]
        fields, support_metrics = load_endpoint_fields(
            endpoint_path, spec, support_recipe
        )
        raft, metrics = build_surface_raft(
            markers,
            previous,
            interval_end - interval_start,
            fields,
            spec,
            model,
        )
        output_path = args.output_directory / f"surface_raft_state_{index:06d}.npz"
        atomic_npz(
            output_path,
            schema=np.int32(SCHEMA),
            output_frame=np.int32(output_frame),
            source_sample_index=np.int32(source_sample_index),
            interval_start=np.float64(interval_start),
            interval_end=np.float64(interval_end),
            endpoint_liquid_source_sample=np.int32(
                endpoint_sample["source_sample_index"]
            ),
            raft=raft,
        )
        for name in ("gas_volume_residual_m3",):
            cumulative[name] += metrics[name]
        for name in (
            "support_fallbacks",
            "anchor_support_repairs",
            "support_bad",
            "hard_tether_hits",
        ):
            cumulative[name] += metrics[name]
        for name in (
            "maximum_anchor_displacement_m",
            "maximum_overlap_m",
            "surface_bubbles",
            "largest_cluster",
        ):
            target = "maximum_surface_bubbles" if name == "surface_bubbles" else name
            cumulative[target] = max(cumulative[target], metrics[name])
        sample_payload = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "output_frame": output_frame,
            "source_sample_index": source_sample_index,
            "endpoint_liquid_source_sample": int(
                endpoint_sample["source_sample_index"]
            ),
            "render_surface_support_metrics": support_metrics,
            "metrics": metrics,
        }
        manifest["samples"].append(sample_payload)
        atomic_json(manifest_path, manifest)
        print(
            f"[v6-surface-raft] source={source_sample_index:04d} "
            f"bubbles={len(raft)} inherited={metrics['inherited']} "
            f"pairs={metrics['compatible_final_pairs']} "
            f"clusters={metrics['clusters']} largest={metrics['largest_cluster']} "
            f"fallback={metrics['support_fallbacks']} "
            f"max_offset_mm={1.0e3 * metrics['maximum_anchor_displacement_m']:.3f}",
            flush=True,
        )
        previous = raft

    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["cumulative"] = cumulative
    atomic_json(manifest_path, manifest)
    print(json.dumps({"complete": True, "samples": len(manifest["samples"]), "cumulative": cumulative}, indent=2))


if __name__ == "__main__":
    main()
