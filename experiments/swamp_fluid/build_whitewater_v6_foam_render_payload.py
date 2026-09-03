"""Build conservative multi-scale v6 foam render primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_render_payload import (
    PATCH_DTYPE,
    RING_DTYPE,
    FoamRenderModel,
    compose_foam_render_payload,
)
from whitewater.liquid_fields import GridSpec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("liquid_field_directory", type=Path)
    parser.add_argument("surface_foam_directory", type=Path)
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


def main():
    args = parse_args()
    liquid_manifest_path = args.liquid_field_directory / "manifest.json"
    foam_manifest_path = args.surface_foam_directory / "manifest.json"
    liquid_manifest = load_json(liquid_manifest_path)
    foam_manifest = load_json(foam_manifest_path)
    if liquid_manifest.get("product") != "whitewater_v6_liquid_fields":
        raise RuntimeError("Unexpected liquid-field product")
    if foam_manifest.get("product") != "whitewater_v6_surface_foam":
        raise RuntimeError("Unexpected surface-foam product")
    if not foam_manifest.get("complete"):
        raise RuntimeError("Surface-foam input is incomplete")
    if liquid_manifest.get("grid") != foam_manifest.get("grid"):
        raise RuntimeError("Liquid and surface-foam grids differ")
    liquid_samples = liquid_manifest["samples"]
    foam_samples = foam_manifest["samples"]
    if len(liquid_samples) != len(foam_samples):
        raise RuntimeError("Liquid and surface-foam sample counts differ")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty render-payload directory: {args.output_directory}"
        )
    grid = liquid_manifest["grid"]
    spec = GridSpec(
        tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"])
    )
    model = FoamRenderModel()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_directory / "manifest.json"
    module_path = Path(__file__).parent / "whitewater" / "foam_render_payload.py"
    manifest = {
        "schema": 1,
        "product": "whitewater_v6_foam_render_payload",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": spec.metadata(),
        "configuration": {"model": model.metadata()},
        "inputs": {
            "liquid_manifest": str(liquid_manifest_path.resolve()),
            "liquid_manifest_sha256": sha256_file(liquid_manifest_path),
            "surface_foam_manifest": str(foam_manifest_path.resolve()),
            "surface_foam_manifest_sha256": sha256_file(foam_manifest_path),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "module": str(module_path.resolve()),
            "module_sha256": sha256_file(module_path),
        },
        "record_contract": {
            "patch_dtype": PATCH_DTYPE.descr,
            "ring_dtype": RING_DTYPE.descr,
            "film_area_balance": "source_film_area = patch_film_area + ring_film_area",
        },
        "samples": [],
        "complete": False,
    }
    atomic_json(manifest_path, manifest)
    cumulative = {
        "source_film_area_m2": 0.0,
        "patch_film_area_m2": 0.0,
        "ring_film_area_m2": 0.0,
        "area_balance_residual_m2": 0.0,
        "affected_patches": 0,
        "active_rings": 0,
        "patch_rows": 0,
        "render_class_counts": {
            "micro_density": 0,
            "bubble_cluster": 0,
            "macro_patch": 0,
        },
    }
    for index, (liquid_sample, foam_sample) in enumerate(
        zip(liquid_samples, foam_samples)
    ):
        liquid_path = args.liquid_field_directory / liquid_sample["file"]
        foam_path = args.surface_foam_directory / foam_sample["file"]
        with np.load(liquid_path) as cache:
            normal = np.asarray(cache["normal"])
        with np.load(foam_path) as cache:
            foam = np.asarray(cache["foam"])
            repellents = np.asarray(cache["repellents"])
        patches, rings, metrics = compose_foam_render_payload(
            foam, repellents, normal, spec, model
        )
        output_path = args.output_directory / f"foam_render_payload_{index:06d}.npz"
        atomic_npz(
            output_path,
            schema=np.int32(1),
            output_frame=np.int32(liquid_sample["output_frame"]),
            source_sample_index=np.int32(liquid_sample["source_sample_index"]),
            patches=patches,
            rings=rings,
        )
        sample_payload = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "output_frame": int(liquid_sample["output_frame"]),
            "source_sample_index": int(liquid_sample["source_sample_index"]),
            "patch_count": len(patches),
            "ring_count": len(rings),
            "metrics": metrics,
        }
        manifest["samples"].append(sample_payload)
        atomic_json(manifest_path, manifest)
        for key in (
            "source_film_area_m2",
            "patch_film_area_m2",
            "ring_film_area_m2",
            "area_balance_residual_m2",
            "affected_patches",
            "active_rings",
        ):
            cumulative[key] += metrics[key]
        cumulative["patch_rows"] += len(patches)
        for key, value in metrics.get("render_class_counts", {}).items():
            cumulative["render_class_counts"][key] += value
        print(
            f"[v6-foam-render] source={liquid_sample['source_sample_index']:04d} "
            f"patches={len(patches)} affected={metrics['affected_patches']} "
            f"rings={len(rings)} ring_area={metrics['ring_film_area_m2']:.6e}",
            flush=True,
        )
    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["cumulative"] = cumulative
    atomic_json(manifest_path, manifest)
    print(json.dumps({"complete": True, "samples": len(manifest["samples"]), "cumulative": cumulative}, indent=2))


if __name__ == "__main__":
    main()
