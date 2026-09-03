"""Build one conservative mesoscopic foam membrane on production Splashsurf."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_membrane_surface import (
    FoamMembraneModel,
    build_conservative_membrane,
    load_triangle_obj,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-obj", type=Path, required=True)
    parser.add_argument("--payload-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--maximum-anchor-distance", type=float, default=0.016)
    parser.add_argument("--minimum-up-normal", type=float, default=0.50)
    parser.add_argument("--minimum-sigma-major", type=float, default=0.0040)
    parser.add_argument("--minimum-sigma-minor", type=float, default=0.0030)
    parser.add_argument("--sigma-radius-scale", type=float, default=1.75)
    parser.add_argument("--target-island-area", type=float, default=0.000150)
    parser.add_argument("--minimum-seed-separation", type=float, default=0.024)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


args = parse_args()
surface_path = args.surface_obj.resolve()
payload_path = args.payload_npz.resolve()
output_path = args.output_npz.resolve()
report_path = output_path.with_suffix(".json")
if output_path.exists() or report_path.exists():
    raise FileExistsError(f"Refusing to overwrite membrane output: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)

model = FoamMembraneModel(
    minimum_abs_up_normal=args.minimum_up_normal,
    maximum_anchor_distance_m=args.maximum_anchor_distance,
    minimum_sigma_major_m=args.minimum_sigma_major,
    minimum_sigma_minor_m=args.minimum_sigma_minor,
    sigma_radius_scale=args.sigma_radius_scale,
    target_island_area_m2=args.target_island_area,
    minimum_seed_separation_m=args.minimum_seed_separation,
)
vertices, triangles = load_triangle_obj(surface_path)
with np.load(payload_path, allow_pickle=False) as payload:
    patches = np.asarray(payload["patches"])
    output_frame = int(payload["output_frame"])
    source_sample_index = int(payload["source_sample_index"])
geometry, metrics = build_conservative_membrane(
    vertices, triangles, patches, model=model
)

with output_path.open("wb") as stream:
    np.savez_compressed(
        stream,
        schema=np.int32(1),
        output_frame=np.int32(output_frame),
        source_sample_index=np.int32(source_sample_index),
        vertices=geometry["vertices"],
        triangles=geometry["triangles"],
        triangle_normals=geometry["triangle_normals"],
        source_face_indices=geometry["source_face_indices"],
        nearest_eligible_patch_rows=geometry["nearest_eligible_patch_rows"],
        aggregated_source_foam_ids=geometry["aggregated_source_foam_ids"],
        target_membrane_area_m2=np.float64(metrics["target_membrane_area_m2"]),
        represented_membrane_area_m2=np.float64(
            metrics["represented_membrane_area_m2"]
        ),
    )

report = {
    "schema": 1,
    "product": "whitewater_v6_conservative_surface_foam_membrane",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": bool(
        abs(metrics["area_residual_m2"]) <= 1.0e-10
        and metrics["relocation_distance_m"]["maximum"]
        <= model.maximum_anchor_distance_m * (1.0 + 1.0e-9)
    ),
    "identity": {
        "output_frame": output_frame,
        "source_sample_index": source_sample_index,
    },
    "model": model.metadata(),
    "metrics": metrics,
    "inputs": {
        "surface_obj": str(surface_path),
        "surface_sha256": sha256_file(surface_path),
        "payload_npz": str(payload_path),
        "payload_sha256": sha256_file(payload_path),
    },
    "output": {
        "npz": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    },
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("WHITEWATER_V6_FOAM_MEMBRANE=" + str(output_path))
print("WHITEWATER_V6_FOAM_MEMBRANE_REPORT=" + str(report_path))
