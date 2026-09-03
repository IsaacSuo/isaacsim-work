"""Audit that every whitewater-v6 render input belongs to one PhysX sample.

The render surface is an independently reconstructed product, so matching a
frame number is not sufficient.  This audit follows hashes from the audited
120 Hz source through liquid fields, marker trajectories, foam payload, PLY
export and Splashsurf, then checks the resulting surface anchors geometrically.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import meshio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("liquid_directory", type=Path)
    parser.add_argument("marker_directory", type=Path)
    parser.add_argument("payload_directory", type=Path)
    parser.add_argument("particle_export_directory", type=Path)
    parser.add_argument("splashsurf_directory", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--source-sample", type=int, required=True)
    parser.add_argument("--marker-frame", type=int, required=True)
    parser.add_argument("--render-frame", type=int, required=True)
    parser.add_argument(
        "--maximum-surface-anchor-distance",
        type=float,
        default=0.024,
        help="Maximum distance from surface bubble/foam/ring anchor to OBJ vertices.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed(items, key, value):
    matches = [item for item in items if int(item[key]) == int(value)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key}={value} record, found {len(matches)}")
    return matches[0]


def check_audited_manifest(directory, expected_product=None):
    manifest_path = directory / "manifest.json"
    audit_path = directory / "audit_report.json"
    manifest = load_json(manifest_path)
    audit = load_json(audit_path)
    if audit.get("valid") is not True:
        raise ValueError(f"Input audit is not valid: {audit_path}")
    expected_hash = audit.get("manifest_sha256")
    if expected_hash is not None and expected_hash != sha256_file(manifest_path):
        raise ValueError(f"Audit does not reference current manifest: {directory}")
    if expected_product is not None and manifest.get("product") != expected_product:
        raise ValueError(f"Unexpected product in {manifest_path}")
    return manifest, manifest_path, audit_path


def read_binary_xyz_ply(path):
    with Path(path).open("rb") as stream:
        header_lines = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("Expected binary little-endian PLY")
        vertex_lines = [line for line in header_lines if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError("Expected one PLY vertex element")
        count = int(vertex_lines[0].split()[-1])
        properties = [line for line in header_lines if line.startswith("property ")]
        if properties != ["property float x", "property float y", "property float z"]:
            raise ValueError("PLY must contain only float32 x/y/z")
        points = np.frombuffer(stream.read(), dtype="<f4")
    if points.size != count * 3:
        raise ValueError("PLY byte count does not match its vertex declaration")
    return points.reshape(count, 3)


def nearest_vertex_distances(anchors, vertices, cell_size, maximum_distance):
    anchors = np.asarray(anchors, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    if not len(anchors):
        return np.empty(0, dtype=np.float64)
    origin = np.minimum(anchors.min(axis=0), vertices.min(axis=0)) - cell_size
    vertex_cells = np.floor((vertices - origin) / cell_size).astype(np.int32)
    buckets = {}
    for index, cell in enumerate(vertex_cells):
        buckets.setdefault(tuple(map(int, cell)), []).append(index)
    radius = int(np.ceil(maximum_distance / cell_size))
    offsets = list(itertools.product(range(-radius, radius + 1), repeat=3))
    distances = np.full(len(anchors), np.inf, dtype=np.float64)
    anchor_cells = np.floor((anchors - origin) / cell_size).astype(np.int32)
    for anchor_index, (anchor, cell) in enumerate(zip(anchors, anchor_cells)):
        candidate_indices = []
        for offset in offsets:
            key = tuple(int(cell[axis] + offset[axis]) for axis in range(3))
            candidate_indices.extend(buckets.get(key, ()))
        if candidate_indices:
            delta = vertices[np.asarray(candidate_indices, dtype=np.int64)] - anchor
            distances[anchor_index] = np.sqrt(np.min(np.sum(delta * delta, axis=1)))
    return distances


args = parse_args()
if args.maximum_surface_anchor_distance <= 0.0:
    raise ValueError("Surface-anchor distance must be positive")
source_directory = args.source_directory.resolve()
liquid_directory = args.liquid_directory.resolve()
marker_directory = args.marker_directory.resolve()
payload_directory = args.payload_directory.resolve()
particle_export_directory = args.particle_export_directory.resolve()
splashsurf_directory = args.splashsurf_directory.resolve()
output_path = args.output_json.resolve()
if output_path.exists():
    raise FileExistsError(f"Refusing to overwrite audit: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)

checks = []


def record(name, passed, detail):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


source_manifest, source_manifest_path, source_audit_path = check_audited_manifest(
    source_directory
)
liquid_manifest, liquid_manifest_path, _ = check_audited_manifest(
    liquid_directory, "whitewater_v6_liquid_fields"
)
marker_manifest, marker_manifest_path, _ = check_audited_manifest(
    marker_directory, "whitewater_v6_marker_trajectories"
)
payload_manifest, payload_manifest_path, _ = check_audited_manifest(
    payload_directory, "whitewater_v6_foam_render_payload"
)

source_item = indexed(source_manifest["samples"], "sample_index", args.source_sample)
source_path = source_directory / source_item["file"]
source_hash = sha256_file(source_path)
record("source_file_hash", source_hash == source_item["sha256"], source_hash)

liquid_item = indexed(
    liquid_manifest["samples"], "source_sample_index", args.source_sample
)
liquid_path = liquid_directory / liquid_item["file"]
record(
    "liquid_source_hash_chain",
    liquid_item["source_sha256"] == source_hash,
    {"liquid": liquid_item["source_sha256"], "source": source_hash},
)

marker_item = indexed(marker_manifest["samples"], "output_frame", args.marker_frame)
marker_path = marker_directory / marker_item["file"]
record(
    "marker_sample_identity",
    int(marker_item["source_sample_index"]) == args.source_sample,
    marker_item,
)
record(
    "marker_liquid_manifest_chain",
    marker_manifest["inputs"]["liquid_manifest_sha256"]
    == sha256_file(liquid_manifest_path),
    marker_manifest["inputs"]["liquid_manifest_sha256"],
)

payload_item = indexed(payload_manifest["samples"], "output_frame", args.marker_frame)
payload_path = payload_directory / payload_item["file"]
record(
    "payload_sample_identity",
    int(payload_item["source_sample_index"]) == args.source_sample,
    payload_item,
)

export_manifest_path = particle_export_directory / "export_manifest.json"
export_manifest = load_json(export_manifest_path)
export_item = indexed(export_manifest["samples"], "source_sample_index", args.source_sample)
ply_path = particle_export_directory / export_item["file"]
record(
    "export_source_hash_chain",
    export_item["source_sha256"] == source_hash,
    {"export": export_item["source_sha256"], "source": source_hash},
)
record("particle_ply_hash", sha256_file(ply_path) == export_item["sha256"], export_item["sha256"])

splashsurf_manifest_path = splashsurf_directory / "splashsurf_manifest.json"
splashsurf_manifest = load_json(splashsurf_manifest_path)
splash_configuration = splashsurf_manifest["configuration"]
record(
    "splashsurf_particle_directory",
    Path(splash_configuration["particle_directory"]).resolve()
    == particle_export_directory,
    splash_configuration["particle_directory"],
)
record(
    "splashsurf_complete",
    splashsurf_manifest["state"].get("complete") is True,
    splashsurf_manifest["state"],
)

surface_path = (
    splashsurf_directory
    / "surface"
    / f"surface_{args.render_frame:04d}_clipped.obj"
)
if not surface_path.is_file():
    raise FileNotFoundError(surface_path)

with np.load(source_path, allow_pickle=False) as source_cache:
    source_positions = np.asarray(source_cache["positions"], dtype=np.float32)
    source_sphere = np.asarray(source_cache["sphere_transform"], dtype=np.float64)[3, :3]
ply_positions = read_binary_xyz_ply(ply_path)
record(
    "particle_ply_exact_positions",
    np.array_equal(source_positions, ply_positions),
    {"source_rows": len(source_positions), "ply_rows": len(ply_positions)},
)

with np.load(liquid_path, allow_pickle=False) as liquid_cache:
    liquid_sphere = np.asarray(liquid_cache["sphere_center"], dtype=np.float64)
sphere_error = float(np.linalg.norm(source_sphere - liquid_sphere))
record(
    "sphere_center_source_to_liquid",
    sphere_error <= 1.0e-6,
    {"error_m": sphere_error, "source": source_sphere.tolist(), "liquid": liquid_sphere.tolist()},
)

with np.load(marker_path, allow_pickle=False) as marker_cache:
    markers = np.asarray(marker_cache["markers"])
with np.load(payload_path, allow_pickle=False) as payload_cache:
    patches = np.asarray(payload_cache["patches"])
    rings = np.asarray(payload_cache["rings"])
surface_bubbles = markers[markers["state"] == np.uint8(4)]["position"]
anchor_groups = (
    ("surface_bubbles", surface_bubbles),
    ("foam_patches", patches["position"]),
    ("hole_rings", rings["position"]),
)
anchors = np.concatenate([values for _, values in anchor_groups], axis=0).astype(
    np.float64
)
surface_vertices = np.asarray(meshio.read(surface_path).points, dtype=np.float64)
distances = nearest_vertex_distances(
    anchors,
    surface_vertices,
    cell_size=0.008,
    maximum_distance=args.maximum_surface_anchor_distance,
)
finite = np.isfinite(distances)
maximum_distance = float(np.max(distances[finite])) if np.any(finite) else float("inf")
group_details = {}
group_start = 0
for group_name, group_anchors in anchor_groups:
    group_stop = group_start + len(group_anchors)
    group_distances = distances[group_start:group_stop]
    group_finite = np.isfinite(group_distances)
    group_bad = ~group_finite | (
        group_distances > args.maximum_surface_anchor_distance
    )
    bad_indices = np.flatnonzero(group_bad)
    worst_order = bad_indices[
        np.argsort(
            np.where(
                np.isfinite(group_distances[bad_indices]),
                group_distances[bad_indices],
                np.inf,
            )
        )[::-1]
    ][:10]
    group_details[group_name] = {
        "anchor_count": len(group_anchors),
        "finite_count": int(np.count_nonzero(group_finite)),
        "bad_count": int(np.count_nonzero(group_bad)),
        "maximum_finite_distance_m": (
            float(np.max(group_distances[group_finite]))
            if np.any(group_finite)
            else None
        ),
        "worst_anchors": [
            {
                "index": int(index),
                "position": np.asarray(group_anchors[index], dtype=float).tolist(),
                "distance_m": (
                    float(group_distances[index])
                    if np.isfinite(group_distances[index])
                    else None
                ),
            }
            for index in worst_order
        ],
    }
    group_start = group_stop
record(
    "surface_anchor_proximity",
    bool(np.all(finite) and maximum_distance <= args.maximum_surface_anchor_distance),
    {
        "anchor_count": len(anchors),
        "finite_count": int(np.count_nonzero(finite)),
        "maximum_distance_m": maximum_distance,
        "p50_distance_m": float(np.percentile(distances[finite], 50)) if np.any(finite) else None,
        "p95_distance_m": float(np.percentile(distances[finite], 95)) if np.any(finite) else None,
        "threshold_m": args.maximum_surface_anchor_distance,
        "groups": group_details,
    },
)

payload = {
    "schema": 1,
    "product": "whitewater_v6_render_alignment_audit",
    "valid": bool(all(item["passed"] for item in checks)),
    "identity": {
        "source_sample_index": args.source_sample,
        "marker_output_frame": args.marker_frame,
        "render_frame": args.render_frame,
    },
    "surface": {
        "file": str(surface_path),
        "sha256": sha256_file(surface_path),
        "vertices": len(surface_vertices),
    },
    "checks": checks,
}
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
temporary.replace(output_path)
print(json.dumps(payload, indent=2))
raise SystemExit(0 if payload["valid"] else 1)
