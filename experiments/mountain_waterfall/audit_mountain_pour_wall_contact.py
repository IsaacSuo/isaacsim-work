"""Independently audit cumulative real-wall contact in a birth-aware source cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_directory", type=Path)
parser.add_argument("run_report", type=Path)
parser.add_argument("wall_target", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

for path in (args.source_directory, args.run_report, args.wall_target):
    if not path.exists():
        raise FileNotFoundError(path)
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")

shared_module_root = Path(__file__).resolve().parents[1] / "swamp_fluid"
sys.path.insert(0, str(shared_module_root))
from whitewater.contact_episodes import MeasuredSurfaceContactTracker


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


manifest_path = args.source_directory / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
run_report = json.loads(args.run_report.read_text(encoding="utf-8"))
wall_target = json.loads(args.wall_target.read_text(encoding="utf-8"))
if manifest.get("schema") != 2 or manifest.get("state", {}).get("complete") is not True:
    raise ValueError("Wall contact audit requires a complete birth-aware schema-2 source")
if wall_target.get("product") != "mountain_pour_wall_target":
    raise ValueError("Unexpected measured wall target")
rows = [row for row in wall_target["rays"] if row.get("accepted") is True]
points = np.asarray([row["location_isaac"] for row in rows], dtype=np.float64)
normals = np.asarray([row["normal_isaac"] for row in rows], dtype=np.float64)
measured = wall_target["measured_wall"]
tracker = MeasuredSurfaceContactTracker(
    points=points,
    normals=normals,
    aabb_minimum=measured["contact_gate_aabb_minimum_isaac"],
    aabb_maximum=measured["contact_gate_aabb_maximum_isaac"],
    initial_particle_count=int(run_report["particle_count"]),
    maximum_contact_distance=0.09,
    episode_entry_distance=0.11,
    minimum_incoming_normal_speed=0.20,
    minimum_cumulative_outward_change=0.30,
    maximum_episode_gap_steps=3,
)

sample_hashes_valid = True
sample_metadata_valid = True
preroll_steps = int(run_report["preroll_physics_steps"])
for expected_index, row in enumerate(manifest["samples"]):
    sample_path = args.source_directory / row["file"]
    if not sample_path.is_file() or sha256_file(sample_path) != row["sha256"]:
        sample_hashes_valid = False
        continue
    with np.load(sample_path, allow_pickle=False) as cache:
        sample_index = int(cache["sample_index"])
        relative_step = int(cache["physics_step"])
        positions = np.asarray(cache["positions"], dtype=np.float32)
        velocities = np.asarray(cache["velocities"], dtype=np.float32)
        particle_ids = np.asarray(cache["particle_ids"], dtype=np.int64)
    if sample_index != expected_index or relative_step != int(row["physics_step"]):
        sample_metadata_valid = False
    tracker.update(preroll_steps + relative_step, particle_ids, positions, velocities)

metrics = tracker.metrics()
criteria = {
    "source_manifest_is_complete_schema_2": (
        manifest.get("schema") == 2 and manifest.get("state", {}).get("complete") is True
    ),
    "all_source_sample_hashes_match": sample_hashes_valid,
    "sample_indices_and_steps_match_manifest": sample_metadata_valid,
    "surface_proximity_is_reached": (
        metrics["minimum_sample_distance_m"] is not None
        and metrics["minimum_sample_distance_m"] <= 0.09
    ),
    "minimum_unique_wall_contacts_are_measured": metrics["contact_unique_particles"] >= 12,
    "contact_fraction_of_true_near_surface_particles_passes": (
        metrics["contact_fraction_of_near_surface"] >= 0.10
    ),
    "resolved_cumulative_normal_deflection_is_present": (
        metrics["maximum_cumulative_outward_change_m_s"] >= 0.30
    ),
}
criteria = {name: bool(value) for name, value in criteria.items()}
report = {
    "schema": 1,
    "product": "mountain_pour_wall_contact_audit",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": metrics,
    "source": {
        "directory": str(args.source_directory.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "run_report": str(args.run_report.resolve()),
        "run_report_sha256": sha256_file(args.run_report),
        "wall_target": str(args.wall_target.resolve()),
        "wall_target_sha256": sha256_file(args.wall_target),
        "samples": len(manifest["samples"]),
        "surface_samples": len(points),
    },
    "method": {
        "contact_denominator": "unique emitted particles within 0.11 m of a measured wall sample",
        "contact_evidence": (
            "distance <= 0.09 m, incoming normal speed >= 0.20 m/s, and cumulative "
            "outward normal velocity change >= 0.30 m/s within one identity-stable episode"
        ),
        "aabb_entry_alone_is_contact": False,
    },
}
atomic_json(args.output, report)
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
