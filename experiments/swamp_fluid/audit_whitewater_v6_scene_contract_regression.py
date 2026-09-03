"""Audit legacy-adapter, explicit-contract and baseline liquid-field equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PHYSICAL_FIELDS = (
    "particle_weight",
    "number_density",
    "fluid_mask",
    "phi",
    "depth",
    "velocity",
    "velocity_valid",
    "normal",
    "curvature",
    "surface_valid",
    "divergence",
    "vorticity",
    "strain_rate",
    "acceleration",
    "acceleration_valid",
    "collision_sdf",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_directory", type=Path)
    parser.add_argument("explicit_directory", type=Path)
    parser.add_argument("baseline_directory", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--source-sample", type=int, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(directory):
    path = Path(directory) / "manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def find_sample(directory, manifest, source_sample):
    matches = [
        row
        for row in manifest["samples"]
        if int(row["source_sample_index"]) == source_sample
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source {source_sample} sample in {directory}, got {len(matches)}"
        )
    path = Path(directory) / matches[0]["file"]
    if sha256_file(path) != matches[0]["sha256"]:
        raise ValueError(f"Frame hash mismatch: {path}")
    return path


def compare_arrays(first, second, fields):
    rows = {}
    valid = True
    for field in fields:
        if field not in first or field not in second:
            rows[field] = {"equal": False, "reason": "missing"}
            valid = False
            continue
        a = np.asarray(first[field])
        b = np.asarray(second[field])
        equal = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        rows[field] = {
            "equal": bool(equal),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "maximum_absolute_difference": (
                float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))
                if a.shape == b.shape
                else None
            ),
        }
        valid &= bool(equal)
    return valid, rows


args = parse_args()
legacy_directory = args.legacy_directory.resolve()
explicit_directory = args.explicit_directory.resolve()
baseline_directory = args.baseline_directory.resolve()
output_report = args.output_report.resolve()
if output_report.exists():
    raise FileExistsError(f"Refusing to overwrite regression report: {output_report}")

legacy_manifest_path, legacy_manifest = load_manifest(legacy_directory)
explicit_manifest_path, explicit_manifest = load_manifest(explicit_directory)
baseline_manifest_path, baseline_manifest = load_manifest(baseline_directory)
legacy_path = find_sample(legacy_directory, legacy_manifest, args.source_sample)
explicit_path = find_sample(explicit_directory, explicit_manifest, args.source_sample)
baseline_path = find_sample(baseline_directory, baseline_manifest, args.source_sample)

errors = []
if legacy_manifest.get("scene_contract", {}).get("mode") != "legacy_adapter":
    errors.append("Legacy build does not declare legacy_adapter mode")
if explicit_manifest.get("scene_contract", {}).get("mode") != "file":
    errors.append("Explicit build does not declare file scene-contract mode")
for directory, manifest in (
    (legacy_directory, legacy_manifest),
    (explicit_directory, explicit_manifest),
):
    audit_path = directory / "audit_report.json"
    if not audit_path.is_file():
        errors.append(f"Missing independent liquid-field audit: {audit_path}")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("valid") is not True:
            errors.append(f"Liquid-field audit failed: {audit_path}")
        if audit.get("manifest_sha256") != sha256_file(directory / "manifest.json"):
            errors.append(f"Liquid-field audit provenance mismatch: {audit_path}")

with np.load(legacy_path, allow_pickle=False) as legacy, np.load(
    explicit_path, allow_pickle=False
) as explicit, np.load(baseline_path, allow_pickle=False) as baseline:
    explicit_has_legacy_alias = "sphere_collision_sdf" in explicit
    legacy_explicit_valid, legacy_explicit = compare_arrays(
        legacy, explicit, PHYSICAL_FIELDS
    )
    legacy_baseline_valid, legacy_baseline = compare_arrays(
        legacy, baseline, PHYSICAL_FIELDS
    )
    role_equivalence = {}
    for name in (
        "collider_collision_sdf",
        "dynamic_collision_sdf",
        "churn_source_sdf",
        "dynamic_churn_source_sdf",
    ):
        equal = name in explicit and np.array_equal(
            np.asarray(explicit[name]), np.asarray(legacy["sphere_collision_sdf"])
        )
        role_equivalence[name] = bool(equal)
        if not equal:
            errors.append(f"Explicit role field does not match legacy impactor: {name}")
    if explicit_has_legacy_alias:
        errors.append("Explicit scene-contract output leaked the legacy sphere alias")
    if "sphere_collision_sdf" not in legacy:
        errors.append("Legacy adapter output omitted its compatibility alias")

if not legacy_explicit_valid:
    errors.append("Legacy and explicit scene-contract physical fields differ")
if not legacy_baseline_valid:
    errors.append("Migrated legacy build and locked production baseline differ")

payload = {
    "schema": 1,
    "product": "whitewater_v6_scene_contract_regression_audit",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": not errors,
    "source_sample_index": args.source_sample,
    "errors": errors,
    "inputs": {
        "legacy_manifest": str(legacy_manifest_path),
        "legacy_manifest_sha256": sha256_file(legacy_manifest_path),
        "explicit_manifest": str(explicit_manifest_path),
        "explicit_manifest_sha256": sha256_file(explicit_manifest_path),
        "baseline_manifest": str(baseline_manifest_path),
        "baseline_manifest_sha256": sha256_file(baseline_manifest_path),
    },
    "criteria": {
        "independent_liquid_audits_passed": not any(
            "liquid-field audit" in error.lower() for error in errors
        ),
        "legacy_and_explicit_fields_bitwise_equal": legacy_explicit_valid,
        "legacy_and_production_baseline_bitwise_equal": legacy_baseline_valid,
        "collider_roles_match_legacy_impactor": all(role_equivalence.values()),
        "legacy_alias_is_compatibility_only": (
            not explicit_has_legacy_alias
        ),
    },
    "role_equivalence": role_equivalence,
    "legacy_vs_explicit": legacy_explicit,
    "legacy_vs_production_baseline": legacy_baseline,
}
output_report.parent.mkdir(parents=True, exist_ok=True)
temporary = output_report.with_suffix(output_report.suffix + ".tmp")
with temporary.open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output_report)
print(json.dumps({"valid": payload["valid"], "errors": errors}, indent=2))
if errors:
    raise SystemExit(1)
