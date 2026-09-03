"""Audit identity, exclusivity, geometry and volume of pending liquid proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pending_proxy_directory", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    directory = args.pending_proxy_directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    domain_manifest_path = Path(manifest["inputs"]["domain_manifest"])
    domain_directory = domain_manifest_path.parent
    domain = json.loads(domain_manifest_path.read_text(encoding="utf-8"))
    source_directory = Path(domain["source"]["directory"])
    particle_volume = float(manifest["configuration"]["liquid_volume_per_particle_m3"])
    errors = []
    maximum_volume_error = 0.0
    maximum_orientation_error = 0.0
    rows = 0

    if manifest.get("product") != "whitewater_v6_pending_liquid_proxy":
        errors.append("unexpected product")
    if not manifest.get("complete"):
        errors.append("manifest incomplete")
    if sha256_file(domain_manifest_path) != manifest["inputs"]["domain_manifest_sha256"]:
        errors.append("domain manifest hash mismatch")
    if len(manifest.get("samples", [])) != len(domain.get("samples", [])):
        errors.append("sample count mismatch")

    for entry, domain_sample in zip(manifest.get("samples", []), domain["samples"]):
        output_path = directory / entry["file"]
        classification_path = domain_directory / entry["classification_file"]
        source_path = source_directory / entry["source_file"]
        if sha256_file(output_path) != entry["sha256"]:
            errors.append(f"{output_path.name}: hash mismatch")
            continue
        with np.load(output_path, allow_pickle=False) as cache:
            proxies = np.asarray(cache["proxies"])
            source_sample = int(cache["source_sample_index"])
        with np.load(classification_path, allow_pickle=False) as cache:
            state = np.asarray(cache["particle_state"], dtype=np.uint8)
            handoff = np.asarray(cache["new_secondary_handoff"], dtype=np.uint8)
        pending = np.flatnonzero(state == 1)
        ids = proxies["particle_id"].astype(np.int64)
        if source_sample != int(domain_sample["sample_index"]):
            errors.append(f"{output_path.name}: source sample mismatch")
        if not np.array_equal(ids, pending):
            errors.append(f"{output_path.name}: pending identity mismatch")
        if len(ids) and np.any(handoff[ids]):
            errors.append(f"{output_path.name}: overlaps confirmed handoff")
        with np.load(source_path, allow_pickle=False) as cache:
            positions = np.asarray(cache["positions"], dtype=np.float32)[pending]
            velocities = np.asarray(cache["velocities"], dtype=np.float32)[pending]
        if not np.array_equal(proxies["position"], positions):
            errors.append(f"{output_path.name}: position provenance mismatch")
        if not np.array_equal(proxies["velocity"], velocities):
            errors.append(f"{output_path.name}: velocity provenance mismatch")
        if np.any(~np.isfinite(proxies["shape_semiaxes"])) or np.any(
            proxies["shape_semiaxes"] <= 0.0
        ):
            errors.append(f"{output_path.name}: invalid shape")
        represented = (4.0 * np.pi / 3.0) * np.prod(
            proxies["shape_semiaxes"].astype(np.float64), axis=1
        )
        if len(represented):
            maximum_volume_error = max(
                maximum_volume_error,
                float(np.max(np.abs(represented - proxies["liquid_volume"]))),
            )
            orientation_length = np.linalg.norm(
                proxies["orientation"].astype(np.float64), axis=1
            )
            maximum_orientation_error = max(
                maximum_orientation_error,
                float(np.max(np.abs(orientation_length - 1.0))),
            )
        expected_volume = len(proxies) * particle_volume
        recorded_volume = float(proxies["liquid_volume"].sum(dtype=np.float64))
        if abs(recorded_volume - expected_volume) > 1.0e-15:
            errors.append(f"{output_path.name}: liquid volume mismatch")
        if abs(recorded_volume - float(entry["owned_liquid_volume_m3"])) > 1.0e-15:
            errors.append(f"{output_path.name}: manifest volume mismatch")
        rows += len(proxies)

    if maximum_volume_error > 1.0e-12:
        errors.append("ellipsoid volume gate exceeded")
    if maximum_orientation_error > 2.0e-6:
        errors.append("orientation unit-length gate exceeded")
    report = {
        "schema": 1,
        "product": "whitewater_v6_pending_liquid_proxy_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": not errors,
        "criteria": {
            "manifest_complete": bool(manifest.get("complete")),
            "identity_matches_pending_state_exactly": not any(
                "identity" in error for error in errors
            ),
            "no_confirmed_handoff_overlap": not any(
                "handoff" in error for error in errors
            ),
            "source_positions_and_velocities_are_exact": not any(
                "provenance" in error for error in errors
            ),
            "ellipsoid_volume_is_conservative": maximum_volume_error <= 1.0e-12,
            "orientation_is_unit_length": maximum_orientation_error <= 2.0e-6,
        },
        "metrics": {
            "samples": len(manifest.get("samples", [])),
            "proxy_rows": rows,
            "maximum_simultaneous_proxies": manifest.get("cumulative", {}).get(
                "maximum_simultaneous_proxies", 0
            ),
            "maximum_ellipsoid_volume_error_m3": maximum_volume_error,
            "maximum_orientation_length_error": maximum_orientation_error,
        },
        "errors": errors,
    }
    report_path = directory / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
