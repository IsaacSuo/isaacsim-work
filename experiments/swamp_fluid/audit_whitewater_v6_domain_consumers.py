"""Audit water-body domain ownership, core consumers and secondary handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.domain_partition import (
    load_domain_partition_contract,
    load_particle_classification,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_manifest", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--liquid-fields", type=Path, action="append", default=[])
    parser.add_argument("--splashsurf-build", type=Path, action="append", default=[])
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_binary_position_ply(path):
    data = Path(path).read_bytes()
    marker = b"end_header\n"
    offset = data.find(marker)
    if offset < 0:
        raise ValueError(f"PLY has no end_header marker: {path}")
    header = data[: offset + len(marker)].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"PLY is not binary little-endian: {path}")
    match = re.search(r"^element vertex (\d+)$", header, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"PLY vertex count is missing: {path}")
    if header.count("property float x") != 1 or header.count("property float y") != 1 or header.count("property float z") != 1:
        raise ValueError(f"PLY does not contain exactly XYZ float properties: {path}")
    count = int(match.group(1))
    payload = data[offset + len(marker) :]
    if len(payload) != count * 3 * 4:
        raise ValueError(f"PLY payload size does not match its vertex count: {path}")
    return np.frombuffer(payload, dtype="<f4").reshape((count, 3)).copy()


def parse_clip_diagnostics(text):
    values = {}
    for token in str(text).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def main():
    args = parse_args()
    if not args.liquid_fields and not args.splashsurf_build:
        raise ValueError("At least one liquid-field or Splashsurf consumer is required")
    output_report = args.output_report.resolve()
    if output_report.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output_report}")

    raw_domain = load_json(args.domain_manifest)
    source_manifest_path = Path(raw_domain["source"]["manifest"])
    source_manifest = load_json(source_manifest_path)
    contract = load_domain_partition_contract(
        args.domain_manifest,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        requested_spacing=raw_domain["configuration"]["domain_spacing"],
        body_id=(raw_domain["bodies"][0]["body_id"] if len(raw_domain["bodies"]) == 1 else None),
    )
    source_directory = Path(raw_domain["source"]["directory"])
    source_rows = {
        int(item["sample_index"]): item for item in source_manifest["samples"]
    }
    particle_count = int(source_manifest["particle_count"])
    errors = []
    classifications = {}
    for sample_index in sorted(contract["sample_rows"]):
        try:
            classifications[sample_index] = load_particle_classification(
                contract, sample_index, particle_count
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"Domain classification {sample_index}: {exc}")

    body_domain = contract["body"]["domain"]
    expected_origin = np.asarray(body_domain["origin"], dtype=np.float64)
    expected_maximum = np.asarray(body_domain["maximum"], dtype=np.float64)
    expected_shape = tuple(map(int, body_domain["shape"]))
    expected_spacing = float(body_domain["spacing"])
    liquid_reports = []
    for directory_value in args.liquid_fields:
        directory = directory_value.resolve()
        manifest_path = directory / "manifest.json"
        audit_path = directory / "audit_report.json"
        manifest = load_json(manifest_path)
        audit = load_json(audit_path)
        local_errors = []
        declared_domain = manifest.get("domain_partition") or {}
        if (
            declared_domain.get("manifest_sha256") != contract["manifest_sha256"]
            or declared_domain.get("membership_sha256") != contract["membership_sha256"]
            or declared_domain.get("body_id") != contract["body"]["body_id"]
        ):
            local_errors.append("domain provenance differs")
        grid = manifest.get("grid", {})
        if (
            tuple(map(int, grid.get("shape", ()))) != expected_shape
            or not np.array_equal(np.asarray(grid.get("origin")), expected_origin)
            or not np.isclose(float(grid.get("spacing", np.nan)), expected_spacing, rtol=0.0, atol=1.0e-12)
        ):
            local_errors.append("grid is not the selected body domain")
        if audit.get("valid") is not True or audit.get("manifest_sha256") != sha256_file(manifest_path):
            local_errors.append("independent liquid-field audit is missing or stale")
        sample_reports = []
        for item in manifest.get("samples", []):
            sample_index = int(item["source_sample_index"])
            classification = classifications.get(sample_index)
            if classification is None:
                local_errors.append(f"sample {sample_index} has no valid classification")
                continue
            core_count = len(classification["selected_core_indices"])
            secondary_count = len(classification["selected_secondary_indices"])
            if (
                int(item.get("liquid_core_input_particles", -1)) != core_count
                or int(item.get("secondary_handoff_particles", -1)) != secondary_count
                or core_count + secondary_count != len(contract["body_particle_indices"])
            ):
                local_errors.append(f"sample {sample_index} particle ledger differs")
            sample_reports.append(
                {
                    "source_sample_index": sample_index,
                    "core_particles": core_count,
                    "secondary_particles": secondary_count,
                }
            )
        errors.extend(f"{directory.name}: {value}" for value in local_errors)
        liquid_reports.append(
            {
                "directory": str(directory),
                "manifest_sha256": sha256_file(manifest_path),
                "valid": not local_errors,
                "samples": sample_reports,
                "errors": local_errors,
            }
        )

    splashsurf_reports = []
    for directory_value in args.splashsurf_build:
        directory = directory_value.resolve()
        manifest_path = directory / "splashsurf_manifest.json"
        manifest = load_json(manifest_path)
        configuration = manifest.get("configuration", {})
        state = manifest.get("state", {})
        declared_domain = configuration.get("domain_partition") or {}
        local_errors = []
        if state.get("complete") is not True:
            local_errors.append("build manifest is incomplete")
        if (
            declared_domain.get("manifest_sha256") != contract["manifest_sha256"]
            or declared_domain.get("membership_sha256") != contract["membership_sha256"]
            or declared_domain.get("body_id") != contract["body"]["body_id"]
        ):
            local_errors.append("domain provenance differs")
        if (
            configuration.get("particle_aabb_source") != "audited_water_body_domain"
            or not np.array_equal(np.asarray(configuration.get("particle_aabb_min")), expected_origin)
            or not np.array_equal(np.asarray(configuration.get("particle_aabb_max")), expected_maximum)
        ):
            local_errors.append("Splashsurf AABB is not the selected body domain")
        terrain_raster = configuration.get("terrain", {}).get("raster", {})
        terrain_raster_path = Path(terrain_raster.get("path", ""))
        if (
            terrain_raster.get("domain_mode") != "automatic_water_body_domain"
            or not terrain_raster_path.is_file()
            or sha256_file(terrain_raster_path) != terrain_raster.get("sha256")
        ):
            local_errors.append("automatic domain terrain raster is missing or stale")
        state_frames = {str(item["frame_id"]): item for item in state.get("frames", [])}
        frame_reports = []
        for frame in declared_domain.get("frames", []):
            sample_index = int(frame["source_sample_index"])
            frame_id = f"{sample_index:06d}"
            classification = classifications.get(sample_index)
            if classification is None:
                local_errors.append(f"sample {sample_index} has no valid classification")
                continue
            source_row = source_rows[sample_index]
            source_path = source_directory / source_row["file"]
            with np.load(source_path, allow_pickle=False) as source:
                source_positions = np.asarray(source["positions"], dtype=np.float32)
                source_velocities = np.asarray(source["velocities"], dtype=np.float32)
            core_indices = classification["selected_core_indices"]
            secondary_indices = classification["selected_secondary_indices"]
            if (
                len(core_indices) + len(secondary_indices)
                != len(contract["body_particle_indices"])
            ):
                local_errors.append(f"sample {sample_index} stable-body ledger is not conserved")
            core_path = Path(declared_domain["core_input_directory"]) / frame["core_ply"]
            try:
                core_positions = read_binary_position_ply(core_path)
                if (
                    sha256_file(core_path) != frame["core_ply_sha256"]
                    or not np.array_equal(core_positions, source_positions[core_indices])
                ):
                    local_errors.append(f"sample {sample_index} core PLY differs from source IDs")
            except (OSError, ValueError) as exc:
                local_errors.append(f"sample {sample_index} core PLY: {exc}")
            secondary_path = Path(declared_domain["secondary_handoff_directory"]) / frame["secondary_handoff"]
            try:
                if sha256_file(secondary_path) != frame["secondary_handoff_sha256"]:
                    raise ValueError("hash differs")
                with np.load(secondary_path, allow_pickle=False) as handoff:
                    if (
                        not np.array_equal(handoff["source_particle_indices"], secondary_indices)
                        or not np.array_equal(handoff["positions"], source_positions[secondary_indices])
                        or not np.array_equal(handoff["velocities"], source_velocities[secondary_indices])
                        or not np.array_equal(
                            handoff["particle_state"],
                            classification["state"][secondary_indices],
                        )
                        or not np.array_equal(
                            handoff["instantaneous_detached"],
                            classification["instantaneous_detached"][secondary_indices],
                        )
                        or not np.array_equal(
                            handoff["new_secondary_handoff"],
                            classification["new_secondary_handoff"][secondary_indices],
                        )
                        or not np.array_equal(
                            handoff["pending_return_to_core"],
                            classification["pending_return_to_core"][secondary_indices],
                        )
                    ):
                        raise ValueError("arrays differ from the source/classification ledger")
            except (OSError, KeyError, ValueError) as exc:
                local_errors.append(f"sample {sample_index} secondary handoff: {exc}")
            surface = state_frames.get(frame_id)
            if surface is None:
                local_errors.append(f"sample {sample_index} has no final surface record")
                diagnostics = {}
            else:
                surface_path = directory / "surface" / surface["surface_file"]
                if (
                    not surface_path.is_file()
                    or sha256_file(surface_path) != surface.get("surface_sha256")
                    or surface_path.stat().st_size != int(surface.get("surface_bytes", -1))
                ):
                    local_errors.append(f"sample {sample_index} final surface hash differs")
                diagnostics = parse_clip_diagnostics(surface.get("diagnostics", ""))
                for key in (
                    "removed_degenerate_faces",
                    "removed_duplicate_faces",
                    "nonmanifold_edges",
                    "winding_conflicts",
                ):
                    if diagnostics.get(key) != "0":
                        local_errors.append(f"sample {sample_index} topology gate {key} failed")
                if diagnostics.get("normals_preserved") != "True":
                    local_errors.append(f"sample {sample_index} normals were not preserved")
            frame_reports.append(
                {
                    "source_sample_index": sample_index,
                    "core_particles": int(len(core_indices)),
                    "secondary_particles": int(len(secondary_indices)),
                    "surface_faces": int(diagnostics.get("output_faces", 0)),
                    "surface_boundary_edges": int(diagnostics.get("boundary_edges", 0)),
                    "surface_nonmanifold_edges": int(diagnostics.get("nonmanifold_edges", -1)),
                }
            )
        errors.extend(f"{directory.name}: {value}" for value in local_errors)
        splashsurf_reports.append(
            {
                "directory": str(directory),
                "manifest_sha256": sha256_file(manifest_path),
                "valid": not local_errors,
                "frames": frame_reports,
                "errors": local_errors,
            }
        )

    criteria = {
        "domain_classifications_are_consumable": (
            len(classifications) == len(contract["sample_rows"])
        ),
        "all_consumer_provenance_and_ledgers_match": not errors,
        "all_liquid_field_consumers_pass": all(item["valid"] for item in liquid_reports),
        "all_splashsurf_consumers_pass": all(item["valid"] for item in splashsurf_reports),
    }
    report = {
        "schema": 1,
        "product": "whitewater_v6_domain_consumer_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": all(criteria.values()),
        "criteria": criteria,
        "domain_manifest": str(contract["manifest_path"]),
        "domain_manifest_sha256": contract["manifest_sha256"],
        "body_id": contract["body"]["body_id"],
        "stable_body_particles": int(len(contract["body_particle_indices"])),
        "liquid_fields": liquid_reports,
        "splashsurf_builds": splashsurf_reports,
        "errors": errors,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
