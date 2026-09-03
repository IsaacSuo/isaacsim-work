"""Independently audit deterministic v6 whitewater marker-birth caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import (
    BIRTH_DTYPE,
    CHANNEL_NAMES,
    BirthChannel,
    PhaseKind,
    marker_ids,
    sample_scalar_trilinear,
)
from whitewater.domain_partition import (
    load_domain_partition_contract,
    load_particle_classification,
)
from whitewater.secondary_handoff import (
    EXTERNAL_SOURCE_NAMESPACE,
    equal_volume_radius,
    external_particle_id,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marker_birth_directory", type=Path)
    parser.add_argument("--maximum-rejection-fraction", type=float, default=0.05)
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


def main():
    args = parse_args()
    if not 0.0 <= args.maximum_rejection_fraction < 1.0:
        raise ValueError("Maximum rejection fraction must lie within 0..1")
    manifest_path = args.marker_birth_directory / "manifest.json"
    manifest = load_json(manifest_path)
    issues = []
    criteria = {}
    if manifest.get("product") != "whitewater_v6_marker_births":
        issues.append("Unexpected marker-birth product")
    if not manifest.get("complete"):
        issues.append("Marker-birth manifest is incomplete")

    input_payload = manifest.get("inputs", {})
    liquid_manifest_path = Path(input_payload.get("liquid_manifest", ""))
    emission_manifest_path = Path(input_payload.get("emission_manifest", ""))
    for label, path, expected_hash in (
        ("liquid", liquid_manifest_path, input_payload.get("liquid_manifest_sha256")),
        (
            "emission",
            emission_manifest_path,
            input_payload.get("emission_manifest_sha256"),
        ),
    ):
        if not path.is_file():
            issues.append(f"Missing {label} input manifest: {path}")
        elif sha256_file(path) != expected_hash:
            issues.append(f"{label.capitalize()} input manifest hash changed")
    if issues:
        liquid_manifest = {"samples": []}
        emission_manifest = {"samples": [], "configuration": {}}
    else:
        liquid_manifest = load_json(liquid_manifest_path)
        emission_manifest = load_json(emission_manifest_path)

    grid = manifest["grid"]
    spec = GridSpec(
        tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"])
    )
    dt = float(manifest["configuration"]["dt_seconds"])
    contact_sample = emission_manifest.get("configuration", {}).get(
        "contact_source_sample"
    )
    model_metadata = manifest["configuration"]["model"]
    radius_settings = model_metadata["radius_distributions"]
    samples = manifest.get("samples", [])
    if len(samples) != len(liquid_manifest.get("samples", [])):
        issues.append("Marker and liquid manifests have different sample counts")

    external_configuration = manifest["configuration"].get(
        "external_liquid_handoff", {"enabled": False}
    )
    domain_contract = None
    domain_source_manifest = None
    domain_source_directory = None
    external_particle_spacing = None
    if external_configuration.get("enabled"):
        try:
            domain_manifest_path = Path(external_configuration["domain_manifest"])
            raw_domain = load_json(domain_manifest_path)
            domain_source_manifest_path = Path(raw_domain["source"]["manifest"])
            domain_source_manifest = load_json(domain_source_manifest_path)
            domain_contract = load_domain_partition_contract(
                domain_manifest_path,
                source_manifest_path=domain_source_manifest_path,
                source_manifest=domain_source_manifest,
                requested_spacing=raw_domain["configuration"]["domain_spacing"],
                body_id=external_configuration["body_id"],
            )
            domain_source_directory = Path(raw_domain["source"]["directory"])
            external_particle_spacing = float(
                external_configuration["particle_spacing"]
            )
            if domain_contract["manifest_sha256"] != external_configuration.get(
                "domain_manifest_sha256"
            ):
                raise ValueError("domain manifest hash differs")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            issues.append(f"External handoff contract is invalid: {exc}")
            domain_contract = None

    all_ids = []
    first_birth = {name: None for name in CHANNEL_NAMES.values()}
    totals = {
        name: {
            "expected_births": 0.0,
            "candidate_births": 0,
            "accepted_births": 0,
            "rejected_births": 0,
            "phase_volume_m3": 0.0,
        }
        for name in CHANNEL_NAMES.values()
    }
    maximum_phase_error = 0.0
    minimum_collision_margin = float("inf")
    maximum_spray_penetration = 0.0
    maximum_bubble_exterior = 0.0
    structural_ok = True
    hashes_ok = True
    provenance_ok = True
    spatial_ok = True
    external_handoff_ok = True
    external_births_total = 0
    external_volume_total = 0.0
    external_outside_grid_total = 0

    liquid_directory = liquid_manifest_path.parent
    emission_directory = emission_manifest_path.parent
    for sample_index, sample in enumerate(samples):
        output_path = args.marker_birth_directory / sample["file"]
        if not output_path.is_file():
            issues.append(f"Missing marker-birth file {output_path.name}")
            structural_ok = False
            continue
        if sha256_file(output_path) != sample.get("sha256"):
            issues.append(f"Marker-birth hash mismatch for {output_path.name}")
            hashes_ok = False
        with np.load(output_path) as cache:
            required = {
                "schema",
                "output_frame",
                "source_sample_index",
                "simulation_time",
                "dt",
                "births",
            }
            missing = required.difference(cache.files)
            if missing:
                issues.append(f"{output_path.name} is missing {sorted(missing)}")
                structural_ok = False
                continue
            births = np.asarray(cache["births"]).copy()
            output_frame = int(cache["output_frame"])
            source_sample = int(cache["source_sample_index"])
            simulation_time = float(cache["simulation_time"])
            cache_dt = float(cache["dt"])
            external_indices = (
                np.asarray(cache["external_source_particle_indices"], dtype=np.int64)
                if "external_source_particle_indices" in cache
                else np.empty(0, dtype=np.int64)
            )
        if births.dtype != BIRTH_DTYPE:
            issues.append(f"{output_path.name} has an unexpected birth dtype")
            structural_ok = False
            continue
        if (
            output_frame != int(sample["output_frame"])
            or source_sample != int(sample["source_sample_index"])
            or simulation_time != float(sample["simulation_time"])
            or cache_dt != dt
            or len(births) != int(sample["birth_count"])
        ):
            issues.append(f"{output_path.name} scalar metadata disagrees with manifest")
            structural_ok = False

        if sample_index >= len(liquid_manifest.get("samples", [])):
            continue
        liquid_sample = liquid_manifest["samples"][sample_index]
        emission_sample = emission_manifest["samples"][sample_index]
        if (
            source_sample != int(liquid_sample["source_sample_index"])
            or source_sample != int(emission_sample["source_sample_index"])
        ):
            issues.append(f"{output_path.name} input provenance sample mismatch")
            provenance_ok = False
            continue
        liquid_path = liquid_directory / liquid_sample["file"]
        emission_path = emission_directory / emission_sample["file"]
        with np.load(liquid_path) as liquid_cache:
            phi_field = np.asarray(liquid_cache["phi"])
            collision_field = np.asarray(liquid_cache["collision_sdf"])
        with np.load(emission_path) as emission_cache:
            emission_fields = {
                name: np.asarray(emission_cache[name]) for name in CHANNEL_NAMES.values()
            }

        external_selection = (
            births["source_node_id"] & np.uint32(EXTERNAL_SOURCE_NAMESPACE)
        ) != 0
        grid_selection = ~external_selection
        if len(births):
            numeric_names = (
                "birth_time",
                "position",
                "velocity",
                "physical_radius",
                "representative_count",
                "phase_volume",
            )
            if any(not np.isfinite(births[name]).all() for name in numeric_names):
                issues.append(f"{output_path.name} contains non-finite marker data")
                structural_ok = False
            if np.any(np.diff(births["id"]) == 0):
                issues.append(f"{output_path.name} contains duplicate IDs")
                provenance_ok = False
            if np.any(births["source_sample"] != source_sample):
                issues.append(f"{output_path.name} has a wrong record source sample")
                provenance_ok = False
            if np.any(births["birth_time"] < simulation_time - 1.0e-12) or np.any(
                births["birth_time"] > simulation_time + dt + 1.0e-12
            ):
                issues.append(f"{output_path.name} has a birth outside its source step")
                provenance_ok = False
            bounds_minimum = np.asarray(spec.origin) - 1.0e-7
            bounds_maximum = np.asarray(spec.maximum) + 1.0e-7
            grid_births = births[grid_selection]
            if np.any(grid_births["position"] < bounds_minimum) or np.any(
                grid_births["position"] > bounds_maximum
            ):
                issues.append(f"{output_path.name} contains an out-of-grid birth")
                spatial_ok = False

            sampled_phi = sample_scalar_trilinear(
                phi_field, grid_births["position"], spec
            )
            sampled_collision = sample_scalar_trilinear(
                collision_field, grid_births["position"], spec
            )
            margin = (
                sampled_collision
                - grid_births["physical_radius"]
                - 0.08 * spec.spacing
            )
            if len(margin):
                minimum_collision_margin = min(
                    minimum_collision_margin, float(np.nanmin(margin))
                )
                if np.any(~np.isfinite(margin)) or np.any(margin < -2.0e-6):
                    issues.append(
                        f"{output_path.name} contains a solid-overlapping birth"
                    )
                    spatial_ok = False
            spray_selection = grid_births["channel"] == np.uint8(BirthChannel.SPRAY)
            bubble_selection = ~spray_selection
            if np.any(spray_selection):
                maximum_spray_penetration = max(
                    maximum_spray_penetration,
                    float(np.maximum(-sampled_phi[spray_selection], 0.0).max()),
                )
                if np.any(sampled_phi[spray_selection] < -0.20 * spec.spacing - 2.0e-6):
                    issues.append(f"{output_path.name} has deeply submerged spray births")
                    spatial_ok = False
            if np.any(bubble_selection):
                maximum_bubble_exterior = max(
                    maximum_bubble_exterior,
                    float(np.maximum(sampled_phi[bubble_selection], 0.0).max()),
                )
                if np.any(sampled_phi[bubble_selection] > 0.20 * spec.spacing + 2.0e-6):
                    issues.append(f"{output_path.name} has exterior bubble births")
                    spatial_ok = False

        if domain_contract is not None:
            try:
                classification = load_particle_classification(
                    domain_contract,
                    source_sample,
                    domain_source_manifest["particle_count"],
                )
                expected_external = classification["selected_new_handoff_indices"]
                external_births = births[external_selection]
                if (
                    not np.array_equal(external_indices, expected_external)
                    or len(external_births) != len(expected_external)
                    or not np.array_equal(
                        external_particle_id(external_births["source_node_id"]),
                        expected_external,
                    )
                ):
                    raise ValueError("source particle ID set differs")
                source_rows = {
                    int(item["sample_index"]): item
                    for item in domain_source_manifest["samples"]
                }
                source_path = domain_source_directory / source_rows[source_sample]["file"]
                with np.load(source_path, allow_pickle=False) as source:
                    expected_positions = np.asarray(
                        source["positions"], dtype=np.float32
                    )[expected_external]
                    expected_velocities = np.asarray(
                        source["velocities"], dtype=np.float32
                    )[expected_external]
                expected_volume = external_particle_spacing**3
                expected_radius = equal_volume_radius(expected_volume)
                if (
                    not np.array_equal(external_births["position"], expected_positions)
                    or not np.array_equal(
                        external_births["velocity"], expected_velocities
                    )
                    or np.any(
                        external_births["phase"] != np.uint8(PhaseKind.LIQUID)
                    )
                    or np.any(
                        external_births["channel"]
                        != np.uint8(BirthChannel.SPRAY)
                    )
                    or np.any(external_births["representative_count"] != 1.0)
                    or not np.allclose(
                        external_births["phase_volume"],
                        expected_volume,
                        rtol=1.0e-7,
                    )
                    or not np.allclose(
                        external_births["physical_radius"],
                        expected_radius,
                        rtol=1.0e-7,
                    )
                ):
                    raise ValueError("physical state or liquid volume differs")
                external_births_total += len(external_births)
                external_volume_total += float(
                    external_births["phase_volume"].sum(dtype=np.float64)
                )
                external_inside = np.all(
                    (external_births["position"] >= np.asarray(spec.origin))
                    & (external_births["position"] <= np.asarray(spec.maximum)),
                    axis=1,
                )
                external_outside_grid_total += int(
                    len(external_births) - np.count_nonzero(external_inside)
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                issues.append(f"{output_path.name} external handoff mismatch: {exc}")
                external_handoff_ok = False

        all_ids.append(births["id"])
        for channel in BirthChannel:
            name = CHANNEL_NAMES[channel]
            selection = births["channel"] == np.uint8(channel)
            selected = births[selection]
            selected_grid = births[selection & grid_selection]
            expected_phase = (
                PhaseKind.LIQUID if channel == BirthChannel.SPRAY else PhaseKind.GAS
            )
            if np.any(selected["phase"] != np.uint8(expected_phase)):
                issues.append(f"{output_path.name} has a channel/phase mismatch")
                structural_ok = False
            if np.any((selected["render_class"] < 0) | (selected["render_class"] > 2)):
                issues.append(f"{output_path.name} has an invalid render class")
                structural_ok = False
            if np.any(selected["representative_count"] < 1.0) or np.any(
                selected["representative_count"] > 12.0001
            ):
                issues.append(f"{output_path.name} has an invalid representative count")
                structural_ok = False
            radius_range = radius_settings[name]["range_m"]
            if np.any(selected_grid["physical_radius"] < radius_range[0] - 1.0e-9) or np.any(
                selected_grid["physical_radius"] > radius_range[1] + 1.0e-9
            ):
                issues.append(f"{output_path.name} has an out-of-range {name} radius")
                structural_ok = False
            if len(selected):
                decoded_ids = marker_ids(
                    channel,
                    selected["source_node_id"],
                    selected["source_emission_index"],
                )
                if not np.array_equal(selected["id"], decoded_ids):
                    issues.append(f"{output_path.name} has invalid packed marker IDs")
                    provenance_ok = False
                flat_emission = emission_fields[name].reshape(-1)
                if np.any(flat_emission[selected_grid["source_node_id"]] <= 0.0):
                    issues.append(f"{output_path.name} has a zero-support {name} birth")
                    provenance_ok = False
                if len(selected_grid) and first_birth[name] is None:
                    first_birth[name] = source_sample
                computed_volume = (
                    (4.0 / 3.0)
                    * np.pi
                    * selected["physical_radius"].astype(np.float64) ** 3
                    * selected["representative_count"].astype(np.float64)
                )
                phase_error = np.max(
                    np.abs(selected["phase_volume"] - computed_volume)
                    / np.maximum(computed_volume, 1.0e-30)
                )
                maximum_phase_error = max(maximum_phase_error, float(phase_error))
            recorded = sample["channels"][name]
            actual_count = int(len(selected_grid))
            if actual_count != int(recorded["accepted_births"]):
                issues.append(f"{output_path.name} {name} count disagrees with manifest")
                structural_ok = False
            for key in totals[name]:
                totals[name][key] += recorded[key]

    concatenated_ids = np.concatenate(all_ids) if all_ids else np.empty(0, np.uint64)
    unique_ids = len(np.unique(concatenated_ids))
    global_ids_unique = unique_ids == len(concatenated_ids)
    if not global_ids_unique:
        issues.append("Marker IDs are not globally unique")
    if maximum_phase_error > 5.0e-7:
        issues.append(
            f"Marker phase-volume relative error is too high: {maximum_phase_error}"
        )
        structural_ok = False

    rate_consistency = True
    rejection_ok = True
    all_channels_activate = True
    activation_follows_contact = True
    for channel in BirthChannel:
        name = CHANNEL_NAMES[channel]
        expected = float(totals[name]["expected_births"])
        candidates = int(totals[name]["candidate_births"])
        statistical_bound = 8.0 * np.sqrt(max(expected, 1.0)) + 8.0
        if abs(candidates - expected) > statistical_bound:
            issues.append(
                f"{name} realized candidate count {candidates} is inconsistent with "
                f"expected budget {expected:.3f}"
            )
            rate_consistency = False
        rejection_fraction = totals[name]["rejected_births"] / max(candidates, 1)
        totals[name]["rejection_fraction"] = rejection_fraction
        totals[name]["candidate_minus_expected"] = candidates - expected
        if rejection_fraction > args.maximum_rejection_fraction:
            issues.append(
                f"{name} rejection fraction {rejection_fraction:.3%} exceeds gate"
            )
            rejection_ok = False
        if first_birth[name] is None:
            issues.append(f"{name} never produced an accepted marker")
            all_channels_activate = False
        if (
            contact_sample is not None
            and first_birth[name] is not None
            and first_birth[name] < int(contact_sample)
        ):
            issues.append(f"{name} produced a pre-contact marker")
            activation_follows_contact = False

    if external_configuration.get("enabled"):
        recorded_external = manifest.get("cumulative", {}).get(
            "external_liquid_handoff", {}
        )
        if (
            external_births_total != int(recorded_external.get("births", -1))
            or not np.isclose(
                external_volume_total,
                float(recorded_external.get("liquid_phase_volume_m3", np.nan)),
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            issues.append("Cumulative external liquid handoff differs from samples")
            external_handoff_ok = False

    criteria.update(
        {
            "manifest_complete": bool(manifest.get("complete")),
            "all_sample_hashes_match": hashes_ok,
            "structural_checks_pass": structural_ok,
            "provenance_checks_pass": provenance_ok,
            "spatial_phase_and_collision_checks_pass": spatial_ok,
            "global_ids_unique": global_ids_unique,
            "rate_realization_is_statistically_consistent": rate_consistency,
            "placement_rejection_within_limit": rejection_ok,
            "all_channels_activate": all_channels_activate,
            "activation_follows_contact": activation_follows_contact,
            "external_liquid_handoff_matches_domain": external_handoff_ok,
        }
    )
    report = {
        "schema": 1,
        "product": "whitewater_v6_marker_births_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "marker_birth_directory": str(args.marker_birth_directory.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "valid": not issues and all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "samples": len(samples),
            "markers": len(concatenated_ids),
            "unique_marker_ids": unique_ids,
            "contact_source_sample": contact_sample,
            "first_birth_source_sample": first_birth,
            "maximum_phase_volume_relative_error": maximum_phase_error,
            "minimum_collision_clearance_margin_m": (
                minimum_collision_margin
                if np.isfinite(minimum_collision_margin)
                else None
            ),
            "maximum_spray_liquid_penetration_m": maximum_spray_penetration,
            "maximum_bubble_exterior_distance_m": maximum_bubble_exterior,
            "external_liquid_handoff": {
                "births": external_births_total,
                "phase_volume_m3": external_volume_total,
                "births_outside_core_grid": external_outside_grid_total,
            },
            "channels": totals,
        },
        "errors": issues,
    }
    output_path = args.marker_birth_directory / "audit_report.json"
    atomic_json(output_path, report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
