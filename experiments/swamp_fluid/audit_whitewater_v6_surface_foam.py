"""Independently audit persistent v6 surface-foam parcels and repellents."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import sample_scalar_trilinear
from whitewater.render_surface_support import (
    build_render_surface_support,
    load_render_surface_support_recipe,
)
from whitewater.surface_foam import (
    FOAM_DTYPE,
    REPELLENT_DTYPE,
    SurfaceFoamModel,
    source_surface_foam,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_foam_directory", type=Path)
    parser.add_argument("--maximum-foam-topology-loss-fraction", type=float, default=0.50)
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
    manifest_path = args.surface_foam_directory / "manifest.json"
    manifest = load_json(manifest_path)
    issues = []
    if manifest.get("product") != "whitewater_v6_surface_foam":
        issues.append("Unexpected surface-foam product")
    if not manifest.get("complete"):
        issues.append("Surface-foam manifest is incomplete")

    inputs = manifest["inputs"]
    liquid_manifest_path = Path(inputs["liquid_manifest"])
    trajectory_manifest_path = Path(inputs["trajectory_manifest"])
    input_hashes_ok = True
    for label, path, expected in (
        ("liquid", liquid_manifest_path, inputs["liquid_manifest_sha256"]),
        (
            "trajectory",
            trajectory_manifest_path,
            inputs["trajectory_manifest_sha256"],
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            issues.append(f"{label.capitalize()} input manifest is missing or changed")
            input_hashes_ok = False
    liquid_manifest = load_json(liquid_manifest_path)
    trajectory_manifest = load_json(trajectory_manifest_path)
    liquid_samples = liquid_manifest["samples"]
    trajectory_samples = trajectory_manifest["samples"]
    samples = manifest["samples"]
    if not (len(samples) == len(liquid_samples) == len(trajectory_samples)):
        issues.append("Surface-foam and input sample counts differ")

    grid = manifest["grid"]
    spec = GridSpec(
        tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"])
    )
    model = SurfaceFoamModel(spec.spacing)
    support_recipe = load_render_surface_support_recipe(
        manifest["configuration"]["render_surface_support"]["recipe_manifest"]
    )
    previous_foam = np.empty(0, dtype=FOAM_DTYPE)
    previous_repellents = np.empty(0, dtype=REPELLENT_DTYPE)
    seen_foam_ids = set()
    seen_repellent_ids = set()
    samples_hash_ok = True
    structural_ok = True
    identity_ok = True
    surface_ok = True
    film_balance_ok = True
    maximum_surface_distance = 0.0
    minimum_collision_sdf = float("inf")
    maximum_frame_displacement = 0.0
    maximum_direction_length_error = 0.0
    maximum_direction_normal_dot = 0.0
    maximum_frame_film_residual = 0.0
    cumulative = {
        "injected": 0.0,
        "drained": 0.0,
        "topology_lost": 0.0,
        "foam_sources": 0,
        "repellent_sources": 0,
        "topology_lost_foam": 0,
        "topology_lost_repellents": 0,
        "expired_foam": 0,
        "expired_repellents": 0,
        "low_weber_rejections": 0,
    }
    liquid_directory = liquid_manifest_path.parent
    trajectory_directory = trajectory_manifest_path.parent

    for index, sample in enumerate(samples):
        output_path = args.surface_foam_directory / sample["file"]
        if not output_path.is_file():
            issues.append(f"Missing surface-foam file {output_path.name}")
            structural_ok = False
            continue
        if sha256_file(output_path) != sample["sha256"]:
            issues.append(f"Surface-foam hash mismatch for {output_path.name}")
            samples_hash_ok = False
        with np.load(output_path) as cache:
            foam = np.asarray(cache["foam"]).copy()
            repellents = np.asarray(cache["repellents"]).copy()
            source_sample = int(cache["source_sample_index"])
            interval_start = float(cache["interval_start"])
            interval_end = float(cache["interval_end"])
        if foam.dtype != FOAM_DTYPE or repellents.dtype != REPELLENT_DTYPE:
            issues.append(f"{output_path.name} has an unexpected dtype")
            structural_ok = False
            continue
        if source_sample != int(sample["source_sample_index"]):
            issues.append(f"{output_path.name} source sample disagrees with manifest")
            structural_ok = False
        if len(foam) != int(sample["active_foam"]) or len(repellents) != int(
            sample["active_repellents"]
        ):
            issues.append(f"{output_path.name} active counts disagree with manifest")
            structural_ok = False
        if len(foam) and np.any(np.diff(foam["id"]) <= 0):
            issues.append(f"{output_path.name} foam IDs are not sorted and unique")
            identity_ok = False
        if len(repellents) and np.any(np.diff(repellents["id"]) <= 0):
            issues.append(f"{output_path.name} repellent IDs are not sorted and unique")
            identity_ok = False
        if any(
            not np.isfinite(foam[name]).all()
            for name in (
                "birth_time",
                "age",
                "position",
                "velocity",
                "film_area",
                "support_area",
                "drainage_time",
                "principal_direction",
                "anisotropy",
            )
        ):
            issues.append(f"{output_path.name} contains non-finite foam data")
            structural_ok = False
        if any(
            not np.isfinite(repellents[name]).all()
            for name in (
                "birth_time",
                "age",
                "position",
                "velocity",
                "radius",
                "strength",
                "lifetime",
            )
        ):
            issues.append(f"{output_path.name} contains non-finite repellent data")
            structural_ok = False
        if np.any(foam["film_area"] <= 0.0) or np.any(foam["support_area"] <= 0.0):
            issues.append(f"{output_path.name} contains non-positive foam areas")
            structural_ok = False
        if np.any((foam["anisotropy"] < 1.0) | (foam["anisotropy"] > 8.0001)):
            issues.append(f"{output_path.name} contains invalid anisotropy")
            structural_ok = False
        if np.any(repellents["age"] >= repellents["lifetime"]):
            issues.append(f"{output_path.name} contains expired repellents")
            structural_ok = False

        trajectory_path = trajectory_directory / trajectory_samples[index]["file"]
        with np.load(trajectory_path) as cache:
            events = np.asarray(cache["terminal_events"])
        new_foam, new_repellents, regenerated_metrics = source_surface_foam(events, model)
        recorded_source = sample["source_metrics"]
        for key, value in regenerated_metrics.items():
            recorded = recorded_source[key]
            if isinstance(value, float):
                matches = np.isclose(value, recorded, rtol=1.0e-10, atol=1.0e-16)
            else:
                matches = value == recorded
            if not matches:
                issues.append(f"{output_path.name} source metric {key} is not reproducible")
                structural_ok = False
        for identifier in new_foam["id"].tolist():
            if identifier in seen_foam_ids:
                issues.append(f"Foam source ID {identifier} was emitted twice")
                identity_ok = False
            seen_foam_ids.add(identifier)
        for identifier in new_repellents["id"].tolist():
            if identifier in seen_repellent_ids:
                issues.append(f"Repellent source ID {identifier} was emitted twice")
                identity_ok = False
            seen_repellent_ids.add(identifier)

        available_foam = np.concatenate((previous_foam["id"], new_foam["id"]))
        available_repellents = np.concatenate(
            (previous_repellents["id"], new_repellents["id"])
        )
        removed_foam = np.setdiff1d(available_foam, foam["id"])
        removed_repellents = np.setdiff1d(available_repellents, repellents["id"])
        advance = sample["advance_metrics"]
        if len(removed_foam) != int(
            advance["topology_lost_foam"] + advance["expired_foam"]
        ):
            issues.append(f"{output_path.name} foam identity sink count is inconsistent")
            identity_ok = False
        if len(removed_repellents) != int(
            advance["topology_lost_repellents"] + advance["expired_repellents"]
        ):
            issues.append(f"{output_path.name} repellent identity sink count is inconsistent")
            identity_ok = False

        previous_film = float(previous_foam["film_area"].sum(dtype=np.float64))
        current_film = float(foam["film_area"].sum(dtype=np.float64))
        frame_residual = (
            previous_film
            + float(recorded_source["injected_film_area_m2"])
            - float(advance["drained_film_area_m2"])
            - float(advance["topology_lost_film_area_m2"])
            - current_film
        )
        maximum_frame_film_residual = max(
            maximum_frame_film_residual, abs(frame_residual)
        )
        if abs(frame_residual) > max(1.0e-12, current_film * 5.0e-7):
            issues.append(f"{output_path.name} does not balance foam film area")
            film_balance_ok = False

        if len(previous_foam) and len(foam):
            shared, old_i, new_i = np.intersect1d(
                previous_foam["id"], foam["id"], return_indices=True
            )
            if len(shared):
                immutable = (
                    "source_event_id",
                    "source_kind",
                    "birth_time",
                    "drainage_time",
                    "random_key",
                )
                for name in immutable:
                    if not np.array_equal(
                        previous_foam[name][old_i], foam[name][new_i]
                    ):
                        issues.append(f"{output_path.name} changed immutable foam field {name}")
                        identity_ok = False
                displacement = np.linalg.norm(
                    foam["position"][new_i] - previous_foam["position"][old_i], axis=1
                )
                maximum_frame_displacement = max(
                    maximum_frame_displacement, float(displacement.max())
                )

        end_liquid_index = min(index + 1, len(liquid_samples) - 1)
        liquid_path = liquid_directory / liquid_samples[end_liquid_index]["file"]
        with np.load(liquid_path) as cache:
            phi = np.asarray(cache["phi"])
            normal_field = np.asarray(cache["normal"])
            collision = np.asarray(cache["collision_sdf"])
            fluid_mask = np.asarray(cache["fluid_mask"])
        render_surface_support, _ = build_render_surface_support(
            fluid_mask, spec, support_recipe
        )
        for records, label in ((foam, "foam"), (repellents, "repellent")):
            if not len(records):
                continue
            sampled_phi = sample_scalar_trilinear(phi, records["position"], spec)
            sampled_collision = sample_scalar_trilinear(
                collision, records["position"], spec
            )
            sampled_support = sample_scalar_trilinear(
                render_surface_support,
                records["position"],
                spec,
                outside=0.0,
            )
            maximum_surface_distance = max(
                maximum_surface_distance, float(np.max(np.abs(sampled_phi)))
            )
            minimum_collision_sdf = min(
                minimum_collision_sdf, float(sampled_collision.min())
            )
            if np.any(~np.isfinite(sampled_phi)) or np.any(
                np.abs(sampled_phi) > 0.004
            ):
                issues.append(f"{output_path.name} active {label} left the surface band")
                surface_ok = False
            if np.any(~np.isfinite(sampled_collision)) or np.any(
                sampled_collision < -0.002
            ):
                issues.append(f"{output_path.name} active {label} is inside a solid")
                surface_ok = False
            if np.any(sampled_support < 0.5):
                issues.append(
                    f"{output_path.name} active {label} lacks render-surface support"
                )
                surface_ok = False
        if len(foam):
            sampled_normal = np.column_stack(
                [
                    sample_scalar_trilinear(
                        normal_field[..., axis], foam["position"], spec, 0.0
                    )
                    for axis in range(3)
                ]
            )
            normal_length = np.linalg.norm(sampled_normal, axis=1)
            valid = normal_length > 0.5
            sampled_normal[valid] /= normal_length[valid, None]
            direction = foam["principal_direction"].astype(np.float64)
            direction_length = np.linalg.norm(direction, axis=1)
            maximum_direction_length_error = max(
                maximum_direction_length_error,
                float(np.max(np.abs(direction_length - 1.0))),
            )
            if np.any(valid):
                maximum_direction_normal_dot = max(
                    maximum_direction_normal_dot,
                    float(
                        np.max(
                            np.abs(
                                np.sum(direction[valid] * sampled_normal[valid], axis=1)
                            )
                        )
                    ),
                )

        cumulative["injected"] += float(recorded_source["injected_film_area_m2"])
        cumulative["drained"] += float(advance["drained_film_area_m2"])
        cumulative["topology_lost"] += float(
            advance["topology_lost_film_area_m2"]
        )
        cumulative["foam_sources"] += int(recorded_source["foam_sources"])
        cumulative["repellent_sources"] += int(
            recorded_source["repellent_sources"]
        )
        cumulative["low_weber_rejections"] += int(
            recorded_source["rejected_low_weber"]
        )
        for key in (
            "topology_lost_foam",
            "topology_lost_repellents",
            "expired_foam",
            "expired_repellents",
        ):
            cumulative[key] += int(advance[key])
        previous_foam = foam
        previous_repellents = repellents

    active_film = float(previous_foam["film_area"].sum(dtype=np.float64))
    cumulative_residual = (
        cumulative["injected"]
        - cumulative["drained"]
        - cumulative["topology_lost"]
        - active_film
    )
    if abs(cumulative_residual) > max(1.0e-12, cumulative["injected"] * 5.0e-7):
        issues.append("Cumulative foam film area is not balanced")
        film_balance_ok = False
    topology_fraction = cumulative["topology_lost_foam"] / max(
        cumulative["foam_sources"], 1
    )
    topology_loss_ok = topology_fraction <= args.maximum_foam_topology_loss_fraction
    if not topology_loss_ok:
        issues.append(
            f"Foam topology-loss fraction {topology_fraction:.3%} exceeds gate"
        )
    direction_ok = (
        maximum_direction_length_error <= 2.0e-5
        and maximum_direction_normal_dot <= 0.20
    )
    if not direction_ok:
        issues.append("Foam principal directions are not unit surface tangents")

    criteria = {
        "manifest_complete": bool(manifest.get("complete")),
        "input_manifest_hashes_match": input_hashes_ok,
        "all_sample_hashes_match": samples_hash_ok,
        "structural_checks_pass": structural_ok,
        "identity_balance_pass": identity_ok,
        "surface_attachment_and_collision_pass": surface_ok,
        "film_area_balance_pass": film_balance_ok,
        "topology_loss_within_limit": topology_loss_ok,
        "principal_directions_are_tangent": direction_ok,
    }
    report = {
        "schema": 1,
        "product": "whitewater_v6_surface_foam_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "surface_foam_directory": str(args.surface_foam_directory.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "valid": not issues and all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "samples": len(samples),
            "foam_sources": cumulative["foam_sources"],
            "repellent_sources": cumulative["repellent_sources"],
            "low_weber_rejections": cumulative["low_weber_rejections"],
            "final_active_foam": len(previous_foam),
            "final_active_repellents": len(previous_repellents),
            "injected_film_area_m2": cumulative["injected"],
            "drained_film_area_m2": cumulative["drained"],
            "topology_lost_film_area_m2": cumulative["topology_lost"],
            "active_film_area_m2": active_film,
            "film_area_balance_residual_m2": cumulative_residual,
            "maximum_frame_film_area_residual_m2": maximum_frame_film_residual,
            "topology_lost_foam": cumulative["topology_lost_foam"],
            "topology_loss_fraction": topology_fraction,
            "maximum_surface_distance_m": maximum_surface_distance,
            "minimum_collision_sdf_m": (
                minimum_collision_sdf if np.isfinite(minimum_collision_sdf) else None
            ),
            "maximum_shared_frame_displacement_m": maximum_frame_displacement,
            "maximum_principal_direction_length_error": maximum_direction_length_error,
            "maximum_principal_direction_normal_dot": maximum_direction_normal_dot,
        },
        "errors": issues,
    }
    atomic_json(args.surface_foam_directory / "audit_report.json", report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
