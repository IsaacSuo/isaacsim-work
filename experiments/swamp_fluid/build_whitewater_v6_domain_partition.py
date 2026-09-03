"""Build audited outlier-resistant domains for one or more stable PBD water bodies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.domain_partition import (
    aligned_domain_bounds,
    body_frame_core,
    reference_water_bodies,
    sha256_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--run-report", type=Path)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--end-sample", type=int)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--component-cell-particles", type=float, default=4.0)
    parser.add_argument("--minimum-body-particles", type=int, default=1024)
    parser.add_argument("--minimum-body-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-fragment-particles", type=int, default=128)
    parser.add_argument("--minimum-fragment-fraction", type=float, default=0.002)
    parser.add_argument("--maximum-unassigned-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-detached-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-outside-domain-fraction", type=float, default=0.001)
    parser.add_argument("--domain-spacing", type=float, default=0.016)
    parser.add_argument("--base-padding-particles", type=float, default=6.0)
    parser.add_argument("--motion-padding-factor", type=float, default=1.25)
    parser.add_argument("--maximum-grid-cells", type=int, default=5_000_000)
    parser.add_argument(
        "--handoff-confirmation-samples",
        type=int,
        default=2,
        help=(
            "Consecutive sampled detached states required before permanent "
            "secondary ownership. Leaving the body domain remains immediate."
        ),
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


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.sample_stride < 1 or args.start_sample < 0:
        raise ValueError("Invalid sample range or stride")
    if not 1 <= args.handoff_confirmation_samples <= 16:
        raise ValueError("handoff-confirmation-samples must lie within 1..16")
    positive = (
        args.component_cell_particles,
        args.minimum_body_particles,
        args.minimum_fragment_particles,
        args.domain_spacing,
        args.base_padding_particles,
        args.motion_padding_factor,
        args.maximum_grid_cells,
    )
    if min(positive) <= 0.0:
        raise ValueError("All size, count and padding options must be positive")
    fractions = (
        args.minimum_body_fraction,
        args.minimum_fragment_fraction,
        args.maximum_unassigned_fraction,
        args.maximum_detached_fraction,
        args.maximum_outside_domain_fraction,
    )
    if any(value < 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("All fraction options must be in [0, 1)")

    source_directory = args.source_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
    manifest_path = source_directory / "manifest.json"
    audit_path = source_directory / "audit_report.json"
    run_report_path = (
        args.run_report.resolve()
        if args.run_report is not None
        else source_directory.parent / "run_complete.json"
    )
    for path in (manifest_path, audit_path, run_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = load_json(manifest_path)
    source_audit = load_json(audit_path)
    run_report = load_json(run_report_path)
    if source_manifest.get("state", {}).get("complete") is not True:
        raise ValueError("Source manifest is incomplete")
    if source_audit.get("valid") is not True:
        raise ValueError("Source particle cache has not passed audit")
    if source_audit.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Source audit does not own the current manifest")
    if run_report.get("valid") is not True:
        raise ValueError("Parent PhysX run is not valid")
    particle_spacing = float(run_report["particle_spacing"])
    component_cell_size = args.component_cell_particles * particle_spacing
    base_padding = args.base_padding_particles * particle_spacing

    rows = {int(row["sample_index"]): row for row in source_manifest["samples"]}
    last_sample = max(rows)
    end_sample = last_sample if args.end_sample is None else args.end_sample
    if not 0 <= args.start_sample <= end_sample <= last_sample:
        raise ValueError(f"Requested samples are outside 0..{last_sample}")
    sample_indices = list(range(args.start_sample, end_sample + 1, args.sample_stride))
    if sample_indices[-1] != end_sample:
        sample_indices.append(end_sample)
    if any(index not in rows for index in sample_indices):
        raise ValueError("Source manifest does not cover every selected sample")

    reference_path = source_directory / rows[sample_indices[0]]["file"]
    with np.load(reference_path, allow_pickle=False) as cache:
        reference_positions = np.asarray(cache["positions"], dtype=np.float64)
    bodies, reference_metadata = reference_water_bodies(
        reference_positions,
        component_cell_size,
        minimum_body_particles=args.minimum_body_particles,
        minimum_body_fraction=args.minimum_body_fraction,
        connectivity=26,
    )
    membership = np.full(len(reference_positions), -1, dtype=np.int32)
    for body_index, body in enumerate(bodies):
        membership[body["particle_indices"]] = body_index

    body_minimum = [np.full(3, np.inf, dtype=np.float64) for _ in bodies]
    body_maximum = [np.full(3, -np.inf, dtype=np.float64) for _ in bodies]
    body_velocity_maximum = [np.zeros(3, dtype=np.float64) for _ in bodies]
    sample_reports = []
    maximum_detached_fraction = 0.0
    sampled_times = []
    sampled_source_hashes = []
    for sample_index in sample_indices:
        row = rows[sample_index]
        sample_path = source_directory / row["file"]
        if sha256_file(sample_path) != row["sha256"]:
            raise ValueError(f"Source sample hash mismatch: {sample_path.name}")
        sampled_source_hashes.append({"sample_index": sample_index, "sha256": row["sha256"]})
        with np.load(sample_path, allow_pickle=False) as cache:
            positions = np.asarray(cache["positions"], dtype=np.float64)
            velocities = np.asarray(cache["velocities"], dtype=np.float64)
            simulation_time = float(np.asarray(cache["simulation_time"]).item())
        if len(positions) != len(reference_positions) or velocities.shape != positions.shape:
            raise ValueError("Stable particle count/order contract changed")
        sampled_times.append(simulation_time)
        body_reports = []
        classifications = []
        for body_index, body in enumerate(bodies):
            frame = body_frame_core(
                positions,
                body["particle_indices"],
                component_cell_size,
                minimum_fragment_particles=args.minimum_fragment_particles,
                minimum_fragment_fraction=args.minimum_fragment_fraction,
                connectivity=26,
            )
            core_indices = frame.pop("core_indices")
            detached_indices = frame.pop("detached_indices")
            body_minimum[body_index] = np.minimum(
                body_minimum[body_index], positions[core_indices].min(axis=0)
            )
            body_maximum[body_index] = np.maximum(
                body_maximum[body_index], positions[core_indices].max(axis=0)
            )
            body_velocity_maximum[body_index] = np.maximum(
                body_velocity_maximum[body_index],
                np.max(np.abs(velocities[core_indices]), axis=0),
            )
            detached_fraction = len(detached_indices) / len(body["particle_indices"])
            maximum_detached_fraction = max(maximum_detached_fraction, detached_fraction)
            frame["body_id"] = body["body_id"]
            frame["detached_fraction"] = detached_fraction
            body_reports.append(frame)
            classifications.append(
                {
                    "body_index": body_index,
                    "core_indices": core_indices,
                    "detached_indices": detached_indices,
                }
            )
        sample_reports.append(
            {
                "sample_index": sample_index,
                "simulation_time": simulation_time,
                "source_file": row["file"],
                "source_sha256": row["sha256"],
                "bodies": body_reports,
                # Private build-time arrays. They are replaced by an audited
                # classification NPZ before the JSON manifest is written.
                "_classifications": classifications,
            }
        )

    maximum_time_gap = max(np.diff(sampled_times)) if len(sampled_times) > 1 else 0.0
    body_outputs = []
    for body_index, body in enumerate(bodies):
        motion_padding = (
            args.motion_padding_factor
            * maximum_time_gap
            * body_velocity_maximum[body_index]
        )
        padding = base_padding + motion_padding
        domain = aligned_domain_bounds(
            body_minimum[body_index],
            body_maximum[body_index],
            args.domain_spacing,
            padding,
        )
        body_outputs.append(
            {
                **{key: value for key, value in body.items() if key != "particle_indices"},
                "sampled_core_bounds_minimum": body_minimum[body_index].astype(float).tolist(),
                "sampled_core_bounds_maximum": body_maximum[body_index].astype(float).tolist(),
                "maximum_absolute_core_velocity": body_velocity_maximum[body_index].astype(float).tolist(),
                "temporal_motion_padding": motion_padding.astype(float).tolist(),
                "domain": domain,
            }
        )
    combined_minimum = np.min(body_minimum, axis=0)
    combined_maximum = np.max(body_maximum, axis=0)
    combined_velocity = np.max(body_velocity_maximum, axis=0)
    combined_padding = (
        base_padding
        + args.motion_padding_factor * maximum_time_gap * combined_velocity
    )
    combined_domain = aligned_domain_bounds(
        combined_minimum,
        combined_maximum,
        args.domain_spacing,
        combined_padding,
    )
    domain_minimum = np.asarray(combined_domain["origin"], dtype=np.float64)
    domain_maximum = np.asarray(combined_domain["maximum"], dtype=np.float64)
    maximum_outside_domain_fraction = 0.0
    maximum_outside_domain_particles = 0
    all_core_particles_inside_own_body_domain = True
    pending_streak = np.zeros(len(membership), dtype=np.uint16)
    secondary_owned = membership == -1
    cumulative_handoff = np.zeros(len(membership), dtype=bool)
    cumulative_handoff_events = 0
    cumulative_pending_returns = 0
    output_directory.mkdir(parents=True, exist_ok=True)
    for sample_report in sample_reports:
        sample_path = source_directory / sample_report["source_file"]
        with np.load(sample_path, allow_pickle=False) as cache:
            positions = np.asarray(cache["positions"], dtype=np.float64)
        instantaneous_state = np.full(len(positions), 2, dtype=np.uint8)
        outside_body_domain = np.zeros(len(positions), dtype=np.uint8)
        for classification, body_report, body_output in zip(
            sample_report.pop("_classifications"),
            sample_report["bodies"],
            body_outputs,
        ):
            core_indices = classification["core_indices"]
            detached_indices = classification["detached_indices"]
            body_indices = np.concatenate((core_indices, detached_indices))
            instantaneous_state[core_indices] = 0
            instantaneous_state[detached_indices] = 1
            body_domain = body_output["domain"]
            body_minimum_bound = np.asarray(body_domain["origin"], dtype=np.float64)
            body_maximum_bound = np.asarray(body_domain["maximum"], dtype=np.float64)
            body_outside = np.any(
                (positions[body_indices] < body_minimum_bound[None, :])
                | (positions[body_indices] > body_maximum_bound[None, :]),
                axis=1,
            )
            outside_indices = body_indices[body_outside]
            outside_body_domain[outside_indices] = 1
            core_outside_indices = core_indices[
                np.any(
                    (positions[core_indices] < body_minimum_bound[None, :])
                    | (positions[core_indices] > body_maximum_bound[None, :]),
                    axis=1,
                )
            ]
            all_core_particles_inside_own_body_domain &= len(core_outside_indices) == 0
            body_report["outside_body_domain_particles"] = int(len(outside_indices))
            body_report["outside_body_domain_particle_indices_sha256"] = (
                sha256_indices(outside_indices)
            )
            body_report["core_outside_body_domain_particles"] = int(
                len(core_outside_indices)
            )
            body_report["core_outside_body_domain_particle_indices_sha256"] = (
                sha256_indices(core_outside_indices)
            )
        assigned = membership >= 0
        instantaneous_detached = instantaneous_state == 1
        candidate = assigned & (
            instantaneous_detached | (outside_body_domain != 0)
        )
        was_pending = pending_streak > 0
        pending_streak[candidate & ~secondary_owned] += 1
        pending_streak[~candidate] = 0
        new_handoff = (
            assigned
            & ~secondary_owned
            & (
                (outside_body_domain != 0)
                | (pending_streak >= args.handoff_confirmation_samples)
            )
        )
        pending_return = assigned & ~secondary_owned & ~candidate & was_pending
        secondary_owned[new_handoff] = True
        cumulative_handoff[new_handoff] = True
        pending_streak[secondary_owned] = 0
        cumulative_handoff_events += int(np.count_nonzero(new_handoff))
        cumulative_pending_returns += int(np.count_nonzero(pending_return))
        particle_state = np.zeros(len(positions), dtype=np.uint8)
        particle_state[membership == -1] = 2
        particle_state[assigned & secondary_owned] = 3
        particle_state[assigned & ~secondary_owned & candidate] = 1

        outside = np.any(
            (positions < domain_minimum[None, :])
            | (positions > domain_maximum[None, :]),
            axis=1,
        )
        outside_indices = np.flatnonzero(outside)
        outside_fraction = len(outside_indices) / len(positions)
        maximum_outside_domain_fraction = max(
            maximum_outside_domain_fraction, outside_fraction
        )
        maximum_outside_domain_particles = max(
            maximum_outside_domain_particles, len(outside_indices)
        )
        sample_report["outside_combined_domain_particles"] = int(len(outside_indices))
        sample_report["outside_combined_domain_fraction"] = outside_fraction
        sample_report["outside_combined_domain_particle_indices_sha256"] = (
            sha256_indices(outside_indices)
        )
        core_indices = np.flatnonzero(particle_state == 0)
        instantaneous_detached_indices = np.flatnonzero(instantaneous_detached)
        pending_indices = np.flatnonzero(particle_state == 1)
        unassigned_indices = np.flatnonzero(particle_state == 2)
        secondary_owned_indices = np.flatnonzero(particle_state == 3)
        new_handoff_indices = np.flatnonzero(new_handoff)
        pending_return_indices = np.flatnonzero(pending_return)
        secondary_indices = np.flatnonzero(
            (particle_state != 0) | (outside_body_domain != 0)
        )
        classification_path = output_directory / (
            f"particle_classification_{sample_report['sample_index']:06d}.npz"
        )
        atomic_npz(
            classification_path,
            schema=np.int32(2),
            source_sample_index=np.int32(sample_report["sample_index"]),
            particle_state=particle_state,
            instantaneous_detached=instantaneous_detached.astype(np.uint8),
            outside_body_domain=outside_body_domain,
            outside_combined_domain=outside.astype(np.uint8),
            new_secondary_handoff=new_handoff.astype(np.uint8),
            pending_return_to_core=pending_return.astype(np.uint8),
            particle_id_contract_sha256=np.asarray(
                source_manifest["particle_ids"]["sha256"]
            ),
        )
        sample_report["classification"] = {
            "file": classification_path.name,
            "sha256": sha256_file(classification_path),
            "bytes": classification_path.stat().st_size,
            "state_codes": {
                "0": "core_owned",
                "1": "pending_detached_proxy",
                "2": "unassigned_secondary_owned",
                "3": "handed_secondary_owned",
            },
            "core_particles": int(len(core_indices)),
            "core_particle_indices_sha256": sha256_indices(core_indices),
            "instantaneous_detached_particles": int(
                len(instantaneous_detached_indices)
            ),
            "instantaneous_detached_particle_indices_sha256": sha256_indices(
                instantaneous_detached_indices
            ),
            "pending_detached_particles": int(len(pending_indices)),
            "pending_detached_particle_indices_sha256": sha256_indices(
                pending_indices
            ),
            "unassigned_particles": int(len(unassigned_indices)),
            "unassigned_particle_indices_sha256": sha256_indices(unassigned_indices),
            "secondary_owned_particles": int(len(secondary_owned_indices)),
            "secondary_owned_particle_indices_sha256": sha256_indices(
                secondary_owned_indices
            ),
            "new_secondary_handoff_particles": int(len(new_handoff_indices)),
            "new_secondary_handoff_particle_indices_sha256": sha256_indices(
                new_handoff_indices
            ),
            "pending_return_to_core_particles": int(len(pending_return_indices)),
            "pending_return_to_core_particle_indices_sha256": sha256_indices(
                pending_return_indices
            ),
            "secondary_candidate_particles": int(len(secondary_indices)),
            "secondary_candidate_particle_indices_sha256": sha256_indices(
                secondary_indices
            ),
        }
    unassigned_fraction = (
        reference_metadata["unassigned_particles"] / reference_metadata["particle_count"]
    )
    criteria = {
        "stable_particle_membership_is_complete": (
            reference_metadata["assigned_particles"]
            + reference_metadata["unassigned_particles"]
            == reference_metadata["particle_count"]
        ),
        "unassigned_reference_fraction_is_bounded": (
            unassigned_fraction <= args.maximum_unassigned_fraction
        ),
        "detached_sample_fraction_is_bounded": (
            maximum_detached_fraction <= args.maximum_detached_fraction
        ),
        "every_body_has_finite_increasing_bounds": all(
            np.isfinite(body_minimum[index]).all()
            and np.isfinite(body_maximum[index]).all()
            and np.all(body_maximum[index] > body_minimum[index])
            for index in range(len(bodies))
        ),
        "combined_domain_respects_memory_gate": (
            combined_domain["cell_count"] <= args.maximum_grid_cells
        ),
        "outside_domain_fraction_is_bounded_and_audited": (
            maximum_outside_domain_fraction <= args.maximum_outside_domain_fraction
        ),
        "all_core_particles_are_inside_their_body_domain": (
            all_core_particles_inside_own_body_domain
        ),
    }
    membership_path = output_directory / "body_membership.npz"
    atomic_npz(
        membership_path,
        schema=np.int32(1),
        particle_body_id=membership,
        reference_sample_index=np.int32(sample_indices[0]),
        particle_id_contract_sha256=np.asarray(
            source_manifest["particle_ids"]["sha256"]
        ),
    )
    report = {
        "schema": 2,
        "product": "whitewater_v6_domain_partition",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": all(criteria.values()),
        "criteria": criteria,
        "source": {
            "directory": str(source_directory),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "run_report": str(run_report_path),
            "run_report_sha256": sha256_file(run_report_path),
            "particle_id_contract": source_manifest["particle_ids"],
            "sampled_files": sampled_source_hashes,
        },
        "configuration": {
            "start_sample": args.start_sample,
            "end_sample": end_sample,
            "sample_stride": args.sample_stride,
            "sample_indices": sample_indices,
            "particle_spacing": particle_spacing,
            "component_cell_size": component_cell_size,
            "component_cell_particles": args.component_cell_particles,
            "minimum_body_particles": args.minimum_body_particles,
            "minimum_body_fraction": args.minimum_body_fraction,
            "minimum_fragment_particles": args.minimum_fragment_particles,
            "minimum_fragment_fraction": args.minimum_fragment_fraction,
            "maximum_unassigned_fraction": args.maximum_unassigned_fraction,
            "maximum_detached_fraction": args.maximum_detached_fraction,
            "maximum_outside_domain_fraction": args.maximum_outside_domain_fraction,
            "domain_spacing": args.domain_spacing,
            "base_padding": base_padding,
            "motion_padding_factor": args.motion_padding_factor,
            "maximum_time_gap": maximum_time_gap,
            "maximum_grid_cells": args.maximum_grid_cells,
            "connectivity": 26,
            "allocation": "sparse_occupied_voxels_only",
            "handoff_confirmation_samples": args.handoff_confirmation_samples,
        },
        "reference_partition": reference_metadata,
        "metrics": {
            "water_body_count": len(bodies),
            "unassigned_reference_fraction": unassigned_fraction,
            "maximum_detached_fraction": maximum_detached_fraction,
            "maximum_outside_domain_fraction": maximum_outside_domain_fraction,
            "maximum_outside_domain_particles": maximum_outside_domain_particles,
            "cumulative_secondary_handoff_events": cumulative_handoff_events,
            "unique_secondary_handoff_particles": int(
                np.count_nonzero(cumulative_handoff)
            ),
            "cumulative_pending_returns_to_core": cumulative_pending_returns,
            "combined_core_bounds_minimum": combined_minimum.astype(float).tolist(),
            "combined_core_bounds_maximum": combined_maximum.astype(float).tolist(),
            "combined_maximum_absolute_velocity": combined_velocity.astype(float).tolist(),
        },
        "bodies": body_outputs,
        "combined_domain": combined_domain,
        "samples": sample_reports,
        "membership": {
            "file": membership_path.name,
            "sha256": sha256_file(membership_path),
            "bytes": membership_path.stat().st_size,
        },
        "interpretation": {
            "core": "connected fragments above the explicit stable-body threshold",
            "detached": "audited particles that cannot enlarge a core reconstruction domain",
            "scope": "liquid reconstruction domain; spray/camera LOD envelopes are separate",
            "classification": (
                "particle_state is a mutually exclusive per-particle ledger; "
                "outside flags are overlapping routing annotations"
            ),
            "secondary_handoff": (
                "new_secondary_handoff is a one-shot ownership transfer; "
                "pending detached proxies return explicitly or become secondary-owned"
            ),
            "ownership_hysteresis": (
                "detachment must persist for the configured consecutive samples; "
                "body-domain escape transfers immediately; ownership is sticky"
            ),
        },
    }
    report_path = output_directory / "domain_manifest.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
