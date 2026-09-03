"""Audit identity, conservation, support, tether and temporal raft metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from whitewater.state_machine import WhitewaterState
from whitewater.surface_raft import RAFT_DTYPE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_directory", type=Path)
    parser.add_argument("raft_directory", type=Path)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantiles(values):
    if not values:
        return [0.0] * 6
    return np.quantile(np.concatenate(values), (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)).tolist()


def main():
    args = parse_args()
    trajectory_manifest = load_json(args.trajectory_directory / "manifest.json")
    raft_manifest = load_json(args.raft_directory / "manifest.json")
    if trajectory_manifest.get("product") != "whitewater_v6_marker_trajectories":
        raise RuntimeError("Unexpected trajectory product")
    if raft_manifest.get("product") != "whitewater_v6_surface_raft":
        raise RuntimeError("Unexpected raft product")
    trajectory_samples = trajectory_manifest.get("samples", [])
    raft_samples = raft_manifest.get("samples", [])
    failures = []
    if not raft_manifest.get("complete"):
        failures.append("raft manifest is incomplete")
    if len(trajectory_samples) != len(raft_samples):
        failures.append("trajectory/raft sample count mismatch")
    maximum_tether = float(
        raft_manifest["configuration"]["model"]["maximum_anchor_displacement_m"]
    )
    support_threshold = float(
        raft_manifest["configuration"]["model"]["surface_projection"][
            "support_threshold"
        ]
    )
    counts = {
        "rows": 0,
        "id_mismatches": 0,
        "immutable_field_mismatches": 0,
        "support_bad": 0,
        "tether_bad": 0,
        "nonfinite": 0,
    }
    gas_residual = 0.0
    maximum_displacement = 0.0
    maximum_overlap = 0.0
    offset_step = []
    offset_acceleration = []
    target_sources = {}
    prior = None
    prior_velocity_by_id = {}
    for trajectory_sample, raft_sample in zip(trajectory_samples, raft_samples):
        if int(trajectory_sample["source_sample_index"]) != int(
            raft_sample["source_sample_index"]
        ):
            failures.append("source sample mismatch")
            break
        with np.load(
            args.trajectory_directory / trajectory_sample["file"], allow_pickle=False
        ) as cache:
            marker_rows = np.asarray(cache["markers"])
        marker_rows = marker_rows[
            marker_rows["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
        ]
        marker_rows = marker_rows[np.argsort(marker_rows["id"])]
        with np.load(args.raft_directory / raft_sample["file"], allow_pickle=False) as cache:
            raft = np.asarray(cache["raft"])
            dt = float(cache["interval_end"] - cache["interval_start"])
        if raft.dtype != RAFT_DTYPE:
            failures.append("raft dtype mismatch")
            break
        counts["rows"] += len(raft)
        if not np.array_equal(marker_rows["id"], raft["marker_id"]):
            counts["id_mismatches"] += 1
            continue
        for marker_name, raft_name in (
            ("physical_radius", "physical_radius"),
            ("representative_count", "representative_count"),
            ("phase_volume", "phase_volume"),
            ("shape", "shape"),
            ("state_age", "state_age"),
            ("random_key", "random_key"),
            ("position", "raw_anchor_position"),
        ):
            if not np.array_equal(marker_rows[marker_name], raft[raft_name]):
                counts["immutable_field_mismatches"] += 1
        marker_gas = float(marker_rows["phase_volume"].sum(dtype=np.float64))
        raft_gas = float(raft["phase_volume"].sum(dtype=np.float64))
        gas_residual += marker_gas - raft_gas
        if len(raft):
            finite_fields = (
                np.isfinite(raft["raft_position"]).all(axis=1)
                & np.isfinite(raft["raft_velocity"]).all(axis=1)
                & np.isfinite(raft["anchor_displacement"])
                & np.isfinite(raft["support_value"])
                & np.isfinite(raft["packing_fraction"])
            )
            counts["nonfinite"] += int(np.count_nonzero(~finite_fields))
            counts["support_bad"] += int(
                np.count_nonzero(raft["support_value"] < support_threshold)
            )
            counts["tether_bad"] += int(
                np.count_nonzero(
                    raft["anchor_displacement"] > maximum_tether + 2.0e-7
                )
            )
            maximum_displacement = max(
                maximum_displacement,
                float(raft["anchor_displacement"].max(initial=0.0)),
            )
        maximum_overlap = max(
            maximum_overlap, float(raft_sample["metrics"]["maximum_overlap_m"])
        )
        current_by_id = {int(row["marker_id"]): row for row in raft}
        current_velocity_by_id = {}
        if prior is not None:
            for identifier in current_by_id.keys() & prior.keys():
                current = current_by_id[identifier]
                old = prior[identifier]
                current_offset = (
                    current["raft_position"].astype(np.float64)
                    - current["raw_anchor_position"].astype(np.float64)
                )
                old_offset = (
                    old["raft_position"].astype(np.float64)
                    - old["raw_anchor_position"].astype(np.float64)
                )
                velocity = (current_offset - old_offset) / dt
                current_velocity_by_id[identifier] = velocity
                offset_step.append(np.array([np.linalg.norm(current_offset - old_offset)]))
                if identifier in prior_velocity_by_id:
                    offset_acceleration.append(
                        np.array(
                            [np.linalg.norm(velocity - prior_velocity_by_id[identifier]) / dt]
                        )
                    )
        prior = current_by_id
        prior_velocity_by_id = current_velocity_by_id
        source = int(raft_sample["source_sample_index"])
        if source in (96, 108, 120):
            cluster_size = raft["cluster_size"].astype(np.int64)
            target_sources[str(source)] = {
                "surface_bubbles": len(raft),
                "support_fallbacks": int(np.count_nonzero(raft["support_fallback"])),
                "anchor_support_repairs": int(
                    np.count_nonzero(raft["anchor_support_repair"])
                ),
                "maximum_anchor_displacement_m": float(
                    raft["anchor_displacement"].max(initial=0.0)
                ),
                "neighbor_count_quantiles": (
                    np.quantile(raft["neighbor_count"], (0, 0.5, 0.9, 0.99, 1)).tolist()
                    if len(raft)
                    else [0.0] * 5
                ),
                "packing_fraction_quantiles": (
                    np.quantile(raft["packing_fraction"], (0, 0.5, 0.9, 0.99, 1)).tolist()
                    if len(raft)
                    else [0.0] * 5
                ),
                "clusters": int(len(np.unique(raft["cluster_id"]))) if len(raft) else 0,
                "largest_cluster": int(cluster_size.max(initial=0)),
            }

    for name, value in counts.items():
        if name != "rows" and value:
            failures.append(f"{name}={value}")
    if gas_residual != 0.0:
        failures.append(f"gas_volume_residual_m3={gas_residual}")
    maximum_compression = max(
        (
            float(sample["metrics"]["maximum_contact_compression_fraction"])
            for sample in raft_samples
        ),
        default=0.0,
    )
    allowed_compression = float(
        raft_manifest["configuration"]["model"][
            "maximum_contact_compression_fraction"
        ]
    )
    excess_compression_pairs = sum(
        int(sample["metrics"]["excess_compression_pairs"])
        for sample in raft_samples
    )
    if excess_compression_pairs:
        failures.append(f"excess_compression_pairs={excess_compression_pairs}")
    report = {
        "valid": not failures,
        "failures": failures,
        "samples": len(raft_samples),
        "counts": counts,
        "gas_volume_residual_m3": gas_residual,
        "maximum_anchor_displacement_m": maximum_displacement,
        "maximum_allowed_anchor_displacement_m": maximum_tether,
        "maximum_overlap_m": maximum_overlap,
        "maximum_contact_compression_fraction": maximum_compression,
        "maximum_allowed_contact_compression_fraction": allowed_compression,
        "excess_compression_pairs": excess_compression_pairs,
        "temporal_offset_step_quantiles_m": quantiles(offset_step),
        "temporal_offset_acceleration_quantiles_m_s2": quantiles(offset_acceleration),
        "target_sources": target_sources,
    }
    report_path = args.raft_directory / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
