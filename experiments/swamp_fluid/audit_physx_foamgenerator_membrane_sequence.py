"""Audit membrane conservation and compare remesh-independent temporal coherence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights, dtype=np.float64))
    return float(np.sum(weights[mask], dtype=np.float64) / total) if total > 0.0 else 1.0


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not len(values):
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights, dtype=np.float64)
    target = q * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, target), len(values) - 1)])


def load_sequence(directory: Path) -> tuple[dict, list[dict], list[str]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    states = []
    for sample in sorted(manifest["samples"], key=lambda item: int(item["output_frame"])):
        path = directory / sample["file"]
        if sha256_file(path) != sample["sha256"]:
            errors.append(f"frame {sample['output_frame']}: SHA-256 mismatch")
        with np.load(path, allow_pickle=False) as cache:
            vertices = np.asarray(cache["vertices"], dtype=np.float64)
            triangles = np.asarray(cache["triangles"], dtype=np.int64)
            points = vertices[triangles]
            centres = points.mean(axis=1)
            areas = 0.5 * np.linalg.norm(
                np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
                axis=1,
            )
            velocity = np.asarray(cache["triangle_advection_velocity"], dtype=np.float32)
            eligible = np.asarray(cache["eligible_patches"])
            nearest_rows = np.asarray(
                cache["nearest_eligible_patch_rows"], dtype=np.int64
            )
            source_foam = np.asarray(cache["source_foam"])
            velocity_by_id = {
                int(marker_id): value
                for marker_id, value in zip(source_foam["id"], source_foam["velocity"])
            }
            expected_velocity = np.asarray(
                [
                    velocity_by_id[int(marker_id)]
                    for marker_id in eligible["source_foam_id"][nearest_rows]
                ],
                dtype=np.float32,
            )
            if not np.array_equal(velocity, expected_velocity):
                errors.append(
                    f"frame {sample['output_frame']}: triangle velocity is not native bit-exact"
                )
            represented = float(np.sum(areas, dtype=np.float64))
            target = float(sample["metrics"]["target_membrane_area_m2"])
            if abs(represented - target) > 1.0e-10:
                errors.append(f"frame {sample['output_frame']}: area ledger mismatch")
            if not all(
                np.all(np.isfinite(array))
                for array in (centres, areas, velocity)
            ):
                errors.append(f"frame {sample['output_frame']}: non-finite geometry")
            states.append(
                {
                    "output_frame": int(sample["output_frame"]),
                    "source_frame": int(sample["source_frame"]),
                    "centres": centres,
                    "areas": areas,
                    "velocity": velocity.astype(np.float64),
                    "target_area": target,
                    "eligible_id": np.asarray(
                        eligible["source_foam_id"], dtype=np.uint64
                    ).copy(),
                    "residual_id": np.asarray(
                        cache["residual_patches"]["source_foam_id"], dtype=np.uint64
                    ).copy(),
                }
            )
    return manifest, states, errors


def temporal_metrics(states: list[dict]) -> dict:
    transitions = []
    for previous, current in zip(states, states[1:]):
        source_step = current["source_frame"] - previous["source_frame"]
        predicted = previous["centres"] + previous["velocity"] * (source_step / 120.0)
        forward = cKDTree(current["centres"]).query(predicted, k=1)[0]
        reverse = cKDTree(predicted).query(current["centres"], k=1)[0]
        previous_all = set(map(int, np.concatenate((previous["eligible_id"], previous["residual_id"]))))
        current_all = set(map(int, np.concatenate((current["eligible_id"], current["residual_id"]))))
        persistent_patch = previous_all & current_all
        previous_eligible = set(map(int, previous["eligible_id"]))
        current_eligible = set(map(int, current["eligible_id"]))
        eligibility_switches = (previous_eligible ^ current_eligible) & persistent_patch
        transitions.append(
            {
                "from_output_frame": previous["output_frame"],
                "to_output_frame": current["output_frame"],
                "source_frame_step": source_step,
                "eligible_residual_switches": len(eligibility_switches),
                "eligible_residual_switch_fraction": (
                    len(eligibility_switches) / len(persistent_patch)
                    if persistent_patch
                    else 0.0
                ),
                "previous_advected_to_current": {
                    "area_retained_within_5mm": weighted_fraction(
                        forward <= 0.005, previous["areas"]
                    ),
                    "area_retained_within_10mm": weighted_fraction(
                        forward <= 0.010, previous["areas"]
                    ),
                    "distance_area_weighted_p50_m": weighted_quantile(
                        forward, previous["areas"], 0.50
                    ),
                    "distance_area_weighted_p95_m": weighted_quantile(
                        forward, previous["areas"], 0.95
                    ),
                },
                "current_to_previous_advected": {
                    "area_supported_within_5mm": weighted_fraction(
                        reverse <= 0.005, current["areas"]
                    ),
                    "area_supported_within_10mm": weighted_fraction(
                        reverse <= 0.010, current["areas"]
                    ),
                    "distance_area_weighted_p50_m": weighted_quantile(
                        reverse, current["areas"], 0.50
                    ),
                    "distance_area_weighted_p95_m": weighted_quantile(
                        reverse, current["areas"], 0.95
                    ),
                },
            }
        )
    summary = {}
    for direction, names in (
        (
            "previous_advected_to_current",
            ("area_retained_within_5mm", "area_retained_within_10mm"),
        ),
        (
            "current_to_previous_advected",
            ("area_supported_within_5mm", "area_supported_within_10mm"),
        ),
    ):
        for name in names:
            values = [item[direction][name] for item in transitions]
            summary[f"mean_{name}"] = float(np.mean(values)) if values else 1.0
            summary[f"minimum_{name}"] = min(values, default=1.0)
    p95_values = [
        max(
            item["previous_advected_to_current"]["distance_area_weighted_p95_m"],
            item["current_to_previous_advected"]["distance_area_weighted_p95_m"],
        )
        for item in transitions
    ]
    summary["maximum_symmetric_area_weighted_p95_m"] = max(p95_values, default=0.0)
    summary["maximum_eligible_residual_switch_fraction"] = max(
        (item["eligible_residual_switch_fraction"] for item in transitions),
        default=0.0,
    )
    chatter_fractions = []
    chatter_counts = []
    for first, middle, last in zip(states, states[1:], states[2:]):
        all_sets = [
            set(map(int, np.concatenate((state["eligible_id"], state["residual_id"]))))
            for state in (first, middle, last)
        ]
        persistent = all_sets[0] & all_sets[1] & all_sets[2]
        eligibility = [set(map(int, state["eligible_id"])) for state in (first, middle, last)]
        chatter = {
            marker_id
            for marker_id in persistent
            if ((marker_id in eligibility[0]) != (marker_id in eligibility[1]))
            and ((marker_id in eligibility[0]) == (marker_id in eligibility[2]))
        }
        chatter_counts.append(len(chatter))
        chatter_fractions.append(len(chatter) / len(persistent) if persistent else 0.0)
    summary["maximum_one_frame_eligible_residual_chatter_fraction"] = max(
        chatter_fractions, default=0.0
    )
    summary["total_one_frame_eligible_residual_chatter_events"] = int(
        sum(chatter_counts)
    )
    return {"summary": summary, "transitions": transitions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_directory", type=Path)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    baseline_directory = args.baseline_directory.resolve()
    candidate_directory = args.candidate_directory.resolve()
    baseline_manifest, baseline_states, baseline_errors = load_sequence(
        baseline_directory
    )
    candidate_manifest, candidate_states, candidate_errors = load_sequence(
        candidate_directory
    )
    baseline_frames = [state["output_frame"] for state in baseline_states]
    candidate_frames = [state["output_frame"] for state in candidate_states]
    if baseline_frames != candidate_frames:
        candidate_errors.append("baseline and candidate frame sets differ")
    baseline_temporal = temporal_metrics(baseline_states)
    candidate_temporal = temporal_metrics(candidate_states)
    baseline_summary = baseline_temporal["summary"]
    candidate_summary = candidate_temporal["summary"]
    comparisons = {
        key: candidate_summary[key] - baseline_summary[key]
        for key in candidate_summary
        if key.startswith("mean_") and key in baseline_summary
    }
    retention_keys = [key for key in comparisons if "within" in key]
    no_mean_retention_regression = all(comparisons[key] >= -1.0e-9 for key in retention_keys)
    at_least_one_mean_retention_improvement = any(
        comparisons[key] > 1.0e-4 for key in retention_keys
    )
    structural_valid = (
        baseline_manifest.get("complete") is True
        and candidate_manifest.get("complete") is True
        and not baseline_errors
        and not candidate_errors
    )
    report = {
        "schema": "physx-foamgenerator-membrane-sequence-ab-audit/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": (
            structural_valid
            and no_mean_retention_regression
            and at_least_one_mean_retention_improvement
        ),
        "structural_valid": structural_valid,
        "finite_difference_velocity_used": False,
        "native_velocity_bit_exact_required": True,
        "comparison_policy": (
            "previous membrane triangle centres are advected with audited native parcel "
            "velocity, then compared to the next remeshed membrane in world space"
        ),
        "baseline": {
            "directory": str(baseline_directory),
            "manifest_sha256": sha256_file(baseline_directory / "manifest.json"),
            "errors": baseline_errors,
            **baseline_temporal,
        },
        "candidate": {
            "directory": str(candidate_directory),
            "manifest_sha256": sha256_file(candidate_directory / "manifest.json"),
            "errors": candidate_errors,
            **candidate_temporal,
        },
        "candidate_minus_baseline_mean_retention": comparisons,
        "no_mean_retention_regression": no_mean_retention_regression,
        "at_least_one_mean_retention_improvement": at_least_one_mean_retention_improvement,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "structural_valid": structural_valid,
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "candidate_minus_baseline": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
