"""Audit conservative multi-scale v6 foam render primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_render_payload import PATCH_DTYPE, RING_DTYPE, FoamRenderClass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_payload_directory", type=Path)
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
    manifest_path = args.render_payload_directory / "manifest.json"
    manifest = load_json(manifest_path)
    issues = []
    if manifest.get("product") != "whitewater_v6_foam_render_payload":
        issues.append("Unexpected render-payload product")
    if not manifest.get("complete"):
        issues.append("Render-payload manifest is incomplete")
    inputs = manifest["inputs"]
    input_hashes_ok = True
    for label, path_text, expected in (
        ("liquid", inputs["liquid_manifest"], inputs["liquid_manifest_sha256"]),
        (
            "surface foam",
            inputs["surface_foam_manifest"],
            inputs["surface_foam_manifest_sha256"],
        ),
    ):
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected:
            issues.append(f"{label.capitalize()} input manifest is missing or changed")
            input_hashes_ok = False

    hashes_ok = True
    structural_ok = True
    area_ok = True
    tangent_ok = True
    global_patch_ids = []
    global_ring_ids = []
    maximum_frame_residual = 0.0
    maximum_normal_error = 0.0
    maximum_tangent_dot = 0.0
    maximum_ring_coverage = 0.0
    minimum_ring_enclosure = 1.0
    cumulative = {
        "source": 0.0,
        "patch": 0.0,
        "ring": 0.0,
        "affected": 0,
        "patch_rows": 0,
        "ring_rows": 0,
        "classes": {"micro_density": 0, "bubble_cluster": 0, "macro_patch": 0},
        "embedded_rings": 0,
        "orphan_repellents": 0,
        "ring_hosted_patch_rows": 0,
    }
    model = manifest["configuration"]["model"]
    for sample in manifest["samples"]:
        path = args.render_payload_directory / sample["file"]
        if not path.is_file():
            issues.append(f"Missing render-payload file {path.name}")
            structural_ok = False
            continue
        if sha256_file(path) != sample["sha256"]:
            issues.append(f"Render-payload hash mismatch for {path.name}")
            hashes_ok = False
        with np.load(path) as cache:
            patches = np.asarray(cache["patches"]).copy()
            rings = np.asarray(cache["rings"]).copy()
        if patches.dtype != PATCH_DTYPE or rings.dtype != RING_DTYPE:
            issues.append(f"{path.name} has an unexpected structured dtype")
            structural_ok = False
            continue
        if len(patches) != sample["patch_count"] or len(rings) != sample["ring_count"]:
            issues.append(f"{path.name} primitive counts disagree with manifest")
            structural_ok = False
        if len(patches) and np.any(np.diff(patches["id"]) <= 0):
            issues.append(f"{path.name} patch IDs are not sorted and unique")
            structural_ok = False
        if len(rings) and np.any(np.diff(rings["id"]) <= 0):
            issues.append(f"{path.name} ring IDs are not sorted and unique")
            structural_ok = False
        for records, names in (
            (
                patches,
                (
                    "position",
                    "normal",
                    "principal_direction",
                    "major_radius",
                    "minor_radius",
                    "film_area",
                    "support_area",
                    "coverage",
                    "hole_fraction",
                    "cluster_id",
                    "cluster_size",
                ),
            ),
            (
                rings,
                (
                    "position",
                    "normal",
                    "inner_radius",
                    "outer_radius",
                    "film_area",
                    "coverage",
                    "strength",
                    "host_cluster_id",
                    "host_patch_count",
                    "enclosure_fraction",
                ),
            ),
        ):
            if any(not np.isfinite(records[name]).all() for name in names):
                issues.append(f"{path.name} contains non-finite primitive data")
                structural_ok = False
        if np.any(patches["major_radius"] < patches["minor_radius"]):
            issues.append(f"{path.name} has inverted patch axes")
            structural_ok = False
        if np.any((patches["coverage"] < 0.0) | (patches["coverage"] > 1.0)):
            issues.append(f"{path.name} has invalid patch coverage")
            structural_ok = False
        if np.any((patches["hole_fraction"] < 0.0) | (patches["hole_fraction"] > 1.0)):
            issues.append(f"{path.name} has invalid hole fractions")
            structural_ok = False
        if np.any(rings["outer_radius"] <= rings["inner_radius"]):
            issues.append(f"{path.name} has non-positive ring widths")
            structural_ok = False
        if len(patches):
            unique_cluster, cluster_counts = np.unique(
                patches["cluster_id"], return_counts=True
            )
            expected_sizes = dict(zip(unique_cluster.tolist(), cluster_counts.tolist()))
            if any(
                int(record["cluster_size"])
                != expected_sizes[int(record["cluster_id"])]
                for record in patches
            ):
                issues.append(f"{path.name} has inconsistent patch cluster sizes")
                structural_ok = False
        if len(rings):
            patch_cluster_ids = set(patches["cluster_id"].tolist())
            if any(
                int(value) not in patch_cluster_ids
                for value in rings["host_cluster_id"]
            ):
                issues.append(f"{path.name} has a ring without a host cluster")
                structural_ok = False
            host_cluster_sizes = {
                int(value): int(np.count_nonzero(patches["cluster_id"] == value))
                for value in np.unique(rings["host_cluster_id"])
            }
            if any(
                int(record["host_patch_count"]) < 1
                or int(record["host_patch_count"])
                > host_cluster_sizes[int(record["host_cluster_id"])]
                for record in rings
            ):
                issues.append(f"{path.name} has invalid ring host counts")
                structural_ok = False
            if np.any(
                rings["enclosure_fraction"]
                < float(model["minimum_enclosure_fraction"]) - 1.0e-6
            ):
                issues.append(f"{path.name} has a ring outside enclosed host film")
                structural_ok = False
            ring_width = rings["outer_radius"] - rings["inner_radius"]
            if np.any(
                ring_width < float(model["minimum_ring_width_m"]) - 1.0e-8
            ) or np.any(
                ring_width > float(model["maximum_ring_width_m"]) + 1.0e-8
            ):
                issues.append(f"{path.name} has a nonphysical ring width")
                structural_ok = False
        if np.any((rings["coverage"] <= 0.0) | (rings["coverage"] > 0.950001)):
            issues.append(f"{path.name} has invalid ring coverage")
            structural_ok = False
        if len(patches):
            normal_length = np.linalg.norm(patches["normal"], axis=1)
            direction_length = np.linalg.norm(patches["principal_direction"], axis=1)
            maximum_normal_error = max(
                maximum_normal_error,
                float(np.max(np.abs(normal_length - 1.0))),
                float(np.max(np.abs(direction_length - 1.0))),
            )
            maximum_tangent_dot = max(
                maximum_tangent_dot,
                float(
                    np.max(
                        np.abs(
                            np.sum(
                                patches["normal"]
                                * patches["principal_direction"],
                                axis=1,
                            )
                        )
                    )
                ),
            )
            expected_class = np.full(
                len(patches), FoamRenderClass.BUBBLE_CLUSTER, dtype=np.uint8
            )
            equivalent = np.sqrt(patches["support_area"] / np.pi)
            expected_class[equivalent < float(model["micro_radius_m"])] = FoamRenderClass.MICRO_DENSITY
            expected_class[equivalent >= float(model["macro_radius_m"])] = FoamRenderClass.MACRO_PATCH
            if not np.array_equal(patches["render_class"], expected_class):
                issues.append(f"{path.name} render classes do not match support scale")
                structural_ok = False
        if len(rings):
            maximum_ring_coverage = max(
                maximum_ring_coverage, float(rings["coverage"].max())
            )
            minimum_ring_enclosure = min(
                minimum_ring_enclosure,
                float(rings["enclosure_fraction"].min()),
            )
        if len(patches):
            effective_support = patches["support_area"].astype(np.float64) * np.maximum(
                1.0 - patches["hole_fraction"].astype(np.float64), 1.0e-6
            )
            expected_coverage = np.clip(
                patches["film_area"].astype(np.float64) / effective_support,
                0.0,
                1.0,
            )
            if not np.allclose(
                patches["coverage"], expected_coverage, rtol=2.0e-6, atol=2.0e-7
            ):
                issues.append(f"{path.name} patch coverage ignores its holes")
                structural_ok = False
        metric = sample["metrics"]
        source_area = float(metric["source_film_area_m2"])
        patch_area = float(patches["film_area"].sum(dtype=np.float64))
        ring_area = float(rings["film_area"].sum(dtype=np.float64))
        residual = source_area - patch_area - ring_area
        maximum_frame_residual = max(maximum_frame_residual, abs(residual))
        if abs(residual) > max(1.0e-12, source_area * 5.0e-7):
            issues.append(f"{path.name} does not conserve render film area")
            area_ok = False
        if int(np.count_nonzero(patches["hole_fraction"] > 0.0)) != int(
            metric["affected_patches"]
        ):
            issues.append(f"{path.name} affected-patch metric is inconsistent")
            structural_ok = False
        global_patch_ids.append(patches["id"])
        global_ring_ids.append(rings["id"])
        cumulative["source"] += source_area
        cumulative["patch"] += patch_area
        cumulative["ring"] += ring_area
        cumulative["affected"] += int(metric["affected_patches"])
        cumulative["patch_rows"] += len(patches)
        cumulative["ring_rows"] += len(rings)
        cumulative["embedded_rings"] += int(metric.get("embedded_rings", 0))
        cumulative["orphan_repellents"] += int(metric.get("orphan_repellents", 0))
        cumulative["ring_hosted_patch_rows"] += int(
            metric.get("ring_hosted_patch_rows", 0)
        )
        for key, value in metric.get("render_class_counts", {}).items():
            cumulative["classes"][key] += int(value)

    if maximum_normal_error > 2.0e-5 or maximum_tangent_dot > 2.0e-5:
        issues.append("Patch normals/directions are not unit surface frames")
        tangent_ok = False
    cumulative_residual = cumulative["source"] - cumulative["patch"] - cumulative["ring"]
    if abs(cumulative_residual) > max(1.0e-12, cumulative["source"] * 5.0e-7):
        issues.append("Cumulative render film area is not conserved")
        area_ok = False
    repellents_effective = (
        cumulative["affected"] > 0
        and cumulative["ring"] > 0.0
        and cumulative["embedded_rings"] == cumulative["ring_rows"]
    )
    if not repellents_effective:
        issues.append("Repellents never created conservative holes/rings")
    all_classes_present = all(value > 0 for value in cumulative["classes"].values())
    if not all_classes_present:
        issues.append("One or more foam render scales are absent")

    criteria = {
        "manifest_complete": bool(manifest.get("complete")),
        "input_manifest_hashes_match": input_hashes_ok,
        "all_sample_hashes_match": hashes_ok,
        "structural_checks_pass": structural_ok,
        "surface_frames_are_orthonormal": tangent_ok,
        "film_area_is_conserved": area_ok,
        "repellents_create_hosted_holes_and_rims": repellents_effective,
        "all_render_scales_present": all_classes_present,
    }
    report = {
        "schema": 1,
        "product": "whitewater_v6_foam_render_payload_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "render_payload_directory": str(args.render_payload_directory.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "valid": not issues and all(criteria.values()),
        "criteria": criteria,
        "metrics": {
            "samples": len(manifest["samples"]),
            "patch_rows": cumulative["patch_rows"],
            "ring_rows": cumulative["ring_rows"],
            "affected_patch_rows": cumulative["affected"],
            "render_class_counts": cumulative["classes"],
            "source_film_area_m2": cumulative["source"],
            "patch_film_area_m2": cumulative["patch"],
            "ring_film_area_m2": cumulative["ring"],
            "area_balance_residual_m2": cumulative_residual,
            "maximum_frame_area_residual_m2": maximum_frame_residual,
            "maximum_surface_frame_length_error": maximum_normal_error,
            "maximum_surface_frame_tangent_dot": maximum_tangent_dot,
            "maximum_ring_coverage": maximum_ring_coverage,
            "minimum_ring_enclosure": (
                minimum_ring_enclosure if cumulative["ring_rows"] else None
            ),
            "embedded_rings": cumulative["embedded_rings"],
            "orphan_repellents": cumulative["orphan_repellents"],
            "ring_hosted_patch_rows": cumulative["ring_hosted_patch_rows"],
        },
        "errors": issues,
    }
    atomic_json(args.render_payload_directory / "audit_report.json", report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
