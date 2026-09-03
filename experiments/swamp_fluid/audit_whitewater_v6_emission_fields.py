"""Independently audit v6 continuous whitewater emission fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.emission_fields import CHANNEL_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emission_directory", type=Path)
    return parser.parse_args()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


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


args = parse_args()
emission_directory = args.emission_directory.resolve()
manifest_path = emission_directory / "manifest.json"
audit_path = emission_directory / "audit_report.json"
if audit_path.exists():
    raise FileExistsError(f"Refusing to overwrite existing audit: {audit_path}")
manifest = load_json(manifest_path)
errors = []
if manifest.get("schema") != 1 or manifest.get("product") != "whitewater_v6_emission_fields":
    errors.append("Unsupported emission manifest")
if manifest.get("state", {}).get("complete") is not True:
    errors.append("Emission manifest is incomplete")

field_manifest_path = Path(manifest["source"]["field_manifest"])
field_audit_path = Path(manifest["source"]["field_audit"])
if sha256_file(field_manifest_path) != manifest["source"]["field_manifest_sha256"]:
    errors.append("Liquid-field manifest provenance mismatch")
if sha256_file(field_audit_path) != manifest["source"]["field_audit_sha256"]:
    errors.append("Liquid-field audit provenance mismatch")
field_manifest = load_json(field_manifest_path)
field_directory = Path(manifest["source"]["field_directory"])
field_rows = {
    int(row["source_sample_index"]): row for row in field_manifest["samples"]
}
shape = tuple(int(value) for value in manifest["grid"]["shape"])
spacing = float(manifest["grid"]["spacing"])
surface_limit = 2.5 * float(manifest["model"]["surface_half_width_cells"]) * spacing
churn_source_limit = float(
    manifest["model"].get(
        "churn_source_influence_cells",
        manifest["model"]["sphere_influence_cells"],
    )
) * spacing
contact_sample = int(manifest["configuration"]["contact_source_sample"])

maximum_precontact = {channel: 0.0 for channel in CHANNEL_NAMES}
maximum_value = {channel: 0.0 for channel in CHANNEL_NAMES}
first_active = {channel: None for channel in CHANNEL_NAMES}
support_violations = {channel: 0 for channel in CHANNEL_NAMES}
channel_vectors = {channel: [] for channel in CHANNEL_NAMES}
cumulative_target_variation = {channel: 0.0 for channel in CHANNEL_NAMES}
cumulative_filtered_variation = {channel: 0.0 for channel in CHANNEL_NAMES}
previous_target = None
previous_filtered = None
audited_samples = 0

for item in manifest.get("samples", []):
    emission_path = emission_directory / item["file"]
    if not emission_path.is_file() or sha256_file(emission_path) != item["sha256"]:
        errors.append(f"Emission frame hash mismatch: {emission_path.name}")
        continue
    source_index = int(item["source_sample_index"])
    field_row = field_rows[source_index]
    field_path = field_directory / field_row["file"]
    if sha256_file(field_path) != field_row["sha256"]:
        errors.append(f"Liquid-field frame hash mismatch: {field_path.name}")
        continue
    with np.load(emission_path, allow_pickle=False) as emission, np.load(
        field_path, allow_pickle=False
    ) as field:
        phi = np.asarray(field["phi"])
        collision_sdf = np.asarray(field["collision_sdf"])
        churn_source_sdf = (
            np.asarray(field["churn_source_sdf"])
            if "churn_source_sdf" in field
            else np.asarray(field["sphere_collision_sdf"])
            if "sphere_collision_sdf" in field
            else None
        )
        target_now = {}
        filtered_now = {}
        for channel in CHANNEL_NAMES:
            raw = np.asarray(emission[channel + "_raw"])
            target = np.asarray(emission[channel + "_target"])
            filtered = np.asarray(emission[channel])
            for name, value in (("raw", raw), ("target", target), ("filtered", filtered)):
                if value.shape != shape:
                    errors.append(
                        f"{emission_path.name}:{channel}_{name} shape {value.shape}"
                    )
                if not np.isfinite(value).all() or np.min(value) < 0.0 or np.max(value) > 1.0:
                    errors.append(
                        f"{emission_path.name}:{channel}_{name} is not finite/bounded"
                    )
            maximum_value[channel] = max(maximum_value[channel], float(np.max(filtered)))
            if source_index < contact_sample:
                maximum_precontact[channel] = max(
                    maximum_precontact[channel], float(np.max(filtered))
                )
            if first_active[channel] is None and np.max(filtered) >= 1.0e-4:
                first_active[channel] = source_index
            active_raw = raw > 1.0e-8
            if channel in ("spray", "entrained_air"):
                supported = (np.abs(phi) <= surface_limit) & (collision_sdf >= 0.0)
            else:
                supported = (
                    (phi <= float(manifest["model"]["surface_half_width_cells"]) * spacing)
                    & (
                        False
                        if churn_source_sdf is None
                        else churn_source_sdf <= churn_source_limit
                    )
                    & (collision_sdf >= 0.0)
                )
            support_violations[channel] += int(np.count_nonzero(active_raw & ~supported))
            if source_index >= contact_sample:
                channel_vectors[channel].append(filtered.ravel())
            target_now[channel] = target
            filtered_now[channel] = filtered
        if previous_target is not None:
            for channel in CHANNEL_NAMES:
                cumulative_target_variation[channel] += float(
                    np.sum(np.abs(target_now[channel] - previous_target[channel]), dtype=np.float64)
                )
                cumulative_filtered_variation[channel] += float(
                    np.sum(
                        np.abs(filtered_now[channel] - previous_filtered[channel]),
                        dtype=np.float64,
                    )
                )
        previous_target = {key: value.copy() for key, value in target_now.items()}
        previous_filtered = {key: value.copy() for key, value in filtered_now.items()}
    audited_samples += 1

distinct_differences = {}
for index, first in enumerate(CHANNEL_NAMES):
    first_values = np.concatenate(channel_vectors[first]) if channel_vectors[first] else np.empty(0)
    for second in CHANNEL_NAMES[index + 1 :]:
        second_values = (
            np.concatenate(channel_vectors[second]) if channel_vectors[second] else np.empty(0)
        )
        key = first + "_vs_" + second
        distinct_differences[key] = (
            float(np.max(np.abs(first_values - second_values), initial=0.0))
            if len(first_values) == len(second_values)
            else 0.0
        )

criteria = {
    "structural_checks_pass": not errors,
    "all_samples_audited": audited_samples == len(manifest.get("samples", [])),
    "strict_precontact_zero": max(maximum_precontact.values()) == 0.0,
    "all_channels_activate": all(value is not None for value in first_active.values()),
    "activation_follows_contact": all(
        value is None or value >= contact_sample for value in first_active.values()
    ),
    "raw_channels_respect_support": sum(support_violations.values()) == 0,
    "channels_are_not_identical": all(value >= 1.0e-4 for value in distinct_differences.values()),
    "temporal_filter_reduces_total_variation": all(
        cumulative_filtered_variation[channel]
        <= 1.05 * cumulative_target_variation[channel] + 1.0e-9
        for channel in CHANNEL_NAMES
    ),
}
report = {
    "schema": 1,
    "product": "whitewater_v6_emission_fields_audit",
    "created_utc": utc_now_iso(),
    "emission_directory": str(emission_directory),
    "manifest_sha256": sha256_file(manifest_path),
    "valid": bool(all(criteria.values())),
    "criteria": {key: bool(value) for key, value in criteria.items()},
    "metrics": {
        "samples": audited_samples,
        "contact_source_sample": contact_sample,
        "maximum_precontact": maximum_precontact,
        "maximum_filtered": maximum_value,
        "first_active_source_sample": first_active,
        "support_violations": support_violations,
        "channel_maximum_absolute_differences": distinct_differences,
        "cumulative_target_variation": cumulative_target_variation,
        "cumulative_filtered_variation": cumulative_filtered_variation,
    },
    "errors": errors,
}
atomic_json(audit_path, report)
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
