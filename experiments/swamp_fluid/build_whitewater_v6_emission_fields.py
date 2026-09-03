"""Build temporally coherent v6 whitewater emission fields without particles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.emission_fields import (
    CHANNEL_NAMES,
    EmissionModel,
    TemporalEmissionFilter,
    raw_emission_fields,
    suppress_precontact_floor,
)
from whitewater.flow_profiles import FlowProfile, default_impact_profile


SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--run-report", type=Path)
    parser.add_argument(
        "--flow-profile",
        type=Path,
        help=(
            "Versioned impact/river/waterfall/wake profile. If omitted, the "
            "historical impact-compatible defaults are used."
        ),
    )
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--conditioning-sigma", type=float)
    parser.add_argument("--attack-seconds", type=float)
    parser.add_argument("--decay-seconds", type=float)
    parser.add_argument("--baseline-quantile", type=float)
    parser.add_argument("--baseline-multiplier", type=float)
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


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_field_frame(path):
    with np.load(path, allow_pickle=False) as cache:
        return {name: np.asarray(cache[name]) for name in cache.files}


args = parse_args()
profile_path = args.flow_profile.resolve() if args.flow_profile is not None else None
profile = FlowProfile.load(profile_path) if profile_path is not None else default_impact_profile()
conditioning_sigma = (
    profile.conditioning_sigma_cells
    if args.conditioning_sigma is None
    else args.conditioning_sigma
)
attack_seconds = profile.attack_seconds if args.attack_seconds is None else args.attack_seconds
decay_seconds = profile.decay_seconds if args.decay_seconds is None else args.decay_seconds
baseline_quantile = (
    profile.baseline_quantile
    if args.baseline_quantile is None
    else args.baseline_quantile
)
baseline_multiplier = (
    profile.baseline_multiplier
    if args.baseline_multiplier is None
    else args.baseline_multiplier
)
if not 0.90 <= baseline_quantile < 1.0:
    raise ValueError("baseline-quantile must lie within 0.90..1.0")
if min(
    args.gravity,
    conditioning_sigma,
    attack_seconds,
    decay_seconds,
    baseline_multiplier,
) <= 0.0:
    raise ValueError("All physical/filter parameters must be positive")
if baseline_multiplier < 1.0:
    raise ValueError("baseline-multiplier must be at least 1")

field_directory = args.field_directory.resolve()
output_directory = args.output_directory.resolve()
field_manifest_path = field_directory / "manifest.json"
field_audit_path = field_directory / "audit_report.json"
for path in (field_manifest_path, field_audit_path):
    if not path.is_file():
        raise FileNotFoundError(path)
field_manifest = load_json(field_manifest_path)
field_audit = load_json(field_audit_path)
if field_manifest.get("product") != "whitewater_v6_liquid_fields":
    raise ValueError("Input is not a v6 liquid-field cache")
if field_manifest.get("state", {}).get("complete") is not True:
    raise ValueError("Input field cache is incomplete")
if field_audit.get("valid") is not True:
    raise ValueError("Input field cache has not passed its independent audit")
if field_audit.get("manifest_sha256") != sha256_file(field_manifest_path):
    raise ValueError("Input audit does not reference the current field manifest")
if output_directory.exists() and any(output_directory.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
output_directory.mkdir(parents=True, exist_ok=True)

run_report_path = (
    args.run_report.resolve()
    if args.run_report is not None
    else Path(field_manifest["source"]["run_report"])
)
run_report = load_json(run_report_path)
if run_report.get("valid") is not True:
    raise ValueError("Parent PhysX run is not valid")
particle_spacing = float(
    field_manifest.get("configuration", {}).get(
        "particle_spacing", run_report["particle_spacing"]
    )
)

rows = sorted(field_manifest["samples"], key=lambda row: int(row["source_sample_index"]))
if not rows:
    raise ValueError("Emission requires at least one liquid-field frame")
shape = tuple(int(value) for value in field_manifest["grid"]["shape"])
spacing = float(field_manifest["grid"]["spacing"])
model = EmissionModel(
    spacing,
    gravity=args.gravity,
    conditioning_sigma=conditioning_sigma,
    surface_half_width_cells=profile.surface_half_width_cells,
    churn_depth_cells=profile.churn_depth_cells,
    sphere_influence_cells=profile.churn_source_influence_cells,
    profile=profile,
)

# Prefer the canonical churn-source field. Historical caches retain the exact
# sphere/water-level contact rule through the legacy branch below.
with np.load(field_directory / rows[0]["file"], allow_pickle=False) as first_cache:
    canonical_contact_fields = "churn_source_sdf" in first_cache
contact_source_sample = int(rows[0]["source_sample_index"])
contact_output_frame = 0
if profile.event_gate == "tagged_source_contact":
    contact_source_sample = None
    contact_output_frame = None
    for output_frame, row in enumerate(rows):
        frame_path = field_directory / row["file"]
        if sha256_file(frame_path) != row["sha256"]:
            raise ValueError(f"Input field hash mismatch: {frame_path}")
        with np.load(frame_path, allow_pickle=False) as cache:
            source_index = int(cache["source_sample_index"])
            if canonical_contact_fields:
                phi = np.asarray(cache["phi"], dtype=np.float32)
                churn_source = np.asarray(cache["churn_source_sdf"], dtype=np.float32)
                contact = bool(
                    np.any(
                        (np.abs(phi) <= 1.5 * spacing)
                        & (churn_source <= particle_spacing + 0.5 * spacing)
                    )
                )
            else:
                water_level = float(run_report["water_level"]["simulated"])
                sphere_radius = float(run_report["impactor"]["radius"])
                sphere_center = np.asarray(cache["sphere_center"], dtype=np.float64)
                contact = bool(
                    sphere_center[1] - sphere_radius
                    <= water_level + particle_spacing
                )
        if contact:
            contact_source_sample = source_index
            contact_output_frame = output_frame
            break
    if contact_source_sample is None:
        raise ValueError("Selected liquid-field range does not contain tagged-source contact")
    if contact_output_frame < 1:
        raise ValueError("At least one strictly pre-contact frame is required")

# Streaming per-frame quantiles avoid retaining all pre-contact 3D channels.
if profile.event_gate == "tagged_source_contact":
    floor_candidates = {channel: [] for channel in CHANNEL_NAMES}
    for row in rows[:contact_output_frame]:
        frame_path = field_directory / row["file"]
        frame = load_field_frame(frame_path)
        raw = raw_emission_fields(frame, model)
        for channel in CHANNEL_NAMES:
            floor_candidates[channel].append(
                float(np.quantile(raw[channel + "_raw"], baseline_quantile))
            )
    floors = {
        channel: float(
            np.clip(
                baseline_multiplier * max(floor_candidates[channel]),
                0.0,
                0.95,
            )
        )
        for channel in CHANNEL_NAMES
    }
else:
    floors = dict(profile.explicit_noise_floors)

manifest_path = output_directory / "manifest.json"
manifest = {
    "schema": SCHEMA,
    "product": "whitewater_v6_emission_fields",
    "created_utc": utc_now_iso(),
    "producer": {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "model_module": str(
            (Path(__file__).parent / "whitewater" / "emission_fields.py").resolve()
        ),
        "model_module_sha256": sha256_file(
            Path(__file__).parent / "whitewater" / "emission_fields.py"
        ),
        "profile_module": str(
            (Path(__file__).parent / "whitewater" / "flow_profiles.py").resolve()
        ),
        "profile_module_sha256": sha256_file(
            Path(__file__).parent / "whitewater" / "flow_profiles.py"
        ),
    },
    "source": {
        "field_directory": str(field_directory),
        "field_manifest": str(field_manifest_path),
        "field_manifest_sha256": sha256_file(field_manifest_path),
        "field_audit": str(field_audit_path),
        "field_audit_sha256": sha256_file(field_audit_path),
        "run_report": str(run_report_path),
        "run_report_sha256": sha256_file(run_report_path),
    },
    "grid": field_manifest["grid"],
    "model": model.metadata(),
    "flow_profile": {
        "mode": "file" if profile_path is not None else "compatibility_default",
        "path": str(profile_path) if profile_path is not None else None,
        "sha256": sha256_file(profile_path) if profile_path is not None else None,
        "resolved": profile.metadata(),
    },
    "configuration": {
        "attack_seconds": attack_seconds,
        "decay_seconds": decay_seconds,
        "baseline_quantile": baseline_quantile,
        "baseline_multiplier": baseline_multiplier,
        "precontact_noise_floors": floors,
        "contact_source_sample": contact_source_sample,
        "contact_output_frame": contact_output_frame,
        "contact_definition": (
            "always enabled by the selected continuous-flow profile"
            if profile.event_gate == "always"
            else (
                "churn_source_sdf intersects the reconstructed liquid surface band"
                if canonical_contact_fields
                else "legacy sphere_center_y - sphere_radius <= "
                "simulated_water_level + particle_spacing"
            )
        ),
        "canonical_contact_fields": canonical_contact_fields,
        "event_gate": profile.event_gate,
        "strict_precontact_zero": profile.event_gate == "tagged_source_contact",
    },
    "channels": {
        "spray": "wave-crest geometry times outward motion/acceleration",
        "entrained_air": "surface convergence times rotation/inward motion",
        "churn": "near-tagged-collider subsurface inward impulse times agitation",
    },
    "samples": [],
    "state": {
        "complete": False,
        "completed_samples": 0,
        "expected_samples": len(rows),
    },
}
atomic_json(manifest_path, manifest)

temporal = TemporalEmissionFilter(
    shape,
    attack_seconds=attack_seconds,
    decay_seconds=decay_seconds,
)
previous_time = None
for output_frame, row in enumerate(rows):
    started = time.perf_counter()
    frame_path = field_directory / row["file"]
    frame = load_field_frame(frame_path)
    simulation_time = float(frame["simulation_time"])
    source_index = int(frame["source_sample_index"])
    if previous_time is None:
        dt = 1.0 / float(field_manifest["configuration"]["whitewater_sample_rate_hz"])
    else:
        dt = simulation_time - previous_time
    if dt <= 0.0:
        raise ValueError("Liquid-field times are not strictly increasing")
    raw = raw_emission_fields(frame, model)
    targets = {
        channel: suppress_precontact_floor(
            raw[channel + "_raw"], floors[channel]
        )
        for channel in CHANNEL_NAMES
    }
    event_enabled = (
        True
        if profile.event_gate == "always"
        else source_index >= contact_source_sample
    )
    filtered = temporal.advance(targets, dt, event_enabled=event_enabled)
    if not event_enabled:
        for channel in CHANNEL_NAMES:
            targets[channel].fill(0.0)

    output_path = output_directory / f"emission_fields_{output_frame:06d}.npz"
    arrays = {
        "schema": np.int32(SCHEMA),
        "output_frame": np.int32(output_frame),
        "source_sample_index": np.int32(source_index),
        "simulation_time": np.float64(simulation_time),
        "event_enabled": np.uint8(event_enabled),
    }
    arrays.update(raw)
    for channel in CHANNEL_NAMES:
        arrays[channel + "_target"] = targets[channel]
        arrays[channel] = filtered[channel]
    atomic_npz(output_path, **arrays)

    voxel_volume = spacing**3
    channel_metrics = {}
    for channel in CHANNEL_NAMES:
        raw_value = raw[channel + "_raw"]
        target = targets[channel]
        value = filtered[channel]
        channel_metrics[channel] = {
            "raw_maximum": float(np.max(raw_value)),
            "target_maximum": float(np.max(target)),
            "filtered_maximum": float(np.max(value)),
            "target_integral_m3": float(np.sum(target, dtype=np.float64) * voxel_volume),
            "filtered_integral_m3": float(np.sum(value, dtype=np.float64) * voxel_volume),
            "active_nodes_1e_4": int(np.count_nonzero(value >= 1.0e-4)),
        }
    item = {
        "output_frame": output_frame,
        "source_sample_index": source_index,
        "simulation_time": simulation_time,
        "event_enabled": event_enabled,
        "file": output_path.name,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "channels": channel_metrics,
    }
    manifest["samples"].append(item)
    manifest["state"]["completed_samples"] = len(manifest["samples"])
    atomic_json(manifest_path, manifest)
    print(
        f"[v6-emission] source={source_index:04d} contact={event_enabled} "
        + " ".join(
            f"{channel}={channel_metrics[channel]['filtered_maximum']:.4f}"
            for channel in CHANNEL_NAMES
        ),
        flush=True,
    )
    previous_time = simulation_time

manifest["state"]["complete"] = True
manifest["state"]["completed_utc"] = utc_now_iso()
atomic_json(manifest_path, manifest)
print(json.dumps(manifest["state"], indent=2))
