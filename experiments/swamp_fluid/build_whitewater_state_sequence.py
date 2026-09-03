"""Build a conservative unified whitewater-marker sequence from audited fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.state_machine import (
    DeterministicEmissionReservoir,
    EmissionChannel,
    MARKER_DTYPE,
    WhitewaterState,
    bubble_shape,
    bubble_terminal_velocity,
    counter_keys,
    make_markers,
    sphere_volume,
    transition_state,
    uniform01_from_keys,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_directory", type=Path)
parser.add_argument("feature_directory", type=Path)
parser.add_argument("output_directory", type=Path)
parser.add_argument("--end-sample", type=int, default=120)
parser.add_argument("--seed", type=int, default=240812)
parser.add_argument("--surface-cell-size", type=float, default=0.016)
parser.add_argument("--maximum-surface-projection-speed", type=float, default=1.5)
parser.add_argument("--spray-volume-coefficient", type=float, default=0.002)
parser.add_argument("--air-volume-coefficient", type=float, default=0.002)
parser.add_argument("--spray-packet-radius", type=float, default=0.001)
parser.add_argument("--air-packet-radius", type=float, default=0.001)
parser.add_argument("--air-packet-weight", type=float, default=8.0)
args = parser.parse_args()

if args.end_sample < 1:
    raise ValueError("end-sample must be positive")
if min(
    args.surface_cell_size,
    args.maximum_surface_projection_speed,
    args.spray_volume_coefficient,
    args.air_volume_coefficient,
    args.spray_packet_radius,
    args.air_packet_radius,
    args.air_packet_weight,
) <= 0.0:
    raise ValueError("All physical scale and volume parameters must be positive")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def smoothstep01(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def sample_height_nearest(points, x_values, z_values, values):
    dx = float(np.median(np.diff(x_values)))
    dz = float(np.median(np.diff(z_values)))
    ix = np.rint((points[:, 0] - x_values[0]) / dx).astype(np.int64)
    iz = np.rint((points[:, 2] - z_values[0]) / dz).astype(np.int64)
    valid = (ix >= 0) & (ix < len(x_values)) & (iz >= 0) & (iz < len(z_values))
    sampled = np.full(len(points), np.nan, dtype=np.float32)
    sampled[valid] = values[ix[valid], iz[valid]]
    valid &= np.isfinite(sampled)
    return sampled, valid


def sample_surface_nearest(points, x_values, z_values, surface_layers):
    low, low_valid = sample_height_nearest(
        points, x_values, z_values, surface_layers[0]
    )
    high, high_valid = sample_height_nearest(
        points, x_values, z_values, surface_layers[1]
    )
    selected = np.full(len(points), np.nan, dtype=np.float32)
    both = low_valid & high_valid
    choose_low = both & (
        np.abs(points[:, 1] - low) <= np.abs(points[:, 1] - high)
    )
    choose_high = both & ~choose_low
    selected[choose_low] = low[choose_low]
    selected[choose_high] = high[choose_high]
    selected[low_valid & ~high_valid] = low[low_valid & ~high_valid]
    selected[high_valid & ~low_valid] = high[high_valid & ~low_valid]
    return selected, low_valid | high_valid


def dynamic_surface_field(
    positions,
    confidence,
    neighbor_count,
    x_values,
    z_values,
    water_level,
):
    dx = float(np.median(np.diff(x_values)))
    dz = float(np.median(np.diff(z_values)))
    ix = np.rint((positions[:, 0] - x_values[0]) / dx).astype(np.int64)
    iz = np.rint((positions[:, 2] - z_values[0]) / dz).astype(np.int64)
    inside = (ix >= 0) & (ix < len(x_values)) & (iz >= 0) & (iz < len(z_values))
    carrier = (
        inside
        & (confidence >= 0.1)
        & (neighbor_count >= 20)
        & (positions[:, 1] <= water_level + 0.20)
    )
    shape = (len(x_values), len(z_values))
    low = np.full(shape, np.inf, dtype=np.float32)
    high = np.full(shape, -np.inf, dtype=np.float32)
    np.minimum.at(low, (ix[carrier], iz[carrier]), positions[carrier, 1])
    np.maximum.at(high, (ix[carrier], iz[carrier]), positions[carrier, 1])
    for _ in range(2):
        expanded_low = low.copy()
        expanded_high = high.copy()
        for ox in (-1, 0, 1):
            for oz in (-1, 0, 1):
                sx = slice(max(0, -ox), min(len(x_values), len(x_values) - ox))
                sz = slice(max(0, -oz), min(len(z_values), len(z_values) - oz))
                tx = slice(max(0, ox), min(len(x_values), len(x_values) + ox))
                tz = slice(max(0, oz), min(len(z_values), len(z_values) + oz))
                expanded_low[tx, tz] = np.minimum(expanded_low[tx, tz], low[sx, sz])
                expanded_high[tx, tz] = np.maximum(expanded_high[tx, tz], high[sx, sz])
        low, high = expanded_low, expanded_high
    low[~np.isfinite(low)] = np.nan
    high[~np.isfinite(high)] = np.nan
    return np.stack((low, high), axis=0)


def marker_snapshot(markers):
    alive = markers["state"] != np.uint8(WhitewaterState.DEAD)
    selected = markers[alive]
    return {name: np.ascontiguousarray(selected[name]) for name in MARKER_DTYPE.names}


source_directory = args.source_directory.resolve()
feature_directory = args.feature_directory.resolve()
output_directory = args.output_directory.resolve()
source_manifest_path = source_directory / "manifest.json"
source_audit_path = source_directory / "audit_report.json"
feature_manifest_path = feature_directory / "manifest.json"
feature_audit_path = feature_directory / "audit_report.json"
for required in (
    source_manifest_path,
    source_audit_path,
    feature_manifest_path,
    feature_audit_path,
):
    if not required.is_file():
        raise FileNotFoundError(required)
source_manifest = load_json(source_manifest_path)
source_audit = load_json(source_audit_path)
feature_manifest = load_json(feature_manifest_path)
feature_audit = load_json(feature_audit_path)
if source_audit.get("valid") is not True or feature_audit.get("valid") is not True:
    raise ValueError("Both source and feature caches must have valid independent audits")
if source_audit["manifest_sha256"] != sha256_file(source_manifest_path):
    raise ValueError("Source audit does not reference the current source manifest")
if feature_audit["manifest_sha256"] != sha256_file(feature_manifest_path):
    raise ValueError("Feature audit does not reference the current feature manifest")
if feature_manifest["source"]["manifest_sha256"] != sha256_file(source_manifest_path):
    raise ValueError("Feature cache was not built from the supplied source cache")
if args.end_sample >= len(source_manifest["samples"]):
    raise ValueError("end-sample exceeds source cache")
if feature_manifest["sample_indices"][: args.end_sample + 1] != list(
    range(args.end_sample + 1)
):
    raise ValueError("Feature cache does not contain consecutive source samples")
if output_directory.exists() and any(output_directory.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
output_directory.mkdir(parents=True, exist_ok=True)

run_report_path = Path(feature_manifest["source"]["run_report"])
run_report = load_json(run_report_path)
terrain_path = Path(feature_manifest["terrain"]["heightfield"])
with np.load(terrain_path, allow_pickle=False) as terrain_cache:
    terrain_x = np.asarray(terrain_cache["x_values"], dtype=np.float64)
    terrain_z = np.asarray(terrain_cache["z_values"], dtype=np.float64)
    terrain_y = np.asarray(terrain_cache["terrain_y"], dtype=np.float32)

particle_count = int(source_manifest["particle_count"])
spacing = float(feature_manifest["configuration"]["particle_spacing"])
water_level = float(feature_manifest["configuration"]["water_level"])
sphere_radius = float(feature_manifest["configuration"]["sphere_radius"])
particle_volume = spacing**3
characteristic_velocity = math.sqrt(9.81 * spacing)
spray_packet_volume = float(sphere_volume(args.spray_packet_radius))
air_packet_volume = float(sphere_volume(args.air_packet_radius) * args.air_packet_weight)

first_source_path = source_directory / source_manifest["samples"][0]["file"]
with np.load(first_source_path, allow_pickle=False) as first_source:
    first_positions = np.asarray(first_source["positions"], dtype=np.float32)
surface_x = np.arange(
    float(first_positions[:, 0].min()) - 0.35,
    float(first_positions[:, 0].max()) + 0.35 + 0.5 * args.surface_cell_size,
    args.surface_cell_size,
    dtype=np.float64,
)
surface_z = np.arange(
    float(first_positions[:, 2].min()) - 0.35,
    float(first_positions[:, 2].max()) + 0.35 + 0.5 * args.surface_cell_size,
    args.surface_cell_size,
    dtype=np.float64,
)

spray_reservoir = DeterministicEmissionReservoir(
    particle_count, EmissionChannel.SPRAY, seed=args.seed
)
air_reservoir = DeterministicEmissionReservoir(
    particle_count, EmissionChannel.ENTRAINED_AIR, seed=args.seed
)
markers = np.empty(0, dtype=MARKER_DTYPE)
next_marker_id = 1
impact_sample = None
previous_time = None
cumulative_requested_spray_volume = 0.0
cumulative_requested_air_volume = 0.0
cumulative_emitted_spray_volume = 0.0
cumulative_emitted_air_volume = 0.0
cumulative_returned_liquid_volume = 0.0
cumulative_released_air_volume = 0.0

manifest = {
    "schema": 1,
    "created_utc": utc_now_iso(),
    "producer": {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "state_module": str(
            (Path(__file__).parent / "whitewater" / "state_machine.py").resolve()
        ),
        "state_module_sha256": sha256_file(
            Path(__file__).parent / "whitewater" / "state_machine.py"
        ),
    },
    "source": {
        "directory": str(source_directory),
        "manifest_sha256": sha256_file(source_manifest_path),
        "audit_sha256": sha256_file(source_audit_path),
    },
    "features": {
        "directory": str(feature_directory),
        "manifest_sha256": sha256_file(feature_manifest_path),
        "audit_sha256": sha256_file(feature_audit_path),
    },
    "terrain": {"path": str(terrain_path), "sha256": sha256_file(terrain_path)},
    "configuration": {
        "end_sample": args.end_sample,
        "seed": args.seed,
        "particle_spacing": spacing,
        "particle_volume": particle_volume,
        "characteristic_velocity": characteristic_velocity,
        "surface_cell_size": args.surface_cell_size,
        "maximum_surface_projection_speed": args.maximum_surface_projection_speed,
        "surface_grid_shape": [len(surface_x), len(surface_z)],
        "spray_volume_coefficient": args.spray_volume_coefficient,
        "air_volume_coefficient": args.air_volume_coefficient,
        "spray_packet_volume": spray_packet_volume,
        "air_packet_volume": air_packet_volume,
        "air_packet_weight": args.air_packet_weight,
        "pre_contact_emission_allowed": False,
    },
    "state": {
        "complete": False,
        "completed_samples": 0,
        "expected_samples": args.end_sample + 1,
    },
    "samples": [],
}
manifest_path = output_directory / "manifest.json"
atomic_write_json(manifest_path, manifest)

for sample_index in range(args.end_sample + 1):
    source_row = source_manifest["samples"][sample_index]
    feature_row = feature_manifest["samples"][sample_index]
    with np.load(source_directory / source_row["file"], allow_pickle=False) as cache:
        positions = np.asarray(cache["positions"], dtype=np.float32)
        velocities = np.asarray(cache["velocities"], dtype=np.float32)
        simulation_time = float(cache["simulation_time"])
        sphere_transform = np.asarray(cache["sphere_transform"], dtype=np.float64)
    with np.load(feature_directory / feature_row["file"], allow_pickle=False) as cache:
        confidence = np.asarray(cache["surface_confidence"], dtype=np.float32)
        normals = np.asarray(cache["surface_normal"], dtype=np.float32)
        neighbor_count = np.asarray(cache["neighbor_count"], dtype=np.int16)
        normal_velocity = np.asarray(cache["normal_velocity"], dtype=np.float32)
        crest = np.asarray(cache["wave_crest_potential"], dtype=np.float32)
        trapped = np.asarray(cache["trapped_air_potential"], dtype=np.float32)
        solid_clearance = np.asarray(cache["solid_clearance"], dtype=np.float32)
    dt = 0.0 if previous_time is None else simulation_time - previous_time
    if previous_time is not None and dt <= 0.0:
        raise RuntimeError("Source times are not strictly increasing")
    previous_time = simulation_time
    sphere_center = sphere_transform[3, :3].astype(np.float32)
    contact_now = float(sphere_center[1]) - sphere_radius <= water_level + spacing
    if contact_now and impact_sample is None:
        impact_sample = sample_index

    surface_layers = dynamic_surface_field(
        positions,
        confidence,
        neighbor_count,
        surface_x,
        surface_z,
        water_level,
    )

    if len(markers) and dt > 0.0:
        alive = markers["state"] != np.uint8(WhitewaterState.DEAD)
        spray = markers["state"] == np.uint8(WhitewaterState.SPRAY)
        bubble = markers["state"] == np.uint8(WhitewaterState.ENTRAINED_BUBBLE)
        surface_bubble = markers["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
        foam = markers["state"] == np.uint8(WhitewaterState.SURFACE_FOAM)
        if np.any(spray):
            radius = markers["radius"][spray]
            drag_time = np.maximum(0.025, 0.08 * radius / 0.001)
            decay = np.exp(-dt / drag_time).astype(np.float32)
            markers["velocity"][spray] *= decay[:, None]
            markers["velocity"][spray, 1] -= np.float32(9.81 * dt)
            markers["position"][spray] += markers["velocity"][spray] * np.float32(dt)
        if np.any(bubble):
            ids = markers["source_particle_id"][bubble]
            rise = bubble_terminal_velocity(markers["radius"][bubble]).astype(np.float32)
            markers["velocity"][bubble] = velocities[ids]
            markers["velocity"][bubble, 1] += rise
            markers["position"][bubble] += markers["velocity"][bubble] * np.float32(dt)
            markers["shape"][bubble] = bubble_shape(markers["radius"][bubble], rise)
        for state_mask in (surface_bubble, foam):
            if np.any(state_mask):
                state_indices = np.flatnonzero(state_mask)
                ids = markers["source_particle_id"][state_indices]
                tangent_velocity = velocities[ids].copy()
                tangent_velocity[:, 1] = 0.0
                markers["velocity"][state_indices] = tangent_velocity
                surface_positions = markers["position"][state_indices].copy()
                surface_positions[:, (0, 2)] += (
                    tangent_velocity[:, (0, 2)] * np.float32(dt)
                )
                markers["position"][state_indices] = surface_positions
        markers["state_age"][alive] += np.float32(dt)

        marker_top, marker_top_valid = sample_surface_nearest(
            markers["position"], surface_x, surface_z, surface_layers
        )
        spray_reentry = spray & marker_top_valid & (
            markers["position"][:, 1] <= marker_top + markers["radius"]
        )
        if np.any(spray_reentry):
            transition_state(markers, spray_reentry, WhitewaterState.SURFACE_FOAM)
        bubble_reached_surface = bubble & marker_top_valid & (
            markers["position"][:, 1] >= marker_top - markers["radius"]
        )
        if np.any(bubble_reached_surface):
            transition_state(
                markers, bubble_reached_surface, WhitewaterState.SURFACE_BUBBLE
            )
        surface_states = (
            (markers["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE))
            | (markers["state"] == np.uint8(WhitewaterState.SURFACE_FOAM))
        )
        supported_surface = surface_states & marker_top_valid
        if np.any(supported_surface):
            maximum_projection = args.maximum_surface_projection_speed * dt
            vertical_correction = np.clip(
                marker_top[supported_surface]
                - markers["position"][supported_surface, 1],
                -maximum_projection,
                maximum_projection,
            )
            markers["position"][supported_surface, 1] += vertical_correction
        terrain_height, terrain_valid = sample_height_nearest(
            markers["position"], terrain_x, terrain_z, terrain_y
        )
        current_spray = markers["state"] == np.uint8(WhitewaterState.SPRAY)
        current_foam = markers["state"] == np.uint8(WhitewaterState.SURFACE_FOAM)
        current_bubble = markers["state"] == np.uint8(
            WhitewaterState.ENTRAINED_BUBBLE
        )
        current_surface_bubble = markers["state"] == np.uint8(
            WhitewaterState.SURFACE_BUBBLE
        )
        terrain_hit = current_spray & terrain_valid & (
            markers["position"][:, 1] <= terrain_height + markers["radius"]
        )
        if np.any(terrain_hit):
            cumulative_returned_liquid_volume += float(
                markers["liquid_volume"][terrain_hit].sum(dtype=np.float64)
            )
            markers["state"][terrain_hit] = np.uint8(WhitewaterState.DEAD)
        bubble_hit = current_bubble & terrain_valid & (
            markers["position"][:, 1] < terrain_height + markers["radius"]
        )
        if np.any(bubble_hit):
            markers["position"][bubble_hit, 1] = (
                terrain_height[bubble_hit] + markers["radius"][bubble_hit]
            )
            markers["velocity"][bubble_hit, 1] = np.maximum(
                markers["velocity"][bubble_hit, 1], 0.0
            )
        dry_surface = (
            (current_foam | current_surface_bubble)
            & terrain_valid
            & (markers["position"][:, 1] <= terrain_height + markers["radius"])
        )
        dry_foam = dry_surface & current_foam
        dry_surface_bubble = dry_surface & current_surface_bubble
        cumulative_returned_liquid_volume += float(
            markers["liquid_volume"][dry_foam].sum(dtype=np.float64)
        )
        cumulative_released_air_volume += float(
            markers["gas_volume"][dry_surface_bubble].sum(dtype=np.float64)
        )
        markers["state"][dry_surface] = np.uint8(WhitewaterState.DEAD)
        expired_spray = current_spray & (markers["state_age"] >= 1.5)
        expired_foam = current_foam & ~dry_surface & (markers["state_age"] >= 2.5)
        expired_bubble = current_bubble & (markers["state_age"] >= 2.0)
        surface_lifetime = 0.25 + 0.75 * np.clip(markers["radius"] / 0.003, 0.0, 1.0)
        expired_surface_bubble = (
            current_surface_bubble
            & ~dry_surface
            & (markers["state_age"] >= surface_lifetime)
        )
        liquid_expired = expired_spray | expired_foam
        cumulative_returned_liquid_volume += float(
            markers["liquid_volume"][liquid_expired].sum(dtype=np.float64)
        )
        gas_expired = expired_bubble | expired_surface_bubble
        cumulative_released_air_volume += float(
            markers["gas_volume"][gas_expired].sum(dtype=np.float64)
        )
        markers["state"][liquid_expired | gas_expired] = np.uint8(
            WhitewaterState.DEAD
        )

        alive = markers["state"] != np.uint8(WhitewaterState.DEAD)
        delta = markers["position"] - sphere_center
        distance = np.linalg.norm(delta, axis=1)
        minimum_distance = sphere_radius + markers["radius"]
        overlap = alive & (distance < minimum_distance) & (distance > 1.0e-8)
        if np.any(overlap):
            direction = delta[overlap] / distance[overlap, None]
            markers["position"][overlap] = (
                sphere_center + direction * minimum_distance[overlap, None]
            )
            inward = np.sum(markers["velocity"][overlap] * direction, axis=1) < 0.0
            overlap_indices = np.flatnonzero(overlap)
            markers["velocity"][overlap_indices[inward]] = 0.0

    if impact_sample is not None and dt > 0.0:
        surface_gate = confidence * (solid_clearance >= 2.0 * spacing)
        spray_rate = (
            args.spray_volume_coefficient
            * particle_volume
            * crest
            / spacing
            * surface_gate
            * smoothstep01(normal_velocity / characteristic_velocity)
        )
        air_rate = (
            args.air_volume_coefficient
            * particle_volume
            * trapped
            / characteristic_velocity
            * surface_gate
        )
        requested_spray = spray_rate * dt
        requested_air = air_rate * dt
        cumulative_requested_spray_volume += float(requested_spray.sum(dtype=np.float64))
        cumulative_requested_air_volume += float(requested_air.sum(dtype=np.float64))
        spray_births = spray_reservoir.advance(requested_spray / spray_packet_volume)
        air_births = air_reservoir.advance(requested_air / air_packet_volume)

        for births, state, packet_volume, channel in (
            (spray_births, WhitewaterState.SPRAY, spray_packet_volume, EmissionChannel.SPRAY),
            (air_births, WhitewaterState.ENTRAINED_BUBBLE, air_packet_volume, EmissionChannel.ENTRAINED_AIR),
        ):
            source_ids = births["source_particle_id"]
            count = len(source_ids)
            if not count:
                continue
            keys = births["random_key"]
            u0 = uniform01_from_keys(keys)
            u1 = uniform01_from_keys(counter_keys(source_ids, channel, births["source_emission_index"] + 17, args.seed))
            if state == WhitewaterState.SPRAY:
                radii = 0.00035 + 0.00145 * u0**2
                birth_positions = positions[source_ids] + normals[source_ids] * radii[:, None]
                birth_velocities = velocities[source_ids] + normals[source_ids] * (
                    0.15 + 0.85 * u1
                )[:, None]
            else:
                radii = np.exp(
                    math.log(0.0003) + u0 * (math.log(0.003) - math.log(0.0003))
                )
                birth_positions = positions[source_ids] - normals[source_ids] * (
                    spacing + radii
                )[:, None]
                birth_velocities = velocities[source_ids]
            weights = packet_volume / sphere_volume(radii)
            marker_ids = np.arange(next_marker_id, next_marker_id + count, dtype=np.uint64)
            next_marker_id += count
            born = make_markers(
                marker_ids,
                state,
                source_ids,
                sample_index,
                simulation_time,
                birth_positions,
                birth_velocities,
                radii,
                weights,
                keys,
            )
            markers = np.concatenate((markers, born))
            if state == WhitewaterState.SPRAY:
                cumulative_emitted_spray_volume += float(
                    born["liquid_volume"].sum(dtype=np.float64)
                )
            else:
                cumulative_emitted_air_volume += float(
                    born["gas_volume"].sum(dtype=np.float64)
                )

    snapshot = marker_snapshot(markers)
    output_path = output_directory / f"states_{sample_index:06d}.npz"
    atomic_write_npz(
        output_path,
        schema=np.asarray(1, dtype="<i4"),
        source_sample_index=np.asarray(sample_index, dtype="<i4"),
        simulation_time=np.asarray(simulation_time, dtype="<f8"),
        **snapshot,
    )
    counts = {
        state.name: int(np.count_nonzero(snapshot["state"] == np.uint8(state)))
        for state in WhitewaterState
        if state != WhitewaterState.DEAD
    }
    row = {
        "source_sample_index": sample_index,
        "simulation_time": simulation_time,
        "file": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "counts": counts,
        "alive_markers": int(len(snapshot["state"])),
        "liquid_volume": float(snapshot["liquid_volume"].sum(dtype=np.float64)),
        "gas_volume": float(snapshot["gas_volume"].sum(dtype=np.float64)),
    }
    manifest["samples"].append(row)
    manifest["state"]["completed_samples"] = sample_index + 1
    manifest["state"]["impact_sample"] = impact_sample
    manifest["ledger"] = {
        "requested_spray_volume": cumulative_requested_spray_volume,
        "emitted_spray_volume": cumulative_emitted_spray_volume,
        "returned_liquid_volume": cumulative_returned_liquid_volume,
        "requested_air_volume": cumulative_requested_air_volume,
        "emitted_air_volume": cumulative_emitted_air_volume,
        "released_air_volume": cumulative_released_air_volume,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"[whitewater-state] sample={sample_index:04d} contact={impact_sample} "
        f"alive={row['alive_markers']} states={counts}",
        flush=True,
    )

manifest["state"]["complete"] = True
manifest["state"]["completed_utc"] = utc_now_iso()
atomic_write_json(manifest_path, manifest)
print(json.dumps({"valid": True, "output": str(output_directory), "ledger": manifest["ledger"]}, indent=2))
