"""Audit whether a v6 whitewater gate ends before its pending bubbles can burst.

This is a read-only model audit.  It does not change marker states or tune any
emission, survival, radius, material, or render parameter.  The forecast is a
buoyancy-only, frozen-surface estimate intended for choosing the next gate
length; it is not a replacement for extending the trajectory simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SURFACE_BUBBLE = 4
ENTRAINED_BUBBLE = 3
BUBBLE_FILM_SURVIVAL = 0.12
STANDARD_HORIZONS_SECONDS = (0.05, 0.10, 0.20, 0.40, 0.80)
CANDIDATE_SOURCE_SAMPLES = (96, 120, 160)
IMPACT_RADII_METERS = (0.25, 0.40)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_directory", type=Path)
    parser.add_argument("liquid_field_directory", type=Path)
    parser.add_argument("surface_foam_directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    levels = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    result = np.quantile(values, levels)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
            result,
        )
    }


def surface_bubble_lifetime(radius, random_key):
    radius = np.asarray(radius, dtype=np.float64)
    key = np.asarray(random_key, dtype=np.uint64)
    jitter = (key >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    size = np.clip((radius - 0.00012) / (0.008 - 0.00012), 0.0, 1.0)
    return np.clip(0.30 - 0.18 * np.sqrt(size) + 0.08 * jitter, 0.06, 0.38)


def bubble_terminal_velocity(radius, iterations=12):
    radius = np.asarray(radius, dtype=np.float64)
    liquid_density = 998.2
    gas_density = 1.204
    viscosity = 1.002e-3
    gravity = 9.81
    diameter = 2.0 * radius
    density_difference = liquid_density - gas_density
    velocity = np.maximum(
        2.0 * density_difference * gravity * radius**2 / (9.0 * viscosity),
        1.0e-6,
    )
    for _ in range(iterations):
        reynolds = np.maximum(
            liquid_density * velocity * diameter / viscosity,
            1.0e-8,
        )
        drag = np.where(
            reynolds < 1000.0,
            24.0 / reynolds * (1.0 + 0.15 * reynolds**0.687),
            0.44,
        )
        velocity = np.sqrt(
            4.0
            * density_difference
            * gravity
            * diameter
            / (3.0 * drag * liquid_density)
        )
    return np.minimum(velocity, 0.35)


def splitmix64(values):
    values = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def uniform_for_stream(keys, stream):
    mask = (1 << 64) - 1
    salt = np.uint64((int(stream) * 0x9E3779B97F4A7C15) & mask)
    mixed = splitmix64(np.asarray(keys, dtype=np.uint64) ^ salt)
    return (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def bubble_drainage_time(random_key):
    random_b = uniform_for_stream(random_key, 102)
    return 1.5 * (6.0 / 1.5) ** random_b


def bubble_film_area(markers):
    radius = markers["physical_radius"].astype(np.float64)
    representative = markers["representative_count"].astype(np.float64)
    return representative * 4.0 * np.pi * radius**2 * BUBBLE_FILM_SURVIVAL


def sample_scalar_trilinear(field, positions, origin, spacing):
    positions = np.asarray(positions, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    coordinates = (positions - origin[None, :]) / float(spacing)
    lower = np.floor(coordinates).astype(np.int64)
    shape = np.asarray(field.shape, dtype=np.int64)
    valid = np.all(lower >= 0, axis=1) & np.all(lower + 1 < shape, axis=1)
    result = np.full(len(positions), np.nan, dtype=np.float64)
    if not np.any(valid):
        return result
    ids = np.flatnonzero(valid)
    base = lower[ids]
    fraction = coordinates[ids] - base
    accumulated = np.zeros(len(ids), dtype=np.float64)
    for dx in (0, 1):
        wx = (1.0 - fraction[:, 0]) if dx == 0 else fraction[:, 0]
        for dy in (0, 1):
            wy = (1.0 - fraction[:, 1]) if dy == 0 else fraction[:, 1]
            for dz in (0, 1):
                wz = (1.0 - fraction[:, 2]) if dz == 0 else fraction[:, 2]
                accumulated += (
                    wx
                    * wy
                    * wz
                    * field[
                        base[:, 0] + dx,
                        base[:, 1] + dy,
                        base[:, 2] + dz,
                    ]
                )
    result[ids] = accumulated
    return result


def released_summary(release_time, film_area, horizon):
    selected = np.isfinite(release_time) & (release_time <= horizon)
    return {
        "marker_count": int(np.count_nonzero(selected)),
        "represented_bubble_count": float(0.0),
        "new_film_area_m2": float(film_area[selected].sum(dtype=np.float64)),
    }


def active_after_release(release_time, film_area, drainage_time, horizon):
    selected = np.isfinite(release_time) & (release_time <= horizon)
    if not np.any(selected):
        return 0.0
    elapsed = horizon - release_time[selected]
    return float(
        np.sum(
            film_area[selected] * np.exp(-elapsed / drainage_time[selected]),
            dtype=np.float64,
        )
    )


def coverage(area, radius):
    return float(area / (np.pi * radius**2))


def main():
    args = parse_args()
    trajectory_manifest_path = args.trajectory_directory / "manifest.json"
    liquid_manifest_path = args.liquid_field_directory / "manifest.json"
    foam_manifest_path = args.surface_foam_directory / "manifest.json"
    trajectory_manifest = load_json(trajectory_manifest_path)
    liquid_manifest = load_json(liquid_manifest_path)
    foam_manifest = load_json(foam_manifest_path)

    trajectory_sample = trajectory_manifest["samples"][-1]
    liquid_sample = liquid_manifest["samples"][-1]
    foam_sample = foam_manifest["samples"][-1]
    source_indices = {
        int(trajectory_sample["source_sample_index"]),
        int(liquid_sample["source_sample_index"]),
        int(foam_sample["source_sample_index"]),
    }
    if len(source_indices) != 1:
        raise RuntimeError(f"Final source samples do not match: {source_indices}")
    current_source_sample = next(iter(source_indices))

    liquid_sample_times = np.asarray(
        [sample["simulation_time"] for sample in liquid_manifest["samples"]],
        dtype=np.float64,
    )
    liquid_source_indices = np.asarray(
        [sample["source_sample_index"] for sample in liquid_manifest["samples"]],
        dtype=np.float64,
    )
    source_dt = float(
        np.median(np.diff(liquid_sample_times) / np.diff(liquid_source_indices))
    )
    candidate_horizons = {
        round((sample - current_source_sample) * source_dt, 12): sample
        for sample in CANDIDATE_SOURCE_SAMPLES
        if sample > current_source_sample
    }
    horizons_seconds = sorted(
        {
            *(round(value, 12) for value in STANDARD_HORIZONS_SECONDS),
            *candidate_horizons.keys(),
        }
    )

    trajectory_path = args.trajectory_directory / trajectory_sample["file"]
    liquid_path = args.liquid_field_directory / liquid_sample["file"]
    foam_path = args.surface_foam_directory / foam_sample["file"]
    with np.load(trajectory_path) as cache:
        markers = np.asarray(cache["markers"]).copy()
        interval_end = float(cache["interval_end"])
        output_frame = int(cache["output_frame"])
    with np.load(liquid_path) as cache:
        phi = np.asarray(cache["phi"], dtype=np.float64)
        simulation_time = float(cache["simulation_time"])
    with np.load(foam_path) as cache:
        foam = np.asarray(cache["foam"]).copy()

    grid = liquid_manifest["grid"]
    if tuple(phi.shape) != tuple(grid["shape"]):
        raise RuntimeError("Liquid phi shape differs from manifest grid")

    surface = markers[markers["state"] == np.uint8(SURFACE_BUBBLE)]
    entrained = markers[markers["state"] == np.uint8(ENTRAINED_BUBBLE)]

    surface_lifetime = surface_bubble_lifetime(
        surface["physical_radius"], surface["random_key"]
    )
    surface_remaining = np.maximum(
        surface_lifetime - surface["state_age"].astype(np.float64), 0.0
    )
    surface_area = bubble_film_area(surface)
    surface_drainage = bubble_drainage_time(surface["random_key"])

    entrained_phi = sample_scalar_trilinear(
        phi,
        entrained["position"],
        grid["origin"],
        grid["spacing"],
    )
    entrained_depth = np.maximum(-entrained_phi, 0.0)
    entrained_radius = entrained["physical_radius"].astype(np.float64)
    # The actual state transition occurs at phi >= -0.5 * bubble radius.
    rise_distance = np.maximum(entrained_depth - 0.5 * entrained_radius, 0.0)
    rise_velocity = bubble_terminal_velocity(entrained_radius)
    arrival_time = rise_distance / rise_velocity
    arrival_time[~np.isfinite(entrained_phi)] = np.nan
    entrained_surface_lifetime = surface_bubble_lifetime(
        entrained_radius, entrained["random_key"]
    )
    entrained_burst_time = arrival_time + entrained_surface_lifetime
    entrained_area = bubble_film_area(entrained)
    entrained_drainage = bubble_drainage_time(entrained["random_key"])

    current_film_area = float(foam["film_area"].sum(dtype=np.float64))
    current_drainage = foam["drainage_time"].astype(np.float64)
    current_source_counts = {
        "spray_reentry": int(np.count_nonzero(foam["source_kind"] == 1)),
        "bubble_burst": int(np.count_nonzero(foam["source_kind"] == 2)),
    }

    horizons = []
    for horizon in horizons_seconds:
        surface_release = released_summary(surface_remaining, surface_area, horizon)
        surface_release["represented_bubble_count"] = float(
            surface["representative_count"][surface_remaining <= horizon].sum(
                dtype=np.float64
            )
        )
        entrained_arrival = released_summary(arrival_time, entrained_area, horizon)
        arrival_selection = np.isfinite(arrival_time) & (arrival_time <= horizon)
        entrained_arrival["represented_bubble_count"] = float(
            entrained["representative_count"][arrival_selection].sum(dtype=np.float64)
        )
        entrained_burst = released_summary(
            entrained_burst_time, entrained_area, horizon
        )
        burst_selection = np.isfinite(entrained_burst_time) & (
            entrained_burst_time <= horizon
        )
        entrained_burst["represented_bubble_count"] = float(
            entrained["representative_count"][burst_selection].sum(dtype=np.float64)
        )

        retained_current = float(
            np.sum(
                foam["film_area"].astype(np.float64)
                * np.exp(-horizon / current_drainage),
                dtype=np.float64,
            )
        )
        retained_surface = active_after_release(
            surface_remaining,
            surface_area,
            surface_drainage,
            horizon,
        )
        retained_entrained = active_after_release(
            entrained_burst_time,
            entrained_area,
            entrained_drainage,
            horizon,
        )
        estimated_active = retained_current + retained_surface + retained_entrained
        gross_inventory = (
            current_film_area
            + surface_release["new_film_area_m2"]
            + entrained_burst["new_film_area_m2"]
        )
        horizons.append(
            {
                "seconds_after_gate": horizon,
                "candidate_source_sample": candidate_horizons.get(horizon),
                "surface_bubble_bursts": surface_release,
                "entrained_bubble_arrivals": entrained_arrival,
                "entrained_bubble_bursts_after_surface_residence": entrained_burst,
                "gross_undrained_inventory_m2": gross_inventory,
                "estimated_active_film_m2": estimated_active,
                "estimated_active_film_components_m2": {
                    "retained_current": retained_current,
                    "from_current_surface_bubbles": retained_surface,
                    "from_current_entrained_bubbles": retained_entrained,
                },
                "estimated_coverage_fraction": {
                    f"radius_{radius:.2f}m": coverage(estimated_active, radius)
                    for radius in IMPACT_RADII_METERS
                },
            }
        )

    all_pending_area = float(
        surface_area.sum(dtype=np.float64) + entrained_area.sum(dtype=np.float64)
    )
    eventual_inventory = current_film_area + all_pending_area
    report = {
        "schema": 1,
        "product": "whitewater_v6_time_budget_audit",
        "scope": {
            "changes_simulation_or_render": False,
            "forecast_type": "frozen-surface buoyancy-only diagnostic",
            "future_births_after_gate_included": False,
            "surface_advection_and_topology_loss_in_forecast": False,
            "foam_drainage_in_estimated_active_film": True,
            "film_area_source": "representative_count * 4*pi*r^2 * 0.12",
        },
        "gate": {
            "output_frame": output_frame,
            "source_sample_index": current_source_sample,
            "source_sample_period_seconds": source_dt,
            "interval_end_seconds": interval_end,
            "simulation_time_seconds": simulation_time,
            "active_marker_count": int(len(markers)),
            "state_counts": {
                "spray": int(np.count_nonzero(markers["state"] == 1)),
                "entrained_bubble": int(len(entrained)),
                "surface_bubble": int(len(surface)),
            },
        },
        "current_surface_foam": {
            "parcel_count": int(len(foam)),
            "source_counts": current_source_counts,
            "film_area_m2": current_film_area,
            "film_area_cm2": current_film_area * 1.0e4,
            "drainage_time_seconds": quantiles(current_drainage),
        },
        "pending_surface_bubbles": {
            "marker_count": int(len(surface)),
            "represented_bubble_count": float(
                surface["representative_count"].sum(dtype=np.float64)
            ),
            "physical_radius_m": quantiles(surface["physical_radius"]),
            "state_age_seconds": quantiles(surface["state_age"]),
            "lifetime_seconds": quantiles(surface_lifetime),
            "remaining_seconds": quantiles(surface_remaining),
            "potential_film_area_m2": float(surface_area.sum(dtype=np.float64)),
            "potential_film_area_cm2": float(surface_area.sum(dtype=np.float64) * 1.0e4),
        },
        "pending_entrained_bubbles": {
            "marker_count": int(len(entrained)),
            "represented_bubble_count": float(
                entrained["representative_count"].sum(dtype=np.float64)
            ),
            "valid_depth_samples": int(np.count_nonzero(np.isfinite(entrained_phi))),
            "invalid_depth_samples": int(np.count_nonzero(~np.isfinite(entrained_phi))),
            "physical_radius_m": quantiles(entrained_radius),
            "sampled_phi_m": quantiles(entrained_phi[np.isfinite(entrained_phi)]),
            "depth_below_zero_isosurface_m": quantiles(
                entrained_depth[np.isfinite(entrained_phi)]
            ),
            "distance_to_state_transition_m": quantiles(
                rise_distance[np.isfinite(entrained_phi)]
            ),
            "terminal_rise_velocity_m_per_s": quantiles(rise_velocity),
            "estimated_arrival_seconds": quantiles(
                arrival_time[np.isfinite(arrival_time)]
            ),
            "surface_residence_after_arrival_seconds": quantiles(
                entrained_surface_lifetime
            ),
            "estimated_burst_seconds": quantiles(
                entrained_burst_time[np.isfinite(entrained_burst_time)]
            ),
            "potential_film_area_m2": float(entrained_area.sum(dtype=np.float64)),
            "potential_film_area_cm2": float(
                entrained_area.sum(dtype=np.float64) * 1.0e4
            ),
        },
        "forecast_by_horizon": horizons,
        "inventory_upper_bound": {
            "current_plus_all_pending_film_area_m2": eventual_inventory,
            "current_plus_all_pending_film_area_cm2": eventual_inventory * 1.0e4,
            "coverage_fraction": {
                f"radius_{radius:.2f}m": coverage(eventual_inventory, radius)
                for radius in IMPACT_RADII_METERS
            },
            "warning": (
                "This is an undrained upper bound and excludes future births after "
                "the gate; it is not simultaneous visible coverage."
            ),
        },
        "inputs": {
            "trajectory_snapshot": str(trajectory_path.resolve()),
            "trajectory_snapshot_sha256": sha256_file(trajectory_path),
            "liquid_snapshot": str(liquid_path.resolve()),
            "liquid_snapshot_sha256": sha256_file(liquid_path),
            "surface_foam_snapshot": str(foam_path.resolve()),
            "surface_foam_snapshot_sha256": sha256_file(foam_path),
        },
    }

    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized)


if __name__ == "__main__":
    main()
