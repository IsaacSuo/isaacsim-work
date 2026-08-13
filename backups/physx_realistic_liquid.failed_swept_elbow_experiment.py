"""High-resolution, PhysX-only PBD liquid rendered with RTX Path Tracing.

This is intentionally independent from ParticlePostProcessingDemo.  It builds a
meter-scale dam-break scene, uses PhysX PBD parameters based on SnippetPBF, and
keeps PhysX's native isosurface so the liquid is real geometry for PathTracing.
"""

import argparse
import asyncio
import json
import math
import os
import struct
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EmitterLayoutPlan:
    spacing: float
    axial_spacing: float
    row_spacing: float
    particle_volume: float
    nozzle_layer_counts: tuple
    nozzle_layer_mean: float
    particle_flux: float
    volume_flux_lps: float
    planned_seconds: float
    reservoir_diameter: float
    reservoir_layers: int
    reservoir_layer_counts: tuple
    reservoir_particles: int
    connector_particles: int
    initial_pool_particles_upper_bound: int
    active_particles_upper_bound: int
    capacity_limit: int
    contraction_length: float
    bend_radius: float
    bend_arc_length: float
    straightening_length: float
    connector_length: float
    emitter_feed_clearance: float
    pipe_sides: int
    bend_segments: int
    contraction_particles: int
    bend_particles: int
    straightening_particles: int
    reservoir_inner_length: float
    reservoir_inner_diameter: float
    pusher_speed: float
    drive_gain: float
    pusher_step_fraction: float
    pusher_planned_stroke: float
    pusher_max_stroke: float
    max_velocity: float
    theory_slices: tuple

    def to_dict(self):
        return asdict(self)


def create_circular_hcp_layer(spacing, diameter, packing_offset=(0.0, 0.0)):
    """Return one circular hexagonally packed cross-section as NumPy X/Z pairs."""
    if diameter < 3.0 * spacing:
        raise ValueError("Emitter nozzle must be at least three particle spacings wide")
    center_limit = 0.5 * diameter - 0.5 * spacing
    row_spacing = 0.5 * math.sqrt(3.0) * spacing
    row_count = int(math.ceil(center_limit / row_spacing))
    column_count = int(math.ceil(center_limit / spacing)) + 1
    points = []
    for row in range(-row_count, row_count + 1):
        z = row * row_spacing + packing_offset[1]
        x_offset = 0.5 * spacing if row & 1 else 0.0
        for column in range(-column_count, column_count + 1):
            x = column * spacing + x_offset + packing_offset[0]
            if x * x + z * z <= center_limit * center_limit:
                points.append((x, z))
    if not points:
        raise ValueError("Emitter cross-section contains no particles")
    return np.asarray(points, dtype=np.float32)


def build_emitter_centerline(
    outlet_x,
    emitter_height,
    feed_clearance,
    contraction_length,
    bend_radius,
    nozzle_diameter,
):
    """Describe a +X taper, 90-degree elbow, and final -Y straight."""
    feed_y = emitter_height + feed_clearance
    straightening_length = feed_clearance - bend_radius
    if straightening_length <= 0.0:
        raise ValueError("Emitter feed clearance must exceed the bend radius")
    bend_arc_length = 0.5 * math.pi * bend_radius
    contraction_end_x = outlet_x - bend_radius
    contraction_start_x = contraction_end_x - contraction_length
    # Keep the taper inside the reservoir envelope and start the physical swept
    # pipe at the nozzle-diameter throat. This prevents reservoir-sized flow from
    # entering the elbow before the final straight can condition it.
    return {
        "outlet_x": outlet_x,
        "emitter_height": emitter_height,
        "feed_y": feed_y,
        "contraction_length": contraction_length,
        "nozzle_diameter": nozzle_diameter,
        "bend_radius": bend_radius,
        "bend_arc_length": bend_arc_length,
        "straightening_length": straightening_length,
        "contraction_start_x": contraction_start_x,
        "contraction_end_x": contraction_end_x,
        "bend_center_x": outlet_x - bend_radius,
        "bend_center_y": feed_y - bend_radius,
        "total_length": contraction_length + bend_arc_length + straightening_length,
    }


def evaluate_emitter_centerline(path, arc_lengths):
    """Evaluate center, tangent, and region along an emitter centerline."""
    arc_lengths = np.asarray(arc_lengths, dtype=np.float64)
    centers = np.empty((len(arc_lengths), 3), dtype=np.float64)
    tangents = np.empty((len(arc_lengths), 3), dtype=np.float64)
    regions = np.empty(len(arc_lengths), dtype=object)
    contraction_end = path["contraction_length"]
    bend_end = contraction_end + path["bend_arc_length"]

    contraction = arc_lengths <= contraction_end + 1.0e-12
    centers[contraction, 0] = path["contraction_start_x"] + arc_lengths[contraction]
    centers[contraction, 1] = path["feed_y"]
    centers[contraction, 2] = 0.0
    tangents[contraction] = (1.0, 0.0, 0.0)
    regions[contraction] = "contraction"

    bend = (arc_lengths > contraction_end + 1.0e-12) & (
        arc_lengths <= bend_end + 1.0e-12
    )
    bend_angle = (arc_lengths[bend] - contraction_end) / path["bend_radius"]
    centers[bend, 0] = path["bend_center_x"] + path["bend_radius"] * np.sin(bend_angle)
    centers[bend, 1] = path["bend_center_y"] + path["bend_radius"] * np.cos(bend_angle)
    centers[bend, 2] = 0.0
    tangents[bend, 0] = np.cos(bend_angle)
    tangents[bend, 1] = -np.sin(bend_angle)
    tangents[bend, 2] = 0.0
    regions[bend] = "bend"

    straight = arc_lengths > bend_end + 1.0e-12
    straight_distance = arc_lengths[straight] - bend_end
    centers[straight, 0] = path["outlet_x"]
    centers[straight, 1] = path["bend_center_y"] - straight_distance
    centers[straight, 2] = 0.0
    tangents[straight] = (0.0, -1.0, 0.0)
    regions[straight] = "straightening"
    return centers, tangents, regions


def parallel_transport_frames(tangents):
    """Return stable normal/binormal vectors for the planar swept path."""
    tangents = np.asarray(tangents, dtype=np.float64)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0], np.zeros(len(tangents))))
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_lengths[:, None], 1.0e-12)
    binormals = np.cross(tangents, normals)
    return normals, binormals


def sample_centerline_by_arc_length(path, axial_spacing):
    """Sample fixed-topology stations, excluding reservoir and outlet boundaries."""
    arc_lengths = np.arange(
        axial_spacing,
        path["total_length"],
        axial_spacing,
        dtype=np.float64,
    )
    centers, tangents, regions = evaluate_emitter_centerline(path, arc_lengths)
    normals, binormals = parallel_transport_frames(tangents)
    return arc_lengths, centers, tangents, normals, binormals, regions


def emitter_particle_diameters(
    arc_lengths, contraction_length, reservoir_diameter, nozzle_diameter
):
    blend = np.clip(np.asarray(arc_lengths) / contraction_length, 0.0, 1.0)
    return reservoir_diameter * (1.0 - blend) + nozzle_diameter * blend


def emitter_path_particle_diameters(path, arc_lengths, reservoir_diameter):
    """Use a short taper followed by nozzle-diameter elbow and straight."""
    arc_lengths = np.asarray(arc_lengths, dtype=np.float64)
    diameters = np.full(len(arc_lengths), path["nozzle_diameter"], dtype=np.float64)
    contraction = arc_lengths <= path["contraction_length"] + 1.0e-12
    diameters[contraction] = emitter_particle_diameters(
        arc_lengths[contraction],
        path["contraction_length"],
        reservoir_diameter,
        path["nozzle_diameter"],
    )
    return diameters


def estimate_initial_pool_count(args):
    if args.emitter_initial_pool_layers <= 0:
        return 0
    spacing = args.spacing
    row_spacing = 0.5 * math.sqrt(3.0) * spacing
    x_min = max(-0.575 + spacing, args.stream_outlet_x - 0.16)
    x_max = min(0.575 - spacing, x_min + args.emitter_initial_pool_span_x)
    z_min = max(-0.265 + spacing, -0.5 * args.emitter_initial_pool_span_z)
    z_max = min(0.265 - spacing, 0.5 * args.emitter_initial_pool_span_z)
    total = 0
    z_values = np.arange(z_min, z_max + 0.5 * row_spacing, row_spacing)
    for layer_index in range(args.emitter_initial_pool_layers):
        for row_index, _ in enumerate(z_values):
            x_offset = 0.5 * spacing if (row_index + layer_index) & 1 else 0.0
            total += len(
                np.arange(x_min + x_offset, x_max + 0.5 * spacing, spacing)
            )
    return total


def validate_args(args):
    positive = {
        "frames": args.frames,
        "capture-every": args.capture_every,
        "width": args.width,
        "height": args.height,
        "spacing": args.spacing,
        "particle-capacity": args.particle_capacity,
        "substeps": args.substeps,
        "solver-iterations": args.solver_iterations,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")
    if args.source_mode == "emitter":
        emitter_positive = {
            "emitter-speed": args.emitter_speed,
            "emitter-nozzle-diameter": args.emitter_nozzle_diameter,
            "emitter-feed-clearance": args.emitter_feed_clearance,
            "emitter-reservoir-max-side": args.emitter_reservoir_max_side,
            "emitter-pusher-max-step-fraction": args.emitter_pusher_max_step_fraction,
            "emitter-drive-gain": args.emitter_drive_gain,
            "emitter-initial-velocity-scale": args.emitter_initial_velocity_scale,
            "emitter-straight-length-diameters": args.emitter_straight_length_diameters,
            "emitter-bend-radius-diameters": args.emitter_bend_radius_diameters,
            "jet-slice-thickness-spacings": args.jet_slice_thickness_spacings,
            "emitter-initial-pool-span-x": args.emitter_initial_pool_span_x,
            "emitter-initial-pool-span-z": args.emitter_initial_pool_span_z,
        }
        for name, value in emitter_positive.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"--{name} must be positive, got {value}")
        for name, value in {
            "emitter-preroll-frames": args.emitter_preroll_frames,
            "emitter-reserve-frames": args.emitter_reserve_frames,
            "emitter-ramp-frames": args.emitter_ramp_frames,
            "emitter-initial-pool-layers": args.emitter_initial_pool_layers,
        }.items():
            if value < 0:
                raise ValueError(f"--{name} must be non-negative, got {value}")
        if args.emitter_pusher_max_step_fraction > 0.5:
            raise ValueError("--emitter-pusher-max-step-fraction must be <= 0.5")
        if args.emitter_initial_velocity_scale > 1.0:
            raise ValueError("--emitter-initial-velocity-scale must be <= 1.0")
        if args.emitter_straight_length_diameters < 6.0:
            raise ValueError("Emitter straightening length must be at least 6 diameters")
        if not 1.0 <= args.emitter_bend_radius_diameters <= 4.0:
            raise ValueError("Emitter bend radius must be between 1 and 4 diameters")
        if not 12 <= args.emitter_pipe_sides <= 32 or args.emitter_pipe_sides % 2:
            raise ValueError("Emitter pipe sides must be an even number from 12 to 32")
        if args.emitter_bend_segments < 6:
            raise ValueError("Emitter bend must use at least 6 segments")
        if args.jet_metrics_warmup_frames < 0:
            raise ValueError("--jet-metrics-warmup-frames must be non-negative")
        try:
            jet_slice_fractions = tuple(
                float(value) for value in args.jet_slice_fractions.split(",")
            )
        except ValueError as exc:
            raise ValueError("--jet-slice-fractions must be comma-separated numbers") from exc
        if (
            not jet_slice_fractions
            or any(not 0.0 < value < 1.0 for value in jet_slice_fractions)
            or tuple(sorted(set(jet_slice_fractions))) != jet_slice_fractions
        ):
            raise ValueError("Jet slice fractions must be unique and strictly increasing in (0, 1)")
        args.jet_slice_fractions_values = jet_slice_fractions
        if args.emitter_nozzle_diameter < 3.0 * args.spacing:
            raise ValueError("Emitter nozzle must be at least three particle spacings wide")
        bend_radius = (
            args.emitter_bend_radius_diameters * args.emitter_nozzle_diameter
        )
        straightening_length = (
            args.emitter_straight_length_diameters * args.emitter_nozzle_diameter
        )
        minimum_feed_clearance = (
            bend_radius
            + straightening_length
            + 0.5 * args.emitter_nozzle_diameter
            + 2.0 * args.spacing
        )
        if args.emitter_feed_clearance + 1.0e-12 < minimum_feed_clearance:
            args.emitter_feed_clearance = minimum_feed_clearance
    elif args.emitter_plan_only:
        raise ValueError("--emitter-plan-only requires --source-mode emitter")


def plan_compact_emitter(args):
    spacing = args.spacing
    axial_spacing = math.sqrt(2.0 / 3.0) * spacing
    row_spacing = 0.5 * math.sqrt(3.0) * spacing
    particle_volume = spacing**3 / math.sqrt(2.0)
    packing_offsets = ((0.0, 0.0), (0.5 * spacing, row_spacing / 3.0))
    nozzle_counts = tuple(
        len(create_circular_hcp_layer(spacing, args.emitter_nozzle_diameter, offset))
        for offset in packing_offsets
    )
    nozzle_mean = 0.5 * sum(nozzle_counts)
    particle_flux = nozzle_mean * args.emitter_speed / axial_spacing
    planned_seconds = (
        args.emitter_preroll_frames + args.frames + args.emitter_reserve_frames
    ) / 60.0
    required_reservoir_particles = int(
        math.ceil(particle_flux * args.emitter_drive_gain * planned_seconds)
    )
    contraction_length = max(
        0.04,
        2.0 * args.emitter_nozzle_diameter,
        16.0 * spacing,
    )
    bend_radius = (
        args.emitter_bend_radius_diameters * args.emitter_nozzle_diameter
    )
    centerline_path = build_emitter_centerline(
        args.stream_outlet_x,
        args.emitter_height,
        args.emitter_feed_clearance,
        contraction_length,
        bend_radius,
        args.emitter_nozzle_diameter,
    )
    straightening_length = centerline_path["straightening_length"]
    if straightening_length + 1.0e-12 < (
        args.emitter_straight_length_diameters * args.emitter_nozzle_diameter
    ):
        raise ValueError("Emitter feed clearance leaves no usable straightening section")
    connector_stations = sample_centerline_by_arc_length(
        centerline_path, axial_spacing
    )
    connector_arc_lengths = connector_stations[0]
    connector_regions = connector_stations[-1]
    grid_spacing = max(0.001, 0.5 * spacing)
    fluid_rest_offset = 0.5 * spacing
    particle_contact_offset = fluid_rest_offset / 0.6
    clearance = particle_contact_offset + grid_spacing
    pusher_thickness = 2.0 * spacing
    pool_upper = estimate_initial_pool_count(args)
    expected_fall_speed = math.sqrt(
        args.emitter_speed**2 + 2.0 * 9.81 * args.emitter_height
    )
    max_velocity = max(5.0, 1.25 * expected_fall_speed)
    if max_velocity > 8.0:
        raise ValueError(
            f"Emitter requires max velocity {max_velocity:.3f} m/s, above 8 m/s safety limit"
        )

    candidates = []
    minimum_diameter = max(0.055, 2.0 * args.emitter_nozzle_diameter)
    maximum_particle_diameter = args.emitter_reservoir_max_side - 2.0 * clearance
    if maximum_particle_diameter < minimum_diameter:
        raise ValueError("Emitter reservoir max side is too small for the nozzle and clearance")
    diameter_steps = int(
        math.floor((maximum_particle_diameter - minimum_diameter) / spacing)
    )
    for diameter_index in range(diameter_steps + 1):
        reservoir_diameter = minimum_diameter + diameter_index * spacing
        reservoir_counts = tuple(
            len(create_circular_hcp_layer(spacing, reservoir_diameter, offset))
            for offset in packing_offsets
        )
        reservoir_mean = 0.5 * sum(reservoir_counts)
        if reservoir_mean <= nozzle_mean:
            continue

        connector_diameters = emitter_path_particle_diameters(
            centerline_path,
            connector_arc_lengths,
            reservoir_diameter,
        )
        connector_counts = [
            len(
                create_circular_hcp_layer(
                    spacing,
                    layer_diameter,
                    packing_offsets[layer_index & 1],
                )
            )
            for layer_index, layer_diameter in enumerate(connector_diameters)
        ]
        connector_particles = sum(connector_counts)
        contraction_particles = sum(
            count
            for count, region in zip(connector_counts, connector_regions)
            if region == "contraction"
        )
        bend_particles = sum(
            count
            for count, region in zip(connector_counts, connector_regions)
            if region == "bend"
        )
        straightening_particles = sum(
            count
            for count, region in zip(connector_counts, connector_regions)
            if region == "straightening"
        )

        reservoir_layers = 2
        reservoir_particles = sum(
            reservoir_counts[layer_index & 1]
            for layer_index in range(reservoir_layers)
        )
        while reservoir_particles < required_reservoir_particles:
            reservoir_particles += reservoir_counts[reservoir_layers & 1]
            reservoir_layers += 1
        water_span = max(0, reservoir_layers - 1) * axial_spacing
        inner_length = water_span + pusher_thickness + 2.0 * clearance
        inner_diameter = reservoir_diameter + 2.0 * clearance
        if inner_length > args.emitter_reservoir_max_side:
            continue
        pusher_speed = (
            args.emitter_speed
            * nozzle_mean
            / reservoir_mean
            * args.emitter_drive_gain
        )
        step_fraction = pusher_speed / (60.0 * args.substeps * spacing)
        if step_fraction > args.emitter_pusher_max_step_fraction:
            continue
        ramp_seconds = args.emitter_ramp_frames / 60.0
        drive_seconds = (args.emitter_preroll_frames + args.frames) / 60.0
        if ramp_seconds > 0.0 and drive_seconds < ramp_seconds:
            planned_stroke = pusher_speed * (
                0.5 * drive_seconds
                - ramp_seconds
                / (2.0 * math.pi)
                * math.sin(math.pi * drive_seconds / ramp_seconds)
            )
        else:
            planned_stroke = pusher_speed * (drive_seconds - 0.5 * ramp_seconds)
        max_stroke = water_span - 2.0 * axial_spacing
        if planned_stroke <= 0.0 or planned_stroke > max_stroke:
            continue
        active_upper = reservoir_particles + connector_particles + pool_upper
        if active_upper > args.particle_capacity:
            continue
        if (
            abs(spacing - 0.0015) <= 1.0e-12
            and abs(args.emitter_nozzle_diameter - 0.028) <= 1.0e-12
            and active_upper > 760_000
        ):
            continue
        max_dimension = max(inner_length, inner_diameter)
        surface_proxy = (
            2.0 * inner_diameter * inner_diameter
            + 4.0 * inner_diameter * inner_length
        )
        score = (
            max_dimension,
            surface_proxy,
            abs(inner_length - inner_diameter),
            active_upper,
        )
        candidates.append(
            (
                score,
                reservoir_diameter,
                reservoir_layers,
                reservoir_counts,
                reservoir_particles,
                connector_particles,
                active_upper,
                inner_length,
                inner_diameter,
                pusher_speed,
                step_fraction,
                planned_stroke,
                max_stroke,
                contraction_particles,
                bend_particles,
                straightening_particles,
            )
        )

    if not candidates:
        raise ValueError(
            "No compact emitter fits the requested duration, pool, capacity, and step limit. "
            f"Estimated source particles={required_reservoir_particles}, "
            f"initial pool upper bound={pool_upper}, capacity={args.particle_capacity}. "
            "Reduce --frames/--emitter-speed, increase --spacing/--substeps, or explicitly "
            "increase --particle-capacity within the GPU budget."
        )
    (
        _,
        reservoir_diameter,
        reservoir_layers,
        reservoir_counts,
        reservoir_particles,
        connector_particles,
        active_upper,
        inner_length,
        inner_diameter,
        pusher_speed,
        step_fraction,
        planned_stroke,
        max_stroke,
        contraction_particles,
        bend_particles,
        straightening_particles,
    ) = min(candidates, key=lambda candidate: candidate[0])
    theory_slices = tuple(
        {
            "height_fraction": fraction,
            "depth_m": args.emitter_height * fraction,
            "theory_speed_mps": math.sqrt(
                args.emitter_speed**2
                + 2.0 * 9.81 * args.emitter_height * fraction
            ),
            "theory_diameter_m": args.emitter_nozzle_diameter
            * math.sqrt(
                args.emitter_speed
                / math.sqrt(
                    args.emitter_speed**2
                    + 2.0 * 9.81 * args.emitter_height * fraction
                )
            ),
        }
        for fraction in args.jet_slice_fractions_values
    )
    return EmitterLayoutPlan(
        spacing=spacing,
        axial_spacing=axial_spacing,
        row_spacing=row_spacing,
        particle_volume=particle_volume,
        nozzle_layer_counts=nozzle_counts,
        nozzle_layer_mean=nozzle_mean,
        particle_flux=particle_flux * args.emitter_drive_gain,
        volume_flux_lps=(
            particle_flux
            * args.emitter_drive_gain
            * particle_volume
            * 1000.0
        ),
        planned_seconds=planned_seconds,
        reservoir_diameter=reservoir_diameter,
        reservoir_layers=reservoir_layers,
        reservoir_layer_counts=reservoir_counts,
        reservoir_particles=reservoir_particles,
        connector_particles=connector_particles,
        initial_pool_particles_upper_bound=pool_upper,
        active_particles_upper_bound=active_upper,
        capacity_limit=args.particle_capacity,
        contraction_length=contraction_length,
        bend_radius=bend_radius,
        bend_arc_length=centerline_path["bend_arc_length"],
        straightening_length=straightening_length,
        connector_length=centerline_path["total_length"],
        emitter_feed_clearance=args.emitter_feed_clearance,
        pipe_sides=args.emitter_pipe_sides,
        bend_segments=args.emitter_bend_segments,
        contraction_particles=contraction_particles,
        bend_particles=bend_particles,
        straightening_particles=straightening_particles,
        reservoir_inner_length=inner_length,
        reservoir_inner_diameter=inner_diameter,
        pusher_speed=pusher_speed,
        drive_gain=args.emitter_drive_gain,
        pusher_step_fraction=step_fraction,
        pusher_planned_stroke=planned_stroke,
        pusher_max_stroke=max_stroke,
        max_velocity=max_velocity,
        theory_slices=theory_slices,
    )

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=60)
parser.add_argument("--capture-every", type=int, default=2)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--spacing", type=float, default=0.0015)
parser.add_argument("--nx", type=int, default=114)
parser.add_argument("--ny", type=int, default=60)
parser.add_argument("--nz", type=int, default=51)
parser.add_argument("--particle-capacity", type=int, default=800_000)
parser.add_argument(
    "--source-mode",
    choices=["emitter", "stream", "block"],
    default="emitter",
)
parser.add_argument("--stream-reservoir-bottom", type=float, default=0.40)
parser.add_argument("--stream-outlet-x", type=float, default=-0.25)
parser.add_argument("--stream-aperture", type=float, default=0.045)
parser.add_argument("--reservoir-settle-frames", type=int, default=180)
parser.add_argument("--emitter-height", type=float, default=0.50)
parser.add_argument("--emitter-nozzle-diameter", type=float, default=0.028)
parser.add_argument("--emitter-speed", type=float, default=1.20)
parser.add_argument("--emitter-preroll-frames", type=int, default=30)
parser.add_argument("--emitter-feed-clearance", type=float, default=0.11)
parser.add_argument("--emitter-reserve-frames", type=int, default=6)
parser.add_argument("--emitter-ramp-frames", type=int, default=6)
parser.add_argument("--emitter-reservoir-max-side", type=float, default=0.16)
parser.add_argument("--emitter-pusher-max-step-fraction", type=float, default=0.25)
parser.add_argument("--emitter-drive-gain", type=float, default=1.0)
parser.add_argument("--emitter-initial-velocity-scale", type=float, default=0.25)
parser.add_argument("--emitter-straight-length-diameters", type=float, default=8.0)
parser.add_argument("--emitter-bend-radius-diameters", type=float, default=1.5)
parser.add_argument("--emitter-pipe-sides", type=int, default=16)
parser.add_argument("--emitter-bend-segments", type=int, default=16)
parser.add_argument("--jet-slice-fractions", default="0.08,0.25,0.50,0.75,0.88")
parser.add_argument("--jet-slice-thickness-spacings", type=float, default=3.0)
parser.add_argument("--jet-metrics-warmup-frames", type=int, default=12)
parser.add_argument("--emitter-initial-pool-layers", type=int, default=3)
parser.add_argument("--emitter-initial-pool-span-x", type=float, default=0.50)
parser.add_argument("--emitter-initial-pool-span-z", type=float, default=0.20)
parser.add_argument("--emitter-plan-only", action="store_true")
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--solver-iterations", type=int, default=8)
parser.add_argument("--path-spp", type=int, default=32)
parser.add_argument("--isosurface-settle-updates", type=int, default=12)
parser.add_argument("--capture-timeout-updates", type=int, default=600)
parser.add_argument("--anisotropy", action="store_true")
parser.add_argument("--diagnostic-material", action="store_true")
parser.add_argument("--dynamic-obstacle", action="store_true")
parser.add_argument(
    "--obstacle-shape",
    choices=["sphere", "elephant"],
    default="sphere",
)
parser.add_argument(
    "--obstacle-mesh",
    default=r"Y:\isaacsim_work\assets\elephant.stl",
)
parser.add_argument("--obstacle-height", type=float, default=0.15)
parser.add_argument(
    "--obstacle-collision",
    choices=["convexHull", "convexDecomposition"],
    default="convexHull",
)
parser.add_argument("--mirror-obstacle", action="store_true")
parser.add_argument("--mirror-roughness", type=float, default=0.16)
parser.add_argument("--dome-light-intensity", type=float, default=650.0)
parser.add_argument("--key-light-intensity", type=float, default=5500.0)
parser.add_argument("--key-light-radius", type=float, default=0.38)
parser.add_argument("--rim-light-intensity", type=float, default=3200.0)
parser.add_argument("--rim-light-width", type=float, default=0.55)
parser.add_argument("--rim-light-height", type=float, default=0.35)
parser.add_argument("--obstacle-mass", type=float, default=0.6)
parser.add_argument("--obstacle-static-friction", type=float, default=0.12)
parser.add_argument("--obstacle-dynamic-friction", type=float, default=0.08)
parser.add_argument("--obstacle-restitution", type=float, default=0.05)
parser.add_argument(
    "--camera-preset",
    choices=["auto", "diagonal", "front", "front-high", "stream-front"],
    default="auto",
)
parser.add_argument("--renderer", choices=["PathTracing", "RaytracedLighting"], default="PathTracing")
parser.add_argument(
    "--output",
    default=r"Y:\isaacsim_work\output\physx_realistic_liquid",
)
args = parser.parse_args()
validate_args(args)
emitter_plan = plan_compact_emitter(args) if args.source_mode == "emitter" else None
if args.emitter_plan_only:
    print(json.dumps(emitter_plan.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": args.renderer,
        "width": args.width,
        "height": args.height,
    }
)

import carb
import numpy as np
import omni.kit.commands
import omni.physx.bindings._physx as physx_settings_bindings
import omni.usd
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils, Vt


def set_camera(camera_prim, eye, target):
    transform = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 1.0, 0.0)).GetInverse()
    UsdGeom.Xformable(camera_prim).AddTransformOp().Set(transform)


def hide(prim):
    if prim:
        UsdGeom.Imageable(prim).MakeInvisible()


def bind_material(stage, prim_path, material_path):
    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=Sdf.Path(prim_path),
        material_path=Sdf.Path(material_path),
        strength=None,
    )


def create_pbr_material(stage, path, color, roughness, metallic=0.0):
    created = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=created,
        bind_selected_prims=False,
        select_new_prim=False,
    )
    generated_path = Sdf.Path(created[0])
    target_path = Sdf.Path(path)
    if generated_path != target_path:
        omni.kit.commands.execute(
            "MovePrim",
            path_from=generated_path,
            path_to=target_path,
        )
    shader = UsdShade.Shader.Get(stage, target_path.AppendChild("Shader"))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    return target_path


def create_block_positions(origin, spacing, nx, ny, nz):
    """Create a regular particle block without Python triple-loop overhead."""
    grid = np.indices((nx, ny, nz), dtype=np.float32)
    points = np.stack(
        (
            origin[0] + grid[0] * spacing,
            origin[1] + grid[1] * spacing,
            origin[2] + grid[2] * spacing,
        ),
        axis=-1,
    ).reshape(-1, 3)
    return Vt.Vec3fArray.FromNumpy(points)


def create_hexagonal_emitter_layer(
    spacing, nozzle_diameter, packing_offset=(0.0, 0.0)
):
    """Create one circular, hexagonally packed cross-section in the XZ plane."""
    return create_circular_hcp_layer(spacing, nozzle_diameter, packing_offset)


def create_hcp_cylinder_positions(
    x_values, center_y, diameter, spacing, packing_offsets
):
    """Create X-axis HCP cylinder points, one circular layer per X value."""
    layers = []
    for layer_index, x_value in enumerate(x_values):
        layer = create_hexagonal_emitter_layer(
            spacing,
            diameter,
            packing_offsets[layer_index & 1],
        )
        points = np.empty((len(layer), 3), dtype=np.float32)
        points[:, 0] = x_value
        points[:, 1] = center_y + layer[:, 0]
        points[:, 2] = layer[:, 1]
        layers.append(points)
    return np.concatenate(layers, axis=0) if layers else np.empty((0, 3), dtype=np.float32)


def create_hcp_swept_tube_positions(
    spacing,
    arc_lengths,
    centers,
    normals,
    binormals,
    diameters,
    packing_offsets,
    bend_start=None,
    bend_end=None,
    bend_radius=None,
):
    """Map alternating circular HCP sections onto a swept centerline."""
    layers = []
    layer_counts = []
    for layer_index, (center, normal, binormal, diameter) in enumerate(
        zip(centers, normals, binormals, diameters)
    ):
        layer = create_circular_hcp_layer(
            spacing,
            float(diameter),
            packing_offsets[layer_index & 1],
        )
        points = (
            center[None, :]
            + layer[:, 0, None] * normal[None, :]
            + layer[:, 1, None] * binormal[None, :]
        )
        if (
            bend_start is not None
            and bend_end is not None
            and bend_radius is not None
            and bend_start < arc_lengths[layer_index] < bend_end
        ):
            # Arc-length HCP stations compress on the inside of a bend. Drop only
            # the innermost radial rows where adjacent layers would be closer than
            # 70% of nominal spacing; keep the full outer half to avoid a hollow jet.
            minimum_normal = -0.30 * bend_radius
            points = points[layer[:, 0] >= minimum_normal]
            layer = layer[layer[:, 0] >= minimum_normal]
        layers.append(points.astype(np.float32))
        layer_counts.append(len(layer))
    return (
        np.concatenate(layers, axis=0)
        if layers
        else np.empty((0, 3), dtype=np.float32),
        np.asarray(layer_counts, dtype=np.int32),
    )


def create_hcp_swept_tube_velocities(
    tangents, layer_counts, particle_flux, axial_spacing, regions=None
):
    """Initialize each swept layer with a conservative tangent velocity."""
    velocities = []
    for layer_index, (tangent, layer_count) in enumerate(zip(tangents, layer_counts)):
        if regions is not None and regions[layer_index] == "bend":
            layer_speed = 0.0
        elif regions is not None and regions[layer_index] == "straightening":
            layer_speed = 0.25 * particle_flux * axial_spacing / max(
                1, int(layer_count)
            )
        else:
            layer_speed = particle_flux * axial_spacing / max(1, int(layer_count))
        velocities.append(
            np.tile(
                (np.asarray(tangent, dtype=np.float32) * layer_speed)[None, :],
                (int(layer_count), 1),
            )
        )
    return (
        np.concatenate(velocities, axis=0)
        if velocities
        else np.empty((0, 3), dtype=np.float32)
    )


def add_swept_polygon_tube_colliders(
    stage,
    path_prefix,
    centers,
    normals,
    binormals,
    clear_radii,
    wall_thickness,
    sides,
):
    """Create one indexed polygonal shell and use exact mesh collision."""
    angles = np.arange(sides, dtype=np.float64) * (2.0 * math.pi / sides)
    polygon_scale = 1.0 / math.cos(math.pi / sides)
    inner_radii = np.asarray(clear_radii, dtype=np.float64) * polygon_scale
    outer_radii = inner_radii + wall_thickness
    inner_rings = []
    outer_rings = []
    for center, normal, binormal, inner_radius, outer_radius in zip(
        centers, normals, binormals, inner_radii, outer_radii
    ):
        directions = (
            np.cos(angles)[:, None] * normal[None, :]
            + np.sin(angles)[:, None] * binormal[None, :]
        )
        inner_rings.append(center[None, :] + inner_radius * directions)
        outer_rings.append(center[None, :] + outer_radius * directions)

    points = np.concatenate(
        [ring for pair in zip(inner_rings, outer_rings) for ring in pair],
        axis=0,
    ).astype(np.float32)

    def vertex(ring_index, outer, side_index):
        return (2 * ring_index + int(outer)) * sides + side_index

    face_counts = []
    face_indices = []
    for ring_index in range(len(centers) - 1):
        for side_index in range(sides):
            next_side = (side_index + 1) % sides
            # The inner wall faces the lumen; outer wall and radial caps make a
            # watertight, zero-seam static shell for triangle-mesh collision.
            face_counts.extend((4, 4))
            face_indices.extend(
                (
                    vertex(ring_index, False, side_index),
                    vertex(ring_index + 1, False, side_index),
                    vertex(ring_index + 1, False, next_side),
                    vertex(ring_index, False, next_side),
                    vertex(ring_index, True, side_index),
                    vertex(ring_index, True, next_side),
                    vertex(ring_index + 1, True, next_side),
                    vertex(ring_index + 1, True, side_index),
                )
            )
    for ring_index in (0, len(centers) - 1):
        for side_index in range(sides):
            next_side = (side_index + 1) % sides
            face_counts.append(4)
            if ring_index == 0:
                face_indices.extend(
                    (
                        vertex(ring_index, False, side_index),
                        vertex(ring_index, False, next_side),
                        vertex(ring_index, True, next_side),
                        vertex(ring_index, True, side_index),
                    )
                )
            else:
                face_indices.extend(
                    (
                        vertex(ring_index, False, side_index),
                        vertex(ring_index, True, side_index),
                        vertex(ring_index, True, next_side),
                        vertex(ring_index, False, next_side),
                    )
                )

    mesh = UsdGeom.Mesh.Define(stage, path_prefix)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    collision_api.CreateApproximationAttr().Set("none")
    return [mesh]


def create_shallow_pool_positions(args, fluid_rest_offset, obstacle_prim):
    """Create an HCP shallow pool near the impact and obstacle region."""
    if args.emitter_initial_pool_layers <= 0:
        return np.empty((0, 3), dtype=np.float32)
    spacing = args.spacing
    row_spacing = 0.5 * np.sqrt(3.0) * spacing
    x_min = max(-0.575 + spacing, args.stream_outlet_x - 0.16)
    x_max = min(0.575 - spacing, x_min + args.emitter_initial_pool_span_x)
    z_min = max(-0.265 + spacing, -0.5 * args.emitter_initial_pool_span_z)
    z_max = min(0.265 - spacing, 0.5 * args.emitter_initial_pool_span_z)
    layers = []
    for layer_index in range(args.emitter_initial_pool_layers):
        y = fluid_rest_offset / 0.6 + 0.001 + layer_index * spacing
        rows = []
        z_values = np.arange(z_min, z_max + 0.5 * row_spacing, row_spacing)
        for row_index, z in enumerate(z_values):
            x_offset = 0.5 * spacing if (row_index + layer_index) & 1 else 0.0
            x_values = np.arange(x_min + x_offset, x_max + 0.5 * spacing, spacing)
            points = np.empty((len(x_values), 3), dtype=np.float32)
            points[:, 0] = x_values
            points[:, 1] = y
            points[:, 2] = z
            rows.append(points)
        layers.append(np.concatenate(rows, axis=0))
    pool = np.concatenate(layers, axis=0)
    if args.obstacle_shape == "sphere":
        obstacle_center = np.asarray((0.10, 0.075, 0.0), dtype=np.float32)
        exclusion_radius = 0.075 + fluid_rest_offset / 0.6 + spacing
        pool = pool[
            np.linalg.norm(pool - obstacle_center[None, :], axis=1)
            > exclusion_radius
        ]
    else:
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        ).ComputeWorldBound(obstacle_prim)
        box = bbox.ComputeAlignedBox()
        bounds_min = np.asarray(box.GetMin(), dtype=np.float32) - 2.0 * spacing
        bounds_max = np.asarray(box.GetMax(), dtype=np.float32) + 2.0 * spacing
        inside = np.logical_and(pool >= bounds_min, pool <= bounds_max).all(axis=1)
        pool = pool[~inside]
    return pool


def intersect_mesh_with_horizontal_plane(
    points, face_counts, face_indices, plane_y, weld_tolerance, axis_x, axis_z, envelope_radius
):
    """Return closed XZ contours where a triangulated mesh crosses one Y plane."""
    if len(points) == 0 or face_counts is None or face_indices is None:
        return []
    indices = np.asarray(face_indices, dtype=np.int64)
    counts = np.asarray(face_counts, dtype=np.int64)
    if len(counts) == 0 or not np.all(counts == 3) or len(indices) != 3 * len(counts):
        return []

    triangle_indices = indices.reshape(-1, 3)
    triangles = points[triangle_indices]
    plane_epsilon = max(1.0e-8, 1.0e-5 * weld_tolerance)
    crossings_by_triangle = {}
    node_samples = {}

    # Key crossings by the indexed mesh edge, not rounded coordinates. Adjacent
    # triangles then share exactly one graph node even when their independently
    # interpolated floating-point positions straddle a quantization boundary.
    for edge_index in range(3):
        first_indices = triangle_indices[:, edge_index]
        second_indices = triangle_indices[:, (edge_index + 1) % 3]
        first = triangles[:, edge_index]
        second = triangles[:, (edge_index + 1) % 3]
        first_delta = first[:, 1] - plane_y
        second_delta = second[:, 1] - plane_y
        first_on = np.abs(first_delta) <= plane_epsilon
        second_on = np.abs(second_delta) <= plane_epsilon
        crossing = (
            ((first_delta < -plane_epsilon) & (second_delta > plane_epsilon))
            | ((first_delta > plane_epsilon) & (second_delta < -plane_epsilon))
            | (first_on & ~second_on)
            | (second_on & ~first_on)
        )
        triangle_ids = np.flatnonzero(crossing)
        for triangle_id in triangle_ids:
            first_index = int(first_indices[triangle_id])
            second_index = int(second_indices[triangle_id])
            if first_on[triangle_id]:
                node_key = ("v", first_index)
                point = first[triangle_id]
            elif second_on[triangle_id]:
                node_key = ("v", second_index)
                point = second[triangle_id]
            else:
                node_key = ("e", min(first_index, second_index), max(first_index, second_index))
                blend = first_delta[triangle_id] / (
                    first_delta[triangle_id] - second_delta[triangle_id]
                )
                point = first[triangle_id] + blend * (second[triangle_id] - first[triangle_id])
            radial = math.hypot(float(point[0]) - axis_x, float(point[2]) - axis_z)
            if radial > envelope_radius:
                continue
            xz = np.asarray((point[0], point[2]), dtype=np.float64)
            crossings_by_triangle.setdefault(int(triangle_id), {})[node_key] = xz
            node_samples.setdefault(node_key, []).append(xz)

    segments = []
    for nodes in crossings_by_triangle.values():
        if len(nodes) < 2:
            continue
        items = list(nodes.items())
        if len(items) == 2:
            pair = items
        else:
            # A plane through a degenerate/non-manifold triangle can yield more
            # than two nodes. Keep the farthest pair rather than inventing a fan.
            best = None
            for first_index in range(len(items) - 1):
                for second_index in range(first_index + 1, len(items)):
                    distance = np.linalg.norm(
                        items[first_index][1] - items[second_index][1]
                    )
                    if best is None or distance > best[0]:
                        best = (distance, items[first_index], items[second_index])
            pair = [best[1], best[2]]
        if np.linalg.norm(pair[0][1] - pair[1][1]) > 0.05 * weld_tolerance:
            segments.append((pair[0][1], pair[1][1]))
    if not segments:
        return []

    # PhysX can duplicate isosurface vertices across triangle patches, so indexed
    # edge identity alone is insufficient. Spatially weld endpoints by checking
    # the current and eight neighboring hash cells, avoiding quantization seams.
    endpoints = np.asarray(
        [point for segment in segments for point in segment], dtype=np.float64
    )
    parents = np.arange(len(endpoints), dtype=np.int64)

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return int(node)

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    cells = {}
    for endpoint_index, point in enumerate(endpoints):
        cell = tuple(np.floor(point / weld_tolerance).astype(np.int64))
        for offset_x in (-1, 0, 1):
            for offset_z in (-1, 0, 1):
                for candidate in cells.get(
                    (cell[0] + offset_x, cell[1] + offset_z), ()
                ):
                    if np.linalg.norm(point - endpoints[candidate]) <= weld_tolerance:
                        union(endpoint_index, candidate)
        cells.setdefault(cell, []).append(endpoint_index)

    cluster_samples = {}
    for endpoint_index, point in enumerate(endpoints):
        cluster_samples.setdefault(find(endpoint_index), []).append(point)
    positions = {
        cluster: np.mean(samples, axis=0)
        for cluster, samples in cluster_samples.items()
    }
    adjacency = {cluster: set() for cluster in positions}
    for segment_index in range(len(segments)):
        first_key = find(2 * segment_index)
        second_key = find(2 * segment_index + 1)
        if first_key == second_key:
            continue
        adjacency[first_key].add(second_key)
        adjacency[second_key].add(first_key)

    contours = []
    visited_edges = set()
    for start, neighbors in adjacency.items():
        if len(neighbors) != 2:
            continue
        for first_neighbor in neighbors:
            edge = frozenset((start, first_neighbor))
            if edge in visited_edges:
                continue
            contour_keys = [start]
            previous = start
            current = first_neighbor
            closed = False
            for _ in range(len(adjacency) + 1):
                contour_keys.append(current)
                visited_edges.add(frozenset((previous, current)))
                if current == start:
                    closed = True
                    break
                current_neighbors = adjacency.get(current, set())
                if len(current_neighbors) != 2:
                    break
                next_candidates = [candidate for candidate in current_neighbors if candidate != previous]
                if not next_candidates:
                    break
                previous, current = current, next_candidates[0]
            if closed and len(contour_keys) >= 4:
                contours.append(
                    np.asarray(
                        [positions[node_key] for node_key in contour_keys[:-1]],
                        dtype=np.float64,
                    )
                )
    return contours


def analyze_closed_contour(contour):
    """Return area, centroid, perimeter, circularity, and high-frequency roughness."""
    x = contour[:, 0]
    z = contour[:, 1]
    x_next = np.roll(x, -1)
    z_next = np.roll(z, -1)
    cross = x * z_next - x_next * z
    signed_area = 0.5 * np.sum(cross)
    area = abs(signed_area)
    if area <= 1.0e-12:
        return None
    centroid_x = np.sum((x + x_next) * cross) / (6.0 * signed_area)
    centroid_z = np.sum((z + z_next) * cross) / (6.0 * signed_area)
    perimeter = float(
        np.sum(np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1))
    )
    circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0e-12)
    centered = contour - np.asarray((centroid_x, centroid_z))
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    radii = np.linalg.norm(centered, axis=1)
    order = np.argsort(angles)
    angles = angles[order]
    radii = radii[order]
    sample_angles = np.linspace(-math.pi, math.pi, 128, endpoint=False)
    angles_extended = np.concatenate((angles - 2.0 * math.pi, angles, angles + 2.0 * math.pi))
    radii_extended = np.tile(radii, 3)
    sampled_radii = np.interp(sample_angles, angles_extended, radii_extended)
    design = np.column_stack(
        (
            np.ones_like(sample_angles),
            np.cos(sample_angles),
            np.sin(sample_angles),
            np.cos(2.0 * sample_angles),
            np.sin(2.0 * sample_angles),
        )
    )
    fitted = design @ np.linalg.lstsq(design, sampled_radii, rcond=None)[0]
    roughness = float(
        np.sqrt(np.mean((sampled_radii - fitted) ** 2))
        / max(np.mean(sampled_radii), 1.0e-12)
    )
    return {
        "area_m2": float(area),
        "centroid_x": float(centroid_x),
        "centroid_z": float(centroid_z),
        "perimeter_m": perimeter,
        "equivalent_diameter_m": float(2.0 * math.sqrt(area / math.pi)),
        "circularity": float(circularity),
        "isosurface_roughness_rms": roughness,
    }


def analyze_jet_slices(
    args,
    mesh_points,
    face_counts,
    face_indices,
    particle_points,
    particle_velocities,
    grid_spacing,
):
    """Measure visible free-jet shape and velocity against inviscid theory."""
    results = []
    slab_half = 0.5 * args.jet_slice_thickness_spacings * args.spacing
    weld_tolerance = max(0.25 * grid_spacing, 0.25 * args.spacing)
    for fraction in args.jet_slice_fractions_values:
        depth = args.emitter_height * fraction
        plane_y = args.emitter_height - depth
        theory_speed = math.sqrt(args.emitter_speed**2 + 2.0 * 9.81 * depth)
        theory_diameter = args.emitter_nozzle_diameter * math.sqrt(
            args.emitter_speed / theory_speed
        )
        contours = intersect_mesh_with_horizontal_plane(
            mesh_points,
            face_counts,
            face_indices,
            plane_y,
            weld_tolerance,
            args.stream_outlet_x,
            0.0,
            2.5 * args.emitter_nozzle_diameter,
        )
        analyzed = [analyze_closed_contour(contour) for contour in contours]
        analyzed = [entry for entry in analyzed if entry is not None]
        analyzed.sort(key=lambda entry: entry["area_m2"], reverse=True)
        valid_contours = [
            entry
            for entry in analyzed
            if math.hypot(
                entry["centroid_x"] - args.stream_outlet_x,
                entry["centroid_z"],
            )
            <= 0.75 * args.emitter_nozzle_diameter
            and entry["equivalent_diameter_m"] >= 0.5 * theory_diameter
        ]
        result = {
            "height_fraction": fraction,
            "depth_m": depth,
            "plane_y": plane_y,
            "theory_speed_mps": theory_speed,
            "theory_diameter_m": theory_diameter,
            "valid": bool(valid_contours),
        }
        if not valid_contours:
            result["invalid_reason"] = "no_closed_axis_contour"
            results.append(result)
            continue
        main = valid_contours[0]
        satellite_area = sum(entry["area_m2"] for entry in valid_contours[1:])
        result.update(main)
        result.update(
            {
                "diameter_relative_error": abs(
                    main["equivalent_diameter_m"] - theory_diameter
                )
                / theory_diameter,
                "centerline_drift_m": math.hypot(
                    main["centroid_x"] - args.stream_outlet_x,
                    main["centroid_z"],
                ),
                "centerline_drift_d0": math.hypot(
                    main["centroid_x"] - args.stream_outlet_x,
                    main["centroid_z"],
                )
                / args.emitter_nozzle_diameter,
                "satellite_loop_count": max(0, len(valid_contours) - 1),
                "satellite_area_ratio": satellite_area / main["area_m2"],
            }
        )
        slab_mask = (
            (np.abs(particle_points[:, 1] - plane_y) <= slab_half)
            & (
                np.hypot(
                    particle_points[:, 0] - main["centroid_x"],
                    particle_points[:, 2] - main["centroid_z"],
                )
                <= 0.75 * max(main["equivalent_diameter_m"], theory_diameter)
            )
        )
        result["particle_count"] = int(np.count_nonzero(slab_mask))
        if particle_velocities is not None and np.any(slab_mask):
            velocities = particle_velocities[slab_mask]
            downward = -velocities[:, 1]
            positive = downward > 0.0
            if np.any(positive):
                mean_axial = float(np.mean(downward[positive]))
                transverse_rms = float(
                    np.sqrt(np.mean(np.sum(velocities[positive][:, (0, 2)] ** 2, axis=1)))
                )
                result.update(
                    {
                        "mean_axial_speed_mps": mean_axial,
                        "axial_speed_relative_error": abs(mean_axial - theory_speed)
                        / theory_speed,
                        "transverse_speed_rms_mps": transverse_rms,
                        "transverse_to_axial_ratio": transverse_rms
                        / max(mean_axial, 1.0e-12),
                    }
                )
        results.append(result)
    return results


def aggregate_jet_slice_metrics(slice_metrics):
    valid = [entry for entry in slice_metrics if entry.get("valid")]
    aggregate = {"jet_slice_valid_fraction": len(valid) / max(1, len(slice_metrics))}
    fields = {
        "diameter_relative_error": ("jet_diameter_error_median", "jet_diameter_error_max"),
        "axial_speed_relative_error": ("jet_speed_error_median", "jet_speed_error_max"),
        "centerline_drift_d0": ("jet_centerline_drift_median_d0", "jet_centerline_drift_max_d0"),
        "circularity": ("jet_circularity_median", "jet_circularity_min"),
        "transverse_to_axial_ratio": ("jet_transverse_axial_ratio_median", "jet_transverse_axial_ratio_max"),
        "isosurface_roughness_rms": ("jet_roughness_median", "jet_roughness_max"),
    }
    for field, (median_name, extreme_name) in fields.items():
        values = [entry[field] for entry in valid if field in entry]
        if not values:
            continue
        aggregate[median_name] = float(np.median(values))
        aggregate[extreme_name] = float(min(values) if field == "circularity" else max(values))
    return aggregate


def aggregate_jet_capture_metrics(captured_metrics, warmup_frame, steady_capture_count=3):
    """Aggregate the final steady captures and report per-slice temporal stability."""
    eligible = [
        metrics
        for metrics in captured_metrics
        if metrics.get("sim_frame", -1) >= warmup_frame
    ]
    selected = eligible[-steady_capture_count:]
    summary = {
        "warmup_frame": warmup_frame,
        "eligible_capture_count": len(eligible),
        "steady_capture_count": len(selected),
        "steady_capture_frames": [entry["sim_frame"] for entry in selected],
    }
    if not selected:
        summary["valid"] = False
        summary["invalid_reason"] = "no_capture_after_warmup"
        return summary

    fractions = sorted(
        {
            slice_metrics["height_fraction"]
            for metrics in selected
            for slice_metrics in metrics.get("jet_slices", [])
        }
    )
    temporal_slices = []
    diameter_cvs = []
    for fraction in fractions:
        samples = []
        for metrics in selected:
            sample = next(
                (
                    entry
                    for entry in metrics.get("jet_slices", [])
                    if abs(entry["height_fraction"] - fraction) <= 1.0e-9
                ),
                None,
            )
            if sample is not None:
                samples.append(sample)
        valid_samples = [sample for sample in samples if sample.get("valid")]
        temporal = {
            "height_fraction": fraction,
            "sample_count": len(samples),
            "valid_sample_count": len(valid_samples),
            "valid_fraction": len(valid_samples) / max(1, len(samples)),
        }
        diameters = [
            sample["equivalent_diameter_m"]
            for sample in valid_samples
            if "equivalent_diameter_m" in sample
        ]
        if diameters:
            mean_diameter = float(np.mean(diameters))
            diameter_cv = float(
                np.std(diameters) / max(mean_diameter, 1.0e-12)
            )
            temporal.update(
                {
                    "diameter_mean_m": mean_diameter,
                    "diameter_cv": diameter_cv,
                }
            )
            diameter_cvs.append(diameter_cv)
        temporal_slices.append(temporal)

    steady_slices = [
        entry
        for metrics in selected
        for entry in metrics.get("jet_slices", [])
    ]
    summary.update(aggregate_jet_slice_metrics(steady_slices))
    summary["jet_temporal_slices"] = temporal_slices
    if diameter_cvs:
        summary["jet_diameter_cv_max"] = max(diameter_cvs)
    summary["valid"] = bool(temporal_slices) and all(
        entry["valid_sample_count"] == len(selected)
        for entry in temporal_slices
    )
    return summary


def add_collider_hexahedron(stage, path, points):
    """Create one closed convex hexahedron and use its hull for collision."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4, 4, 4, 4, 4, 4]))
    mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray(
            [
                0, 1, 2, 3,
                4, 7, 6, 5,
                0, 4, 5, 1,
                1, 5, 6, 2,
                2, 6, 7, 3,
                3, 7, 4, 0,
            ]
        )
    )
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    collision_api.CreateApproximationAttr().Set("convexHull")
    return mesh


def create_binary_stl_mesh(stage, prim_path, file_path, target_height):
    """Load a binary STL as a welded, indexed USD mesh in Y-up coordinates."""
    with open(file_path, "rb") as stl_file:
        stl_data = stl_file.read()
    if len(stl_data) < 84:
        raise ValueError(f"STL file is too small: {file_path}")
    triangle_count = struct.unpack_from("<I", stl_data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(stl_data) != expected_size:
        raise ValueError(
            f"Expected binary STL size {expected_size}, got {len(stl_data)}: {file_path}"
        )

    record_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    records = np.frombuffer(
        stl_data,
        dtype=record_dtype,
        count=triangle_count,
        offset=84,
    )
    triangle_vertices = records["vertices"].reshape(-1, 3)
    points, face_vertex_indices = np.unique(
        triangle_vertices,
        axis=0,
        return_inverse=True,
    )

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    source_extents = bounds_max - bounds_min
    if source_extents[1] <= 0.0:
        raise ValueError(f"STL has no Y extent: {file_path}")
    mesh_scale = target_height / source_extents[1]
    source_origin = np.array(
        [
            0.5 * (bounds_min[0] + bounds_max[0]),
            bounds_min[1],
            0.5 * (bounds_min[2] + bounds_max[2]),
        ],
        dtype=np.float32,
    )
    points = ((points - source_origin) * mesh_scale).astype(np.float32)

    normals = records["normal"].astype(np.float32)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1.0e-8)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set(
        Vt.IntArray.FromNumpy(
            np.full(triangle_count, 3, dtype=np.int32)
        )
    )
    mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(face_vertex_indices.astype(np.int32))
    )
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(False)

    return mesh, len(points), triangle_count, source_extents * mesh_scale


os.makedirs(args.output, exist_ok=True)
context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()

UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# GPU PhysX scene in SI units.
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
physx_scene.CreateEnableExternalForcesEveryIterationAttr().Set(True)
if args.source_mode == "emitter":
    physx_scene.CreateTimeStepsPerSecondAttr().Set(60 * args.substeps)

basin_material = create_pbr_material(
    stage,
    "/World/Looks/Basin",
    Gf.Vec3f(0.32, 0.35, 0.39),
    roughness=0.24,
    metallic=0.05,
)
if args.mirror_obstacle:
    obstacle_material = create_pbr_material(
        stage,
        "/World/Looks/Obstacle",
        Gf.Vec3f(0.92, 0.92, 0.92),
        roughness=args.mirror_roughness,
        metallic=1.0,
    )
else:
    obstacle_material = create_pbr_material(
        stage,
        "/World/Looks/Obstacle",
        Gf.Vec3f(0.08, 0.10, 0.13),
        roughness=0.12,
        metallic=0.75,
    )

# Open-front basin.  Geometry dimensions are full extents.
floor = physicsUtils.add_collider_cube(
    stage,
    "/World/BasinFloor",
    Gf.Vec3f(1.20, 0.025, 0.58),
    Gf.Vec3f(0.0, -0.0125, 0.0),
)
source_colliders = []
stream_gate = None
stream_settle_colliders = []
emitter_piston = None
emitter_piston_translate_attr = None
emitter_piston_start = None
source_colliders.append(
    physicsUtils.add_collider_cube(
        stage,
        "/World/BasinLeft",
        Gf.Vec3f(0.025, 0.34, 0.58),
        Gf.Vec3f(-0.60, 0.17, 0.0),
    )
)
if args.source_mode == "emitter":
    # The visible horizontal pipe is render-only.  The physical supply is a
    # compact pressure reservoir beside the elbow, keeping hidden liquid out of
    # the multi-meter isosurface footprint used by the legacy implementation.
    emitter_feed_y = args.emitter_height + emitter_plan.emitter_feed_clearance
    tube_inner_half = (
        0.5 * args.emitter_nozzle_diameter
        + 0.5 * args.spacing / 0.6
        + max(0.001, 0.5 * args.spacing)
    )
    tube_wall_thickness = 0.006
    tube_outer_width = 2.0 * (tube_inner_half + tube_wall_thickness)
    emitter_centerline = build_emitter_centerline(
        args.stream_outlet_x,
        args.emitter_height,
        emitter_plan.emitter_feed_clearance,
        emitter_plan.contraction_length,
        emitter_plan.bend_radius,
        args.emitter_nozzle_diameter,
    )

    nozzle = UsdGeom.Cylinder.Define(stage, "/World/EmitterNozzle")
    nozzle.CreateAxisAttr().Set(UsdGeom.Tokens.y)
    nozzle.CreateRadiusAttr().Set(tube_inner_half + tube_wall_thickness)
    nozzle.CreateHeightAttr().Set(emitter_plan.straightening_length + 0.02)
    nozzle.AddTranslateOp().Set(
        Gf.Vec3d(
            args.stream_outlet_x,
            args.emitter_height + 0.5 * emitter_plan.straightening_length,
            0.0,
        )
    )
    bind_material(stage, nozzle.GetPath(), basin_material)

    reservoir_wall = max(0.006, 2.0 * max(0.001, 0.5 * args.spacing))
    reservoir_inner_length = emitter_plan.reservoir_inner_length
    reservoir_inner_diameter = emitter_plan.reservoir_inner_diameter
    reservoir_right_x = emitter_centerline["contraction_start_x"]
    reservoir_left_x = reservoir_right_x - reservoir_inner_length
    reservoir_center_x = 0.5 * (reservoir_left_x + reservoir_right_x)
    reservoir_half = 0.5 * reservoir_inner_diameter
    reservoir_shell = [
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterReservoirTop",
            Gf.Vec3f(reservoir_inner_length + 2.0 * reservoir_wall, reservoir_wall, reservoir_inner_diameter + 2.0 * reservoir_wall),
            Gf.Vec3f(reservoir_center_x, emitter_feed_y + reservoir_half + 0.5 * reservoir_wall, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterReservoirBottom",
            Gf.Vec3f(reservoir_inner_length + 2.0 * reservoir_wall, reservoir_wall, reservoir_inner_diameter + 2.0 * reservoir_wall),
            Gf.Vec3f(reservoir_center_x, emitter_feed_y - reservoir_half - 0.5 * reservoir_wall, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterReservoirBack",
            Gf.Vec3f(reservoir_inner_length + 2.0 * reservoir_wall, reservoir_inner_diameter, reservoir_wall),
            Gf.Vec3f(reservoir_center_x, emitter_feed_y, -reservoir_half - 0.5 * reservoir_wall),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterReservoirFront",
            Gf.Vec3f(reservoir_inner_length + 2.0 * reservoir_wall, reservoir_inner_diameter, reservoir_wall),
            Gf.Vec3f(reservoir_center_x, emitter_feed_y, reservoir_half + 0.5 * reservoir_wall),
        ),
    ]
    source_colliders.extend(reservoir_shell)
    for supply_collider in reservoir_shell:
        hide(supply_collider)

    # One shared centerline controls the fluid seed and the 16-sided physical
    # channel. Segment stations include analytic boundaries and uniformly sampled
    # bend rings so each convex sector remains a close approximation to the arc.
    collider_arc_lengths = list(
        np.arange(
            0.0,
            emitter_centerline["total_length"]
            + 0.5 * emitter_plan.axial_spacing,
            max(4.0 * emitter_plan.axial_spacing, 0.25 * args.emitter_nozzle_diameter),
        )
    )
    collider_arc_lengths.extend(
        np.linspace(
            emitter_plan.contraction_length,
            emitter_plan.contraction_length + emitter_plan.bend_arc_length,
            args.emitter_bend_segments + 1,
        )
    )
    collider_arc_lengths.extend(
        (
            0.0,
            emitter_plan.contraction_length,
            emitter_plan.contraction_length + emitter_plan.bend_arc_length,
            emitter_centerline["total_length"],
        )
    )
    collider_arc_lengths = np.unique(
        np.clip(collider_arc_lengths, 0.0, emitter_centerline["total_length"])
    )
    (
        collider_centers,
        collider_tangents,
        collider_regions,
    ) = evaluate_emitter_centerline(emitter_centerline, collider_arc_lengths)
    collider_normals, collider_binormals = parallel_transport_frames(
        collider_tangents
    )
    collider_particle_diameters = emitter_path_particle_diameters(
        emitter_centerline,
        collider_arc_lengths,
        emitter_plan.reservoir_diameter,
    )
    collider_clear_radii = (
        0.5 * collider_particle_diameters
        + 0.5 * args.spacing / 0.6
        + max(0.001, 0.5 * args.spacing)
    )
    bend_start = emitter_plan.contraction_length
    bend_end = bend_start + emitter_plan.bend_arc_length
    collider_clear_radii[
        (collider_arc_lengths >= bend_start - 1.0e-12)
        & (collider_arc_lengths <= bend_end + 1.0e-12)
    ] += 1.5 * args.spacing
    emitter_tube_colliders = add_swept_polygon_tube_colliders(
        stage,
        "/World/EmitterSweptTube",
        collider_centers,
        collider_normals,
        collider_binormals,
        collider_clear_radii,
        tube_wall_thickness,
        args.emitter_pipe_sides,
    )
    source_colliders.extend(emitter_tube_colliders)
    for tube_collider in emitter_tube_colliders:
        hide(tube_collider.GetPrim())

    contraction_left_x = reservoir_right_x
    contraction_right_x = args.stream_outlet_x
    # A large-area pusher supplies nozzle flux at a small, contact-safe speed.
    piston_thickness = 2.0 * args.spacing
    emitter_piston = physicsUtils.add_collider_cube(
        stage,
        "/World/EmitterPiston",
        Gf.Vec3f(
            piston_thickness,
            reservoir_inner_diameter - 0.5 * args.spacing,
            reservoir_inner_diameter - 0.5 * args.spacing,
        ),
        Gf.Vec3f(
            reservoir_left_x + 0.5 * piston_thickness,
            emitter_feed_y,
            0.0,
        ),
    )
    piston_rigid_api = UsdPhysics.RigidBodyAPI.Apply(emitter_piston.GetPrim())
    piston_rigid_api.CreateKinematicEnabledAttr().Set(True)
    piston_physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(emitter_piston.GetPrim())
    piston_physx_api.CreateDisableGravityAttr().Set(True)
    emitter_piston_start = Gf.Vec3d(
        reservoir_left_x + 0.5 * piston_thickness,
        emitter_feed_y,
        0.0,
    )
    emitter_piston_translate_attr = emitter_piston.GetPrim().GetAttribute("xformOp:translate")
    source_colliders.append(emitter_piston)
    hide(emitter_piston)

    # Opaque housing is required because hiding colliders does not hide the
    # reservoir liquid generated by the shared particle-system isosurface.
    housing_margin = max(0.004, 2.0 * max(0.001, 0.5 * args.spacing))
    housing_left_x = reservoir_left_x - reservoir_wall - housing_margin
    housing_right_x = contraction_right_x + housing_margin
    housing = UsdGeom.Cube.Define(stage, "/World/EmitterReservoirHousing")
    housing.CreateSizeAttr().Set(1.0)
    housing.AddTranslateOp().Set(
        Gf.Vec3d(0.5 * (housing_left_x + housing_right_x), emitter_feed_y, 0.0)
    )
    housing.AddScaleOp().Set(
        Gf.Vec3d(
            housing_right_x - housing_left_x,
            reservoir_inner_diameter + 2.0 * (reservoir_wall + housing_margin),
            reservoir_inner_diameter + 2.0 * (reservoir_wall + housing_margin),
        )
    )
    bind_material(stage, housing.GetPath(), basin_material)

    # Preserve the established pipe silhouette without physical water inside it.
    visual_feed_left_x = -0.82
    visual_feed_right_x = args.stream_outlet_x
    visual_feed_length = visual_feed_right_x - visual_feed_left_x
    feed_cover = UsdGeom.Cylinder.Define(stage, "/World/EmitterFeedCover")
    feed_cover.CreateAxisAttr().Set(UsdGeom.Tokens.x)
    feed_cover.CreateRadiusAttr().Set(0.75 * tube_outer_width)
    feed_cover.CreateHeightAttr().Set(visual_feed_length)
    feed_cover.AddTranslateOp().Set(
        Gf.Vec3d(0.5 * (visual_feed_left_x + visual_feed_right_x), emitter_feed_y, 0.0)
    )
    bind_material(stage, feed_cover.GetPath(), basin_material)
elif args.source_mode == "stream":
    aperture_half = 0.5 * args.stream_aperture
    tank_left = -0.55
    tank_right = 0.05
    tank_back = -0.14
    tank_front = 0.14
    outlet_left = args.stream_outlet_x - aperture_half
    outlet_right = args.stream_outlet_x + aperture_half
    if (
        outlet_left <= tank_left
        or outlet_right >= tank_right
        or args.stream_aperture >= tank_front - tank_back
    ):
        raise ValueError("Stream aperture must fit inside the overhead reservoir")

    bottom_y = args.stream_reservoir_bottom
    floor_center_y = bottom_y - 0.0125
    left_floor_width = outlet_left - tank_left
    right_floor_width = tank_right - outlet_right
    side_floor_depth = 0.5 * (
        tank_front - tank_back - args.stream_aperture
    )
    side_floor_center = aperture_half + 0.5 * side_floor_depth
    # Let the front/back plates overlap the left/right plates slightly.  This
    # removes four collider seams around the outlet without narrowing the
    # actual square opening.
    cross_plate_width = args.stream_aperture + 0.020
    wall_bottom = bottom_y - 0.025
    wall_top = bottom_y + 0.35
    wall_height = wall_top - wall_bottom
    wall_center_y = 0.5 * (wall_bottom + wall_top)
    source_colliders.extend(
        [
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyBottomLeft",
                Gf.Vec3f(left_floor_width, 0.025, 0.28),
                Gf.Vec3f(
                    0.5 * (tank_left + outlet_left), floor_center_y, 0.0
                ),
            ),
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyBottomRight",
                Gf.Vec3f(right_floor_width, 0.025, 0.28),
                Gf.Vec3f(
                    0.5 * (outlet_right + tank_right), floor_center_y, 0.0
                ),
            ),
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyBottomBack",
                Gf.Vec3f(
                    cross_plate_width, 0.025, side_floor_depth
                ),
                Gf.Vec3f(
                    args.stream_outlet_x,
                    floor_center_y,
                    -side_floor_center,
                ),
            ),
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyBottomFront",
                Gf.Vec3f(
                    cross_plate_width, 0.025, side_floor_depth
                ),
                Gf.Vec3f(
                    args.stream_outlet_x,
                    floor_center_y,
                    side_floor_center,
                ),
            ),
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyLeft",
                Gf.Vec3f(0.025, wall_height, 0.28),
                Gf.Vec3f(tank_left, wall_center_y, 0.0),
            ),
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyRight",
                Gf.Vec3f(0.025, wall_height, 0.28),
                Gf.Vec3f(tank_right, wall_center_y, 0.0),
            ),
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyBack",
                Gf.Vec3f(0.60, wall_height, 0.025),
                Gf.Vec3f(-0.25, wall_center_y, tank_back),
            ),
            physicsUtils.add_collider_cube(
                stage,
                "/World/SupplyFront",
                Gf.Vec3f(0.60, wall_height, 0.025),
                Gf.Vec3f(-0.25, wall_center_y, tank_front),
            ),
        ]
    )
    # A temporary invisible valve seals the aperture while the initially
    # authored particle lattice relaxes to hydrostatic equilibrium.  Its
    # collision is disabled only after the unrecorded settling phase.
    stream_gate = physicsUtils.add_collider_cube(
        stage,
        "/World/SupplyGate",
        Gf.Vec3f(0.65, 0.20, 0.33),
        Gf.Vec3f(-0.25, bottom_y - 0.10, 0.0),
    )
    source_colliders.append(stream_gate)
    hide(stream_gate)
    settle_wall_bottom = bottom_y - 0.10
    settle_wall_top = bottom_y + 0.60
    settle_wall_height = settle_wall_top - settle_wall_bottom
    settle_wall_center_y = 0.5 * (settle_wall_bottom + settle_wall_top)
    stream_settle_colliders = [
        stream_gate,
        physicsUtils.add_collider_cube(
            stage,
            "/World/SupplySettleLeft",
            Gf.Vec3f(0.050, settle_wall_height, 0.33),
            Gf.Vec3f(tank_left, settle_wall_center_y, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/SupplySettleRight",
            Gf.Vec3f(0.050, settle_wall_height, 0.33),
            Gf.Vec3f(tank_right, settle_wall_center_y, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/SupplySettleBack",
            Gf.Vec3f(0.65, settle_wall_height, 0.050),
            Gf.Vec3f(-0.25, settle_wall_center_y, tank_back),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/SupplySettleFront",
            Gf.Vec3f(0.65, settle_wall_height, 0.050),
            Gf.Vec3f(-0.25, settle_wall_center_y, tank_front),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/SupplySettleCeiling",
            Gf.Vec3f(0.65, 0.050, 0.33),
            Gf.Vec3f(-0.25, settle_wall_top, 0.0),
        ),
    ]
    for settle_collider in stream_settle_colliders[1:]:
        source_colliders.append(settle_collider)
        hide(settle_collider)
    # Keep the near wall as a collider, but render the reservoir as a cutaway
    # so the preallocated water body and the outlet remain visible.
    hide(stage.GetPrimAtPath("/World/SupplyFront"))
right_wall = physicsUtils.add_collider_cube(
    stage,
    "/World/BasinRight",
    Gf.Vec3f(0.025, 0.34, 0.58),
    Gf.Vec3f(0.60, 0.17, 0.0),
)
back_wall = physicsUtils.add_collider_cube(
    stage,
    "/World/BasinBack",
    Gf.Vec3f(1.20, 0.34, 0.025),
    Gf.Vec3f(0.0, 0.17, -0.29),
)
front_wall = physicsUtils.add_collider_cube(
    stage,
    "/World/BasinFrontCollider",
    Gf.Vec3f(1.20, 0.34, 0.025),
    Gf.Vec3f(0.0, 0.17, 0.29),
)
hide(front_wall)
if args.source_mode == "emitter":
    # Invisible upper guards contain rare high-energy droplets so a few escaped
    # particles cannot stretch the isosurface allocation across several meters.
    splash_guard_height = args.emitter_height + 0.20
    splash_guards = [
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinLeftSplashGuard",
            Gf.Vec3f(0.025, splash_guard_height, 0.58),
            Gf.Vec3f(-0.60, 0.5 * splash_guard_height, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinRightSplashGuard",
            Gf.Vec3f(0.025, splash_guard_height, 0.58),
            Gf.Vec3f(0.60, 0.5 * splash_guard_height, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinBackSplashGuard",
            Gf.Vec3f(1.20, splash_guard_height, 0.025),
            Gf.Vec3f(0.0, 0.5 * splash_guard_height, -0.29),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinFrontSplashGuard",
            Gf.Vec3f(1.20, splash_guard_height, 0.025),
            Gf.Vec3f(0.0, 0.5 * splash_guard_height, 0.29),
        ),
    ]
    for guard in splash_guards:
        source_colliders.append(guard)
        hide(guard)
    # A second, larger catch tray prevents a handful of tunneled droplets from
    # expanding the global isosurface bounds even if they cross the primary basin.
    safety_floor = physicsUtils.add_collider_cube(
        stage,
        "/World/BasinSafetyFloor",
        Gf.Vec3f(1.60, 0.200, 0.80),
        Gf.Vec3f(0.0, -0.100, 0.0),
    )
    safety_wall_height = args.emitter_height + 0.50
    safety_guards = [
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinSafetyLeft",
            Gf.Vec3f(0.200, safety_wall_height, 0.80),
            Gf.Vec3f(-0.70, 0.5 * safety_wall_height - 0.10, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinSafetyRight",
            Gf.Vec3f(0.200, safety_wall_height, 0.80),
            Gf.Vec3f(0.70, 0.5 * safety_wall_height - 0.10, 0.0),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinSafetyBack",
            Gf.Vec3f(1.60, safety_wall_height, 0.200),
            Gf.Vec3f(0.0, 0.5 * safety_wall_height - 0.10, -0.39),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/BasinSafetyFront",
            Gf.Vec3f(1.60, safety_wall_height, 0.200),
            Gf.Vec3f(0.0, 0.5 * safety_wall_height - 0.10, 0.39),
        ),
    ]
    safety_ceiling_bottom = args.emitter_height + 0.40
    safety_ceiling = physicsUtils.add_collider_cube(
        stage,
        "/World/BasinSafetyCeiling",
        Gf.Vec3f(1.60, 0.100, 0.80),
        Gf.Vec3f(0.0, safety_ceiling_bottom + 0.050, 0.0),
    )
    source_colliders.append(safety_floor)
    source_colliders.append(safety_ceiling)
    source_colliders.extend(safety_guards)
    hide(safety_floor)
    hide(safety_ceiling)
    for guard in safety_guards:
        hide(guard)
for collider in (floor, right_wall, back_wall, *source_colliders):
    bind_material(stage, collider.GetPath(), basin_material)

# The obstacle stays static by default for backwards compatibility, but can be
# made a low-friction dynamic rigid body so particle contacts push it.
obstacle_mesh_info = ""
if args.obstacle_shape == "sphere":
    obstacle = UsdGeom.Sphere.Define(stage, "/World/FlowObstacle")
    obstacle.CreateRadiusAttr().Set(0.075)
    obstacle.AddTranslateOp().Set(Gf.Vec3d(0.10, 0.075, 0.0))
    obstacle_prim = obstacle.GetPrim()
    obstacle_collision_prim = obstacle_prim
    UsdPhysics.CollisionAPI.Apply(obstacle_collision_prim)
    bind_material(stage, obstacle.GetPath(), obstacle_material)
else:
    obstacle = UsdGeom.Xform.Define(stage, "/World/FlowObstacle")
    obstacle.AddTranslateOp().Set(Gf.Vec3d(0.10, 0.001, 0.0))
    obstacle_prim = obstacle.GetPrim()
    elephant_mesh, mesh_vertex_count, mesh_triangle_count, mesh_extents = (
        create_binary_stl_mesh(
            stage,
            "/World/FlowObstacle/ElephantMesh",
            args.obstacle_mesh,
            args.obstacle_height,
        )
    )
    obstacle_collision_prim = elephant_mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(obstacle_collision_prim)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(
        obstacle_collision_prim
    )
    mesh_collision_api.CreateApproximationAttr().Set(
        args.obstacle_collision
    )
    bind_material(stage, elephant_mesh.GetPath(), obstacle_material)
    obstacle_mesh_info = (
        f", mesh_vertices={mesh_vertex_count}, mesh_triangles={mesh_triangle_count}, "
        f"mesh_extents=({mesh_extents[0]:.3f},{mesh_extents[1]:.3f},"
        f"{mesh_extents[2]:.3f})m"
    )

obstacle_rigid_api = None
if args.dynamic_obstacle:
    obstacle_rigid_api = UsdPhysics.RigidBodyAPI.Apply(obstacle_prim)
    obstacle_rigid_api.CreateRigidBodyEnabledAttr().Set(True)
    obstacle_rigid_api.CreateKinematicEnabledAttr().Set(False)

    obstacle_mass_api = UsdPhysics.MassAPI.Apply(obstacle_prim)
    obstacle_mass_api.CreateMassAttr().Set(args.obstacle_mass)

    obstacle_physics_material = UsdShade.Material.Define(
        stage, "/World/Looks/ObstaclePhysics"
    )
    obstacle_material_api = UsdPhysics.MaterialAPI.Apply(
        obstacle_physics_material.GetPrim()
    )
    obstacle_material_api.CreateStaticFrictionAttr().Set(
        args.obstacle_static_friction
    )
    obstacle_material_api.CreateDynamicFrictionAttr().Set(
        args.obstacle_dynamic_friction
    )
    obstacle_material_api.CreateRestitutionAttr().Set(args.obstacle_restitution)
    physicsUtils.add_physics_material_to_prim(
        stage,
        obstacle_collision_prim,
        obstacle_physics_material.GetPath(),
    )

# PhysX uses a rest distance of two fluid-rest-offsets.
fluid_rest_offset = 0.5 * args.spacing
particle_contact_offset = fluid_rest_offset / 0.6
rest_offset = particle_contact_offset
particle_system_path = Sdf.Path("/World/ParticleSystem")
particle_system = particleUtils.add_physx_particle_system(
    stage,
    particle_system_path,
    simulation_owner=scene.GetPath(),
    contact_offset=rest_offset + 0.001,
    rest_offset=rest_offset,
    particle_contact_offset=particle_contact_offset,
    solid_rest_offset=rest_offset,
    fluid_rest_offset=fluid_rest_offset,
    enable_ccd=args.source_mode == "emitter",
    solver_position_iterations=args.solver_iterations,
    max_neighborhood=96,
    neighborhood_scale=1.01,
    max_velocity=emitter_plan.max_velocity if args.source_mode == "emitter" else 5.0,
)

# Water-like values based on PhysX 5's SnippetPBF.  Vorticity is deliberately
# reduced from the snippet's showcase value of 10 to avoid perpetual agitation.
pbd_material_path = Sdf.Path("/World/Looks/WaterPhysics")
particleUtils.add_pbd_particle_material(
    stage,
    pbd_material_path,
    density=1000.0,
    friction=0.05,
    damping=0.02 if args.source_mode == "emitter" else 0.05,
    viscosity=0.001,
    vorticity_confinement=0.0 if args.source_mode == "emitter" else 0.5,
    surface_tension=0.00704,
    cohesion=0.0704,
    adhesion=0.0,
    cfl_coefficient=1.0,
)
physicsUtils.add_physics_material_to_prim(stage, particle_system.GetPrim(), pbd_material_path)

particles_path = Sdf.Path("/World/WaterParticles")

emitter_layer = None
emitter_velocities_np = None
emitter_region_counts = {}
if args.source_mode == "emitter":
    emitter_axial_spacing = emitter_plan.axial_spacing
    emitter_row_spacing = emitter_plan.row_spacing
    emitter_packing_offsets = (
        (0.0, 0.0),
        (0.5 * args.spacing, emitter_row_spacing / 3.0),
    )
    emitter_layer = create_hexagonal_emitter_layer(
        args.spacing,
        args.emitter_nozzle_diameter,
    )

    # Fill the compact reservoir with fixed-topology HCP particles.
    reservoir_x = (
        reservoir_left_x
        + piston_thickness
        + particle_contact_offset
        + emitter_axial_spacing
        + np.arange(emitter_plan.reservoir_layers, dtype=np.float32)
        * emitter_axial_spacing
    )
    reservoir_points = create_hcp_cylinder_positions(
        reservoir_x,
        emitter_feed_y,
        emitter_plan.reservoir_diameter,
        args.spacing,
        emitter_packing_offsets,
    )

    # Seed the contraction, elbow, and long final straight from the exact same
    # centerline used by the collision shell. HCP parity stays continuous through
    # region boundaries, and velocity rotates continuously with the local tangent.
    (
        connector_arc_lengths,
        connector_centers,
        connector_tangents,
        connector_normals,
        connector_binormals,
        connector_regions,
    ) = sample_centerline_by_arc_length(
        emitter_centerline, emitter_axial_spacing
    )
    connector_diameters = emitter_path_particle_diameters(
        emitter_centerline,
        connector_arc_lengths,
        emitter_plan.reservoir_diameter,
    )
    bend_seed_start = emitter_plan.contraction_length
    bend_seed_end = bend_seed_start + emitter_plan.bend_arc_length
    bend_seed_mask = (
        (connector_arc_lengths >= bend_seed_start - 1.0e-12)
        & (connector_arc_lengths <= bend_seed_end + 1.0e-12)
    )
    connector_diameters[bend_seed_mask] = np.maximum(
        3.0 * args.spacing,
        connector_diameters[bend_seed_mask]
        * (
            1.0
            - 0.22
            * np.sin(
                math.pi
                * (connector_arc_lengths[bend_seed_mask] - bend_seed_start)
                / emitter_plan.bend_arc_length
            )
        ),
    )
    connector_points, connector_layer_counts = create_hcp_swept_tube_positions(
        args.spacing,
        connector_arc_lengths,
        connector_centers,
        connector_normals,
        connector_binormals,
        connector_diameters,
        emitter_packing_offsets,
        bend_seed_start,
        bend_seed_end,
        emitter_plan.bend_radius,
    )
    connector_velocity = create_hcp_swept_tube_velocities(
        connector_tangents,
        connector_layer_counts,
        emitter_plan.particle_flux * args.emitter_initial_velocity_scale,
        emitter_axial_spacing,
        connector_regions,
    )
    connector_particle_regions = np.repeat(
        connector_regions, connector_layer_counts
    )

    initial_pool_points = create_shallow_pool_positions(
        args, fluid_rest_offset, obstacle_prim
    )
    supply_points = np.concatenate(
        (reservoir_points, connector_points, initial_pool_points),
        axis=0,
    )
    emitter_velocities_np = np.concatenate(
        (
            np.zeros_like(reservoir_points),
            connector_velocity,
            np.zeros_like(initial_pool_points),
        ),
        axis=0,
    )
    emitter_region_counts = {
        "reservoir": int(len(reservoir_points)),
        "contraction": int(np.count_nonzero(connector_particle_regions == "contraction")),
        "bend": int(np.count_nonzero(connector_particle_regions == "bend")),
        "straightening": int(
            np.count_nonzero(connector_particle_regions == "straightening")
        ),
        "initial_pool": int(len(initial_pool_points)),
    }
    if len(supply_points) > args.particle_capacity:
        raise ValueError(
            "Compact emitter needs "
            f"{len(supply_points)} particles {emitter_region_counts}, "
            f"but --particle-capacity is {args.particle_capacity}"
        )
    positions = Vt.Vec3fArray.FromNumpy(supply_points)
    particle_capacity = len(supply_points)
    source_description = (
        f"compact_reservoir={len(reservoir_points)}, "
        f"connector={len(connector_points)}, "
        f"initial_pool={len(initial_pool_points)}, "
        f"active={len(supply_points)}/{args.particle_capacity}, "
        f"reservoir={emitter_plan.reservoir_inner_length:.3f}x"
        f"{emitter_plan.reservoir_inner_diameter:.3f}m, "
        f"nozzle={args.emitter_nozzle_diameter:.3f}m, "
        f"nozzle_speed={args.emitter_speed:.3f}m/s, "
        f"pusher_speed={emitter_plan.pusher_speed:.4f}m/s, packing=HCP"
    )
elif args.source_mode == "block":
    block_width = (args.nx - 1) * args.spacing
    block_depth = (args.nz - 1) * args.spacing
    block_origin = (
        -0.55 + 0.5 * args.spacing,
        fluid_rest_offset + 0.001,
        -0.5 * block_depth,
    )
    positions = create_block_positions(
        block_origin,
        args.spacing,
        args.nx,
        args.ny,
        args.nz,
    )
    source_description = (
        f"block={block_width:.3f}x{(args.ny - 1) * args.spacing:.3f}"
        f"x{block_depth:.3f}m"
    )
else:
    # Reuse the same particle budget and spacing as the block version, but
    # preallocate it in the reservoir directly above the basin.  Gravity drives
    # particles through its fixed bottom aperture without topology changes.
    reservoir_nx = args.nx
    reservoir_ny = args.ny
    reservoir_nz = args.nz
    reservoir_width = (reservoir_nx - 1) * args.spacing
    reservoir_height = (reservoir_ny - 1) * args.spacing
    reservoir_depth = (reservoir_nz - 1) * args.spacing
    reservoir_origin = (
        args.stream_outlet_x - 0.5 * reservoir_width,
        args.stream_reservoir_bottom + fluid_rest_offset + 0.002,
        -0.5 * reservoir_depth,
    )
    positions = create_block_positions(
        reservoir_origin,
        args.spacing,
        reservoir_nx,
        reservoir_ny,
        reservoir_nz,
    )
    source_description = (
        f"preallocated_overhead_pool={reservoir_width:.3f}x"
        f"{reservoir_height:.3f}x{reservoir_depth:.3f}m, "
        f"outlet={args.stream_aperture:.3f}x{args.stream_aperture:.3f}m"
        f"@({args.stream_outlet_x:.3f},{args.stream_reservoir_bottom:.3f})"
    )

active_particle_count = len(positions)
if args.source_mode != "emitter":
    particle_capacity = active_particle_count
velocities = Vt.Vec3fArray.FromNumpy(
    emitter_velocities_np
    if args.source_mode == "emitter"
    else np.zeros((active_particle_count, 3), dtype=np.float32)
)
particles_prim = particleUtils.add_physx_particleset_pointinstancer(
    stage,
    particles_path,
    positions,
    velocities,
    particle_system_path,
    self_collision=True,
    fluid=True,
    particle_group=0,
    particle_mass=0.0,
    density=1000.0,
)
particles_prim.CreateAttribute(
    "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
).Set(particle_capacity)
particle_instancer = UsdGeom.PointInstancer(particles_prim)
particle_set_api = PhysxSchema.PhysxParticleSetAPI(particles_prim)
prototype = UsdGeom.Imageable.Get(
    stage,
    particles_path.AppendChild("particlePrototype0"),
)
hide(prototype)

# The controlled thin stream uses the isosurface-oriented anisotropy settings
# from NVIDIA's ParticlePostProcessingDemo.  Async capture is already held
# until the generated mesh settles, so anisotropy no longer races screenshots.
particleUtils.add_physx_particle_smoothing(
    stage,
    particle_system_path,
    enabled=True,
    strength=0.80 if args.source_mode == "emitter" else 0.12,
)
use_anisotropy = args.anisotropy or args.source_mode == "emitter"
if use_anisotropy:
    particleUtils.add_physx_particle_anisotropy(
        stage,
        particle_system_path,
        enabled=True,
        scale=5.0 if args.source_mode == "emitter" else 1.35,
        min=1.0 if args.source_mode == "emitter" else 0.9,
        max=2.0 if args.source_mode == "emitter" else 1.8,
    )
particleUtils.add_physx_particle_isosurface(
    stage,
    particle_system_path,
    enabled=True,
    # Keep the reconstruction grid near 1 mm while the compact hidden source
    # limits spatial subgrid demand and leaves budget for the visible jet/pool.
    grid_spacing=(
        max(0.001, 0.5 * args.spacing)
        if args.source_mode == "emitter"
        else 0.5 * args.spacing
    ),
    surface_distance=(
        1.10 * args.spacing
        if args.source_mode == "emitter"
        else None
    ),
    num_mesh_smoothing_passes=2,
    num_mesh_normal_smoothing_passes=4,
    max_vertices=4_000_000,
    max_triangles=8_000_000,
    max_subgrids=16384,
)

if args.diagnostic_material:
    water_render_path = create_pbr_material(
        stage,
        "/World/Looks/WaterRender",
        Gf.Vec3f(0.025, 0.20, 0.46),
        roughness=0.055,
        metallic=0.0,
    )
else:
    CreateAndBindMdlMaterialFromLibrary(
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        bind_selected_prims=False,
        prim_name="WaterRender",
    ).do()
    water_render_path = Sdf.Path("/World/Looks/WaterRender")
    water_shader = UsdShade.Shader.Get(stage, water_render_path.AppendChild("Shader"))
    water_shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.90, 0.97, 1.0)
    )
    water_shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
    water_shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.018)
    water_shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    water_shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.08)
bind_material(stage, particle_system_path, water_render_path)

# Studio-like lighting gives the transparent surface readable highlights.
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr().Set(args.dome_light_intensity)
dome.CreateColorAttr().Set(Gf.Vec3f(0.72, 0.82, 1.0))
key = UsdLux.DiskLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr().Set(args.key_light_intensity)
key.CreateRadiusAttr().Set(args.key_light_radius)
key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.88, 0.72))
set_camera(key.GetPrim(), Gf.Vec3d(-0.45, 1.05, -0.35), Gf.Vec3d(0.08, 0.08, 0.0))
rim = UsdLux.RectLight.Define(stage, "/World/RimLight")
rim.CreateIntensityAttr().Set(args.rim_light_intensity)
rim.CreateWidthAttr().Set(args.rim_light_width)
rim.CreateHeightAttr().Set(args.rim_light_height)
rim.CreateColorAttr().Set(Gf.Vec3f(0.55, 0.72, 1.0))
set_camera(rim.GetPrim(), Gf.Vec3d(0.55, 0.62, -0.58), Gf.Vec3d(0.12, 0.10, 0.0))

camera = UsdGeom.Camera.Define(stage, "/World/RenderCamera")
camera_preset = args.camera_preset
if camera_preset == "auto":
    camera_preset = "front" if args.dynamic_obstacle else "diagonal"
camera_settings = {
    "diagonal": (
        46.0,
        Gf.Vec3d(1.02, 0.48, 0.88),
        Gf.Vec3d(-0.03, 0.11, 0.0),
    ),
    "front": (
        30.0,
        Gf.Vec3d(0.02, 0.52, 1.70),
        Gf.Vec3d(0.03, 0.10, 0.0),
    ),
    "front-high": (
        32.0,
        Gf.Vec3d(0.02, 0.82, 1.65),
        Gf.Vec3d(0.03, 0.08, 0.0),
    ),
    "stream-front": (
        26.0,
        Gf.Vec3d(0.02, 0.68, 2.15),
        Gf.Vec3d(-0.03, 0.34, 0.0),
    ),
}
camera_focal_length, camera_eye, camera_target = camera_settings[camera_preset]
camera.CreateFocalLengthAttr(camera_focal_length)
camera.CreateHorizontalApertureAttr(20.955)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
set_camera(camera.GetPrim(), camera_eye, camera_target)
viewport = get_active_viewport()
viewport.camera_path = camera.GetPath()
viewport.set_texture_resolution((args.width, args.height))

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", args.renderer)
settings.set("/persistent/app/viewport/displayOptions", 0)
settings.set(physx_settings_bindings.SETTING_UPDATE_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)
settings.set("/rtx/translucency/maxRefractionBounces", 12)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/indirectDiffuse/enabled", True)
settings.set("/rtx/pathtracing/fractionalCutoutOpacity", True)
if args.renderer == "PathTracing":
    settings.set("/rtx/pathtracing/spp", args.path_spp)
    settings.set("/rtx/pathtracing/totalSpp", args.path_spp)
    settings.set("/rtx/pathtracing/maxBounces", 12)


def render_updates(count):
    for _ in range(max(0, count)):
        simulation_app.update()


def capture_viewport(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    capture = capture_viewport_to_file(viewport, file_path=file_path)
    task = asyncio.ensure_future(capture.wait_for_result(completion_frames=2))
    for _ in range(args.capture_timeout_updates):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        task.cancel()
        raise RuntimeError(f"Timed out waiting for capture: {file_path}")
    if not task.result():
        raise RuntimeError(f"Capture returned no AOVs: {file_path}")
    for _ in range(args.capture_timeout_updates):
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            return
        simulation_app.update()
    raise RuntimeError(f"Capture did not produce a non-empty file: {file_path}")


stage_path = os.path.join(args.output, "physx_realistic_liquid.usda")
stage.GetRootLayer().Export(stage_path)
render_updates(3)

simulation = get_physx_simulation_interface()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
simulation.attach_stage(stage_id)
metrics_file = None
try:

    if emitter_plan:
        emitter_plan_data = emitter_plan.to_dict()
        emitter_plan_data["actual_active_particles"] = active_particle_count
        emitter_plan_data["regions"] = emitter_region_counts
        emitter_plan_data["source_description"] = source_description
        emitter_plan_path = os.path.join(args.output, "emitter_plan.json")
        with open(emitter_plan_path, "w", encoding="utf-8") as plan_file:
            json.dump(emitter_plan_data, plan_file, indent=2, sort_keys=True)
        print(
            f"[emitter-plan] active={active_particle_count}/{args.particle_capacity}, "
            f"regions={emitter_region_counts}, flux={emitter_plan.particle_flux:.0f} particles/s "
            f"({emitter_plan.volume_flux_lps:.3f} L/s), "
            f"reservoir={emitter_plan.reservoir_inner_length:.3f}x"
            f"{emitter_plan.reservoir_inner_diameter:.3f}m, "
            f"pusher={emitter_plan.pusher_speed:.4f}m/s, "
            f"step={emitter_plan.pusher_step_fraction:.3f} spacing, "
            f"stroke={emitter_plan.pusher_planned_stroke:.3f}/"
            f"{emitter_plan.pusher_max_stroke:.3f}m"
        )

    print(
        f"[physx-liquid] capacity={particle_capacity}, "
        f"active_particles={active_particle_count}, source={source_description}, "
        f"spacing={args.spacing:.4f}m, iterations={args.solver_iterations}, "
        f"substeps={args.substeps}, renderer={args.renderer}, "
        f"anisotropy={use_anisotropy}, "
        f"dynamic_obstacle={args.dynamic_obstacle}, obstacle_mass={args.obstacle_mass:.3f}kg, "
        f"obstacle_shape={args.obstacle_shape}, obstacle_collision={args.obstacle_collision}, "
        f"mirror_obstacle={args.mirror_obstacle}, mirror_roughness={args.mirror_roughness:.3f}, "
        f"lights=(dome={args.dome_light_intensity:.0f}, "
        f"key={args.key_light_intensity:.0f}@r{args.key_light_radius:.3f}, "
        f"rim={args.rim_light_intensity:.0f}@{args.rim_light_width:.3f}x{args.rim_light_height:.3f}), "
        f"camera={camera_preset}{obstacle_mesh_info}"
    )

    simulation_step = 0


    def emitter_pusher_displacement(elapsed):
        if not emitter_plan:
            return 0.0
        ramp_seconds = args.emitter_ramp_frames / 60.0
        if ramp_seconds > 0.0 and elapsed < ramp_seconds:
            return emitter_plan.pusher_speed * (
                0.5 * elapsed
                - ramp_seconds
                / (2.0 * math.pi)
                * math.sin(math.pi * elapsed / ramp_seconds)
            )
        return emitter_plan.pusher_speed * (elapsed - 0.5 * ramp_seconds)


    def update_emitter_piston_target():
        if not emitter_piston_translate_attr:
            return
        elapsed = simulation_step / (60.0 * args.substeps)
        displacement = emitter_pusher_displacement(elapsed)
        if displacement > emitter_plan.pusher_max_stroke + 1.0e-9:
            raise RuntimeError(
                f"Emitter pusher exceeded stroke: {displacement:.6f} > "
                f"{emitter_plan.pusher_max_stroke:.6f} m"
            )
        emitter_piston_translate_attr.Set(
            Gf.Vec3d(
                emitter_piston_start[0] + displacement,
                emitter_piston_start[1],
                emitter_piston_start[2],
            )
        )


    if args.source_mode == "emitter":
        print(
            f"[emitter] pre-rolling compact pressure reservoir for "
            f"{args.emitter_preroll_frames} frames"
        )
        for _ in range(max(0, args.emitter_preroll_frames)):
            for _ in range(args.substeps):
                update_emitter_piston_target()
                dt = 1.0 / (60.0 * args.substeps)
                simulation.simulate(dt, simulation_step * dt)
                simulation.fetch_results()
                simulation_step += 1
            simulation_app.update()
        print("[emitter] pre-roll complete; capture timeline starts now")

    if stream_gate:
        print(
            f"[reservoir] settling closed tank for "
            f"{args.reservoir_settle_frames} frames"
        )
        for _ in range(max(0, args.reservoir_settle_frames)):
            for _ in range(args.substeps):
                dt = 1.0 / (60.0 * args.substeps)
                simulation.simulate(dt, simulation_step * dt)
                simulation.fetch_results()
                simulation_step += 1
            simulation_app.update()
        for settle_collider in stream_settle_colliders:
            UsdPhysics.CollisionAPI(
                settle_collider
            ).GetCollisionEnabledAttr().Set(False)
        render_updates(2)
        print("[reservoir] valve opened; capture timeline starts now")

    capture_index = 0
    previous_capture_points = None
    previous_capture_frame = None
    empty_visible_surface_captures = 0
    last_metrics = None
    captured_metrics = []
    metrics_file = None
    if args.source_mode == "emitter":
        metrics_file = open(
            os.path.join(args.output, "metrics.jsonl"),
            "w",
            encoding="utf-8",
        )

    for frame in range(args.frames):
        for _ in range(args.substeps):
            update_emitter_piston_target()
            dt = 1.0 / (60.0 * args.substeps)
            simulation.simulate(dt, simulation_step * dt)
            simulation.fetch_results()
            simulation_step += 1
        simulation_app.update()

        should_capture = frame % args.capture_every == 0 or frame == args.frames - 1
        if should_capture:
            # Allow the asynchronously generated PhysX mesh to finish, then let the
            # path tracer accumulate on a static geometry state.
            render_updates(args.isosurface_settle_updates)
            output_path = os.path.join(args.output, f"rgb_{capture_index:04d}.png")
            capture_viewport(output_path)
            iso_mesh = UsdGeom.Mesh.Get(stage, particle_system_path.AppendChild("Isosurface"))
            mesh_points_np = np.empty((0, 3), dtype=np.float32)
            triangle_count = 0
            face_counts_np = None
            face_indices_np = None
            if iso_mesh:
                mesh_points = iso_mesh.GetPointsAttr().Get()
                if mesh_points is not None and len(mesh_points):
                    mesh_points_np = np.asarray(mesh_points, dtype=np.float32)
                face_counts = iso_mesh.GetFaceVertexCountsAttr().Get()
                face_indices = iso_mesh.GetFaceVertexIndicesAttr().Get()
                face_counts_np = np.asarray(face_counts, dtype=np.int32) if face_counts is not None else None
                face_indices_np = np.asarray(face_indices, dtype=np.int64) if face_indices is not None else None
                triangle_count = len(face_counts) if face_counts is not None else 0
            vertex_count = len(mesh_points_np)

            simulation_points = particle_set_api.GetSimulationPointsAttr().Get()
            if simulation_points is None or len(simulation_points) == 0:
                simulation_points = particle_instancer.GetPositionsAttr().Get()
            point_array = np.asarray(simulation_points, dtype=np.float32)
            if len(point_array) != active_particle_count:
                raise RuntimeError(
                    f"Fixed particle topology changed: {len(point_array)} != "
                    f"{active_particle_count}"
                )
            finite_mask = np.isfinite(point_array).all(axis=1)
            finite_count = int(finite_mask.sum())
            if finite_count != active_particle_count:
                raise RuntimeError(
                    f"Non-finite particles detected: {finite_count}/{active_particle_count}"
                )
            finite_points = point_array[finite_mask]
            bounds_min = finite_points.min(axis=0)
            bounds_max = finite_points.max(axis=0)
            particle_state = (
                f", finite_particles={finite_count}, bounds="
                f"({bounds_min[0]:.3f},{bounds_min[1]:.3f},{bounds_min[2]:.3f}).."
                f"({bounds_max[0]:.3f},{bounds_max[1]:.3f},{bounds_max[2]:.3f})"
            )

            metrics = {
                "sim_frame": frame,
                "output_index": capture_index,
                "active_particles": active_particle_count,
                "finite_particles": finite_count,
                "isosurface_vertices": vertex_count,
                "isosurface_triangles": triangle_count,
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "path": output_path,
            }
            if emitter_plan:
                jet_half_width = 2.0 * args.emitter_nozzle_diameter
                reservoir_mask = (
                    (point_array[:, 0] >= reservoir_left_x - 0.01)
                    & (point_array[:, 0] <= reservoir_right_x + 0.01)
                    & (np.abs(point_array[:, 1] - emitter_feed_y) <= 0.75 * emitter_plan.reservoir_inner_diameter)
                    & (np.abs(point_array[:, 2]) <= 0.75 * emitter_plan.reservoir_inner_diameter)
                )
                jet_mask = (
                    (np.abs(point_array[:, 0] - args.stream_outlet_x) <= jet_half_width)
                    & (point_array[:, 1] >= 0.0)
                    & (point_array[:, 1] <= args.emitter_height)
                    & (np.abs(point_array[:, 2]) <= jet_half_width)
                )
                basin_mask = (
                    (point_array[:, 0] >= -0.575)
                    & (point_array[:, 0] <= 0.575)
                    & (point_array[:, 1] >= -0.01)
                    & (point_array[:, 1] <= 0.34)
                    & (point_array[:, 2] >= -0.265)
                    & (point_array[:, 2] <= 0.265)
                )
                pool_mask = basin_mask & (point_array[:, 1] <= 0.05)
                escaped_mask = (
                    (point_array[:, 0] < min(-0.79, reservoir_left_x - 0.20))
                    | (point_array[:, 0] > 0.79)
                    | (point_array[:, 1] < -0.20)
                    | (point_array[:, 1] > args.emitter_height + 0.40)
                    | (np.abs(point_array[:, 2]) > 0.49)
                )
                reservoir_envelope = (
                    (point_array[:, 0] >= reservoir_left_x - 0.02)
                    & (point_array[:, 0] <= reservoir_right_x + 0.02)
                    & (np.abs(point_array[:, 1] - emitter_feed_y) <= reservoir_half + 0.02)
                    & (np.abs(point_array[:, 2]) <= reservoir_half + 0.02)
                )
                swept_distances = np.full(len(point_array), np.inf, dtype=np.float32)
                diagnostics_chunk_size = 100_000
                for chunk_start in range(0, len(point_array), diagnostics_chunk_size):
                    chunk_end = min(
                        len(point_array), chunk_start + diagnostics_chunk_size
                    )
                    deltas = (
                        point_array[chunk_start:chunk_end, None, :]
                        - collider_centers[None, :, :]
                    )
                    swept_distances[chunk_start:chunk_end] = np.sqrt(
                        np.min(np.sum(deltas * deltas, axis=2), axis=1)
                    )
                swept_channel_envelope = swept_distances <= (
                    float(np.max(collider_clear_radii)) + 0.02
                )
                outlet_spray_half = 3.0 * args.emitter_nozzle_diameter
                outlet_spray_envelope = (
                    (np.abs(point_array[:, 0] - args.stream_outlet_x) <= outlet_spray_half)
                    & (point_array[:, 1] >= args.emitter_height - 0.02)
                    & (point_array[:, 1] <= emitter_feed_y + tube_inner_half + 0.04)
                    & (np.abs(point_array[:, 2]) <= outlet_spray_half)
                )
                source_neighborhood = (
                    (point_array[:, 0] >= reservoir_left_x - 0.08)
                    & (point_array[:, 0] <= contraction_right_x + 0.08)
                    & (point_array[:, 1] >= args.emitter_height + 0.02)
                    & (point_array[:, 1] <= emitter_feed_y + reservoir_half + 0.08)
                    & (np.abs(point_array[:, 2]) <= reservoir_half + 0.08)
                )
                source_leak_mask = (
                    source_neighborhood
                    & ~reservoir_envelope
                    & ~swept_channel_envelope
                    & ~outlet_spray_envelope
                )
                jet_bin_count = max(1, int(math.ceil(args.emitter_height / 0.01)))
                jet_bins = np.floor(point_array[jet_mask, 1] / 0.01).astype(np.int32)
                jet_bin_coverage = len(np.unique(jet_bins)) / jet_bin_count
                crossing_count = 0
                measured_flux = None
                velocity_array = None
                current_velocities = particle_instancer.GetVelocitiesAttr().Get()
                if current_velocities is not None and len(current_velocities) == len(point_array):
                    velocity_array = np.asarray(current_velocities, dtype=np.float32)
                    nozzle_slab = (
                        (point_array[:, 1] >= args.emitter_height - emitter_axial_spacing)
                        & (point_array[:, 1] <= args.emitter_height + emitter_axial_spacing)
                        & (np.abs(point_array[:, 0] - args.stream_outlet_x) <= jet_half_width)
                        & (np.abs(point_array[:, 2]) <= jet_half_width)
                        & (velocity_array[:, 1] < 0.0)
                    )
                    measured_flux = float(
                        np.sum(-velocity_array[nozzle_slab, 1])
                        / (2.0 * emitter_axial_spacing)
                    )
                if (
                    previous_capture_points is not None
                    and len(previous_capture_points) == len(point_array)
                ):
                    crossing_count = int(
                        np.count_nonzero(
                            (previous_capture_points[:, 1] >= args.emitter_height)
                            & (point_array[:, 1] < args.emitter_height)
                            & (np.abs(point_array[:, 0] - args.stream_outlet_x) <= jet_half_width)
                            & (np.abs(point_array[:, 2]) <= jet_half_width)
                        )
                    )
                jet_iso_mask = (
                    (np.abs(mesh_points_np[:, 0] - args.stream_outlet_x) <= jet_half_width)
                    & (mesh_points_np[:, 1] >= 0.0)
                    & (mesh_points_np[:, 1] <= args.emitter_height + 0.02)
                    & (np.abs(mesh_points_np[:, 2]) <= jet_half_width)
                ) if vertex_count else np.zeros(0, dtype=bool)
                pool_iso_mask = (
                    (mesh_points_np[:, 0] >= -0.575)
                    & (mesh_points_np[:, 0] <= 0.575)
                    & (mesh_points_np[:, 1] >= -0.01)
                    & (mesh_points_np[:, 1] <= 0.05)
                    & (np.abs(mesh_points_np[:, 2]) <= 0.265)
                ) if vertex_count else np.zeros(0, dtype=bool)
                visible_iso_mask = jet_iso_mask | pool_iso_mask
                pusher_displacement = emitter_pusher_displacement(
                    simulation_step / (60.0 * args.substeps)
                )
                jet_slices = analyze_jet_slices(
                    args,
                    mesh_points_np,
                    face_counts_np,
                    face_indices_np,
                    point_array,
                    velocity_array,
                    max(0.001, 0.5 * args.spacing),
                )
                jet_shape = aggregate_jet_slice_metrics(jet_slices)
                metrics.update(
                    {
                        "reservoir_particles": int(np.count_nonzero(reservoir_mask)),
                        "jet_particles": int(np.count_nonzero(jet_mask)),
                        "basin_particles": int(np.count_nonzero(basin_mask)),
                        "pool_particles": int(np.count_nonzero(pool_mask)),
                        "escaped_particles": int(np.count_nonzero(escaped_mask)),
                        "source_leak_particles": int(np.count_nonzero(source_leak_mask)),
                        "nozzle_crossings": crossing_count,
                        "measured_particle_flux": measured_flux,
                        "target_particle_flux": emitter_plan.particle_flux,
                        "jet_vertical_bin_coverage": jet_bin_coverage,
                        "visible_isosurface_vertices": int(np.count_nonzero(visible_iso_mask)),
                        "jet_isosurface_vertices": int(np.count_nonzero(jet_iso_mask)),
                        "pool_isosurface_vertices": int(np.count_nonzero(pool_iso_mask)),
                        "pusher_displacement": pusher_displacement,
                        "pusher_fraction": pusher_displacement / emitter_plan.pusher_max_stroke,
                        "pusher_remaining_stroke": emitter_plan.pusher_max_stroke - pusher_displacement,
                        "jet_slices": jet_slices,
                        **jet_shape,
                    }
                )
                if metrics["visible_isosurface_vertices"] == 0:
                    empty_visible_surface_captures += 1
                else:
                    empty_visible_surface_captures = 0
                if empty_visible_surface_captures >= 2:
                    raise RuntimeError("Visible isosurface was empty for two captures")
                metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
                metrics_file.flush()
                last_metrics = metrics
                captured_metrics.append(metrics)
                if metrics["source_leak_particles"] > max(1, int(0.001 * active_particle_count)):
                    print(
                        "[warning] High-source spray exceeded 0.1%: "
                        f"{metrics['source_leak_particles']} particles"
                    )
                if metrics["escaped_particles"] > max(1, int(0.001 * active_particle_count)):
                    raise RuntimeError(
                        f"Escaped particles exceeded 0.1%: {metrics['escaped_particles']}"
                    )
                previous_capture_points = point_array.copy()
                previous_capture_frame = frame

            obstacle_state = ""
            if obstacle_rigid_api:
                obstacle_transform = UsdGeom.Xformable(
                    obstacle_prim
                ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                obstacle_position = obstacle_transform.ExtractTranslation()
                obstacle_velocity = obstacle_rigid_api.GetVelocityAttr().Get()
                obstacle_state = (
                    f", obstacle_position=({obstacle_position[0]:.4f},"
                    f"{obstacle_position[1]:.4f},{obstacle_position[2]:.4f})"
                    f", obstacle_velocity=({obstacle_velocity[0]:.4f},"
                    f"{obstacle_velocity[1]:.4f},{obstacle_velocity[2]:.4f})"
                )
            print(
                f"[capture] sim_frame={frame}, output={capture_index}, "
                f"active_particles={active_particle_count}, "
                f"capacity={particle_capacity}, "
                f"isosurface_vertices={vertex_count}, triangles={triangle_count}"
                f"{particle_state}{obstacle_state}, path={output_path}"
            )
            capture_index += 1

    jet_capture_summary = None
    if emitter_plan:
        if last_metrics is None:
            raise RuntimeError("Emitter completed without capture metrics")
        jet_capture_summary = aggregate_jet_capture_metrics(
            captured_metrics,
            args.jet_metrics_warmup_frames,
        )
        with open(
            os.path.join(args.output, "jet_metrics_summary.json"),
            "w",
            encoding="utf-8",
        ) as summary_file:
            json.dump(jet_capture_summary, summary_file, indent=2, sort_keys=True)
        acceptance_failures = []
        if last_metrics["finite_particles"] != active_particle_count:
            acceptance_failures.append("finite particle count changed")
        if last_metrics["escaped_particles"] != 0:
            acceptance_failures.append(
                f"escaped_particles={last_metrics['escaped_particles']}"
            )
        if last_metrics["source_leak_particles"] > max(
            1, int(0.001 * active_particle_count)
        ):
            acceptance_failures.append(
                f"source_leak_particles={last_metrics['source_leak_particles']}"
            )
        if last_metrics["jet_vertical_bin_coverage"] < 0.80:
            acceptance_failures.append(
                "jet_vertical_bin_coverage="
                f"{last_metrics['jet_vertical_bin_coverage']:.3f}"
            )
        if last_metrics["jet_isosurface_vertices"] <= 0:
            acceptance_failures.append("jet isosurface is empty")
        if last_metrics["pool_isosurface_vertices"] <= 0:
            acceptance_failures.append("pool isosurface is empty")
        measured_flux = last_metrics.get("measured_particle_flux")
        if measured_flux is None or measured_flux < 0.5 * emitter_plan.particle_flux:
            acceptance_failures.append(
                f"measured_particle_flux={measured_flux}, "
                f"target={emitter_plan.particle_flux:.1f}"
            )
        if acceptance_failures:
            raise RuntimeError(
                "Emitter acceptance failed: " + "; ".join(acceptance_failures)
            )

    completion = {
        "frames_completed": args.frames,
        "captures_completed": capture_index,
        "active_particles": active_particle_count,
        "source_mode": args.source_mode,
    }
    if jet_capture_summary is not None:
        completion["jet_metrics_summary"] = jet_capture_summary
    with open(
        os.path.join(args.output, "run_complete.json"),
        "w",
        encoding="utf-8",
    ) as completion_file:
        json.dump(completion, completion_file, indent=2, sort_keys=True)
finally:
    if metrics_file is not None:
        metrics_file.close()
    simulation.detach_stage()
    simulation_app.close()
