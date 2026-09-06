"""Prepare a glass-cabinet pour or pre-filled pool impact run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCENES_ROOT = Path(r"Y:\scenes") if os.name == "nt" else Path("/mnt/y/scenes")
os.environ.setdefault("SCENES_ROOT", str(SCENES_ROOT))

from experiments.model_material.run_experiments import (  # noqa: E402
    SCENE_CONFIG_PATH,
    load_scene_context,
)


PHYSICS_FRAMES_PER_SECOND = 60
DEEP_POUR_NOZZLE_ASPECT_RATIO = 72.0 / 52.0
DEEP_POUR_PHYSICS_SUBSTEPS = 12
DEEP_POUR_PARTICLE_ITERATIONS = 64
DEEP_POUR_DEFORMABLE_ITERATIONS = 32
DEEP_POUR_COLLISION_TRIANGLE_COUNT = 20_000
DEEP_POUR_MAXIMUM_SPEED = 4.2
DEEP_POUR_MAX_BIAS_COEFFICIENT = 240.0


def deep_pour_spec(
    inner_size,
    spacing: float,
    *,
    target_water_depth: float = 0.030,
    prewarm_seconds: float = 2.0,
    inlet_seconds: float = 5.0,
    post_inlet_seconds: float = 1.5,
    inlet_speed: float = 0.96,
) -> dict:
    """Return the discrete server-scale inlet and timing for a deep pour.

    The requested water depth is converted to volume using the cabinet floor
    area.  Nozzle dimensions are then snapped up to complete particle lattice
    columns, so the generated volume cannot silently fall below the requested
    depth because ``_axis_samples`` truncates partial columns.
    """
    inner_x, _inner_y, inner_z = (float(value) for value in inner_size)
    spacing = float(spacing)
    values = {
        "cabinet inner x": inner_x,
        "cabinet inner z": inner_z,
        "spacing": spacing,
        "target water depth": float(target_water_depth),
        "prewarm seconds": float(prewarm_seconds),
        "inlet seconds": float(inlet_seconds),
        "post-inlet seconds": float(post_inlet_seconds),
        "inlet speed": float(inlet_speed),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError(f"Deep-pour dimensions and timing must be positive: {values}")

    prewarm_frames = int(round(prewarm_seconds * PHYSICS_FRAMES_PER_SECOND))
    inlet_frames = int(round(inlet_seconds * PHYSICS_FRAMES_PER_SECOND))
    post_inlet_frames = int(
        round(post_inlet_seconds * PHYSICS_FRAMES_PER_SECOND)
    )
    if min(prewarm_frames, inlet_frames, post_inlet_frames) < 1:
        raise ValueError("Deep-pour timing must allocate at least one frame per phase")

    target_volume = inner_x * inner_z * target_water_depth
    lattice_column_area = spacing**2
    inlet_distance = inlet_speed * inlet_frames / PHYSICS_FRAMES_PER_SECOND
    minimum_columns = int(
        math.ceil(target_volume / (lattice_column_area * inlet_distance))
    )
    nozzle_columns_x = int(
        math.ceil(math.sqrt(minimum_columns * DEEP_POUR_NOZZLE_ASPECT_RATIO))
    )
    nozzle_columns_z = int(math.ceil(minimum_columns / nozzle_columns_x))
    nozzle_size_x = nozzle_columns_x * spacing
    nozzle_size_z = nozzle_columns_z * spacing
    effective_area = nozzle_columns_x * nozzle_columns_z * lattice_column_area
    projected_volume = effective_area * inlet_distance
    projected_depth = projected_volume / (inner_x * inner_z)
    projected_particles = int(round(projected_volume / spacing**3))
    source_start = prewarm_frames + 1
    source_stop = prewarm_frames + inlet_frames
    total_frames = source_stop + post_inlet_frames
    return {
        "prewarm_frames": prewarm_frames,
        "inlet_frames": inlet_frames,
        "post_inlet_frames": post_inlet_frames,
        "source_start_frame": source_start,
        "source_stop_frame": source_stop,
        "total_frames": total_frames,
        "visible_start_frame": prewarm_frames,
        "visible_stop_frame": total_frames,
        "target_water_depth_m": target_water_depth,
        "target_water_volume_m3": target_volume,
        "target_water_volume_liters": 1000.0 * target_volume,
        "nozzle_columns": [nozzle_columns_x, nozzle_columns_z],
        "nozzle_size_m": [nozzle_size_x, nozzle_size_z],
        "projected_water_depth_m": projected_depth,
        "projected_water_volume_m3": projected_volume,
        "projected_water_volume_liters": 1000.0 * projected_volume,
        "projected_particle_count": projected_particles,
        "inlet_speed_m_s": inlet_speed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="warehouse")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--spacing", type=float, default=0.004)
    parser.add_argument(
        "--max-depenetration-velocity",
        type=float,
        default=1.0,
        help=(
            "Maximum speed added by particle penetration correction. This is "
            "separate from the overall particle maximum speed."
        ),
    )
    parser.add_argument(
        "--continuous-inlet-probe",
        action="store_true",
        help=(
            "Use one density-driven shared particle set and a true 240 Hz "
            "continuous invisible inlet instead of the retained v18 source."
        ),
    )
    parser.add_argument(
        "--compact-impact-probe",
        action="store_true",
        help=(
            "Build the compact v22 cabinet: smaller bodies and enclosure, a "
            "lower-flow inlet, and first impact on the central deformable body. "
            "This mode always uses the continuous shared particle set."
        ),
    )
    parser.add_argument(
        "--pool-drop-probe",
        action="store_true",
        help=(
            "Fill a smaller glass cabinet with a pre-settled 4 mm pool, then "
            "release the mixed rigid/deformable bodies into it."
        ),
    )
    parser.add_argument(
        "--server-deep-pour",
        action="store_true",
        help=(
            "Build the original-scale, 4 mm, 3 cm target-depth production pour "
            "for a high-memory server. The preset includes two seconds of dry "
            "body prewarm, five seconds of inlet flow, 64/32 solvers, "
            "finite particle/deformable depenetration speeds, and enlarged GPU "
            "contact buffers. It overrides --frames and "
            "--max-depenetration-velocity."
        ),
    )
    parser.add_argument(
        "--deep-pour-deformable-only",
        action="store_true",
        help=(
            "With --server-deep-pour, retain only the deformable body. This "
            "matches the locally validated fluid-softbody isolation run; omit "
            "it for the full rigid/deformable production scene."
        ),
    )
    parser.add_argument(
        "--deep-pour-contact-probe",
        action="store_true",
        help=(
            "With --server-deep-pour, emit only one production-width inlet "
            "frame onto the deformable body while retaining the production "
            "4.2 m/s safety ceiling. This short run validates contact before "
            "launching the full cache."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. By default each authored scene gets its own "
            "<scene>_glass_cabinet_pour_v18_4mm directory."
        ),
    )
    return parser.parse_args()


def windows_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        return str(resolved)
    text = str(resolved)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        relative = text[7:].replace("/", "\\")
        return str(PureWindowsPath(f"{drive}:\\{relative}"))
    raise ValueError(f"Path is not on a mounted Windows drive: {resolved}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    exclusive_modes = (
        args.pool_drop_probe,
        args.compact_impact_probe,
        args.continuous_inlet_probe,
        args.server_deep_pour,
    )
    if sum(bool(value) for value in exclusive_modes) > 1:
        raise ValueError(
            "Choose only one of --pool-drop-probe, --compact-impact-probe, "
            "--continuous-inlet-probe, and --server-deep-pour"
        )
    if args.deep_pour_deformable_only and not args.server_deep_pour:
        raise ValueError(
            "--deep-pour-deformable-only requires --server-deep-pour"
        )
    if args.deep_pour_contact_probe and not args.server_deep_pour:
        raise ValueError("--deep-pour-contact-probe requires --server-deep-pour")
    if args.server_deep_pour and not math.isclose(
        args.spacing, 0.004, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("--server-deep-pour is calibrated for --spacing 0.004")
    if args.pool_drop_probe and args.frames != 36:
        raise ValueError(
            "The compact pool-drop shot is validated for exactly 36 frames; "
            "longer runs enter the current PhysX deformable-contact overflow."
        )
    if not args.pool_drop_probe and not args.server_deep_pour and args.frames < 45:
        raise ValueError("Glass-cabinet smoke runs require at least 45 frames")
    if args.max_depenetration_velocity <= 0.0:
        raise ValueError("--max-depenetration-velocity must be positive")
    if args.pool_drop_probe and (
        args.compact_impact_probe or args.continuous_inlet_probe
    ):
        raise ValueError("--pool-drop-probe cannot be combined with a pour probe")
    scene_configs = json.loads(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))
    scene = load_scene_context(args.scene, scene_configs)
    multi = json.loads(
        (ROOT / "configs" / "multi_object_scene_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    profiles = json.loads(
        (ROOT / "configs" / "model_material_experiments.json").read_text(
            encoding="utf-8"
        )
    )["physics_profiles"]
    experiment = next(
        (item for item in multi["experiments"] if item["scene"] == args.scene), None
    )
    if experiment is None:
        raise KeyError(f"No mixed-material baseline exists for scene {args.scene}")

    use_pool_drop = bool(args.pool_drop_probe)
    use_continuous_inlet = bool(
        args.continuous_inlet_probe
        or args.compact_impact_probe
        or args.server_deep_pour
    )

    # The apparatus sits on the already validated scene-local support.  The
    # visible glass remains 15 mm.  Its collision proxy extends outward to
    # 200 mm while preserving the same inner contact plane.  This exceeds the
    # maximum 133 mm displacement of one exact 60 Hz simulation call at the
    # capped 8 m/s particle speed, without moving the rendered water boundary.
    render_thickness = 0.015
    collision_thickness = 0.200
    floor_y = scene["support_y"] + render_thickness
    centre_x = scene["spawn_x"]
    centre_z = scene["spawn_z"]
    deep_spec = None
    if use_pool_drop:
        # Keep the 4 mm water resolution, but frame the event as a compact
        # tabletop-scale cabinet.  The earlier 0.74 m basin required 541k
        # particles and exhausted the 12 GB GPU during peak fluid/deformable
        # contact; this size retains comfortable clearance around all three
        # bodies with roughly half as many particles.
        body_scale = 0.30
        inner_size = (0.62, 0.34, 0.46)
        water_depth = 0.072
        placements = (
            (-0.17, water_depth + 0.10, 0.055),
            (0.17, water_depth + 0.14, 0.055),
            (0.00, water_depth + 0.18, -0.065),
        )
        body_contact_offset = 0.012
        body_rest_offset = 0.002
        layout = "glass_cabinet_pool_drop"
    elif args.server_deep_pour:
        # This is the original object/cabinet scale used by the v45-v59
        # coupling investigation.  Keep the smaller deformable contact shell
        # instead of the legacy 30/10 mm shell, which was too large for 4 mm
        # fluid particles.
        body_scale = 1.0
        inner_size = (2.0, 1.1272727272727272, 1.4909090909090907)
        placements = (
            (-0.5636363636363636, 0.05454545454545454, 0.20),
            (0.5636363636363636, 0.06363636363636363, 0.20),
            (0.0, 0.07272727272727272, -0.18181818181818182),
        )
        body_contact_offset = None
        body_rest_offset = None
        layout = "glass_cabinet_compact_pour"
        deep_spec = deep_pour_spec(inner_size, args.spacing)
        args.frames = (
            int(deep_spec["source_start_frame"]) + 29
            if args.deep_pour_contact_probe
            else int(deep_spec["total_frames"])
        )
    elif args.compact_impact_probe:
        body_scale = 0.55
        inner_size = (1.10, 0.62, 0.82)
        placements = (
            (-0.31, 0.030, 0.11),
            (0.31, 0.035, 0.11),
            (0.00, 0.040, -0.10),
        )
        body_contact_offset = 0.018
        body_rest_offset = 0.006
        layout = "glass_cabinet_compact_pour"
    else:
        body_scale = 1.0
        inner_size = (1.80, 1.25, 1.25)
        placements = (
            (-0.55, 0.05, 0.18),
            (0.55, 0.06, 0.18),
            (0.00, 0.08, -0.30),
        )
        body_contact_offset = 0.030
        body_rest_offset = 0.010
        layout = "glass_cabinet_pour"
    collision_policies = multi["rigid_collision_policies"]
    bodies = []
    for body, (offset_x, lift, offset_z) in zip(experiment["bodies"], placements):
        profile = profiles[body["physics_profile"]]
        physics_kind = "rigid" if profile["expected_behavior"] == "hard" else "deformable"
        collision_policy = collision_policies[body["model"]]
        collision_approximation = (
            collision_policy["approximation"] if physics_kind == "rigid" else "tetrahedral"
        )
        if args.server_deep_pour:
            collision_contact_offset = 0.006 if physics_kind == "deformable" else 0.018
            collision_rest_offset = 0.0 if physics_kind == "deformable" else 0.006
        else:
            collision_contact_offset = body_contact_offset
            collision_rest_offset = body_rest_offset
        bodies.append(
            {
                "model": windows_path(
                    ROOT / "assets" / "simulation_ready_models" / body["model"]
                ),
                "model_height": body_scale * body["model_height"],
                "model_yaw": body.get("model_yaw", 0.0),
                "spawn_x": centre_x + offset_x,
                "drop_height": floor_y + lift,
                "spawn_z": centre_z + offset_z,
                "density": profile["density"],
                "youngs_modulus": profile["youngs_modulus"],
                "poissons_ratio": profile["poissons_ratio"],
                "linear_damping": profile["linear_damping"],
                "settling_damping": profile["settling_damping"],
                # Water contact in the compact impact probe should not inherit
                # the dry collision bounce from the material profile.
                "restitution": (
                    0.0
                    if args.compact_impact_probe
                    or use_pool_drop
                    or args.server_deep_pour
                    else profile["restitution"]
                ),
                # Keep the collision shell proportional when the apparatus is
                # compacted; otherwise it would become larger relative to the
                # bodies even though the visible geometry got smaller.
                "collision_contact_offset": collision_contact_offset,
                "collision_rest_offset": collision_rest_offset,
                "physics_kind": physics_kind,
                "collision_approximation": collision_approximation,
                "sdf_resolution": (
                    int(collision_policy.get("sdf_resolution", 384))
                    if collision_approximation == "sdf"
                    else None
                ),
                "sdf_enable_remeshing": (
                    bool(collision_policy.get("sdf_enable_remeshing", False))
                    if collision_approximation == "sdf"
                    else None
                ),
                "physics_profile": body["physics_profile"],
                "material_preset": body["material_preset"],
            }
        )
    if args.deep_pour_deformable_only or args.deep_pour_contact_probe:
        bodies = [body for body in bodies if body["physics_kind"] == "deformable"]
        if len(bodies) != 1:
            raise RuntimeError(
                "The deep-pour isolation preset expected exactly one deformable body"
            )

    output = (
        args.output
        if args.output is not None
        else ROOT
        / "output"
        / "coupled_scenes"
        / (
            f"{args.scene}_glass_cabinet_deep_pour_server_3cm_4mm"
            if args.server_deep_pour and not args.deep_pour_contact_probe
            else f"{args.scene}_glass_cabinet_deep_pour_contact_probe_4mm"
            if args.deep_pour_contact_probe
            else (
                f"{args.scene}_glass_cabinet_pool_drop_v27_compact_4mm"
                if use_pool_drop
                else (
                    f"{args.scene}_glass_cabinet_pour_v22_compact_4mm"
                    if args.compact_impact_probe
                    else (
                        f"{args.scene}_glass_cabinet_pour_v20_probe_4mm"
                        if use_continuous_inlet
                        else f"{args.scene}_glass_cabinet_pour_v18_4mm"
                    )
                )
            )
        )
    ).resolve()
    body_config_path = output / "bodies.json"
    event_config_path = output / "glass_cabinet_pour.json"
    body_config = {
        "schema": 1,
        "product": "coupled_scene_mixed_body_configuration",
        "scene": args.scene,
        "source_experiment": experiment["id"],
        "deformable_collision_policy": multi["deformable_collision_policy"],
        "bodies": bodies,
    }
    if deep_spec is not None:
        source_start = int(deep_spec["source_start_frame"])
        source_stop = (
            source_start
            if args.deep_pour_contact_probe
            else int(deep_spec["source_stop_frame"])
        )
    elif use_pool_drop:
        source_start = source_stop = 1
    else:
        source_start = max(16, int(round(0.27 * args.frames)))
        source_stop = min(args.frames - 12, source_start + 47)
    particle_contact_offset = 0.5 * args.spacing / 0.6
    pool_bottom_y = floor_y + particle_contact_offset + 0.001
    pool_layer_count = max(
        1,
        int((water_depth - (particle_contact_offset + 0.001)) // args.spacing) + 1,
    ) if use_pool_drop else None
    pool_size_y = pool_layer_count * args.spacing if use_pool_drop else None
    pool_centre_y = (
        pool_bottom_y + 0.5 * (pool_layer_count - 1) * args.spacing
        if use_pool_drop
        else None
    )
    event_config = {
        "schema": 1,
        "product": "coupled_scene_pbd_pour_event",
        "layout": layout,
        "scene": args.scene,
        "particle_set_strategy": (
            "preauthored_frame_sets"
            if args.server_deep_pour
            else "single_shared_density_set"
            if use_continuous_inlet or use_pool_drop
            else "per_emission_frame_sets"
        ),
        "production_target": deep_spec,
        # Match the established swamp workflow: 1.5 seconds at 240 Hz is
        # simulated off-camera before the bodies are released.
        "settle_steps": 360 if use_pool_drop else 0,
        "cabinet": {
            # centre.y is deliberately the interior floor height.
            "centre": [centre_x, floor_y, centre_z],
            "inner_size": list(inner_size),
            "wall_thickness": render_thickness,
            "render_thickness": render_thickness,
            "collision_thickness": collision_thickness,
        },
        "source": {
            "spacing": args.spacing,
            "centre": (
                [centre_x, pool_centre_y, centre_z]
                if use_pool_drop
                else (
                    [
                        centre_x - 0.03,
                        floor_y + 0.625,
                        centre_z - 0.2018181818181818,
                    ]
                    if args.server_deep_pour
                    else (
                        [centre_x - 0.03, floor_y + 0.70, centre_z - 0.12]
                        if args.compact_impact_probe
                        else [
                            centre_x + 0.10,
                            floor_y + inner_size[1] + 0.24,
                            centre_z + 0.03,
                        ]
                    )
                )
            ),
            # size.y retains the legacy v18 source-band description.  The
            # continuous inlet derives its flux from cross-section * speed and
            # uses only the few millimetres upstream of its invisible plane.
            "size": (
                [inner_size[0] - 0.016, pool_size_y, inner_size[2] - 0.016]
                if use_pool_drop
                else (
                    [
                        deep_spec["nozzle_size_m"][0],
                        deep_spec["inlet_speed_m_s"] / PHYSICS_FRAMES_PER_SECOND,
                        deep_spec["nozzle_size_m"][1],
                    ]
                    if deep_spec is not None
                    else (
                        [0.072, 0.024, 0.052]
                        if args.compact_impact_probe
                        else [0.140, 0.024, 0.096]
                    )
                )
            ),
            "velocity": (
                [0.0, 0.0, 0.0]
                if use_pool_drop
                else (
                    [0.0, -deep_spec["inlet_speed_m_s"], 0.0]
                    if deep_spec is not None
                    else (
                        [0.0, -0.90, 0.0]
                        if args.compact_impact_probe
                        else [0.0, -1.25, 0.0]
                    )
                )
            ),
            "emission_model": (
                "static_slab"
                if use_pool_drop
                else "preauthored_frame_inlet"
                if args.server_deep_pour
                else "continuous_inlet"
                if use_continuous_inlet
                else "continuous_subframe_ballistic"
            ),
            "start_frame": source_start,
            "stop_frame": source_stop,
            "interval_frames": 1,
            "maximum_speed": (
                DEEP_POUR_MAXIMUM_SPEED if args.server_deep_pour else 8.0
            ),
            "max_depenetration_velocity": (
                0.25 if args.server_deep_pour else args.max_depenetration_velocity
            ),
            "solver_position_iterations": (
                DEEP_POUR_PARTICLE_ITERATIONS if args.server_deep_pour else 16
            ),
            "density": 1000.0,
            "friction": 0.05,
            "damping": 0.01,
            "settle_damping": 0.5 if use_pool_drop else 0.01,
            "viscosity": 0.002,
            "vorticity_confinement": 0.02,
            "surface_tension": 0.0074,
            "cohesion": 0.01,
            "adhesion": 0.0,
        },
        "maximum_lateral_escape_fraction": 0.0001,
        "gpu_collision_stack_size": (
            1073741824
            if args.deep_pour_contact_probe
            else
            2147483648
            if args.server_deep_pour
            else 1342177280
            if use_pool_drop
            else 536870912
        ),
        "gpu_max_deformable_volume_contacts": (
            8388608
            if args.deep_pour_contact_probe
            else
            16777216
            if args.server_deep_pour or use_pool_drop
            else 4194304
        ),
        "gpu_max_deformable_surface_contacts": (
            2097152
            if args.deep_pour_contact_probe
            else
            4194304
            if args.server_deep_pour
            else 2097152
            if use_pool_drop
            else 1048576
        ),
        "gpu_resource_maximum_utilization": 0.90,
        "maximum_below_floor_fraction": 0.0001,
        "maximum_speed_cap_fraction": 0.005,
        "body_contact_threshold": 0.040,
        "body_visible_surface_threshold": max(0.012, 3.0 * args.spacing),
        "minimum_body_contact_particles": 8,
        "minimum_visible_surface_particles": 1,
        "minimum_consecutive_visible_surface_frames": 2,
        "required_body_kinds": sorted({body["physics_kind"] for body in bodies}),
        "gpu_max_particle_contacts": (
            2097152
            if args.deep_pour_contact_probe
            else
            8388608
            if args.server_deep_pour
            else 4194304
            if use_pool_drop
            else 2097152
        ),
    }
    write_json(body_config_path, body_config)
    write_json(event_config_path, event_config)

    exact_collision = ROOT / "output" / "scene_collision_assets" / f"{args.scene}_exact_collision.usdc"
    if scene["needs_exact_collision"] and not exact_collision.is_file():
        raise FileNotFoundError(exact_collision)
    target_y = floor_y + (
        0.16
        if use_pool_drop
        else 0.28
        if args.compact_impact_probe
        else 0.55
    )
    camera_target = (
        centre_x,
        target_y,
        centre_z - (0.02 if args.compact_impact_probe else 0.0),
    )
    if args.compact_impact_probe or use_pool_drop:
        camera_distance_scale = 0.46 if use_pool_drop else 0.62
        camera_eye = tuple(
            camera_target[index]
            + camera_distance_scale
            * (scene["camera_eye"][index] - scene["camera_target"][index])
            for index in range(3)
        )
    else:
        camera_eye = tuple(scene["camera_eye"])
    command = [
        r"Y:\isaacsim\python.bat",
        windows_path(ROOT / "soft_body_bounce_hero.py"),
        "--frames", str(args.frames),
        "--substeps",
        str(DEEP_POUR_PHYSICS_SUBSTEPS if args.server_deep_pour else 4),
        "--width", "480", "--height", "480",
        "--renderer", "RaytracedLighting",
        "--output", windows_path(output),
        "--model", bodies[0]["model"],
        "--model-height", str(bodies[0]["model_height"]),
        "--model-scale-mode", "max_extent",
        "--model-yaw", str(bodies[0]["model_yaw"]),
        "--drop-height", str(bodies[0]["drop_height"]),
        "--youngs-modulus", str(bodies[0]["youngs_modulus"]),
        "--poissons-ratio", str(bodies[0]["poissons_ratio"]),
        "--linear-damping", str(bodies[0]["linear_damping"]),
        "--settling-damping", str(bodies[0]["settling_damping"]),
        "--restitution", str(bodies[0]["restitution"]),
        "--density", str(bodies[0]["density"]),
        "--expected-behavior", "soft",
        "--validation-profile", "generic",
        "--deformable-resolution", "24",
        "--deformable-solver-position-iterations",
        str(DEEP_POUR_DEFORMABLE_ITERATIONS) if args.server_deep_pour else "24",
        "--deformable-collision-remeshing",
        "--deformable-remeshing-resolution", "0",
        "--deformable-target-triangle-count",
        str(DEEP_POUR_COLLISION_TRIANGLE_COUNT)
        if args.server_deep_pour
        else "0",
        "--deformable-force-conforming",
        "--environment-usd", windows_path(scene["usd"]),
        "--environment-ground-only",
        "--support-top-y", str(scene["support_y"]),
        "--spawn-x", str(centre_x),
        "--spawn-z", str(centre_z),
        "--camera-eye", *map(str, camera_eye),
        "--camera-target", *map(str, camera_target),
        "--body-config", windows_path(body_config_path),
        "--coupled-event-config", windows_path(event_config_path),
        "--export-blender-usd", "--blender-usd-name", "mixed_bodies.usdc",
        "--skip-preview-render",
    ]
    if args.server_deep_pour:
        command.extend(
            [
                "--physics-max-bias-coefficient",
                str(DEEP_POUR_MAX_BIAS_COEFFICIENT),
            ]
        )
    if args.server_deep_pour:
        command.extend(
            [
                "--deformable-max-depenetration-velocity",
                "0.25",
                "--audit-tet-trajectory",
            ]
        )
    if scene["needs_exact_collision"]:
        command.extend(["--prebuilt-collision-usd", windows_path(exact_collision)])
    command_path = output / "run_command.json"
    write_json(
        command_path,
        {
            "schema": 1,
            "scene": args.scene,
            "layout": layout,
            "frames": args.frames,
            "production_target": deep_spec,
            "contact_stability_probe": bool(args.deep_pour_contact_probe),
            "suggested_capture": (
                {
                    "physics_frame_start": deep_spec["visible_start_frame"],
                    "physics_frame_stop": (
                        args.frames
                        if args.deep_pour_contact_probe
                        else deep_spec["visible_stop_frame"]
                    ),
                    "physics_frame_stride_for_30fps": 2,
                }
                if deep_spec is not None
                else None
            ),
            "command": command,
        },
    )
    print(
        json.dumps(
            {
                "scene": args.scene,
                "layout": layout,
                "output": str(output),
                "body_config": str(body_config_path),
                "event_config": str(event_config_path),
                "command_file": str(command_path),
                "source_frames": [source_start, source_stop],
                "production_target": deep_spec,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
