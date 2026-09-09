"""Optional PhysX PBD pour event embedded in an existing authored scene.

The event owns only local apparatus and primary-fluid state.  The surrounding
scene and the rigid/deformable bodies remain owned by ``soft_body_bounce_hero``.
Keeping that boundary explicit lets one event layout move between the fourteen
validated environments without treating the apparatus as a replacement scene.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from coupled_scene.water_defaults import water_vorticity


PRODUCT = "coupled_scene_pbd_pour_event"
SCHEMA = 1
PREAUTHORED_SET_STRATEGIES = {
    "per_emission_frame_sets",
    "preauthored_frame_sets",
    "preauthored_density_sets",
}
MULTI_SET_STRATEGIES = PREAUTHORED_SET_STRATEGIES | {"chunked_density_sets"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            # Windows can briefly deny replacement while a validator has the
            # previous manifest open.  The cache frame itself is already safe;
            # retry only the small manifest handoff.
            time.sleep(0.05)


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _triplet(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain three finite values")
    return result


def _positive(value, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _load_configuration(path: Path, frame_count: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("product") != PRODUCT:
        raise ValueError(f"Unsupported coupled event configuration: {path}")
    if payload.get("layout") not in {
        "glass_cabinet_pour",
        "glass_cabinet_compact_pour",
        "glass_cabinet_pool_drop",
    }:
        raise ValueError(
            "Only glass-cabinet pour and pool-drop layouts are currently "
            "implemented"
        )
    cabinet = payload.get("cabinet") or {}
    source = payload.get("source") or {}
    centre = _triplet(cabinet.get("centre"), "cabinet.centre")
    inner_size = _triplet(cabinet.get("inner_size"), "cabinet.inner_size")
    if np.any(inner_size <= 0.0):
        raise ValueError("cabinet.inner_size must be positive")
    render_thickness = _positive(
        cabinet.get("render_thickness", cabinet.get("wall_thickness")),
        "cabinet.render_thickness",
    )
    collision_thickness = _positive(
        cabinet.get("collision_thickness", cabinet.get("wall_thickness")),
        "cabinet.collision_thickness",
    )
    spacing = _positive(source.get("spacing"), "source.spacing")
    nozzle_centre = _triplet(source.get("centre"), "source.centre")
    nozzle_size = _triplet(source.get("size"), "source.size")
    if np.any(nozzle_size <= 0.0):
        raise ValueError("source.size must be positive")
    velocity = _triplet(source.get("velocity"), "source.velocity")
    emission_model = str(source.get("emission_model", "static_slab"))
    if emission_model != "static_slab" and np.linalg.norm(velocity) <= 0.0:
        raise ValueError("source.velocity must be non-zero")
    start_frame = int(source.get("start_frame"))
    stop_frame = int(source.get("stop_frame"))
    interval_frames = int(source.get("interval_frames", 1))
    if not (1 <= start_frame <= stop_frame <= frame_count):
        raise ValueError("source frame interval must lie inside the physics run")
    if interval_frames < 1:
        raise ValueError("source.interval_frames must be positive")
    maximum_speed = _positive(source.get("maximum_speed", 8.0), "source.maximum_speed")
    max_depenetration_velocity = source.get("max_depenetration_velocity")
    if max_depenetration_velocity is not None:
        max_depenetration_velocity = _positive(
            max_depenetration_velocity,
            "source.max_depenetration_velocity",
        )
    density = _positive(source.get("density", 1000.0), "source.density")
    return {
        **payload,
        "cabinet": {
            **cabinet,
            "centre": centre,
            "inner_size": inner_size,
            "wall_thickness": render_thickness,
            "render_thickness": render_thickness,
            "collision_thickness": collision_thickness,
        },
        "source": {
            **source,
            "spacing": spacing,
            "centre": nozzle_centre,
            "size": nozzle_size,
            "velocity": velocity,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "interval_frames": interval_frames,
            "maximum_speed": maximum_speed,
            "max_depenetration_velocity": max_depenetration_velocity,
            "density": density,
        },
    }


def _axis_samples(centre: float, extent: float, spacing: float) -> np.ndarray:
    count = max(1, int(np.floor(extent / spacing)))
    offsets = (np.arange(count, dtype=np.float64) - 0.5 * (count - 1)) * spacing
    return centre + offsets


def _emission_chunks(batch_records: list[dict], capacity: int) -> list[list[dict]]:
    """Pack consecutive births without changing their timing or ID order.

    Only the last chunk may grow during emission. All earlier chunks retain
    their native simulated state, bounding each append's array traffic.
    """
    if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
        raise ValueError("emission_chunk_particles must be a positive integer")
    chunks = []
    current_count = 0
    for record in batch_records:
        count = len(record["positions"])
        if count > capacity:
            raise ValueError(
                f"Emission birth has {count} particles, exceeding chunk capacity "
                f"{capacity}; increase emission_chunk_particles"
            )
        if not chunks or current_count + count > capacity:
            chunks.append([])
            current_count = 0
        chunks[-1].append(record)
        current_count += count
    return chunks


class PbdPourEvent:
    """Runtime owner for a pre-authored, append-only PBD pour."""

    def __init__(
        self,
        stage,
        physics_scene,
        configuration_path,
        output_directory,
        frame_count,
        physics_substeps,
    ):
        import carb
        import omni.physx.bindings._physx as physx_settings_bindings
        from omni.physx.scripts import particleUtils, physicsUtils
        from pxr import (
            Gf,
            PhysicsSchemaTools,
            PhysxSchema,
            Sdf,
            UsdGeom,
            UsdPhysics,
            UsdUtils,
            Vt,
        )

        self.stage = stage
        self.configuration_path = Path(configuration_path).resolve()
        self.output_directory = Path(output_directory).resolve()
        self.frame_count = int(frame_count)
        self._Vt = Vt
        self.physics_substeps = int(physics_substeps)
        if self.physics_substeps < 1:
            raise ValueError("physics_substeps must be positive")
        self.configuration = _load_configuration(
            self.configuration_path, self.frame_count
        )
        self.preroll_steps = int(self.configuration.get("settle_steps", 0))
        if self.preroll_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        self.preroll_prepared = False
        self.preroll_body_gravity_attrs = []
        self.fluid_directory = self.output_directory / "primary_fluid"
        self.fluid_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.fluid_directory / "manifest.json"
        self.particle_instancer = None
        self.active_instancers = []
        self.active_id_chunks = []
        self.batch_records = []
        self.enabled_batches = 0
        self.emission_statistics = {
            "activation_count": 0,
            "old_particle_rows_read": 0,
            "particle_rows_written": 0,
            "maximum_old_particle_rows_per_activation": 0,
            "authoring_seconds": 0.0,
            "attribute_read_seconds": 0.0,
            "array_build_seconds": 0.0,
            "attribute_write_seconds": 0.0,
            "activation_update_seconds": 0.0,
        }
        self._physx_bindings = physx_settings_bindings
        self.solver_statistics_interface = None
        self.solver_statistics_setup_error = None
        self.solver_resource_peaks = {}
        self.solver_resource_peak_steps = {}
        self.maximum_speed_seen = 0.0
        self.maximum_speed_cap_fraction = 0.0
        self.maximum_lateral_escape_fraction = 0.0
        self.maximum_below_floor_fraction = 0.0
        self.maximum_lateral_boundary_penetration_fraction = 0.0
        self.maximum_floor_boundary_penetration_fraction = 0.0
        self.maximum_over_rim_fraction = 0.0
        self.over_rim_particle_ids = set()
        self.side_tunnel_particle_ids = set()
        self.bottom_tunnel_particle_ids = set()
        self.samples = []
        self.body_interaction_states = {}

        settings = carb.settings.get_settings()
        settings.set(physx_settings_bindings.SETTING_UPDATE_TO_USD, True)
        settings.set(physx_settings_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
        settings.set(physx_settings_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
        settings.set(physx_settings_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)

        cabinet = self.configuration["cabinet"]
        source = self.configuration["source"]
        centre = cabinet["centre"]
        inner = cabinet["inner_size"]
        thickness = cabinet["collision_thickness"]
        floor_y = float(centre[1])
        contact_offset = max(0.004, 0.35 * source["spacing"])

        def add_panel(name: str, panel_centre, size):
            path = Sdf.Path(f"/World/CoupledEvent/GlassCabinet/{name}")
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr().Set(2.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, panel_centre)))
            xform.AddScaleOp().Set(Gf.Vec3f(*(0.5 * np.asarray(size))))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(
                True
            )
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateContactOffsetAttr().Set(contact_offset)
            collision.CreateRestOffsetAttr().Set(0.0)
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
            return str(path)

        half_x, _height, half_z = 0.5 * inner
        cabinet_top = floor_y + inner[1]
        panel_paths = [
            add_panel(
                "Bottom",
                (centre[0], floor_y - 0.5 * thickness, centre[2]),
                (inner[0] + 2.0 * thickness, thickness, inner[2] + 2.0 * thickness),
            ),
            add_panel(
                "Left",
                (
                    centre[0] - half_x - 0.5 * thickness,
                    floor_y + 0.5 * inner[1] - 0.5 * thickness,
                    centre[2],
                ),
                (thickness, inner[1] + thickness, inner[2] + 2.0 * thickness),
            ),
            add_panel(
                "Right",
                (
                    centre[0] + half_x + 0.5 * thickness,
                    floor_y + 0.5 * inner[1] - 0.5 * thickness,
                    centre[2],
                ),
                (thickness, inner[1] + thickness, inner[2] + 2.0 * thickness),
            ),
            add_panel(
                "Front",
                (
                    centre[0],
                    floor_y + 0.5 * inner[1] - 0.5 * thickness,
                    centre[2] - half_z - 0.5 * thickness,
                ),
                (inner[0], inner[1] + thickness, thickness),
            ),
            add_panel(
                "Back",
                (
                    centre[0],
                    floor_y + 0.5 * inner[1] - 0.5 * thickness,
                    centre[2] + half_z + 0.5 * thickness,
                ),
                (inner[0], inner[1] + thickness, thickness),
            ),
        ]

        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        physx_scene.CreateGpuMaxParticleContactsAttr().Set(
            int(self.configuration.get("gpu_max_particle_contacts", 1_000_000))
        )
        gpu_max_deformable_volume_contacts = int(
            self.configuration.get(
                "gpu_max_deformable_volume_contacts", 4 * 1024 * 1024
            )
        )
        gpu_max_deformable_surface_contacts = int(
            self.configuration.get(
                "gpu_max_deformable_surface_contacts", 4 * 1024 * 1024
            )
        )
        if gpu_max_deformable_volume_contacts < 1024 * 1024:
            raise ValueError(
                "gpu_max_deformable_volume_contacts must be at least 1048576"
            )
        if gpu_max_deformable_surface_contacts < 1024 * 1024:
            raise ValueError(
                "gpu_max_deformable_surface_contacts must be at least 1048576"
            )
        physx_scene.CreateGpuMaxDeformableVolumeContactsAttr().Set(
            gpu_max_deformable_volume_contacts
        )
        physx_scene.CreateGpuMaxDeformableSurfaceContactsAttr().Set(
            gpu_max_deformable_surface_contacts
        )
        gpu_collision_stack_size = int(
            self.configuration.get("gpu_collision_stack_size", 128 * 1024 * 1024)
        )
        if gpu_collision_stack_size < 64 * 1024 * 1024:
            raise ValueError("gpu_collision_stack_size must be at least 64 MiB")
        physx_scene.CreateGpuCollisionStackSizeAttr().Set(gpu_collision_stack_size)
        try:
            self.solver_statistics_interface = (
                physx_settings_bindings.acquire_physx_statistics_interface()
            )
            stage_cache = UsdUtils.StageCache.Get()
            stage_cache_id = stage_cache.GetId(stage)
            if not stage_cache_id.IsValid():
                stage_cache_id = stage_cache.Insert(stage)
            self.solver_statistics_stage_id = stage_cache_id.ToLongInt()
            self.solver_statistics_scene_path = PhysicsSchemaTools.encodeSdfPath(
                physics_scene.GetPath()
            )[0]
        except Exception as exc:
            self.solver_statistics_setup_error = f"{type(exc).__name__}: {exc}"
        spacing = source["spacing"]
        self.body_contact_threshold = _positive(
            self.configuration.get("body_contact_threshold", 2.0 * spacing),
            "body_contact_threshold",
        )
        self.body_visible_surface_threshold = _positive(
            self.configuration.get(
                "body_visible_surface_threshold", max(3.0 * spacing, 0.012)
            ),
            "body_visible_surface_threshold",
        )
        if self.body_visible_surface_threshold > self.body_contact_threshold:
            raise ValueError(
                "body_visible_surface_threshold cannot exceed body_contact_threshold"
            )
        self.minimum_body_contact_particles = int(
            self.configuration.get("minimum_body_contact_particles", 3)
        )
        if self.minimum_body_contact_particles < 1:
            raise ValueError("minimum_body_contact_particles must be positive")
        self.minimum_visible_surface_particles = int(
            self.configuration.get("minimum_visible_surface_particles", 1)
        )
        if self.minimum_visible_surface_particles < 1:
            raise ValueError("minimum_visible_surface_particles must be positive")
        self.minimum_consecutive_visible_surface_frames = int(
            self.configuration.get("minimum_consecutive_visible_surface_frames", 2)
        )
        if self.minimum_consecutive_visible_surface_frames < 1:
            raise ValueError(
                "minimum_consecutive_visible_surface_frames must be positive"
            )
        self.required_body_kinds = tuple(
            str(value)
            for value in self.configuration.get(
                "required_body_kinds", ["rigid", "deformable"]
            )
        )
        if not self.required_body_kinds:
            raise ValueError("required_body_kinds must not be empty")
        fluid_rest_offset = 0.5 * spacing
        particle_contact_offset = fluid_rest_offset / 0.6
        authored_particle_contact_offset = float(source.get('particle_contact_offset', particle_contact_offset))
        if 'particle_contact_offset' in source and (not np.isfinite(authored_particle_contact_offset) or authored_particle_contact_offset <= particle_contact_offset):
            raise ValueError('Explicit particle_contact_offset must exceed solid_rest_offset')
        particle_system_path = Sdf.Path("/World/CoupledEvent/ParticleSystem")
        particle_system = particleUtils.add_physx_particle_system(
            stage,
            particle_system_path,
            simulation_owner=physics_scene.GetPath(),
            contact_offset=particle_contact_offset + 0.001,
            rest_offset=particle_contact_offset,
            particle_contact_offset=authored_particle_contact_offset,
            solid_rest_offset=particle_contact_offset,
            fluid_rest_offset=fluid_rest_offset,
            enable_ccd=True,
            solver_position_iterations=int(source.get("solver_position_iterations", 16)),
            max_depenetration_velocity=source.get("max_depenetration_velocity"),
            max_neighborhood=int(source.get("max_neighborhood", 96)),
            neighborhood_scale=1.01,
            max_velocity=source["maximum_speed"],
        )
        material_path = Sdf.Path("/World/CoupledEvent/Looks/WaterPhysics")
        particleUtils.add_pbd_particle_material(
            stage,
            material_path,
            density=source["density"],
            friction=float(source.get("friction", 0.05)),
            damping=float(source.get("damping", 0.01)),
            viscosity=float(source.get("viscosity", 0.002)),
            vorticity_confinement=water_vorticity(source),
            surface_tension=float(source.get("surface_tension", 0.0074)),
            cohesion=float(source.get("cohesion", 0.01)),
            adhesion=float(source.get("adhesion", 0.0)),
            cfl_coefficient=1.0,
        )
        physicsUtils.add_physics_material_to_prim(
            stage, particle_system.GetPrim(), material_path
        )
        self.water_damping_attr = stage.GetPrimAtPath(material_path).GetAttribute(
            "physxPBDMaterial:damping"
        )

        xs = _axis_samples(source["centre"][0], source["size"][0], spacing)
        zs = _axis_samples(source["centre"][2], source["size"][2], spacing)
        emission_model = str(source.get("emission_model", "static_slab"))
        continuous_inlet_schedule = None
        if emission_model == "continuous_inlet":
            if source["interval_frames"] != 1:
                raise ValueError(
                    "continuous_inlet requires source.interval_frames == 1"
                )
            velocity_magnitude = float(np.linalg.norm(source["velocity"]))
            if abs(float(source["velocity"][1])) / velocity_magnitude < 0.999:
                raise ValueError(
                    "continuous_inlet currently requires an axis-aligned vertical source"
                )

            gravity_direction_value = physics_scene.GetGravityDirectionAttr().Get()
            gravity_magnitude_value = physics_scene.GetGravityMagnitudeAttr().Get()
            gravity_direction = np.asarray(
                gravity_direction_value
                if gravity_direction_value is not None
                else (0.0, -1.0, 0.0),
                dtype=np.float64,
            )
            direction_norm = float(np.linalg.norm(gravity_direction))
            if direction_norm <= 0.0:
                raise RuntimeError("Physics gravity direction is zero")
            gravity = gravity_direction / direction_norm * float(
                gravity_magnitude_value
                if gravity_magnitude_value is not None
                else 9.81
            )

            # Each x/z lattice column crosses the invisible inlet once per
            # spacing / speed seconds.  A deterministic low-discrepancy phase
            # offset prevents the columns from entering as synchronized,
            # perfectly planar sheets while preserving Q = A * |v|.
            xz_positions = np.stack(
                np.meshgrid(xs, zs, indexing="ij"), axis=-1
            ).reshape((-1, 2))
            column_count = int(len(xz_positions))
            crossing_period = float(spacing / velocity_magnitude)
            golden_fraction = 0.6180339887498949
            next_crossing_times = (
                np.mod(np.arange(column_count, dtype=np.float64) * golden_fraction, 1.0)
                * crossing_period
            )
            substep_dt = 1.0 / (60.0 * self.physics_substeps)
            birth_frames = list(
                range(
                    source["start_frame"],
                    source["stop_frame"] + 1,
                    source["interval_frames"],
                )
            )
            continuous_inlet_schedule = []
            time_epsilon = 1.0e-12
            for local_frame, birth_frame in enumerate(birth_frames):
                for birth_substep in range(self.physics_substeps):
                    local_step = local_frame * self.physics_substeps + birth_substep
                    step_start = local_step * substep_dt
                    step_stop = step_start + substep_dt
                    due_columns_chunks = []
                    crossing_time_chunks = []
                    while True:
                        due_columns = np.flatnonzero(
                            next_crossing_times < step_stop - time_epsilon
                        )
                        if not len(due_columns):
                            break
                        due_columns_chunks.append(due_columns)
                        crossing_time_chunks.append(next_crossing_times[due_columns].copy())
                        next_crossing_times[due_columns] += crossing_period
                    if not due_columns_chunks:
                        continue
                    due_columns = np.concatenate(due_columns_chunks)
                    crossing_times = np.concatenate(crossing_time_chunks)
                    within_step = np.clip(
                        crossing_times - step_start, 0.0, substep_dt
                    )
                    crossing_positions = np.empty(
                        (len(due_columns), 3), dtype=np.float64
                    )
                    crossing_positions[:, 0] = xz_positions[due_columns, 0]
                    crossing_positions[:, 1] = source["centre"][1]
                    crossing_positions[:, 2] = xz_positions[due_columns, 1]
                    # Insert at the start of the substep at the state that will
                    # cross the inlet plane with the requested source velocity
                    # at its individual phase time.  This is only an inlet
                    # boundary condition; all downstream motion is native PBD.
                    positions = (
                        crossing_positions
                        - within_step[:, None] * source["velocity"][None, :]
                        + 0.5 * within_step[:, None] ** 2 * gravity[None, :]
                    ).astype(np.float32)
                    velocities = (
                        source["velocity"][None, :]
                        - within_step[:, None] * gravity[None, :]
                    ).astype(np.float32)
                    order = np.argsort(crossing_times, kind="stable")
                    continuous_inlet_schedule.append(
                        (
                            int(birth_frame),
                            int(birth_substep),
                            positions[order],
                            velocities[order],
                        )
                    )
            slice_count = float(velocity_magnitude / (spacing * 60.0))
            subbatch_templates = []
        elif emission_model == "substep_batches":
            layer_count = max(1, int(np.floor(source["size"][1] / spacing)))
            layer_groups = np.array_split(
                np.arange(layer_count, dtype=np.int64), self.physics_substeps
            )
            subbatch_templates = []
            for birth_substep, layer_group in enumerate(layer_groups):
                if not len(layer_group):
                    continue
                ys = _axis_samples(
                    source["centre"][1], len(layer_group) * spacing, spacing
                )
                positions = np.stack(
                    np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1
                ).reshape((-1, 3)).astype(np.float32)
                velocities = np.repeat(
                    source["velocity"].astype(np.float32)[None, :],
                    len(positions),
                    axis=0,
                )
                subbatch_templates.append(
                    (int(birth_substep), positions, velocities)
                )
            slice_count = layer_count
        elif emission_model in {
            "continuous_subframe_ballistic",
            "preauthored_frame_inlet",
        }:
            slice_count = max(1, int(np.floor(source["size"][1] / spacing)))
            frame_duration = 1.0 / 60.0
            output_ages = (
                (np.arange(slice_count, dtype=np.float64) + 0.5)
                / slice_count
                * frame_duration
            )
            # These particles all enter the stage at the start of the output
            # frame. Back-time their initial state so the native physics
            # substeps land each layer at the same frame-boundary state as a
            # genuinely continuous emitter would have at output_ages.
            initial_ages = output_ages - frame_duration
            gravity_direction_value = physics_scene.GetGravityDirectionAttr().Get()
            gravity_magnitude_value = physics_scene.GetGravityMagnitudeAttr().Get()
            gravity_direction = np.asarray(
                gravity_direction_value
                if gravity_direction_value is not None
                else (0.0, -1.0, 0.0),
                dtype=np.float64,
            )
            direction_norm = float(np.linalg.norm(gravity_direction))
            if direction_norm <= 0.0:
                raise RuntimeError("Physics gravity direction is zero")
            gravity = gravity_direction / direction_norm * float(
                gravity_magnitude_value
                if gravity_magnitude_value is not None
                else 9.81
            )
            xz_positions = np.stack(
                np.meshgrid(xs, zs, indexing="ij"), axis=-1
            ).reshape((-1, 2))
            position_layers = []
            velocity_layers = []
            for age in initial_ages:
                ballistic_centre = (
                    source["centre"]
                    + source["velocity"] * age
                    + 0.5 * gravity * age**2
                )
                layer_positions = np.empty((len(xz_positions), 3), dtype=np.float64)
                layer_positions[:, 0] = (
                    xz_positions[:, 0]
                    + ballistic_centre[0]
                    - source["centre"][0]
                )
                layer_positions[:, 1] = ballistic_centre[1]
                layer_positions[:, 2] = (
                    xz_positions[:, 1]
                    + ballistic_centre[2]
                    - source["centre"][2]
                )
                layer_velocity = source["velocity"] + gravity * age
                position_layers.append(layer_positions)
                velocity_layers.append(
                    np.repeat(layer_velocity[None, :], len(layer_positions), axis=0)
                )
            batch_positions = np.concatenate(position_layers).astype(np.float32)
            batch_velocities = np.concatenate(velocity_layers).astype(np.float32)
            subbatch_templates = [(0, batch_positions, batch_velocities)]
        elif emission_model == "static_slab":
            ys = _axis_samples(source["centre"][1], source["size"][1], spacing)
            batch_positions = np.stack(
                np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1
            ).reshape((-1, 3)).astype(np.float32)
            batch_velocities = np.repeat(
                source["velocity"].astype(np.float32)[None, :],
                len(batch_positions),
                axis=0,
            )
            slice_count = len(ys)
            subbatch_templates = [(0, batch_positions, batch_velocities)]
        else:
            raise ValueError(f"Unsupported source.emission_model: {emission_model}")
        self.manual_substep_emission = emission_model in {
            "substep_batches",
            "continuous_inlet",
            "preauthored_frame_inlet",
        }
        next_id = 0
        birth_frames = list(
            range(
                source["start_frame"],
                source["stop_frame"] + 1,
                source["interval_frames"],
            )
        )
        if continuous_inlet_schedule is not None:
            scheduled_batches = continuous_inlet_schedule
        else:
            scheduled_batches = [
                (birth_frame, birth_substep, batch_positions, batch_velocities)
                for birth_frame in birth_frames
                for birth_substep, batch_positions, batch_velocities in subbatch_templates
            ]
        for (
            birth_frame,
            birth_substep,
            batch_positions,
            batch_velocities,
        ) in scheduled_batches:
            ids = np.arange(
                next_id, next_id + len(batch_positions), dtype=np.int64
            )
            next_id += len(batch_positions)
            self.batch_records.append(
                {
                    "birth_frame": int(birth_frame),
                    "birth_substep": int(birth_substep),
                    "positions": batch_positions,
                    "velocities": batch_velocities,
                    "ids": ids,
                }
            )

        particle_set_strategy = str(
            self.configuration.get(
                "particle_set_strategy", "single_shared_append_only_set"
            )
        )
        chunk_capacity = self.configuration.get("emission_chunk_particles", 16384)
        particle_set_count = 1
        if particle_set_strategy == "chunked_density_sets":
            chunks = _emission_chunks(self.batch_records, chunk_capacity)
            particle_set_count = len(chunks)
            self.particle_mass_attr = None
            for chunk_index, records in enumerate(chunks):
                particle_path = Sdf.Path(
                    f"/World/CoupledEvent/EmitterChunk_{chunk_index:04d}"
                )
                particle_prim = particleUtils.add_physx_particleset_pointinstancer(
                    stage, particle_path, Vt.Vec3fArray(), Vt.Vec3fArray(),
                    particle_system_path, self_collision=True, fluid=True,
                    particle_group=0, particle_mass=0.0, density=source["density"],
                )
                UsdPhysics.MassAPI(particle_prim).GetMassAttr().Clear()
                particle_prim.CreateAttribute(
                    "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
                ).Set(sum(len(record["positions"]) for record in records))
                enabled = particle_prim.GetAttribute("physxParticle:particleEnabled")
                if not enabled:
                    raise RuntimeError("PBD emitter chunk has no particleEnabled attribute")
                enabled.Set(False)
                instancer = UsdGeom.PointInstancer.Get(stage, particle_path)
                for index, record in enumerate(records):
                    record["instancer"] = instancer
                    record["enabled"] = enabled
                    record["starts_chunk"] = index == 0
                prototype = UsdGeom.Imageable.Get(
                    stage, particle_path.AppendChild("particlePrototype0")
                )
                if prototype:
                    prototype.MakeInvisible()
        elif particle_set_strategy in PREAUTHORED_SET_STRATEGIES:
            particle_set_count = len(self.batch_records)
            for batch_index, record in enumerate(self.batch_records):
                particle_path = Sdf.Path(
                    f"/World/CoupledEvent/EmitterBatch_{batch_index:04d}"
                )
                particle_prim = particleUtils.add_physx_particleset_pointinstancer(
                    stage,
                    particle_path,
                    Vt.Vec3fArray.FromNumpy(record["positions"]),
                    Vt.Vec3fArray.FromNumpy(record["velocities"]),
                    particle_system_path,
                    self_collision=True,
                    fluid=True,
                    particle_group=0,
                    # This helper takes PER-PARTICLE mass and multiplies by N
                    # itself. Passing total batch mass here would create N
                    # times too much water mass. Density sets match the proven
                    # shared emitter's density-derived mass policy instead.
                    particle_mass=(
                        0.0
                        if particle_set_strategy == "preauthored_density_sets"
                        else source["density"] * spacing**3
                    ),
                    density=source["density"],
                )
                if particle_set_strategy == "preauthored_density_sets":
                    UsdPhysics.MassAPI(particle_prim).GetMassAttr().Clear()
                else:
                    # Verify the installed helper's contract against the USD
                    # it actually authored, before attaching the physics stage.
                    expected_mass = len(record["positions"]) * source["density"] * spacing**3
                    authored_mass = UsdPhysics.MassAPI(particle_prim).GetMassAttr().Get()
                    if authored_mass is None or not np.isclose(
                        authored_mass, expected_mass, rtol=1e-6, atol=0.0
                    ):
                        raise RuntimeError(
                            f"Particle batch mass mismatch: {authored_mass} kg, "
                            f"expected {expected_mass} kg for {len(record['positions'])} particles"
                        )
                particle_prim.CreateAttribute(
                    "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
                ).Set(len(record["positions"]))
                enabled = particle_prim.GetAttribute(
                    "physxParticle:particleEnabled"
                )
                if not enabled:
                    raise RuntimeError(
                        "PBD emitter batch has no particleEnabled attribute"
                    )
                enabled.Set(False)
                record["enabled"] = enabled
                record["instancer"] = UsdGeom.PointInstancer.Get(
                    stage, particle_path
                )
                prototype = UsdGeom.Imageable.Get(
                    stage, particle_path.AppendChild("particlePrototype0")
                )
                if prototype:
                    prototype.MakeInvisible()
        elif particle_set_strategy in {
            "single_shared_append_only_set",
            "single_shared_density_set",
        }:
            particle_path = Sdf.Path("/World/CoupledEvent/FluidParticles")
            particle_prim = particleUtils.add_physx_particleset_pointinstancer(
                stage,
                particle_path,
                Vt.Vec3fArray(),
                Vt.Vec3fArray(),
                particle_system_path,
                self_collision=True,
                fluid=True,
                particle_group=0,
                # A zero authored mass makes PhysX derive stable per-particle
                # mass from density as this shared set grows during emission.
                particle_mass=0.0,
                density=source["density"],
            )
            particle_prim.CreateAttribute(
                "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
            ).Set(next_id)
            enabled = particle_prim.GetAttribute("physxParticle:particleEnabled")
            if not enabled:
                raise RuntimeError(
                    "Shared PBD particle set has no particleEnabled attribute"
                )
            enabled.Set(True)
            self.particle_instancer = UsdGeom.PointInstancer.Get(
                stage, particle_path
            )
            mass_api = UsdPhysics.MassAPI(particle_prim)
            self.particle_mass_attr = mass_api.GetMassAttr()
            if not self.particle_mass_attr:
                raise RuntimeError("Shared PBD particle set has no mass attribute")
            self.per_particle_mass = float(source["density"] * spacing**3)
            if particle_set_strategy == "single_shared_density_set":
                # MassAPI.mass is total set mass in USD.  Re-authoring it while
                # the array grows forces an avoidable physics reparse.  Clear
                # that stronger opinion and use density, whose derived
                # per-particle mass remains stable as particles are appended.
                self.particle_mass_attr.Clear()
                density_attr = mass_api.GetDensityAttr()
                density_attr.Set(float(source["density"]))
                self.particle_mass_attr = None
            prototype = UsdGeom.Imageable.Get(
                stage, particle_path.AppendChild("particlePrototype0")
            )
            if prototype:
                prototype.MakeInvisible()
        else:
            raise ValueError(
                f"Unsupported particle_set_strategy: {particle_set_strategy}"
            )
        self.particle_set_strategy = particle_set_strategy

        if continuous_inlet_schedule is not None:
            subbatch_particle_counts = [
                int(len(positions))
                for _frame, _substep, positions, _velocities in continuous_inlet_schedule
            ]
            particles_per_emission_frame = float(
                next_id / max(1, len(birth_frames))
            )
        else:
            subbatch_particle_counts = [
                int(len(positions))
                for _substep, positions, _velocities in subbatch_templates
            ]
            particles_per_emission_frame = int(sum(subbatch_particle_counts))

        self.metadata = {
            "configuration": str(self.configuration_path),
            "configuration_sha256": _sha256_file(self.configuration_path),
            "layout": self.configuration["layout"],
            "scene": self.configuration.get("scene"),
            "cabinet": {
                "centre": centre.tolist(),
                "floor_y": floor_y,
                "top_y": float(cabinet_top),
                "inner_size": inner.tolist(),
                "wall_thickness": cabinet["render_thickness"],
                "render_thickness": cabinet["render_thickness"],
                "collision_thickness": thickness,
                "collision_panels": panel_paths,
                "render_policy": "transparent prop is reconstructed by the downstream scene renderer",
            },
            "source": {
                "spacing": spacing,
                "vorticity_confinement": float(stage.GetPrimAtPath(material_path).GetAttribute(
                    "physxPBDMaterial:vorticityConfinement").Get()),
                "centre": source["centre"].tolist(),
                "size": source["size"].tolist(),
                "velocity": source["velocity"].tolist(),
                "start_frame": source["start_frame"],
                "stop_frame": source["stop_frame"],
                "interval_frames": source["interval_frames"],
                "emission_model": emission_model,
                "emission_timing": (
                    "manual_physics_substep_activation"
                    if self.manual_substep_emission
                    else "native_frame_step_with_back_time_compensated_layers"
                    if emission_model == "continuous_subframe_ballistic"
                    else "native_frame_step"
                ),
                "emission_subframe_slices": (
                    float(slice_count)
                    if emission_model == "continuous_inlet"
                    else int(slice_count)
                ),
                "emission_physics_substeps": int(self.physics_substeps),
                "solver_position_iterations": int(
                    source.get("solver_position_iterations", 16)
                ),
                "max_depenetration_velocity_m_s": source.get(
                    "max_depenetration_velocity"
                ),
                "maximum_velocity_m_s": source["maximum_speed"],
                "subbatch_particle_counts": subbatch_particle_counts,
                "particles_per_batch": particles_per_emission_frame,
                "particles_per_emission_frame": particles_per_emission_frame,
                "emission_frame_count": int(len(birth_frames)),
                "batch_count": int(len(self.batch_records)),
                "particle_set_count": particle_set_count,
                "emission_chunk_particles": (
                    int(chunk_capacity)
                    if particle_set_strategy == "chunked_density_sets" else None
                ),
                "particle_set_strategy": particle_set_strategy,
                "mass_policy": (
                    "density_derived_constant_per_particle_mass"
                    if particle_set_strategy in {
                        "single_shared_density_set", "preauthored_density_sets",
                        "chunked_density_sets",
                    }
                    else "authored_total_mass"
                ),
                "maximum_particles": int(next_id),
            },
            "gpu_resources": {
                "maximum_particle_contacts": int(
                    self.configuration.get("gpu_max_particle_contacts", 1_000_000)
                ),
                "maximum_deformable_volume_contacts": (
                    gpu_max_deformable_volume_contacts
                ),
                "maximum_deformable_surface_contacts": (
                    gpu_max_deformable_surface_contacts
                ),
                "collision_stack_size_bytes": gpu_collision_stack_size,
                "maximum_measured_utilization": float(
                    self.configuration.get(
                        "gpu_resource_maximum_utilization", 0.90
                    )
                ),
            },
            "body_interaction_gate": {
                "method": (
                    "visible_surface_vertex_proximity_with_temporal_gate_and_"
                    "context_envelope"
                ),
                "physics_contact_envelope_m": self.body_contact_threshold,
                "visible_surface_proximity_m": self.body_visible_surface_threshold,
                "minimum_context_envelope_particles_per_body_frame": (
                    self.minimum_body_contact_particles
                ),
                "minimum_visible_surface_particles_per_body_frame": (
                    self.minimum_visible_surface_particles
                ),
                "minimum_consecutive_visible_surface_frames": (
                    self.minimum_consecutive_visible_surface_frames
                ),
                "required_body_kinds": list(self.required_body_kinds),
            },
            "preroll": {
                "steps": self.preroll_steps,
                "physics_hz": 60 * self.physics_substeps,
                "duration_s": self.preroll_steps
                / float(60 * self.physics_substeps),
                "settle_damping": float(
                    source.get("settle_damping", source.get("damping", 0.01))
                ),
                "runtime_damping": float(source.get("damping", 0.01)),
                "completed": False,
                "final_maximum_speed_m_s": None,
            },
        }
        self.conditioned_inlet = None
        if self.configuration.get('conditioned_inlet'):
            from coupled_scene.conditioned_inlet import ConditionedInlet
            self.conditioned_inlet = ConditionedInlet(self, self.configuration['conditioned_inlet'])
        self._write_manifest(complete=False)

    def after_substep(self, simulation_app, frame, substep):
        if self.conditioned_inlet is not None:
            self.conditioned_inlet.after_substep(simulation_app, frame, substep)

    def prepare_preroll(self, bodies, simulation_app) -> None:
        """Author a full pool and suspend falling bodies before stage attach."""
        if self.preroll_steps == 0:
            return
        if str(self.configuration["source"].get("emission_model")) != "static_slab":
            raise ValueError("settle_steps currently requires a static_slab pool")
        if self.configuration["source"]["start_frame"] != 1:
            raise ValueError("A pre-rolled static pool must start at frame 1")
        self.enable_due(1, 0, simulation_app)
        settle_damping = float(
            self.configuration["source"].get(
                "settle_damping", self.configuration["source"].get("damping", 0.01)
            )
        )
        if not np.isfinite(settle_damping) or settle_damping < 0.0:
            raise ValueError("source.settle_damping must be finite and non-negative")
        self.water_damping_attr.Set(settle_damping)
        for body in bodies:
            attribute_name = (
                "physxRigidBody:disableGravity"
                if body["physics_kind"] == "rigid"
                else "physxDeformableBody:disableGravity"
            )
            attribute = body["root"].GetAttribute(attribute_name)
            if not attribute or not attribute.IsValid():
                raise RuntimeError(
                    f"Body {body['index']} has no {attribute_name} attribute"
                )
            attribute.Set(True)
            self.preroll_body_gravity_attrs.append(attribute)
        self.preroll_prepared = True

    def run_preroll(self, simulation, simulation_app) -> None:
        """Settle the authored pool off-camera, then release every body."""
        if self.preroll_steps == 0:
            return
        if not self.preroll_prepared:
            raise RuntimeError("prepare_preroll must run before PhysX stage attach")
        timestep = 1.0 / float(60 * self.physics_substeps)
        for step in range(self.preroll_steps):
            simulation.simulate(
                timestep,
                (step - self.preroll_steps) * timestep,
            )
            simulation.fetch_results()
            self.sample_solver_resources(0, step)
            simulation_app.update()
            if step == 0 or (step + 1) % 60 == 0 or step + 1 == self.preroll_steps:
                print(
                    f"[coupled-preroll] step={step + 1}/{self.preroll_steps}",
                    flush=True,
                )
        positions, velocities, _ids = self.state_arrays()
        final_maximum_speed = float(
            np.linalg.norm(velocities, axis=1).max(initial=0.0)
        )
        self.water_damping_attr.Set(
            float(self.configuration["source"].get("damping", 0.01))
        )
        for attribute in self.preroll_body_gravity_attrs:
            attribute.Set(False)
        for _ in range(4):
            simulation_app.update()
        self.metadata["preroll"].update(
            {
                "completed": True,
                "particle_count": int(len(positions)),
                "final_maximum_speed_m_s": final_maximum_speed,
            }
        )
        self._write_manifest(complete=False)

    def _write_manifest(self, complete: bool, valid: bool | None = None) -> None:
        payload = {
            "schema": 1,
            "product": "coupled_scene_primary_fluid_cache",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **self.metadata,
            "emission_statistics": dict(self.emission_statistics),
            "state": {
                "complete": bool(complete),
                "valid": valid,
                "completed_frames": len(self.samples),
                "expected_frames": self.frame_count,
            },
            "samples": self.samples,
        }
        _atomic_json(self.manifest_path, payload)

    def enable_due(self, frame: int, substep: int, simulation_app) -> None:
        due = (int(frame), int(substep))
        changed = False
        while (
            self.enabled_batches < len(self.batch_records)
            and (
                self.batch_records[self.enabled_batches]["birth_frame"],
                self.batch_records[self.enabled_batches]["birth_substep"],
            )
            == due
        ):
            record = self.batch_records[self.enabled_batches]
            authoring_start = time.perf_counter()
            if self.particle_set_strategy in PREAUTHORED_SET_STRATEGIES:
                record["enabled"].Set(True)
                self.active_instancers.append(record["instancer"])
            else:
                if self.particle_set_strategy == "chunked_density_sets":
                    instancer = record["instancer"]
                else:
                    instancer = self.particle_instancer
                positions_attr = instancer.GetPositionsAttr()
                velocities_attr = instancer.GetVelocitiesAttr()
                proto_indices_attr = instancer.GetProtoIndicesAttr()
                read_start = time.perf_counter()
                current_positions_value = positions_attr.Get()
                current_velocities_value = velocities_attr.Get()
                current_proto_indices_value = proto_indices_attr.Get()
                self.emission_statistics["attribute_read_seconds"] += time.perf_counter() - read_start
                build_start = time.perf_counter()
                current_positions = np.asarray(
                    current_positions_value
                    if current_positions_value is not None
                    else [],
                    dtype=np.float32,
                ).reshape((-1, 3))
                current_velocities = np.asarray(
                    current_velocities_value
                    if current_velocities_value is not None
                    else [],
                    dtype=np.float32,
                ).reshape((-1, 3))
                current_proto_indices = np.asarray(
                    current_proto_indices_value
                    if current_proto_indices_value is not None
                    else [],
                    dtype=np.int32,
                )
                positions = np.concatenate(
                    (current_positions, record["positions"]), axis=0
                )
                velocities = np.concatenate(
                    (current_velocities, record["velocities"]), axis=0
                )
                proto_indices = np.concatenate(
                    (
                        current_proto_indices,
                        np.zeros(len(record["positions"]), dtype=np.int32),
                    )
                )
                self.emission_statistics["array_build_seconds"] += time.perf_counter() - build_start
                write_start = time.perf_counter()
                positions_attr.Set(self._Vt.Vec3fArray.FromNumpy(positions))
                velocities_attr.Set(self._Vt.Vec3fArray.FromNumpy(velocities))
                proto_indices_attr.Set(self._Vt.IntArray.FromNumpy(proto_indices))
                self.emission_statistics["attribute_write_seconds"] += time.perf_counter() - write_start
                self.emission_statistics["old_particle_rows_read"] += len(current_positions)
                self.emission_statistics["particle_rows_written"] += len(positions)
                self.emission_statistics["maximum_old_particle_rows_per_activation"] = max(
                    self.emission_statistics["maximum_old_particle_rows_per_activation"],
                    len(current_positions),
                )
                if self.particle_set_strategy == "chunked_density_sets" and record["starts_chunk"]:
                    record["enabled"].Set(True)
                    self.active_instancers.append(instancer)
                if self.particle_mass_attr is not None:
                    self.particle_mass_attr.Set(
                        float(len(positions) * self.per_particle_mass)
                    )
            self.emission_statistics["authoring_seconds"] += time.perf_counter() - authoring_start
            self.emission_statistics["activation_count"] += 1
            self.active_id_chunks.append(record["ids"])
            self.enabled_batches += 1
            changed = True
        if self.enabled_batches < len(self.batch_records):
            next_record = self.batch_records[self.enabled_batches]
            next_due = (
                next_record["birth_frame"],
                next_record["birth_substep"],
            )
            if next_due < due:
                raise RuntimeError(
                    f"Missed PBD emitter activation {next_due} before {due}"
                )
        if changed:
            update_start = time.perf_counter()
            simulation_app.update()
            self.emission_statistics["activation_update_seconds"] += time.perf_counter() - update_start

    def state_arrays(self):
        if not self.active_id_chunks:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        if self.particle_set_strategy in MULTI_SET_STRATEGIES:
            positions = np.ascontiguousarray(
                np.concatenate(
                    [
                        np.asarray(
                            instancer.GetPositionsAttr().Get(), dtype=np.float32
                        ).reshape((-1, 3))
                        for instancer in self.active_instancers
                    ]
                ),
                dtype=np.float32,
            )
            velocities = np.ascontiguousarray(
                np.concatenate(
                    [
                        np.asarray(
                            instancer.GetVelocitiesAttr().Get(), dtype=np.float32
                        ).reshape((-1, 3))
                        for instancer in self.active_instancers
                    ]
                ),
                dtype=np.float32,
            )
        else:
            positions = np.ascontiguousarray(
                np.asarray(
                    self.particle_instancer.GetPositionsAttr().Get(),
                    dtype=np.float32,
                ).reshape((-1, 3)),
                dtype=np.float32,
            )
            velocities = np.ascontiguousarray(
                np.asarray(
                    self.particle_instancer.GetVelocitiesAttr().Get(),
                    dtype=np.float32,
                ).reshape((-1, 3)),
                dtype=np.float32,
            )
        ids = np.ascontiguousarray(np.concatenate(self.active_id_chunks), dtype=np.int64)
        if positions.shape != velocities.shape or positions.shape != (len(ids), 3):
            raise RuntimeError("PBD state arrays have inconsistent shapes")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError("PBD state contains non-finite values")
        return positions, velocities, ids

    def sample_solver_resources(self, frame: int, substep: int) -> None:
        if self.solver_statistics_interface is None:
            return
        stats = self._physx_bindings.PhysicsSceneStats()
        try:
            success = self.solver_statistics_interface.get_physx_scene_statistics(
                self.solver_statistics_stage_id,
                self.solver_statistics_scene_path,
                stats,
            )
        except Exception as exc:
            self.solver_statistics_setup_error = f"{type(exc).__name__}: {exc}"
            self.solver_statistics_interface = None
            return
        if not success:
            self.solver_statistics_setup_error = "statistics query returned false"
            self.solver_statistics_interface = None
            return
        for name in (
            "gpu_mem_collision_stack_size",
            "gpu_mem_deformable_volume_contacts",
            "gpu_mem_deformable_surface_contacts",
            "gpu_mem_particle_contacts",
        ):
            value = int(getattr(stats, name))
            if value > self.solver_resource_peaks.get(name, -1):
                self.solver_resource_peaks[name] = value
                self.solver_resource_peak_steps[name] = {
                    "frame": int(frame),
                    "substep": int(substep),
                }

    def _solver_resource_report(self) -> dict:
        capacities = {
            "gpu_mem_collision_stack_size": self.metadata["gpu_resources"][
                "collision_stack_size_bytes"
            ],
            "gpu_mem_deformable_volume_contacts": self.metadata["gpu_resources"][
                "maximum_deformable_volume_contacts"
            ],
            "gpu_mem_deformable_surface_contacts": self.metadata["gpu_resources"][
                "maximum_deformable_surface_contacts"
            ],
            "gpu_mem_particle_contacts": self.metadata["gpu_resources"][
                "maximum_particle_contacts"
            ],
        }
        maximum_utilization = self.metadata["gpu_resources"][
            "maximum_measured_utilization"
        ]
        resources = {}
        for name, capacity in capacities.items():
            peak = self.solver_resource_peaks.get(name)
            utilization = None if peak is None else float(peak / capacity)
            resources[name] = {
                "peak": peak,
                "capacity": int(capacity),
                "utilization": utilization,
                "peak_step": self.solver_resource_peak_steps.get(name),
                "valid": bool(
                    peak is not None and utilization <= maximum_utilization
                ),
            }
        available = bool(
            self.solver_statistics_interface is not None
            and len(self.solver_resource_peaks) == len(capacities)
        )
        return {
            "available": available,
            "setup_error": self.solver_statistics_setup_error,
            "resources": resources,
            "valid": bool(
                available and all(item["valid"] for item in resources.values())
            ),
        }

    def capture(self, frame: int, body_snapshots) -> None:
        positions, velocities, ids = self.state_arrays()
        path = self.fluid_directory / f"frame_{frame:04d}.npz"
        _atomic_npz(
            path,
            schema=np.asarray("coupled-scene-primary-fluid-frame/v1"),
            output_frame=np.int32(frame),
            positions=positions,
            velocities=velocities,
            particle_ids=ids,
        )
        speed = np.linalg.norm(velocities, axis=1) if len(velocities) else np.empty(0)
        maximum_speed = float(speed.max(initial=0.0))
        self.maximum_speed_seen = max(self.maximum_speed_seen, maximum_speed)
        speed_cap_fraction = (
            float(np.mean(speed >= self.configuration["source"]["maximum_speed"] - 0.01))
            if len(speed)
            else 0.0
        )
        self.maximum_speed_cap_fraction = max(
            self.maximum_speed_cap_fraction, speed_cap_fraction
        )
        cabinet = self.configuration["cabinet"]
        centre = cabinet["centre"]
        half = 0.5 * cabinet["inner_size"]
        collision_thickness = cabinet["collision_thickness"]
        spacing = self.configuration["source"]["spacing"]
        if len(positions):
            lateral_boundary_penetration = (
                (positions[:, 0] < centre[0] - half[0] - 2.0 * spacing)
                | (positions[:, 0] > centre[0] + half[0] + 2.0 * spacing)
                | (positions[:, 2] < centre[2] - half[2] - 2.0 * spacing)
                | (positions[:, 2] > centre[2] + half[2] + 2.0 * spacing)
            )
            floor_boundary_penetration = positions[:, 1] < centre[1] - 2.0 * spacing
            over_rim_now = (
                lateral_boundary_penetration
                & (positions[:, 1] >= centre[1] + cabinet["inner_size"][1] - 2.0 * spacing)
            )
            self.over_rim_particle_ids.update(map(int, ids[over_rim_now]))
            lateral_outside_proxy = (
                (
                    positions[:, 0]
                    < centre[0] - half[0] - collision_thickness - 2.0 * spacing
                )
                | (
                    positions[:, 0]
                    > centre[0] + half[0] + collision_thickness + 2.0 * spacing
                )
                | (
                    positions[:, 2]
                    < centre[2] - half[2] - collision_thickness - 2.0 * spacing
                )
                | (
                    positions[:, 2]
                    > centre[2] + half[2] + collision_thickness + 2.0 * spacing
                )
            )
            for particle_id in ids[lateral_outside_proxy]:
                particle_id = int(particle_id)
                if particle_id not in self.over_rim_particle_ids:
                    self.side_tunnel_particle_ids.add(particle_id)
            below_floor_proxy = (
                positions[:, 1]
                < centre[1] - collision_thickness - 2.0 * spacing
            )
            for particle_id in ids[below_floor_proxy]:
                particle_id = int(particle_id)
                if particle_id not in self.over_rim_particle_ids:
                    self.bottom_tunnel_particle_ids.add(particle_id)
            lateral_boundary_fraction = float(np.mean(lateral_boundary_penetration))
            floor_boundary_fraction = float(np.mean(floor_boundary_penetration))
            over_rim_fraction = len(self.over_rim_particle_ids) / len(ids)
            lateral_fraction = len(self.side_tunnel_particle_ids) / len(ids)
            below_fraction = len(self.bottom_tunnel_particle_ids) / len(ids)
        else:
            lateral_boundary_fraction = 0.0
            floor_boundary_fraction = 0.0
            over_rim_fraction = 0.0
            lateral_fraction = 0.0
            below_fraction = 0.0

        body_surface_proximity = []
        for body_snapshot in body_snapshots:
            body_index = int(body_snapshot["body_index"])
            physics_kind = str(body_snapshot["physics_kind"])
            world_points = np.asarray(
                body_snapshot["world_points"], dtype=np.float64
            )
            if world_points.ndim != 2 or world_points.shape[1:] != (3,):
                raise RuntimeError(
                    f"Body {body_index} has invalid surface points for fluid contact audit"
                )
            if len(positions):
                distances, _indices = cKDTree(world_points).query(
                    positions.astype(np.float64, copy=False), k=1
                )
                nearest_distance = float(np.min(distances))
                contact_envelope_count = int(
                    np.count_nonzero(distances <= self.body_contact_threshold)
                )
                visible_surface_count = int(
                    np.count_nonzero(
                        distances <= self.body_visible_surface_threshold
                    )
                )
            else:
                nearest_distance = None
                contact_envelope_count = 0
                visible_surface_count = 0
            state = self.body_interaction_states.setdefault(
                body_index,
                {
                    "body_index": body_index,
                    "physics_kind": physics_kind,
                    "first_physics_contact_envelope_frame": None,
                    "maximum_physics_contact_envelope_particles": 0,
                    "first_visible_surface_proximity_frame": None,
                    "maximum_visible_surface_proximity_particles": 0,
                    "visible_surface_proximity_frame_count": 0,
                    "current_consecutive_visible_surface_proximity_frames": 0,
                    "maximum_consecutive_visible_surface_proximity_frames": 0,
                    "minimum_surface_vertex_distance_m": None,
                },
            )
            state["maximum_physics_contact_envelope_particles"] = max(
                state["maximum_physics_contact_envelope_particles"],
                contact_envelope_count,
            )
            state["maximum_visible_surface_proximity_particles"] = max(
                state["maximum_visible_surface_proximity_particles"],
                visible_surface_count,
            )
            if nearest_distance is not None:
                previous = state["minimum_surface_vertex_distance_m"]
                state["minimum_surface_vertex_distance_m"] = (
                    nearest_distance
                    if previous is None
                    else min(previous, nearest_distance)
                )
            if (
                contact_envelope_count >= self.minimum_body_contact_particles
                and state["first_physics_contact_envelope_frame"] is None
            ):
                state["first_physics_contact_envelope_frame"] = int(frame)
            if visible_surface_count >= self.minimum_visible_surface_particles:
                if state["first_visible_surface_proximity_frame"] is None:
                    state["first_visible_surface_proximity_frame"] = int(frame)
                state["visible_surface_proximity_frame_count"] += 1
                state["current_consecutive_visible_surface_proximity_frames"] += 1
                state["maximum_consecutive_visible_surface_proximity_frames"] = max(
                    state["maximum_consecutive_visible_surface_proximity_frames"],
                    state["current_consecutive_visible_surface_proximity_frames"],
                )
            else:
                state["current_consecutive_visible_surface_proximity_frames"] = 0
            body_surface_proximity.append(
                {
                    "body_index": body_index,
                    "physics_kind": physics_kind,
                    "nearest_surface_vertex_distance_m": nearest_distance,
                    "physics_contact_envelope_particle_count": contact_envelope_count,
                    "visible_surface_proximity_particle_count": visible_surface_count,
                }
            )
        self.maximum_lateral_escape_fraction = max(
            self.maximum_lateral_escape_fraction, lateral_fraction
        )
        self.maximum_below_floor_fraction = max(
            self.maximum_below_floor_fraction, below_fraction
        )
        self.maximum_over_rim_fraction = max(
            self.maximum_over_rim_fraction, over_rim_fraction
        )
        self.maximum_lateral_boundary_penetration_fraction = max(
            self.maximum_lateral_boundary_penetration_fraction,
            lateral_boundary_fraction,
        )
        self.maximum_floor_boundary_penetration_fraction = max(
            self.maximum_floor_boundary_penetration_fraction,
            floor_boundary_fraction,
        )
        self.samples.append(
            {
                "frame": int(frame),
                "file": path.name,
                "sha256": _sha256_file(path),
                "particle_count": int(len(ids)),
                "maximum_speed_m_s": maximum_speed,
                "speed_cap_fraction": speed_cap_fraction,
                "lateral_boundary_penetration_fraction": lateral_boundary_fraction,
                "floor_boundary_penetration_fraction": floor_boundary_fraction,
                "over_rim_fraction": over_rim_fraction,
                "over_rim_particle_count": len(self.over_rim_particle_ids),
                "side_tunnel_particle_count": len(self.side_tunnel_particle_ids),
                "bottom_tunnel_particle_count": len(self.bottom_tunnel_particle_ids),
                "lateral_escape_fraction": lateral_fraction,
                "below_floor_fraction": below_fraction,
                "body_surface_proximity": body_surface_proximity,
            }
        )
        self._write_manifest(complete=False)

    def finalize(self) -> dict:
        source = self.configuration["source"]
        solver_resources = self._solver_resource_report()
        self.metadata["solver_resources"] = solver_resources
        interaction_states = [
            self.body_interaction_states[index]
            for index in sorted(self.body_interaction_states)
        ]
        contacted_body_kinds = sorted(
            {
                state["physics_kind"]
                for state in interaction_states
                if state["maximum_consecutive_visible_surface_proximity_frames"]
                >= self.minimum_consecutive_visible_surface_frames
            }
        )
        missing_body_kinds = sorted(
            set(self.required_body_kinds) - set(contacted_body_kinds)
        )
        valid = bool(
            len(self.samples) == self.frame_count
            and self.maximum_speed_seen <= source["maximum_speed"] + 0.05
            and self.maximum_speed_cap_fraction <= float(
                self.configuration.get("maximum_speed_cap_fraction", 0.005)
            )
            and self.maximum_lateral_escape_fraction <= float(
                self.configuration.get("maximum_lateral_escape_fraction", 0.02)
            )
            and self.maximum_below_floor_fraction <= float(
                self.configuration.get("maximum_below_floor_fraction", 0.002)
            )
            and solver_resources["valid"]
            and not missing_body_kinds
        )
        self._write_manifest(complete=True, valid=valid)
        return {
            **self.metadata,
            "manifest": str(self.manifest_path),
            "manifest_sha256": _sha256_file(self.manifest_path),
            "frames": len(self.samples),
            "maximum_speed_m_s": self.maximum_speed_seen,
            "maximum_speed_cap_fraction": self.maximum_speed_cap_fraction,
            "maximum_lateral_escape_fraction": self.maximum_lateral_escape_fraction,
            "maximum_below_floor_fraction": self.maximum_below_floor_fraction,
            "maximum_over_rim_fraction": self.maximum_over_rim_fraction,
            "over_rim_particle_count": len(self.over_rim_particle_ids),
            "side_tunnel_particle_count": len(self.side_tunnel_particle_ids),
            "bottom_tunnel_particle_count": len(self.bottom_tunnel_particle_ids),
            "maximum_lateral_boundary_penetration_fraction": (
                self.maximum_lateral_boundary_penetration_fraction
            ),
            "maximum_floor_boundary_penetration_fraction": (
                self.maximum_floor_boundary_penetration_fraction
            ),
            "body_interaction": {
                **self.metadata["body_interaction_gate"],
                "bodies": interaction_states,
                "contacted_body_kinds": contacted_body_kinds,
                "missing_required_body_kinds": missing_body_kinds,
                "valid": not missing_body_kinds,
            },
            "valid": valid,
        }
