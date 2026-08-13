"""Capability probe for fixed-topology PhysX PBD particle recycling.

The probe keeps one particle set at a constant size, lets it fall out of a small
visible region, then rewrites those existing particles to their original hidden
spawn slots.  It succeeds only when the rewritten simulation points survive the
next PhysX step and the topology remains unchanged across multiple cycles.
"""

import argparse
import json
import os

import numpy as np


os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=180)
parser.add_argument("--substeps", type=int, default=2)
parser.add_argument("--spacing", type=float, default=0.003)
parser.add_argument("--recycle-y", type=float, default=-0.05)
parser.add_argument("--spawn-y", type=float, default=0.18)
parser.add_argument("--gpu-max-particle-contacts", type=int, default=100_000)
parser.add_argument(
    "--output",
    default=r"Y:\isaacsim_work\output\long_video_probe\particle_state_reset",
)
args = parser.parse_args()

if args.frames <= 0 or args.substeps <= 0 or args.spacing <= 0.0:
    raise ValueError("frames, substeps, and spacing must be positive")
if args.spawn_y <= args.recycle_y:
    raise ValueError("spawn-y must be above recycle-y")

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": 320,
        "height": 180,
    }
)

import carb
import omni.usd
from omni.physx import get_physx_interface, get_physx_simulation_interface
import omni.physx.bindings._physx as physx_settings_bindings
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdUtils, Vt


os.makedirs(args.output, exist_ok=True)
context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
physx_scene.CreateEnableExternalForcesEveryIterationAttr().Set(True)
create_contacts_attr = getattr(physx_scene, "CreateGpuMaxParticleContactsAttr", None)
if create_contacts_attr is None:
    raise RuntimeError("This Isaac Sim build lacks gpuMaxParticleContacts")
create_contacts_attr().Set(args.gpu_max_particle_contacts)
physx_scene.CreateTimeStepsPerSecondAttr().Set(60 * args.substeps)

fluid_rest_offset = 0.5 * args.spacing
particle_contact_offset = fluid_rest_offset / 0.6
particle_system_path = Sdf.Path("/World/ParticleSystem")
particle_system = particleUtils.add_physx_particle_system(
    stage,
    particle_system_path,
    simulation_owner=scene.GetPath(),
    contact_offset=particle_contact_offset + 0.001,
    rest_offset=particle_contact_offset,
    particle_contact_offset=particle_contact_offset,
    solid_rest_offset=particle_contact_offset,
    fluid_rest_offset=fluid_rest_offset,
    enable_ccd=True,
    solver_position_iterations=8,
    max_neighborhood=96,
    neighborhood_scale=1.01,
    max_velocity=5.0,
)
material_path = Sdf.Path("/World/WaterPhysics")
particleUtils.add_pbd_particle_material(
    stage,
    material_path,
    density=1000.0,
    friction=0.05,
    damping=0.02,
    viscosity=0.001,
    vorticity_confinement=0.0,
    surface_tension=0.00704,
    cohesion=0.0704,
    adhesion=0.0,
    cfl_coefficient=1.0,
)
physicsUtils.add_physics_material_to_prim(
    stage, particle_system.GetPrim(), material_path
)

positions, velocities = particleUtils.create_particles_grid(
    Gf.Vec3f(-3.5 * args.spacing, args.spawn_y, -3.5 * args.spacing),
    args.spacing,
    8,
    8,
    8,
)
initial_points = np.asarray(positions, dtype=np.float32)
active_count = len(initial_points)
particles_path = Sdf.Path("/World/WaterParticles")
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
).Set(active_count)
instancer = UsdGeom.PointInstancer(particles_prim)
particle_set = PhysxSchema.PhysxParticleSetAPI(particles_prim)
prototype = UsdGeom.Imageable.Get(
    stage, particles_path.AppendChild("particlePrototype0")
)
if prototype:
    prototype.MakeInvisible()

particleUtils.add_physx_particle_smoothing(
    stage, particle_system_path, enabled=True, strength=0.45
)
particleUtils.add_physx_particle_anisotropy(
    stage,
    particle_system_path,
    enabled=True,
    scale=4.0,
    min=1.0,
    max=2.0,
)
particleUtils.add_physx_particle_isosurface(
    stage,
    particle_system_path,
    enabled=True,
    grid_spacing=max(0.001, 0.5 * args.spacing),
    surface_distance=args.spacing,
    num_mesh_smoothing_passes=2,
    num_mesh_normal_smoothing_passes=4,
    max_vertices=250_000,
    max_triangles=500_000,
    max_subgrids=2048,
)

simulation_points_attr = particle_set.GetSimulationPointsAttr()

settings = carb.settings.get_settings()
settings.set(physx_settings_bindings.SETTING_UPDATE_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)

stage.GetRootLayer().Export(os.path.join(args.output, "probe_scene.usda"))
for _ in range(3):
    simulation_app.update()

simulation = get_physx_simulation_interface()
physx_interface = get_physx_interface()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
simulation.attach_stage(stage_id)
print(
    f"[recycle-probe] attached active={active_count}, frames={args.frames}",
    flush=True,
)

simulation_step = 0
pending_indices = None
pending_targets = None
recycle_cycles = 0
recycled_particles = 0
max_post_step_error = 0.0
max_authored_error = 0.0
last_state_source = "unknown"

try:
    for frame in range(args.frames):
        for _ in range(args.substeps):
            dt = 1.0 / (60.0 * args.substeps)
            simulation.simulate(dt, simulation_step * dt)
            simulation.fetch_results()
            simulation_step += 1
        physx_interface.update_transformations(False, True, True)

        simulation_points = simulation_points_attr.Get()
        state_source = "simulationPoints"
        if simulation_points is None:
            simulation_points = instancer.GetPositionsAttr().Get()
            state_source = "positions"
        if simulation_points is None:
            raise RuntimeError("Particle positions disappeared during the probe")
        last_state_source = state_source
        points = np.asarray(simulation_points, dtype=np.float32).copy()
        if len(points) != active_count:
            raise RuntimeError(
                f"Fixed particle topology changed: {len(points)} != {active_count}"
            )
        velocity_values = instancer.GetVelocitiesAttr().Get()
        if velocity_values is None or len(velocity_values) != active_count:
            raise RuntimeError("Particle velocities disappeared or changed size")
        velocity_array = np.asarray(velocity_values, dtype=np.float32).copy()
        if not np.isfinite(points).all() or not np.isfinite(velocity_array).all():
            raise RuntimeError("Probe produced non-finite particle state")
        if frame % 30 == 0:
            print(
                f"[recycle-probe] frame={frame}, min_y={float(points[:, 1].min()):.6f}, "
                f"max_y={float(points[:, 1].max()):.6f}, "
                f"state_source={state_source}, "
                f"simulation_points_authored={simulation_points_attr.HasAuthoredValue()}",
                flush=True,
            )

        if pending_indices is not None:
            post_step_error = np.linalg.norm(
                points[pending_indices] - pending_targets, axis=1
            )
            max_post_step_error = max(
                max_post_step_error, float(np.max(post_step_error, initial=0.0))
            )
            if np.any(post_step_error > 0.05):
                raise RuntimeError(
                    "Rewritten simulation points were not accepted by the next PhysX "
                    f"step; max error={float(np.max(post_step_error)):.6f} m"
                )
            pending_indices = None
            pending_targets = None

        recycle_indices = np.flatnonzero(points[:, 1] < args.recycle_y)
        if len(recycle_indices) and frame < args.frames - 1:
            target_points = initial_points[recycle_indices].copy()
            target_points[:, 1] += args.spawn_y - float(
                np.min(initial_points[:, 1])
            )
            points[recycle_indices] = target_points
            velocity_array[recycle_indices] = np.asarray(
                [0.0, -0.05, 0.0], dtype=np.float32
            )

            # simulationPoints is the unsmoothed physical state.  The point-instancer
            # arrays are kept in lockstep because PhysX validates equal-sized authored
            # particle arrays and writes smoothed display positions separately.
            with Sdf.ChangeBlock():
                if state_source == "simulationPoints":
                    simulation_points_attr.Set(
                        Vt.Vec3fArray.FromNumpy(points)
                    )
                instancer.GetPositionsAttr().Set(
                    Vt.Vec3fArray.FromNumpy(points)
                )
                instancer.GetVelocitiesAttr().Set(
                    Vt.Vec3fArray.FromNumpy(velocity_array)
                )
            simulation.flush_changes()

            authored_values = (
                simulation_points_attr.Get()
                if state_source == "simulationPoints"
                else instancer.GetPositionsAttr().Get()
            )
            authored_points = np.asarray(authored_values, dtype=np.float32)
            authored_error = np.linalg.norm(
                authored_points[recycle_indices] - target_points, axis=1
            )
            max_authored_error = max(
                max_authored_error, float(np.max(authored_error, initial=0.0))
            )
            if np.any(authored_error > 1.0e-6):
                raise RuntimeError(
                    "Authored simulationPoints did not retain the recycle targets"
                )

            pending_indices = recycle_indices
            pending_targets = target_points
            recycle_cycles += 1
            recycled_particles += len(recycle_indices)
            print(
                f"[recycle-probe] frame={frame}, cycle={recycle_cycles}, "
                f"recycled={len(recycle_indices)}, active={active_count}",
                flush=True,
            )

        simulation_app.update()

    if pending_indices is not None:
        raise RuntimeError("Probe ended before verifying the final recycle write")
    if recycle_cycles < 2:
        raise RuntimeError(
            f"Probe completed only {recycle_cycles} recycle cycles; expected at least 2"
        )

    completion = {
        "active_particles": active_count,
        "fixed_topology": True,
        "frames_completed": args.frames,
        "gpu_max_particle_contacts": args.gpu_max_particle_contacts,
        "max_authored_error_m": max_authored_error,
        "max_post_step_error_m": max_post_step_error,
        "recycle_cycles": recycle_cycles,
        "recycled_particles": recycled_particles,
        "state_source": last_state_source,
        "valid": True,
    }
    with open(
        os.path.join(args.output, "probe_complete.json"),
        "w",
        encoding="utf-8",
    ) as completion_file:
        json.dump(completion, completion_file, indent=2, sort_keys=True)
    print(json.dumps(completion, sort_keys=True), flush=True)
except Exception as exc:
    import traceback

    traceback.print_exc()
    with open(
        os.path.join(args.output, "probe_failed.json"),
        "w",
        encoding="utf-8",
    ) as failure_file:
        json.dump(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "frames_requested": args.frames,
                "recycle_cycles": recycle_cycles,
                "valid": False,
            },
            failure_file,
            indent=2,
            sort_keys=True,
        )
    raise
finally:
    simulation.detach_stage()
    simulation_app.close()
