"""Low-density end-to-end gate for the PhysX secondary-particle bridge.

The gate consumes the same BGEO birth-event contract as FoamGenerator, appends
births to one shared non-fluid PhysX particle pool, exports native PhysX state,
compacts expired ids, and verifies collisions against both a static wall and a
moving dynamic rigid body. It uses at most fourteen particles.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import numpy as np

from foam_bgeo_io import read_birth_events, write_birth_events, write_motion_state


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--physics-fps", type=int, default=120)
parser.add_argument("--steps", type=int, default=90)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to reuse output directory: {args.output}")
if args.physics_fps != 120 or args.steps != 90:
    raise ValueError("This calibrated low-load gate requires 120 FPS and 90 steps")
args.output.mkdir(parents=True, exist_ok=False)
birth_directory = args.output / "births"
motion_directory = args.output / "external_motion"
birth_directory.mkdir()
motion_directory.mkdir()
report_path = args.output / "gate_report.json"
obsolete_path = args.output / "OBSOLETE.json"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


static_ids = np.arange(0, 6, dtype=np.int32)
moving_ids = np.arange(6, 12, dtype=np.int32)
initial_ids = np.concatenate((static_ids, moving_ids))
static_positions = np.column_stack(
    (
        np.full(6, -0.35, dtype=np.float32),
        np.full(6, 0.055, dtype=np.float32),
        np.linspace(-0.36, -0.10, 6, dtype=np.float32),
    )
)
static_velocities = np.tile(np.asarray((1.5, 0.0, 0.0), dtype=np.float32), (6, 1))
moving_positions = np.column_stack(
    (
        np.full(6, -0.02, dtype=np.float32),
        np.full(6, 0.055, dtype=np.float32),
        np.linspace(0.12, 0.36, 6, dtype=np.float32),
    )
)
moving_velocities = np.zeros((6, 3), dtype=np.float32)
initial_positions = np.concatenate((static_positions, moving_positions))
initial_velocities = np.concatenate((static_velocities, moving_velocities))
write_birth_events(
    birth_directory / "birth_000000.bgeo",
    initial_positions,
    initial_velocities,
    initial_ids,
    np.full(12, 1.0, dtype=np.float32),
    0,
    np.arange(12, dtype=np.int32),
)

transient_ids = np.asarray((12, 13), dtype=np.int32)
write_birth_events(
    birth_directory / "birth_000020.bgeo",
    np.asarray(((-0.30, 0.18, -0.18), (-0.30, 0.24, -0.26)), dtype=np.float32),
    np.asarray(((1.2, 0.0, 0.0), (1.2, 0.0, 0.0)), dtype=np.float32),
    transient_ids,
    np.full(2, 0.05, dtype=np.float32),
    20,
    np.asarray((100, 101), dtype=np.int32),
)

report: dict[str, object] = {
    "schema": "physx-secondary-bridge-collision-gate/v1",
    "started_utc": utc_now(),
    "valid": False,
    "physics_fps": args.physics_fps,
    "steps": args.steps,
    "max_particles": 64,
    "maximum_expected_active_particles": 14,
}
atomic_json(report_path, report)

simulation_app = None
simulation = None

try:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "renderer": "MinimalRendering",
            "disable_viewport_updates": True,
            "limit_cpu_threads": 2,
            "multi_gpu": False,
            "max_gpu_count": 1,
        },
        experience=r"Y:\isaacsim\apps\isaacsim.exp.base.python.kit",
    )

    import carb
    import omni.physx.bindings._physx as physx_bindings
    import omni.usd
    from omni.physx import get_physx_simulation_interface
    from omni.physx.scripts import particleUtils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdUtils, Vt

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    def add_cube(path: str, translation: tuple[float, float, float], scale: tuple[float, float, float]):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr().Set(2.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        return cube

    add_cube("/World/Ground", (0.0, -0.05, 0.0), (0.8, 0.05, 0.6))
    add_cube("/World/StaticWall", (0.20, 0.20, -0.23), (0.03, 0.20, 0.20))
    pusher = add_cube("/World/DynamicPusher", (-0.32, 0.06, 0.24), (0.04, 0.04, 0.18))
    pusher_rigid = UsdPhysics.RigidBodyAPI.Apply(pusher.GetPrim())
    pusher_rigid.CreateRigidBodyEnabledAttr().Set(True)
    pusher_rigid.CreateKinematicEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(pusher.GetPrim()).CreateMassAttr().Set(5.0)
    pusher_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(pusher.GetPrim())
    pusher_physx.CreateDisableGravityAttr().Set(True)
    pusher_physx.CreateLinearDampingAttr().Set(0.0)
    pusher_physx.CreateAngularDampingAttr().Set(0.0)
    pusher_translate_op = UsdGeom.Xformable(pusher.GetPrim()).GetOrderedXformOps()[0]

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
    physx_scene.CreateGpuMaxParticleContactsAttr().Set(512)
    physx_scene.CreateTimeStepsPerSecondAttr().Set(args.physics_fps)

    particle_system_path = Sdf.Path("/World/SecondaryParticleSystem")
    particleUtils.add_physx_particle_system(
        stage,
        particle_system_path,
        simulation_owner=scene.GetPath(),
        contact_offset=0.022,
        rest_offset=0.018,
        particle_contact_offset=0.020,
        solid_rest_offset=0.015,
        fluid_rest_offset=0.012,
        enable_ccd=True,
        solver_position_iterations=6,
        max_neighborhood=32,
        neighborhood_scale=1.01,
        max_velocity=20.0,
        global_self_collision_enabled=False,
        non_particle_collision_enabled=True,
    )
    particles_path = Sdf.Path("/World/SecondaryParticles")
    particles_prim = particleUtils.add_physx_particleset_pointinstancer(
        stage,
        particles_path,
        Vt.Vec3fArray(),
        Vt.Vec3fArray(),
        particle_system_path,
        self_collision=False,
        fluid=False,
        particle_group=1,
        particle_mass=0.001,
        density=0.0,
    )
    particles_prim.CreateAttribute("physxParticle:maxParticles", Sdf.ValueTypeNames.Int).Set(64)
    instancer = UsdGeom.PointInstancer.Get(stage, particles_path)

    settings = carb.settings.get_settings()
    settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
    settings.set(physx_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)
    settings.set(physx_bindings.SETTING_SUPPRESS_READBACK, False)

    stable_ids = np.empty(0, dtype=np.int64)
    remaining_lifetimes = np.empty(0, dtype=np.float64)
    birth_frames = np.empty(0, dtype=np.int64)

    def readback() -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(instancer.GetPositionsAttr().Get(), dtype=np.float32).reshape((-1, 3))
        velocities = np.asarray(instancer.GetVelocitiesAttr().Get(), dtype=np.float32).reshape((-1, 3))
        if positions.shape != velocities.shape or len(positions) != len(stable_ids):
            raise RuntimeError("PhysX state and stable-id arrays are not aligned")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError("PhysX returned non-finite state")
        return positions, velocities

    def author_state(positions: np.ndarray, velocities: np.ndarray) -> None:
        if positions.shape != velocities.shape or len(positions) > 64:
            raise ValueError("Authored PhysX state is invalid or exceeds maxParticles")
        instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(positions, dtype=np.float32)))
        instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(velocities, dtype=np.float32)))
        instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(positions)))

    def inject_birth_frame(frame: int) -> list[int]:
        global stable_ids, remaining_lifetimes, birth_frames
        path = birth_directory / f"birth_{frame:06d}.bgeo"
        if not path.exists():
            return []
        birth = read_birth_events(path)
        if np.any(birth["birth_frame"] != frame):
            raise ValueError(f"{path}: birth_frame does not match filename")
        born_ids = birth["id"].astype(np.int64)
        if np.intersect1d(stable_ids, born_ids).size:
            raise ValueError(f"{path}: birth stable id already exists in the pool")
        positions, velocities = readback()
        author_state(
            np.concatenate((positions, birth["position"].astype(np.float32))),
            np.concatenate((velocities, birth["velocity"].astype(np.float32))),
        )
        stable_ids = np.concatenate((stable_ids, born_ids))
        remaining_lifetimes = np.concatenate(
            (remaining_lifetimes, birth["remaining_lifetime"].astype(np.float64))
        )
        birth_frames = np.concatenate((birth_frames, birth["birth_frame"].astype(np.int64)))
        return born_ids.tolist()

    # Author frame-zero births before stage attachment; later births exercise runtime append.
    injected_zero = inject_birth_frame(0)
    stage_path = args.output / "physx_secondary_bridge_collision_gate.usda"
    stage.GetRootLayer().Export(str(stage_path))
    simulation = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    simulation.attach_stage(stage_id)

    dt = 1.0 / args.physics_fps
    snapshots: dict[str, dict[str, object]] = {}
    events: list[dict[str, object]] = [{"frame": 0, "event": "birth", "stable_ids": injected_zero}]
    maximum_active = len(stable_ids)
    for frame in range(1, args.steps + 1):
        # Drive a true PhysX kinematic rigid body. The particle positions are
        # never authored here; contact response remains PhysX-owned.
        pusher_translate_op.Set(
            Gf.Vec3d(-0.32 + 0.9 * frame * dt, 0.06, 0.24)
        )
        simulation.simulate(dt, (frame - 1) * dt)
        simulation.fetch_results()
        positions, velocities = readback()
        write_motion_state(
            motion_directory / f"external_{frame:06d}.bgeo",
            positions,
            velocities,
            stable_ids.astype(np.int32),
        )
        if frame in (1, 20, 21, 26, 27, 28, 50, args.steps):
            snapshots[f"frame_{frame:06d}"] = {
                "count": len(stable_ids),
                "stable_ids": stable_ids.tolist(),
                "positions": positions.tolist(),
                "velocities": velocities.tolist(),
                "kinematic_pusher_translation": list(pusher_translate_op.Get()),
            }

        # FoamGenerator chronological lifetime is evaluated after this frame's
        # state is consumed. Particles that expire here disappear next frame.
        remaining_lifetimes -= dt
        keep = remaining_lifetimes > 0.0
        if not np.all(keep):
            removed = stable_ids[~keep].tolist()
            author_state(positions[keep], velocities[keep])
            stable_ids = stable_ids[keep]
            remaining_lifetimes = remaining_lifetimes[keep]
            birth_frames = birth_frames[keep]
            events.append({"frame": frame, "event": "compact_after_output", "stable_ids": removed})

        if frame < args.steps:
            born = inject_birth_frame(frame)
            if born:
                events.append({"frame": frame, "event": "birth_after_output", "stable_ids": born})
        maximum_active = max(maximum_active, len(stable_ids))

    final_positions, final_velocities = readback()
    frame_50 = snapshots["frame_000050"]
    frame_50_ids = np.asarray(frame_50["stable_ids"], dtype=np.int64)
    frame_50_positions = np.asarray(frame_50["positions"], dtype=np.float64)
    frame_50_velocities = np.asarray(frame_50["velocities"], dtype=np.float64)
    static_mask = np.isin(frame_50_ids, static_ids)
    moving_mask = np.isin(frame_50_ids, moving_ids)
    static_maximum_x = float(np.max(frame_50_positions[static_mask, 0]))
    static_minimum_vx = float(np.min(frame_50_velocities[static_mask, 0]))
    moving_maximum_x = float(np.max(frame_50_positions[moving_mask, 0]))
    moving_displacement = moving_maximum_x - float(np.max(moving_positions[:, 0]))
    final_id_set = stable_ids.tolist()
    transient_absent_at_28 = not any(
        stable_id in snapshots["frame_000028"]["stable_ids"] for stable_id in transient_ids.tolist()
    )

    checks = {
        "frame_zero_birth_contract_consumed": injected_zero == initial_ids.tolist(),
        "runtime_append_reached_maximum_14": maximum_active == 14,
        "short_lifetime_particles_compacted": transient_absent_at_28,
        "stable_ids_survived_compaction": final_id_set == initial_ids.tolist(),
        "static_wall_not_crossed": static_maximum_x <= 0.19,
        "static_wall_changed_horizontal_velocity": static_minimum_vx < 1.0,
        "moving_kinematic_rigid_body_displaced_particles": moving_displacement >= 0.04,
        "all_final_state_finite": bool(
            np.all(np.isfinite(final_positions)) and np.all(np.isfinite(final_velocities))
        ),
        "native_physx_state_exported": len(list(motion_directory.glob("external_*.bgeo"))) == args.steps,
    }
    report.update(
        {
            "completed_utc": utc_now(),
            "valid": all(checks.values()),
            "checks": checks,
            "events": events,
            "snapshots": snapshots,
            "maximum_active_particles": maximum_active,
            "final_stable_ids": final_id_set,
            "static_wall_maximum_particle_x_at_frame_50": static_maximum_x,
            "static_wall_minimum_particle_vx_at_frame_50": static_minimum_vx,
            "moving_rigid_body_maximum_particle_x_at_frame_50": moving_maximum_x,
            "moving_rigid_body_particle_displacement_x_at_frame_50": moving_displacement,
            "position_authority": "PhysX PointInstancer native readback",
            "velocity_authority": "PhysX PointInstancer velocities native readback",
            "finite_difference_velocity_used": False,
        }
    )
    atomic_json(report_path, report)
    if not report["valid"]:
        atomic_json(
            obsolete_path,
            {
                "obsolete": True,
                "safe_for_production": False,
                "reason": "One or more PhysX bridge collision checks failed.",
                "checks": checks,
            },
        )
        raise RuntimeError("PhysX secondary bridge collision gate failed")
except Exception as exc:
    report.update(
        {
            "completed_utc": utc_now(),
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    )
    atomic_json(report_path, report)
    if not obsolete_path.exists():
        atomic_json(
            obsolete_path,
            {
                "obsolete": True,
                "safe_for_production": False,
                "reason": "PhysX secondary bridge collision gate raised an exception.",
                "error": report["error"],
            },
        )
    raise
finally:
    if simulation is not None:
        try:
            simulation.detach_stage()
        except Exception:
            pass
    if simulation_app is not None:
        simulation_app.close()
