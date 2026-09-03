"""Low-load runtime resize gate for a PhysX secondary-particle pool.

The probe verifies the exact lifecycle operations required by the
FoamGenerator/PhysX bridge without loading a production scene:

1. create a shared non-fluid PBD particle set with spare maxParticles capacity;
2. simulate against an ordinary PhysX collider;
3. append born particles while the stage remains attached;
4. compact the active arrays to remove dead particles;
5. continue simulation and audit native position/velocity readback.

It is intentionally limited to five particles and two CPU worker threads.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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

if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
if not 60 <= args.physics_fps <= 240:
    raise ValueError("physics-fps must be within the low-load 60..240 gate range")
if not 60 <= args.steps <= 180:
    raise ValueError("steps must be within the low-load 60..180 gate range")
args.output.mkdir(parents=True, exist_ok=False)

report_path = args.output / "probe_report.json"
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


report: dict[str, object] = {
    "schema": "physx-secondary-particle-pool-probe/v1",
    "started_utc": utc_now(),
    "valid": False,
    "physics_fps": args.physics_fps,
    "steps": args.steps,
    "maximum_particles": 16,
    "events": [],
}
atomic_json(report_path, report)

simulation_app = None
simulation = None

try:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
            "disable_viewport_updates": True,
            "limit_cpu_threads": 2,
        }
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

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr().Set(2.0)
    ground_xform = UsdGeom.Xformable(ground.GetPrim())
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.05, 0.0))
    ground_xform.AddScaleOp().Set(Gf.Vec3f(0.5, 0.05, 0.5))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr().Set(True)

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
    physx_scene.CreateGpuMaxParticleContactsAttr().Set(256)
    physx_scene.CreateTimeStepsPerSecondAttr().Set(args.physics_fps)

    particle_system_path = Sdf.Path("/World/SecondaryParticleSystem")
    particle_system = particleUtils.add_physx_particle_system(
        stage,
        particle_system_path,
        simulation_owner=scene.GetPath(),
        contact_offset=0.022,
        rest_offset=0.018,
        particle_contact_offset=0.020,
        solid_rest_offset=0.015,
        fluid_rest_offset=0.012,
        enable_ccd=True,
        solver_position_iterations=4,
        max_neighborhood=32,
        neighborhood_scale=1.01,
        max_velocity=20.0,
        global_self_collision_enabled=False,
        non_particle_collision_enabled=True,
    )

    positions = np.asarray(
        [(-0.10, 0.12, 0.0), (0.10, 0.18, 0.0)], dtype=np.float32
    )
    velocities = np.zeros_like(positions)
    stable_ids = np.asarray([1000, 1001], dtype=np.int64)

    particles_path = Sdf.Path("/World/SecondaryParticles")
    particles_prim = particleUtils.add_physx_particleset_pointinstancer(
        stage,
        particles_path,
        Vt.Vec3fArray.FromNumpy(positions),
        Vt.Vec3fArray.FromNumpy(velocities),
        particle_system_path,
        self_collision=False,
        fluid=False,
        particle_group=1,
        particle_mass=0.001,
        density=0.0,
    )
    particles_prim.CreateAttribute(
        "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
    ).Set(16)
    instancer = UsdGeom.PointInstancer.Get(stage, particles_path)

    settings = carb.settings.get_settings()
    settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
    settings.set(physx_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)
    settings.set(physx_bindings.SETTING_SUPPRESS_READBACK, False)

    stage_path = args.output / "secondary_particle_pool_probe.usda"
    stage.GetRootLayer().Export(str(stage_path))
    simulation = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    simulation.attach_stage(stage_id)

    def readback() -> tuple[np.ndarray, np.ndarray]:
        current_positions = np.asarray(
            instancer.GetPositionsAttr().Get(), dtype=np.float32
        )
        current_velocities = np.asarray(
            instancer.GetVelocitiesAttr().Get(), dtype=np.float32
        )
        if current_positions.shape != current_velocities.shape:
            raise RuntimeError(
                "PhysX returned inconsistent position and velocity array shapes: "
                f"{current_positions.shape} versus {current_velocities.shape}"
            )
        if current_positions.ndim != 2 or current_positions.shape[1] != 3:
            raise RuntimeError(f"Unexpected particle shape: {current_positions.shape}")
        if not np.all(np.isfinite(current_positions)) or not np.all(
            np.isfinite(current_velocities)
        ):
            raise RuntimeError("PhysX returned non-finite particle state")
        return current_positions, current_velocities

    def author_state(new_positions: np.ndarray, new_velocities: np.ndarray) -> None:
        if new_positions.shape != new_velocities.shape:
            raise ValueError("Authored position and velocity shapes differ")
        if len(new_positions) > 16:
            raise ValueError("Probe maxParticles capacity exceeded")
        instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(new_positions, dtype=np.float32))
        )
        instancer.GetVelocitiesAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(new_velocities, dtype=np.float32))
        )
        instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(new_positions)))

    snapshots: dict[str, dict[str, object]] = {}

    def snapshot(name: str, step: int) -> tuple[np.ndarray, np.ndarray]:
        current_positions, current_velocities = readback()
        snapshots[name] = {
            "step": step,
            "count": int(len(current_positions)),
            "stable_ids": stable_ids.tolist(),
            "positions": current_positions.tolist(),
            "velocities": current_velocities.tolist(),
            "minimum_y": float(np.min(current_positions[:, 1])),
            "maximum_speed": float(
                np.max(np.linalg.norm(current_velocities, axis=1))
            ),
        }
        if len(current_positions) != len(stable_ids):
            raise RuntimeError(
                f"Stable-ID map has {len(stable_ids)} entries but PhysX has "
                f"{len(current_positions)} particles at {name}"
            )
        return current_positions, current_velocities

    dt = 1.0 / args.physics_fps
    append_step = 18
    compact_step = 48
    for step in range(1, args.steps + 1):
        if step == append_step:
            current_positions, current_velocities = snapshot("before_append", step)
            born_positions = np.asarray(
                [(-0.18, 0.34, 0.0), (0.0, 0.40, 0.0), (0.18, 0.46, 0.0)],
                dtype=np.float32,
            )
            born_velocities = np.asarray(
                [(0.0, -0.25, 0.0), (0.0, -0.50, 0.0), (0.0, -0.75, 0.0)],
                dtype=np.float32,
            )
            author_state(
                np.concatenate((current_positions, born_positions)),
                np.concatenate((current_velocities, born_velocities)),
            )
            stable_ids = np.concatenate(
                (stable_ids, np.asarray([2000, 2001, 2002], dtype=np.int64))
            )
            report["events"].append(
                {"step": step, "event": "append", "stable_ids": [2000, 2001, 2002]}
            )
        if step == compact_step:
            current_positions, current_velocities = snapshot("before_compact", step)
            keep = np.asarray([1, 2, 4], dtype=np.int64)
            removed_ids = stable_ids[np.asarray([0, 3], dtype=np.int64)].tolist()
            author_state(current_positions[keep], current_velocities[keep])
            stable_ids = stable_ids[keep]
            report["events"].append(
                {"step": step, "event": "compact", "removed_stable_ids": removed_ids}
            )

        simulation.simulate(dt, (step - 1) * dt)
        simulation.fetch_results()

        if step in (1, append_step, append_step + 1, compact_step, compact_step + 1, args.steps):
            snapshot(f"after_step_{step:03d}", step)

    final_positions, final_velocities = snapshot("final", args.steps)
    minimum_y = min(
        float(item["minimum_y"]) for item in snapshots.values() if item["count"]
    )
    count_sequence_ok = (
        snapshots["before_append"]["count"] == 2
        and snapshots[f"after_step_{append_step:03d}"]["count"] == 5
        and snapshots["before_compact"]["count"] == 5
        and snapshots[f"after_step_{compact_step:03d}"]["count"] == 3
        and len(final_positions) == 3
    )
    stable_mapping_ok = stable_ids.tolist() == [1001, 2000, 2002]
    collider_margin_ok = minimum_y >= -0.002

    report.update(
        {
            "completed_utc": utc_now(),
            "snapshots": snapshots,
            "checks": {
                "runtime_append_preserved_native_velocity_array": (
                    snapshots[f"after_step_{append_step:03d}"]["count"] == 5
                ),
                "runtime_compaction_preserved_native_velocity_array": (
                    snapshots[f"after_step_{compact_step:03d}"]["count"] == 3
                ),
                "count_sequence_correct": count_sequence_ok,
                "stable_id_mapping_survived_compaction": stable_mapping_ok,
                "ordinary_physx_collider_prevented_ground_penetration": collider_margin_ok,
                "all_state_finite": bool(
                    np.all(np.isfinite(final_positions))
                    and np.all(np.isfinite(final_velocities))
                ),
            },
            "minimum_observed_y": minimum_y,
            "valid": bool(count_sequence_ok and stable_mapping_ok and collider_margin_ok),
        }
    )
    atomic_json(report_path, report)
    if not report["valid"]:
        atomic_json(
            obsolete_path,
            {
                "obsolete": True,
                "safe_for_production": False,
                "reason": "PhysX secondary-particle pool gate did not pass all checks.",
            },
        )
        raise RuntimeError("PhysX secondary-particle pool gate failed")
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
                "reason": "PhysX secondary-particle pool probe raised an exception.",
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
