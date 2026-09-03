"""Replay FoamGenerator births through an existing PhysX liquid scene.

The bridge deliberately keeps responsibilities narrow:

* cached primary positions and native velocities are authoritative;
* FoamGenerator birth events own stable ids and initial lifetimes;
* PhysX owns all secondary motion, liquid interaction, and scene collisions;
* only native PointInstancer positions and velocities are exported.

Frame semantics match FoamGenerator's chronological contract. Births authored
for source frame F enter after output F and first appear in external state
F+1. Therefore a run from 48 through 54 exports frames 49 through 54.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any

import numpy as np

from foam_bgeo_io import read_birth_events, write_motion_state


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--birth-pattern", required=True)
    parser.add_argument("--primary-source-pattern", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--source-fps", type=int, default=120)
    parser.add_argument("--physics-fps", type=int, default=240)
    parser.add_argument("--max-secondary-particles", type=int, default=2048)
    parser.add_argument(
        "--secondary-dynamics",
        choices=("physx-ballistic", "foamgenerator-warp"),
        default="physx-ballistic",
        help=(
            "Optional type-aware dynamics. foamgenerator-warp reproduces the "
            "FoamGenerator neighborhood model on CUDA and applies its velocity "
            "coupling before PhysX integration."
        ),
    )
    parser.add_argument("--particle-radius", type=float, default=0.004)
    parser.add_argument("--foamgenerator-buoyancy", type=float, default=2.0)
    parser.add_argument("--foamgenerator-drag", type=float, default=0.8)
    parser.add_argument("--warp-device", default="cuda:0")
    parser.add_argument("--physics-scene-path", default="/World/PhysicsScene")
    parser.add_argument("--particle-system-path", default="/World/ParticleSystem")
    parser.add_argument("--primary-particles-path", default="/World/WaterParticles")
    parser.add_argument("--kinematic-prim-path", default="/World/DropSphere")
    parser.add_argument(
        "--strip-prim",
        action="append",
        default=[],
        help="Prim to remove before simulation; repeat for multiple render-only prims.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs and USD contracts without starting SimulationApp.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def frame_path(pattern: str, frame: int) -> Path:
    if "{frame" in pattern:
        return Path(pattern.format(frame=frame))
    match = re.search(r"(#+)", pattern)
    if match:
        width = len(match.group(1))
        return Path(pattern[: match.start()] + f"{frame:0{width}d}" + pattern[match.end() :])
    if "%" in pattern:
        try:
            return Path(pattern % frame)
        except (TypeError, ValueError):
            pass
    raise ValueError(
        "Frame pattern must contain {frame}, a run of # characters, or a printf integer token: "
        + pattern
    )


def require_array(
    path: Path,
    archive: Any,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"{path}: missing array {name}")
    value = np.asarray(archive[name])
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(
            f"{path}: {name} must be {dtype} {shape}, got {value.dtype} {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{path}: {name} contains non-finite values")
    return value


def inspect_npz(path: Path, expected_count: int | None) -> tuple[int, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if "positions" not in archive:
            raise ValueError(f"{path}: missing positions")
        positions = np.asarray(archive["positions"])
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"{path}: positions must have shape (N, 3)")
        count = len(positions)
        if expected_count is not None and count != expected_count:
            raise ValueError(f"{path}: particle count changed: {count} != {expected_count}")
        require_array(path, archive, "positions", (count, 3), np.dtype("float32"))
        require_array(path, archive, "velocities", (count, 3), np.dtype("float32"))
        require_array(path, archive, "sphere_transform", (4, 4), np.dtype("float64"))
        require_array(path, archive, "sphere_linear_velocity", (3,), np.dtype("float32"))
        require_array(path, archive, "sphere_angular_velocity", (3,), np.dtype("float32"))
    return count, {"file": str(path.resolve()), "bytes": path.stat().st_size}


def inspect_usda_text_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect an ASCII USDA structurally without importing Kit or pxr."""
    if args.scene_usd.suffix.lower() != ".usda":
        raise RuntimeError("Text-only USD preflight supports .usda scenes only")

    prims: dict[str, dict[str, Any]] = {}
    stack: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    brace_depth = 0
    declaration = re.compile(
        r'^\s*(?:def|over|class)\s+(?:(\w+)\s+)?"([^"]+)"'
    )
    relationship = re.compile(
        r'rel\s+physxParticle:particleSystem\s*=\s*<([^>]+)>'
    )
    with args.scene_usd.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            match = declaration.match(line)
            if match:
                pending = {
                    "name": match.group(2),
                    "type": match.group(1) or "",
                    "api_schemas": [],
                    "declared_line": line_number,
                }
            if pending is not None and "apiSchemas" in line:
                pending["api_schemas"].extend(re.findall(r'"([^"]+)"', line))

            opens = line.count("{")
            closes = line.count("}")
            if pending is not None and opens:
                parent_path = stack[-1]["path"] if stack else ""
                pending["path"] = parent_path + "/" + pending["name"]
                pending["open_depth"] = brace_depth + 1
                pending["particle_system_target"] = None
                pending["kinematic_enabled"] = None
                prims[pending["path"]] = pending
                stack.append(pending)
                pending = None

            if stack:
                current = stack[-1]
                relation_match = relationship.search(line)
                if relation_match:
                    current["particle_system_target"] = relation_match.group(1)
                if "physics:kinematicEnabled" in line:
                    current["kinematic_enabled"] = bool(
                        re.search(r"physics:kinematicEnabled\s*=\s*(?:1|true)", line)
                    )

            brace_depth += opens - closes
            while stack and brace_depth < stack[-1]["open_depth"]:
                stack.pop()

    required = {
        "physics_scene": args.physics_scene_path,
        "particle_system": args.particle_system_path,
        "primary_particles": args.primary_particles_path,
        "kinematic_prim": args.kinematic_prim_path,
    }
    for role, prim_path in required.items():
        if prim_path not in prims:
            raise ValueError(f"Scene is missing {role}: {prim_path}")
    expected_types = {
        args.physics_scene_path: "PhysicsScene",
        args.particle_system_path: "PhysxParticleSystem",
        args.primary_particles_path: "PointInstancer",
    }
    for prim_path, expected_type in expected_types.items():
        if prims[prim_path]["type"] != expected_type:
            raise ValueError(
                f"{prim_path} must be {expected_type}, got {prims[prim_path]['type']}"
            )
    primary = prims[args.primary_particles_path]
    if "PhysxParticleSetAPI" not in primary["api_schemas"]:
        raise ValueError(f"{args.primary_particles_path} has no PhysxParticleSetAPI")
    if primary["particle_system_target"] != args.particle_system_path:
        raise ValueError(
            f"Primary particle-system relationship is {primary['particle_system_target']}, "
            f"expected {args.particle_system_path}"
        )
    kinematic = prims[args.kinematic_prim_path]
    if "PhysicsRigidBodyAPI" not in kinematic["api_schemas"]:
        raise ValueError(f"{args.kinematic_prim_path} has no PhysicsRigidBodyAPI")

    collision_prims = [
        {
            "path": path,
            "type": record["type"],
            "rigid_body": "PhysicsRigidBodyAPI" in record["api_schemas"],
        }
        for path, record in prims.items()
        if "PhysicsCollisionAPI" in record["api_schemas"]
    ]
    if not collision_prims:
        raise ValueError("Scene has no PhysicsCollisionAPI prims")
    return {
        "inspection_backend": "streaming USDA text parser",
        "required_prims": required,
        "collision_prims": collision_prims,
        "collision_prim_count": len(collision_prims),
        "primary_particle_system_relationship": [primary["particle_system_target"]],
    }


def inspect_usd_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Use pxr when available, otherwise retain a no-Kit USDA path."""
    try:
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    except ImportError:
        return inspect_usda_text_contract(args)

    stage = Usd.Stage.Open(str(args.scene_usd.resolve()), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {args.scene_usd}")

    required = {
        "physics_scene": args.physics_scene_path,
        "particle_system": args.particle_system_path,
        "primary_particles": args.primary_particles_path,
        "kinematic_prim": args.kinematic_prim_path,
    }
    for role, prim_path in required.items():
        if not stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"Scene is missing {role}: {prim_path}")

    scene_prim = stage.GetPrimAtPath(args.physics_scene_path)
    system_prim = stage.GetPrimAtPath(args.particle_system_path)
    primary_prim = stage.GetPrimAtPath(args.primary_particles_path)
    kinematic_prim = stage.GetPrimAtPath(args.kinematic_prim_path)
    if not scene_prim.IsA(UsdPhysics.Scene):
        raise ValueError(f"{args.physics_scene_path} is not a PhysicsScene")
    if not system_prim.IsA(PhysxSchema.PhysxParticleSystem):
        raise ValueError(f"{args.particle_system_path} is not a PhysxParticleSystem")
    if not primary_prim.IsA(UsdGeom.PointInstancer):
        raise ValueError(f"{args.primary_particles_path} is not a PointInstancer")
    if not primary_prim.HasAPI(PhysxSchema.PhysxParticleSetAPI):
        raise ValueError(f"{args.primary_particles_path} has no PhysxParticleSetAPI")
    relation = primary_prim.GetRelationship("physxParticle:particleSystem").GetTargets()
    if [str(path) for path in relation] != [args.particle_system_path]:
        raise ValueError(
            f"Primary particle-system relationship is {relation}, expected {args.particle_system_path}"
        )
    if not kinematic_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"{args.kinematic_prim_path} is not a rigid body")

    collision_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                }
            )
    if not collision_prims:
        raise ValueError("Scene has no PhysicsCollisionAPI prims")
    return {
        "inspection_backend": "pxr Usd.Stage.LoadNone",
        "required_prims": required,
        "collision_prims": collision_prims,
        "collision_prim_count": len(collision_prims),
        "primary_particle_system_relationship": [str(path) for path in relation],
    }


def preflight(args: argparse.Namespace, inspect_usd: bool) -> dict[str, Any]:
    if args.start_frame < 0 or args.end_frame <= args.start_frame:
        raise ValueError("Frame range must satisfy 0 <= start-frame < end-frame")
    if args.source_fps <= 0 or args.physics_fps <= 0:
        raise ValueError("FPS values must be positive")
    if args.physics_fps % args.source_fps != 0:
        raise ValueError("physics-fps must be an integer multiple of source-fps")
    if args.max_secondary_particles <= 0:
        raise ValueError("max-secondary-particles must be positive")
    if args.particle_radius <= 0.0:
        raise ValueError("particle-radius must be positive")
    if args.foamgenerator_buoyancy < 0.0:
        raise ValueError("foamgenerator-buoyancy must be non-negative")
    if not 0.0 <= args.foamgenerator_drag <= 1.0:
        raise ValueError("foamgenerator-drag must be in [0, 1]")
    if args.secondary_dynamics == "foamgenerator-warp" and not args.warp_device.startswith(
        "cuda:"
    ):
        raise ValueError("foamgenerator-warp requires a CUDA Warp device")
    if not args.scene_usd.is_file():
        raise FileNotFoundError(args.scene_usd)

    primary_count: int | None = None
    source_files = []
    for frame in range(args.start_frame, args.end_frame + 1):
        path = frame_path(args.primary_source_pattern, frame)
        if not path.is_file():
            raise FileNotFoundError(path)
        primary_count, record = inspect_npz(path, primary_count)
        record["frame"] = frame
        source_files.append(record)

    all_ids: list[np.ndarray] = []
    birth_files = []
    running_count = 0
    conservative_maximum_active = 0
    for frame in range(args.start_frame, args.end_frame + 1):
        path = frame_path(args.birth_pattern, frame)
        if not path.exists():
            birth_files.append({"frame": frame, "file": str(path.resolve()), "particles": 0})
            continue
        birth = read_birth_events(path)
        if np.any(birth["birth_frame"] != frame):
            raise ValueError(f"{path}: birth_frame does not match filename frame {frame}")
        ids = birth["id"].astype(np.int64, copy=False)
        if len(ids):
            all_ids.append(ids.copy())
        running_count += len(ids)
        conservative_maximum_active = max(conservative_maximum_active, running_count)
        birth_files.append(
            {
                "frame": frame,
                "file": str(path.resolve()),
                "particles": len(ids),
                "id_minimum": int(ids.min()) if len(ids) else None,
                "id_maximum": int(ids.max()) if len(ids) else None,
            }
        )
    concatenated = np.concatenate(all_ids) if all_ids else np.empty(0, dtype=np.int64)
    if len(np.unique(concatenated)) != len(concatenated):
        raise ValueError("Birth-event stable ids are duplicated across frames")
    if len(concatenated):
        expected = np.arange(int(concatenated.min()), int(concatenated.max()) + 1)
        if not np.array_equal(np.sort(concatenated), expected) or int(expected[0]) != 0:
            raise ValueError("Birth-event stable ids are not one contiguous range starting at zero")
    if conservative_maximum_active > args.max_secondary_particles:
        raise ValueError(
            "Conservative birth count exceeds max-secondary-particles: "
            f"{conservative_maximum_active} > {args.max_secondary_particles}"
        )

    report: dict[str, Any] = {
        "schema": (
            "physx-secondary-scene-bridge-preflight/v2"
            if args.secondary_dynamics == "foamgenerator-warp"
            else "physx-secondary-scene-bridge-preflight/v1"
        ),
        "valid": True,
        "checked_utc": utc_now(),
        "scene_usd": str(args.scene_usd.resolve()),
        "scene_usd_sha256": sha256_file(args.scene_usd),
        "frame_range": [args.start_frame, args.end_frame],
        "external_output_frame_range": [args.start_frame + 1, args.end_frame],
        "source_fps": args.source_fps,
        "physics_fps": args.physics_fps,
        "physics_steps_per_source_frame": args.physics_fps // args.source_fps,
        "primary_particle_count": primary_count,
        "source_files": source_files,
        "birth_files": birth_files,
        "birth_particle_count": len(concatenated),
        "stable_id_range": (
            [int(concatenated.min()), int(concatenated.max())] if len(concatenated) else None
        ),
        "conservative_maximum_active_particles": conservative_maximum_active,
        "max_secondary_particles": args.max_secondary_particles,
        "capacity_headroom": args.max_secondary_particles - conservative_maximum_active,
        "secondary_dynamics": {
            "mode": args.secondary_dynamics,
            "particle_radius_m": args.particle_radius,
            "support_radius_m": 4.0 * args.particle_radius,
            "foamgenerator_buoyancy": args.foamgenerator_buoyancy,
            "foamgenerator_drag": args.foamgenerator_drag,
            "warp_device": (
                args.warp_device
                if args.secondary_dynamics == "foamgenerator-warp"
                else None
            ),
        },
        "native_primary_velocity_required": True,
        "finite_difference_velocity_allowed": False,
    }
    if inspect_usd:
        report["usd_contract"] = inspect_usd_contract(args)
    return report


args = parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to reuse output directory: {args.output}")
args.output.mkdir(parents=True, exist_ok=False)
report_path = args.output / "bridge_report.json"
preflight_path = args.output / "preflight_report.json"
obsolete_path = args.output / "OBSOLETE.json"

try:
    preflight_report = preflight(args, inspect_usd=args.preflight_only)
    atomic_json(preflight_path, preflight_report)
except Exception as exc:
    failure = {
        "schema": "physx-secondary-scene-bridge-failure/v1",
        "valid": False,
        "obsolete": True,
        "failed_utc": utc_now(),
        "phase": "preflight",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    atomic_json(obsolete_path, failure)
    atomic_json(report_path, failure)
    raise

if args.preflight_only:
    print(json.dumps(preflight_report, indent=2, ensure_ascii=False))
    raise SystemExit(0)


simulation_app = None
simulation = None
stage = None
report: dict[str, Any] = {
    "schema": (
        "physx-secondary-scene-bridge/v2"
        if args.secondary_dynamics == "foamgenerator-warp"
        else "physx-secondary-scene-bridge/v1"
    ),
    "valid": False,
    "started_utc": utc_now(),
    "preflight": str(preflight_path.resolve()),
    "frame_range": [args.start_frame, args.end_frame],
    "external_output_frame_range": [args.start_frame + 1, args.end_frame],
    "max_secondary_particles": args.max_secondary_particles,
}
atomic_json(report_path, report)

try:
    gc.collect()
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

    if args.secondary_dynamics == "foamgenerator-warp":
        from foamgenerator_warp_dynamics import (
            BUBBLES,
            FOAM,
            SPRAY,
            FoamGeneratorWarpNeighborhood,
            apply_physx_velocity_control,
        )

    context = omni.usd.get_context()
    if not context.open_stage(str(args.scene_usd.resolve())):
        raise RuntimeError(f"Failed to open scene: {args.scene_usd}")
    for _ in range(4):
        simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("USD context returned no stage")

    physics_scene_prim = stage.GetPrimAtPath(args.physics_scene_path)
    particle_system_prim = stage.GetPrimAtPath(args.particle_system_path)
    primary_prim = stage.GetPrimAtPath(args.primary_particles_path)
    kinematic_prim = stage.GetPrimAtPath(args.kinematic_prim_path)
    required_runtime_prims = {
        args.physics_scene_path: physics_scene_prim,
        args.particle_system_path: particle_system_prim,
        args.primary_particles_path: primary_prim,
        args.kinematic_prim_path: kinematic_prim,
    }
    missing_runtime_prims = [path for path, prim in required_runtime_prims.items() if not prim]
    if missing_runtime_prims:
        raise ValueError(f"Scene is missing required prims: {missing_runtime_prims}")
    if not physics_scene_prim.IsA(UsdPhysics.Scene):
        raise ValueError(f"{args.physics_scene_path} is not a PhysicsScene")
    physics_scene = UsdPhysics.Scene(physics_scene_prim)
    gravity_direction = np.asarray(
        physics_scene.GetGravityDirectionAttr().Get(), dtype=np.float32
    )
    gravity_direction_norm = float(np.linalg.norm(gravity_direction.astype(np.float64)))
    if gravity_direction.shape != (3,) or gravity_direction_norm <= 0.0:
        raise ValueError("PhysicsScene has an invalid gravity direction")
    gravity_magnitude = float(physics_scene.GetGravityMagnitudeAttr().Get())
    if not np.isfinite(gravity_magnitude) or gravity_magnitude < 0.0:
        raise ValueError("PhysicsScene has an invalid gravity magnitude")
    gravity = np.ascontiguousarray(
        gravity_direction / np.float32(gravity_direction_norm) * np.float32(gravity_magnitude),
        dtype=np.float32,
    )
    if not particle_system_prim.IsA(PhysxSchema.PhysxParticleSystem):
        raise ValueError(f"{args.particle_system_path} is not a PhysxParticleSystem")
    if not primary_prim.IsA(UsdGeom.PointInstancer):
        raise ValueError(f"{args.primary_particles_path} is not a PointInstancer")
    if not primary_prim.HasAPI(PhysxSchema.PhysxParticleSetAPI):
        raise ValueError(f"{args.primary_particles_path} has no PhysxParticleSetAPI")
    primary_system_targets = primary_prim.GetRelationship(
        "physxParticle:particleSystem"
    ).GetTargets()
    if [str(path) for path in primary_system_targets] != [args.particle_system_path]:
        raise ValueError(
            f"Primary particle-system relationship is {primary_system_targets}, "
            f"expected {args.particle_system_path}"
        )

    removed_prims = []
    for prim_path in args.strip_prim:
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            stage.RemovePrim(prim_path)
            removed_prims.append(prim_path)

    particle_system = PhysxSchema.PhysxParticleSystem(particle_system_prim)
    particle_system.GetMaxVelocityAttr().Set(20.0)
    if particle_system_prim.HasAPI(PhysxSchema.PhysxParticleSmoothingAPI):
        PhysxSchema.PhysxParticleSmoothingAPI(
            particle_system_prim
        ).GetParticleSmoothingEnabledAttr().Set(False)
    if particle_system_prim.HasAPI(PhysxSchema.PhysxParticleAnisotropyAPI):
        PhysxSchema.PhysxParticleAnisotropyAPI(
            particle_system_prim
        ).GetParticleAnisotropyEnabledAttr().Set(False)
    if particle_system_prim.HasAPI(PhysxSchema.PhysxParticleIsosurfaceAPI):
        PhysxSchema.PhysxParticleIsosurfaceAPI(
            particle_system_prim
        ).GetIsosurfaceEnabledAttr().Set(False)
    generated_isosurface_path = Sdf.Path(args.particle_system_path).AppendChild("Isosurface")
    if stage.GetPrimAtPath(generated_isosurface_path):
        stage.RemovePrim(generated_isosurface_path)
        removed_prims.append(str(generated_isosurface_path))

    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physx_scene_api.GetEnableGPUDynamicsAttr().Set(True)
    physx_scene_api.GetBroadphaseTypeAttr().Set("GPU")
    physx_scene_api.GetTimeStepsPerSecondAttr().Set(args.physics_fps)

    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(kinematic_prim)
    rigid_body_api.GetRigidBodyEnabledAttr().Set(True)
    rigid_body_api.GetKinematicEnabledAttr().Set(True)
    kinematic_xform_op = UsdGeom.Xformable(kinematic_prim).MakeMatrixXform()

    secondary_path = Sdf.Path("/World/SecondaryParticles")
    if stage.GetPrimAtPath(secondary_path):
        raise ValueError(f"Scene already contains reserved bridge prim {secondary_path}")
    secondary_prim = particleUtils.add_physx_particleset_pointinstancer(
        stage,
        secondary_path,
        Vt.Vec3fArray(),
        Vt.Vec3fArray(),
        Sdf.Path(args.particle_system_path),
        self_collision=False,
        fluid=False,
        particle_group=1,
        particle_mass=0.0001,
        density=0.0,
    )
    secondary_prim.CreateAttribute(
        "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
    ).Set(args.max_secondary_particles)
    secondary_instancer = UsdGeom.PointInstancer.Get(stage, secondary_path)
    primary_instancer = UsdGeom.PointInstancer.Get(stage, args.primary_particles_path)

    settings = carb.settings.get_settings()
    settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
    settings.set(physx_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)
    settings.set(physx_bindings.SETTING_SUPPRESS_READBACK, False)

    dynamics_controller = (
        FoamGeneratorWarpNeighborhood(
            particle_radius=args.particle_radius,
            device=args.warp_device,
        )
        if args.secondary_dynamics == "foamgenerator-warp"
        else None
    )

    pool: dict[str, np.ndarray] = {
        "stable_ids": np.empty(0, dtype=np.int64),
        # FoamGenerator is built with single-precision Real.  Keep the bridge's
        # lifecycle arithmetic in the same dtype so particles that land within
        # a few ulps of zero expire on the identical source frame.
        "remaining_lifetimes": np.empty(0, dtype=np.float32),
        "birth_frames": np.empty(0, dtype=np.int64),
        "source_particle_indices": np.empty(0, dtype=np.int64),
        "particle_types": np.empty(0, dtype=np.int32),
    }

    def numpy_to_gf_matrix(values: np.ndarray) -> Gf.Matrix4d:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (4, 4) or not np.all(np.isfinite(values)):
            raise ValueError("Kinematic transform must be a finite 4x4 matrix")
        matrix = Gf.Matrix4d()
        for row in range(4):
            matrix.SetRow(row, Gf.Vec4d(*[float(value) for value in values[row]]))
        return matrix

    def set_kinematic_transform(values: np.ndarray) -> None:
        kinematic_xform_op.Set(numpy_to_gf_matrix(values))

    def load_source(frame: int) -> dict[str, np.ndarray]:
        path = frame_path(args.primary_source_pattern, frame)
        expected_count = int(preflight_report["primary_particle_count"])
        with np.load(path, allow_pickle=False) as archive:
            result = {
                "positions": np.ascontiguousarray(
                    require_array(
                        path,
                        archive,
                        "positions",
                        (expected_count, 3),
                        np.dtype("float32"),
                    )
                ),
                "velocities": np.ascontiguousarray(
                    require_array(
                        path,
                        archive,
                        "velocities",
                        (expected_count, 3),
                        np.dtype("float32"),
                    )
                ),
                "sphere_transform": np.ascontiguousarray(
                    require_array(
                        path,
                        archive,
                        "sphere_transform",
                        (4, 4),
                        np.dtype("float64"),
                    )
                ),
            }
        return result

    def author_primary(source: dict[str, np.ndarray]) -> None:
        primary_instancer.GetPositionsAttr().Set(
            Vt.Vec3fArray.FromNumpy(source["positions"])
        )
        primary_instancer.GetVelocitiesAttr().Set(
            Vt.Vec3fArray.FromNumpy(source["velocities"])
        )

    def read_secondary() -> tuple[np.ndarray, np.ndarray]:
        positions_value = secondary_instancer.GetPositionsAttr().Get()
        velocities_value = secondary_instancer.GetVelocitiesAttr().Get()
        positions = np.asarray(positions_value, dtype=np.float32).reshape((-1, 3)).copy()
        velocities = np.asarray(velocities_value, dtype=np.float32).reshape((-1, 3)).copy()
        expected = len(pool["stable_ids"])
        if positions.shape != velocities.shape or len(positions) != expected:
            raise RuntimeError(
                "PhysX secondary state and stable-id arrays are not aligned: "
                f"{positions.shape}, {velocities.shape}, ids={expected}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise RuntimeError("PhysX returned non-finite secondary state")
        return positions, velocities

    def author_secondary(positions: np.ndarray, velocities: np.ndarray) -> None:
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        velocities = np.ascontiguousarray(velocities, dtype=np.float32)
        if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Secondary state arrays must both have shape (N, 3)")
        if len(positions) > args.max_secondary_particles:
            raise ValueError(
                f"Secondary pool capacity exceeded: {len(positions)} > "
                f"{args.max_secondary_particles}"
            )
        secondary_instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
        secondary_instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(velocities))
        secondary_instancer.GetProtoIndicesAttr().Set(
            Vt.IntArray.FromNumpy(np.zeros(len(positions), dtype=np.int32))
        )

    def inject_births(frame: int) -> list[int]:
        path = frame_path(args.birth_pattern, frame)
        if not path.exists():
            return []
        birth = read_birth_events(path)
        if np.any(birth["birth_frame"] != frame):
            raise ValueError(f"{path}: birth_frame does not match frame {frame}")
        born_ids = birth["id"].astype(np.int64)
        if np.intersect1d(pool["stable_ids"], born_ids).size:
            raise ValueError(f"{path}: stable id is already active")
        positions, velocities = read_secondary()
        author_secondary(
            np.concatenate((positions, birth["position"].astype(np.float32))),
            np.concatenate((velocities, birth["velocity"].astype(np.float32))),
        )
        pool["stable_ids"] = np.concatenate((pool["stable_ids"], born_ids))
        pool["remaining_lifetimes"] = np.concatenate(
            (
                pool["remaining_lifetimes"],
                birth["remaining_lifetime"].astype(np.float32),
            )
        )
        pool["birth_frames"] = np.concatenate(
            (pool["birth_frames"], birth["birth_frame"].astype(np.int64))
        )
        pool["source_particle_indices"] = np.concatenate(
            (
                pool["source_particle_indices"],
                birth["source_particle_index"].astype(np.int64),
            )
        )
        pool["particle_types"] = np.concatenate(
            (
                pool["particle_types"],
                np.full(len(born_ids), 3, dtype=np.int32),
            )
        )
        if len(pool["stable_ids"]) > args.max_secondary_particles:
            raise ValueError("Birth injection exceeded the configured secondary capacity")
        return born_ids.tolist()

    collision_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                    "kinematic": (
                        bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                        else False
                    ),
                }
            )
    if not collision_prims:
        raise RuntimeError("Loaded scene has no collision prims")

    current_source = load_source(args.start_frame)
    author_primary(current_source)
    set_kinematic_transform(current_source["sphere_transform"])
    initial_born = inject_births(args.start_frame)

    configured_stage_path = args.output / "configured_bridge_stage.usda"
    stage.GetRootLayer().Export(str(configured_stage_path))

    simulation = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    simulation.attach_stage(stage_id)

    external_directory = args.output / "external_motion"
    external_directory.mkdir(parents=True, exist_ok=False)
    dynamics_control_directory = None
    if dynamics_controller is not None:
        dynamics_control_directory = args.output / "dynamics_control"
        dynamics_control_directory.mkdir(parents=True, exist_ok=False)
    source_dt = 1.0 / args.source_fps
    physics_dt = 1.0 / args.physics_fps
    substeps = args.physics_fps // args.source_fps
    frame_records = []
    events: list[dict[str, Any]] = [
        {
            "frame": args.start_frame,
            "event": "birth_after_output",
            "stable_ids": initial_born,
        }
    ]
    maximum_active = len(pool["stable_ids"])
    maximum_speed = 0.0
    maximum_control_delta_speed = 0.0
    controlled_particle_states = 0
    controlled_type_totals = {"foam": 0, "spray": 0, "bubbles": 0}

    for output_frame in range(args.start_frame + 1, args.end_frame + 1):
        next_source = load_source(output_frame)
        start_transform = current_source["sphere_transform"]
        end_transform = next_source["sphere_transform"]
        dynamics_result = None
        control_record: dict[str, Any] | None = None
        if dynamics_controller is not None:
            query_positions, query_velocities = read_secondary()
            dynamics_result = dynamics_controller.query(
                current_source["positions"],
                current_source["velocities"],
                query_positions,
            )
            pool["particle_types"] = dynamics_result.particle_types.copy()
            type_counts = {
                "foam": int(np.count_nonzero(pool["particle_types"] == FOAM)),
                "spray": int(np.count_nonzero(pool["particle_types"] == SPRAY)),
                "bubbles": int(np.count_nonzero(pool["particle_types"] == BUBBLES)),
            }
            for name, count in type_counts.items():
                controlled_type_totals[name] += count
            controlled_particle_states += len(pool["stable_ids"])
            control_path = dynamics_control_directory / f"control_{output_frame:06d}.npz"
            if control_path.exists():
                raise FileExistsError(control_path)
            np.savez_compressed(
                control_path,
                id=pool["stable_ids"].astype(np.int64),
                query_position=query_positions.astype(np.float32),
                query_velocity=query_velocities.astype(np.float32),
                particle_type=dynamics_result.particle_types.astype(np.int32),
                neighbor_count=dynamics_result.neighbor_counts.astype(np.int32),
                local_fluid_velocity=dynamics_result.fluid_velocities.astype(np.float32),
                kernel_weight_sum=dynamics_result.weight_sums.astype(np.float32),
                source_state_frame=np.asarray(output_frame - 1, dtype=np.int64),
                target_output_frame=np.asarray(output_frame, dtype=np.int64),
            )
            control_record = {
                "source_state_frame": output_frame - 1,
                "target_output_frame": output_frame,
                "particle_count": len(pool["stable_ids"]),
                "type_counts": type_counts,
                "neighbor_count_minimum": (
                    int(dynamics_result.neighbor_counts.min())
                    if len(pool["stable_ids"])
                    else None
                ),
                "neighbor_count_maximum": (
                    int(dynamics_result.neighbor_counts.max())
                    if len(pool["stable_ids"])
                    else None
                ),
                "file": str(control_path.resolve()),
                "bytes": control_path.stat().st_size,
            }
        frame_maximum_control_delta = 0.0
        for substep in range(1, substeps + 1):
            alpha = substep / substeps
            # The configured cache currently drives a sphere, so translation is
            # interpolated at 240 Hz while the cached end rotation is retained.
            # This keeps every collision response PhysX-owned without inventing
            # a finite-difference rigid-body velocity.
            interpolated = end_transform.copy()
            interpolated[3, :3] = (
                (1.0 - alpha) * start_transform[3, :3]
                + alpha * end_transform[3, :3]
            )
            set_kinematic_transform(interpolated)
            if dynamics_result is not None:
                controlled_positions, current_velocities = read_secondary()
                controlled_velocities, velocity_delta = apply_physx_velocity_control(
                    current_velocities,
                    dynamics_result.particle_types,
                    dynamics_result.fluid_velocities,
                    gravity,
                    physics_dt,
                    substeps,
                    args.foamgenerator_buoyancy,
                    args.foamgenerator_drag,
                )
                frame_maximum_control_delta = max(
                    frame_maximum_control_delta,
                    (
                        float(
                            np.linalg.norm(
                                velocity_delta.astype(np.float64), axis=1
                            ).max()
                        )
                        if len(velocity_delta)
                        else 0.0
                    ),
                )
                author_secondary(controlled_positions, controlled_velocities)
            absolute_step = (output_frame - args.start_frame - 1) * substeps + substep
            simulation.simulate(
                physics_dt,
                args.start_frame / args.source_fps + (absolute_step - 1) * physics_dt,
            )
            simulation.fetch_results()

        positions, velocities = read_secondary()
        exported_ids = pool["stable_ids"].copy()
        output_path = external_directory / f"external_{output_frame:06d}.bgeo"
        write_motion_state(
            output_path,
            positions,
            velocities,
            pool["stable_ids"].astype(np.int32),
        )
        frame_maximum_speed = (
            float(np.linalg.norm(velocities.astype(np.float64), axis=1).max())
            if len(velocities)
            else 0.0
        )
        maximum_speed = max(maximum_speed, frame_maximum_speed)
        maximum_control_delta_speed = max(
            maximum_control_delta_speed, frame_maximum_control_delta
        )

        # FoamGenerator subtracts chronological lifetime while writing this
        # frame, then removes expired ids at the beginning of the next frame.
        pool["remaining_lifetimes"] -= np.float32(source_dt)
        keep = pool["remaining_lifetimes"] > 0.0
        removed_ids = pool["stable_ids"][~keep].tolist()
        if not np.all(keep):
            author_secondary(positions[keep], velocities[keep])
            for name in tuple(pool):
                pool[name] = pool[name][keep]
            events.append(
                {
                    "frame": output_frame,
                    "event": "compact_after_output",
                    "stable_ids": removed_ids,
                }
            )

        # Primary cache remains authoritative at every 120 Hz boundary. The
        # secondary pool is intentionally never reset here.
        author_primary(next_source)
        set_kinematic_transform(next_source["sphere_transform"])
        born_ids = inject_births(output_frame) if output_frame < args.end_frame else []
        if born_ids:
            events.append(
                {
                    "frame": output_frame,
                    "event": "birth_after_output",
                    "stable_ids": born_ids,
                }
            )
        maximum_active = max(maximum_active, len(pool["stable_ids"]))
        frame_records.append(
            {
                "frame": output_frame,
                "exported_particles": len(positions),
                "id_minimum": int(exported_ids.min()) if len(exported_ids) else None,
                "id_maximum": int(exported_ids.max()) if len(exported_ids) else None,
                "expired_after_output": len(removed_ids),
                "born_after_output": len(born_ids),
                "active_for_next_frame": len(pool["stable_ids"]),
                "maximum_speed": frame_maximum_speed,
                "maximum_type_control_delta_speed": frame_maximum_control_delta,
                "type_dynamics_control": control_record,
                "position_minimum": positions.min(axis=0).astype(float).tolist() if len(positions) else None,
                "position_maximum": positions.max(axis=0).astype(float).tolist() if len(positions) else None,
                "file": str(output_path.resolve()),
                "bytes": output_path.stat().st_size,
            }
        )
        print(
            f"[secondary-bridge] frame={output_frame:06d} "
            f"exported={len(positions)} active_next={len(pool['stable_ids'])}",
            flush=True,
        )
        current_source = next_source
        gc.collect()

    expected_output_count = args.end_frame - args.start_frame
    output_files = sorted(external_directory.glob("external_*.bgeo"))
    control_files = (
        sorted(dynamics_control_directory.glob("control_*.npz"))
        if dynamics_control_directory is not None
        else []
    )
    checks = {
        "expected_external_frames_written": len(output_files) == expected_output_count,
        "capacity_not_exceeded": maximum_active <= args.max_secondary_particles,
        "all_exported_state_finite": all(
            record["maximum_speed"] is not None and np.isfinite(record["maximum_speed"])
            for record in frame_records
        ),
        "scene_has_collision_prims": bool(collision_prims),
        "primary_and_secondary_share_particle_system": True,
        "native_physx_positions_exported": True,
        "native_physx_velocities_exported": True,
        "finite_difference_velocity_not_used": True,
        "type_dynamics_control_frames_complete": (
            len(control_files) == expected_output_count
            if dynamics_controller is not None
            else True
        ),
        "foamgenerator_neighborhood_runs_on_cuda": (
            str(dynamics_controller.device).startswith("cuda:")
            if dynamics_controller is not None
            else True
        ),
        "type_dynamics_uses_native_primary_velocity": True,
    }
    report.update(
        {
            "valid": all(checks.values()),
            "completed_utc": utc_now(),
            "scene_usd": str(args.scene_usd.resolve()),
            "configured_stage": str(configured_stage_path.resolve()),
            "configured_stage_sha256": sha256_file(configured_stage_path),
            "physics_scene_path": args.physics_scene_path,
            "particle_system_path": args.particle_system_path,
            "primary_particles_path": args.primary_particles_path,
            "secondary_particles_path": str(secondary_path),
            "kinematic_prim_path": args.kinematic_prim_path,
            "removed_prims": removed_prims,
            "source_fps": args.source_fps,
            "physics_fps": args.physics_fps,
            "physics_steps_per_source_frame": substeps,
            "primary_particle_count": preflight_report["primary_particle_count"],
            "maximum_active_secondary_particles": maximum_active,
            "maximum_secondary_speed": maximum_speed,
            "maximum_type_control_delta_speed": maximum_control_delta_speed,
            "controlled_particle_states": controlled_particle_states,
            "controlled_type_totals": controlled_type_totals,
            "external_motion_directory": str(external_directory.resolve()),
            "dynamics_control_directory": (
                str(dynamics_control_directory.resolve())
                if dynamics_control_directory is not None
                else None
            ),
            "position_authority": "PhysX PointInstancer native readback",
            "velocity_authority": (
                "PhysX PointInstancer native readback after audited FoamGenerator-type "
                "velocity control"
                if dynamics_controller is not None
                else "PhysX PointInstancer velocities native readback"
            ),
            "collision_authority": "loaded PhysX scene",
            "primary_state_authority": "cached native PhysX PointInstancer positions and velocities",
            "lifetime_authority": "FoamGenerator chronological lifetime contract",
            "secondary_dynamics": {
                "mode": args.secondary_dynamics,
                "particle_radius_m": args.particle_radius,
                "support_radius_m": 4.0 * args.particle_radius,
                "classification_thresholds": {
                    "spray_below_neighbor_count": 6,
                    "bubbles_above_neighbor_count": 20,
                },
                "foam_velocity_model": (
                    "kernel-weighted local primary velocity with per-substep gravity compensation"
                    if dynamics_controller is not None
                    else None
                ),
                "bubble_buoyancy": args.foamgenerator_buoyancy,
                "bubble_drag_per_source_step": args.foamgenerator_drag,
                "bubble_drag_substep_conversion": "1-(1-drag)^(1/substeps)",
                "gravity_m_per_s2": gravity.astype(float).tolist(),
                "warp_device": (
                    str(dynamics_controller.device)
                    if dynamics_controller is not None
                    else None
                ),
                "finite_difference_velocity_used": False,
            },
            "collision_prims": collision_prims,
            "events": events,
            "frames": frame_records,
            "checks": checks,
        }
    )
    atomic_json(report_path, report)
    if not report["valid"]:
        atomic_json(
            obsolete_path,
            {
                "schema": "physx-secondary-scene-bridge-obsolete/v1",
                "obsolete": True,
                "reason": "One or more bridge invariants failed",
                "checks": checks,
                "report": str(report_path.resolve()),
            },
        )
        raise RuntimeError(f"Scene bridge invariants failed: {checks}")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))

except Exception as exc:
    failure = {
        "schema": "physx-secondary-scene-bridge-failure/v1",
        "valid": False,
        "obsolete": True,
        "failed_utc": utc_now(),
        "phase": "physx_bridge",
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "partial_report": report,
    }
    atomic_json(obsolete_path, failure)
    atomic_json(report_path, failure)
    raise
finally:
    if simulation is not None:
        try:
            simulation.detach_stage()
        except Exception:
            pass
    if simulation_app is not None:
        simulation_app.close()
