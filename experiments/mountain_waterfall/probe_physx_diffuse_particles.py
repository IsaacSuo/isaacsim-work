"""Low-load gate for PhysX GPU diffuse-particle generation and Python readback.

This probe deliberately does not load the mountain scene.  It creates a small
water slug above a shallow particle pool, enables PhysX diffuse particles, and
records every USD/Fabric-facing clue that Isaac Sim exposes.  A failed or
inconclusive run is preserved and marked obsolete instead of being reused.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
# Keep incidental CPU libraries from occupying the whole workstation.  PhysX
# particle dynamics remain explicitly GPU-backed below.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--physics-fps", type=int, default=240)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--spacing", type=float, default=0.03)
parser.add_argument("--diffuse-threshold", type=float, default=1.0e-4)
parser.add_argument("--diffuse-lifetime", type=float, default=1.0)
parser.add_argument("--diffuse-multiplier", type=float, default=2.0)
parser.add_argument(
    "--native-diffuse-bridge",
    action="store_true",
    help="Use the pinned C++/CUDA bridge and require complete native Diffuse readback",
)
parser.add_argument(
    "--recover-native-labels",
    action="store_true",
    help="Recover PhysX-compatible spray/foam/bubble labels from primary-neighbor counts",
)
parser.add_argument(
    "--full-diffuse-advection",
    action="store_true",
    help="Enable PhysX's native 27-cell Diffuse advection for exact full-neighborhood recovery",
)
parser.add_argument(
    "--native-frame-stride",
    type=int,
    default=0,
    help=(
        "Additionally save a native Diffuse frame every N physics steps. "
        "Zero keeps the original checkpoint-only audit behavior."
    ),
)
parser.add_argument(
    "--omnipvd",
    action="store_true",
    help="Record the native simulation to OVD and audit the converted OmniPVD USDA",
)
args = parser.parse_args()

if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
if not 60 <= args.physics_fps <= 480 or not 24 <= args.steps <= 240:
    raise ValueError("Probe is restricted to 60..480 Hz and 24..240 steps")
if not 0.02 <= args.spacing <= 0.05:
    raise ValueError("Probe spacing must remain within 0.02..0.05 m")
if not 0.0 <= args.diffuse_threshold <= 0.1:
    raise ValueError("Diffuse threshold is outside the probe safety range")
if not 0.05 <= args.diffuse_lifetime <= 2.0:
    raise ValueError("Diffuse lifetime is outside the probe safety range")
if not 0.0 < args.diffuse_multiplier <= 3.0:
    raise ValueError("Diffuse multiplier must be in (0, 3]")
if (args.recover_native_labels or args.full_diffuse_advection) and not args.native_diffuse_bridge:
    raise ValueError("Native label recovery and full Diffuse advection require --native-diffuse-bridge")
if not 0 <= args.native_frame_stride <= args.steps:
    raise ValueError("native-frame-stride must be zero or within 1..steps")
if args.native_frame_stride and not args.native_diffuse_bridge:
    raise ValueError("native-frame-stride requires --native-diffuse-bridge")

args.output.mkdir(parents=True, exist_ok=False)
report_path = args.output / "probe_report.json"
obsolete_path = args.output / "OBSOLETE.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def set_below_normal_priority() -> bool:
    if os.name != "nt":
        return False
    below_normal_priority_class = 0x00004000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal_priority_class))


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory() -> dict[str, float | None]:
    if os.name != "nt":
        return {"working_set_mib": None, "peak_working_set_mib": None, "private_mib": None}
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCountersEx),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    success = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not success:
        return {"working_set_mib": None, "peak_working_set_mib": None, "private_mib": None}
    scale = 1024.0 * 1024.0
    return {
        "working_set_mib": counters.WorkingSetSize / scale,
        "peak_working_set_mib": counters.PeakWorkingSetSize / scale,
        "private_mib": counters.PrivateUsage / scale,
    }


def gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=True
        )
        devices = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                devices.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "memory_used_mib": int(fields[2]),
                        "memory_total_mib": int(fields[3]),
                        "utilization_percent": int(fields[4]),
                    }
                )
        return {"available": True, "devices": devices}
    except Exception as exc:  # diagnostic only; absence must not crash physics
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def resource_sample(label: str, start_wall: float, start_cpu: float) -> dict[str, object]:
    return {
        "label": label,
        "elapsed_seconds": time.perf_counter() - start_wall,
        "process_cpu_seconds": time.process_time() - start_cpu,
        "process_memory": process_memory(),
        "gpu": gpu_snapshot(),
    }


def filtered_dir(value: object) -> list[str]:
    needles = ("particle", "diffuse", "buffer", "fabric")
    return sorted(name for name in dir(value) if any(term in name.lower() for term in needles))


def lattice(minimum, counts, spacing, velocity) -> tuple[np.ndarray, np.ndarray]:
    axes = [
        minimum[index] + np.arange(counts[index], dtype=np.float32) * spacing
        for index in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    velocities = np.repeat(np.asarray(velocity, dtype=np.float32)[None, :], len(grid), axis=0)
    return np.ascontiguousarray(grid), np.ascontiguousarray(velocities)


def array_length(attribute) -> int | None:
    try:
        value = attribute.Get()
        return len(value) if value is not None and hasattr(value, "__len__") else None
    except Exception:
        return None


priority_lowered = set_below_normal_priority()
wall_start = time.perf_counter()
cpu_start = time.process_time()
report: dict[str, object] = {
    "schema": 1,
    "product": "physx_diffuse_particles_probe",
    "created_utc": utc_now(),
    "status": "running",
    "valid": False,
    "inconclusive": True,
    "purpose": [
        "confirm native PhysX diffuse generation",
        "test Python/USD readback of positions velocities lifetime and type",
        "bound workstation load before any mountain-scene integration",
    ],
    "configuration": {
        "physics_fps": args.physics_fps,
        "steps": args.steps,
        "duration_seconds": args.steps / args.physics_fps,
        "spacing_m": args.spacing,
        "diffuse_threshold": args.diffuse_threshold,
        "diffuse_lifetime_seconds": args.diffuse_lifetime,
        "max_diffuse_particle_multiplier": args.diffuse_multiplier,
        "maximum_primary_particles": 3000,
        "maximum_diffuse_multiplier_gate": 3.0,
        "headless": True,
        "cycles_or_splashsurf": False,
        "mountain_scene_loaded": False,
        "omnipvd_enabled": args.omnipvd,
        "native_diffuse_bridge_enabled": args.native_diffuse_bridge,
        "native_frame_stride": args.native_frame_stride,
        "recover_native_labels": args.recover_native_labels,
        "full_diffuse_advection_requested": args.full_diffuse_advection,
        "cpu_thread_environment_cap": 2,
        "process_priority_below_normal": priority_lowered,
    },
    "resource_samples": [resource_sample("before_isaac_startup", wall_start, cpu_start)],
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
            # SimulationApp otherwise replaces the library environment caps
            # with the machine's full core count in Kit's task schedulers.
            "limit_cpu_threads": 2,
        }
    )

    import carb
    import omni.physx
    import omni.physx.bindings._physx as physx_bindings
    import omni.usd
    import usdrt
    from omni.physx import get_physx_simulation_interface, get_physx_statistics_interface
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt

    native_diffuse_bridge = None
    recover_diffuse_labels = None
    diffuse_label_histogram = None
    if args.native_diffuse_bridge:
        bridge_directory = Path(__file__).resolve().parent / "native_bridge"
        sys.path.insert(0, str(bridge_directory))
        import physx_diffuse_bridge as native_diffuse_bridge

        report["native_bridge_abi"] = native_diffuse_bridge.abi_info()
        if args.recover_native_labels:
            from physx_diffuse_labels import label_histogram as diffuse_label_histogram
            from physx_diffuse_labels import recover_labels as recover_diffuse_labels

    report["resource_samples"].append(
        resource_sample("after_isaac_startup", wall_start, cpu_start)
    )

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    def add_static_box(path, centre, half_extents):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr().Set(2.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
        xform.AddScaleOp().Set(Gf.Vec3f(*map(float, half_extents)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        return cube

    # A closed, shallow catch basin prevents escape without any scene mesh.
    add_static_box("/World/Basin/Bottom", (0.0, -0.08, 0.0), (0.55, 0.05, 0.48))
    add_static_box("/World/Basin/WallXNeg", (-0.52, 0.18, 0.0), (0.03, 0.31, 0.48))
    add_static_box("/World/Basin/WallXPos", (0.52, 0.18, 0.0), (0.03, 0.31, 0.48))
    add_static_box("/World/Basin/WallZNeg", (0.0, 0.18, -0.45), (0.55, 0.31, 0.03))
    add_static_box("/World/Basin/WallZPos", (0.0, 0.18, 0.45), (0.55, 0.31, 0.03))

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
    physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
    physx_scene.CreateGpuMaxParticleContactsAttr().Set(100_000)
    physx_scene.CreateTimeStepsPerSecondAttr().Set(args.physics_fps)

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
        solver_position_iterations=6,
        max_neighborhood=96,
        neighborhood_scale=1.01,
        max_velocity=10.0,
    )
    material_path = Sdf.Path("/World/WaterPhysics")
    particleUtils.add_pbd_particle_material(
        stage,
        material_path,
        density=1000.0,
        friction=0.05,
        damping=0.01,
        viscosity=0.002,
        vorticity_confinement=0.02,
        surface_tension=0.0074,
        cohesion=0.01,
        adhesion=0.0,
        cfl_coefficient=1.0,
    )
    physicsUtils.add_physics_material_to_prim(stage, particle_system.GetPrim(), material_path)

    pool_positions, pool_velocities = lattice(
        (-0.30, 0.00, -0.27), (21, 3, 19), args.spacing, (0.0, 0.0, 0.0)
    )
    slug_positions, slug_velocities = lattice(
        (-0.15, 0.50, -0.15), (11, 7, 11), args.spacing, (0.0, -2.8, 0.0)
    )
    initial_positions = np.concatenate((pool_positions, slug_positions)).astype(np.float32)
    initial_velocities = np.concatenate((pool_velocities, slug_velocities)).astype(np.float32)
    if len(initial_positions) > 3000:
        raise RuntimeError(f"Primary-particle safety cap exceeded: {len(initial_positions)}")

    particles_path = Sdf.Path("/World/WaterParticles")
    particles_prim = particleUtils.add_physx_particleset_pointinstancer(
        stage,
        particles_path,
        Vt.Vec3fArray.FromNumpy(initial_positions),
        Vt.Vec3fArray.FromNumpy(initial_velocities),
        particle_system_path,
        self_collision=True,
        fluid=True,
        particle_group=0,
        particle_mass=1000.0 * args.spacing**3,
        density=1000.0,
    )
    particles_prim.CreateAttribute("physxParticle:maxParticles", Sdf.ValueTypeNames.Int).Set(
        len(initial_positions)
    )
    instancer = UsdGeom.PointInstancer.Get(stage, particles_path)
    prototype = UsdGeom.Imageable.Get(stage, particles_path.AppendChild("particlePrototype0"))
    if prototype:
        prototype.MakeInvisible()

    diffuse_apply_succeeded = particleUtils.add_physx_diffuse_particles(
        stage,
        particles_path,
        enabled=True,
        max_diffuse_particle_multiplier=args.diffuse_multiplier,
        threshold=args.diffuse_threshold,
        lifetime=args.diffuse_lifetime,
        air_drag=0.1,
        bubble_drag=0.5,
        buoyancy=0.8,
        kinetic_energy_weight=1.0,
        pressure_weight=1.0,
        divergence_weight=5.0,
        collision_decay=0.25,
        use_accurate_velocity=True,
    )

    settings = carb.settings.get_settings()
    settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
    settings.set(physx_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
    settings.set(physx_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)

    omnipvd_recording_directory = args.output / "omnipvd_recording"
    if args.omnipvd:
        omnipvd_recording_directory.mkdir(exist_ok=True)
        omnipvd_directory_setting = omnipvd_recording_directory.resolve().as_posix() + "/"
        # OmniPVD reads these settings when the stage is attached.  NVIDIA's
        # own tests require the output directory to be set before recording is
        # enabled, followed by one app update so the setting takes effect.
        settings.set(
            physx_bindings.SETTING_OMNIPVD_OVD_RECORDING_DIRECTORY,
            omnipvd_directory_setting,
        )
        # Isaac's CUDA device manager may enable suppressReadback for maximum
        # RL throughput.  OmniPVD needs host-visible particle state, so disable
        # suppression explicitly without disabling GPU dynamics themselves.
        settings.set(physx_bindings.SETTING_SUPPRESS_READBACK, False)
        settings.set(physx_bindings.SETTING_OMNIPVD_ENABLED, True)
        simulation_app.update()
        report["omnipvd"] = {
            "recording_directory": omnipvd_directory_setting,
            "recording_enabled_before_attach": bool(
                settings.get_as_bool(physx_bindings.SETTING_OMNIPVD_ENABLED)
            ),
            "suppress_readback_setting_before_attach": bool(
                settings.get_as_bool(physx_bindings.SETTING_SUPPRESS_READBACK)
            ),
        }

    stage_path = args.output / "physx_diffuse_probe.usda"
    stage.GetRootLayer().Export(str(stage_path))
    simulation = get_physx_simulation_interface()
    statistics = get_physx_statistics_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    simulation.attach_stage(stage_id)
    if native_diffuse_bridge is not None:
        report["native_particle_system_before_configuration"] = (
            native_diffuse_bridge.particle_system_info(str(particle_system_path))
        )
        if args.full_diffuse_advection:
            native_diffuse_bridge.set_full_diffuse_advection(str(particle_system_path), True)
        report["native_particle_system"] = native_diffuse_bridge.particle_system_info(
            str(particle_system_path)
        )
        if args.full_diffuse_advection and not report["native_particle_system"][
            "full_diffuse_advection"
        ]:
            raise RuntimeError("PhysX rejected the requested full Diffuse advection flag")
    if args.omnipvd:
        report["omnipvd"]["readback_suppressed_after_attach"] = bool(
            omni.physx.get_physx_interface().is_readback_suppressed()
        )
    runtime_stage = usdrt.Usd.Stage.Attach(context.get_stage_id())

    report["primary_particles"] = {
        "pool": len(pool_positions),
        "impact_slug": len(slug_positions),
        "total": len(initial_positions),
        "maximum_possible_diffuse": int(math.ceil(len(initial_positions) * args.diffuse_multiplier)),
    }
    diffuse_api = PhysxSchema.PhysxDiffuseParticlesAPI.Get(stage, particles_path)
    report["diffuse_api"] = {
        "helper_returned": bool(diffuse_apply_succeeded),
        "applied": particles_prim.HasAPI(PhysxSchema.PhysxDiffuseParticlesAPI),
        "attributes": {
            attribute.GetName(): attribute.Get()
            for attribute in diffuse_api.GetPrim().GetAttributes()
            if "physxDiffuseParticles" in attribute.GetName()
        },
    }
    report["python_interfaces"] = {
        "omni_physx": filtered_dir(omni.physx),
        "physx_bindings": filtered_dir(physx_bindings),
        "simulation_interface": filtered_dir(simulation),
        "statistics_interface": filtered_dir(statistics),
    }

    native_samples: list[dict[str, object]] = []

    def inspect_native_diffuse(step: int) -> dict[str, object]:
        if native_diffuse_bridge is None:
            return {"requested": False, "step": step}

        sample = native_diffuse_bridge.read_particle_frame(str(particles_path))
        primary_position_inv_mass = sample["primary_position_inv_mass"]
        primary_velocity = sample["primary_velocity"]
        position_lifetime = sample["diffuse_position_lifetime"]
        velocity = sample["diffuse_velocity"]
        primary_active_count = int(sample["primary_active_count"])
        primary_max_count = int(sample["primary_max_count"])
        active_count = int(sample["diffuse_active_count"])
        max_count = int(sample["diffuse_max_count"])
        expected_max_count = int(math.ceil(len(initial_positions) * args.diffuse_multiplier))

        checks = {
            "primary_position_shape": list(primary_position_inv_mass.shape)
            == [primary_active_count, 4],
            "primary_velocity_shape": list(primary_velocity.shape) == [primary_active_count, 4],
            "primary_count_matches_authored": primary_active_count == len(initial_positions),
            "primary_active_not_above_max": 0 <= primary_active_count <= primary_max_count,
            "primary_position_finite": bool(np.isfinite(primary_position_inv_mass).all()),
            "primary_velocity_finite": bool(np.isfinite(primary_velocity).all()),
            "position_lifetime_shape": list(position_lifetime.shape) == [active_count, 4],
            "velocity_shape": list(velocity.shape) == [active_count, 4],
            "position_lifetime_dtype": position_lifetime.dtype == np.dtype("float32"),
            "velocity_dtype": velocity.dtype == np.dtype("float32"),
            "active_not_above_max": 0 <= active_count <= max_count,
            "max_matches_authored_capacity": max_count == expected_max_count,
            "position_lifetime_finite": bool(np.isfinite(position_lifetime).all()),
            "velocity_finite": bool(np.isfinite(velocity).all()),
            "lifetime_in_authored_range": bool(
                active_count == 0
                or (
                    np.min(position_lifetime[:, 3]) >= -1.0e-5
                    and np.max(position_lifetime[:, 3]) <= args.diffuse_lifetime + 1.0e-4
                )
            ),
            "velocity_xyz_not_all_zero": bool(
                active_count > 0 and np.any(np.abs(velocity[:, :3]) > 1.0e-6)
            ),
        }
        recovered_labels = None
        neighbor_counts = None
        recovered_histogram = None
        if args.recover_native_labels:
            contact_distance = float(report["native_particle_system"]["diffuse_neighbor_radius"])
            recovered_labels, neighbor_counts = recover_diffuse_labels(
                primary_position_inv_mass[:, :3],
                position_lifetime[:, :3],
                contact_distance,
                maximum_neighbors=16,
            )
            recovered_histogram = diffuse_label_histogram(recovered_labels)
            checks.update(
                {
                    "label_shape": list(recovered_labels.shape) == [active_count],
                    "neighbor_count_shape": list(neighbor_counts.shape) == [active_count],
                    "label_dtype": recovered_labels.dtype == np.dtype("uint8"),
                    "neighbor_count_dtype": neighbor_counts.dtype == np.dtype("uint8"),
                    "neighbor_counts_in_physx_range": bool(
                        np.all(neighbor_counts <= 16)
                    ),
                    "label_histogram_conserves_active_count": sum(
                        recovered_histogram.values()
                    )
                    == active_count,
                    "full_diffuse_advection_enabled": bool(
                        report["native_particle_system"]["full_diffuse_advection"]
                    ),
                    "kernel_neighbor_radius_is_twice_particle_contact_offset": bool(
                        np.isclose(
                            contact_distance,
                            2.0
                            * float(
                                report["native_particle_system"]["particle_contact_offset"]
                            ),
                            rtol=1.0e-6,
                            atol=1.0e-8,
                        )
                    ),
                }
            )
        record = {
            "requested": True,
            "step": step,
            "simulation_time": step / args.physics_fps,
            "active_count": active_count,
            "max_count": max_count,
            "primary_active_count": primary_active_count,
            "primary_max_count": primary_max_count,
            "expected_max_count": expected_max_count,
            "checks": checks,
            "all_structural_checks_passed": all(
                value for name, value in checks.items() if name != "velocity_xyz_not_all_zero"
            ),
            "lifetime_range": (
                [float(np.min(position_lifetime[:, 3])), float(np.max(position_lifetime[:, 3]))]
                if active_count
                else None
            ),
            "speed_range": (
                [
                    float(np.min(np.linalg.norm(velocity[:, :3], axis=1))),
                    float(np.max(np.linalg.norm(velocity[:, :3], axis=1))),
                ]
                if active_count
                else None
            ),
            "label_histogram": recovered_histogram,
            "neighbor_count_range": (
                [int(np.min(neighbor_counts)), int(np.max(neighbor_counts))]
                if neighbor_counts is not None and active_count
                else None
            ),
        }

        samples_directory = args.output / "native_diffuse_samples"
        samples_directory.mkdir(exist_ok=True)
        atomic_npz(
            samples_directory / f"diffuse_native_{step:04d}.npz",
            schema=np.asarray("physx_diffuse_labeled_frame/1"),
            physics_step=np.asarray(step, dtype=np.int64),
            simulation_time=np.asarray(step / args.physics_fps, dtype=np.float64),
            particle_set_path=np.asarray(str(particles_path)),
            particle_system_path=np.asarray(str(particle_system_path)),
            particle_contact_offset=np.asarray(
                report["native_particle_system"]["particle_contact_offset"], dtype=np.float32
            ),
            diffuse_neighbor_radius=np.asarray(
                report["native_particle_system"]["diffuse_neighbor_radius"], dtype=np.float32
            ),
            full_diffuse_advection=np.asarray(
                report["native_particle_system"]["full_diffuse_advection"], dtype=np.bool_
            ),
            primary_active_count=np.asarray(primary_active_count, dtype=np.uint32),
            primary_max_count=np.asarray(primary_max_count, dtype=np.uint32),
            active_count=np.asarray(active_count, dtype=np.uint32),
            max_count=np.asarray(max_count, dtype=np.uint32),
            primary_position_inv_mass=np.ascontiguousarray(primary_position_inv_mass),
            primary_velocity=np.ascontiguousarray(primary_velocity),
            position_lifetime=np.ascontiguousarray(position_lifetime),
            velocity=np.ascontiguousarray(velocity),
            recovered_label=(
                np.ascontiguousarray(recovered_labels)
                if recovered_labels is not None
                else np.empty(0, dtype=np.uint8)
            ),
            primary_neighbor_count=(
                np.ascontiguousarray(neighbor_counts)
                if neighbor_counts is not None
                else np.empty(0, dtype=np.uint8)
            ),
        )
        native_samples.append(record)
        return record

    def inspect_stage(step: int) -> dict[str, object]:
        prims = []
        readable_candidates = []
        for prim in stage.Traverse():
            attributes = []
            for attribute in prim.GetAttributes():
                name = str(attribute.GetName())
                length = array_length(attribute)
                if (
                    "diffuse" in name.lower()
                    or "particle" in name.lower()
                    or name in ("points", "positions", "velocities", "ids", "protoIndices")
                ):
                    attributes.append({"name": name, "array_length": length})
                    if length is not None and "diffuseparticles" in str(prim.GetPath()).lower():
                        readable_candidates.append(
                            {"prim": str(prim.GetPath()), "attribute": name, "array_length": length}
                        )
            prims.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "attributes": attributes,
                }
            )
        primary_positions = np.asarray(instancer.GetPositionsAttr().Get(), dtype=np.float32)
        primary_velocities = np.asarray(instancer.GetVelocitiesAttr().Get(), dtype=np.float32)
        diffuse_prim = stage.GetPrimAtPath("/World/ParticleSystem/DiffuseParticles")
        diffuse_arrays = {}
        if diffuse_prim:
            for source_name, output_name, dtype in (
                ("points", "positions", np.float32),
                ("velocities", "velocities", np.float32),
                ("ids", "ids", np.int64),
            ):
                value = diffuse_prim.GetAttribute(source_name).Get()
                if value is not None:
                    array = np.ascontiguousarray(value, dtype=dtype)
                    if array.ndim >= 1:
                        diffuse_arrays[output_name] = array
        diffuse_positions = diffuse_arrays.get("positions")
        diffuse_data = {
            "position_count": int(len(diffuse_positions)) if diffuse_positions is not None else 0,
            "velocity_count": int(len(diffuse_arrays.get("velocities", ()))),
            "id_count": int(len(diffuse_arrays.get("ids", ()))),
            "positions_finite": bool(np.isfinite(diffuse_positions).all())
            if diffuse_positions is not None
            else None,
            "position_bounds": [
                diffuse_positions.min(axis=0).tolist(), diffuse_positions.max(axis=0).tolist()
            ]
            if diffuse_positions is not None and len(diffuse_positions)
            else None,
        }
        if diffuse_arrays:
            samples_directory = args.output / "diffuse_samples"
            samples_directory.mkdir(exist_ok=True)
            atomic_npz(
                samples_directory / f"diffuse_{step:04d}.npz",
                physics_step=np.asarray(step, dtype=np.int64),
                simulation_time=np.asarray(step / args.physics_fps, dtype=np.float64),
                **diffuse_arrays,
            )

        runtime_data = {
            "prim_valid": False,
            "property_names": [],
            "attributes": [],
            "arrays": {},
        }
        runtime_prim = runtime_stage.GetPrimAtPath("/World/ParticleSystem/DiffuseParticles")
        if runtime_prim:
            runtime_data["prim_valid"] = True
            runtime_data["property_names"] = sorted(
                str(name) for name in runtime_prim.GetPropertyNames()
            )
            runtime_arrays = {}
            for runtime_attribute in runtime_prim.GetAttributes():
                attribute_name = str(runtime_attribute.GetName())
                record = {
                    "name": attribute_name,
                    "type_name": str(runtime_attribute.GetTypeName()),
                    "cpu_valid_before": bool(runtime_attribute.IsCpuDataValid()),
                    "gpu_valid_before": bool(runtime_attribute.IsGpuDataValid()),
                    "sync_to_cpu_attempted": False,
                    "sync_to_cpu_succeeded": None,
                    "read_error": None,
                    "array_shape": None,
                    "array_dtype": None,
                    "array_length": None,
                    "value_preview": None,
                }
                try:
                    if record["gpu_valid_before"] and not record["cpu_valid_before"]:
                        record["sync_to_cpu_attempted"] = True
                        record["sync_to_cpu_succeeded"] = bool(runtime_attribute.SyncDataToCpu())
                    value = runtime_attribute.Get()
                    record["cpu_valid_after"] = bool(runtime_attribute.IsCpuDataValid())
                    record["gpu_valid_after"] = bool(runtime_attribute.IsGpuDataValid())
                    if value is not None:
                        try:
                            array = np.asarray(value)
                            record["array_shape"] = list(array.shape)
                            record["array_dtype"] = str(array.dtype)
                            record["array_length"] = int(len(array)) if array.ndim else None
                            record["value_preview"] = str(value)[:240]
                            if array.ndim >= 1 and array.dtype != object:
                                key = "usdrt_" + "".join(
                                    character if character.isalnum() else "_"
                                    for character in attribute_name
                                )
                                runtime_arrays[key] = np.ascontiguousarray(array)
                                runtime_data["arrays"][attribute_name] = {
                                    "npz_key": key,
                                    "shape": list(array.shape),
                                    "dtype": str(array.dtype),
                                }
                        except Exception as array_error:
                            record["value_preview"] = str(value)[:240]
                            record["array_conversion_error"] = (
                                f"{type(array_error).__name__}: {array_error}"
                            )
                except Exception as read_error:
                    record["read_error"] = f"{type(read_error).__name__}: {read_error}"
                runtime_data["attributes"].append(record)
            if runtime_arrays:
                samples_directory = args.output / "diffuse_samples"
                samples_directory.mkdir(exist_ok=True)
                atomic_npz(
                    samples_directory / f"diffuse_usdrt_{step:04d}.npz",
                    physics_step=np.asarray(step, dtype=np.int64),
                    simulation_time=np.asarray(step / args.physics_fps, dtype=np.float64),
                    **runtime_arrays,
                )

        related_runtime_prims = {}
        related_runtime_arrays = {}
        for related_path in (
            "/World/ParticleSystem",
            "/World/WaterParticles",
            "/World/ParticleSystem/DiffuseParticles",
        ):
            related_prim = runtime_stage.GetPrimAtPath(related_path)
            related_record = {
                "valid": bool(related_prim),
                "property_names": [],
                "attributes": [],
            }
            if related_prim:
                related_record["property_names"] = sorted(
                    str(name) for name in related_prim.GetPropertyNames()
                )
                for related_attribute in related_prim.GetAttributes():
                    related_name = str(related_attribute.GetName())
                    attribute_record = {
                        "name": related_name,
                        "type_name": str(related_attribute.GetTypeName()),
                        "cpu_valid_before": bool(related_attribute.IsCpuDataValid()),
                        "gpu_valid_before": bool(related_attribute.IsGpuDataValid()),
                        "sync_to_cpu_attempted": False,
                        "sync_to_cpu_succeeded": None,
                        "shape": None,
                        "dtype": None,
                        "length": None,
                        "read_error": None,
                    }
                    try:
                        if attribute_record["gpu_valid_before"] and not attribute_record["cpu_valid_before"]:
                            attribute_record["sync_to_cpu_attempted"] = True
                            attribute_record["sync_to_cpu_succeeded"] = bool(
                                related_attribute.SyncDataToCpu()
                            )
                        related_value = related_attribute.Get()
                        attribute_record["cpu_valid_after"] = bool(
                            related_attribute.IsCpuDataValid()
                        )
                        attribute_record["gpu_valid_after"] = bool(
                            related_attribute.IsGpuDataValid()
                        )
                        if related_value is not None:
                            related_array = np.asarray(related_value)
                            attribute_record["shape"] = list(related_array.shape)
                            attribute_record["dtype"] = str(related_array.dtype)
                            attribute_record["length"] = (
                                int(len(related_array)) if related_array.ndim else None
                            )
                            if related_array.ndim >= 1 and related_array.dtype != object:
                                related_key = "internal_" + "".join(
                                    character if character.isalnum() else "_"
                                    for character in related_path + "_" + related_name
                                )
                                related_runtime_arrays[related_key] = np.ascontiguousarray(
                                    related_array
                                )
                    except Exception as related_error:
                        attribute_record["read_error"] = (
                            f"{type(related_error).__name__}: {related_error}"
                        )
                    related_record["attributes"].append(attribute_record)
            related_runtime_prims[related_path] = related_record
        if related_runtime_arrays:
            samples_directory = args.output / "diffuse_samples"
            samples_directory.mkdir(exist_ok=True)
            atomic_npz(
                samples_directory / f"diffuse_internal_{step:04d}.npz",
                physics_step=np.asarray(step, dtype=np.int64),
                simulation_time=np.asarray(step / args.physics_fps, dtype=np.float64),
                **related_runtime_arrays,
            )
        return {
            "step": step,
            "primary_position_count": len(primary_positions),
            "primary_velocity_count": len(primary_velocities),
            "primary_bounds": [primary_positions.min(axis=0).tolist(), primary_positions.max(axis=0).tolist()],
            "maximum_primary_speed": float(np.linalg.norm(primary_velocities, axis=1).max()),
            "prims": prims,
            "readable_non_primary_candidates": readable_candidates,
            "diffuse_data": diffuse_data,
            "usdrt_diffuse": runtime_data,
            "usdrt_related_prims": related_runtime_prims,
        }

    checkpoints = {0, args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps}
    snapshots = [inspect_stage(0)]
    report["resource_samples"].append(resource_sample("before_simulation", wall_start, cpu_start))
    for step in range(1, args.steps + 1):
        simulation.simulate(1.0 / args.physics_fps, (step - 1) / args.physics_fps)
        simulation.fetch_results()
        simulation_app.update()
        if step in checkpoints:
            snapshot = inspect_stage(step)
            snapshot["native_diffuse"] = inspect_native_diffuse(step)
            snapshots.append(snapshot)
            report["resource_samples"].append(
                resource_sample(f"physics_step_{step}", wall_start, cpu_start)
            )
        elif args.native_frame_stride and step % args.native_frame_stride == 0:
            inspect_native_diffuse(step)

    stage.GetRootLayer().Export(str(stage_path))
    fabric_private_layer_path = args.output / "fabric_runtime_private.usda"
    runtime_stage.WriteToLayer(str(fabric_private_layer_path), True, False)
    report["snapshots"] = snapshots

    omnipvd_audit = {
        "requested": args.omnipvd,
        "ovd_file": None,
        "converted_stage": None,
        "candidate_attributes": [],
        "position_lifetime_readable": False,
        "velocities_readable": False,
        "active_count_readable": False,
    }
    if args.omnipvd:
        # Detaching closes the PhysX scene and forces the OVD writer to finish
        # the recording.  Keep Kit alive afterwards so the shipped Python PVD
        # importer can convert and inspect the recording in the same process.
        simulation.detach_stage()
        simulation = None
        simulation_app.update()

        ovd_files = sorted(
            (
                path
                for path in omnipvd_recording_directory.glob("*.ovd")
                if not path.name.endswith("tmp.ovd")
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not ovd_files:
            raise RuntimeError(
                f"OmniPVD recording produced no finalized OVD file in {omnipvd_recording_directory}"
            )
        ovd_path = ovd_files[-1]
        omnipvd_audit["ovd_file"] = str(ovd_path.resolve())
        omnipvd_audit["ovd_size_bytes"] = ovd_path.stat().st_size

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.physx.pvd", True)
        simulation_app.update()
        from omni.physxpvd.scripts.extension import get_physx_pvd_interface

        omnipvd_usd_directory = args.output / "omnipvd_usd"
        omnipvd_usd_directory.mkdir(exist_ok=True)
        conversion_result = get_physx_pvd_interface().ovd_to_usd(
            str(ovd_path.resolve()),
            omnipvd_usd_directory.resolve().as_posix() + "/",
            0,  # Y up
            1,  # USDA output for an auditable, non-proprietary cache
        )
        simulation_app.update()
        omnipvd_audit["conversion_result"] = (
            int(conversion_result)
            if isinstance(conversion_result, (int, np.integer))
            else str(conversion_result)
        )

        omnipvd_stage_path = omnipvd_usd_directory / "stage.usda"
        if not omnipvd_stage_path.is_file():
            raise RuntimeError(
                f"OmniPVD conversion did not create {omnipvd_stage_path}"
            )
        omnipvd_audit["converted_stage"] = str(omnipvd_stage_path.resolve())
        omnipvd_stage = Usd.Stage.Open(str(omnipvd_stage_path.resolve()))
        if not omnipvd_stage:
            raise RuntimeError(f"Unable to open converted OmniPVD stage: {omnipvd_stage_path}")

        omnipvd_arrays = {}
        for omnipvd_prim in omnipvd_stage.Traverse():
            for omnipvd_attribute in omnipvd_prim.GetAttributes():
                attribute_name = str(omnipvd_attribute.GetName())
                normalized_name = "".join(
                    character.lower() for character in attribute_name if character.isalnum()
                )
                if not any(
                    token in normalized_name
                    for token in (
                        "diffusepositionlifetime",
                        "diffusevelocit",
                        "nbactivediffuse",
                        "maxdiffuseparticle",
                    )
                ):
                    continue

                time_samples = list(omnipvd_attribute.GetTimeSamples())
                sample_times = time_samples if time_samples else [None]
                attribute_record = {
                    "prim": str(omnipvd_prim.GetPath()),
                    "name": attribute_name,
                    "type_name": str(omnipvd_attribute.GetTypeName()),
                    "time_samples": time_samples,
                    "samples": [],
                }
                for sample_index, sample_time in enumerate(sample_times):
                    value = omnipvd_attribute.Get(
                        Usd.TimeCode(sample_time) if sample_time is not None else Usd.TimeCode.Default()
                    )
                    sample_record = {
                        "time": sample_time,
                        "readable": value is not None,
                        "shape": None,
                        "dtype": None,
                    }
                    if value is not None:
                        try:
                            array = np.asarray(value)
                            sample_record["shape"] = list(array.shape)
                            sample_record["dtype"] = str(array.dtype)
                            if array.size and np.issubdtype(array.dtype, np.number):
                                sample_record["minimum"] = float(np.nanmin(array))
                                sample_record["maximum"] = float(np.nanmax(array))
                            if array.ndim >= 1 and array.dtype != object:
                                key = "omnipvd_" + "".join(
                                    character if character.isalnum() else "_"
                                    for character in (
                                        str(omnipvd_prim.GetPath())
                                        + "_"
                                        + attribute_name
                                        + f"_{sample_index:04d}"
                                    )
                                )
                                omnipvd_arrays[key] = np.ascontiguousarray(array)
                                sample_record["npz_key"] = key
                        except Exception as conversion_error:
                            sample_record["conversion_error"] = (
                                f"{type(conversion_error).__name__}: {conversion_error}"
                            )
                    attribute_record["samples"].append(sample_record)

                has_nonempty_array = any(
                    sample["readable"]
                    and sample["shape"] is not None
                    and len(sample["shape"]) >= 1
                    and sample["shape"][0] > 0
                    for sample in attribute_record["samples"]
                )
                has_readable_scalar = any(
                    sample["readable"] and sample["shape"] == []
                    for sample in attribute_record["samples"]
                )
                if "diffusepositionlifetime" in normalized_name and has_nonempty_array:
                    omnipvd_audit["position_lifetime_readable"] = True
                if "diffusevelocit" in normalized_name and has_nonempty_array:
                    omnipvd_audit["velocities_readable"] = True
                if "nbactivediffuse" in normalized_name and has_readable_scalar:
                    omnipvd_audit["active_count_readable"] = True
                omnipvd_audit["candidate_attributes"].append(attribute_record)

        if omnipvd_arrays:
            atomic_npz(args.output / "omnipvd_diffuse_samples.npz", **omnipvd_arrays)
        omnipvd_audit["sample_arrays_written"] = len(omnipvd_arrays)
        settings.set(physx_bindings.SETTING_OMNIPVD_ENABLED, False)
        simulation_app.update()

    report["omnipvd_audit"] = omnipvd_audit

    all_prim_paths = {
        prim["path"] for snapshot in snapshots for prim in snapshot["prims"]
    }
    diffuse_prim_paths = sorted(path for path in all_prim_paths if "diffuse" in path.lower())
    readable_candidates = [
        candidate
        for snapshot in snapshots
        for candidate in snapshot["readable_non_primary_candidates"]
        if "diffuse" in candidate["prim"].lower()
        or "diffuse" in candidate["attribute"].lower()
    ]
    generated_count = max(
        (candidate["array_length"] for candidate in readable_candidates), default=0
    )
    interface_names = report["python_interfaces"]
    exposed_interface_names = sorted(
        {
            name
            for names in interface_names.values()
            for name in names
            if "diffuse" in name.lower()
        }
    )

    # Generation is only proven by dynamic diffuse data, not by successful API
    # authoring.  A hidden /DiffuseParticles prim is acceptable if its arrays
    # can be read.  API application by itself leaves the run inconclusive.
    generated = generated_count > 0 and any(
        snapshot["diffuse_data"]["positions_finite"] is True for snapshot in snapshots
    )
    positions_readable = any(
        candidate["array_length"] > 0
        and candidate["attribute"].lower() in ("points", "positions")
        for candidate in readable_candidates
    )
    velocities_readable = any(
        candidate["array_length"] > 0 and "velocit" in candidate["attribute"].lower()
        for candidate in readable_candidates
    )
    lifetime_readable = any(
        candidate["array_length"] > 0
        and any(token in candidate["attribute"].lower() for token in ("life", "lifetime"))
        for candidate in readable_candidates
    )
    type_readable = any(
        candidate["array_length"] > 0
        and any(token in candidate["attribute"].lower() for token in ("type", "phase"))
        for candidate in readable_candidates
    )
    # This gate is intentionally stricter than the original position-only
    # discovery probe.  The downstream whitewater cache needs the native
    # PxParticleAndDiffuseBuffer contract: xyz + remaining lifetime, velocity,
    # and the active element count.  A render/display points array is not a
    # substitute for that contract.
    omnipvd_complete_diffuse_readback = bool(
        omnipvd_audit["position_lifetime_readable"]
        and omnipvd_audit["velocities_readable"]
        and omnipvd_audit["active_count_readable"]
    )
    native_nonempty_samples = [sample for sample in native_samples if sample["active_count"] > 0]
    native_complete_diffuse_readback = bool(
        args.native_diffuse_bridge
        and native_nonempty_samples
        and all(sample["all_structural_checks_passed"] for sample in native_samples)
        and any(sample["checks"]["velocity_xyz_not_all_zero"] for sample in native_nonempty_samples)
    )
    native_label_recovery_complete = bool(
        args.recover_native_labels
        and native_nonempty_samples
        and all(
            sample["checks"].get("label_histogram_conserves_active_count", False)
            and sample["checks"].get("neighbor_counts_in_physx_range", False)
            and sample["checks"].get("full_diffuse_advection_enabled", False)
            for sample in native_nonempty_samples
        )
    )
    required_readback_proven = bool(
        native_complete_diffuse_readback
        and (not args.recover_native_labels or native_label_recovery_complete)
        if args.native_diffuse_bridge
        else (generated and args.omnipvd and omnipvd_complete_diffuse_readback)
    )
    report["result"] = {
        "diffuse_prim_paths": diffuse_prim_paths,
        "readable_diffuse_candidates": readable_candidates,
        "generated_diffuse_count_lower_bound": generated_count,
        "native_diffuse_generation_proven": generated,
        "positions_readable": positions_readable,
        "velocities_readable": velocities_readable,
        "lifetime_readable": lifetime_readable,
        "type_or_phase_readable": type_readable,
        "omnipvd_complete_diffuse_readback": omnipvd_complete_diffuse_readback,
        "native_complete_diffuse_readback": native_complete_diffuse_readback,
        "native_label_recovery_complete": native_label_recovery_complete,
        "native_diffuse_samples": native_samples,
        "explicit_diffuse_interface_names": exposed_interface_names,
        "primary_pointinstancer_remained_primary_only": all(
            snapshot["primary_position_count"] == len(initial_positions) for snapshot in snapshots
        ),
        "diffuse_samples_written": len(list((args.output / "diffuse_samples").glob("*.npz")))
        if (args.output / "diffuse_samples").is_dir()
        else 0,
        "usdrt_runtime_attribute_names": sorted(
            {
                attribute["name"]
                for snapshot in snapshots
                for attribute in snapshot["usdrt_diffuse"]["attributes"]
            }
        ),
        "usdrt_runtime_arrays": {
            name: metadata
            for snapshot in snapshots
            for name, metadata in snapshot["usdrt_diffuse"]["arrays"].items()
        },
        "usdrt_related_property_names": {
            related_path: sorted(
                {
                    name
                    for snapshot in snapshots
                    for name in snapshot["usdrt_related_prims"][related_path]["property_names"]
                }
            )
            for related_path in (
                "/World/ParticleSystem",
                "/World/WaterParticles",
                "/World/ParticleSystem/DiffuseParticles",
            )
        },
        "fabric_private_layer": str(fabric_private_layer_path.resolve()),
    }
    report["status"] = "passed" if required_readback_proven else "inconclusive"
    report["valid"] = required_readback_proven
    report["inconclusive"] = not report["valid"]
    report["completed_utc"] = utc_now()
    report["elapsed_seconds"] = time.perf_counter() - wall_start
    report["process_cpu_seconds"] = time.process_time() - cpu_start
    atomic_json(report_path, report)
    if not report["valid"]:
        atomic_json(
            obsolete_path,
            {
                "schema": 1,
                "product": "obsolete_marker",
                "created_utc": utc_now(),
                "reason": (
                    "Complete PhysX diffuse readback was not proven: the gate requires "
                    "position+lifetime, velocity, and active count"
                ),
                "replacement": None,
                "preserve_for_diagnostics": True,
                "report": str(report_path.resolve()),
            },
        )

except Exception as exc:
    report["status"] = "failed"
    report["valid"] = False
    report["inconclusive"] = False
    report["failure"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    report["completed_utc"] = utc_now()
    report["elapsed_seconds"] = time.perf_counter() - wall_start
    report["process_cpu_seconds"] = time.process_time() - cpu_start
    atomic_json(report_path, report)
    atomic_json(
        obsolete_path,
        {
            "schema": 1,
            "product": "obsolete_marker",
            "created_utc": utc_now(),
            "reason": f"PhysX diffuse micro gate failed: {type(exc).__name__}: {exc}",
            "replacement": None,
            "preserve_for_diagnostics": True,
            "report": str(report_path.resolve()),
        },
    )
    traceback.print_exc()
    sys.exit_code = 1

finally:
    if simulation is not None:
        try:
            simulation.detach_stage()
        except Exception:
            pass
    if simulation_app is not None:
        simulation_app.close()

if report.get("status") == "failed":
    raise SystemExit(1)
