"""Run identity-stable PhysX PBD water against the authored mountain scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mountain-usd", type=Path, default=Path(r"Y:\scenes\mountain\mountain.usdc"))
parser.add_argument("--frames", type=int, default=45)
parser.add_argument("--output-fps", type=int, default=30)
parser.add_argument("--source-fps", type=int, default=120)
parser.add_argument("--physics-fps", type=int, default=240)
parser.add_argument("--spacing", type=float, default=0.03)
parser.add_argument("--solver-iterations", type=int, default=8)
parser.add_argument("--source-speed", type=float, default=1.2)
parser.add_argument("--maximum-speed", type=float, default=8.0)
parser.add_argument("--continuous-emitter", action="store_true")
parser.add_argument("--emitter-interval-steps", type=int, default=6)
parser.add_argument("--preroll-seconds", type=float, default=0.0)
parser.add_argument(
    "--pour-source",
    type=Path,
    help="Versioned primary-liquid PourSource JSON; replaces the legacy waterfall emitter.",
)
parser.add_argument(
    "--wall-target",
    type=Path,
    help="Measured mountain wall target JSON required by --pour-source.",
)
parser.add_argument(
    "--wall-collision-adapter",
    type=Path,
    help="Audited Isaac-space OBJ exported from the rendered Blender wall.",
)
parser.add_argument(
    "--wall-collision-selection",
    type=Path,
    help="Selection/provenance JSON paired with --wall-collision-adapter.",
)
parser.add_argument(
    "--ground-collision-adapter",
    type=Path,
    help="Audited Isaac-space OBJ sampled from the rendered Blender ground.",
)
parser.add_argument(
    "--ground-collision-selection",
    type=Path,
    help="Selection/provenance JSON paired with --ground-collision-adapter.",
)
parser.add_argument(
    "--ground-collision-audit",
    type=Path,
    help="Independent valid audit JSON for --ground-collision-adapter.",
)
args = parser.parse_args()

if (args.pour_source is None) != (args.wall_target is None):
    raise ValueError("--pour-source and --wall-target must be supplied together")
if args.pour_source is not None and args.continuous_emitter:
    raise ValueError("--pour-source already defines emission; omit --continuous-emitter")
if (args.wall_collision_adapter is None) != (args.wall_collision_selection is None):
    raise ValueError("Wall collision adapter and selection must be supplied together")
if args.wall_collision_adapter is not None and args.wall_target is None:
    raise ValueError("A wall collision adapter requires the independently measured wall target")
ground_adapter_inputs = (
    args.ground_collision_adapter,
    args.ground_collision_selection,
    args.ground_collision_audit,
)
if any(value is not None for value in ground_adapter_inputs) and not all(
    value is not None for value in ground_adapter_inputs
):
    raise ValueError("Ground collision adapter, selection, and audit must be supplied together")
if args.ground_collision_adapter is not None and args.wall_collision_adapter is None:
    raise ValueError("A Blender ground adapter requires the registered Blender wall adapter")

if args.output.exists() and any(args.output.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
if not args.mountain_usd.is_file():
    raise FileNotFoundError(args.mountain_usd)
if args.frames < 2 or not 0.02 <= args.spacing <= 0.05:
    raise ValueError("Preview requires frames >= 2 and spacing within 0.02..0.05 m")
for rate in (args.output_fps, args.source_fps):
    if rate <= 0 or args.physics_fps % rate:
        raise ValueError("output/source FPS must divide physics FPS")
if args.solver_iterations < 1 or min(args.source_speed, args.maximum_speed) <= 0.0:
    raise ValueError("Solver iterations and speeds must be positive")
if args.emitter_interval_steps < 1:
    raise ValueError("Emitter interval must be at least one physics step")
if args.preroll_seconds < 0.0:
    raise ValueError("Preroll seconds cannot be negative")

pour_source = None
wall_target_payload = None
wall_target_points = None
wall_target_normals = None
if args.pour_source is not None:
    if not args.pour_source.is_file() or not args.wall_target.is_file():
        raise FileNotFoundError("Pour source and measured wall target must both exist")
    shared_module_root = Path(__file__).resolve().parents[1] / "swamp_fluid"
    sys.path.insert(0, str(shared_module_root))
    from whitewater.pour_source import PourSource
    from whitewater.contact_episodes import MeasuredSurfaceContactTracker

    pour_source = PourSource.load(args.pour_source)
    wall_target_payload = json.loads(args.wall_target.read_text(encoding="utf-8"))
    if (
        wall_target_payload.get("schema") != 1
        or wall_target_payload.get("product") != "mountain_pour_wall_target"
    ):
        raise ValueError("Unsupported measured mountain wall target")
    accepted_wall_rays = [
        row for row in wall_target_payload.get("rays", []) if row.get("accepted") is True
    ]
    if not accepted_wall_rays:
        raise ValueError("Measured wall target contains no accepted samples")
    wall_target_points = np.asarray(
        [row["location_isaac"] for row in accepted_wall_rays], dtype=np.float64
    )
    wall_target_normals = np.asarray(
        [row["normal_isaac"] for row in accepted_wall_rays], dtype=np.float64
    )
    if not np.isfinite(wall_target_points).all() or not np.isfinite(wall_target_normals).all():
        raise ValueError("Measured wall target contains non-finite samples")

continuous_mode = args.continuous_emitter or pour_source is not None

args.output.mkdir(parents=True, exist_ok=True)
particles_directory = args.output / "particles"
source_directory = args.output / "whitewater_source"
particles_directory.mkdir()
source_directory.mkdir()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import carb
import omni.physx.bindings._physx as physx_settings_bindings
import omni.usd
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Vt


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_obj_triangles(path):
    vertices = []
    faces = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"Incomplete OBJ vertex at line {line_number}")
            vertices.append(tuple(map(float, fields[1:4])))
        elif fields[0] == "f":
            indices = [int(token.split("/", 1)[0]) - 1 for token in fields[1:]]
            if len(indices) != 3:
                raise ValueError("Wall collision adapter must be fully triangulated")
            faces.append(tuple(indices))
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(vertices)
        or not len(faces)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("Wall collision adapter OBJ is invalid")
    return vertices, faces


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_ply(path, positions):
    points = np.ascontiguousarray(positions, dtype="<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment mountain waterfall PhysX PBD positions\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        points.tofile(stream)


def lattice(minimum, maximum, spacing, velocity):
    axes = [
        np.arange(minimum[index], maximum[index] + 0.25 * spacing, spacing, dtype=np.float32)
        for index in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    values = np.repeat(np.asarray(velocity, dtype=np.float32)[None, :], len(grid), axis=0)
    return grid, values


def add_static_box(stage, path, centre, half_extents, rotate_y_degrees=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(2.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    if rotate_y_degrees:
        xform.AddRotateYOp().Set(float(rotate_y_degrees))
    xform.AddScaleOp().Set(Gf.Vec3f(*map(float, half_extents)))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    return cube


def add_static_triangle_mesh(stage, path, vertices, faces, contact_offset):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr().Set(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32))
    )
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    mesh.CreateExtentAttr().Set(
        Vt.Vec3fArray(
            [Gf.Vec3f(*map(float, minimum)), Gf.Vec3f(*map(float, maximum))]
        )
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr().Set(
        float(contact_offset)
    )
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
    return mesh


wall_collision_vertices = None
wall_collision_faces = None
wall_collision_selection = None
if args.wall_collision_adapter is not None:
    if not args.wall_collision_adapter.is_file() or not args.wall_collision_selection.is_file():
        raise FileNotFoundError("Wall collision adapter and selection must both exist")
    wall_collision_selection = json.loads(
        args.wall_collision_selection.read_text(encoding="utf-8")
    )
    if wall_collision_selection.get("product") != "mountain_blender_wall_collision_selection":
        raise ValueError("Unexpected Blender wall collision selection product")
    selected_mesh = wall_collision_selection.get("selected_mesh", {})
    if selected_mesh.get("sha256") != sha256_file(args.wall_collision_adapter):
        raise ValueError("Blender wall collision adapter hash does not match selection")
    if (
        wall_collision_selection.get("source", {}).get("wall_target_sha256")
        != sha256_file(args.wall_target)
    ):
        raise ValueError("Blender wall collision selection targets a different wall report")
    wall_collision_vertices, wall_collision_faces = load_obj_triangles(
        args.wall_collision_adapter
    )
    if (
        len(wall_collision_vertices) != int(selected_mesh.get("vertex_count", -1))
        or len(wall_collision_faces) != int(selected_mesh.get("triangle_count", -1))
    ):
        raise ValueError("Blender wall collision adapter counts do not match selection")

ground_collision_vertices = None
ground_collision_faces = None
ground_collision_selection = None
ground_collision_audit = None
if args.ground_collision_adapter is not None:
    if not all(path.is_file() for path in ground_adapter_inputs):
        raise FileNotFoundError("Ground collision adapter, selection, and audit must all exist")
    ground_collision_selection = json.loads(
        args.ground_collision_selection.read_text(encoding="utf-8")
    )
    if (
        ground_collision_selection.get("product")
        != "mountain_blender_ground_collision_selection"
    ):
        raise ValueError("Unexpected Blender ground collision selection product")
    selected_ground = ground_collision_selection.get("selected_mesh", {})
    if selected_ground.get("sha256") != sha256_file(args.ground_collision_adapter):
        raise ValueError("Blender ground collision adapter hash does not match selection")
    source_blend = Path(ground_collision_selection.get("source", {}).get("blend_path", ""))
    if (
        not source_blend.is_file()
        or ground_collision_selection.get("source", {}).get("blend_sha256")
        != sha256_file(source_blend)
    ):
        raise ValueError("Blender ground adapter source scene hash does not match")
    ground_collision_vertices, ground_collision_faces = load_obj_triangles(
        args.ground_collision_adapter
    )
    if (
        len(ground_collision_vertices) != int(selected_ground.get("vertex_count", -1))
        or len(ground_collision_faces) != int(selected_ground.get("triangle_count", -1))
    ):
        raise ValueError("Blender ground collision adapter counts do not match selection")
    ground_collision_audit = json.loads(args.ground_collision_audit.read_text(encoding="utf-8"))
    if (
        ground_collision_audit.get("product")
        != "mountain_blender_ground_collision_audit"
        or ground_collision_audit.get("valid") is not True
        or ground_collision_audit.get("mesh_sha256")
        != sha256_file(args.ground_collision_adapter)
    ):
        raise ValueError("Blender ground collision adapter has no matching valid audit")


# Isaac coordinates are Y-up.  Selected Blender site mapping is (x, y, z) -> (x, z, -y).
# The catch-basin bottom surface is y=-0.525.  Start the first pool layer at
# the PBD particle-contact equilibrium instead of dropping the whole pool
# onto the support during the recorded timeline.
pool_contact_equilibrium_y = -0.525 + (0.5 * args.spacing / 0.6)
pool_points, pool_velocities = lattice(
    (2.72, pool_contact_equilibrium_y, 1.12),
    (4.26, -0.28, 2.26),
    args.spacing,
    (0.0, 0.0, 0.0),
)
# The scanned cliff has no closed natural pool, so this is an explicit artist
# footprint.  An ellipse avoids turning the invisible rectangular catch basin
# into a visible rectangular water brick.
pool_centre_xz = np.asarray((3.49, 1.69), dtype=np.float32)
pool_radii_xz = np.asarray((0.79, 0.57), dtype=np.float32)
pool_planar = (pool_points[:, (0, 2)] - pool_centre_xz) / pool_radii_xz
pool_mask = np.sum(pool_planar * pool_planar, axis=1) <= 1.0
pool_points = np.ascontiguousarray(pool_points[pool_mask])
pool_velocities = np.ascontiguousarray(pool_velocities[pool_mask])
reservoir_points, reservoir_velocities = lattice(
    # X is aligned to the lip lattice anchored at 3.72 m.  A separate
    # 4.06 m anchor created 1 cm near-duplicate interface pairs at 3 cm spacing.
    (4.08, 1.13, 1.48), (4.68, 1.34, 1.92), args.spacing,
    (-args.source_speed, 0.0, 0.0),
)
lip_points, lip_velocities = lattice(
    (3.72, 1.04, 1.48), (4.18, 1.19, 1.92), args.spacing,
    (-args.source_speed, -0.12, 0.0),
)
sheet_points, sheet_velocities = lattice(
    (3.72, -0.25, 1.48), (3.84, 1.08, 1.92), args.spacing,
    (-0.35 * args.source_speed, -0.65, 0.0),
)
region_chunks = (
    (("pool", pool_points, pool_velocities),)
    if continuous_mode
    else (
        ("pool", pool_points, pool_velocities),
        ("reservoir", reservoir_points, reservoir_velocities),
        ("lip_transition", lip_points, lip_velocities),
        ("falling_sheet", sheet_points, sheet_velocities),
    )
)
combined_positions = np.concatenate([chunk[1] for chunk in region_chunks])
combined_velocities = np.concatenate([chunk[2] for chunk in region_chunks])
combined_region_labels = np.concatenate(
    [np.full(len(chunk[1]), index, dtype=np.int16) for index, chunk in enumerate(region_chunks)]
)
# Adjacent artist-authored lattice volumes intentionally touch, but must not
# contribute coincident PBD particles.  A 0.1-spacing key only merges points
# that are effectively identical; it cannot collapse legitimate neighbours.
spatial_keys = np.rint(combined_positions / (0.1 * args.spacing)).astype(np.int64)
_, unique_indices = np.unique(spatial_keys, axis=0, return_index=True)
unique_indices.sort()
duplicate_initial_particles_removed = int(len(combined_positions) - len(unique_indices))
initial_positions = np.ascontiguousarray(combined_positions[unique_indices], dtype=np.float32)
initial_velocities = np.ascontiguousarray(combined_velocities[unique_indices], dtype=np.float32)
initial_region_labels = np.ascontiguousarray(combined_region_labels[unique_indices])
initial_regions = {
    chunk[0]: int(np.count_nonzero(initial_region_labels == index))
    for index, chunk in enumerate(region_chunks)
}

context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
environment = UsdGeom.Xform.Define(stage, "/World/Environment")
environment.GetPrim().GetReferences().AddReference(str(args.mountain_usd.resolve()))
UsdGeom.Xformable(environment.GetPrim()).AddRotateXOp().Set(-90.0)

collision_meshes = []
disabled_referenced_collision_meshes = []
for prim in Usd.PrimRange(environment.GetPrim()):
    if prim.IsA(UsdGeom.Mesh):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        if wall_collision_vertices is not None:
            collision_api.CreateCollisionEnabledAttr().Set(False)
            disabled_referenced_collision_meshes.append(str(prim.GetPath()))
        else:
            collision_api.CreateCollisionEnabledAttr().Set(True)
            # In Isaac Sim 6.0 the mesh approximation schema is owned by
            # UsdPhysics, while PhysxCollisionAPI carries PhysX-specific tuning.
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
            PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr().Set(
                max(0.006, 0.4 * args.spacing)
            )
            collision_meshes.append(str(prim.GetPath()))
if wall_collision_vertices is not None:
    wall_adapter_path = "/World/BlenderWallCollisionAdapter"
    add_static_triangle_mesh(
        stage,
        wall_adapter_path,
        wall_collision_vertices,
        wall_collision_faces,
        max(0.006, 0.4 * args.spacing),
    )
    collision_meshes.append(wall_adapter_path)
if ground_collision_vertices is not None:
    ground_adapter_path = "/World/BlenderGroundCollisionAdapter"
    add_static_triangle_mesh(
        stage,
        ground_adapter_path,
        ground_collision_vertices,
        ground_collision_faces,
        max(0.006, 0.4 * args.spacing),
    )
    collision_meshes.append(ground_adapter_path)
if not collision_meshes:
    raise RuntimeError("Referenced mountain scene contains no mesh collision geometry")

# An explicitly authored, invisible catch basin contains the shallow plunge pool.
add_static_box(stage, "/World/CatchBasin/Bottom", (3.49, -0.57, 1.69), (0.92, 0.045, 0.68))
catch_wall_segments = 32
catch_wall_centre_xz = np.asarray((3.49, 1.69), dtype=np.float64)
catch_wall_radii_xz = np.asarray((0.87, 0.65), dtype=np.float64)
for segment_index in range(catch_wall_segments):
    first_angle = 2.0 * math.pi * segment_index / catch_wall_segments
    second_angle = 2.0 * math.pi * (segment_index + 1) / catch_wall_segments
    first = catch_wall_centre_xz + catch_wall_radii_xz * np.asarray(
        (math.cos(first_angle), math.sin(first_angle))
    )
    second = catch_wall_centre_xz + catch_wall_radii_xz * np.asarray(
        (math.cos(second_angle), math.sin(second_angle))
    )
    midpoint = 0.5 * (first + second)
    chord = second - first
    chord_length = float(np.linalg.norm(chord))
    rotate_y = math.degrees(math.atan2(-float(chord[1]), float(chord[0])))
    add_static_box(
        stage,
        f"/World/CatchBasin/Wall_{segment_index:02d}",
        (float(midpoint[0]), -0.34, float(midpoint[1])),
        (0.5 * chord_length + 0.015, 0.28, 0.035),
        rotate_y_degrees=rotate_y,
    )

scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
physx_scene.CreateGpuMaxParticleContactsAttr().Set(1_000_000)
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
    solver_position_iterations=args.solver_iterations,
    max_neighborhood=96,
    neighborhood_scale=1.01,
    max_velocity=args.maximum_speed,
)
material_path = Sdf.Path("/World/Looks/WaterPhysics")
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

# Isaac Sim exposes no primary-fluid emitter schema.  Continuous mode uses a
# fixed set of pre-authored, disabled particle batches and enables one source
# layer at a deterministic cadence.  Existing IDs are never recycled or moved.
total_steps = args.frames * (args.physics_fps // args.output_fps)
preroll_steps = int(round(args.preroll_seconds * args.physics_fps))
if not math.isclose(preroll_steps / args.physics_fps, args.preroll_seconds, abs_tol=1.0e-9):
    raise ValueError("Preroll seconds must align to an integer physics step")
emitter_batch_records = []
emitter_batch_particle_counts = []
global_end_seconds = (preroll_steps + total_steps) / args.physics_fps
if pour_source is not None:
    if pour_source.start_seconds > 0.0 or pour_source.stop_seconds < global_end_seconds:
        raise ValueError(
            "PourSource timing must cover the complete preroll plus recorded gate"
        )
    emitter_birth_steps = pour_source.schedule_for_spacing(
        args.physics_fps, args.spacing, global_end_seconds
    )
    pour_schedule_audit = pour_source.schedule_audit(
        args.physics_fps, args.spacing, global_end_seconds
    )
    emitter_layer_speed = pour_source.flow.speed_m_s
else:
    emitter_birth_steps = tuple(
        range(
            args.emitter_interval_steps,
            preroll_steps + total_steps + 1,
            args.emitter_interval_steps,
        )
    )
    pour_schedule_audit = None
    emitter_layer_speed = (
        args.spacing * args.physics_fps / args.emitter_interval_steps
        if args.continuous_emitter
        else 0.65
    )
    legacy_emitter_points, legacy_emitter_velocities = lattice(
        # Historical waterfall source retained only for reproducibility.
        (3.72, 1.08, 1.48),
        (3.84, 1.08, 1.92),
        args.spacing,
        (-0.15 * emitter_layer_speed, -emitter_layer_speed, 0.0),
    )
next_particle_id = len(initial_positions)
if continuous_mode:
    for batch_index, birth_step in enumerate(emitter_birth_steps):
        if pour_source is not None:
            authored_batch = pour_source.particle_batch(
                batch_index, birth_step, args.physics_fps, args.spacing
            )
            emitter_batch_points = authored_batch.positions
            emitter_batch_velocities = authored_batch.velocities
        else:
            emitter_batch_points = legacy_emitter_points
            emitter_batch_velocities = legacy_emitter_velocities
        batch_path = Sdf.Path(f"/World/EmitterBatch_{batch_index:04d}")
        batch_prim = particleUtils.add_physx_particleset_pointinstancer(
            stage,
            batch_path,
            Vt.Vec3fArray.FromNumpy(emitter_batch_points),
            Vt.Vec3fArray.FromNumpy(emitter_batch_velocities),
            particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=1000.0 * args.spacing**3,
            density=1000.0,
        )
        batch_prim.CreateAttribute(
            "physxParticle:maxParticles", Sdf.ValueTypeNames.Int
        ).Set(len(emitter_batch_points))
        enabled_attribute = batch_prim.GetAttribute("physxParticle:particleEnabled")
        if not enabled_attribute:
            raise RuntimeError("Emitter batch does not expose physxParticle:particleEnabled")
        enabled_attribute.Set(False)
        batch_instancer = UsdGeom.PointInstancer.Get(stage, batch_path)
        batch_prototype = UsdGeom.Imageable.Get(
            stage, batch_path.AppendChild("particlePrototype0")
        )
        if batch_prototype:
            batch_prototype.MakeInvisible()
        batch_ids = np.arange(
            next_particle_id,
            next_particle_id + len(emitter_batch_points),
            dtype="<i8",
        )
        next_particle_id += len(emitter_batch_points)
        emitter_batch_particle_counts.append(int(len(emitter_batch_points)))
        emitter_batch_records.append(
            {
                "birth_step": int(birth_step),
                "enabled_attribute": enabled_attribute,
                "instancer": batch_instancer,
                "particle_ids": batch_ids,
            }
        )

settings = carb.settings.get_settings()
settings.set(physx_settings_bindings.SETTING_UPDATE_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)

stage_path = args.output / "mountain_waterfall_physx.usda"
stage.GetRootLayer().Export(str(stage_path))
simulation = get_physx_simulation_interface()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
simulation.attach_stage(stage_id)

particle_ids = np.arange(len(initial_positions), dtype="<i8")
particle_ids_sha256 = hashlib.sha256(particle_ids.tobytes()).hexdigest()
source_manifest_path = source_directory / "manifest.json"
source_stride = args.physics_fps // args.source_fps
expected_steps = list(range(0, total_steps + 1, source_stride))
source_manifest = {
    "schema": 2 if continuous_mode else 1,
    "producer": {"script": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve())},
    "created_utc": utc_now(),
    "source": (
        "raw PhysX PBD PointInstancer state; append-only births; versioned pour source"
        if pour_source is not None
        else "raw PhysX PBD PointInstancer state; append-only particle births; waterfall emitter"
        if args.continuous_emitter
        else "raw PhysX PBD PointInstancer state; fixed particle identities; waterfall slug"
    ),
    "coordinates": "Isaac/USD world XYZ in metres",
    "particle_count": len(initial_positions),
    "initial_particle_count": len(initial_positions),
    "maximum_particle_count": next_particle_id,
    "particle_ids": (
        {"storage": "per_sample", "dtype": "int64", "births": "append_only"}
        if continuous_mode
        else {"storage": "implicit array index", "dtype": "int64", "sha256": particle_ids_sha256}
    ),
    "arrays": {
        "positions": {
            "dtype": "float32",
            "shape": ["active_particles", 3]
            if continuous_mode
            else [len(initial_positions), 3],
        },
        "velocities": {
            "dtype": "float32",
            "shape": ["active_particles", 3]
            if continuous_mode
            else [len(initial_positions), 3],
        },
        "particle_ids": {
            "dtype": "int64",
            "shape": ["active_particles"]
            if continuous_mode
            else [len(initial_positions)],
        },
        "sphere_transform": {"dtype": "float64", "shape": [4, 4]},
        "sphere_linear_velocity": {"dtype": "float32", "shape": [3]},
        "sphere_angular_velocity": {"dtype": "float32", "shape": [3]},
    },
    "sampling": {
        "physics_fps": args.physics_fps,
        "output_fps": args.output_fps,
        "impact_fps": args.source_fps,
        "impact_seconds": args.frames / args.output_fps,
        "impact_end_physics_step": total_steps,
        "tail_fps": args.output_fps,
        "total_physics_steps": total_steps,
        "expected_physics_steps": expected_steps,
    },
    "emitter": {
        "continuous": continuous_mode,
        "method": "preauthored_disabled_batches_enabled_at_birth_step",
        "interval_physics_steps": (
            args.emitter_interval_steps if pour_source is None else None
        ),
        "batch_particles": (
            emitter_batch_particle_counts[0]
            if emitter_batch_particle_counts
            and min(emitter_batch_particle_counts) == max(emitter_batch_particle_counts)
            else None
        ),
        "minimum_batch_particles": min(emitter_batch_particle_counts, default=0),
        "maximum_batch_particles": max(emitter_batch_particle_counts, default=0),
        "layer_speed_m_s": emitter_layer_speed,
        "planned_batches": len(emitter_batch_records),
        "preroll_physics_steps": preroll_steps,
        "preroll_seconds": args.preroll_seconds,
        "pour_source": (
            {
                "path": str(args.pour_source.resolve()),
                "file_sha256": sha256_file(args.pour_source),
                "configuration_sha256": pour_source.configuration_sha256(),
                "schedule_audit": pour_schedule_audit,
            }
            if pour_source is not None
            else None
        ),
    },
    "state": {"complete": False, "completed_samples": 0, "expected_samples": len(expected_steps), "parent_run_valid": None, "updated_utc": utc_now()},
    "samples": [],
}
atomic_json(source_manifest_path, source_manifest)

identity = np.eye(4, dtype="<f8")
zero3 = np.zeros(3, dtype="<f4")
metrics = []
maximum_outside = 0
maximum_below_collision_support = 0
maximum_speed = 0.0
active_instancers = [instancer]
active_particle_id_chunks = [particle_ids]
enabled_emitter_batches = 0
if wall_target_payload is not None:
    measured_wall = wall_target_payload["measured_wall"]
    contact_tracker = MeasuredSurfaceContactTracker(
        points=wall_target_points,
        normals=wall_target_normals,
        aabb_minimum=measured_wall["contact_gate_aabb_minimum_isaac"],
        aabb_maximum=measured_wall["contact_gate_aabb_maximum_isaac"],
        initial_particle_count=len(initial_positions),
        maximum_contact_distance=0.09,
        episode_entry_distance=0.11,
        minimum_incoming_normal_speed=0.20,
        minimum_cumulative_outward_change=0.30,
        maximum_episode_gap_steps=3,
    )
else:
    contact_tracker = None


def state_arrays():
    position_chunks = []
    velocity_chunks = []
    for active_instancer in active_instancers:
        position_chunks.append(
            np.ascontiguousarray(active_instancer.GetPositionsAttr().Get(), dtype="<f4")
        )
        velocity_chunks.append(
            np.ascontiguousarray(active_instancer.GetVelocitiesAttr().Get(), dtype="<f4")
        )
    positions = np.ascontiguousarray(np.concatenate(position_chunks), dtype="<f4")
    velocities = np.ascontiguousarray(np.concatenate(velocity_chunks), dtype="<f4")
    active_ids = np.ascontiguousarray(np.concatenate(active_particle_id_chunks), dtype="<i8")
    if positions.shape != velocities.shape or positions.shape != (len(active_ids), 3):
        raise RuntimeError(
            "PhysX particle state shape changed: "
            f"positions={positions.shape}, velocities={velocities.shape}, ids={active_ids.shape}"
        )
    if not continuous_mode and positions.shape != initial_positions.shape:
        raise RuntimeError("Fixed-particle PhysX state shape changed")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise RuntimeError("PhysX particle state contains non-finite values")
    return positions, velocities, active_ids


def audit_wall_contact(global_step, positions, velocities, active_ids):
    if contact_tracker is not None:
        contact_tracker.update(global_step, active_ids, positions, velocities)


def capture_source(step, sample_index, positions, velocities, active_ids):
    path = source_directory / f"source_{sample_index:06d}.npz"
    atomic_npz(
        path,
        schema=np.asarray(1, dtype="<i4"),
        sample_index=np.asarray(sample_index, dtype="<i4"),
        physics_step=np.asarray(step, dtype="<i8"),
        simulation_time=np.asarray(step / args.physics_fps, dtype="<f8"),
        positions=positions,
        velocities=velocities,
        particle_ids=active_ids,
        sphere_transform=identity,
        sphere_linear_velocity=zero3,
        sphere_angular_velocity=zero3,
    )
    source_manifest["samples"].append(
        {
            "sample_index": sample_index,
            "physics_step": step,
            "simulation_time": step / args.physics_fps,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
    source_manifest["state"].update(completed_samples=sample_index + 1, updated_utc=utc_now())
    atomic_json(source_manifest_path, source_manifest)


def enable_emitter_batches_due(global_step):
    global enabled_emitter_batches
    changed = False
    while (
        continuous_mode
        and enabled_emitter_batches < len(emitter_batch_records)
        and global_step == emitter_batch_records[enabled_emitter_batches]["birth_step"]
    ):
        record = emitter_batch_records[enabled_emitter_batches]
        record["enabled_attribute"].Set(True)
        active_instancers.append(record["instancer"])
        active_particle_id_chunks.append(record["particle_ids"])
        enabled_emitter_batches += 1
        changed = True
    if changed:
        simulation_app.update()


if contact_tracker is not None:
    contact_tracker.update(0, particle_ids, initial_positions, initial_velocities)


try:
    for preroll_step in range(1, preroll_steps + 1):
        simulation.simulate(
            1.0 / args.physics_fps,
            (preroll_step - preroll_steps - 1) / args.physics_fps,
        )
        simulation.fetch_results()
        simulation_app.update()
        enable_emitter_batches_due(preroll_step)
        positions, velocities, active_ids = state_arrays()
        audit_wall_contact(preroll_step, positions, velocities, active_ids)
    positions, velocities, active_ids = state_arrays()
    write_ply(particles_directory / "particles_0000.ply", positions)
    capture_source(0, 0, positions, velocities, active_ids)
    source_index = 1
    output_stride = args.physics_fps // args.output_fps
    for step in range(1, total_steps + 1):
        global_step = preroll_steps + step
        simulation.simulate(1.0 / args.physics_fps, (step - 1) / args.physics_fps)
        simulation.fetch_results()
        simulation_app.update()
        enable_emitter_batches_due(global_step)
        positions, velocities, active_ids = state_arrays()
        audit_wall_contact(global_step, positions, velocities, active_ids)
        if step % source_stride == 0:
            capture_source(step, source_index, positions, velocities, active_ids)
            source_index += 1
        if step % output_stride == 0:
            output_frame = step // output_stride
            write_ply(particles_directory / f"particles_{output_frame:04d}.ply", positions)
            outside = (
                (positions[:, 0] < 2.25) | (positions[:, 0] > 5.10)
                | (positions[:, 1] < -1.0) | (positions[:, 1] > 2.0)
                | (positions[:, 2] < 0.65) | (positions[:, 2] > 2.75)
            )
            speed = np.linalg.norm(velocities, axis=1)
            maximum_outside = max(maximum_outside, int(np.count_nonzero(outside)))
            below_collision_support = positions[:, 1] < -1.0
            maximum_below_collision_support = max(
                maximum_below_collision_support,
                int(np.count_nonzero(below_collision_support)),
            )
            maximum_speed = max(maximum_speed, float(speed.max(initial=0.0)))
            metrics.append(
                {
                    "output_frame": output_frame,
                    "simulation_time": step / args.physics_fps,
                    "position_minimum": positions.min(axis=0).astype(float).tolist(),
                    "position_maximum": positions.max(axis=0).astype(float).tolist(),
                    "maximum_speed": float(speed.max(initial=0.0)),
                    "outside_preview_domain": int(np.count_nonzero(outside)),
                    "below_collision_support": int(
                        np.count_nonzero(below_collision_support)
                    ),
                }
            )
            if output_frame % 10 == 0 or output_frame == args.frames:
                print(
                    f"[mountain-waterfall] frame={output_frame:03d}/{args.frames:03d} "
                    f"particles={len(positions)} outside={int(np.count_nonzero(outside))} "
                    f"vmax={float(speed.max(initial=0.0)):.3f}",
                    flush=True,
                )
finally:
    simulation.detach_stage()

source_complete = len(source_manifest["samples"]) == len(expected_steps)
outside_preview_fraction = maximum_outside / len(initial_positions)
below_collision_support_fraction = maximum_below_collision_support / len(initial_positions)
wall_contact_metrics = contact_tracker.metrics() if contact_tracker is not None else None
wall_contact_valid = (
    pour_source is None
    or (
        wall_contact_metrics["contact_unique_particles"] >= 12
        and wall_contact_metrics["contact_fraction_of_near_surface"] >= 0.10
    )
)
run_valid = (
    source_complete
    and outside_preview_fraction <= 0.02
    and below_collision_support_fraction <= 0.002
    and maximum_speed <= args.maximum_speed + 0.05
    and wall_contact_valid
)
source_manifest["state"].update(
    complete=source_complete,
    completed_samples=len(source_manifest["samples"]),
    parent_run_valid=run_valid,
    completed_utc=utc_now(),
    updated_utc=utc_now(),
)
atomic_json(source_manifest_path, source_manifest)
report = {
    "schema": 1,
    "product": (
        "mountain_wall_pour_physx_preview"
        if pour_source is not None
        else "mountain_waterfall_physx_preview"
    ),
    "created_utc": utc_now(),
    "valid": run_valid,
    "scene": str(args.mountain_usd.resolve()),
    "scene_sha256": sha256_file(args.mountain_usd),
    "physics": "PhysX GPU PBD fluid",
    "mode": (
        "configured_pour_against_geometry_registered_blender_wall_and_ground_with_artist_authored_catch_basin"
        if pour_source is not None and ground_collision_vertices is not None
        else "configured_pour_against_geometry_registered_blender_wall_with_artist_authored_catch_basin"
        if pour_source is not None and wall_collision_vertices is not None
        else "configured_pour_against_measured_real_wall_with_artist_authored_catch_basin"
        if pour_source is not None
        else "continuous_batched_waterfall_with_artist_authored_catch_basin"
        if args.continuous_emitter
        else "finite_waterfall_slug_with_artist_authored_catch_basin"
    ),
    "particle_count": len(initial_positions),
    "maximum_active_particle_count": int(len(positions)),
    "initial_regions": initial_regions,
    "duplicate_initial_particles_removed": duplicate_initial_particles_removed,
    "initial_pool_contact_equilibrium_y": pool_contact_equilibrium_y,
    "initial_pool_footprint": {
        "kind": "artist_authored_ellipse",
        "centre_xz": pool_centre_xz.astype(float).tolist(),
        "radii_xz": pool_radii_xz.astype(float).tolist(),
    },
    "particle_spacing": args.spacing,
    "physics_fps": args.physics_fps,
    "output_fps": args.output_fps,
    "source_fps": args.source_fps,
    "frames": args.frames,
    "duration_seconds": args.frames / args.output_fps,
    "preroll_seconds": args.preroll_seconds,
    "preroll_physics_steps": preroll_steps,
    "collision_meshes": len(collision_meshes),
    "collision_registration": (
        {
            "method": (
                "audited_local_blender_wall_and_ground_adapters"
                if ground_collision_vertices is not None
                else "audited_local_blender_wall_adapter"
            ),
            "adapter": str(args.wall_collision_adapter.resolve()),
            "adapter_sha256": sha256_file(args.wall_collision_adapter),
            "selection": str(args.wall_collision_selection.resolve()),
            "selection_sha256": sha256_file(args.wall_collision_selection),
            "vertices": int(len(wall_collision_vertices)),
            "triangles": int(len(wall_collision_faces)),
            "disabled_unregistered_referenced_meshes": len(
                disabled_referenced_collision_meshes
            ),
            "wall_target_sha256": sha256_file(args.wall_target),
            "ground_adapter": (
                str(args.ground_collision_adapter.resolve())
                if args.ground_collision_adapter is not None
                else None
            ),
            "ground_adapter_sha256": (
                sha256_file(args.ground_collision_adapter)
                if args.ground_collision_adapter is not None
                else None
            ),
            "ground_selection": (
                str(args.ground_collision_selection.resolve())
                if args.ground_collision_selection is not None
                else None
            ),
            "ground_selection_sha256": (
                sha256_file(args.ground_collision_selection)
                if args.ground_collision_selection is not None
                else None
            ),
            "ground_audit": (
                str(args.ground_collision_audit.resolve())
                if args.ground_collision_audit is not None
                else None
            ),
            "ground_audit_sha256": (
                sha256_file(args.ground_collision_audit)
                if args.ground_collision_audit is not None
                else None
            ),
            "ground_vertices": (
                int(len(ground_collision_vertices))
                if ground_collision_vertices is not None
                else 0
            ),
            "ground_triangles": (
                int(len(ground_collision_faces))
                if ground_collision_faces is not None
                else 0
            ),
        }
        if wall_collision_vertices is not None
        else {
            "method": "referenced_mountain_usd_meshes",
            "disabled_unregistered_referenced_meshes": 0,
        }
    ),
    "catch_basin": {
        "kind": "artist_authored_invisible_bottom_plus_elliptical_segmented_wall",
        "wall_segments": catch_wall_segments,
        "wall_centre_xz": catch_wall_centre_xz.astype(float).tolist(),
        "wall_radii_xz": catch_wall_radii_xz.astype(float).tolist(),
        "reason": "The scanned ground is not a closed depression at the selected cliff.",
    },
    "source": {
        "kind": (
            "versioned_pour_source_precomputed_batches_enabled_at_mass_flow_birth_steps"
            if pour_source is not None
            else "preauthored_disabled_batches_enabled_at_deterministic_birth_steps"
            if args.continuous_emitter
            else "finite_initialized_reservoir_and_falling_sheet"
        ),
        "continuous_emitter": continuous_mode,
        "initial_speed_m_s": (
            pour_source.flow.speed_m_s if pour_source is not None else args.source_speed
        ),
        "emitter_interval_physics_steps": (
            None if pour_source is not None else args.emitter_interval_steps
        ),
        "emitter_batch_particles": (
            emitter_batch_particle_counts[0]
            if emitter_batch_particle_counts
            and min(emitter_batch_particle_counts) == max(emitter_batch_particle_counts)
            else None
        ),
        "minimum_emitter_batch_particles": min(
            emitter_batch_particle_counts, default=0
        ),
        "maximum_emitter_batch_particles": max(
            emitter_batch_particle_counts, default=0
        ),
        "emitter_layer_speed_m_s": emitter_layer_speed,
        "enabled_emitter_batches": enabled_emitter_batches,
        "pour_source": (
            {
                "path": str(args.pour_source.resolve()),
                "file_sha256": sha256_file(args.pour_source),
                "configuration_sha256": pour_source.configuration_sha256(),
                "metadata": pour_source.metadata(),
                "schedule_audit": pour_schedule_audit,
            }
            if pour_source is not None
            else None
        ),
    },
    "measured_wall_contact": (
        {
            "valid": wall_contact_valid,
            "method": (
                "identity_stable_measured_surface_contact_episode_with_cumulative_"
                "resolved_normal_velocity_change"
            ),
            "target_path": str(args.wall_target.resolve()),
            "target_sha256": sha256_file(args.wall_target),
            "surface_samples": int(len(wall_target_points)),
            "metrics": wall_contact_metrics,
            "acceptance": {
                "minimum_unique_contact_particles": 12,
                "minimum_contact_fraction_of_near_surface": 0.10,
            },
            "limitation": (
                "Isaac Sim 6.0 exposes no verified PBD-particle mesh contact callback; "
                "the gate combines the independently measured real wall with cumulative "
                "resolved normal velocity deflection and does not count AABB entry alone."
            ),
        }
        if pour_source is not None
        else None
    ),
    "maximum_outside_preview_domain_particles": maximum_outside,
    "maximum_outside_preview_domain_fraction": outside_preview_fraction,
    "maximum_below_collision_support_particles": maximum_below_collision_support,
    "maximum_below_collision_support_fraction": below_collision_support_fraction,
    "containment_acceptance": {
        "maximum_outside_preview_domain_fraction": 0.02,
        "maximum_below_collision_support_fraction": 0.002,
        "interpretation": "Lateral camera-crop loss is distinct from particles falling below all registered support geometry.",
    },
    "maximum_particle_speed_m_s": maximum_speed,
    # This key is part of the version-1 source/run contract consumed by
    # audit_whitewater_source.py.  Keep it stable across scene adapters.
    "whitewater_source_cache": {
        "directory": str(source_directory),
        "manifest": str(source_manifest_path),
        "manifest_sha256": sha256_file(source_manifest_path),
        "samples": len(source_manifest["samples"]),
        "complete": source_complete,
    },
    "particles": {"directory": str(particles_directory), "frames": args.frames + 1},
    "metrics": metrics,
}
atomic_json(args.output / "run_complete.json", report)
print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))
try:
    simulation_app.close()
except SystemExit:
    if not run_valid:
        raise SystemExit(1)
    raise
if not run_valid:
    raise RuntimeError("Mountain water PhysX preview failed its physical gate")
