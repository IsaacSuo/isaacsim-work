"""High-resolution, PhysX-only PBD liquid rendered with RTX Path Tracing.

This is intentionally independent from ParticlePostProcessingDemo.  It builds a
meter-scale dam-break scene, uses PhysX PBD parameters based on SnippetPBF, and
keeps PhysX's native isosurface so the liquid is real geometry for PathTracing.
"""

import argparse
import asyncio
import os
import struct

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=60)
parser.add_argument("--capture-every", type=int, default=2)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--spacing", type=float, default=0.0014)
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
parser.add_argument("--emitter-height", type=float, default=0.62)
parser.add_argument("--emitter-nozzle-diameter", type=float, default=0.024)
parser.add_argument("--emitter-speed", type=float, default=1.20)
parser.add_argument("--emitter-preroll-frames", type=int, default=30)
parser.add_argument("--emitter-feed-clearance", type=float, default=0.11)
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
    if nozzle_diameter < 3.0 * spacing:
        raise ValueError(
            "Emitter nozzle must be at least three particle spacings wide"
        )
    center_limit = 0.5 * nozzle_diameter - 0.5 * spacing
    row_spacing = 0.5 * np.sqrt(3.0) * spacing
    row_count = int(np.ceil(center_limit / row_spacing))
    column_count = int(np.ceil(center_limit / spacing)) + 1
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
    # The preallocated pool travels through a supported horizontal feed pipe
    # and turns down through a compact elbow.  Unlike a tall falling column,
    # gravity cannot continuously accelerate and deplete the hidden supply.
    emitter_feed_length = args.emitter_speed * (
        (args.emitter_preroll_frames + args.frames) / 60.0 + 0.75
    )
    emitter_feed_y = args.emitter_height + args.emitter_feed_clearance
    emitter_feed_start_x = args.stream_outlet_x - emitter_feed_length
    nozzle = UsdGeom.Cylinder.Define(stage, "/World/EmitterNozzle")
    nozzle.CreateAxisAttr().Set(UsdGeom.Tokens.y)
    nozzle.CreateRadiusAttr().Set(0.75 * args.emitter_nozzle_diameter)
    nozzle.CreateHeightAttr().Set(0.10)
    nozzle.AddTranslateOp().Set(
        Gf.Vec3d(
            args.stream_outlet_x,
            args.emitter_height + 0.05,
            0.0,
        )
    )
    bind_material(stage, nozzle.GetPath(), basin_material)
    tube_bottom = args.emitter_height - 0.02
    tube_top = emitter_feed_y + 0.5 * args.emitter_nozzle_diameter + args.spacing
    tube_height = tube_top - tube_bottom
    tube_center_y = 0.5 * (tube_bottom + tube_top)
    tube_inner_half = 0.5 * args.emitter_nozzle_diameter + args.spacing
    tube_wall_thickness = 0.006
    tube_outer_width = 2.0 * tube_inner_half + 2.0 * tube_wall_thickness
    feed_end_x = args.stream_outlet_x + tube_inner_half
    feed_length = feed_end_x - emitter_feed_start_x
    feed_center_x = 0.5 * (feed_end_x + emitter_feed_start_x)
    feed_outer_height = tube_outer_width
    emitter_tube_colliders = [
        # Hidden box colliders form a horizontal pipe.  The bottom stops at
        # the elbow opening, while the end cap redirects flow downward.
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterFeedTop",
            Gf.Vec3f(feed_length, tube_wall_thickness, tube_outer_width),
            Gf.Vec3f(
                feed_center_x,
                emitter_feed_y + tube_inner_half + 0.5 * tube_wall_thickness,
                0.0,
            ),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterFeedBottom",
            Gf.Vec3f(
                args.stream_outlet_x - tube_inner_half - emitter_feed_start_x,
                tube_wall_thickness,
                tube_outer_width,
            ),
            Gf.Vec3f(
                0.5 * (
                    emitter_feed_start_x
                    + args.stream_outlet_x
                    - tube_inner_half
                ),
                emitter_feed_y - tube_inner_half - 0.5 * tube_wall_thickness,
                0.0,
            ),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterFeedBack",
            Gf.Vec3f(feed_length, feed_outer_height, tube_wall_thickness),
            Gf.Vec3f(
                feed_center_x,
                emitter_feed_y,
                -tube_inner_half - 0.5 * tube_wall_thickness,
            ),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterFeedFront",
            Gf.Vec3f(feed_length, feed_outer_height, tube_wall_thickness),
            Gf.Vec3f(
                feed_center_x,
                emitter_feed_y,
                tube_inner_half + 0.5 * tube_wall_thickness,
            ),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterFeedEnd",
            Gf.Vec3f(tube_wall_thickness, feed_outer_height, tube_outer_width),
            Gf.Vec3f(
                feed_end_x + 0.5 * tube_wall_thickness,
                emitter_feed_y,
                0.0,
            ),
        ),
        # Vertical guide below the elbow.  Its left wall starts below the
        # horizontal passage so it does not close the inlet.
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterTubeLeft",
            Gf.Vec3f(
                tube_wall_thickness,
                emitter_feed_y - tube_inner_half - tube_bottom,
                tube_outer_width,
            ),
            Gf.Vec3f(
                args.stream_outlet_x - tube_inner_half - 0.5 * tube_wall_thickness,
                0.5 * (tube_bottom + emitter_feed_y - tube_inner_half),
                0.0,
            ),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterTubeRight",
            Gf.Vec3f(tube_wall_thickness, tube_height, tube_outer_width),
            Gf.Vec3f(
                args.stream_outlet_x + tube_inner_half + 0.5 * tube_wall_thickness,
                tube_center_y,
                0.0,
            ),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterTubeBack",
            Gf.Vec3f(2.0 * tube_inner_half, tube_height, tube_wall_thickness),
            Gf.Vec3f(
                args.stream_outlet_x,
                tube_center_y,
                -tube_inner_half - 0.5 * tube_wall_thickness,
            ),
        ),
        physicsUtils.add_collider_cube(
            stage,
            "/World/EmitterTubeFront",
            Gf.Vec3f(2.0 * tube_inner_half, tube_height, tube_wall_thickness),
            Gf.Vec3f(
                args.stream_outlet_x,
                tube_center_y,
                tube_inner_half + 0.5 * tube_wall_thickness,
            ),
        ),
    ]
    for tube_collider in emitter_tube_colliders:
        source_colliders.append(tube_collider)
        hide(tube_collider)

    # A heavy, gravity-free piston supplies the pressure that an actual pump
    # would provide.  It is a normal PhysX rigid body, so the water is driven
    # through contacts instead of by rewriting particle state at runtime.
    piston_thickness = 2.0 * args.spacing
    emitter_piston = physicsUtils.add_collider_cube(
        stage,
        "/World/EmitterPiston",
        Gf.Vec3f(
            piston_thickness,
            2.0 * (tube_inner_half - 0.001),
            2.0 * (tube_inner_half - 0.001),
        ),
        Gf.Vec3f(
            emitter_feed_start_x + 0.5 * piston_thickness,
            emitter_feed_y,
            0.0,
        ),
    )
    piston_rigid_api = UsdPhysics.RigidBodyAPI.Apply(emitter_piston.GetPrim())
    piston_rigid_api.CreateKinematicEnabledAttr().Set(True)
    piston_physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(
        emitter_piston.GetPrim()
    )
    piston_physx_api.CreateDisableGravityAttr().Set(True)
    emitter_piston_start = Gf.Vec3d(
        emitter_feed_start_x + 0.5 * piston_thickness,
        emitter_feed_y,
        0.0,
    )
    emitter_piston_translate_attr = emitter_piston.GetPrim().GetAttribute(
        "xformOp:translate"
    )
    source_colliders.append(emitter_piston)
    hide(emitter_piston)

    # Opaque render-only cover hides the simulated supply without adding a
    # second collision representation.  It reads as a metal feed pipe.
    feed_cover = UsdGeom.Cylinder.Define(stage, "/World/EmitterFeedCover")
    feed_cover.CreateAxisAttr().Set(UsdGeom.Tokens.x)
    feed_cover.CreateRadiusAttr().Set(0.75 * tube_outer_width)
    feed_cover.CreateHeightAttr().Set(feed_length)
    feed_cover.AddTranslateOp().Set(Gf.Vec3d(feed_center_x, emitter_feed_y, 0.0))
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
    enable_ccd=False,
    solver_position_iterations=args.solver_iterations,
    max_neighborhood=96,
    neighborhood_scale=1.01,
    max_velocity=5.0,
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
if args.source_mode == "emitter":
    particle_capacity = args.particle_capacity
    emitter_axial_spacing = np.sqrt(2.0 / 3.0) * args.spacing
    emitter_row_spacing = 0.5 * np.sqrt(3.0) * args.spacing
    emitter_packing_offsets = (
        (0.0, 0.0),
        (0.5 * args.spacing, emitter_row_spacing / 3.0),
    )
    emitter_layer = create_hexagonal_emitter_layer(
        args.spacing,
        args.emitter_nozzle_diameter,
    )
    layer_positions = []
    layer_velocities = []

    # Pre-fill the horizontal supply.  The pipe floor carries its weight, so
    # its transport velocity does not acquire the unphysical vertical ramp of
    # the previous hanging-column approximation.
    horizontal_x = np.arange(
        emitter_feed_start_x + piston_thickness + emitter_axial_spacing,
        args.stream_outlet_x - tube_inner_half,
        emitter_axial_spacing,
        dtype=np.float32,
    )
    for layer_index, layer_x in enumerate(horizontal_x):
        layer = create_hexagonal_emitter_layer(
            args.spacing,
            args.emitter_nozzle_diameter,
            emitter_packing_offsets[layer_index & 1],
        )
        points = np.empty((len(layer), 3), dtype=np.float32)
        points[:, 0] = layer_x
        points[:, 1] = emitter_feed_y + layer[:, 0]
        points[:, 2] = layer[:, 1]
        layer_positions.append(points)
        layer_velocities.append(
            np.tile(
                np.asarray(
                    [[args.emitter_speed, 0.0, 0.0]], dtype=np.float32
                ),
                (len(layer), 1),
            )
        )

    # Seed the compact down-leg and the visible free jet with the matching
    # gravity solution.  This only initializes a continuous state; after
    # attach, the elbow, nozzle and basin are all advanced by PhysX.
    supply_bottom = fluid_rest_offset + 0.001
    vertical_y = np.arange(
        supply_bottom,
        emitter_feed_y - tube_inner_half - args.spacing,
        emitter_axial_spacing,
        dtype=np.float32,
    )
    for layer_index, layer_y in enumerate(vertical_y):
        fall_distance = emitter_feed_y - float(layer_y)
        layer_speed = np.sqrt(
            args.emitter_speed * args.emitter_speed
            + 2.0 * 9.81 * fall_distance
        )
        # Constant volume flux Q=A*v gives d ~ sqrt(v0/v).  Preserve the
        # particle rest spacing by reducing the number across the section
        # rather than squeezing the same particles unnaturally together.
        layer_diameter = max(
            3.001 * args.spacing,
            args.emitter_nozzle_diameter
            * np.sqrt(args.emitter_speed / layer_speed),
        )
        layer = create_hexagonal_emitter_layer(
            args.spacing,
            layer_diameter,
            emitter_packing_offsets[layer_index & 1],
        )
        points = np.empty((len(layer), 3), dtype=np.float32)
        points[:, 0] = args.stream_outlet_x + layer[:, 0]
        points[:, 1] = layer_y
        points[:, 2] = layer[:, 1]
        layer_positions.append(points)
        layer_velocities.append(
            np.tile(
                np.asarray([[0.0, -layer_speed, 0.0]], dtype=np.float32),
                (len(layer), 1),
            )
        )
    supply_points = np.concatenate(layer_positions, axis=0)
    emitter_velocities_np = np.concatenate(layer_velocities, axis=0)
    if len(supply_points) > particle_capacity:
        raise ValueError(
            f"Ballistic emitter needs {len(supply_points)} particles, "
            f"but capacity is {particle_capacity}"
        )
    positions = Vt.Vec3fArray.FromNumpy(supply_points)
    source_description = (
        f"preallocated_pipe_pool={len(supply_points)}/{particle_capacity}, "
        f"feed_length={emitter_feed_length:.3f}m, "
        f"nozzle={args.emitter_nozzle_diameter:.3f}m, "
        f"speed={args.emitter_speed:.3f}m/s, "
        f"layer_particles={len(emitter_layer)}, packing=HCP"
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
    # Keep the reconstruction grid near 1 mm.  Going below that for a
    # half-million-particle pipe consumes the subgrid budget without adding
    # useful visible detail to a 24 mm jet.
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


def update_emitter_piston_target():
    if not emitter_piston_translate_attr:
        return
    elapsed = simulation_step / (60.0 * args.substeps)
    emitter_piston_translate_attr.Set(
        Gf.Vec3d(
            emitter_piston_start[0] + args.emitter_speed * elapsed,
            emitter_piston_start[1],
            emitter_piston_start[2],
        )
    )


if args.source_mode == "emitter":
    print(
        f"[emitter] pre-rolling pipe-fed particle pool for "
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
        vertex_count = 0
        if iso_mesh:
            mesh_points = iso_mesh.GetPointsAttr().Get()
            vertex_count = len(mesh_points) if mesh_points else 0
        simulation_points = particle_set_api.GetSimulationPointsAttr().Get()
        particle_state = ""
        if simulation_points:
            point_array = np.asarray(simulation_points, dtype=np.float32)
            finite_mask = np.isfinite(point_array).all(axis=1)
            finite_count = int(finite_mask.sum())
            if finite_count:
                finite_points = point_array[finite_mask]
                bounds_min = finite_points.min(axis=0)
                bounds_max = finite_points.max(axis=0)
                particle_state = (
                    f", finite_particles={finite_count}, bounds="
                    f"({bounds_min[0]:.3f},{bounds_min[1]:.3f},"
                    f"{bounds_min[2]:.3f}).."
                    f"({bounds_max[0]:.3f},{bounds_max[1]:.3f},"
                    f"{bounds_max[2]:.3f})"
                )
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
            f"isosurface_vertices={vertex_count}"
            f"{particle_state}{obstacle_state}, path={output_path}"
        )
        capture_index += 1

simulation.detach_stage()
simulation_app.close()
