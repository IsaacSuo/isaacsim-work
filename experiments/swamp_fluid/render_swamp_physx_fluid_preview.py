"""Render a short PhysX particle-fluid proof inside the authored Swamp scene.

This deliberately replaces the authored metallic ``Water`` plane with a real
GPU PBD liquid and PhysX's native isosurface.  The first proof uses one rigid
sphere so fluid/rigid two-way coupling is unambiguous before the mixed
rigid/deformable experiment is integrated.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frames", type=int, default=48, help="Output frames at 30 fps.")
parser.add_argument("--physics-fps", type=int, default=240)
parser.add_argument("--spacing", type=float, default=0.008)
parser.add_argument(
    "--water-level-offset",
    type=float,
    default=-0.055,
    help="Offset from the authored puddle level; lowering it reduces water volume naturally.",
)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--resolution", type=int, default=640)
parser.add_argument("--settle-frames", type=int, default=360)
parser.add_argument(
    "--settle-damping",
    type=float,
    default=0.5,
    help="Temporary damping used only during the unrecorded hydrostatic pre-roll.",
)
parser.add_argument(
    "--minimum-water-layers",
    type=int,
    default=1,
    help="Minimum particle layers used by the physical basin fill.",
)
parser.add_argument(
    "--render-minimum-water-layers",
    type=int,
    default=8,
    help="Render-only depth threshold that hides unresolved shoreline films.",
)
parser.add_argument("--solver-iterations", type=int, default=8)
parser.add_argument(
    "--collision-margin",
    type=float,
    default=0.75,
    help="Dry-terrain collision apron beyond the nominal pool radii.",
)
parser.add_argument(
    "--collision-scope",
    choices=("full", "local"),
    default="full",
    help=(
        "Use the full authored Object_12 terrain for formal runs; local is a "
        "faster diagnostic crop controlled by --collision-margin."
    ),
)
parser.add_argument("--damping", type=float, default=0.02)
parser.add_argument("--viscosity", type=float, default=0.001)
parser.add_argument("--vorticity-confinement", type=float, default=0.0)
parser.add_argument("--surface-tension", type=float, default=0.0074)
parser.add_argument("--cohesion", type=float, default=0.01)
parser.add_argument(
    "--particle-smoothing-strength",
    type=float,
    default=0.12,
    help="Render-only PhysX particle-position smoothing coefficient.",
)
parser.add_argument(
    "--mesh-normal-smoothing-passes",
    type=int,
    default=32,
    help="Render-only isosurface normal smoothing; does not damp the fluid motion.",
)
parser.add_argument(
    "--glass-roughness",
    type=float,
    default=0.015,
    help="OmniGlass frosting roughness. Real water should remain nearly specular.",
)
parser.add_argument(
    "--render-shoreline-erosion-cells",
    type=int,
    default=4,
    help="Render-only inward shoreline mask used to remove isosurface closure walls.",
)
parser.add_argument("--debug-opaque-water", action="store_true")
parser.add_argument(
    "--debug-render-particles",
    action="store_true",
    help="Render raw PBD particles as spheres with smoothing, anisotropy, and isosurface disabled.",
)
parser.add_argument(
    "--enable-anisotropy",
    dest="enable_anisotropy",
    action="store_true",
    help="Enable conservative anisotropic reconstruction after isotropic validation.",
)
parser.add_argument(
    "--disable-anisotropy",
    dest="enable_anisotropy",
    action="store_false",
    help="Disable anisotropic surface reconstruction for diagnostics.",
)
parser.set_defaults(enable_anisotropy=True)
parser.add_argument(
    "--no-capture",
    action="store_true",
    help="Run physics and write diagnostics without spending time on image capture.",
)
parser.add_argument(
    "--export-particle-ply",
    action="store_true",
    help="Export selected output-frame particle positions as binary little-endian PLY.",
)
parser.add_argument(
    "--export-particle-stride",
    type=int,
    default=1,
    help="Export one particle PLY every N output frames when PLY export is enabled.",
)
parser.add_argument(
    "--export-whitewater-source",
    action="store_true",
    help=(
        "Export raw high-frequency PBD positions, native velocities, exact physics "
        "timestamps, and rigid impactor state for the unified whitewater solver."
    ),
)
parser.add_argument(
    "--whitewater-impact-fps",
    type=int,
    default=120,
    help="Source-cache rate from release through --whitewater-impact-seconds.",
)
parser.add_argument(
    "--whitewater-source-fps",
    type=int,
    default=None,
    help=(
        "Export the complete whitewater source cache at one fixed rate. This "
        "overrides the impact/tail schedule and is the required mode for tools "
        "such as SPlisHSPlasH FoamGenerator that accept one constant timestep."
    ),
)
parser.add_argument(
    "--whitewater-impact-seconds",
    type=float,
    default=2.5,
    help="Duration of the high-frequency whitewater source-cache window.",
)
parser.add_argument(
    "--enable-physx-diffuse",
    action="store_true",
    help="Enable PhysX render-only diffuse particles after the hydrostatic settle.",
)
parser.add_argument(
    "--diffuse-max-multiplier",
    type=float,
    default=0.10,
    help="Maximum diffuse-particle count relative to the primary particle count.",
)
parser.add_argument(
    "--diffuse-threshold",
    type=float,
    default=0.01,
    help="PhysX diffuse-particle kinetic-energy emission threshold.",
)
parser.add_argument("--diffuse-lifetime", type=float, default=1.5)
parser.add_argument("--diffuse-air-drag", type=float, default=1.0)
parser.add_argument("--diffuse-bubble-drag", type=float, default=0.5)
parser.add_argument("--diffuse-buoyancy", type=float, default=0.8)
parser.add_argument(
    "--diffuse-kinetic-energy-weight", type=float, default=0.01
)
parser.add_argument("--diffuse-pressure-weight", type=float, default=1.0)
parser.add_argument("--diffuse-divergence-weight", type=float, default=5.0)
parser.add_argument("--diffuse-collision-decay", type=float, default=0.5)
parser.add_argument(
    "--diffuse-use-accurate-velocity",
    action="store_true",
    help="Ask PhysX to use its more accurate diffuse-particle velocity estimate.",
)
parser.add_argument(
    "--diffuse-enable-before-settle",
    action="store_true",
    help=(
        "Enable diffuse before PhysX first parses the scene. This tests whether "
        "runtime activation is responsible for a diffuse GPU-buffer failure."
    ),
)
parser.add_argument(
    "--diffuse-debug-visualization",
    action="store_true",
    help=(
        "Enable PhysX diffuse debug visualization while hiding the primary-particle "
        "debug layer; intended only for the isolated feasibility gate."
    ),
)
parser.add_argument(
    "--diffuse-probe",
    action="store_true",
    help=(
        "Write per-output runtime discovery snapshots for the PhysX diffuse export "
        "feasibility audit. Does not claim diffuse buffers are exportable."
    ),
)
parser.add_argument(
    "--export-native-diffuse",
    action="store_true",
    help=(
        "Export the live PxParticleAndDiffuseBuffer at every non-empty output "
        "frame through the pinned Isaac Sim 6.0 native bridge. The cache contains "
        "primary and Diffuse positions/velocities plus Diffuse remaining lifetime; "
        "it does not infer render labels during simulation."
    ),
)
parser.add_argument(
    "--whitewater-tail-fps",
    type=int,
    default=30,
    help="Source-cache rate after the high-frequency impact window.",
)
parser.add_argument(
    "--initialization-only",
    action="store_true",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--export-collision-heightfield",
    type=Path,
    help=(
        "Atomically export the exact rasterized collision heightfield used by "
        "the containment audit. Pair with --initialization-only to build the "
        "asset without running the recorded simulation."
    ),
)
parser.add_argument("--sphere-radius", type=float, default=0.08)
parser.add_argument("--sphere-mass", type=float, default=0.65)
parser.add_argument("--sphere-start-y", type=float, default=-0.55)
parser.add_argument(
    "--analytic-rigid-coupling",
    action="store_true",
    help="Add analytic buoyancy/drag; disabled by default to expose native contacts.",
)
parser.add_argument(
    "--swamp-usd", type=Path, default=Path(r"Y:\scenes\swamp\swamp.usdc")
)
args = parser.parse_args()

if args.frames < 2 or args.physics_fps < 30 or args.physics_fps % 30:
    raise ValueError("frames must be >= 2 and physics-fps must be a multiple of 30")
if not 0.006 <= args.spacing <= 0.06:
    raise ValueError("spacing must stay within the preview range 0.006..0.06 m")
if not -0.12 <= args.water_level_offset <= 0.02:
    raise ValueError("water-level-offset must stay within -0.12..0.02 m")
for required in (args.swamp_usd,):
    if not required.is_file():
        raise FileNotFoundError(required)
if min(
    args.damping,
    args.settle_damping,
    args.viscosity,
    args.vorticity_confinement,
    args.surface_tension,
    args.cohesion,
) < 0.0:
    raise ValueError("fluid material coefficients must be non-negative")
if args.solver_iterations < 1:
    raise ValueError("solver-iterations must be at least 1")
if args.minimum_water_layers < 1:
    raise ValueError("minimum-water-layers must be at least 1")
if args.render_minimum_water_layers < 1:
    raise ValueError("render-minimum-water-layers must be at least 1")
if not 0.0 <= args.particle_smoothing_strength <= 1.0:
    raise ValueError("particle-smoothing-strength must be within 0..1")
if args.mesh_normal_smoothing_passes < 0:
    raise ValueError("mesh-normal-smoothing-passes must be non-negative")
if not 0.0 <= args.glass_roughness <= 1.0:
    raise ValueError("glass-roughness must be within 0..1")
if args.render_shoreline_erosion_cells < 0:
    raise ValueError("render-shoreline-erosion-cells must be non-negative")
if args.export_particle_stride < 1:
    raise ValueError("export-particle-stride must be at least 1")
if args.whitewater_impact_seconds < 0.0:
    raise ValueError("whitewater-impact-seconds must be non-negative")
for option_name, source_fps in (
    ("whitewater-impact-fps", args.whitewater_impact_fps),
    ("whitewater-tail-fps", args.whitewater_tail_fps),
):
    if source_fps < 1 or source_fps > args.physics_fps:
        raise ValueError(f"{option_name} must be within 1..physics-fps")
    if args.physics_fps % source_fps:
        raise ValueError(f"{option_name} must divide physics-fps exactly")
if args.whitewater_source_fps is not None:
    if not 1 <= args.whitewater_source_fps <= args.physics_fps:
        raise ValueError("whitewater-source-fps must be within 1..physics-fps")
    if args.physics_fps % args.whitewater_source_fps:
        raise ValueError("whitewater-source-fps must divide physics-fps exactly")
if not 0.30 <= args.collision_margin <= 2.0:
    raise ValueError("collision-margin must stay within 0.30..2.0 m")
if not 0.0 <= args.diffuse_max_multiplier <= 4.0:
    raise ValueError("diffuse-max-multiplier must stay within 0..4")
for option_name, option_value in (
    ("diffuse-threshold", args.diffuse_threshold),
    ("diffuse-lifetime", args.diffuse_lifetime),
    ("diffuse-air-drag", args.diffuse_air_drag),
    ("diffuse-bubble-drag", args.diffuse_bubble_drag),
    ("diffuse-buoyancy", args.diffuse_buoyancy),
    ("diffuse-kinetic-energy-weight", args.diffuse_kinetic_energy_weight),
    ("diffuse-pressure-weight", args.diffuse_pressure_weight),
    ("diffuse-divergence-weight", args.diffuse_divergence_weight),
):
    if option_value < 0.0 or not math.isfinite(option_value):
        raise ValueError(f"{option_name} must be finite and non-negative")
if not 0.0 <= args.diffuse_collision_decay <= 1.0:
    raise ValueError("diffuse-collision-decay must stay within 0..1")
if args.diffuse_enable_before_settle and not args.enable_physx_diffuse:
    raise ValueError(
        "diffuse-enable-before-settle requires --enable-physx-diffuse"
    )
if args.diffuse_debug_visualization and not args.enable_physx_diffuse:
    raise ValueError(
        "diffuse-debug-visualization requires --enable-physx-diffuse"
    )
if args.export_native_diffuse and not args.enable_physx_diffuse:
    raise ValueError("export-native-diffuse requires --enable-physx-diffuse")

args.output.mkdir(parents=True, exist_ok=True)
whitewater_source_dir = args.output / "whitewater_source"
if args.export_whitewater_source and whitewater_source_dir.exists():
    existing_whitewater_files = list(whitewater_source_dir.iterdir())
    if existing_whitewater_files:
        raise FileExistsError(
            "Refusing to overwrite non-empty whitewater source cache: "
            f"{whitewater_source_dir}"
        )
if args.export_whitewater_source:
    whitewater_source_dir.mkdir(parents=True, exist_ok=True)
frames_dir = args.output / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
for stale in frames_dir.glob("frame_*.png"):
    stale.unlink()
particles_dir = args.output / "particles"
if args.export_particle_ply:
    particles_dir.mkdir(parents=True, exist_ok=True)
    for stale in particles_dir.glob("particles_*.ply"):
        stale.unlink()
native_diffuse_dir = args.output / "native_diffuse"
native_diffuse_manifest_path = native_diffuse_dir / "manifest.json"
if args.export_native_diffuse:
    if native_diffuse_dir.exists() and any(native_diffuse_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite an existing native Diffuse cache: "
            f"{native_diffuse_dir}"
        )
    native_diffuse_dir.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "PathTracing",
        "width": args.resolution,
        "height": args.resolution,
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
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils, Vt
from whitewater.physx_diffuse_probe import PhysxDiffuseProbe

native_diffuse_bridge = None
if args.export_native_diffuse:
    native_bridge_directory = (
        Path(__file__).resolve().parents[1] / "mountain_waterfall" / "native_bridge"
    )
    if not native_bridge_directory.is_dir():
        raise FileNotFoundError(native_bridge_directory)
    sys.path.insert(0, str(native_bridge_directory))
    import physx_diffuse_bridge as native_diffuse_bridge


WATER_PRIM = (
    "/World/Environment/Sketchfab_model/root/GLTF_SceneRootNode/"
    "Water_5/Object_14/Object_5"
)
TERRAIN_PRIM = (
    "/World/Environment/Sketchfab_model/root/GLTF_SceneRootNode/"
    "Terrain_4/Object_12/Object_4"
)
AUTHORED_WATER_LEVEL = -1.3259903192520142
WATER_LEVEL = AUTHORED_WATER_LEVEL + args.water_level_offset
POOL_CENTER = np.asarray((-0.73, 0.78), dtype=np.float32)
POOL_RADII = np.asarray((1.28, 1.75), dtype=np.float32)
POOL_FLOOR_MINIMUM = -1.525
SPHERE_RADIUS = args.sphere_radius
SPHERE_MASS = args.sphere_mass
RENDER_WET_XS = None
RENDER_WET_ZS = None
RENDER_WET_MASK = None
LAST_SURFACE_EXTRACTION_DIAGNOSTICS = None


def set_camera(prim, eye, target, up=Gf.Vec3d(0, 1, 0)):
    matrix = Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()
    UsdGeom.Xformable(prim).AddTransformOp().Set(matrix)


def bind_material(prim_path, material_path):
    omni.kit.commands.execute(
        "BindMaterialCommand",
        prim_path=Sdf.Path(prim_path),
        material_path=Sdf.Path(material_path),
        strength=None,
    )


def authored_terrain_triangles():
    """Return every authored Terrain triangle in transformed world space."""
    terrain = UsdGeom.Mesh.Get(stage, TERRAIN_PRIM)
    if not terrain:
        raise RuntimeError(f"Authored Swamp terrain is missing: {TERRAIN_PRIM}")
    points = list(terrain.GetPointsAttr().Get() or [])
    counts = list(terrain.GetFaceVertexCountsAttr().Get() or [])
    indices = list(terrain.GetFaceVertexIndicesAttr().Get() or [])
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(terrain.GetPrim())
    world_points = np.asarray([matrix.Transform(point) for point in points], dtype=np.float64)
    triangles = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            triangles.append(world_points[[face[0], face[offset], face[offset + 1]]])
    if not triangles:
        raise RuntimeError("Authored Swamp terrain contains no triangles")
    return np.asarray(triangles, dtype=np.float64)


def terrain_fill_triangles(triangles):
    """Select only low triangles that can define the initialized water floor."""
    sampling_margin = 0.30
    minimum_x = float(POOL_CENTER[0] - POOL_RADII[0] - sampling_margin)
    maximum_x = float(POOL_CENTER[0] + POOL_RADII[0] + sampling_margin)
    minimum_z = float(POOL_CENTER[1] - POOL_RADII[1] - sampling_margin)
    maximum_z = float(POOL_CENTER[1] + POOL_RADII[1] + sampling_margin)
    selected = triangles[
        (triangles[:, :, 1].min(axis=1) < WATER_LEVEL)
        & (triangles[:, :, 0].max(axis=1) >= minimum_x)
        & (triangles[:, :, 0].min(axis=1) <= maximum_x)
        & (triangles[:, :, 2].max(axis=1) >= minimum_z)
        & (triangles[:, :, 2].min(axis=1) <= maximum_z)
    ].copy()
    if not len(selected):
        raise RuntimeError("Authored Swamp terrain has no triangles below the water level")
    return selected


def create_terrain_collision(triangles):
    """Cook an exact static collider from the current, transformed Terrain.

    The old external collision USD was cropped before the Blender Z-up to
    Isaac Y-up coordinate correction and covered the wrong side of the scene.
    Building from world-space triangles here keeps rendering, filling, and
    collision in the same coordinate frame.
    """
    margin = args.collision_margin
    if args.collision_scope == "full":
        selected = triangles.copy()
    else:
        minimum_x = float(POOL_CENTER[0] - POOL_RADII[0] - margin)
        maximum_x = float(POOL_CENTER[0] + POOL_RADII[0] + margin)
        minimum_z = float(POOL_CENTER[1] - POOL_RADII[1] - margin)
        maximum_z = float(POOL_CENTER[1] + POOL_RADII[1] + margin)
        selected = triangles[
            (triangles[:, :, 0].max(axis=1) >= minimum_x)
            & (triangles[:, :, 0].min(axis=1) <= maximum_x)
            & (triangles[:, :, 2].max(axis=1) >= minimum_z)
            & (triangles[:, :, 2].min(axis=1) <= maximum_z)
        ].copy()
    if not len(selected):
        raise RuntimeError("No authored Terrain triangles overlap the Swamp basin")

    normals = np.cross(selected[:, 1] - selected[:, 0], selected[:, 2] - selected[:, 0])
    downward = normals[:, 1] < 0.0
    selected[downward, 1], selected[downward, 2] = (
        selected[downward, 2].copy(),
        selected[downward, 1].copy(),
    )

    # Preserve shared vertices rather than emitting triangle soup.  This keeps
    # internal edges welded during PhysX triangle-mesh cooking.
    vertex_map = {}
    points = []
    indices = []
    for triangle in selected:
        for point in triangle:
            key = tuple(float(value) for value in point)
            index = vertex_map.get(key)
            if index is None:
                index = len(points)
                vertex_map[key] = index
                points.append(Gf.Vec3f(*key))
            indices.append(index)

    path = "/World/FluidBasinCollision/LocalTerrain"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(points)
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(selected))
    mesh.CreateFaceVertexIndicesAttr().Set(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()).CreateContactOffsetAttr().Set(0.012)
    PhysxSchema.PhysxCollisionAPI(mesh.GetPrim()).CreateRestOffsetAttr().Set(0.0)
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()

    bounds_minimum = selected.reshape(-1, 3).min(axis=0)
    bounds_maximum = selected.reshape(-1, 3).max(axis=0)
    report = {
        "source": TERRAIN_PRIM,
        "mode": (
            "runtime_exact_full_authored_terrain"
            if args.collision_scope == "full"
            else "runtime_exact_local_triangle_mesh_with_dry_apron"
        ),
        "scope": args.collision_scope,
        "selection_margin": margin if args.collision_scope == "local" else None,
        "vertices": len(points),
        "triangles": len(selected),
        "flipped_downward_triangles": int(np.count_nonzero(downward)),
        "bounds_minimum": bounds_minimum.astype(float).tolist(),
        "bounds_maximum": bounds_maximum.astype(float).tolist(),
        "safety_bottom": False,
        "safety_ring": False,
    }
    return selected, report


def sample_floor_y(x, z, triangles, maximum_y=WATER_LEVEL):
    """Find the highest selected Terrain intersection at ``(x, z)``."""
    candidates = triangles[
        (triangles[:, :, 0].min(axis=1) <= x)
        & (triangles[:, :, 0].max(axis=1) >= x)
        & (triangles[:, :, 2].min(axis=1) <= z)
        & (triangles[:, :, 2].max(axis=1) >= z)
    ]
    heights = []
    point = np.asarray((x, z), dtype=np.float64)
    for triangle in candidates:
        planar = triangle[:, (0, 2)]
        a, b, c = planar
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
            a[1] - c[1]
        )
        if abs(denominator) <= 1.0e-12:
            continue
        wa = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (point[1] - c[1])) / denominator
        wb = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (point[1] - c[1])) / denominator
        wc = 1.0 - wa - wb
        if min(wa, wb, wc) >= -1.0e-7:
            y = wa * triangle[0, 1] + wb * triangle[1, 1] + wc * triangle[2, 1]
            if maximum_y is None or y < maximum_y:
                heights.append(float(y))
    return max(heights) if heights else None


def rasterize_terrain_heightfield(triangles, spacing):
    """Rasterize the collision apron once for inexpensive per-frame audits."""
    bounds_minimum = triangles.reshape(-1, 3).min(axis=0)
    bounds_maximum = triangles.reshape(-1, 3).max(axis=0)
    xs = np.arange(
        bounds_minimum[0], bounds_maximum[0] + 0.5 * spacing, spacing,
        dtype=np.float64,
    )
    zs = np.arange(
        bounds_minimum[2], bounds_maximum[2] + 0.5 * spacing, spacing,
        dtype=np.float64,
    )
    heights = np.full((len(xs), len(zs)), np.nan, dtype=np.float32)
    for triangle in triangles:
        ix0 = max(0, int(math.floor((triangle[:, 0].min() - xs[0]) / spacing)))
        ix1 = min(len(xs) - 1, int(math.ceil((triangle[:, 0].max() - xs[0]) / spacing)))
        iz0 = max(0, int(math.floor((triangle[:, 2].min() - zs[0]) / spacing)))
        iz1 = min(len(zs) - 1, int(math.ceil((triangle[:, 2].max() - zs[0]) / spacing)))
        grid_x, grid_z = np.meshgrid(
            xs[ix0 : ix1 + 1], zs[iz0 : iz1 + 1], indexing="ij"
        )
        a, b, c = triangle[:, (0, 2)]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
            a[1] - c[1]
        )
        if abs(denominator) <= 1.0e-12:
            continue
        wa = (
            (b[1] - c[1]) * (grid_x - c[0])
            + (c[0] - b[0]) * (grid_z - c[1])
        ) / denominator
        wb = (
            (c[1] - a[1]) * (grid_x - c[0])
            + (a[0] - c[0]) * (grid_z - c[1])
        ) / denominator
        wc = 1.0 - wa - wb
        inside = np.minimum(np.minimum(wa, wb), wc) >= -1.0e-7
        triangle_y = (
            wa * triangle[0, 1] + wb * triangle[1, 1] + wc * triangle[2, 1]
        )
        target = heights[ix0 : ix1 + 1, iz0 : iz1 + 1]
        target[inside] = np.fmax(target[inside], triangle_y[inside])
    if not np.isfinite(heights).any():
        raise RuntimeError("Collision terrain audit heightfield is empty")
    return xs, zs, heights


def audit_particle_containment(current, xs, zs, terrain_y):
    """Classify domain escape and terrain tunnelling without hiding either."""
    spacing_x = float(np.median(np.diff(xs)))
    spacing_z = float(np.median(np.diff(zs)))
    ix = np.rint((current[:, 0] - xs[0]) / spacing_x).astype(np.int64)
    iz = np.rint((current[:, 2] - zs[0]) / spacing_z).astype(np.int64)
    inside_domain = (
        (ix >= 0) & (ix < len(xs)) & (iz >= 0) & (iz < len(zs))
    )
    sampled_terrain = np.full(len(current), np.nan, dtype=np.float32)
    sampled_terrain[inside_domain] = terrain_y[ix[inside_domain], iz[inside_domain]]
    terrain_coverage = inside_domain & np.isfinite(sampled_terrain)
    outside_domain = ~inside_domain
    missing_terrain = inside_domain & ~terrain_coverage
    penetration_tolerance = args.spacing
    terrain_penetration = terrain_coverage & (
        current[:, 1] < sampled_terrain - penetration_tolerance
    )
    escaped = outside_domain | missing_terrain | terrain_penetration
    return escaped, {
        "outside_collision_domain_count": int(np.count_nonzero(outside_domain)),
        "missing_collision_terrain_count": int(np.count_nonzero(missing_terrain)),
        "terrain_penetration_count": int(np.count_nonzero(terrain_penetration)),
        "penetration_tolerance": penetration_tolerance,
    }


def create_terrain_fitted_pool(spacing, triangles):
    """Flood the Terrain-connected depression without an artificial ellipse."""
    global RENDER_WET_XS, RENDER_WET_ZS, RENDER_WET_MASK
    contact_offset = 0.5 * spacing / 0.6
    top = WATER_LEVEL - 0.45 * spacing
    search_margin = 0.30
    xs = np.arange(
        POOL_CENTER[0] - POOL_RADII[0] - search_margin + spacing,
        POOL_CENTER[0] + POOL_RADII[0] + search_margin - spacing,
        spacing,
        dtype=np.float32,
    )
    zs = np.arange(
        POOL_CENTER[1] - POOL_RADII[1] - search_margin + spacing,
        POOL_CENTER[1] + POOL_RADII[1] + search_margin - spacing,
        spacing,
        dtype=np.float32,
    )
    floor_clearance = contact_offset + 0.001
    floor_grid = np.full((len(xs), len(zs)), np.nan, dtype=np.float64)
    layer_grid = np.zeros((len(xs), len(zs)), dtype=np.int32)
    rejected_deep_columns = 0
    for ix, x in enumerate(xs):
        for iz, z in enumerate(zs):
            floor_y = sample_floor_y(float(x), float(z), triangles)
            if floor_y is None:
                continue
            if floor_y < POOL_FLOOR_MINIMUM:
                rejected_deep_columns += 1
                continue
            bottom = floor_y + floor_clearance
            if bottom > top:
                continue
            layer_count = int(math.floor((top - bottom) / spacing)) + 1
            if layer_count < args.minimum_water_layers:
                continue
            floor_grid[ix, iz] = floor_y
            layer_grid[ix, iz] = layer_count

    # Select only the wet cells hydraulically connected to the seed at the
    # authored puddle centre.  This prevents unrelated low terrain from being
    # flooded without introducing a render-visible ellipse or retaining wall.
    candidates = np.argwhere(layer_grid > 0)
    if not len(candidates):
        raise RuntimeError("Terrain-fitted Swamp pool contains no candidate cells")
    seed_distance = (
        (xs[candidates[:, 0]] - POOL_CENTER[0]) ** 2
        + (zs[candidates[:, 1]] - POOL_CENTER[1]) ** 2
    )
    seed = tuple(int(value) for value in candidates[int(np.argmin(seed_distance))])
    stack = [seed]
    connected = np.zeros_like(layer_grid, dtype=bool)
    connected[seed] = True
    while stack:
        ix, iz = stack.pop()
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, nz = ix + dx, iz + dz
            if (
                0 <= nx < len(xs)
                and 0 <= nz < len(zs)
                and layer_grid[nx, nz] > 0
                and not connected[nx, nz]
            ):
                connected[nx, nz] = True
                stack.append((nx, nz))

    boundary_contact = bool(
        connected[0].any()
        or connected[-1].any()
        or connected[:, 0].any()
        or connected[:, -1].any()
    )
    # Erode only the render mask.  PhysX's isosurface closes the particle volume
    # with side faces at its shoreline; those faces show up as a blue water
    # slope.  Keeping physics untouched while dropping a narrow boundary band
    # leaves an open free-surface edge that the authored Terrain occludes.
    render_mask = connected & (layer_grid >= args.render_minimum_water_layers)
    for _ in range(args.render_shoreline_erosion_cells):
        eroded = np.zeros_like(render_mask)
        eroded[1:-1, 1:-1] = (
            render_mask[1:-1, 1:-1]
            & render_mask[:-2, 1:-1]
            & render_mask[2:, 1:-1]
            & render_mask[1:-1, :-2]
            & render_mask[1:-1, 2:]
        )
        render_mask = eroded
    RENDER_WET_XS = xs
    RENDER_WET_ZS = zs
    RENDER_WET_MASK = render_mask
    columns = []
    sampled_floors = []
    for ix, iz in np.argwhere(connected):
        x, z = xs[ix], zs[iz]
        floor_y = float(floor_grid[ix, iz])
        layer_count = int(layer_grid[ix, iz])
        sampled_floors.append(floor_y)
        ys = top - np.arange(layer_count, dtype=np.float32) * spacing
        columns.extend((float(x), float(y), float(z)) for y in ys)
    if not columns:
        raise RuntimeError("Terrain-fitted Swamp pool contains no particles")
    points = np.ascontiguousarray(columns, dtype=np.float32)
    report = {
        "method": "vertical_sampling_of_authored_terrain",
        "shoreline_selection": "four_connected_component_from_pool_center",
        "artificial_ellipse_clip": False,
        "sampling_boundary_contact": boundary_contact,
        "simulation_domain": (
            "open_crop_outside_render_view"
            if boundary_contact
            else "terrain_closed_depression"
        ),
        "candidate_columns": int(len(candidates)),
        "accepted_columns": len(sampled_floors),
        "render_mask_columns": int(np.count_nonzero(render_mask)),
        "render_minimum_water_layers": args.render_minimum_water_layers,
        "render_shoreline_erosion_cells": args.render_shoreline_erosion_cells,
        "rejected_deep_columns": rejected_deep_columns,
        "sampled_floor_minimum": min(sampled_floors),
        "sampled_floor_maximum": max(sampled_floors),
        "free_surface_target": top,
        "fill_direction": "shared_surface_down_to_local_floor",
        "minimum_water_layers": args.minimum_water_layers,
        "particle_minimum": points.min(axis=0).astype(float).tolist(),
        "particle_maximum": points.max(axis=0).astype(float).tolist(),
    }
    return points, report


def write_binary_particle_ply(path, positions):
    """Write Splashsurf-compatible XYZ particles without a Python per-point loop."""
    xyz = np.ascontiguousarray(positions, dtype="<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment exported from Isaac Sim PhysX PBD particles\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        xyz.tofile(stream)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def capture(viewport, path):
    if path.exists():
        path.unlink()
    task = asyncio.ensure_future(
        capture_viewport_to_file(viewport, file_path=str(path)).wait_for_result(
            completion_frames=2
        )
    )
    for _ in range(500):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        task.cancel()
        raise RuntimeError(f"Timed out capturing {path}")
    if not task.result():
        raise RuntimeError(f"Capture failed: {path}")
    for _ in range(300):
        if path.is_file() and path.stat().st_size > 0:
            return
        simulation_app.update()
    raise RuntimeError(f"Capture produced no non-empty file: {path}")


context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

environment = UsdGeom.Xform.Define(stage, "/World/Environment")
environment.GetPrim().GetReferences().AddReference(str(args.swamp_usd.resolve()))
# Blender's USD is Z-up.  References do not perform stage-axis conversion, so
# rotate it explicitly into this Y-up physics stage: (x, y, z) -> (x, z, -y).
UsdGeom.Xformable(environment.GetPrim()).AddRotateXOp().Set(-90.0)

# A stale export may still contain the old plastic water mesh.  Disable it so
# only the dynamic PhysX isosurface can represent the water.
water_prim = stage.GetPrimAtPath(WATER_PRIM)
authored_water_present = bool(water_prim and water_prim.IsValid())
if authored_water_present:
    UsdGeom.Imageable(water_prim).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(water_prim).CreateCollisionEnabledAttr().Set(False)
terrain_visual_report = {
    "mode": "authored_terrain_from_reexported_usd",
    "runtime_visual_cut": False,
}
all_authored_terrain_triangles = authored_terrain_triangles()
fill_terrain_triangles = terrain_fill_triangles(all_authored_terrain_triangles)
terrain_below_water_bounds = {
    "minimum": fill_terrain_triangles.reshape(-1, 3).min(axis=0).astype(float).tolist(),
    "maximum": fill_terrain_triangles.reshape(-1, 3).max(axis=0).astype(float).tolist(),
    "triangles": int(len(fill_terrain_triangles)),
    "authored_terrain_triangles": int(len(all_authored_terrain_triangles)),
}
print(
    "[swamp-fluid] below-water-terrain "
    + json.dumps(terrain_below_water_bounds, ensure_ascii=False),
    flush=True,
)
collision_terrain_triangles, basin_collision_report = create_terrain_collision(
    all_authored_terrain_triangles
)
collision_audit_xs, collision_audit_zs, collision_audit_terrain_y = (
    rasterize_terrain_heightfield(collision_terrain_triangles, args.spacing)
)
if args.export_collision_heightfield is not None:
    collision_heightfield_path = args.export_collision_heightfield.resolve()
    if collision_heightfield_path.exists():
        raise FileExistsError(
            "Refusing to overwrite collision heightfield: "
            f"{collision_heightfield_path}"
        )
    collision_heightfield_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_npz(
        collision_heightfield_path,
        schema=np.asarray(1, dtype="<i4"),
        x_values=np.asarray(collision_audit_xs, dtype="<f8"),
        z_values=np.asarray(collision_audit_zs, dtype="<f8"),
        terrain_y=np.asarray(collision_audit_terrain_y, dtype="<f4"),
        spacing=np.asarray(args.spacing, dtype="<f8"),
        collision_scope=np.asarray(args.collision_scope),
        terrain_prim=np.asarray(TERRAIN_PRIM),
        swamp_usd=np.asarray(str(args.swamp_usd.resolve())),
    )
    print(
        "[swamp-fluid] collision-heightfield "
        + json.dumps(
            {
                "path": str(collision_heightfield_path),
                "bytes": collision_heightfield_path.stat().st_size,
                "sha256": sha256_file(collision_heightfield_path),
                "shape": list(collision_audit_terrain_y.shape),
                "valid_cells": int(
                    np.count_nonzero(np.isfinite(collision_audit_terrain_y))
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
basin_collision_report["audit_heightfield"] = {
    "shape": [len(collision_audit_xs), len(collision_audit_zs)],
    "spacing": args.spacing,
    "valid_cells": int(np.count_nonzero(np.isfinite(collision_audit_terrain_y))),
}
print(
    "[swamp-fluid] basin-collision "
    + json.dumps(basin_collision_report, ensure_ascii=False),
    flush=True,
)

scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, -1, 0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
physx_scene.CreateGpuMaxParticleContactsAttr().Set(1_000_000)
physx_scene.CreateEnableExternalForcesEveryIterationAttr().Set(True)
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
    max_velocity=6.0,
)

water_physics_path = Sdf.Path("/World/Looks/WaterPhysics")
particleUtils.add_pbd_particle_material(
    stage,
    water_physics_path,
    density=1000.0,
    friction=0.06,
    damping=args.settle_damping,
    viscosity=args.viscosity,
    vorticity_confinement=args.vorticity_confinement,
    surface_tension=args.surface_tension,
    cohesion=args.cohesion,
    adhesion=0.0,
    cfl_coefficient=1.0,
)
physicsUtils.add_physics_material_to_prim(
    stage, particle_system.GetPrim(), water_physics_path
)

pool_points, pool_initialization_report = create_terrain_fitted_pool(
    args.spacing, fill_terrain_triangles
)
initial_escaped_mask, initial_containment_audit = audit_particle_containment(
    pool_points,
    collision_audit_xs,
    collision_audit_zs,
    collision_audit_terrain_y,
)
pool_initialization_report["containment_audit"] = initial_containment_audit
pool_initialization_report["escaped_particles"] = int(
    np.count_nonzero(initial_escaped_mask)
)
if np.any(initial_escaped_mask):
    raise RuntimeError(
        "Initialized pool is not fully contained by the collision terrain: "
        + json.dumps(initial_containment_audit)
    )
print(
    "[swamp-fluid] pool-initialization "
    + json.dumps(
        {"particles": len(pool_points), **pool_initialization_report},
        ensure_ascii=False,
    ),
    flush=True,
)
if args.initialization_only:
    simulation_app.close()
    raise SystemExit(0)
velocities = np.zeros_like(pool_points)
# Each particle represents one cell in the rest-spacing lattice.  Using the
# volume of a sphere with radius=fluidRestOffset underestimates bulk mass by
# roughly 48 percent for this cubic initialization.
particle_mass = 1000.0 * args.spacing**3
particles_path = Sdf.Path("/World/WaterParticles")
particles_prim = particleUtils.add_physx_particleset_pointinstancer(
    stage,
    particles_path,
    Vt.Vec3fArray.FromNumpy(pool_points),
    Vt.Vec3fArray.FromNumpy(velocities),
    particle_system_path,
    self_collision=True,
    fluid=True,
    particle_group=0,
    particle_mass=particle_mass,
    density=1000.0,
)
particle_instancer = UsdGeom.PointInstancer.Get(stage, particles_path)
particle_ids_sha256 = None
if args.export_whitewater_source:
    # PhysX preserves particle-set array order; that implicit index is the
    # identity consumed by all downstream caches.  The audit verifies it through
    # per-index native-velocity/displacement continuity over every interval.
    particle_ids = np.arange(len(pool_points), dtype="<i8")
    particle_ids_sha256 = hashlib.sha256(particle_ids.tobytes()).hexdigest()
particles_prim.CreateAttribute("physxParticle:maxParticles", Sdf.ValueTypeNames.Int).Set(
    len(pool_points)
)
diffuse_api = None
if args.diffuse_probe or args.enable_physx_diffuse:
    particleUtils.add_physx_diffuse_particles(
        stage,
        particles_path,
        enabled=args.enable_physx_diffuse and args.diffuse_enable_before_settle,
        max_diffuse_particle_multiplier=args.diffuse_max_multiplier,
        threshold=args.diffuse_threshold,
        lifetime=args.diffuse_lifetime,
        air_drag=args.diffuse_air_drag,
        bubble_drag=args.diffuse_bubble_drag,
        buoyancy=args.diffuse_buoyancy,
        kinetic_energy_weight=args.diffuse_kinetic_energy_weight,
        pressure_weight=args.diffuse_pressure_weight,
        divergence_weight=args.diffuse_divergence_weight,
        collision_decay=args.diffuse_collision_decay,
        use_accurate_velocity=args.diffuse_use_accurate_velocity,
    )
    diffuse_api = PhysxSchema.PhysxDiffuseParticlesAPI.Get(
        stage, particles_path
    )
    if not diffuse_api:
        raise RuntimeError("PhysX diffuse API did not apply to the water particle set")
if args.diffuse_debug_visualization:
    settings = carb.settings.get_settings()
    settings.set(physx_settings_bindings.SETTING_DISPLAY_PARTICLES, True)
    settings.set(physx_settings_bindings.SETTING_DISPLAY_PARTICLES_SHOW_DIFFUSE, True)
    settings.set(
        physx_settings_bindings.SETTING_DISPLAY_PARTICLES_SHOW_PARTICLE_SET_PARTICLES,
        False,
    )
    settings.set(
        physx_settings_bindings.SETTING_DISPLAY_PARTICLES_SHOW_FLUID_SURFACE, False
    )
prototype = UsdGeom.Imageable.Get(
    stage, particles_path.AppendChild("particlePrototype0")
)
if prototype and not args.debug_render_particles:
    prototype.MakeInvisible()
elif prototype:
    prototype.MakeVisible()
    UsdGeom.Sphere(prototype.GetPrim()).CreateRadiusAttr().Set(0.42 * args.spacing)

particleUtils.add_physx_particle_smoothing(
    stage,
    particle_system_path,
    enabled=not args.debug_render_particles,
    strength=args.particle_smoothing_strength,
)
particleUtils.add_physx_particle_anisotropy(
    stage,
    particle_system_path,
    enabled=args.enable_anisotropy and not args.debug_render_particles,
    scale=1.08,
    min=0.95,
    max=1.30,
)
particleUtils.add_physx_particle_isosurface(
    stage,
    particle_system_path,
    enabled=not args.debug_render_particles,
    grid_spacing=0.5 * args.spacing,
    num_mesh_smoothing_passes=2,
    num_mesh_normal_smoothing_passes=args.mesh_normal_smoothing_passes,
    max_vertices=4_000_000,
    max_triangles=8_000_000,
    max_subgrids=16384,
)

CreateAndBindMdlMaterialFromLibrary(
    mdl_name="OmniPBR.mdl" if args.debug_opaque_water else "OmniGlass.mdl",
    mtl_name="OmniPBR" if args.debug_opaque_water else "OmniGlass",
    bind_selected_prims=False,
    prim_name="SwampWaterRender",
).do()
water_render_path = Sdf.Path("/World/Looks/SwampWaterRender")
water_shader = UsdShade.Shader.Get(stage, water_render_path.AppendChild("Shader"))
if args.debug_opaque_water:
    water_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.05, 0.65, 0.22)
    )
    water_shader.CreateInput(
        "reflection_roughness_constant", Sdf.ValueTypeNames.Float
    ).Set(0.18)
else:
    water_shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.76, 0.90, 0.80)
    )
    water_shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
    water_shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(
        args.glass_roughness
    )
    # The render mesh is the air/water interface, not a closed glass solid.
    water_shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    water_shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.85)
if args.debug_render_particles:
    bind_material(particles_path.AppendChild("particlePrototype0"), water_render_path)
else:
    bind_material(particle_system_path, water_render_path)

shoreline_render_mesh = None
if not args.debug_render_particles:
    shoreline_render_mesh = UsdGeom.Mesh.Define(stage, "/World/WaterSurfaceClipped")
    shoreline_render_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    shoreline_render_mesh.CreateDoubleSidedAttr().Set(True)
    bind_material(shoreline_render_mesh.GetPath(), water_render_path)


def update_shoreline_clipped_surface(iso_mesh, iso_array):
    """Extract the local top envelope from PhysX's closed liquid volume."""
    global LAST_SURFACE_EXTRACTION_DIAGNOSTICS
    if shoreline_render_mesh is None or not len(iso_array):
        return 0
    counts_value = iso_mesh.GetFaceVertexCountsAttr().Get()
    indices_value = iso_mesh.GetFaceVertexIndicesAttr().Get()
    if counts_value is None or indices_value is None:
        return 0
    counts = np.asarray(counts_value, dtype=np.int32)
    indices = np.asarray(indices_value, dtype=np.int64)
    if not len(counts) or not np.all(counts == 3) or len(indices) != 3 * len(counts):
        raise RuntimeError("Unexpected PhysX isosurface topology")
    triangles = indices.reshape(-1, 3)
    centroids = iso_array[triangles].mean(axis=1)
    dynamic_core = np.linalg.norm(
        centroids[:, (0, 2)] - POOL_CENTER[None, :], axis=1
    ) <= max(0.40, 5.0 * SPHERE_RADIUS)
    ix = np.rint((centroids[:, 0] - RENDER_WET_XS[0]) / args.spacing).astype(np.int64)
    iz = np.rint((centroids[:, 2] - RENDER_WET_ZS[0]) / args.spacing).astype(np.int64)
    inside = (
        (ix >= 0)
        & (ix < len(RENDER_WET_XS))
        & (iz >= 0)
        & (iz < len(RENDER_WET_ZS))
    )
    wet = np.zeros(len(triangles), dtype=bool)
    wet[inside] = RENDER_WET_MASK[ix[inside], iz[inside]]

    # PhysX emits a closed isosurface.  Selecting merely upward-facing faces is
    # insufficient: the sloping closure skin around a shallow shore also has a
    # positive Y normal and appears as liquid climbing the Terrain.  Build a
    # horizontal top envelope and retain only faces near the highest surface in
    # each sampled column.  Around the impact, also retain above-water side
    # faces so crowns and near-vertical splash sheets remain visible.
    flat_cell = ix * len(RENDER_WET_ZS) + iz
    top_y = np.full(len(RENDER_WET_XS) * len(RENDER_WET_ZS), -np.inf)
    valid_cells = inside & wet
    np.maximum.at(top_y, flat_cell[valid_cells], centroids[valid_cells, 1])
    local_top = np.full(len(triangles), -np.inf)
    local_top[valid_cells] = top_y[flat_cell[valid_cells]]
    top_shell = centroids[:, 1] >= local_top - 1.5 * args.spacing
    dynamic_splash = dynamic_core & (
        centroids[:, 1] >= WATER_LEVEL - max(0.04, 5.0 * args.spacing)
    )
    keep = np.zeros(len(triangles), dtype=bool)
    keep[valid_cells] = (top_shell | dynamic_splash)[valid_cells]
    kept = triangles[keep]
    kept_centroid_y = centroids[keep, 1]
    LAST_SURFACE_EXTRACTION_DIAGNOSTICS = {
        "kept_faces": int(len(kept)),
        "centroid_y_quantiles": np.quantile(
            kept_centroid_y, (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0)
        ).astype(float).tolist(),
        "near_impact_faces": int(np.count_nonzero(keep & dynamic_core)),
    }
    shoreline_render_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(iso_array))
    shoreline_render_mesh.GetFaceVertexCountsAttr().Set([3] * len(kept))
    shoreline_render_mesh.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(kept.reshape(-1), dtype=np.int32))
    )
    UsdGeom.Imageable(iso_mesh.GetPrim()).MakeInvisible()
    return int(len(kept))

# A buoyant rigid sphere makes two-way interaction and the impact splash visible.
sphere = UsdGeom.Sphere.Define(stage, "/World/DropSphere")
sphere.CreateRadiusAttr().Set(SPHERE_RADIUS)
sphere_xform = UsdGeom.Xformable(sphere)
sphere_xform.AddTranslateOp().Set(
    Gf.Vec3d(float(POOL_CENTER[0]), args.sphere_start_y, float(POOL_CENTER[1]))
)
UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr().Set(True)
rigid_api = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
rigid_api.CreateRigidBodyEnabledAttr().Set(True)
kinematic_attr = rigid_api.CreateKinematicEnabledAttr()
kinematic_attr.Set(True)
mass_api = UsdPhysics.MassAPI.Apply(sphere.GetPrim())
mass_api.CreateMassAttr().Set(SPHERE_MASS)
force_api = PhysxSchema.PhysxForceAPI.Apply(sphere.GetPrim())
force_attr = force_api.CreateForceAttr()
force_api.CreateTorqueAttr().Set(Gf.Vec3f(0.0))
force_api.CreateModeAttr().Set("force")
force_api.CreateForceEnabledAttr().Set(True)
force_api.CreateWorldFrameEnabledAttr().Set(True)
last_hydrodynamic_force = np.zeros(3, dtype=np.float64)


def update_hydrodynamic_force():
    """Optionally add analytic buoyancy/drag on top of native particle contacts."""
    global last_hydrodynamic_force
    if not args.analytic_rigid_coupling:
        last_hydrodynamic_force.fill(0.0)
        force_attr.Set(Gf.Vec3f(0.0))
        return
    center_y = float(
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(sphere.GetPrim())
        .ExtractTranslation()[1]
    )
    velocity_value = rigid_api.GetVelocityAttr().Get()
    velocity = (
        np.asarray(velocity_value, dtype=np.float64)
        if velocity_value is not None
        else np.zeros(3, dtype=np.float64)
    )
    radius = SPHERE_RADIUS
    height = float(np.clip(WATER_LEVEL - (center_y - radius), 0.0, 2.0 * radius))
    submerged_volume = math.pi * height**2 * (radius - height / 3.0)
    buoyancy = np.asarray((0.0, 1000.0 * 9.81 * submerged_volume, 0.0))
    area = math.pi * radius**2 * (height / (2.0 * radius))
    speed = float(np.linalg.norm(velocity))
    drag = -0.5 * 1000.0 * 0.47 * area * speed * velocity
    last_hydrodynamic_force = buoyancy + drag
    force_attr.Set(Gf.Vec3f(*last_hydrodynamic_force.astype(float)))

created = []
omni.kit.commands.execute(
    "CreateAndBindMdlMaterialFromLibrary",
    mdl_name="OmniPBR.mdl",
    mtl_name="OmniPBR",
    mtl_created_list=created,
    bind_selected_prims=False,
    select_new_prim=False,
)
sphere_material_path = Sdf.Path(created[0])
sphere_shader = UsdShade.Shader.Get(
    stage, sphere_material_path.AppendChild("Shader")
)
sphere_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(0.95, 0.18, 0.035)
)
sphere_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
    0.24
)
bind_material(sphere.GetPath(), sphere_material_path)

dome = UsdLux.DomeLight.Define(stage, "/World/Lights/SwampHDRI")
dome.CreateIntensityAttr().Set(900.0)
dome.CreateColorAttr().Set(Gf.Vec3f(0.75, 0.82, 1.0))
dome.CreateTextureFileAttr().Set(
    Sdf.AssetPath(r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr")
)
dome.CreateTextureFormatAttr().Set("latlong")
key = UsdLux.DiskLight.Define(stage, "/World/Lights/PreviewKey")
key.CreateIntensityAttr().Set(7500.0)
key.CreateRadiusAttr().Set(1.8)
key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.86, 0.70))
set_camera(
    key.GetPrim(),
    Gf.Vec3d(-2.2, 3.8, 2.5),
    Gf.Vec3d(float(POOL_CENTER[0]), -1.1, float(POOL_CENTER[1])),
)

camera = UsdGeom.Camera.Define(stage, "/World/RenderCamera")
camera.CreateFocalLengthAttr().Set(40.0)
camera.CreateHorizontalApertureAttr().Set(24.0)
camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
set_camera(
    camera.GetPrim(),
    Gf.Vec3d(
        float(POOL_CENTER[0] + 0.85),
        1.15,
        float(POOL_CENTER[1] - 2.35),
    ),
    Gf.Vec3d(float(POOL_CENTER[0]), WATER_LEVEL - 0.03, float(POOL_CENTER[1])),
)

viewport = get_active_viewport()
viewport.camera_path = str(camera.GetPath())
viewport.set_texture_resolution((args.resolution, args.resolution))

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", "PathTracing")
settings.set("/persistent/app/viewport/displayOptions", 0)
settings.set(physx_settings_bindings.SETTING_UPDATE_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_UPDATE_VELOCITIES_TO_USD, True)
settings.set(physx_settings_bindings.SETTING_ENABLE_PARTICLE_AUTHORING, True)
settings.set("/rtx/pathtracing/spp", args.samples)
settings.set("/rtx/pathtracing/totalSpp", args.samples)
settings.set("/rtx/pathtracing/maxBounces", 12)
settings.set("/rtx/translucency/maxRefractionBounces", 12)
settings.set("/rtx/reflections/enabled", True)

stage_path = args.output / "swamp_physx_fluid_preview.usda"
stage.GetRootLayer().Export(str(stage_path))

simulation = get_physx_simulation_interface()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
simulation.attach_stage(stage_id)
native_diffuse_manifest = None
if native_diffuse_bridge is not None:
    native_particle_system_before = native_diffuse_bridge.particle_system_info(
        str(particle_system_path)
    )
    native_diffuse_bridge.set_full_diffuse_advection(
        str(particle_system_path), True
    )
    native_particle_system = native_diffuse_bridge.particle_system_info(
        str(particle_system_path)
    )
    if not native_particle_system["full_diffuse_advection"]:
        raise RuntimeError("PhysX rejected the full Diffuse advection flag")
    native_diffuse_manifest = {
        "schema": 1,
        "product": "swamp_physx_native_diffuse_frames",
        "created_utc": utc_now_iso(),
        "state": {
            "complete": False,
            "captured_nonempty_frames": 0,
            "updated_utc": utc_now_iso(),
        },
        "contract": {
            "source": "live PxParticleAndDiffuseBuffer after fetchResults",
            "native_bridge_abi": native_diffuse_bridge.abi_info(),
            "particle_set_path": str(particles_path),
            "particle_system_path": str(particle_system_path),
            "empty_output_frames_are_not_written": True,
            "labels_in_raw_cache": False,
            "label_policy": (
                "Recover later from same-frame native primary positions using "
                "the pinned PhysX 5.9 strict-radius thresholds; never infer from "
                "height or water-surface position."
            ),
        },
        "particle_system_before_configuration": native_particle_system_before,
        "particle_system": native_particle_system,
        "configuration": {
            "particle_spacing_m": args.spacing,
            "physics_fps": args.physics_fps,
            "output_fps": 30,
            "output_frames": args.frames,
            "full_diffuse_advection": True,
            "diffuse_max_multiplier": args.diffuse_max_multiplier,
            "diffuse_threshold": args.diffuse_threshold,
            "diffuse_lifetime_s": args.diffuse_lifetime,
        },
        "frames": [],
    }
    atomic_write_json(native_diffuse_manifest_path, native_diffuse_manifest)
capture_stride = args.physics_fps // 30
metrics = []
previous_sphere_center = None
reference_local_particle_p99_y = None
reference_local_isosurface_max_y = None
reference_local_isosurface_p999_y = None
whitewater_manifest = None
whitewater_manifest_path = whitewater_source_dir / "manifest.json"
whitewater_sample_cursor = 0
total_physics_steps = args.frames * capture_stride
whitewater_impact_end_step = min(
    total_physics_steps,
    int(round(args.whitewater_impact_seconds * args.physics_fps)),
)
whitewater_impact_stride = args.physics_fps // args.whitewater_impact_fps
whitewater_tail_stride = args.physics_fps // args.whitewater_tail_fps
whitewater_uniform_stride = (
    args.physics_fps // args.whitewater_source_fps
    if args.whitewater_source_fps is not None
    else None
)
whitewater_progress_fps = (
    args.whitewater_source_fps
    if args.whitewater_source_fps is not None
    else args.whitewater_impact_fps
)


def should_capture_whitewater_step(physics_step):
    if whitewater_uniform_stride is not None:
        return physics_step % whitewater_uniform_stride == 0
    if physics_step <= whitewater_impact_end_step:
        return physics_step % whitewater_impact_stride == 0
    return physics_step % whitewater_tail_stride == 0


whitewater_expected_steps = (
    [
        physics_step
        for physics_step in range(total_physics_steps + 1)
        if should_capture_whitewater_step(physics_step)
    ]
    if args.export_whitewater_source
    else []
)


def capture_native_diffuse(output_frame, physics_step):
    if native_diffuse_bridge is None:
        return
    metadata = native_diffuse_bridge.probe(str(particles_path))
    active_count = int(metadata["active_count"])
    if active_count == 0:
        return
    sample = native_diffuse_bridge.read_particle_frame(str(particles_path))
    primary_positions = sample["primary_position_inv_mass"]
    primary_velocities = sample["primary_velocity"]
    diffuse_positions = sample["diffuse_position_lifetime"]
    diffuse_velocities = sample["diffuse_velocity"]
    primary_count = int(sample["primary_active_count"])
    diffuse_count = int(sample["diffuse_active_count"])
    checks = {
        "probe_count_matches_copy": active_count == diffuse_count,
        "primary_count_matches_authored": primary_count == len(pool_points),
        "primary_positions_shape": list(primary_positions.shape) == [primary_count, 4],
        "primary_velocities_shape": list(primary_velocities.shape) == [primary_count, 4],
        "diffuse_positions_shape": list(diffuse_positions.shape) == [diffuse_count, 4],
        "diffuse_velocities_shape": list(diffuse_velocities.shape) == [diffuse_count, 4],
        "primary_positions_finite": bool(np.isfinite(primary_positions).all()),
        "primary_velocities_finite": bool(np.isfinite(primary_velocities).all()),
        "diffuse_positions_finite": bool(np.isfinite(diffuse_positions).all()),
        "diffuse_velocities_finite": bool(np.isfinite(diffuse_velocities).all()),
        "lifetime_in_physx_kernel_range": bool(
            np.min(diffuse_positions[:, 3]) > 0.0
            and np.max(diffuse_positions[:, 3])
            <= max(1.0, args.diffuse_lifetime) + 1.0e-4
        ),
    }
    structural_check_names = (
        "probe_count_matches_copy",
        "primary_count_matches_authored",
        "primary_positions_shape",
        "primary_velocities_shape",
        "diffuse_positions_shape",
        "diffuse_velocities_shape",
        "primary_positions_finite",
        "primary_velocities_finite",
        "diffuse_positions_finite",
        "diffuse_velocities_finite",
    )
    if not all(checks[name] for name in structural_check_names):
        raise RuntimeError(
            "Native Diffuse frame failed its structural contract: "
            f"frame={output_frame}, checks={checks}"
        )
    output_path = native_diffuse_dir / f"native_diffuse_{output_frame:04d}.npz"
    atomic_write_npz(
        output_path,
        schema=np.asarray("physx_diffuse_native_frame/1"),
        output_frame=np.asarray(output_frame, dtype=np.int64),
        physics_step=np.asarray(physics_step, dtype=np.int64),
        simulation_time=np.asarray(physics_step / args.physics_fps, dtype=np.float64),
        particle_set_path=np.asarray(str(particles_path)),
        particle_system_path=np.asarray(str(particle_system_path)),
        particle_contact_offset=np.asarray(
            native_diffuse_manifest["particle_system"]["particle_contact_offset"],
            dtype=np.float32,
        ),
        diffuse_neighbor_radius=np.asarray(
            native_diffuse_manifest["particle_system"]["diffuse_neighbor_radius"],
            dtype=np.float32,
        ),
        full_diffuse_advection=np.asarray(True, dtype=np.bool_),
        primary_active_count=np.asarray(primary_count, dtype=np.uint32),
        primary_max_count=np.asarray(sample["primary_max_count"], dtype=np.uint32),
        diffuse_active_count=np.asarray(diffuse_count, dtype=np.uint32),
        diffuse_max_count=np.asarray(sample["diffuse_max_count"], dtype=np.uint32),
        primary_position_inv_mass=np.ascontiguousarray(primary_positions),
        primary_velocity=np.ascontiguousarray(primary_velocities),
        diffuse_position_lifetime=np.ascontiguousarray(diffuse_positions),
        diffuse_velocity=np.ascontiguousarray(diffuse_velocities),
    )
    record = {
        "output_frame": output_frame,
        "physics_step": physics_step,
        "simulation_time_s": physics_step / args.physics_fps,
        "file": output_path.name,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "primary_active_count": primary_count,
        "primary_max_count": int(sample["primary_max_count"]),
        "diffuse_active_count": diffuse_count,
        "diffuse_max_count": int(sample["diffuse_max_count"]),
        "remaining_lifetime_range_s": [
            float(np.min(diffuse_positions[:, 3])),
            float(np.max(diffuse_positions[:, 3])),
        ],
        "speed_range_mps": [
            float(np.min(np.linalg.norm(diffuse_velocities[:, :3], axis=1))),
            float(np.max(np.linalg.norm(diffuse_velocities[:, :3], axis=1))),
        ],
        "checks": checks,
        "semantic_warnings": (
            []
            if checks["lifetime_in_physx_kernel_range"]
            else ["remaining_lifetime_outside_physx_kernel_range"]
        ),
    }
    native_diffuse_manifest["frames"].append(record)
    native_diffuse_manifest["state"].update(
        {
            "captured_nonempty_frames": len(native_diffuse_manifest["frames"]),
            "updated_utc": utc_now_iso(),
        }
    )
    atomic_write_json(native_diffuse_manifest_path, native_diffuse_manifest)


def rigid_vec3(attribute, name):
    value = attribute.Get()
    if value is None:
        raise RuntimeError(f"PhysX did not expose native rigid {name}")
    result = np.asarray(value, dtype="<f4")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise RuntimeError(f"Native rigid {name} is invalid: {result!r}")
    return np.ascontiguousarray(result)


def capture_whitewater_source(physics_step):
    global whitewater_sample_cursor
    if not args.export_whitewater_source:
        return
    if whitewater_sample_cursor >= len(whitewater_expected_steps):
        raise RuntimeError("Whitewater source produced more samples than planned")
    expected_step = whitewater_expected_steps[whitewater_sample_cursor]
    if physics_step != expected_step:
        raise RuntimeError(
            "Whitewater source schedule mismatch: "
            f"expected physics step {expected_step}, received {physics_step}"
        )

    positions_value = particle_instancer.GetPositionsAttr().Get()
    velocities_value = particle_instancer.GetVelocitiesAttr().Get()
    if positions_value is None:
        raise RuntimeError("PhysX did not expose PointInstancer positions")
    if velocities_value is None:
        raise RuntimeError(
            "PhysX did not expose native PointInstancer velocities; "
            "finite-difference fallback is intentionally forbidden"
        )
    positions = np.ascontiguousarray(positions_value, dtype="<f4")
    particle_velocities = np.ascontiguousarray(velocities_value, dtype="<f4")
    expected_shape = (len(pool_points), 3)
    if positions.shape != expected_shape or particle_velocities.shape != expected_shape:
        raise RuntimeError(
            "Whitewater particle state shape changed: "
            f"positions={positions.shape}, velocities={particle_velocities.shape}, "
            f"expected={expected_shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(particle_velocities).all():
        raise RuntimeError("Whitewater particle state contains non-finite values")
    particle_speeds = np.linalg.norm(particle_velocities.astype(np.float64), axis=1)
    position_minimum = positions.min(axis=0).astype(float).tolist()
    position_maximum = positions.max(axis=0).astype(float).tolist()
    speed_quantiles = np.quantile(
        particle_speeds, (0.0, 0.5, 0.9, 0.99, 1.0)
    ).astype(float).tolist()

    sphere_transform_value = UsdGeom.XformCache().GetLocalToWorldTransform(
        sphere.GetPrim()
    )
    sphere_transform = np.asarray(
        [
            [float(sphere_transform_value[row][column]) for column in range(4)]
            for row in range(4)
        ],
        dtype="<f8",
    )
    if not np.isfinite(sphere_transform).all():
        raise RuntimeError("Sphere world transform contains non-finite values")
    sphere_linear_velocity = rigid_vec3(
        rigid_api.GetVelocityAttr(), "linear velocity"
    )
    sphere_angular_velocity = rigid_vec3(
        rigid_api.GetAngularVelocityAttr(), "angular velocity"
    )

    sample_index = whitewater_sample_cursor
    sample_path = whitewater_source_dir / f"source_{sample_index:06d}.npz"
    simulation_time = physics_step / args.physics_fps
    atomic_write_npz(
        sample_path,
        schema=np.asarray(1, dtype="<i4"),
        sample_index=np.asarray(sample_index, dtype="<i4"),
        physics_step=np.asarray(physics_step, dtype="<i8"),
        simulation_time=np.asarray(simulation_time, dtype="<f8"),
        positions=positions,
        velocities=particle_velocities,
        sphere_transform=sphere_transform,
        sphere_linear_velocity=sphere_linear_velocity,
        sphere_angular_velocity=sphere_angular_velocity,
    )
    sample_row = {
        "sample_index": sample_index,
        "physics_step": physics_step,
        "simulation_time": simulation_time,
        "file": sample_path.name,
        "bytes": sample_path.stat().st_size,
        "sha256": sha256_file(sample_path),
        "position_minimum_m": position_minimum,
        "position_maximum_m": position_maximum,
        "speed_quantiles_mps": {
            "minimum": speed_quantiles[0],
            "p50": speed_quantiles[1],
            "p90": speed_quantiles[2],
            "p99": speed_quantiles[3],
            "maximum": speed_quantiles[4],
        },
        "checks": {
            "particle_count_matches_contract": len(positions) == len(pool_points),
            "positions_shape_matches_contract": positions.shape == expected_shape,
            "velocities_shape_matches_contract": particle_velocities.shape
            == expected_shape,
            "positions_finite": bool(np.isfinite(positions).all()),
            "velocities_finite": bool(np.isfinite(particle_velocities).all()),
            "finite_difference_velocity_fallback": False,
        },
    }
    whitewater_manifest["samples"].append(sample_row)
    whitewater_sample_cursor += 1
    whitewater_manifest["state"].update(
        {
            "completed_samples": whitewater_sample_cursor,
            "updated_utc": utc_now_iso(),
        }
    )
    atomic_write_json(whitewater_manifest_path, whitewater_manifest)
    if sample_index == 0 or sample_index % max(1, whitewater_progress_fps) == 0:
        print(
            "[whitewater-source] "
            f"sample={sample_index:04d}/{len(whitewater_expected_steps) - 1:04d} "
            f"physics_step={physics_step} time={simulation_time:.6f}s",
            flush=True,
        )


if args.export_whitewater_source:
    whitewater_manifest = {
        "schema": 1,
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "created_utc": utc_now_iso(),
        "source": "raw PhysX PBD PointInstancer state immediately after fetch_results",
        "velocity_authority": (
            "PhysX-authored UsdGeom.PointInstancer velocities; no temporal "
            "finite-difference fallback"
        ),
        "coordinates": "Isaac/USD world XYZ in metres",
        "particle_count": len(pool_points),
        "particle_ids": {
            "storage": "implicit array index",
            "dtype": "int64",
            "sha256": particle_ids_sha256,
        },
        "arrays": {
            "positions": {"dtype": "float32", "shape": [len(pool_points), 3]},
            "velocities": {"dtype": "float32", "shape": [len(pool_points), 3]},
            "sphere_transform": {"dtype": "float64", "shape": [4, 4]},
            "sphere_linear_velocity": {"dtype": "float32", "shape": [3]},
            "sphere_angular_velocity": {"dtype": "float32", "shape": [3]},
        },
        "sampling": {
            "mode": (
                "uniform"
                if args.whitewater_source_fps is not None
                else "impact_then_tail"
            ),
            "physics_fps": args.physics_fps,
            "output_fps": 30,
            "uniform_source_fps": args.whitewater_source_fps,
            "uniform_timestep_s": (
                1.0 / args.whitewater_source_fps
                if args.whitewater_source_fps is not None
                else None
            ),
            "impact_fps": args.whitewater_impact_fps,
            "impact_seconds": args.whitewater_impact_seconds,
            "impact_end_physics_step": whitewater_impact_end_step,
            "tail_fps": args.whitewater_tail_fps,
            "total_physics_steps": total_physics_steps,
            "expected_physics_steps": whitewater_expected_steps,
        },
        "state": {
            "complete": False,
            "completed_samples": 0,
            "expected_samples": len(whitewater_expected_steps),
            "parent_run_valid": None,
            "updated_utc": utc_now_iso(),
        },
        "samples": [],
    }
    atomic_write_json(whitewater_manifest_path, whitewater_manifest)

diffuse_probe = None
if args.diffuse_probe:
    diffuse_probe = PhysxDiffuseProbe(
        stage=stage,
        scene_path=scene.GetPath(),
        particle_path=particles_path,
        output_path=args.output / "physx_diffuse_probe.json",
        enabled_requested=args.enable_physx_diffuse,
        configuration={
            "primary_particle_count": len(pool_points),
            "particle_spacing": args.spacing,
            "physics_fps": args.physics_fps,
            "frames": args.frames,
            "settle_frames": args.settle_frames,
            "max_diffuse_particle_multiplier": args.diffuse_max_multiplier,
            "maximum_diffuse_particles": int(
                math.floor(len(pool_points) * args.diffuse_max_multiplier)
            ),
            "threshold": args.diffuse_threshold,
            "lifetime": args.diffuse_lifetime,
            "air_drag": args.diffuse_air_drag,
            "bubble_drag": args.diffuse_bubble_drag,
            "buoyancy": args.diffuse_buoyancy,
            "kinetic_energy_weight": args.diffuse_kinetic_energy_weight,
            "pressure_weight": args.diffuse_pressure_weight,
            "divergence_weight": args.diffuse_divergence_weight,
            "collision_decay": args.diffuse_collision_decay,
            "use_accurate_velocity": args.diffuse_use_accurate_velocity,
        },
    )
try:
    # Keep the impactor fixed while the irregular terrain rejects overlapping
    # initial particles and the free surface relaxes.  The recorded timeline
    # starts only after this pre-roll.
    for settle_frame in range(args.settle_frames):
        simulation.simulate(
            1.0 / args.physics_fps,
            (settle_frame - args.settle_frames) / args.physics_fps,
        )
        simulation.fetch_results()
        simulation_app.update()
    stage.GetPrimAtPath(water_physics_path).GetAttribute(
        "physxPBDMaterial:damping"
    ).Set(args.damping)
    if diffuse_probe is not None:
        diffuse_probe.capture(
            0,
            0.0,
            "post_settle_diffuse_pre_enabled"
            if args.diffuse_enable_before_settle else "post_settle_diffuse_disabled",
        )
    if args.enable_physx_diffuse and not args.diffuse_enable_before_settle:
        if diffuse_api is None:
            raise RuntimeError("Diffuse enable requested without an applied API")
        diffuse_api.GetDiffuseParticlesEnabledAttr().Set(True)
    kinematic_attr.Set(False)
    update_hydrodynamic_force()
    for _ in range(8):
        simulation_app.update()
    if diffuse_probe is not None:
        diffuse_probe.capture(
            0,
            0.0,
            "release_diffuse_enabled"
            if args.enable_physx_diffuse
            else "release_diffuse_disabled",
        )
    capture_whitewater_source(0)
    for output_frame in range(args.frames):
        for subframe in range(capture_stride):
            sim_frame = output_frame * capture_stride + subframe
            update_hydrodynamic_force()
            simulation.simulate(1.0 / args.physics_fps, sim_frame / args.physics_fps)
            simulation.fetch_results()
            simulation_app.update()
            physics_step = sim_frame + 1
            if (
                args.export_whitewater_source
                and should_capture_whitewater_step(physics_step)
            ):
                capture_whitewater_source(physics_step)
        for _ in range(3):
            simulation_app.update()

        positions_value = UsdGeom.PointInstancer.Get(
            stage, particles_path
        ).GetPositionsAttr().Get()
        current = np.asarray(positions_value, dtype=np.float32)
        if len(current) != len(pool_points) or not np.isfinite(current).all():
            raise RuntimeError("Particle state became invalid")
        recorded_physics_step = (output_frame + 1) * capture_stride
        capture_native_diffuse(output_frame, recorded_physics_step)
        probe_interval = max(1, args.frames // 4)
        if diffuse_probe is not None and (
            output_frame < 3
            or output_frame == args.frames - 1
            or (output_frame + 1) % probe_interval == 0
        ):
            diffuse_probe.capture(
                recorded_physics_step,
                recorded_physics_step / args.physics_fps,
                f"recorded_output_frame_{output_frame:04d}",
            )
        if (
            args.export_particle_ply
            and output_frame % args.export_particle_stride == 0
        ):
            write_binary_particle_ply(
                particles_dir / f"particles_{output_frame:04d}.ply", current
            )
        iso_mesh = UsdGeom.Mesh.Get(stage, particle_system_path.AppendChild("Isosurface"))
        iso_points = iso_mesh.GetPointsAttr().Get() if iso_mesh else None
        iso_count = len(iso_points) if iso_points is not None else 0
        iso_array = (
            np.asarray(iso_points, dtype=np.float32)
            if iso_points is not None and iso_count
            else np.empty((0, 3), dtype=np.float32)
        )
        if output_frame >= 2 and iso_count == 0 and not args.debug_render_particles:
            raise RuntimeError("PhysX liquid isosurface is empty")
        clipped_surface_faces = (
            update_shoreline_clipped_surface(iso_mesh, iso_array)
            if not args.debug_render_particles
            else 0
        )
        for _ in range(2):
            simulation_app.update()

        frame_path = frames_dir / f"frame_{output_frame:04d}.png"
        if not args.no_capture:
            capture(viewport, frame_path)
        escaped_mask, containment_audit = audit_particle_containment(
            current,
            collision_audit_xs,
            collision_audit_zs,
            collision_audit_terrain_y,
        )
        escaped_particle_count = int(np.count_nonzero(escaped_mask))
        lowest_particle = current[int(np.argmin(current[:, 1]))]
        sphere_center = np.asarray(
            UsdGeom.XformCache()
            .GetLocalToWorldTransform(sphere.GetPrim())
            .ExtractTranslation(),
            dtype=np.float64,
        )
        sphere_velocity = (
            np.zeros(3, dtype=np.float64)
            if previous_sphere_center is None
            else (sphere_center - previous_sphere_center) * 30.0
        )
        local_radius = max(0.30, 3.0 * SPHERE_RADIUS)
        particle_planar_distance = np.linalg.norm(
            current[:, [0, 2]] - sphere_center[[0, 2]], axis=1
        )
        local_particles = current[particle_planar_distance <= local_radius]
        iso_planar_distance = (
            np.linalg.norm(iso_array[:, [0, 2]] - sphere_center[[0, 2]], axis=1)
            if iso_count
            else np.empty(0, dtype=np.float32)
        )
        local_iso = iso_array[iso_planar_distance <= local_radius]
        local_particle_p99_y = (
            float(np.percentile(local_particles[:, 1], 99.0))
            if len(local_particles)
            else None
        )
        local_isosurface_max_y = (
            float(local_iso[:, 1].max()) if len(local_iso) else None
        )
        # A raw maximum is useful for diagnosing detached reconstruction spikes,
        # but it is not a trustworthy measure of a visible splash.  P99.9 still
        # follows a thin crown while ignoring one-off isosurface vertices.
        local_isosurface_p999_y = (
            float(np.percentile(local_iso[:, 1], 99.9)) if len(local_iso) else None
        )
        if reference_local_particle_p99_y is None:
            reference_local_particle_p99_y = local_particle_p99_y
        if reference_local_isosurface_max_y is None:
            reference_local_isosurface_max_y = local_isosurface_max_y
        if reference_local_isosurface_p999_y is None:
            reference_local_isosurface_p999_y = local_isosurface_p999_y
        metrics.append(
            {
                "output_frame": output_frame,
                "particle_minimum": current.min(axis=0).astype(float).tolist(),
                "particle_maximum": current.max(axis=0).astype(float).tolist(),
                "isosurface_vertices": iso_count,
                "shoreline_clipped_surface_faces": clipped_surface_faces,
                "surface_extraction": LAST_SURFACE_EXTRACTION_DIAGNOSTICS,
                "isosurface_minimum": (
                    iso_array.min(axis=0).astype(float).tolist() if iso_count else None
                ),
                "isosurface_maximum": (
                    iso_array.max(axis=0).astype(float).tolist() if iso_count else None
                ),
                "escaped_particle_count": escaped_particle_count,
                "containment_audit": containment_audit,
                "lowest_particle": lowest_particle.astype(float).tolist(),
                "sphere_center": sphere_center.tolist(),
                "sphere_velocity_30fps_difference": sphere_velocity.tolist(),
                "hydrodynamic_force": last_hydrodynamic_force.tolist(),
                "local_radius": local_radius,
                "local_particle_count": int(len(local_particles)),
                "local_particle_p99_y": local_particle_p99_y,
                "local_particle_splash_height": (
                    local_particle_p99_y - reference_local_particle_p99_y
                    if local_particle_p99_y is not None
                    and reference_local_particle_p99_y is not None
                    else None
                ),
                "local_isosurface_vertices": int(len(local_iso)),
                "local_isosurface_max_y": local_isosurface_max_y,
                "local_isosurface_splash_height": (
                    local_isosurface_max_y - reference_local_isosurface_max_y
                    if local_isosurface_max_y is not None
                    and reference_local_isosurface_max_y is not None
                    else None
                ),
                "local_isosurface_p999_y": local_isosurface_p999_y,
                "local_isosurface_robust_splash_height": (
                    local_isosurface_p999_y - reference_local_isosurface_p999_y
                    if local_isosurface_p999_y is not None
                    and reference_local_isosurface_p999_y is not None
                    else None
                ),
                "spray_particle_count": int(
                    np.count_nonzero(current[:, 1] > WATER_LEVEL + args.spacing)
                ),
            }
        )
        previous_sphere_center = sphere_center
        if output_frame % 10 == 0 or output_frame == args.frames - 1:
            print(
                f"[swamp-fluid] frame={output_frame:03d}/{args.frames - 1:03d} "
                f"particles={len(current)} iso_vertices={iso_count}",
                flush=True,
            )
    if (
        args.export_whitewater_source
        and whitewater_sample_cursor != len(whitewater_expected_steps)
    ):
        raise RuntimeError(
            "Whitewater source sample count mismatch: "
            f"{whitewater_sample_cursor} != {len(whitewater_expected_steps)}"
        )
finally:
    simulation.detach_stage()

maximum_escaped_particles = max(row["escaped_particle_count"] for row in metrics)
maximum_outside_collision_domain = max(
    row["containment_audit"]["outside_collision_domain_count"] for row in metrics
)
maximum_missing_collision_terrain = max(
    row["containment_audit"]["missing_collision_terrain_count"] for row in metrics
)
maximum_terrain_penetrations = max(
    row["containment_audit"]["terrain_penetration_count"] for row in metrics
)
preimpact_metrics = [
    row
    for row in metrics
    if row["sphere_center"][1] - SPHERE_RADIUS > WATER_LEVEL
]
maximum_preimpact_spray = max(
    (row["spray_particle_count"] for row in preimpact_metrics), default=0
)
particle_count = len(pool_points)
maximum_escaped_fraction = maximum_escaped_particles / particle_count
maximum_preimpact_spray_fraction = maximum_preimpact_spray / particle_count
maximum_local_isosurface_splash = max(
    (
        row["local_isosurface_splash_height"]
        for row in metrics
        if row["local_isosurface_splash_height"] is not None
    ),
    default=0.0,
)
maximum_local_isosurface_robust_splash = max(
    (
        row["local_isosurface_robust_splash_height"]
        for row in metrics
        if row["local_isosurface_robust_splash_height"] is not None
    ),
    default=0.0,
)
impact_observed = any(
    row["sphere_center"][1] - SPHERE_RADIUS <= WATER_LEVEL for row in metrics
)
visible_splash = (
    max(
        (
            row["local_particle_splash_height"]
            for row in metrics
            if row["local_particle_splash_height"] is not None
        ),
        default=0.0,
    )
    >= args.spacing
    if args.debug_render_particles
    else maximum_local_isosurface_robust_splash >= args.spacing
)
fluid_audit = {
    # A formal cache is valid only if every particle remains over authored
    # collision terrain and none tunnels more than one particle spacing.
    "no_material_leakage": maximum_escaped_particles == 0,
    "maximum_escaped_particles": maximum_escaped_particles,
    "maximum_escaped_fraction": maximum_escaped_fraction,
    "maximum_outside_collision_domain_particles": maximum_outside_collision_domain,
    "maximum_missing_collision_terrain_particles": maximum_missing_collision_terrain,
    "maximum_terrain_penetration_particles": maximum_terrain_penetrations,
    "terrain_penetration_tolerance": args.spacing,
    "maximum_preimpact_spray_particles": maximum_preimpact_spray,
    "maximum_preimpact_spray_fraction": maximum_preimpact_spray_fraction,
    "stable_before_impact": maximum_preimpact_spray_fraction <= 0.001,
    "maximum_local_isosurface_splash_height": maximum_local_isosurface_splash,
    "maximum_local_isosurface_robust_splash_height": (
        maximum_local_isosurface_robust_splash
    ),
    "minimum_visible_splash_height": args.spacing,
    "splash_audit_applicable": impact_observed,
    "visible_splash": visible_splash if impact_observed else None,
}
physx_diffuse_report = None
if diffuse_probe is not None:
    completed_diffuse_probe = diffuse_probe.complete()
    diffuse_probe_path = args.output / "physx_diffuse_probe.json"
    physx_diffuse_report = {
        "api_applied": diffuse_api is not None,
        "enabled_during_recording": args.enable_physx_diffuse,
        "probe": str(diffuse_probe_path),
        "probe_sha256": sha256_file(diffuse_probe_path),
        "probe_complete": completed_diffuse_probe["state"]["complete"],
        "snapshots": completed_diffuse_probe["state"]["snapshots"],
        "public_buffer_export_discovered": completed_diffuse_probe["discovery"][
            "public_buffer_export_discovered"
        ],
    }

native_diffuse_complete = True
native_diffuse_report = None
if native_diffuse_manifest is not None:
    native_diffuse_complete = bool(native_diffuse_manifest["frames"])
    native_diffuse_manifest["state"].update(
        {
            "complete": native_diffuse_complete,
            "captured_nonempty_frames": len(native_diffuse_manifest["frames"]),
            "completed_utc": utc_now_iso(),
            "updated_utc": utc_now_iso(),
        }
    )
    atomic_write_json(native_diffuse_manifest_path, native_diffuse_manifest)
    native_diffuse_report = {
        "directory": str(native_diffuse_dir),
        "manifest": str(native_diffuse_manifest_path),
        "manifest_sha256": sha256_file(native_diffuse_manifest_path),
        "complete": native_diffuse_complete,
        "captured_nonempty_frames": len(native_diffuse_manifest["frames"]),
        "labels_in_raw_cache": False,
    }

whitewater_source_complete = (
    not args.export_whitewater_source
    or whitewater_sample_cursor == len(whitewater_expected_steps)
)
run_valid = all(
    (
        fluid_audit["no_material_leakage"],
        fluid_audit["stable_before_impact"],
        fluid_audit["visible_splash"] if impact_observed else True,
        whitewater_source_complete,
        native_diffuse_complete,
    )
)
whitewater_source_report = None
if args.export_whitewater_source:
    whitewater_manifest["state"].update(
        {
            "complete": whitewater_source_complete,
            "completed_samples": whitewater_sample_cursor,
            "parent_run_valid": run_valid,
            "completed_utc": utc_now_iso(),
            "updated_utc": utc_now_iso(),
        }
    )
    atomic_write_json(whitewater_manifest_path, whitewater_manifest)
    whitewater_source_report = {
        "schema": whitewater_manifest["schema"],
        "directory": str(whitewater_source_dir),
        "manifest": str(whitewater_manifest_path),
        "manifest_sha256": sha256_file(whitewater_manifest_path),
        "complete": whitewater_source_complete,
        "samples": whitewater_sample_cursor,
        "expected_samples": len(whitewater_expected_steps),
        "sampling_mode": whitewater_manifest["sampling"]["mode"],
        "uniform_source_fps": args.whitewater_source_fps,
        "native_particle_velocities": True,
        "finite_difference_velocity_fallback": False,
    }
report = {
    "valid": run_valid,
    "physics": "PhysX GPU PBD fluid",
    "surface": (
        "raw PhysX particle spheres"
        if args.debug_render_particles
        else "PhysX native isosurface with render-only free-surface extraction"
    ),
    "render_mode": "raw_particles" if args.debug_render_particles else "isosurface",
    "authored_water_prim": WATER_PRIM if authored_water_present else None,
    "authored_water_render_role": (
        "hidden" if authored_water_present else "deleted_in_source_usd"
    ),
    "terrain_visual_hole": terrain_visual_report,
    "basin_collision": basin_collision_report,
    "particle_count": len(pool_points),
    "particle_cache": {
        "format": "binary_little_endian_ply" if args.export_particle_ply else None,
        "stride": args.export_particle_stride if args.export_particle_ply else None,
        "directory": str(particles_dir) if args.export_particle_ply else None,
    },
    "whitewater_source_cache": whitewater_source_report,
    "physx_diffuse_particles": physx_diffuse_report,
    "native_diffuse_cache": native_diffuse_report,
    "pool_initialization": pool_initialization_report,
    "water_level": {
        "authored": AUTHORED_WATER_LEVEL,
        "offset": args.water_level_offset,
        "simulated": WATER_LEVEL,
    },
    "particle_mass": particle_mass,
    "fluid_material": {
        "damping": args.damping,
        "settle_damping": args.settle_damping,
        "viscosity": args.viscosity,
        "vorticity_confinement": args.vorticity_confinement,
        "surface_tension": args.surface_tension,
        "cohesion": args.cohesion,
    },
    "solver_iterations": args.solver_iterations,
    "particle_smoothing_strength": args.particle_smoothing_strength,
    "surface_reconstruction": {
        "mesh_smoothing_passes": 2,
        "normal_smoothing_passes": args.mesh_normal_smoothing_passes,
        "glass_roughness": args.glass_roughness,
        "anisotropy_enabled": args.enable_anisotropy,
        "shoreline_render_mode": "local_top_envelope_with_dynamic_impact_core",
        "dynamic_impact_core_radius": max(0.40, 5.0 * SPHERE_RADIUS),
    },
    "particle_spacing": args.spacing,
    "physics_fps": args.physics_fps,
    "output_fps": 30,
    "frames": args.frames,
    "resolution": [args.resolution, args.resolution],
    "path_samples": args.samples,
    "impactor": {"shape": "sphere", "radius": SPHERE_RADIUS, "mass": SPHERE_MASS},
    "rigid_fluid_coupling": (
        "native PhysX particle contacts plus analytic submerged-volume buoyancy and drag"
        if args.analytic_rigid_coupling
        else "native PhysX particle contacts only"
    ),
    "maximum_escaped_particles": maximum_escaped_particles,
    "fluid_audit": fluid_audit,
    "metrics": metrics,
}
(args.output / "run_complete.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))
if not report["valid"]:
    raise RuntimeError(f"Swamp fluid audit failed: {json.dumps(fluid_audit)}")
simulation_app.close()
