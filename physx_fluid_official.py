"""Headless Isaac Sim 6.0 PhysX fluid post-processing test.

Uses NVIDIA's own particle smoothing, anisotropy and isosurface pipeline.
It does not use the custom Warp/Blender surface reconstruction.
"""

import argparse
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=240)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--nx", type=int, default=18)
parser.add_argument("--ny", type=int, default=24)
parser.add_argument("--nz", type=int, default=18)
parser.add_argument("--output", default=r"Y:\isaacsim_work\output\official_fluid")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": args.width,
        "height": args.height,
    }
)

import carb
import omni.kit.app
import omni.kit.commands
import omni.usd
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdUtils, Vt


def set_camera(camera, eye, target):
    matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 1, 0)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)


extension_manager = omni.kit.app.get_app().get_extension_manager()
extension_manager.set_extension_enabled_immediate("omni.physx.demos", True)

from omni.physxdemos.scenes.ParticlePostProcessingDemo import ParticlePostProcessingDemo
from omni.physxdemos.utils.room_helper import RoomHelper

os.makedirs(args.output, exist_ok=True)
stage_path = os.path.join(args.output, "official_fluid_scene.usda")

context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# Build NVIDIA's official particle post-processing demo with every surface stage enabled.
fluid_demo = ParticlePostProcessingDemo()
fluid_demo.create(stage, Anisotropy=True, Smoothing=True, Isosurface=True)

particle_system_path = Sdf.Path("/World/particleSystem")
particle_system = PhysxSchema.PhysxParticleSystem.Get(stage, particle_system_path)
if not particle_system:
    raise RuntimeError("Official demo did not create /World/particleSystem")

# Explicit values are NVIDIA's recommended isosurface/anisotropy tuning from the demo.
anisotropy = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
anisotropy.CreateScaleAttr().Set(5.0)
anisotropy.CreateMinAttr().Set(1.0)
anisotropy.CreateMaxAttr().Set(2.0)
PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())
PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())

# Replace the demo's ~42k-particle cube with a configurable smaller block.
contact_offset = float(particle_system.GetParticleContactOffsetAttr().Get())
rest_offset = 0.99 * 0.6 * contact_offset
spacing = 2.0 * rest_offset
positions, velocities = particleUtils.create_particles_grid(
    Gf.Vec3f(
        -0.5 * (args.nx - 1) * spacing,
        -0.5 * (args.ny - 1) * spacing,
        -0.5 * (args.nz - 1) * spacing,
    ),
    spacing,
    args.nx,
    args.ny,
    args.nz,
)

instancer = UsdGeom.PointInstancer.Get(stage, "/World/particles")
instancer.GetPositionsAttr().Set(Vt.Vec3fArray(positions))
instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray(velocities))
instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(positions)))
physicsUtils.set_or_add_translate_op(instancer, Gf.Vec3f(-18.0, 34.0, 0.0))
# The point instancer remains the physical state, but must not be rendered on
# top of the generated isosurface.
instancer.MakeInvisible()

# Keep the catch tank collision active while hiding its demo visualization.
tank = UsdGeom.Imageable.Get(stage, "/World/box")
if tank:
    tank.MakeInvisible()

# Bind NVIDIA's refractive OmniGlass to the generated isosurface.
glass_path = RoomHelper.get_glass_material(stage)
omni.kit.commands.execute(
    "BindMaterialCommand",
    prim_path=particle_system_path,
    material_path=glass_path,
    strength=None,
)

# Camera looks into the official catch tank.
camera = UsdGeom.Camera.Define(stage, "/World/RenderCamera")
camera.CreateFocalLengthAttr(45.0)
camera.CreateHorizontalApertureAttr(24.0)
set_camera(
    camera.GetPrim(),
    Gf.Vec3d(100.0, 70.0, 115.0),
    Gf.Vec3d(0.0, 20.0, 0.0),
)
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

# RTX settings needed for a refractive liquid surface.
settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", "RaytracedLighting")
settings.set("/rtx/translucency/maxRefractionBounces", 12)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/indirectDiffuse/enabled", True)

stage.GetRootLayer().Export(stage_path)

viewport = get_active_viewport()
viewport.camera_path = camera.GetPath()
viewport.set_texture_resolution((args.width, args.height))

simulation = get_physx_simulation_interface()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
simulation.attach_stage(stage_id)

print(f"[fluid] particles={len(positions)}, frames={args.frames}, output={args.output}")
for frame in range(args.frames):
    simulation.simulate(1.0 / 60.0, frame / 60.0)
    simulation.fetch_results()
    simulation_app.update()

    frame_path = os.path.join(args.output, f"frame_{frame:04d}.png")
    capture = capture_viewport_to_file(viewport, file_path=frame_path)
    # Capture is asynchronous. App updates advance rendering but not physics because
    # the timeline remains stopped and physics is stepped explicitly above.
    for _ in range(30):
        simulation_app.update()
        if os.path.exists(frame_path):
            break
    if not os.path.exists(frame_path):
        raise RuntimeError(f"Timed out capturing {frame_path}")
    if frame % 30 == 0:
        print(f"[fluid] frame {frame}/{args.frames}")

simulation.detach_stage()
fluid_demo.on_shutdown()
simulation_app.close()
