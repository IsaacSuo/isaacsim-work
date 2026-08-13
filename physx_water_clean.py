"""Clean meter-scale PhysX PBD water scene for Isaac Sim 6.0."""

import argparse
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--spacing", type=float, default=0.003)
parser.add_argument("--nx", type=int, default=60)
parser.add_argument("--ny", type=int, default=12)
parser.add_argument("--nz", type=int, default=20)
parser.add_argument("--output", default=r"Y:\isaacsim_work\output\water_clean")
parser.add_argument("--diagnostic-only", action="store_true")
parser.add_argument("--capture-every", type=int, default=1)
parser.add_argument("--substeps", type=int, default=8)
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
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from omni.physx import get_physx_simulation_interface
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdUtils, Vt


def set_camera(camera, eye, target):
    matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 1, 0)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)


def hide(prim):
    if prim:
        UsdGeom.Imageable(prim).MakeInvisible()


os.makedirs(args.output, exist_ok=True)
context = omni.usd.get_context()
context.new_stage()
stage = context.get_stage()

UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# Physics scene: GPU dynamics, SI gravity.
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
physx_scene.CreateEnableExternalForcesEveryIterationAttr().Set(True)

# White floor plus invisible tank walls. The walls remain colliders.
floor = physicsUtils.add_collider_cube(
    stage,
    "/World/TankFloor",
    Gf.Vec3f(1.40, 0.02, 0.90),
    Gf.Vec3f(0.0, -0.01, 0.0),
    color=Gf.Vec3f(0.55, 0.58, 0.62),
)

# Smooth visible rigid obstacle for judging splitting and runoff.
obstacle = UsdGeom.Sphere.Define(stage, "/World/FlowObstacle")
obstacle.CreateRadiusAttr().Set(0.06)
obstacle.AddTranslateOp().Set(Gf.Vec3d(0.06, 0.06, 0.0))
obstacle.CreateDisplayColorAttr().Set([Gf.Vec3f(0.12, 0.14, 0.18)])
UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())

walls = [
    physicsUtils.add_collider_cube(stage, "/World/WallLeft", Gf.Vec3f(0.02, 0.40, 0.90), Gf.Vec3f(-0.70, 0.20, 0.0)),
    physicsUtils.add_collider_cube(stage, "/World/WallRight", Gf.Vec3f(0.02, 0.40, 0.90), Gf.Vec3f(0.70, 0.20, 0.0)),
    physicsUtils.add_collider_cube(stage, "/World/WallBack", Gf.Vec3f(1.40, 0.40, 0.02), Gf.Vec3f(0.0, 0.20, -0.45)),
    physicsUtils.add_collider_cube(stage, "/World/WallFront", Gf.Vec3f(1.40, 0.40, 0.02), Gf.Vec3f(0.0, 0.20, 0.45)),
]
for wall in walls:
    hide(wall)
collider_particle_contact_offset = (0.5 * args.spacing) / (0.99 * 0.6)
for collider in [floor, obstacle.GetPrim(), *walls]:
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider)
    collision_api.CreateContactOffsetAttr().Set(2.0 * collider_particle_contact_offset)
    collision_api.CreateRestOffsetAttr().Set(collider_particle_contact_offset)

# Resolution is defined by particle spacing = 2 * fluid rest offset.
fluid_rest_offset = 0.5 * args.spacing
particle_contact_offset = fluid_rest_offset / (0.99 * 0.6)
particle_system_path = Sdf.Path("/World/ParticleSystem")
particle_system = particleUtils.add_physx_particle_system(
    stage,
    particle_system_path,
    simulation_owner=scene.GetPath(),
    particle_contact_offset=particle_contact_offset,
    solver_position_iterations=4,
    max_neighborhood=96,
    enable_ccd=True,
    max_velocity=2.5,
)

# NVIDIA's official meter-scale Water preset.
pbd_material_path = Sdf.Path("/World/Looks/WaterPhysics")
particleUtils.add_pbd_particle_material(
    stage,
    pbd_material_path,
    density=1000.0,
    friction=0.1,
    damping=0.0,
    viscosity=0.0000017,
    vorticity_confinement=0.0,
    surface_tension=0.0074,
    cohesion=0.01,
)
physicsUtils.add_physics_material_to_prim(stage, particle_system.GetPrim(), pbd_material_path)

# Start empty and emit one tightly spaced cross-section per render frame. This
# follows NVIDIA's FluidBallEmitterDemo instead of launching a pre-filled block.
positions = Vt.Vec3fArray([])
velocities = Vt.Vec3fArray([])
particles_prim = particleUtils.add_physx_particleset_pointinstancer(
    stage,
    Sdf.Path("/World/WaterParticles"),
    Vt.Vec3fArray(positions),
    Vt.Vec3fArray(velocities),
    particle_system_path,
    self_collision=True,
    fluid=True,
    particle_group=0,
    particle_mass=0.0,
    density=1000.0,
    num_prototypes=0,
)
particle_prototype_path = Sdf.Path("/World/WaterParticles/particlePrototype0")
particle_prototype = UsdGeom.Sphere.Define(stage, particle_prototype_path)
particle_prototype.CreateRadiusAttr().Set(fluid_rest_offset)
UsdGeom.PointInstancer(particles_prim).GetPrototypesRel().AddTarget(particle_prototype_path)
particles_prim.CreateAttribute("physxParticle:maxParticles", Sdf.ValueTypeNames.Int).Set(
    max(50_000, args.frames * 128)
)

# Render-only post processing. Moderate smoothing avoids a rubbery surface.
particleUtils.add_physx_particle_smoothing(
    stage, particle_system_path, enabled=True, strength=0.18
)
particleUtils.add_physx_particle_anisotropy(
    stage, particle_system_path, enabled=True, scale=1.25, min=0.8, max=1.8
)
particleUtils.add_physx_particle_isosurface(
    stage,
    particle_system_path,
    enabled=True,
    grid_spacing=0.0012,
    surface_distance=0.0025,
    grid_filtering_passes="SRSRS",
    grid_smoothing_radius=1.0,
    num_mesh_smoothing_passes=2,
    num_mesh_normal_smoothing_passes=4,
    max_vertices=2_000_000,
    max_triangles=2_000_000,
    max_subgrids=4096,
)

# Opaque diagnostic material keeps physics validation separate from refraction.
created = []
omni.kit.commands.execute(
    "CreateAndBindMdlMaterialFromLibrary",
    mdl_name="OmniPBR.mdl",
    mtl_name="OmniPBR",
    mtl_created_list=created,
    bind_selected_prims=False,
    select_new_prim=False,
)
water_render_path = Sdf.Path(created[0])
water_shader = UsdShade.Shader.Get(stage, water_render_path.AppendChild("Shader"))
water_shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(0.035, 0.22, 0.48)
)
water_shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.08)
omni.kit.commands.execute(
    "BindMaterialCommand",
    prim_path=particle_system_path,
    material_path=water_render_path,
    strength=None,
)

# Neutral lighting makes refraction readable without a patterned floor.
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr().Set(700.0)
dome.CreateColorAttr().Set(Gf.Vec3f(0.82, 0.90, 1.0))
key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
key.CreateIntensityAttr().Set(2500.0)
key.CreateAngleAttr().Set(3.0)
UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 35.0, 20.0))

camera = rep.create.camera(
    position=(0.75, 0.38, 0.68),
    look_at=(-0.02, 0.11, 0.0),
    look_at_up_axis=(0.0, 1.0, 0.0),
    focal_length=42.0,
    clipping_range=(0.01, 100.0),
)

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", "RaytracedLighting")
settings.set("/rtx/translucency/maxRefractionBounces", 12)
settings.set("/rtx/reflections/enabled", True)
settings.set("/rtx/indirectDiffuse/enabled", True)

stage.GetRootLayer().Export(os.path.join(args.output, "water_clean_scene.usda"))

# Let the deferred isosurface authoring complete before attaching PhysX.
simulation_app.update()
simulation_app.update()

# Use an explicit render product instead of the editor viewport. This prevents
# headless captures from silently falling back to the perspective/grid camera.
render_product = rep.create.render_product(camera, (args.width, args.height))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=args.output, rgb=True)
writer.attach([render_product])

simulation = get_physx_simulation_interface()
stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
simulation.attach_stage(stage_id)

def print_particle_diagnostics(label):
    values = UsdGeom.PointInstancer(particles_prim).GetPositionsAttr().Get()
    if values:
        mins = [min(p[i] for p in values) for i in range(3)]
        maxs = [max(p[i] for p in values) for i in range(3)]
        print(f"[diag] {label} bounds min={mins}, max={maxs}")
    print(
        f"[diag] offsets contact={particle_system.GetContactOffsetAttr().Get()}, "
        f"rest={particle_system.GetRestOffsetAttr().Get()}, "
        f"particleContact={particle_system.GetParticleContactOffsetAttr().Get()}, "
        f"solidRest={particle_system.GetSolidRestOffsetAttr().Get()}, "
        f"fluidRest={particle_system.GetFluidRestOffsetAttr().Get()}"
    )
    mass_api = UsdPhysics.MassAPI(particles_prim)
    print(f"[diag] mass={mass_api.GetMassAttr().Get()}, density={mass_api.GetDensityAttr().Get()}")


def extend_array_attribute(attribute, elements):
    current = attribute.Get()
    values = list(current) if current is not None else []
    values.extend(elements)
    attribute.Set(values)


def emit_stream_slice():
    # A 21 x 39 mm vertical ribbon, one 3 mm layer per 60 Hz frame.
    emitted_positions = []
    for ix in range(7):
        for iz in range(14):
            emitted_positions.append(
                Gf.Vec3f(
                    0.021 + ix * args.spacing,
                    0.19,
                    (iz - 6.5) * args.spacing,
                )
            )
    emitted_velocities = [Gf.Vec3f(0.0, -0.20, 0.0)] * len(emitted_positions)

    particle_set = PhysxSchema.PhysxParticleSetAPI(particles_prim)
    simulation_points = particle_set.GetSimulationPointsAttr()
    if not simulation_points.HasAuthoredValue():
        simulation_points.Set(Vt.Vec3fArray([]))
    extend_array_attribute(simulation_points, emitted_positions)

    instancer = UsdGeom.PointInstancer(particles_prim)
    extend_array_attribute(instancer.GetPositionsAttr(), emitted_positions)
    extend_array_attribute(instancer.GetVelocitiesAttr(), emitted_velocities)
    extend_array_attribute(instancer.GetProtoIndicesAttr(), [0] * len(emitted_positions))
    extend_array_attribute(
        instancer.GetOrientationsAttr(),
        [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * len(emitted_positions),
    )
    extend_array_attribute(
        instancer.GetScalesAttr(), [Gf.Vec3f(1.0)] * len(emitted_positions)
    )


print_particle_diagnostics("before")
print(
    f"[water] emitter=98 particles/frame, spacing={args.spacing:.4f}m, "
        f"iterations=4, frames={args.frames}"
)
if args.diagnostic_only:
    for step in range(3):
        simulation.simulate(1.0 / 60.0, step / 60.0)
        simulation.fetch_results()
        simulation_app.update()
        print_particle_diagnostics(f"after_step_{step + 1}")
    simulation.detach_stage()
    simulation_app.close()
    raise SystemExit(0)

for frame in range(args.frames):
    emit_stream_slice()
    simulation_app.update()
    for substep in range(args.substeps):
        step_index = frame * args.substeps + substep
        dt = 1.0 / (60.0 * args.substeps)
        simulation.simulate(dt, step_index * dt)
        simulation.fetch_results()
    simulation_app.update()
    if frame % 10 == 0:
        print_particle_diagnostics(f"frame_{frame}")

    if frame % args.capture_every == 0 or frame == args.frames - 1:
        rep.orchestrator.step(rt_subframes=4 if frame == 0 else 1, delta_time=0.0)
    if frame % 30 == 0:
        print(f"[water] frame {frame}/{args.frames}")

simulation.detach_stage()
writer.detach()
simulation_app.close()
