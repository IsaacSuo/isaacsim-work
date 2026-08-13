"""Headless probe for PhysX's bundled FluidBallEmitterDemo update ordering."""

import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": 320,
        "height": 180,
    }
)

import omni.physx
import omni.kit.app
import omni.timeline
import omni.usd

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "omni.physx.demos", True
)

from omni.physxdemos.scenes.FluidBallEmitterDemo import FluidBallEmitterDemo
from omni.physx.scripts import particleUtils
from pxr import PhysxSchema, UsdGeom


context = omni.usd.get_context()
context.new_stage()
for _ in range(3):
    simulation_app.update()
stage = context.get_stage()
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

demo = FluidBallEmitterDemo()
demo.create(stage, Single_Particle_Set=True, Use_Instancer=True)
particleUtils.add_physx_particle_isosurface(
    stage,
    demo._particleSystemPath,
    enabled=True,
    grid_spacing=0.09,
    surface_distance=0.135,
    max_vertices=2_500_000,
    max_triangles=2_500_000,
    max_subgrids=8192,
)

timeline = omni.timeline.get_timeline_interface()
timeline.set_time_codes_per_second(60.0)
timeline.set_target_framerate(60.0)
timeline.play()
timeline.commit()

for frame in range(180):
    # The demo browser invokes scene.update on the main thread before the next
    # Kit update.  Runtime USD topology must be authored at this point, not
    # from a generic GLOBAL_EVENT_UPDATE observer.
    demo.update(stage, 1.0 / 60.0, None, omni.physx.get_physx_interface())
    simulation_app.update()
    if frame in (30, 60, 90, 120, 150, 179):
        prim = stage.GetPrimAtPath("/particles")
        if prim:
            sim_points = PhysxSchema.PhysxParticleSetAPI(
                prim
            ).GetSimulationPointsAttr().Get()
            positions = UsdGeom.PointInstancer(prim).GetPositionsAttr().Get()
            print(
                f"[probe] frame={frame}, sim_points={len(sim_points or [])}, "
                f"positions={len(positions or [])}, "
                f"first_sim={sim_points[0] if sim_points else None}, "
                f"first_pos={positions[0] if positions else None}"
            )

timeline.stop()
timeline.commit()
simulation_app.update()
demo.on_shutdown()
simulation_app.close()
