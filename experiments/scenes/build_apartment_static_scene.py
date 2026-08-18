"""Build and validate a static-collision wrapper for the Blender apartment USD."""

from pathlib import Path

from isaacsim import SimulationApp


SOURCE_USD = Path(r"Y:\scenes\apartment\apartment.usdc")
OUTPUT_USD = Path(r"Y:\scenes\apartment\apartment_sim.usda")
ROOT_PATH = "/World"
APARTMENT_PATH = "/World/Apartment"
PROBE_PATH = "/World/_CollisionProbe"


simulation_app = SimulationApp({"headless": True, "renderer": "MinimalRendering"})

try:
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    if not SOURCE_USD.is_file():
        raise FileNotFoundError(f"Apartment source USD does not exist: {SOURCE_USD}")

    stage = Usd.Stage.CreateNew(str(OUTPUT_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, ROOT_PATH)
    stage.SetDefaultPrim(world.GetPrim())

    apartment = UsdGeom.Xform.Define(stage, APARTMENT_PATH)
    apartment.GetPrim().GetReferences().AddReference("./apartment.usdc")

    physics_scene = UsdPhysics.Scene.Define(stage, f"{ROOT_PATH}/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateTimeStepsPerSecondAttr().Set(60)

    mesh_paths = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.GetPath().HasPrefix(Sdf.Path(APARTMENT_PATH)):
            continue
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
        mesh_paths.append(str(prim.GetPath()))

    if not mesh_paths:
        raise RuntimeError("No apartment meshes were found through the USD reference")

    stage.GetRootLayer().Save()
    print(f"[build] output={OUTPUT_USD}")
    print(f"[build] static_triangle_mesh_colliders={len(mesh_paths)}")

    # Keep the validation body out of the saved root layer.
    stage.SetEditTarget(stage.GetSessionLayer())
    probe = UsdGeom.Cube.Define(stage, PROBE_PATH)
    probe.CreateSizeAttr().Set(0.5)
    probe.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
    UsdPhysics.MassAPI.Apply(probe.GetPrim()).CreateMassAttr().Set(1.0)

    simulation = get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    simulation.attach_stage(stage_id)
    initial_y = float(
        UsdGeom.Xformable(probe).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()[1]
    )
    minimum_y = initial_y
    final_y = initial_y
    for frame in range(240):
        simulation.simulate(1.0 / 60.0, frame / 60.0)
        simulation.fetch_results()
        simulation_app.update()
        final_y = float(
            UsdGeom.Xformable(probe).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            ).ExtractTranslation()[1]
        )
        minimum_y = min(minimum_y, final_y)
    simulation.detach_stage()

    # The broad apartment floor around the origin is near y=-0.49 m, so a
    # 0.5 m cube should settle with its center near y=-0.24 m.
    passed = -0.8 < final_y < 0.5 and minimum_y > -1.0
    print(
        f"[probe] initial_y={initial_y:.6f} minimum_y={minimum_y:.6f} "
        f"final_y={final_y:.6f} passed={passed}"
    )
    if not passed:
        raise RuntimeError(
            "The collision probe did not settle on the expected floor near the origin"
        )
finally:
    simulation_app.close()
