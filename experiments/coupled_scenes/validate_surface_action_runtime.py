"""Native kinematic movement smoke test, no particle simulation or rendered output."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from coupled_scene.surface_actions import plan_action,action_pose


def main():
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':64,'height':64})
    import carb
    import omni.usd
    from omni.physx import get_physx_interface,get_physx_simulation_interface
    import omni.physx.bindings._physx as bindings
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,Gf
    simulation=get_physx_simulation_interface()
    attached=False
    try:
        carb.settings.get_settings().set(bindings.SETTING_UPDATE_TO_USD,True)
        for name in ['02_sloshing','04_wave_reflection','05_surface_recovery']:
            context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
            directory=ROOT/'output/coupled_scenes/surface_study_group01_v1'/name
            spec=json.loads((directory/'scene_spec.json').read_text(encoding='utf-8'))
            plan=plan_action(spec,.079)
            stage.GetRootLayer().subLayerPaths.append(str(directory/'colliders.usda'))
            UsdGeom.SetStageUpAxis(stage,'Y');UsdGeom.SetStageMetersPerUnit(stage,1.)
            scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,-1,0));scene.CreateGravityMagnitudeAttr().Set(9.81)
            PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateEnableGPUDynamicsAttr().Set(True)
            target=stage.GetPrimAtPath(plan['target_path'])
            op=UsdGeom.Xformable(target).MakeMatrixXform()
            pose,_=action_pose(plan,0.,False);op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pose)))
            app.update()
            cache=UsdUtils.StageCache.Get();sid=cache.GetId(stage)
            if not sid.IsValid():sid=cache.Insert(stage)
            simulation.attach_stage(sid.ToLongInt());attached=True
            results=[]
            for i,(t,recording) in enumerate([(0.,False),(.5,False),(1.,False),(2.1,True),(2.3,True),(2.5,True),(4.,True),(7.,True)]):
                pose,velocity=action_pose(plan,t,recording)
                op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pose)))
                simulation.simulate(1/720,i/720);simulation.fetch_results();app.update()
                actual=get_physx_interface().get_rigidbody_transformation(plan['target_path'])
                assert actual['ret_val'],actual
                error=max(abs(a-b) for a,b in zip(actual['position'],pose))
                assert error<1e-4,(name,t,error,actual,pose)
                results.append(dict(seconds=t,prewarm=not recording,error_m=error))
            simulation.detach_stage();attached=False
            print('[native-action-pass] '+json.dumps(dict(case=name,plan=plan,samples=results)),flush=True)
    finally:
        if attached:simulation.detach_stage()
        app.close()


if __name__=='__main__':main()
