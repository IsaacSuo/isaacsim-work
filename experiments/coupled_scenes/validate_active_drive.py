"""Read-only active-drive USD and motion validation; usd-core, no GPU/Isaac app."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from coupled_scene.active_drive import body_pose, rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    args=parser.parse_args()
    paths=sorted(args.input.glob('*/scene_spec.json'))
    assert len(paths)==3
    report=[]
    for path in paths:
        spec=json.loads(path.read_text());case=spec['case'];bodies=spec['geometry']['bodies']
        stage=Usd.Stage.Open(str(path.parent/'colliders.usda'))
        assert stage and UsdGeom.GetStageUpAxis(stage)=='Y' and UsdGeom.GetStageMetersPerUnit(stage)==1
        assert stage.GetEndTimeCode()==spec['frame_count']
        colliders=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(colliders)==sum(len(b['collision']) for b in bodies)
        assert not any('REFERENCE' in str(p.GetPath()) for p in stage.Traverse())
        cameras=json.loads((path.parent/'cameras.json').read_text())
        assert len(cameras)==8 and len({c['name'] for c in cameras})==8
        for camera in cameras:
            assert camera['resolution']==[1280,960]
            assert camera['K'][0][0]>0 and camera['K'][1][1]>0
        samples=json.loads((path.parent/'motion_samples.json').read_text())
        assert len(samples)==spec['frame_count']
        moving=next(b for b in bodies if b['moving'])
        for body in bodies:
            prim=stage.GetPrimAtPath('/ActiveDrive/'+body['name'])
            if body['moving']:
                assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
                assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
            else: assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        error=0.
        for sample in samples:
            frame=sample['frame'];t=sample['seconds']
            assert math.isclose(t,(frame-1)/spec['fps'],abs_tol=1e-12)
            pose=body_pose(moving,case['motion'],t,spec['origin_isaac_m'])
            for key in ('angle_rad','rotation_axis'): assert pose[key]==sample[key]
            for key in ('position_m','linear_velocity_m_s','angular_velocity_rad_s'):
                assert all(abs(a-b)<1e-10 for a,b in zip(pose[key],sample[key]))
            cache=UsdGeom.XformCache(Usd.TimeCode(frame))
            for body in bodies:
                pose=body_pose(body,case['motion'],t,spec['origin_isaac_m'])
                matrix=cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/ActiveDrive/'+body['name']))
                r=rotation(pose['rotation_axis'],pose['angle_rad'])
                for local in ([0,0,0],[.17,.23,.31]):
                    expected=[pose['position_m'][i]+sum(r[i][j]*local[j] for j in range(3)) for i in range(3)]
                    actual=matrix.Transform(Gf.Vec3d(*local))
                    error=max(error,max(abs(a-b) for a,b in zip(actual,expected)))
        assert error<1e-7,error
        report.append(dict(case=case['id'],colliders=len(colliders),samples=len(samples),max_transform_error_m=error))
    print(json.dumps(dict(valid=True,scope='Layout transforms, colliders and cameras only; no fluid or contact solver run',cases=report),indent=2))


if __name__=='__main__': main()
