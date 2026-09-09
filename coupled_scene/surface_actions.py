"""Native-substep action planning shared by simulation and Blender playback."""
from copy import deepcopy
from coupled_scene.surface_study import actuator_spec, motion_state


def plan_action(spec, measured_level):
    case=deepcopy(spec['case'])
    actuator=actuator_spec(case['actuator'],spec['tank'])
    origin=spec['tank_origin_isaac']
    base=list(origin) if actuator is None else [a+b for a,b in zip(origin,actuator['centre'])]
    if case['actuator']=='Plunger':
        initial_bottom=actuator['centre'][1]-actuator['height']/2
        case['motion']['amplitude_m']=measured_level-.02-initial_bottom
    insertion=0.
    if case['actuator']=='Paddle':
        insertion=max(0.,measured_level+actuator['size'][1]/2+.02-actuator['centre'][1])
    return dict(case=case,base_isaac=base,measured_level_m=measured_level,
                insertion_raise_m=insertion,insertion_seconds=1.,
                target_path='/SurfaceStudy/'+case['motion']['target'])


def action_pose(plan, seconds, recording):
    base=plan['base_isaac']
    if recording:
        displacement,velocity=motion_state(plan['case']['motion'],seconds)
    else:
        u=min(1.,max(0.,seconds/plan['insertion_seconds']))
        blend=u*u*(3-2*u)
        displacement=[0.,plan['insertion_raise_m']*(1-blend),0.]
        velocity=[0.,-plan['insertion_raise_m']*6*u*(1-u)/plan['insertion_seconds'],0.]
    return [a+b for a,b in zip(base,displacement)],velocity
