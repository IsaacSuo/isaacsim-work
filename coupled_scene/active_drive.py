"""Solver-independent active apparatus geometry and analytic rigid motion.

All lengths are metres, Y-up; angles and angular velocities are radians.
No particles, forces on particles, or fluid animation are synthesized here.
"""
import math
from coupled_scene.surface_study import collision_panels


def smooth(u):
    u = max(0., min(1., u))
    return u**3 * (10 - 15*u + 6*u*u), 30*u*u*(1-u)**2


def motion_state(motion, seconds):
    p, v, angle, omega = [0.]*3, [0.]*3, 0., 0.
    start, stop = motion['start_s'], motion['stop_s']
    if stop <= start:
        raise ValueError('Motion duration must be positive')
    kind = motion['kind']
    if kind == 'push_hold':
        f, df = smooth((seconds-start)/(stop-start))
        p[motion['axis']] = motion['stroke_m']*f
        v[motion['axis']] = motion['stroke_m']*df/(stop-start)
    elif kind == 'tilt_hold_return':
        peak, hold = motion['tilt_end_s'], motion['hold_end_s']
        if not start < peak <= hold < stop:
            raise ValueError('Invalid tilt phases')
        if seconds <= peak:
            f, df = smooth((seconds-start)/(peak-start)); rate = df/(peak-start)
        elif seconds <= hold:
            f, rate = 1., 0.
        else:
            blend, df = smooth((seconds-hold)/(stop-hold)); f, rate = 1-blend, -df/(stop-hold)
        angle = math.radians(motion['angle_deg'])*f
        omega = math.radians(motion['angle_deg'])*rate
    elif kind == 'spin_ramp':
        ramp = motion['ramp_s']; duration = stop-start
        if not 0 < 2*ramp <= duration:
            raise ValueError('Spin ramps must fit duration')
        t = max(0., min(duration, seconds-start))
        # Integral of smoothstep 3u²-2u³; no finite-difference velocities.
        def integrated(u): return u**3 - .5*u**4
        if t < ramp:
            u=t/ramp; swept=ramp*integrated(u); speed=u*u*(3-2*u)
        elif t <= duration-ramp:
            swept=t-ramp/2; speed=1.
        else:
            u=(t-duration+ramp)/ramp
            swept=duration-1.5*ramp+ramp*(u-integrated(u)); speed=1-u*u*(3-2*u)
        angle=motion['speed_rad_s']*swept
        omega=motion['speed_rad_s']*speed if start < seconds < stop else 0.
    else:
        raise ValueError(kind)
    angular=[0.]*3; angular[motion['axis']]=omega
    return dict(displacement_m=p, linear_velocity_m_s=v,
                angle_rad=angle, angular_velocity_rad_s=angular, rotation_axis=motion['axis'])


def cube(name, centre, size, material):
    return dict(name=name, shape='Cube', centre=centre, size=size, material=material)


def tank_body(name, base, size, offset=None):
    offset=offset or [0.,0.,0.]
    proxy=collision_panels(dict(inner_size_m=size, proxy_thickness_m=.03))
    visible=collision_panels(dict(inner_size_m=size, proxy_thickness_m=.012))
    def shapes(panels):
        return [cube(n,[a+b for a,b in zip(p,offset)],s,'floor' if n=='Floor' else 'glass') for n,p,s in panels]
    return dict(name=name, base_m=base, moving=False, collision=shapes(proxy), visual=shapes(visible))


def layout(case):
    """Positions relative to the apparatus floor origin, except local shapes."""
    key=case['id']
    if key=='01_container_transfer':
        x,h,z=case['donor_size_m']; pivot=case['donor_lip_pivot_m']
        receiver=tank_body('Receiver',[0,0,0],case['receiver_size_m'])
        donor=tank_body('Donor',pivot,case['donor_size_m'],[-x/2,-h,0])
        donor['moving']=True
        fill=dict(body='Donor',centre_m=[-x/2,-h+case['initial_depth_m']/2,0],
                  size_m=[x-.016,case['initial_depth_m'],z-.016])
        bodies=[receiver,donor]; target=[-.10,.48,0]
    elif key=='02_stirring':
        tank=tank_body('Tank',[0,0,0],case['tank_size_m'])
        span,height,thickness=case['blade_span_m'],case['blade_height_m'],case['blade_thickness_m']
        shape=[cube('BladeX',[0,0,0],[span,height,thickness],'copper'),
               cube('BladeZ',[0,0,0],[thickness,height,span],'copper'),
               dict(name='Shaft',shape='Cylinder',centre=[0,.235,0],radius=.012,height=.57,material='steel')]
        bodies=[tank,dict(name='Impeller',base_m=[0,case['blade_centre_height_m'],0],moving=True,collision=shape,visual=shape)]
        x,h,z=case['tank_size_m']; depth=case['initial_depth_m']
        fill=dict(body='Tank',centre_m=[0,depth/2,0],size_m=[x-.016,depth,z-.016],exclude_bodies=['Impeller'])
        target=[0,.23,0]
    elif key=='03_piston_push':
        x,h,z=case['tank_size_m']; gap=case['piston_gap_m']; thickness=case['piston_thickness_m']
        plate=cube('Plate',[0,h/2+gap/2,0],[thickness,h-gap,z-2*gap],'copper')
        rod=dict(name='Rod',shape='Cube',centre=[-.465,h+.04,0],size=[.95,.026,.026],material='steel')
        clevis=cube('RodClevis',[0,h+.02,0],[.045,.07,.07],'copper')
        bodies=[tank_body('Tank',[0,0,0],case['tank_size_m']),
                dict(name='Piston',base_m=[case['piston_start_x_m'],0,0],moving=True,collision=[plate],visual=[plate,rod,clevis])]
        low=case['piston_start_x_m']+thickness/2+.008; high=x/2-.008; depth=case['initial_depth_m']
        fill=dict(body='Tank',centre_m=[(low+high)/2,depth/2,0],size_m=[high-low,depth,z-.016],dry_region='Behind Piston')
        target=[0,.16,0]
    else:
        raise ValueError(key)
    fill.update(purpose='Initial fill design only, exclude all solid occupied space and settle before action',
                exact_particle_count_or_volume=False)
    return dict(bodies=bodies,fill=fill,camera_target_m=target)


def rotation(axis, angle):
    """Column-vector rotation matrix in Isaac world axes."""
    c,s=math.cos(angle),math.sin(angle)
    if axis==0: return [[1,0,0],[0,c,-s],[0,s,c]]
    if axis==1: return [[c,0,s],[0,1,0],[-s,0,c]]
    return [[c,-s,0],[s,c,0],[0,0,1]]


def body_pose(body, motion, seconds, origin=(0,0,0)):
    state=motion_state(motion,seconds) if body['moving'] else dict(
        displacement_m=[0.]*3,linear_velocity_m_s=[0.]*3,angle_rad=0.,angular_velocity_rad_s=[0.]*3,rotation_axis=0)
    return dict(position_m=[a+b+c for a,b,c in zip(origin,body['base_m'],state['displacement_m'])], **state)
