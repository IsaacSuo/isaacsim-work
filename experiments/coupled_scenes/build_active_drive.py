"""Blender: three editable active-drive layouts, collision USD, cameras, previews.

Only apparatus is animated. Reference fill is visible at t=0 only; it is not
fluid and is never exported into collision assets. No simulator is launched.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from coupled_scene.active_drive import layout, body_pose, motion_state
from coupled_scene.water_defaults import DEFAULT_VORTICITY_CONFINEMENT
from experiments.coupled_scenes.build_surface_study import (
    box, empty, material, move_to, convert, camera_report, text_label,
)


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_usd(path, bodies, case, origin, fps):
    end=round(case['duration_s']*fps)+1
    lines=['#usda 1.0','(', '    defaultPrim = "ActiveDrive"', '    upAxis = "Y"',
           '    metersPerUnit = 1',f'    timeCodesPerSecond = {fps}',
           '    startTimeCode = 1',f'    endTimeCode = {end}',')','def Xform "ActiveDrive" {']
    def vec(values): return '('+', '.join(f'{v:.12g}' for v in values)+')'
    for body in bodies:
        moving=body['moving']; name=body['name']; axis='XYZ'[case['motion']['axis']]
        api=' (prepend apiSchemas = ["PhysicsRigidBodyAPI"])' if moving else ''
        lines.append(f'    def Xform "{name}"{api} {{')
        if moving:
            lines += ['        bool physics:rigidBodyEnabled = true','        bool physics:kinematicEnabled = true',
                      '        double3 xformOp:translate.timeSamples = {']
            for frame in range(1,end+1):
                pose=body_pose(body,case['motion'],(frame-1)/fps,origin)
                lines.append(f'            {frame}: {vec(pose["position_m"])},')
            lines += ['        }',f'        double xformOp:rotate{axis}.timeSamples = {{']
            for frame in range(1,end+1):
                angle=motion_state(case['motion'],(frame-1)/fps)['angle_rad']
                lines.append(f'            {frame}: {math.degrees(angle):.12g},')
            lines += ['        }',f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotate{axis}"]']
        else:
            lines += [f'        double3 xformOp:translate = {vec(body_pose(body,case["motion"],0,origin)["position_m"])}',
                      '        uniform token[] xformOpOrder = ["xformOp:translate"]']
        for shape in body['collision']:
            lines += [f'        def {shape["shape"]} "{shape["name"]}" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{',
                      '            bool physics:collisionEnabled = true',
                      f'            double3 xformOp:translate = {vec(shape["centre"])}']
            if shape['shape']=='Cube':
                lines += ['            double size = 1',f'            double3 xformOp:scale = {vec(shape["size"])}',
                          '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]']
            else:
                lines += [f'            double radius = {shape["radius"]}',f'            double height = {shape["height"]}',
                          '            uniform token axis = "Y"','            uniform token[] xformOpOrder = ["xformOp:translate"]']
            lines.append('        }')
        lines.append('    }')
    lines.append('}')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def create_shape(shape, parent, collection, mats, prefix='', proxy=False):
    if shape['shape']=='Cube':
        obj=box(prefix+shape['name'],shape['centre'],shape['size'],None if proxy else mats[shape['material']],
                collection,parent,0 if proxy else .0015)
    else:
        bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=shape['radius'],depth=shape['height'])
        obj=move_to(bpy.context.object,collection);obj.name=prefix+shape['name'];obj.parent=parent
        obj.location=convert(shape['centre'])
        if not proxy: obj.data.materials.append(mats[shape['material']])
    if proxy:
        obj.hide_render=True;obj.display_type='WIRE';obj.hide_set(True)
    return obj


def build(cfg,case,args):
    bpy.ops.wm.open_mainfile(filepath=str(args.environment))
    scene=bpy.context.scene
    collection=bpy.data.collections.new('ActiveDrive');scene.collection.children.link(collection)
    proxies=bpy.data.collections.new('COLLISION_PROXIES_inspection_only');scene.collection.children.link(proxies)
    origin=list(cfg['placement_isaac']);origin[1]+=cfg['floor_above_support_m']
    recipe=layout(case);fps=cfg['fps'];end=round(case['duration_s']*fps)+1
    mats=dict(steel=material('Drive slate',(0.06,.085,.095),.28,metallic=.45),
              glass=material('Observation glass',(.97,.99,1),.055,1),
              floor=material('Basin ceramic',(.5,.6,.61),.38),
              copper=material('Driven copper',(.65,.20,.06),.28,metallic=.25),
              water=material('REFERENCE_WATER',(.58,.84,.88),.08,.9),
              text=material('Markings',(.90,.94,.94),.45))
    def world(local): return [a+b for a,b in zip(origin,local)]
    box('Workbench',world([0,-.10,0]),[2.25,.10,1.24],mats['steel'],collection,bevel=.015)
    for xx in [-.95,.95]:
        for zz in [-.49,.49]:
            box('Stand leg',world([xx,-.435,zz]),[.065,.57,.065],mats['steel'],collection,bevel=.006)
    roots={};visuals=[];samples=[]
    for body in recipe['bodies']:
        parent=empty(body['name'],world(body['base_m']),collection);roots[body['name']]=parent
        parent['native_usd_path']='/ActiveDrive/'+body['name']
        for shape in body['visual']: visuals.append(create_shape(shape,parent,collection,mats,body['name']+'_'))
        for shape in body['collision']: create_shape(shape,parent,proxies,mats,'Proxy_'+body['name']+'_',True)
        if body['moving']:
            parent.rotation_mode='QUATERNION'
            axis=[0,0,0];axis[case['motion']['axis']]=1;axis=convert(axis)
            for frame in range(1,end+1):
                t=(frame-1)/fps;pose=body_pose(body,case['motion'],t,origin)
                parent.location=convert(pose['position_m']);parent.rotation_quaternion=Quaternion(axis,pose['angle_rad'])
                parent.keyframe_insert(data_path='location',frame=frame)
                parent.keyframe_insert(data_path='rotation_quaternion',frame=frame)
                samples.append(dict(frame=frame,seconds=t,target=body['name'],**pose))
            # Keyed samples are design playback; native physics evaluates the function at each substep.
            for layer in parent.animation_data.action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for curve in bag.fcurves:
                            for point in curve.keyframe_points: point.interpolation='LINEAR'
    fill=recipe['fill']
    water=box('REFERENCE_FILL_NOT_SIMULATED',fill['centre_m'],fill['size_m'],mats['water'],collection,roots[fill['body']])
    water['purpose']='Initial fill design only; not simulated water. Exclude solid occupied volume at initialization.'
    # Do not animate a rigid water block through the pour or piston movement.
    water.hide_render=False;water.hide_viewport=False
    water.keyframe_insert(data_path='hide_render',frame=1);water.keyframe_insert(data_path='hide_viewport',frame=1)
    water.hide_render=True;water.hide_viewport=True
    water.keyframe_insert(data_path='hide_render',frame=2);water.keyframe_insert(data_path='hide_viewport',frame=2)
    stationary=empty('DecorativeStand',origin,collection)
    text_label('Design label','APPARATUS DESIGN / NO FLUID SIMULATION',[-1.04,-.115,.625],.026,mats['text'],collection,stationary)
    if case['id']=='01_container_transfer':
        pivot=case['donor_lip_pivot_m']
        for z in [-.50,.50]:
            box('Tilt bearing post',world([pivot[0],pivot[1]/2,z]),[.045,pivot[1],.045],mats['steel'],collection,bevel=.004)
            box('Tilt axle',world([pivot[0],pivot[1],z*.70]),[.025,.025,.30],mats['copper'],collection,bevel=.004)
        box('Tilt drive',world([pivot[0],pivot[1],.57]),[.12,.12,.12],mats['copper'],collection,bevel=.012)
    elif case['id']=='02_stirring':
        for z in [-.50,.50]: box('Mixer post',world([0,.345,z]),[.04,.69,.04],mats['steel'],collection,bevel=.004)
        box('Mixer bridge',world([0,.69,0]),[.065,.045,1.06],mats['steel'],collection,bevel=.005)
        box('Mixer drive',world([0,.66,0]),[.13,.12,.13],mats['copper'],collection,bevel=.012)
    else:
        box('Piston actuator',world([-.98,.38,0]),[.25,.12,.13],mats['steel'],collection,bevel=.008)
        for z in [-.14,.14]: box('Actuator support',world([-.98,.16,z]),[.045,.42,.045],mats['steel'],collection,bevel=.003)
    target=convert(world(recipe['camera_target_m']))
    scene.world=bpy.data.worlds.new('Active drive environment');scene.world.use_nodes=True
    nodes=scene.world.node_tree.nodes;nodes.clear()
    env=nodes.new('ShaderNodeTexEnvironment');env.image=bpy.data.images.load(str(args.environment.parent/'HDRI/bryanston_park_sunrise_8k.exr'),check_existing=True)
    bg=nodes.new('ShaderNodeBackground');bg.inputs['Strength'].default_value=.65
    output=nodes.new('ShaderNodeOutputWorld');scene.world.node_tree.links.new(env.outputs['Color'],bg.inputs['Color']);scene.world.node_tree.links.new(bg.outputs['Background'],output.inputs['Surface'])
    for name,local,power in [('Key',[-.8,2.3,-.8],700),('Rim',[.8,1.8,.8],500)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=1.8
        obj=bpy.data.objects.new(name,data);collection.objects.link(obj);obj.location=convert(world(local));obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
    scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for device in prefs.devices: device.use=device.type!='CPU'
    scene.cycles.device='GPU';scene.view_settings.view_transform='AgX';scene.view_settings.exposure=0
    scene.render.resolution_x,scene.render.resolution_y=cfg['camera']['resolution'];scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.fps=fps;scene.frame_start=1;scene.frame_end=end
    scene.render.use_motion_blur=False
    cameras=[]
    for i in range(cfg['camera']['count']):
        a=math.radians(-55+i*360/cfg['camera']['count']);data=bpy.data.cameras.new(f'View_{i:02d}')
        data.lens=cfg['camera']['lens_mm'];data.sensor_width=36;data.sensor_fit='HORIZONTAL'
        cam=bpy.data.objects.new(data.name,data);collection.objects.link(cam)
        cam.location=target+Vector((cfg['camera']['radius_m']*math.cos(a),cfg['camera']['radius_m']*math.sin(a),cfg['camera']['height_m']))
        cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cameras.append(cam)
    scene.camera=cameras[0];scene.frame_set(1);bpy.context.view_layer.update()
    folder=args.output/case['id'];folder.mkdir(parents=True,exist_ok=False)
    reports=[camera_report(c,scene) for c in cameras]
    save_json(folder/'cameras.json',reports);save_json(folder/'motion_samples.json',samples)
    spec=dict(schema=1,product='active_drive_layout',status='layout_only_runtime_adapter_required',
              environment=cfg['environment'],origin_isaac_m=origin,case=case,geometry=recipe,fps=fps,frame_count=end,
              fluid=dict(simulated=False,initialization='prefill_excluding_solids_then_settle',reference_fill_only=True,
                         defaults=dict(spacing_m=.004,hz=720,iterations=64,vorticity_confinement=DEFAULT_VORTICITY_CONFINEMENT),
                         note='Defaults are a starting point, not a validated coupled-fluid run.'),
              motion_function='coupled_scene.active_drive.body_pose',collider_asset='colliders.usda',
              runtime_note='Adapt native kinematic rotation/translation at every substep. Existing surface/pour runners do not directly support these layouts.',
              camera_file='cameras.json')
    save_json(folder/'scene_spec.json',spec);write_usd(folder/'colliders.usda',recipe['bodies'],case,origin,fps)
    # Verify actual Blender transforms and camera coverage at all design samples.
    worst=0.;angle_error=0.;coverage=[];swept_points=[]
    for sample in samples:
        scene.frame_set(sample['frame']);bpy.context.view_layer.update()
        parent=roots[sample['target']]
        worst=max(worst,(parent.matrix_world.translation-convert(sample['position_m'])).length)
        axis=[0.,0.,0.];axis[sample['rotation_axis']]=1.
        expected=Quaternion(convert(axis),sample['angle_rad'])
        # Dot-product comparison treats q and -q as the same rotation.
        angle_error=max(angle_error,1-abs(parent.matrix_world.to_quaternion().dot(expected)))
        if (sample['frame']-1)%10==0:
            swept_points.extend(obj.matrix_world@Vector(p) for obj in visuals for p in obj.bound_box)
    assert worst<1e-6, worst
    assert angle_error<1e-6,angle_error
    # Fit every fixed camera to the swept apparatus, keeping its direction.
    # Cameras remain stationary throughout recording; no auto-tracking motion.
    for cam in cameras:
        for attempt in range(20):
            projected=[world_to_camera_view(scene,cam,p) for p in swept_points]
            if all(.045<p.x<.955 and .045<p.y<.955 and p.z>0 for p in projected): break
            cam.location=target+(cam.location-target)*1.07
            bpy.context.view_layer.update()
        else: raise RuntimeError('Cannot frame apparatus sweep: '+cam.name)
    for t in case['review_times_s']:
        scene.frame_set(round(t*fps)+1);bpy.context.view_layer.update()
        for cam in cameras:
            projected=[world_to_camera_view(scene,cam,obj.matrix_world@Vector(p)) for obj in visuals for p in obj.bound_box]
            fits=all(.025<p.x<.975 and .025<p.y<.975 and p.z>0 for p in projected)
            coverage.append(dict(seconds=t,camera=cam.name,fits= fits))
    assert all(row['fits'] for row in coverage),coverage
    save_json(folder/'cameras.json',[camera_report(c,scene) for c in cameras])
    save_json(folder/'layout_audit.json',dict(blender_translation_max_error_m=worst,
        quaternion_dot_error=angle_error,camera_coverage=coverage,swept_framing_sample_hz=fps/10,fluid_simulated=False))
    scene.frame_set(1);bpy.ops.wm.save_as_mainfile(filepath=str(folder/(case['id']+'.blend')))
    for index,t in enumerate(case['review_times_s']):
        scene.frame_set(round(t*fps)+1);scene.camera=cameras[0]
        scene.render.filepath=str(folder/f'pose_{index:02d}.png');bpy.ops.render.render(write_still=True)
    scene.camera=cameras[2];scene.frame_set(1);scene.render.filepath=str(folder/'alternate.png');bpy.ops.render.render(write_still=True)
    print('COMPLETE '+case['id'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/active_drive_scene.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--environment',type=Path,default=Path(r'Y:\scenes\warehouse.blend'))
    parser.add_argument('--samples',type=int,default=32)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for case in cfg['cases']: build(cfg,case,args)
    save_json(args.output/'build_complete.json',dict(complete=True,cases=[c['id'] for c in cfg['cases']],fluid_simulated=False))


if __name__=='__main__': main()
