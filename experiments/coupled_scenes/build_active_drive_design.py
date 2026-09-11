"""Blender-only v3 appearance study. No physics assets or fluid are produced.

The redesigned vessel geometry intentionally does NOT reuse v2 collision USD.
Approve the appearance before matching collision geometry and native adapters.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector, Quaternion
from bpy_extras.object_utils import world_to_camera_view

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from coupled_scene.active_drive import motion_state
from experiments.coupled_scenes.build_surface_study import (
    box, empty, material, move_to, convert, camera_report,
)


def mesh_object(name,verts,faces,mat,collection,parent=None):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata([convert(v) for v in verts],[],faces);mesh.update()
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
    obj=bpy.data.objects.new(name,mesh);collection.objects.link(obj);obj.parent=parent
    mesh.materials.append(mat)
    for polygon in mesh.polygons: polygon.use_smooth=True
    return obj


def vessel(name,profile,mat,collection,parent=None,centre=(0,0,0),ellipse=1.,spout=False):
    """Closed lathed solid cross-section with real wall thickness and open mouth."""
    n=128;verts=[];rings=[]
    for radius,height in profile:
        if radius==0:
            rings.append([len(verts)]);verts.append((centre[0],centre[1]+height,centre[2]));continue
        ring=[]
        for j in range(n):
            angle=2*math.pi*j/n
            beak=max(0.,math.cos(angle))**24*max(0.,min(1.,(height-.20)/.10)) if spout else 0.
            rr=radius+.060*beak
            ring.append(len(verts));verts.append((centre[0]+rr*math.cos(angle),centre[1]+height-.010*beak,centre[2]+radius*math.sin(angle)*ellipse))
        rings.append(ring)
    faces=[]
    for a,b in zip(rings,rings[1:]):
        for j in range(n):
            k=(j+1)%n
            if len(a)==1: faces.append((a[0],b[j],b[k]))
            elif len(b)==1: faces.append((a[j],b[0],a[k]))
            else: faces.append((a[j],b[j],b[k],a[k]))
    return mesh_object(name,verts,faces,mat,collection,parent)


def tube(name,points,radius,mat,collection,parent=None):
    data=bpy.data.curves.new(name,'CURVE');data.dimensions='3D';data.resolution_u=16
    data.bevel_depth=radius;data.bevel_resolution=4;data.use_fill_caps=True
    spline=data.splines.new('BEZIER');spline.bezier_points.add(len(points)-1)
    for p,co in zip(spline.bezier_points,points):
        p.co=convert(co);p.handle_left_type='AUTO';p.handle_right_type='AUTO'
    obj=bpy.data.objects.new(name,data);collection.objects.link(obj);obj.parent=parent;data.materials.append(mat)
    return obj


def cylinder(name,centre,radius,height,mat,collection,parent=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=radius,depth=height)
    obj=move_to(bpy.context.object,collection);obj.name=name;obj.parent=parent;obj.location=convert(centre)
    obj.data.materials.append(mat)
    bevel=obj.modifiers.new('Soft machined edges','BEVEL');bevel.width=min(.004,height/5);bevel.segments=3
    for p in obj.data.polygons:p.use_smooth=True
    return obj


def animate(obj,case,base,fps):
    axis=[0.,0.,0.];axis[case['motion']['axis']]=1.;axis=convert(axis)
    obj.rotation_mode='QUATERNION'
    for f in range(1,round(case['duration_s']*fps)+2):
        state=motion_state(case['motion'],(f-1)/fps)
        obj.location=convert([a+b for a,b in zip(base,state['displacement_m'])])
        obj.rotation_quaternion=Quaternion(axis,state['angle_rad'])
        obj.keyframe_insert(data_path='location',frame=f);obj.keyframe_insert(data_path='rotation_quaternion',frame=f)
    for layer in obj.animation_data.action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for point in curve.keyframe_points: point.interpolation='LINEAR'


def build(case,args):
    bpy.ops.wm.open_mainfile(filepath=str(args.environment));scene=bpy.context.scene
    collection=bpy.data.collections.new('ActiveDrive_DesignOnly');scene.collection.children.link(collection)
    origin=[.041443,.2199969271881278,-8.076415]
    root=empty('DesignRoot',origin,collection)
    mats=dict(stone=material('Warm honed limestone',(.26,.225,.18),.68),
        ceramic=material('Ivory satin ceramic',(.64,.58,.46),.27),
        dark=material('Graphite anodised',(.018,.028,.029),.32,metallic=.55),
        bronze=material('Brushed warm alloy',(.23,.115,.045),.31,metallic=.72),
        glass=material('Thin clear borosilicate',(.99,.995,1),.025,1.),
        steel=material('Brushed stainless',(.29,.34,.35),.27,metallic=.85),
        teal=material('Deep petrol enamel',(.025,.10,.105),.29))
    # Subtle stone texture; no large-frequency bump competing with the water.
    nodes=mats['stone'].node_tree.nodes;bsdf=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    noise=nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=160
    bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.0004
    mats['stone'].node_tree.links.new(noise.outputs['Fac'],bump.inputs['Height']);mats['stone'].node_tree.links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
    key=case['id'];moving=None
    table_width=1.75 if key!='02_stirring' else 1.20
    box('Honed stone countertop',[0,-.047,0],[table_width,.065,.96],mats['stone'],collection,root,.028)
    box('Recessed dark underside',[0,-.087,0],[table_width-.10,.020,.86],mats['dark'],collection,root,.012)
    for xx in [-table_width/2+.12,table_width/2-.12]:
        for zz in [-.34,.34]: cylinder('Slender leg',[xx,-.402,zz],.022,.62,mats['dark'],collection,root)
    if key=='01_container_transfer':
        basin=vessel('Low oval receiving basin',[(0,0),(.37,0),(.43,.007),(.47,.025),(.52,.13),(.525,.145),
            (.521,.150),(.509,.150),(.505,.14),(.49,.105),(.452,.035),(.425,.024),(0,.024)],mats['ceramic'],collection,root,ellipse=.62)
        base=[-.23,.68,0];moving=empty('PouringPitcher',base,collection);moving.parent=root
        profile=[(0,0),(.095,0),(.104,.007),(.107,.023),(.112,.08),(.132,.24),(.145,.302),
                 (.146,.31),(.142,.313),(.138,.31),(.136,.301),(.124,.24),(.104,.08),(.099,.025),(0,.014)]
        centre=[-.205,-.303,0]
        vessel('Spouted ceramic pitcher',profile,mats['ceramic'],collection,moving,centre=centre,spout=True)
        handle=[(-.135,.26,0),(-.177,.267,0),(-.222,.205,0),(-.218,.126,0),(-.176,.084,0),(-.113,.09,0)]
        tube('Rounded pitcher handle',[[a+b for a,b in zip(p,centre)] for p in handle],.012,mats['ceramic'],collection,moving)
        cylinder('Rear tilt base',[-.23,.018,-.34],.065,.025,mats['dark'],collection,root)
        tube('Single rear support',[[-.23,.03,-.34],[-.23,.54,-.34],[-.23,.65,-.31],[-.23,.68,-.245]],.013,mats['dark'],collection,root)
        tube('Tilt axis',[[-.23,.68,-.26],[-.23,.68,-.18]],.016,mats['bronze'],collection,root)
        tube('Rear rotating cradle',[[0,0,-.19],[-.115,-.08,-.19],[-.205,-.205,-.108]],.009,mats['dark'],collection,moving)
        target=[-.08,.46,0];radius=2.0;elevation=.95
        note='Ceramic spouted pitcher + low oval basin; single rear tilt support. Matching new collision geometry is pending.'
    elif key=='02_stirring':
        cylinder('Round recessed pedestal',[0,.012,0],.334,.024,mats['dark'],collection,root)
        vessel('Open round glass bowl',[(0,.026),(.268,.026),(.285,.032),(.296,.05),(.304,.30),
            (.305,.327),(.302,.332),(.298,.332),(.297,.327),(.296,.30),(.288,.052),(.274,.036),(0,.036)],mats['glass'],collection,root)
        base=[0,.105,0];moving=empty('QuietThreeBladeImpeller',base,collection);moving.parent=root
        cylinder('Slim spindle',[0,.243,0],.006,.535,mats['steel'],collection,moving)
        cylinder('Impeller hub',[0,0,0],.024,.026,mats['bronze'],collection,moving)
        for i in range(3):
            a=i*2*math.pi/3
            blade=box('Rounded paddle', [.110*math.cos(a),-.01,.110*math.sin(a)],[.17,.016,.040],mats['teal'],collection,moving,.007)
            blade.rotation_euler=Quaternion(convert([0,1,0]),-a).to_euler()
        tube('Single rear mixer arm',[[0,.032,-.415],[0,.56,-.415],[0,.71,-.37],[0,.72,-.12],[0,.72,0]],.012,mats['dark'],collection,root)
        cylinder('Compact mixer head',[0,.678,0],.032,.093,mats['dark'],collection,root)
        cylinder('Head trim',[0,.631,0],.033,.007,mats['bronze'],collection,root)
        cylinder('Rear pedestal',[0,.012,-.415],.052,.024,mats['dark'],collection,root)
        target=[0,.34,0];radius=1.65;elevation=.82
        note='Round thin-wall glass vessel, slim shaft and three small paddles; motor is above the vessel, support behind it.'
    else:
        box('Rounded concealed drive base',[-.08,.012,0],[1.49,.065,.405],mats['dark'],collection,root,.035)
        box('Channel ceramic floor',[.025,.054,0],[1.26,.03,.326],mats['ceramic'],collection,root,.012)
        for zz in [-.169,.169]:
            box('Thin observation side',[.025,.168,zz],[1.26,.21,.004],mats['glass'],collection,root,.0015)
            box('Recessed edge rail',[.025,.059,zz],[1.28,.016,.016],mats['dark'],collection,root,.005)
        box('Rounded far end',[.655,.168,0],[.045,.23,.344],mats['ceramic'],collection,root,.019)
        box('Integrated drive housing',[-.692,.134,0],[.20,.255,.374],mats['dark'],collection,root,.043)
        box('Housing trim',[-.591,.134,0],[.004,.19,.28],mats['bronze'],collection,root,.001)
        base=[-.50,0,0];moving=empty('ConcealedCarriagePiston',base,collection);moving.parent=root
        box('Piston face',[0,.163,0],[.027,.184,.323],mats['teal'],collection,moving,.009)
        box('Piston top edge',[0,.258,0],[.039,.012,.323],mats['steel'],collection,moving,.004)
        target=[-.06,.18,0];radius=1.95;elevation=.90
        note='Drive and carriage concealed below opaque base; no exposed long rear rod. Seal/slot/collision engineering not yet defined.'
    animate(moving,case,base,60)
    for obj in collection.objects:
        for modifier in obj.modifiers:
            if modifier.type=='BEVEL': modifier.segments=6
    scene.world=bpy.data.worlds.new('Soft daylight');scene.world.use_nodes=True
    nodes=scene.world.node_tree.nodes;nodes.clear();env=nodes.new('ShaderNodeTexEnvironment')
    env.image=bpy.data.images.load(str(args.environment.parent/'HDRI/bryanston_park_sunrise_8k.exr'),check_existing=True)
    bg=nodes.new('ShaderNodeBackground');bg.inputs['Strength'].default_value=.30;out=nodes.new('ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(env.outputs['Color'],bg.inputs['Color']);scene.world.node_tree.links.new(bg.outputs['Background'],out.inputs['Surface'])
    focus=convert([a+b for a,b in zip(origin,target)])
    for name,pos,power,size in [('Softbox',[-.60,1.8,.8],130,1.5),('Edge',[.5,1.2,-.7],90,1.0)]:
        light=bpy.data.lights.new(name,'AREA');light.energy=power;light.shape='DISK';light.size=size
        obj=bpy.data.objects.new(name,light);collection.objects.link(obj);obj.location=convert([a+b for a,b in zip(origin,pos)])
        obj.rotation_euler=(focus-obj.location).to_track_quat('-Z','Y').to_euler()
    scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for device in prefs.devices:device.use=device.type!='CPU'
    scene.cycles.device='GPU';scene.render.resolution_x=1280;scene.render.resolution_y=960;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.fps=60;scene.frame_start=1;scene.frame_end=601
    scene.render.use_motion_blur=False;scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.4
    cameras=[]
    for i in range(8):
        a=math.radians(-55+45*i);data=bpy.data.cameras.new(f'DesignView_{i:02d}');data.lens=44;data.sensor_width=36;data.sensor_fit='HORIZONTAL'
        camera=bpy.data.objects.new(data.name,data);collection.objects.link(camera)
        camera.location=focus+Vector((radius*math.cos(a),radius*math.sin(a),elevation))
        camera.rotation_euler=(focus-camera.location).to_track_quat('-Z','Y').to_euler();cameras.append(camera)
    scene.camera=cameras[0];scene.frame_set(1);bpy.context.view_layer.update()
    apparatus=[obj for obj in collection.objects if obj.type in ('MESH','CURVE') and not
               obj.name.startswith(('Slender leg','Honed stone','Recessed dark underside'))]
    sweep=[]
    for frame in range(1,602,30):
        scene.frame_set(frame);bpy.context.view_layer.update()
        sweep.extend(obj.matrix_world@Vector(p) for obj in apparatus for p in obj.bound_box)
    for camera in cameras:
        for attempt in range(20):
            projected=[world_to_camera_view(scene,camera,p) for p in sweep]
            if all(.035<p.x<.965 and .035<p.y<.965 and p.z>0 for p in projected): break
            camera.location=focus+(camera.location-focus)*1.06;bpy.context.view_layer.update()
        else: raise RuntimeError('Cannot frame full apparatus motion')
    scene.frame_set(1);bpy.context.view_layer.update()
    folder=args.output/key;folder.mkdir()
    spec=dict(case=case,product='appearance_concept_only',physics_ready=False,fluid_simulated=False,
        appearance_revision=3,note=note,old_collision_usd_compatible=False,
        review_times_s=[0,3.5,5.5],reference_water_shown=False,
        origin_isaac_m=origin,environment=str(args.environment),body_root=moving.name)
    (folder/'design_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'cameras.json').write_text(json.dumps([camera_report(c,scene) for c in cameras],indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/(key+'.blend')))
    for i,t in enumerate(spec['review_times_s']):
        scene.frame_set(round(t*60)+1);scene.render.filepath=str(folder/f'pose_{i:02d}.png');bpy.ops.render.render(write_still=True)
    scene.frame_set(1);scene.camera=cameras[1];scene.render.filepath=str(folder/'alternate.png');bpy.ops.render.render(write_still=True)
    print('COMPLETE '+key,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--environment',type=Path,default=Path(r'Y:\scenes\warehouse.blend'))
    parser.add_argument('--samples',type=int,default=48)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/active_drive_scene.json').read_text(encoding='utf-8'))
    for case in cfg['cases']:build(case,args)
    (args.output/'build_complete.json').write_text(json.dumps(dict(complete=True,cases=[c['id'] for c in cfg['cases']],
        physics_ready=False,fluid_simulated=False,appearance_revision=3),indent=2))


if __name__=='__main__':main()
