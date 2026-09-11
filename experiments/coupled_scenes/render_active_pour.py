"""Render accepted v3 design using native water and recorded pitcher transforms."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector,Quaternion

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.coupled_scenes.blender_coupled_event_overlay import _read_obj,_water_material
from experiments.coupled_scenes.run_active_pour_probe import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend',type=Path,required=True)
    parser.add_argument('--surfaces',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    sequence=json.loads((args.surfaces/'sequence.json').read_text(encoding='utf-8'));assert sequence['complete']
    if sequence.get('source_blend_sha256'):
        assert hashlib.sha256(args.blend.read_bytes()).hexdigest()==sequence['source_blend_sha256'],'Mismatched pitcher layout'
    args.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(args.blend));scene=bpy.context.scene
    scene.camera=bpy.data.objects['DesignView_00'];scene.render.fps=30
    scene.render.resolution_x=1280;scene.render.resolution_y=960;scene.render.resolution_percentage=100
    scene.cycles.samples=64;scene.cycles.seed=0;scene.cycles.use_animated_seed=False;scene.render.use_motion_blur=False
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for device in prefs.devices:device.use=device.type!='CPU'
    scene.cycles.device='GPU'
    donor=bpy.data.objects[sequence.get('moving_object','PouringPitcher')];donor.animation_data_clear();donor.rotation_mode='QUATERNION'
    parent_inverse=donor.parent.matrix_world.inverted()
    water=bpy.data.objects.new('NativeWater',bpy.data.meshes.new('EmptyWater'));scene.collection.objects.link(water)
    material=_water_material();rendered=[]
    for index,frame in enumerate(sequence['frames']):
        scene.frame_set(index+1)
        x,y,z=frame['native_position_m'];donor.location=parent_inverse@Vector((x,-z,y))
        qx,qy,qz,qw=frame['native_rotation_xyzw'];donor.rotation_quaternion=Quaternion((qw,qx,-qz,qy))
        bpy.context.view_layer.update()
        vertices,faces=_read_obj(args.surfaces/frame['surface'])
        mesh=bpy.data.meshes.new(f'Water_{index:04d}');mesh.from_pydata(vertices,[],faces);mesh.update()
        mesh.materials.append(material)
        for polygon in mesh.polygons:polygon.use_smooth=True
        old=water.data;water.data=mesh
        if old.users==0:bpy.data.meshes.remove(old)
        del vertices,faces
        filename=f'frame_{index:04d}.png';scene.render.filepath=str(args.output/filename);bpy.ops.render.render(write_still=True)
        rendered.append(dict(file=filename,action_seconds=frame['action_seconds']))
        atomic_json(args.output/'render_manifest.json',dict(complete=len(rendered)==len(sequence['frames']),frames=rendered,fps=30,
            diagnostic_only=sequence.get('diagnostic_only',False)))
        print(f'RENDER {index+1}/{len(sequence["frames"])}',flush=True)


if __name__=='__main__':main()
