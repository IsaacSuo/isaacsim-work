"""Blender: matched-camera animation from reconstructed native snapshots."""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.coupled_scenes.blender_coupled_event_overlay import _read_obj,_water_material


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend',type=Path,required=True)
    parser.add_argument('--surfaces',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=96)
    parser.add_argument('--width',type=int,default=1600)
    parser.add_argument('--height',type=int,default=1200)
    parser.add_argument('--frame-limit',type=int,help='Render a prefix for launch validation; not a complete sequence')
    parser.add_argument('--resume',action='store_true',help='Resume only frames recorded by this renderer with matching inputs')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    manifest=json.loads((args.surfaces/'sequence.json').read_text())
    assert manifest['complete'] and len(manifest['frames'])>=2
    if args.frame_limit is not None and args.frame_limit < 1:
        parser.error('--frame-limit must be positive')
    identity=dict(blend=str(args.blend.resolve()),surfaces=str(args.surfaces.resolve()),
                  samples=args.samples,width=args.width,height=args.height,
                  source_sequence=manifest)
    checkpoint=args.output/'render_manifest.json'
    rendered=[]
    if args.resume and args.output.exists():
        previous=json.loads(checkpoint.read_text())
        if previous.get('identity')!=identity:
            raise RuntimeError('Resume inputs/settings differ; refusing to reuse rendered frames')
        rendered=previous['frames']
        for i,entry in enumerate(rendered):
            path=args.output/entry['file']
            if entry['file']!=f'frame_{i:04d}.png' or not path.is_file() or path.stat().st_size<100:
                raise RuntimeError(f'Invalid resume frame: {path}')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
    def save_progress():
        temporary=checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(complete=len(rendered)==len(manifest['frames']),
            fps=30,identity=identity,frames=rendered),indent=2))
        temporary.replace(checkpoint)
    save_progress()
    bpy.ops.wm.open_mainfile(filepath=str(args.blend))
    scene=bpy.context.scene
    scene.camera=bpy.data.objects.get('WaterSurfaceLowAngle') or bpy.data.objects['View_00']
    scene.render.fps=30
    scene.render.resolution_x=args.width;scene.render.resolution_y=args.height
    scene.cycles.samples=args.samples;scene.cycles.seed=0
    scene.cycles.use_animated_seed=False
    scene.render.use_motion_blur=False
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX';prefs.get_devices()
    for device in prefs.devices:device.use=device.type!='CPU'
    scene.cycles.device='GPU'
    obj=bpy.data.objects.get('NativeFinalWater_t8s')
    if obj is None:
        placeholder=bpy.data.objects.get('REFERENCE_WATER_NOT_SIMULATED')
        if placeholder:bpy.data.objects.remove(placeholder,do_unlink=True)
        obj=bpy.data.objects.new('NativeActionWater',bpy.data.meshes.new('EmptyWater'))
        bpy.data.collections['SurfaceStudy'].objects.link(obj)
        mat=_water_material()
    else:mat=obj.data.materials[0]
    action=manifest.get('action')
    drive=None
    if action:
        drive=bpy.data.objects[action['case']['motion']['target']]
        drive.animation_data_clear()
    label=bpy.data.objects.get('Reference label')
    if label:label.data.body='PHYSX WATER / NORMAL DAMPING / 25x SURFACE SMOOTHING'
    for i,frame in enumerate(manifest['frames']):
        if args.frame_limit is not None and i>=args.frame_limit:break
        if i<len(rendered):continue
        vertices,faces=_read_obj(args.surfaces/frame['surface'])
        mesh=bpy.data.meshes.new(f'WaterFrame_{i:04d}')
        mesh.from_pydata(vertices,[],faces);mesh.update()
        for poly in mesh.polygons:poly.use_smooth=True
        mesh.materials.append(mat)
        old=obj.data;obj.data=mesh
        if old.users==0:bpy.data.meshes.remove(old)
        del vertices,faces
        scene.frame_set(i+1)
        if drive:
            pose=frame['target_pose_isaac_m']
            drive.location=Vector((pose[0],-pose[2],pose[1]))
            bpy.context.view_layer.update()
        scene.render.filepath=str(args.output/f'frame_{i:04d}.png')
        bpy.ops.render.render(write_still=True)
        rendered.append(dict(file=f'frame_{i:04d}.png',recording_seconds=frame['recording_seconds'],target_pose_isaac_m=frame.get('target_pose_isaac_m')))
        save_progress()
        print(f'[sequence-render] {i+1}/{len(manifest["frames"])}',flush=True)


if __name__=='__main__':main()
