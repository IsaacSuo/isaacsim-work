"""Blender: lower the accepted pitcher and its support without changing vessels.

Writes a new design directory; no rendering or fluid simulation.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.coupled_scenes.build_active_drive_design import animate
from experiments.coupled_scenes.build_surface_study import camera_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lowering-m',type=float,default=.20)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    assert math.isfinite(args.lowering_m) and 0<args.lowering_m<=.20
    assert not args.output.exists()
    spec=json.loads((args.design/'design_spec.json').read_text(encoding='utf-8'))
    assert spec['appearance_revision']==3 and not spec.get('pitcher_lowering_m')
    source=args.design/'01_container_transfer.blend'
    bpy.ops.wm.open_mainfile(filepath=str(source))
    scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update()
    donor=bpy.data.objects['PouringPitcher']
    old=donor.location.copy();base=[old.x,old.z-args.lowering_m,-old.y]
    donor.animation_data_clear()
    animate(donor,spec['case'],base,scene.render.fps)
    # Cradle and handle follow the pitcher parent. Only fixed support geometry
    # needs adjustment; keep its base and the receiving basin unchanged.
    support=bpy.data.objects['Single rear support'].data.splines[0].bezier_points
    assert len(support)==4
    for point in list(support)[1:]:
        point.co.z-=args.lowering_m
        point.handle_left.z-=args.lowering_m
        point.handle_right.z-=args.lowering_m
    for point in bpy.data.objects['Tilt axis'].data.splines[0].bezier_points:
        point.co.z-=args.lowering_m
        point.handle_left.z-=args.lowering_m
        point.handle_right.z-=args.lowering_m
    scene.frame_set(1);bpy.context.view_layer.update()
    shell=bpy.data.objects['Spouted ceramic pitcher']
    basin=bpy.data.objects['Low oval receiving basin']
    local=donor.matrix_world.inverted()@shell.matrix_world
    # Exact vertical minimum of each vertex over the full tilt angle interval.
    # Rotation around native Z becomes Blender -Y. Local native (x,y) = (x,z).
    low=math.radians(spec['case']['motion']['angle_deg']);high=0.
    assert low<high
    min_y=float('inf')
    for vertex in shell.data.vertices:
        p=local@vertex.co;a=p.x;b=p.z
        critical=math.atan2(a,b)
        angles=[low,high]+[critical+k*math.pi for k in range(-3,4) if low<=critical+k*math.pi<=high]
        min_y=min(min_y,min(a*math.sin(t)+b*math.cos(t) for t in angles)+donor.matrix_world.translation.z)
    rim=max((basin.matrix_world@v.co).z for v in basin.data.vertices)
    clearance=min_y-rim
    assert clearance>=.02,clearance
    scene.frame_set(1);bpy.context.view_layer.update()
    spec.update(layout_revision=4,pitcher_lowering_m=args.lowering_m,
        source_design=str(args.design.resolve()),source_blend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        note='Accepted ceramic geometry; pitcher lowered with matching fixed support. Basin, cameras and motion timing unchanged.',
        minimum_full_tilt_vertical_clearance_m=clearance,physics_ready=False,fluid_simulated=False)
    spec['case']['donor_lip_pivot_m']=base
    args.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/'01_container_transfer.blend'))
    (args.output/'design_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    cameras=sorted((o for o in bpy.data.objects if o.name.startswith('DesignView_')),key=lambda o:o.name)
    (args.output/'cameras.json').write_text(json.dumps([camera_report(c,scene) for c in cameras],indent=2))
    print(json.dumps(dict(complete=True,lowering_m=args.lowering_m,pivot_local_native_m=base,
        minimum_full_tilt_vertical_clearance_m=clearance,output=str(args.output)),indent=2),flush=True)


if __name__=='__main__':
    main()
