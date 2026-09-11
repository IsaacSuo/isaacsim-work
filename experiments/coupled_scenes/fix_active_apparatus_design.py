"""Blender: derive seam-free impeller and close-fitting piston from accepted v3."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import bpy

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.coupled_scenes.build_surface_study import box


def apply_modifiers(obj):
    bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
    for modifier in list(obj.modifiers):bpy.ops.object.modifier_apply(modifier=modifier.name)


def union_objects(base,others):
    apply_modifiers(base)
    for obj in others:
        apply_modifiers(obj)
        bpy.context.view_layer.objects.active=base
        modifier=base.modifiers.new('Continuous welded solid','BOOLEAN');modifier.operation='UNION';modifier.solver='EXACT';modifier.object=obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(obj,do_unlink=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    spec=json.loads((args.design/'design_spec.json').read_text(encoding='utf-8'));key=spec['case']['id']
    assert key in ('02_stirring','03_piston_push') and not args.output.exists()
    source=args.design/(key+'.blend');bpy.ops.wm.open_mainfile(filepath=str(source));bpy.context.scene.frame_set(1);bpy.context.view_layer.update()
    moving=bpy.data.objects[spec['body_root']]
    if key=='02_stirring':
        hub=bpy.data.objects['Impeller hub']
        # Widen only the hub: overlap all three blade roots before boolean union.
        for v in hub.data.vertices:v.co.x*=.036/.024;v.co.y*=.036/.024
        union_objects(hub,[o for o in list(moving.children) if o.type=='MESH' and o!=hub])
        hub.name='Continuous impeller and spindle'
        changes='Hub radius 24 to 36 mm; exact union with all three paddles and spindle, no internal contact faces'
    else:
        floor=bpy.data.objects['Channel ceramic floor']
        # Close floor/side seam and remove large top corner rounding.
        for v in floor.data.vertices:v.co.y*=.338/.326
        for modifier in floor.modifiers:
            if modifier.type=='BEVEL':modifier.width=.0005
        face=bpy.data.objects['Piston face'];collection=face.users_collection[0]
        seal=box('Close fitting piston seal',[0,.165,0],[.029,.1904,.331],face.data.materials[0],collection,moving,.0003)
        union_objects(face,[seal,bpy.data.objects['Piston top edge']]);face.name='Sealed piston face'
        changes='Floor width 338 mm, edge radius .5 mm; integrated piston seal bottom .0698 m, sides +/- .1655 m, .3 mm edge radius'
    bpy.context.view_layer.update()
    spec.update(layout_revision=4,contact_revision='seam_seal_v2',source_blend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                contact_changes=changes,physics_ready=False,fluid_simulated=False)
    args.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/(key+'.blend')))
    (args.output/'design_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'cameras.json').write_bytes((args.design/'cameras.json').read_bytes())
    print('COMPLETED '+key+' '+changes,flush=True)


if __name__=='__main__':main()
