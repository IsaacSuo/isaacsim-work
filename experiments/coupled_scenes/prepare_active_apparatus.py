"""Extract evaluated accepted stirring/piston meshes and safe 4 mm fills.

Blender only. Original scenes are read, never overwritten. No simulation/render.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def native(p):
    return [p[0],p[2],-p[1]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spacing',type=float,choices=[.003,.004],default=.004)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    spec=json.loads((args.design/'design_spec.json').read_text(encoding='utf-8'))
    key=spec['case']['id'];assert key in ('02_stirring','03_piston_push')
    source=args.design/(key+'.blend');assert not args.output.exists()
    bpy.ops.wm.open_mainfile(filepath=str(source));scene=bpy.context.scene
    scene.frame_set(1);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
    root=bpy.data.objects['DesignRoot'];moving=bpy.data.objects[spec['body_root']]
    if key=='02_stirring':
        static=[bpy.data.objects['Open round glass bowl']]
        moving_objects=[o for o in moving.children if o.type=='MESH']
        low=np.array([-.278,.046,-.278]);high=np.array([.278,.230,.278])
    else:
        names=['Channel ceramic floor','Rounded far end']
        static=[bpy.data.objects[n] for n in names]+[o for o in root.children if o.name.startswith('Thin observation side')]
        moving_objects=[o for o in moving.children if o.type=='MESH']
        low=np.array([-.478,.080,-.153]);high=np.array([.620,.210,.153])
    arrays={};report={};local_parts={'donor':[],'receiver':[]};world_bvhs=[]
    origin=np.array(native(root.matrix_world.translation))
    for body,objects,parent in [('donor',moving_objects,moving),('receiver',static,root)]:
        verts=[];faces=[];offset=0;parts=[]
        for obj in objects:
            evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
            matrix=parent.matrix_world.inverted()@evaluated.matrix_world
            v=np.asarray([native(matrix@vert.co) for vert in mesh.vertices],dtype=np.float32)
            f=np.asarray([list(t.vertices) for t in mesh.loop_triangles],dtype=np.int32)
            edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
            _,counts=np.unique(edges,axis=0,return_counts=True);assert np.all(counts==2),obj.name
            tri=v[f].astype(float);volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
            assert volume>0,(obj.name,volume)
            world=v+np.array(native(parent.matrix_world.translation),dtype=np.float32)
            bvh=BVHTree.FromPolygons([Vector(p) for p in world],f.tolist(),all_triangles=True)
            world_bvhs.append((bvh,body=='donor' or key=='03_piston_push'))
            local_parts[body].append(v);verts.append(v);faces.append(f+offset);offset+=len(v)
            parts.append(dict(name=obj.name,vertices=len(v),triangles=len(f),volume_m3=float(volume)))
            evaluated.to_mesh_clear()
        arrays[body+'_vertices']=np.concatenate(verts);arrays[body+'_triangles']=np.concatenate(faces)
        report[body]=dict(position_m=native(parent.matrix_world.translation),vertices=offset,
            triangles=len(arrays[body+'_triangles']),parts=parts,kinematic=body=='donor',approximation='none')
    positions=[];minimum=float('inf');spacing=args.spacing;clearance=.008
    for y in np.arange(low[1],high[1]+1e-8,spacing):
        for x in np.arange(low[0],high[0]+1e-8,spacing):
            for z in np.arange(low[2],high[2]+1e-8,spacing):
                if key=='02_stirring' and x*x+z*z>.278**2:continue
                candidate=Vector((x+origin[0],y+origin[1],z+origin[2]));distance=float('inf');valid=True
                for bvh,convex in world_bvhs:
                    hit,normal,_,dist=bvh.find_nearest(candidate)
                    if dist<clearance or (convex and (candidate-hit).dot(normal)<0):valid=False;break
                    distance=min(distance,dist)
                if valid:positions.append(tuple(candidate));minimum=min(minimum,distance)
    arrays['positions']=np.asarray(positions,dtype=np.float32);arrays['velocities']=np.zeros_like(arrays['positions'])
    assert 10000<len(positions)<2500000
    # A conservative whole-sweep separation check independent of sampling rate.
    v=arrays['donor_vertices'];pivot=np.array(report['donor']['position_m'])-origin
    if key=='02_stirring':
        radius=np.hypot(v[:,0],v[:,2]).max();bottom=v[:,1].min()+pivot[1]
        assert radius<.24 and bottom>.060,(radius,bottom)
        sweep=dict(maximum_rotating_radius_m=float(radius),lowest_moving_y_m=float(bottom),
            note='Rotating mesh fits inside bowl inner radius >= .274 above floor; spindle extends through open top')
    else:
        extent_min=v.min(0)+pivot;extent_max=v.max(0)+pivot
        end=extent_max[0]+spec['case']['motion']['stroke_m']
        assert extent_min[1]>.069 and extent_max[2]<.167 and extent_min[2]>-.167 and end<.632
        sweep=dict(bottom_geometric_gap_m=float(extent_min[1]-.069),
            minimum_side_geometric_gap_m=float(min(.167-extent_max[2],extent_min[2]+.167)),
            final_front_x_m=float(end),note='Small geometric gaps: native particle sealing/leakage still needs testing')
    args.output.mkdir(parents=True,exist_ok=False);np.savez(args.output/'geometry_and_fill.npz',**arrays)
    static_v=arrays['receiver_vertices'];bounds=[static_v.min(0).tolist(),static_v.max(0).tolist()]
    report.update(product='active_apparatus_exact_mesh_assets',case=spec['case'],moving_object=spec['body_root'],
        source_blend=str(source.resolve()),source_blend_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        geometry_sha256=hashlib.sha256((args.output/'geometry_and_fill.npz').read_bytes()).hexdigest(),
        particle_count=len(positions),particle_spacing_m=spacing,nominal_lattice_liters=len(positions)*spacing**3*1000,
        fill_cavity_axis_local_m=[0.,0.,0.],diagnostic_origin_m=origin.tolist(),containment_bounds_local_m=bounds,
        initial_fill_bounds_local_m=[low.tolist(),high.tolist()],minimum_initial_distance_to_shell_m=minimum,
        evaluated_modifiers_in_collision=True,initial_velocities_zero=True,sweep_clearance=sweep,
        physics_validated=False,excluded_visuals='support, motor/housing and worktop; no artificial liquid forces')
    report['contact_revision']=spec.get('contact_revision','original_v3')
    (args.output/'assets.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,case=key,particles=len(positions),clearance=minimum,sweep=sweep),indent=2),flush=True)


if __name__=='__main__':main()
