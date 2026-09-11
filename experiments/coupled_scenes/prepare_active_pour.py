"""Blender: extract exact accepted v3 vessel meshes and a safe native fill lattice."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def native(p): return [p[0],p[2],-p[1]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fill-height',type=float,default=.18)
    parser.add_argument('--particle-limit',type=int,help='Lower the filled surface to retain this many lattice particles; never thin the bulk')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    assert .08<=args.fill_height<=.24
    spec=json.loads((args.design/'design_spec.json').read_text(encoding='utf-8'))
    assert spec['appearance_revision']==3 and spec['case']['id']=='01_container_transfer'
    blend=args.design/'01_container_transfer.blend'
    bpy.ops.wm.open_mainfile(filepath=str(blend));bpy.context.scene.frame_set(1);bpy.context.view_layer.update()
    args.output.mkdir(parents=True,exist_ok=False)
    roots={'donor':bpy.data.objects['PouringPitcher'],'receiver':bpy.data.objects['DesignRoot']}
    names={'donor':'Spouted ceramic pitcher','receiver':'Low oval receiving basin'}
    data={};report={}
    for key,name in names.items():
        obj=bpy.data.objects[name];mesh=obj.data;mesh.calc_loop_triangles()
        matrix=roots[key].matrix_world.inverted()@obj.matrix_world
        vertices=np.asarray([native(matrix@v.co) for v in mesh.vertices],dtype=np.float64)
        faces=np.asarray([list(t.vertices) for t in mesh.loop_triangles],dtype=np.int32)
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        assert np.all(counts==2),'Vessel solid shell is not watertight'
        triangles=vertices[faces]
        volume=np.sum(np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2])))/6
        assert volume>0,'Vessel normals must face out of the solid'
        data[key+'_vertices']=vertices.astype(np.float32);data[key+'_triangles']=faces
        report[key]=dict(blender_object=name,position_m=native(roots[key].matrix_world.translation),
                        vertices=len(vertices),triangles=len(faces),solid_shell_volume_m3=float(volume),
                        watertight=True,approximation='none',kinematic=key=='donor')
    vertices=data['donor_vertices'];faces=data['donor_triangles']
    bvh=BVHTree.FromPolygons([Vector(v) for v in vertices],faces.tolist(),all_triangles=True)
    # Same cavity axis as the approved pitcher. Radial limits come from exact
    # inner mesh ray intersections, not a box/convex hull covering the mouth.
    axis=np.array([-.205,-.303,0.]);spacing=.004;clearance=.008;positions=[];distances=[]
    for yy in np.arange(.030,args.fill_height+1e-8,spacing):
        centre=axis+np.array([0,yy,0])
        for xx in np.arange(-.132,.132+1e-8,spacing):
            for zz in np.arange(-.132,.132+1e-8,spacing):
                radius=float(np.hypot(xx,zz));direction=Vector((xx,0,zz)).normalized() if radius>1e-9 else Vector((1,0,0))
                hit,_,_,distance=bvh.ray_cast(Vector(centre),direction,.3)
                if hit is None or radius>distance-clearance: continue
                candidate=centre+np.array([xx,0,zz]);nearest=bvh.find_nearest(Vector(candidate))
                if nearest[3]<clearance: continue
                positions.append(candidate);distances.append(nearest[3])
    positions=np.asarray(positions,dtype=np.float32)
    original_count=len(positions)
    if args.particle_limit is not None:
        assert 10000<args.particle_limit<=original_count
        # Whole lower layers first; any partial top layer is a centred patch.
        # Keep the original order afterwards so lower-layer rows are unchanged.
        radial_squared=(positions[:,0]-axis[0])**2+(positions[:,2]-axis[2])**2
        chosen=np.sort(np.lexsort((radial_squared,positions[:,1]))[:args.particle_limit])
        positions=positions[chosen]
        distances=np.asarray(distances)[chosen]
    actual_top=float(positions[:,1].max()-axis[1])
    positions=positions+np.asarray(report['donor']['position_m'],dtype=np.float32)
    assert len(positions)>10000
    data['positions']=positions;data['velocities']=np.zeros_like(positions)
    np.savez(args.output/'geometry_and_fill.npz',**data)
    report.update(product='active_pour_exact_mesh_assets',source_blend=str(blend.resolve()),
        source_blend_sha256=hashlib.sha256(blend.read_bytes()).hexdigest(),case=spec['case'],
        particle_spacing_m=spacing,particle_count=len(positions),nominal_lattice_liters=len(positions)*spacing**3*1000,
        fill_cavity_axis_local_m=axis.tolist(),initial_fill_top_above_pitcher_bottom_m=actual_top,
        requested_fill_height_m=args.fill_height,particle_limit=args.particle_limit,unlimited_lattice_count=original_count,
        minimum_initial_distance_to_shell_m=min(distances),initial_velocities_zero=True,
        geometry_sha256=hashlib.sha256((args.output/'geometry_and_fill.npz').read_bytes()).hexdigest(),
        excluded_visuals='handle, rear support and worktop are not part of the two liquid-contact vessels',
        physics_validated=False)
    (args.output/'assets.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
