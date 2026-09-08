"""Opt-in guide geometry and upstream-only particle sink for a PBD inlet."""
import json
import numpy as np


def guide_panels(config):
    outlet = np.asarray(config['outlet_centre'], dtype=float)
    width, depth = config['inner_size_xz']
    thickness = float(config['wall_thickness'])
    height = float(config['upstream_height']) + float(config['headroom'])
    values = np.array([width, depth, thickness, height, config['upstream_height'], config['headroom']], dtype=float)
    if outlet.shape != (3,) or not np.isfinite(outlet).all() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid conditioned-inlet dimensions')
    x,y,z = outlet
    cy = y + height/2
    return [
        ('Left', (x-(width+thickness)/2,cy,z), (thickness,height,depth+2*thickness)),
        ('Right', (x+(width+thickness)/2,cy,z), (thickness,height,depth+2*thickness)),
        ('Front', (x,cy,z-(depth+thickness)/2), (width,height,thickness)),
        ('Back', (x,cy,z+(depth+thickness)/2), (width,height,thickness)),
    ]


def upstream_sink_mask(positions, ids, discharged, outlet_y, top_y):
    """Once discharged, a particle is never eligible for the upstream sink."""
    discharged[ids[positions[:,1] < outlet_y]] = True
    return (positions[:,1] > top_y) & ~discharged[ids]


class ConditionedInlet:
    def __init__(self, event, config):
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
        from omni.physx.scripts import physicsUtils
        if event.particle_set_strategy != 'chunked_density_sets' or not event.manual_substep_emission:
            raise ValueError('Conditioned inlet requires chunked density sets and manual substeps')
        self.event, self.config = event, config
        source = event.configuration['source']
        outlet = np.asarray(config['outlet_centre'], dtype=float)
        if not np.allclose(source['centre'], outlet + [0,config['upstream_height'],0], atol=1e-7, rtol=0):
            raise ValueError('Birth plane must be above the configured outlet')
        self.outlet_y = float(outlet[1])
        self.top_y = self.outlet_y + config['upstream_height'] + config['headroom']
        self.discharged = np.zeros(event.metadata['source']['maximum_particles'],dtype=bool)
        self.removed = set()
        self.log = event.fluid_directory/'upstream_removals.jsonl'
        self.removal_updates = 0
        stage=event.stage
        mat=UsdShade.Material.Define(stage,'/World/CoupledEvent/GuideMaterial')
        api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr().Set(.05)
        api.CreateDynamicFrictionAttr().Set(.05)
        api.CreateRestitutionAttr().Set(0.)
        self.panels=guide_panels(config)
        for name,centre,size in self.panels:
            cube=UsdGeom.Cube.Define(stage,'/World/CoupledEvent/InletGuide/'+name)
            cube.CreateSizeAttr().Set(1.)
            xf=UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            collision=PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateRestOffsetAttr().Set(0.)
            collision.CreateContactOffsetAttr().Set(.001)
            physicsUtils.add_physics_material_to_prim(stage,cube.GetPrim(),mat.GetPath())
            # Downstream Blender overlay creates the matching opaque shell.
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        self.refresh_metadata()

    def refresh_metadata(self):
        e=self.event
        e.metadata['conditioned_inlet'] = dict(self.config,
            removal_policy='above_top_only_before_first_outlet_discharge',
            removed_count=len(self.removed),
            removed_nominal_liters=len(self.removed)*e.configuration['source']['spacing']**3*1000,
            discharged_unique_count=int(self.discharged.sum()),
            removal_updates=self.removal_updates, removal_log=self.log.name)

    def after_substep(self, app, frame, substep):
        from pxr import Sdf, UsdPhysics, Vt
        e=self.event
        p,v,ids=e.state_arrays()
        if not len(ids): return
        mask=upstream_sink_mask(p,ids,self.discharged,self.outlet_y,self.top_y)
        removed=ids[mask]
        if len(removed):
            if self.removed.intersection(removed.tolist()):
                raise RuntimeError('Removed particle ID reappeared')
            cursor=0
            with Sdf.ChangeBlock():
                for inst in e.active_instancers:
                    count=len(inst.GetPositionsAttr().Get())
                    keep=~mask[cursor:cursor+count]
                    if not keep.all():
                        mass=UsdPhysics.MassAPI(inst.GetPrim())
                        if mass.GetMassAttr().HasAuthoredValueOpinion() or mass.GetDensityAttr().Get()!=e.configuration['source']['density']:
                            raise RuntimeError('Upstream removal requires constant density-derived particle mass')
                        inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(p[cursor:cursor+count][keep]))
                        inst.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(v[cursor:cursor+count][keep]))
                        proto=np.asarray(inst.GetProtoIndicesAttr().Get(),dtype=np.int32)
                        inst.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(proto[keep]))
                    cursor+=count
            if cursor!=len(ids): raise RuntimeError('Chunk count mismatch during removal')
            e.active_id_chunks=[chunk[~np.isin(chunk,removed)] for chunk in e.active_id_chunks]
            app.update()
            pp,vv,ii=e.state_arrays()
            if not (np.array_equal(pp,p[~mask]) and np.array_equal(vv,v[~mask]) and np.array_equal(ii,ids[~mask])):
                raise RuntimeError('Removing upstream particles altered surviving state')
            self.removed.update(removed.tolist())
            self.removal_updates+=1
            with self.log.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(frame=frame,substep=substep,removed_ids=removed.tolist(),
                                             removed_positions=p[mask].tolist(),retained_count=len(ii)))+'\n')
        self.refresh_metadata()
