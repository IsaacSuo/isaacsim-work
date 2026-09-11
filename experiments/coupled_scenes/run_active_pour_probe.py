"""Native PhysX: prefilled v3 pitcher, validated prewarm, then six-second tilt.

Exact triangle shells, kinematic pitcher; optional audited outside-basin sink.
This is a bounded local test, not yet a general active-drive runtime.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from coupled_scene.active_drive import motion_state
from coupled_scene.surface_settling import SettlingGate,prewarm_damping
from coupled_scene.water_defaults import DEFAULT_VORTICITY_CONFINEMENT
from experiments.coupled_scenes.build_cabinet_liquid_surfaces import atomic_json as _atomic_json


def atomic_json(path, data):
    """Bounded retry for Windows readers briefly locking the replace target."""
    for attempt in range(7):
        try:
            return _atomic_json(path, data)
        except PermissionError:
            if attempt==6:raise
            time.sleep(.05*2**attempt)


def official_water_settings(spacing):
    offsets=resolution_offsets(spacing)
    contact=offsets['fluid_rest']/(.99*.6)
    offsets.update(particle_contact=contact,system_contact=contact,
                   solid_rest=.99*contact)
    material=dict(cohesion=.01,damping=0.,friction=.1,surface_tension=.0074,
                  viscosity=.0000017,vorticity_confinement=0.)
    return offsets,material


def restored_vorticity(elapsed, initial=.02, target=DEFAULT_VORTICITY_CONFINEMENT):
    """One-second smooth ramp, followed by constant runtime vorticity."""
    u=max(0.,min(1.,elapsed))
    return initial+(target-initial)*(u*u*(3.-2.*u)) if u<1. else target


def resolution_offsets(spacing):
    if spacing not in (.003,.004):
        raise ValueError('Validated probe spacing must be 3 or 4 mm')
    scale=spacing/.004
    return dict(fluid_rest=.002*scale,solid_rest=(.002/.6)*scale,
                particle_contact=(.002/.6)*scale,
                system_contact=(.002/.6+.001)*scale,wall_contact=.004*scale)


def restoration_gate(minimum_seconds, relaxed=False):
    gate=SettlingGate(minimum_seconds=minimum_seconds,window_seconds=2.)
    if relaxed:
        gate.limits.update(rms_speed_m_s=.08,speed_p99_m_s=.20)
        gate.policy_id='pitcher_motion_ready_v1_relaxed_restored_speed'
    return gate


def escaped_particle_mask(positions,velocities,receiver_position,receiver_vertices):
    """Sink below the basin and outside its expanded elliptical footprint.

    Specific to the centred, axis-aligned elliptical receiver in this probe.
    No speed-magnitude criterion; rising splash and particles above the bottom
    plane remain physical. Particles below the basin footprint are NOT hidden.
    """
    lower=receiver_vertices.min(axis=0);upper=receiver_vertices.max(axis=0)
    if not np.allclose(lower[[0,2]],-upper[[0,2]],atol=1e-6):
        raise ValueError('Recycling requires a centred elliptical receiver')
    radii=upper[[0,2]]+.01
    local=positions-np.asarray(receiver_position)
    outside=np.sum((local[:,[0,2]]/radii)**2,axis=1)>1.
    return (local[:,1]<lower[1]-.02)&outside&(velocities[:,1]<0.)


def piston_prewarm_sink_mask(positions,velocities,receiver_position,receiver_vertices,donor_position,donor_vertices):
    """Sideways startup escape only; never retire under-floor or behind-plate leaks."""
    origin=np.asarray(receiver_position);local=positions-origin
    lower=receiver_vertices.min(axis=0);upper=receiver_vertices.max(axis=0)
    plate_front=float(donor_vertices[:,0].max()+np.asarray(donor_position)[0]-origin[0])
    side=(local[:,2]<lower[2]-.01)|(local[:,2]>upper[2]+.01)
    ahead=(local[:,0]>plate_front+.01)&(local[:,0]<upper[0]-.01)
    return side&ahead&(local[:,1]<lower[1]-.02)&(velocities[:,1]<0.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--official-water-probe',action='store_true',help='Stationary 10s using installed Water preset at metre units, official system offsets, no assisted damping or vorticity ramp; diagnostic only')
    parser.add_argument('--official-water-prewarm',action='store_true',help='With official-water-probe: damping 5 for 1.5s, smooth release over .5s to 0; validate only at runtime damping')
    parser.add_argument('--stirring-speed',type=float,help='Stirring-only peak angular speed override in rad/s; geometry and ramp timing unchanged')
    parser.add_argument('--stationary-resolution-probe',action='store_true',help='Eight-second diagnostic, no drive motion or readiness certification: low vorticity until 3s, ramp until 4s, then runtime value; cache 3s, 4s, and 6-8s')
    parser.add_argument('--settling-only',action='store_true',help='Observe the upright vessel for the full ten seconds; never start pouring')
    parser.add_argument('--settling-vorticity',type=float,help='Diagnostic override, allowed only with --settling-only')
    parser.add_argument('--restore-vorticity-after-settling',action='store_true',help='Settle at 0.02, ramp to runtime vorticity over 1s, require a 2s ready window before pouring')
    parser.add_argument('--relaxed-restored-readiness',action='store_true',help='Pitcher-only restoration gate: RMS <= 0.08, p99 <= 0.20 m/s; all other limits unchanged')
    parser.add_argument('--recycle-escaped',action='store_true',help='Delete descending particles below and outside the receiver; keep stable IDs and a removal audit')
    parser.add_argument('--receiver-friction',type=float,default=.05,help='Static/dynamic rigid material friction for the receiver only; donor and PBD material stay 0.05')
    parser.add_argument('--seconds',type=float,help='Action duration override; default 6s pour or full apparatus duration')
    parser.add_argument('--recycle-stirring-prewarm',action='store_true',help='Stirring only: up to 64 descending exterior particles below bowl during prewarm; never hide leakage under bowl footprint')
    parser.add_argument('--recycle-piston-prewarm',action='store_true',help='Piston only: first 2s of prewarm, at most 64 descending side escapes ahead of plate; never behind plate or under channel')
    args=parser.parse_args()
    if args.official_water_prewarm and not args.official_water_probe:
        parser.error('--official-water-prewarm requires --official-water-probe')
    if args.official_water_probe and (args.stationary_resolution_probe or args.settling_only or args.restore_vorticity_after_settling or args.relaxed_restored_readiness or args.settling_vorticity is not None or args.seconds is not None or args.stirring_speed is not None or args.recycle_escaped):
        parser.error('Official Water diagnostic has fixed material and timing; do not combine with action or other diagnostic modes')
    if args.stationary_resolution_probe and (args.settling_only or args.restore_vorticity_after_settling or args.relaxed_restored_readiness or args.settling_vorticity is not None or args.seconds is not None or args.recycle_escaped):
        parser.error('Stationary resolution probe has its own fixed schedule; do not combine with other timing/readiness modes')
    if not math.isfinite(args.receiver_friction) or args.receiver_friction<0:
        parser.error('--receiver-friction must be finite and nonnegative')
    if args.recycle_escaped and args.settling_only:
        parser.error('--recycle-escaped is a pouring option, not a settling-only option')
    if args.relaxed_restored_readiness and not args.restore_vorticity_after_settling:
        parser.error('--relaxed-restored-readiness requires --restore-vorticity-after-settling')
    if args.restore_vorticity_after_settling and (args.settling_only or args.settling_vorticity is not None):
        parser.error('--restore-vorticity-after-settling cannot be combined with settling-only overrides')
    if args.settling_vorticity is not None and (not args.settling_only or not math.isfinite(args.settling_vorticity) or args.settling_vorticity<0):
        parser.error('--settling-vorticity requires --settling-only and a finite nonnegative value')
    vorticity=DEFAULT_VORTICITY_CONFINEMENT if args.settling_vorticity is None else args.settling_vorticity
    if args.restore_vorticity_after_settling: vorticity=.02
    if args.stationary_resolution_probe: vorticity=.02
    if args.official_water_probe:vorticity=0.
    meta=json.loads((args.assets/'assets.json').read_text(encoding='utf-8'));geometry=args.assets/'geometry_and_fill.npz'
    assert hashlib.sha256(geometry.read_bytes()).hexdigest()==meta['geometry_sha256']
    apparatus=meta.get('product')=='active_apparatus_exact_mesh_assets'
    original_motion=dict(meta['case']['motion'])
    if args.stirring_speed is not None:
        if not apparatus or meta['case']['id']!='02_stirring' or args.stationary_resolution_probe or not math.isfinite(args.stirring_speed) or args.stirring_speed<=0:
            parser.error('--stirring-speed requires the stirring action case and a finite positive speed')
        meta['case']['motion']['speed_rad_s']=args.stirring_speed
    if args.stationary_resolution_probe and not apparatus:
        parser.error('Stationary resolution diagnostic is apparatus-only')
    if args.official_water_probe and not apparatus:
        parser.error('Official Water diagnostic is apparatus-only')
    if args.recycle_piston_prewarm and (not apparatus or meta['case']['id']!='03_piston_push' or args.recycle_escaped or args.recycle_stirring_prewarm):
        parser.error('Piston prewarm recycling requires the piston case and cannot combine with other sinks')
    if args.recycle_stirring_prewarm and (not apparatus or meta['case']['id']!='02_stirring' or args.recycle_escaped):
        parser.error('Prewarm recycling is restricted to the stirring bowl')
    if apparatus and args.recycle_escaped:parser.error('Pitcher recycling is not supported for other apparatus')
    arrays=np.load(geometry);count=len(arrays['positions']);hz=720
    spacing=float(meta.get('particle_spacing_m',.004));offsets=resolution_offsets(spacing)
    if args.official_water_probe:offsets,water_preset=official_water_settings(spacing)
    seconds=args.seconds if args.seconds is not None else (float(meta['case']['duration_s']) if apparatus else 6.)
    capture_start=0. if apparatus else 1.5
    if not math.isfinite(seconds) or seconds<=capture_start or abs(seconds*30-round(seconds*30))>1e-6:
        parser.error('Action duration must exceed capture start and align with 30 fps')
    expected_frames=round((seconds-capture_start)*30)+1
    report=dict(status='prepared',particle_count=count,spacing_m=spacing,hz=hz,iterations=64,
        vorticity_confinement=vorticity,requested_action_seconds=0. if args.settling_only else seconds,
        capture_start_action_seconds=capture_start,nominal_lattice_liters=meta['nominal_lattice_liters'],
        geometry_sha256=meta['geometry_sha256'],source_blend_sha256=meta['source_blend_sha256'],
        geometry_policy='Exact accepted render vessel triangles; no convex closure or synthetic jet',
        solver_checkpoint=False,rows=[])
    report['settling_only']=args.settling_only
    report['motion']=dict(meta['case']['motion'])
    report['source_asset_motion']=original_motion
    report['stirring_speed_override_rad_s']=args.stirring_speed
    report['offsets_m']=offsets
    report['stationary_resolution_probe']=args.stationary_resolution_probe
    report['official_water_probe']=args.official_water_probe
    if args.stationary_resolution_probe:
        report.update(requested_action_seconds=0.,diagnostic_only=True,physics_gate_passed=False,
            diagnostic_schedule='stationary 8s: vorticity .02 until 3s, ramp to 10 at 4s; unchanged damping; no readiness bypass claimed as success')
    report['contact_materials']=dict(donor_friction=.05,receiver_friction=args.receiver_friction,
        particle_material_friction=.05,restitution=0.,receiver_scope='entire exact receiver shell, bottom and walls',
        note='Authored coefficients; not a measured fluid-solid effective friction or isolated friction-effect test')
    if args.official_water_probe:
        report.update(requested_action_seconds=0.,diagnostic_only=True,physics_gate_passed=False,
            water_preset=water_preset,diagnostic_schedule='stationary 10s; installed Water preset constant from step 1; no high damping or vorticity switch',
            preset_scope='PBD material and official auto-derived system offsets only; existing wall geometry/material, density=1000, solver and caps retained')
        report['contact_materials']['particle_material_friction']=water_preset['friction']
        report['assisted_prewarm']=args.official_water_prewarm
        if args.official_water_prewarm:
            report['diagnostic_schedule']='stationary 10s; official Water except damping 5 for 1.5s then .5s release to 0; vorticity always 0; validate only after full release'
    report['recycling']=dict(enabled=args.recycle_escaped,removed_count=0,updates=0,
        policy='descending, below receiver bottom by 2cm, outside elliptical footprint expanded by 1cm',
        audit_file='escaped_removals.jsonl' if args.recycle_escaped else None,
        no_replacement_emission=True)
    if args.recycle_stirring_prewarm:
        report['recycling'].update(enabled=True,policy='prewarm only; descending below bowl bottom by 2cm and outside radius plus 1cm; at most 64 total',
            maximum_removed_count=64,audit_file='escaped_removals.jsonl')
    if args.recycle_piston_prewarm:
        report['recycling'].update(enabled=True,policy='first 2s prewarm only; descending below channel bottom by 2cm, beyond side walls by 1cm, ahead of plate front by 1cm; at most 64 total',
            maximum_removed_count=64,cutoff_simulated_seconds=2.,audit_file='escaped_removals.jsonl',
            excluded='behind or within piston, under channel footprint, beyond far end, action phase')
    live_ids=np.arange(count,dtype=np.int64)
    report['restore_vorticity_after_settling']=args.restore_vorticity_after_settling
    if args.restore_vorticity_after_settling:
        policy_gate=restoration_gate(0.,args.relaxed_restored_readiness)
        report['restoration_policy']=dict(initial=.02,target=DEFAULT_VORTICITY_CONFINEMENT,ramp_seconds=1.,validation_seconds=2.,
            unchanged_thresholds=not args.relaxed_restored_readiness,gate_policy=policy_gate.policy_id,limits=dict(policy_gate.limits),
            interpretation='Motion-ready with accepted small-scale particle motion; not strict still water' if args.relaxed_restored_readiness else 'Original still-water thresholds')
    if args.dry_run: print(json.dumps(report,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'capture').mkdir()
    def save():atomic_json(args.output/'probe_report.json',report)
    save()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':64,'height':64})
    attached=False;frames=[]
    try:
        import carb
        import omni.usd
        import omni.physx.bindings._physx as bindings
        from omni.physx import get_physx_simulation_interface,get_physx_interface
        from omni.physx.scripts import particleUtils,physicsUtils
        from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,UsdUtils,Sdf,Vt,Gf
        for setting in (bindings.SETTING_UPDATE_TO_USD,bindings.SETTING_UPDATE_PARTICLES_TO_USD,
                        bindings.SETTING_UPDATE_VELOCITIES_TO_USD,bindings.SETTING_ENABLE_PARTICLE_AUTHORING):
            carb.settings.get_settings().set(setting,True)
        context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
        UsdGeom.SetStageUpAxis(stage,'Y');UsdGeom.SetStageMetersPerUnit(stage,1.)
        scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,-1,0));scene.CreateGravityMagnitudeAttr().Set(9.81)
        api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());api.CreateEnableGPUDynamicsAttr().Set(True)
        api.CreateBroadphaseTypeAttr().Set('GPU');api.CreateSolverTypeAttr().Set('TGS');api.CreateTimeStepsPerSecondAttr().Set(hz)
        api.CreateMaxBiasCoefficientAttr().Set(240.);api.CreateEnableExternalForcesEveryIterationAttr().Set(True)
        api.CreateGpuMaxParticleContactsAttr().Set(2097152);api.CreateGpuCollisionStackSizeAttr().Set(536870912)
        api.CreateGpuMaxDeformableVolumeContactsAttr().Set(1024);api.CreateGpuMaxDeformableSurfaceContactsAttr().Set(1024)
        mat=UsdShade.Material.Define(stage,'/World/WallMaterial');wall=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        wall.CreateStaticFrictionAttr().Set(.05);wall.CreateDynamicFrictionAttr().Set(.05);wall.CreateRestitutionAttr().Set(0.)
        receiver_mat=UsdShade.Material.Define(stage,'/World/ReceiverMaterial')
        receiver_wall=UsdPhysics.MaterialAPI.Apply(receiver_mat.GetPrim())
        receiver_wall.CreateStaticFrictionAttr().Set(args.receiver_friction)
        receiver_wall.CreateDynamicFrictionAttr().Set(args.receiver_friction)
        receiver_wall.CreateRestitutionAttr().Set(0.)
        pivot=np.array(meta['donor']['position_m']);pitcher_floor=np.array(meta['diagnostic_origin_m']) if apparatus else pivot+np.array(meta['fill_cavity_axis_local_m'])
        for key in ('donor','receiver'):
            body=UsdGeom.Xform.Define(stage,'/World/'+key)
            op=body.AddTransformOp();op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*meta[key]['position_m'])))
            if key=='donor':
                rigid=UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());rigid.CreateRigidBodyEnabledAttr().Set(True);rigid.CreateKinematicEnabledAttr().Set(True)
                drive=op
            mesh=UsdGeom.Mesh.Define(stage,'/World/'+key+'/Shell')
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(arrays[key+'_vertices']))
            triangles=arrays[key+'_triangles'];mesh.CreateFaceVertexCountsAttr([3]*len(triangles));mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr('none');UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set('none')
            contact=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim());contact.CreateContactOffsetAttr().Set(offsets['wall_contact']);contact.CreateRestOffsetAttr().Set(0.)
            bound_mat=receiver_mat if key=='receiver' else mat
            physicsUtils.add_physics_material_to_prim(stage,mesh.GetPrim(),bound_mat.GetPath())
            assert UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial('physics')[0].GetPath()==bound_mat.GetPath()
        fluid_rest=offsets['fluid_rest'];solid_rest=offsets['solid_rest'];system_path=Sdf.Path('/World/ParticleSystem')
        particleUtils.add_physx_particle_system(stage,system_path,simulation_owner=scene.GetPath(),
            contact_offset=offsets['system_contact'],rest_offset=solid_rest,particle_contact_offset=offsets['particle_contact'],
            solid_rest_offset=solid_rest,fluid_rest_offset=fluid_rest,enable_ccd=True,
            solver_position_iterations=64,max_depenetration_velocity=.25,max_neighborhood=96,neighborhood_scale=1.01,max_velocity=4.2)
        water_path=Sdf.Path('/World/WaterMaterial')
        particleUtils.add_pbd_particle_material(stage,water_path,density=1000.,friction=.05,damping=5.,
            viscosity=.002,vorticity_confinement=vorticity,surface_tension=.0074,cohesion=.01,adhesion=0.,cfl_coefficient=1.)
        if args.official_water_probe:
            particleUtils.AddPBDMaterialWater(stage.GetPrimAtPath(water_path))
            attr_names=dict(cohesion='cohesion',damping='damping',friction='friction',surface_tension='surfaceTension',viscosity='viscosity',vorticity_confinement='vorticityConfinement')
            authored={k:float(stage.GetPrimAtPath(water_path).GetAttribute('physxPBDMaterial:'+n).Get()) for k,n in attr_names.items()}
            for k,value in water_preset.items():
                assert math.isclose(authored[k],value,rel_tol=1e-6,abs_tol=1e-12),(k,authored[k],value)
            import inspect
            preset_source=Path(inspect.getsourcefile(particleUtils.AddPBDMaterialWater))
            report['installed_water_preset']=dict(source=str(preset_source),sha256=hashlib.sha256(preset_source.read_bytes()).hexdigest(),
                function_source=inspect.getsource(particleUtils.AddPBDMaterialWater),meters_per_unit=1.,authored=authored)
        physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system_path),water_path)
        damping=stage.GetPrimAtPath(water_path).GetAttribute('physxPBDMaterial:damping')
        vortex_attr=stage.GetPrimAtPath(water_path).GetAttribute('physxPBDMaterial:vorticityConfinement')
        report['authored_vorticity']=float(vortex_attr.Get())
        prim=particleUtils.add_physx_particleset_pointinstancer(stage,Sdf.Path('/World/Water'),
            Vt.Vec3fArray.FromNumpy(arrays['positions']),Vt.Vec3fArray.FromNumpy(arrays['velocities']),system_path,
            self_collision=True,fluid=True,particle_group=0,particle_mass=0.,density=1000.)
        mass=UsdPhysics.MassAPI(prim);mass.GetMassAttr().Clear();mass.GetDensityAttr().Set(1000.)
        UsdGeom.Imageable(prim).MakeInvisible();instancer=UsdGeom.PointInstancer(prim)
        report['buffers']=dict(collision_stack_bytes=536870912,particle_contacts=2097152,no_softbodies=True)
        report['offsets_m']=offsets
        report['authored_receiver_friction']=dict(static=float(receiver_wall.GetStaticFrictionAttr().Get()),
            dynamic=float(receiver_wall.GetDynamicFrictionAttr().Get()),material_path=str(receiver_mat.GetPath()))
        app.update();simulation=get_physx_simulation_interface();cache=UsdUtils.StageCache.Get();sid=cache.GetId(stage)
        if not sid.IsValid():sid=cache.Insert(stage)
        simulation.attach_stage(sid.ToLongInt());attached=True
        gate=SettlingGate(minimum_seconds=3.);ready_step=None;started=time.perf_counter();total_limit=math.ceil(14+seconds)*hz
        restore_step=None;restore_gate=None
        report['status']='settling';save()
        for step in range(1,total_limit+1):
            if time.perf_counter()-started>5400: raise RuntimeError('Wall-time limit reached; partial cache retained')
            action_time=(step-ready_step)/hz if ready_step is not None else 0.
            motion=motion_state(meta['case']['motion'],action_time);angle=motion['angle_rad']
            axis=np.zeros(3);axis[motion['rotation_axis']]=1.
            target_position=pivot+np.asarray(motion['displacement_m'])
            transform=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*axis),math.degrees(angle)))
            transform.SetTranslateOnly(Gf.Vec3d(*target_position));drive.Set(transform)
            coefficient,phase=prewarm_damping((step-1)/hz,5.) if ready_step is None else (.01,'recording')
            if restore_step is not None:
                vortex_attr.Set(restored_vorticity((step-restore_step)/hz))
                if ready_step is None:
                    phase='vorticity_ramp' if step-restore_step<hz else 'restored_validation'
            if args.stationary_resolution_probe:
                vortex_attr.Set(restored_vorticity(step/hz-3.))
                phase='diagnostic_low_vorticity' if step<=3*hz else ('diagnostic_ramp' if step<4*hz else 'diagnostic_runtime_vorticity')
            if args.official_water_probe:
                coefficient,phase=prewarm_damping((step-1)/hz,5.,runtime_damping=0.) if args.official_water_prewarm else (0.,'diagnostic_official_water')
                assert float(vortex_attr.Get())==0.
            damping.Set(coefficient)
            simulation.simulate(1/hz,(step-1)/hz);simulation.fetch_results();app.update()
            capture_due=ready_step is not None and step-ready_step>=round(capture_start*hz) and (step-ready_step)%24==0
            if args.stationary_resolution_probe:
                capture_due=step in (3*hz,4*hz) or (6*hz<=step<=8*hz and step%24==0)
            if args.official_water_probe:
                capture_due=step in (hz,2*hz,3*hz,4*hz,5*hz) or (6*hz<=step<=10*hz and step%24==0)
            if step==1 or step%72==0 or capture_due:
                p=np.asarray(instancer.GetPositionsAttr().Get(),dtype=np.float32)
                v=np.asarray(instancer.GetVelocitiesAttr().Get(),dtype=np.float32)
                assert p.shape==(len(live_ids),3) and v.shape==p.shape and np.isfinite(p).all() and np.isfinite(v).all()
                piston_sink_due=args.recycle_piston_prewarm and ready_step is None and step<=2*hz
                if (args.recycle_escaped and ready_step is not None) or (args.recycle_stirring_prewarm and ready_step is None) or piston_sink_due:
                    if piston_sink_due:
                        remove=piston_prewarm_sink_mask(p,v,meta['receiver']['position_m'],arrays['receiver_vertices'],meta['donor']['position_m'],arrays['donor_vertices'])
                    else:
                        remove=escaped_particle_mask(p,v,meta['receiver']['position_m'],arrays['receiver_vertices'])
                    if (args.recycle_stirring_prewarm or args.recycle_piston_prewarm) and report['recycling']['removed_count']+int(remove.sum())>64:
                        raise RuntimeError('Prewarm escape exceeds 64-particle allowance; not masking a leak')
                    if remove.any():
                        keep=~remove
                        if not keep.any(): raise RuntimeError('All particles escaped; refusing an empty-water result')
                        removed_ids=live_ids[remove].copy()
                        removal=dict(step=step,simulated_seconds=step/hz,action_seconds=action_time,
                            removed_ids=removed_ids.tolist(),positions=p[remove].tolist(),velocities=v[remove].tolist(),
                            retained_count=int(keep.sum()))
                        assert not mass.GetMassAttr().HasAuthoredValueOpinion() and mass.GetDensityAttr().Get()==1000.
                        proto=np.asarray(instancer.GetProtoIndicesAttr().Get(),dtype=np.int32)
                        assert proto.shape==(len(live_ids),)
                        with Sdf.ChangeBlock():
                            instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(p[keep]))
                            instancer.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(v[keep]))
                            instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(proto[keep]))
                        app.update()
                        pp=np.asarray(instancer.GetPositionsAttr().Get(),dtype=np.float32)
                        vv=np.asarray(instancer.GetVelocitiesAttr().Get(),dtype=np.float32)
                        assert np.array_equal(pp,p[keep]) and np.array_equal(vv,v[keep]),'Recycling altered surviving state'
                        p,v=pp,vv;live_ids=live_ids[keep]
                        report['recycling']['removed_count']+=len(removed_ids)
                        report['recycling']['updates']+=1
                        assert len(live_ids)+report['recycling']['removed_count']==count
                        with (args.output/'escaped_removals.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(removal)+'\n')
                report['active_particle_count']=len(live_ids)
                speed=np.linalg.norm(v,axis=1);cap=float(np.mean(speed>=4.2*.999))
                if cap>.005:raise RuntimeError('More than 0.5% of particles reached native speed ceiling')
                actual=get_physx_interface().get_rigidbody_transformation('/World/donor')
                assert actual['ret_val'],actual
                q=np.asarray(actual['rotation']);expected=np.r_[axis*math.sin(angle/2),math.cos(angle/2)]
                assert np.linalg.norm(np.asarray(actual['position'])-target_position)<1e-4,actual
                assert abs(float(q@expected))>1-1e-5,(actual,expected.tolist())
                local=p-pitcher_floor;levels=[]
                for positive_x,positive_z in ((False,False),(False,True),(True,False),(True,True)):
                    mask=((local[:,0]>=0)==positive_x)&((local[:,2]>=0)==positive_z)
                    levels.append(float(np.quantile(local[mask,1],.99)) if mask.any() else None)
                outside=(np.hypot(local[:,0],local[:,2])>.155)|(local[:,1]<.006)
                over=local[:,1]>.30
                if apparatus:
                    bounds=np.asarray(meta['containment_bounds_local_m'])
                    outside=(local[:,1]<bounds[0,1]-.004)|np.any(local[:,[0,2]]<bounds[0,[0,2]]-.008,axis=1)|np.any(local[:,[0,2]]>bounds[1,[0,2]]+.008,axis=1)
                    over=local[:,1]>bounds[1,1]+.008
                row=dict(step=step,simulated_seconds=step/hz,action_seconds=action_time if ready_step is not None else None,
                    wall_seconds=time.perf_counter()-started,phase=phase,normal_damping=coefficient,
                    vorticity_confinement=float(vortex_attr.Get()),
                    rms_speed_m_s=float(np.sqrt(np.mean(speed**2))),speed_p99_m_s=float(np.quantile(speed,.99)),
                    max_speed_m_s=float(speed.max()),speed_cap_fraction=cap,
                    outside_side_or_floor_count=int(outside.sum()),over_rim_particle_count=int(over.sum()),
                    regional_level_p99_m=levels,position_bounds_m=[p.min(axis=0).tolist(),p.max(axis=0).tolist()],
                    particle_count=len(p),recycled_count=report['recycling']['removed_count'],native_donor_position_m=list(actual['position']),native_donor_rotation_xyzw=list(actual['rotation']))
                report['rows'].append(row)
                report['authored_vorticity']=row['vorticity_confinement']
                report['last_simulated_seconds']=step/hz;report['last_action_seconds']=action_time
                report['prewarm_over_rim_max']=max(report.get('prewarm_over_rim_max',0),int(over.sum()) if ready_step is None else 0)
                if args.stationary_resolution_probe or args.official_water_probe:
                    row['strict_still_water_passed']=coefficient==(0. if args.official_water_probe else .01) and gate.observe(row)
                    if capture_due:
                        filename=f'frame_{len(frames):04d}.npz'
                        np.savez(args.output/'capture'/filename,positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                        frames.append(dict(file=filename,simulated_seconds=step/hz,particle_count=len(live_ids)))
                        report['recorded_frames']=len(frames)
                    if step>=(10 if args.official_water_probe else 8)*hz:
                        report.update(status='completed',simulated_action_seconds=0.,loop_wall_seconds=time.perf_counter()-started,
                                      final_strict_still_water_passed=row['strict_still_water_passed'])
                        atomic_json(args.output/'capture/manifest.json',dict(complete=True,diagnostic_only=True,physics_gate_passed=False,
                            frames=frames,source_assets=meta,note='Irregular diagnostic samples; not an action/video-ready cache'))
                        save();return
                    save();continue
                if ready_step is None:
                    # A transient overshoot during initialization is not a ready
                    # state, but can settle back without deleting any particles.
                    # Both readiness policies reject every contaminated window.
                    if restore_step is not None:
                        is_ready=False
                        if step-restore_step>=hz:
                            assert row['vorticity_confinement']==DEFAULT_VORTICITY_CONFINEMENT and coefficient==.01
                            is_ready=restore_gate.observe(row)
                        row['restored_readiness_passed']=is_ready
                        if step-restore_step>=3*hz:
                            report['restoration_passed']=is_ready
                            np.savez(args.output/'restored_state.npz',positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                            if not is_ready:
                                raise RuntimeError('Restored vorticity failed the selected thresholds over the two-second validation window; pouring not started')
                            ready_step=step;report.update(status='recording',settled_at_seconds=step/hz,settled_level_m=levels)
                            np.savez(args.output/'settled_state.npz',positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                    elif args.restore_vorticity_after_settling:
                        is_ready=coefficient==.01 and gate.observe(row)
                        row['low_vorticity_readiness_passed']=is_ready
                        if is_ready:
                            restore_step=step
                            restore_gate=restoration_gate(step/hz+3.,args.relaxed_restored_readiness)
                            report.update(status='restoring_vorticity',low_vorticity_ready_at_seconds=step/hz)
                            np.savez(args.output/'low_vorticity_settled_state.npz',positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                        elif step>=10*hz:
                            raise RuntimeError('Low-vorticity water did not settle within ten seconds')
                    elif args.settling_only:
                        is_ready=coefficient==.01 and gate.observe(row)
                        row['readiness_passed']=is_ready
                        if is_ready and 'first_ready_at_seconds' not in report:
                            report['first_ready_at_seconds']=step/hz
                        report['last_simulated_seconds']=step/hz
                        report['last_action_seconds']=0.
                        if step>=10*hz:
                            np.savez(args.output/'settling_final_state.npz',positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                            report['final_readiness_passed']=is_ready
                            if not is_ready:
                                raise RuntimeError('Ten-second settling-only control did not pass unchanged readiness thresholds')
                            report.update(status='completed',simulated_action_seconds=0.,loop_wall_seconds=time.perf_counter()-started)
                            save()
                            return
                    elif coefficient==.01 and gate.observe(row):
                        ready_step=step;report.update(status='recording',settled_at_seconds=step/hz,settled_level_m=levels)
                        np.savez(args.output/'settled_state.npz',positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                    elif step>=10*hz:raise RuntimeError('Upright water did not pass existing readiness thresholds in 10 seconds')
                elif capture_due:
                    filename=f'frame_{len(frames):04d}.npz'
                    np.savez(args.output/'capture'/filename,positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                    frames.append(dict(file=filename,recording_seconds=action_time-capture_start,action_seconds=action_time,
                        particle_count=len(live_ids),recycled_count=report['recycling']['removed_count'],
                        native_position_m=list(actual['position']),native_rotation_xyzw=list(actual['rotation'])))
                    report['recorded_frames']=len(frames)
                if ready_step==step and capture_start==0.:
                    filename='frame_0000.npz'
                    np.savez(args.output/'capture'/filename,positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
                    frames.append(dict(file=filename,recording_seconds=0.,action_seconds=0.,particle_count=len(live_ids),
                        recycled_count=report['recycling']['removed_count'],native_position_m=list(actual['position']),native_rotation_xyzw=list(actual['rotation'])))
                    report['recorded_frames']=len(frames)
                report['last_simulated_seconds']=step/hz;report['last_action_seconds']=action_time;save()
            if ready_step is not None and step-ready_step>=round(seconds*hz):break
        assert ready_step is not None and len(frames)==expected_frames,len(frames)
        report.update(status='completed',simulated_action_seconds=seconds,loop_wall_seconds=time.perf_counter()-started)
        stage.GetRootLayer().Export(str(args.output/'final_stage.usda'))
        atomic_json(args.output/'capture/manifest.json',dict(complete=True,fps=30,frames=frames,source_assets=meta,
            moving_object=meta.get('moving_object','PouringPitcher'),
            recycling=dict(report['recycling']),initial_particle_count=count,active_particle_count=len(live_ids),
            note='Native arrays; explicit stable ids, no interpolation or replacement emission. Optional audited escaped-particle deletion. Initial settling excluded.'))
        save()
    except BaseException as exc:
        if 'p' in locals() and 'v' in locals():
            np.savez(args.output/'failed_state.npz',positions=p,velocities=v,ids=live_ids,simulated_seconds=step/hz)
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise
    finally:
        if attached:simulation.detach_stage()
        app.close()


if __name__=='__main__':main()
