"""Bounded native PhysX still-water capacity probe; no softbodies or emitter.

Run with Isaac Sim python.bat. The short startup test is not a settled-water
production cache. It uses the authored layout colliders and original water size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from coupled_scene.water_defaults import surface_vorticity, LEGACY_DECAY_VORTICITY_CONFINEMENT


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=.004)
    parser.add_argument("--hz", type=int, default=720)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=.25)
    parser.add_argument("--wall-limit", type=float, default=180)
    parser.add_argument("--settle", action="store_true", help="Prewarm to a sustained low-motion gate before recording")
    parser.add_argument("--settle-min-seconds", type=float, default=2.)
    parser.add_argument("--settle-max-seconds", type=float, default=8.)
    parser.add_argument("--prewarm-damping", type=float, help="Optional off-camera damping, fully restored before readiness testing")
    parser.add_argument("--damping-hold-seconds", type=float, default=1.5)
    parser.add_argument("--damping-release-seconds", type=float, default=.5)
    parser.add_argument("--normal-validation-seconds", type=float, default=1.)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--initial-state", type=Path, help="Restore native positions AND velocities; revalidate under normal damping")
    parser.add_argument("--action-case",action='store_true',help="Run the layout's native kinematic action after prewarm")
    parser.add_argument('--decay-test',action='store_true',help='Diagnostic only: restore stopped slosh, freeze tank, vary only iterations')
    parser.add_argument('--initial-report',type=Path,help='Explicit source report for decay snapshot inside settled_capture')
    parser.add_argument('--decay-viscosity',type=float,help='Isolated 64-iteration decay diagnostic viscosity override; production defaults unchanged')
    parser.add_argument('--decay-timestep-test',action='store_true',help='Isolated decay frequency test: 360/720/1440 Hz, 64 iterations, original material')
    parser.add_argument('--decay-friction-test',action='store_true',help='Isolated decay test: zero particle and wall contact friction, retain collisions')
    parser.add_argument('--decay-vorticity',type=float,help='Isolated decay vorticity confinement override; all other baseline settings retained')
    parser.add_argument('--decay-capture-fps',type=int,choices=(2,30),default=2,help='Decay snapshot output cadence only; physics timestep unchanged')
    args = parser.parse_args()
    if args.decay_capture_fps!=2 and not args.decay_test:
        raise ValueError('High-rate diagnostic capture requires decay-test mode')
    if not all(math.isfinite(v) and v > 0 for v in (args.spacing, args.hz, args.iterations, args.seconds, args.wall_limit)):
        raise ValueError("Probe arguments must be finite and positive")
    if args.seconds > (4 if args.decay_test else 15 if args.action_case else 2) or args.wall_limit > (5400 if args.decay_test else 10800 if args.action_case else 5400 if args.settle else 600):
        raise ValueError("Capture limited to 2 seconds; wall limit at most 5400 with settling, otherwise 600")
    if not (math.isfinite(args.settle_min_seconds) and math.isfinite(args.settle_max_seconds)
            and 0 <= args.settle_min_seconds < args.settle_max_seconds <= 30):
        raise ValueError("Settling requires 0 <= minimum < maximum <= 30 seconds")
    if args.settle and args.hz % 30:
        raise ValueError("Settled capture requires a timestep divisible by 30 Hz")
    if args.prewarm_damping is not None:
        if not args.settle or not math.isfinite(args.prewarm_damping) or args.prewarm_damping < .01:
            raise ValueError("Assisted damping requires --settle and damping >= 0.01")
        if not all(math.isfinite(v) and v > 0 for v in (args.damping_hold_seconds,args.damping_release_seconds,args.normal_validation_seconds)):
            raise ValueError("Assisted phase durations must be positive")
        if args.damping_hold_seconds+args.damping_release_seconds+args.normal_validation_seconds >= args.settle_max_seconds:
            raise ValueError("Settling limit must leave time for normal-parameter validation")
    if args.decay_timestep_test and (not args.decay_test or args.hz not in (360,720,1440) or args.iterations!=64 or args.decay_viscosity is not None):
        raise ValueError('Timestep test requires decay mode, 360/720/1440 Hz, 64 iterations and unchanged viscosity')
    if args.decay_test and (not args.initial_state or not args.initial_report or args.settle or args.action_case or args.prewarm_damping is not None or (args.hz != 720 and not args.decay_timestep_test) or args.iterations not in (16,32,64)):
        raise ValueError('Decay test requires source arrays/report, 720 Hz and 16/32/64 iterations, no action or prewarm')
    if args.decay_viscosity is not None and (not args.decay_test or args.iterations!=64 or not math.isfinite(args.decay_viscosity) or args.decay_viscosity<0):
        raise ValueError('Viscosity override requires the 64-iteration decay diagnostic and a finite nonnegative value')
    if args.decay_friction_test and (not args.decay_test or args.hz!=720 or args.iterations!=64 or args.decay_viscosity is not None or args.decay_timestep_test):
        raise ValueError('Friction test requires 720 Hz, 64 iterations, original viscosity, and no timestep test')
    if args.decay_vorticity is not None and (not args.decay_test or args.hz!=720 or args.iterations!=64 or args.decay_viscosity is not None or args.decay_timestep_test or args.decay_friction_test or not math.isfinite(args.decay_vorticity) or args.decay_vorticity<0):
        raise ValueError('Vorticity test requires 720 Hz, 64 iterations, no other overrides, finite nonnegative vorticity')
    viscosity=.002 if args.decay_viscosity is None else args.decay_viscosity
    friction=0. if args.decay_friction_test else .05
    vorticity=surface_vorticity(decay_test=args.decay_test, decay_override=args.decay_vorticity)
    if args.initial_report and not args.decay_test:
        raise ValueError('--initial-report is restricted to the explicit decay diagnostic')
    if args.initial_state and not args.decay_test and (not args.settle or (args.prewarm_damping is not None and not args.action_case)):
        raise ValueError("Snapshot restore requires --settle without assisted damping")
    spec = json.loads((args.layout / "scene_spec.json").read_text(encoding="utf-8"))
    if spec["case"]["id"] != "01_still_water" and not (args.action_case or args.decay_test):
        raise ValueError("Only the static tank is supported by this probe")
    if args.action_case and (not spec['case']['motion'] or not args.initial_state or not args.settle):
        raise ValueError('Action requires a moving layout, restored water, and prewarm validation')
    tank = spec["tank"]
    origin = spec["tank_origin_isaac"]
    x, height, z = tank["inner_size_m"]
    depth = tank["reference_water_depth_m"]
    # Preserve the original 4 mm lattice. Leave the same 8 mm side clearance
    # as the existing pool-drop initializer, avoiding initial wall penetration.
    clearance = max(.008, 2 * args.spacing)
    counts = [math.floor((x-2*clearance)/args.spacing+1e-8),
              math.floor((depth-.5*args.spacing-clearance)/args.spacing+1e-8)+1,
              math.floor((z-2*clearance)/args.spacing+1e-8)]
    if min(counts) < 1:
        raise ValueError("Water volume smaller than particle boundary clearance")
    report = dict(status="prepared", product="surface_study_capacity_probe",
                  production_cache=False, settled_water=False,
                  layout=str(args.layout.resolve()), spacing_m=args.spacing,
                  lattice_counts=counts, particle_count=math.prod(counts),
                  tank_size_m=[x,height,z], reference_depth_m=depth,
                  bottom_particle_centre_clearance_m=clearance,
                  nominal_lattice_volume_l=math.prod(counts)*args.spacing**3*1000,
                  hz=args.hz, iterations=args.iterations, requested_seconds=args.seconds,
                  wall_limit_seconds=args.wall_limit,
                  vorticity_confinement=vorticity,
                  baseline="v74 offsets and other material values; vorticity recorded explicitly; no deformable or emitter",
                  rows=[])
    from coupled_scene.surface_settling import SettlingGate, prewarm_damping
    restored_at = args.damping_hold_seconds+args.damping_release_seconds if args.prewarm_damping is not None else 0.
    gate = SettlingGate(max(args.settle_min_seconds,restored_at+args.normal_validation_seconds) if args.prewarm_damping is not None else args.settle_min_seconds)
    report["settling"] = dict(enabled=args.settle, minimum_seconds=args.settle_min_seconds,
        maximum_seconds=args.settle_max_seconds, window_seconds=gate.window_seconds,
        limits=gate.limits, policy_id=gate.policy_id, runtime_damping=.01,
        assisted_damping=args.prewarm_damping, restored_at_seconds=restored_at,
        normal_validation_seconds=args.normal_validation_seconds if args.prewarm_damping is not None else None,
        readiness_policy="project-local numerical thresholds, not a universal physical standard")
    restored_arrays = None
    action_plan=None
    if args.initial_state:
        import numpy as np
        source_report_path=args.initial_report or args.initial_state.parent/'probe_report.json'
        source_report=json.loads(source_report_path.read_text(encoding='utf-8'))
        source_vorticity=source_report.get('vorticity_confinement', LEGACY_DECAY_VORTICITY_CONFINEMENT)
        report['restored_material_transition']=dict(source_vorticity=source_vorticity,
            target_vorticity=vorticity, changed=source_vorticity!=vorticity,
            revalidate_before_recording=bool(args.settle))
        if args.decay_test and source_vorticity!=LEGACY_DECAY_VORTICITY_CONFINEMENT:
            raise ValueError('Historical decay diagnostics require the original 0.02 source material')
        for key in ('spacing_m','hz','iterations','particle_count','tank_size_m','reference_depth_m','native_offsets_m','gpu_buffers'):
            if key in report and source_report[key] != report[key] and not (args.decay_test and key=='iterations') and not (args.decay_timestep_test and key=='hz'):
                raise ValueError(f"Snapshot configuration mismatch: {key}")
        source_spec=json.loads((Path(source_report['layout'])/'scene_spec.json').read_text(encoding='utf-8'))
        if source_spec['tank_origin_isaac'] != origin or source_spec['tank'] != tank:
            raise ValueError("Snapshot tank placement differs; no translation or scaling allowed")
        if args.decay_test:
            if spec['case']['id']!='02_sloshing' or spec['case']!=source_spec['case'] or source_report['status']!='completed_settled_capture':
                raise ValueError('Decay source must be the completed matching slosh case')
            capture_path=source_report_path.parent/'settled_capture'
            source_capture=json.loads((capture_path/'manifest.json').read_text())
            entries=[f for f in source_capture['frames'] if (capture_path/f['file']).resolve()==args.initial_state.resolve()]
            if not source_capture['complete'] or len(entries)!=1:
                raise ValueError('Snapshot not registered in complete source capture')
            entry=entries[0]
            if abs(entry['recording_seconds']-spec['case']['motion']['stop_s'])>1e-9 or entry['target_pose_isaac_m']!=origin:
                raise ValueError('Decay must start exactly at stopped tank pose/time')
            report.update(product='surface_study_iteration_decay_diagnostic',
                decay=dict(source_recording_seconds=entry['recording_seconds'],source_iterations=source_report['iterations'],
                           frozen_tank=True,prewarm=False,only_parameter_changed='iterations',
                           metric_hz=10,checkpoint_hz=args.decay_capture_fps))
            if args.decay_viscosity is not None:
                report['product']='surface_study_viscosity_decay_diagnostic'
                report['decay'].update(only_parameter_changed='viscosity',baseline_viscosity=.002,viscosity=viscosity)
            if args.decay_timestep_test:
                report['product']='surface_study_timestep_decay_diagnostic'
                report['decay'].update(only_parameter_changed='hz',baseline_hz=source_report['hz'],
                                       hz=args.hz,dt_seconds=1/args.hz,viscosity=.002)
            if args.decay_friction_test:
                report['product']='surface_study_friction_decay_diagnostic'
                report['decay'].update(only_parameter_changed='contact_friction',baseline_friction=.05,
                    particle_friction=0.,wall_static_friction=0.,wall_dynamic_friction=0.,collisions_retained=True)
            if args.decay_vorticity is not None:
                report['product']='surface_study_vorticity_decay_diagnostic'
                report['decay'].update(only_parameter_changed='vorticity_confinement',baseline_vorticity=.02,
                    vorticity=vorticity,note='Vorticity compensation adds motion; persistent agitation is not proof of accurate wave preservation.')
        with np.load(args.initial_state,allow_pickle=False) as data:
            p=data['positions'].copy();v=data['velocities'].copy();source_seconds=float(data['simulated_seconds'])
        if p.shape != (report['particle_count'],3) or v.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError("Invalid snapshot arrays")
        restored_arrays=(p,v)
        if args.action_case:
            from coupled_scene.surface_actions import plan_action,action_pose
            action_plan=plan_action(spec,float(np.quantile(p[:,1]-origin[1],.99)))
            report['action']=action_plan
        with args.initial_state.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        report['initial_state']=dict(path=str(args.initial_state.resolve()),sha256=digest,
            source_seconds=source_seconds,positions_and_velocities_preserved=True,
            exact_solver_checkpoint=False,note="Native arrays restored; solver internals rebuilt. Revalidate before recording.")
        if args.decay_test:
            from coupled_scene.surface_decay import wave_metrics
            report['initial_state']['note']='Shared native arrays; solver internals rebuilt equally for all arms, including the 64-iteration control. No settling.'
            report['decay']['initial_metrics']=wave_metrics(p,v,origin,[x,height,z])
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "probe_report.json"
    write_report(report_path, report)
    print('[probe-plan] ' + json.dumps(report), flush=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 64, "height": 64})
    simulation = None
    attached = False
    try:
        import numpy as np
        import carb
        import omni.usd
        import omni.physx.bindings._physx as bindings
        from omni.physx import get_physx_simulation_interface
        from omni.physx.scripts import particleUtils, physicsUtils
        from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdUtils, UsdShade, Vt

        settings = carb.settings.get_settings()
        for name in (bindings.SETTING_UPDATE_TO_USD, bindings.SETTING_UPDATE_PARTICLES_TO_USD,
                     bindings.SETTING_UPDATE_VELOCITIES_TO_USD, bindings.SETTING_ENABLE_PARTICLE_AUTHORING):
            settings.set(name, True)
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, "Y")
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        stage.GetRootLayer().subLayerPaths.append(str((args.layout / "colliders.usda").resolve()))
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,-1,0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        api.CreateEnableGPUDynamicsAttr().Set(True)
        api.CreateBroadphaseTypeAttr().Set("GPU")
        api.CreateSolverTypeAttr().Set("TGS")
        api.CreateTimeStepsPerSecondAttr().Set(args.hz)
        if args.decay_test:
            report['decay']['authored_hz']=int(api.GetTimeStepsPerSecondAttr().Get())
            assert report['decay']['authored_hz']==args.hz
        api.CreateMaxBiasCoefficientAttr().Set(240.)
        api.CreateEnableExternalForcesEveryIterationAttr().Set(True)
        api.CreateGpuMaxParticleContactsAttr().Set(2097152)
        api.CreateGpuCollisionStackSizeAttr().Set(536870912)
        # No deformables exist. Do not allocate the server's large softbody pools.
        api.CreateGpuMaxDeformableVolumeContactsAttr().Set(1024)
        api.CreateGpuMaxDeformableSurfaceContactsAttr().Set(1024)
        wall_material = UsdShade.Material.Define(stage, "/World/WallMaterial")
        wall_api = UsdPhysics.MaterialAPI.Apply(wall_material.GetPrim())
        wall_api.CreateStaticFrictionAttr().Set(friction)
        wall_api.CreateDynamicFrictionAttr().Set(friction)
        wall_api.CreateRestitutionAttr().Set(0.)
        if args.decay_test:
            report['decay']['authored_wall_material']=dict(static_friction=float(wall_api.GetStaticFrictionAttr().Get()),
                dynamic_friction=float(wall_api.GetDynamicFrictionAttr().Get()),restitution=float(wall_api.GetRestitutionAttr().Get()))
        collider_count = 0
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collider_count += 1
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr().Set(.004)
                collision.CreateRestOffsetAttr().Set(0.)
                physicsUtils.add_physics_material_to_prim(stage, prim, wall_material.GetPath())
        assert collider_count == (6 if args.action_case and spec['case']['actuator'] else 5)
        motion_xform=None
        if args.decay_test:
            frozen_target=stage.GetPrimAtPath('/SurfaceStudy/Tank')
            assert frozen_target.HasAPI(UsdPhysics.RigidBodyAPI)
            motion_xform=UsdGeom.Xformable(frozen_target).MakeMatrixXform()
            motion_xform.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*origin)))
        if action_plan:
            target=stage.GetPrimAtPath(action_plan['target_path'])
            assert target.HasAPI(UsdPhysics.RigidBodyAPI)
            assert UsdPhysics.RigidBodyAPI(target).GetKinematicEnabledAttr().Get()
            motion_xform=UsdGeom.Xformable(target).MakeMatrixXform()
            pose,_=action_pose(action_plan,0.,False)
            motion_xform.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pose)))
        fluid_rest = args.spacing * .5
        solid_rest = fluid_rest / .6
        system_path = Sdf.Path("/World/ParticleSystem")
        system = particleUtils.add_physx_particle_system(
            stage, system_path, simulation_owner=scene.GetPath(),
            contact_offset=solid_rest+.001, rest_offset=solid_rest,
            particle_contact_offset=solid_rest, solid_rest_offset=solid_rest,
            fluid_rest_offset=fluid_rest, enable_ccd=True,
            solver_position_iterations=args.iterations, max_depenetration_velocity=.25,
            max_neighborhood=96, neighborhood_scale=1.01, max_velocity=4.2)
        if args.decay_test:
            report['decay']['authored_solver_iterations']=int(system.GetSolverPositionIterationCountAttr().Get())
            assert report['decay']['authored_solver_iterations']==args.iterations
        material_path = Sdf.Path("/World/WaterMaterial")
        initial_damping, _ = prewarm_damping(0,args.prewarm_damping,args.damping_hold_seconds,args.damping_release_seconds)
        particleUtils.add_pbd_particle_material(stage, material_path, density=1000.,
            friction=friction, damping=initial_damping, viscosity=viscosity, vorticity_confinement=vorticity,
            surface_tension=.0074, cohesion=.01, adhesion=0., cfl_coefficient=1.)
        physicsUtils.add_physics_material_to_prim(stage, system.GetPrim(), material_path)
        damping_attr = stage.GetPrimAtPath(material_path).GetAttribute("physxPBDMaterial:damping")
        report['authored_vorticity_confinement']=float(stage.GetPrimAtPath(material_path).GetAttribute(
            'physxPBDMaterial:vorticityConfinement').Get())
        assert math.isclose(report['authored_vorticity_confinement'],vorticity,rel_tol=1e-6)
        if args.decay_test:
            material_prim=stage.GetPrimAtPath(material_path)
            report['decay']['authored_material']={key:float(material_prim.GetAttribute('physxPBDMaterial:'+key).Get())
                for key in ('density','friction','damping','viscosity','vorticityConfinement','surfaceTension','cohesion','adhesion','cflCoefficient')}
            assert math.isclose(report['decay']['authored_material']['viscosity'],viscosity,rel_tol=1e-6,abs_tol=1e-12)
        authored_damping = initial_damping
        if restored_arrays is not None:
            positions,velocities=restored_arrays
            restored_arrays=None
        else:
            axes = [(np.arange(n,dtype=np.float32)-(n-1)/2)*args.spacing for n in counts]
            axes[0] += origin[0]
            axes[1] = origin[1]+clearance+np.arange(counts[1],dtype=np.float32)*args.spacing
            axes[2] += origin[2]
            positions = np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3)
            velocities = np.zeros_like(positions)
        particle_path = Sdf.Path("/World/FluidParticles")
        prim = particleUtils.add_physx_particleset_pointinstancer(stage,particle_path,
            Vt.Vec3fArray.FromNumpy(positions),Vt.Vec3fArray.FromNumpy(velocities),system_path,
            self_collision=True,fluid=True,particle_group=0,particle_mass=0.,density=1000.)
        mass = UsdPhysics.MassAPI(prim)
        mass.GetMassAttr().Clear()
        mass.GetDensityAttr().Set(1000.)
        UsdGeom.Imageable(prim).MakeInvisible()
        instancer = UsdGeom.PointInstancer(prim)
        report["native_offsets_m"] = dict(fluid_rest=fluid_rest,solid_rest=solid_rest,
            particle_contact=solid_rest,contact=solid_rest+.001,wall_contact=.004)
        report["gpu_buffers"] = dict(particle_contacts=2097152,collision_stack_bytes=536870912,
            deformable_volume_contacts=1024,deformable_surface_contacts=1024)
        if args.initial_state:
            for key in ('native_offsets_m','gpu_buffers'):
                if report[key] != source_report[key]:
                    raise ValueError(f"Snapshot native configuration mismatch: {key}")
        report["initial_bounds_m"] = [positions.min(axis=0).tolist(),positions.max(axis=0).tolist()]
        if args.decay_test:
            (args.output/'decay_capture').mkdir()
            np.savez(args.output/'decay_capture/frame_0000.npz',positions=positions,velocities=velocities,simulated_seconds=0.)
        del positions, velocities
        app.update()
        simulation = get_physx_simulation_interface()
        cache = UsdUtils.StageCache.Get()
        stage_id = cache.GetId(stage)
        if not stage_id.IsValid():
            stage_id = cache.Insert(stage)
        simulation.attach_stage(stage_id.ToLongInt())
        attached = True
        start = time.perf_counter()
        report["status"] = "settling" if args.settle else "running"
        write_report(report_path, report)
        requested_steps = math.ceil(args.seconds*args.hz)
        max_settle_steps = math.ceil(args.settle_max_seconds*args.hz) if args.settle else 0
        total_steps = max_settle_steps+requested_steps
        capture_start_step = None if args.settle else 0
        capture_frames = []
        if args.decay_test:
            capture_frames=[dict(file='frame_0000.npz',recording_seconds=0.)]
        sample_stride = max(1,args.hz//10) if args.settle else 12
        if args.decay_test:sample_stride=args.hz//10
        capture_stride = args.hz//30
        if args.settle:
            (args.output/"settled_capture").mkdir()
        solve_total = update_total = 0.
        previous_sample = start
        checkpoint_wall = start
        for step in range(1,total_steps+1):
            if action_plan:
                action_time=(step-capture_start_step)/args.hz if capture_start_step is not None else step/args.hz
                target_pose,target_velocity=action_pose(action_plan,action_time,capture_start_step is not None)
                motion_xform.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*target_pose)))
            # Author at the native substep boundary, not at output-frame cadence.
            damping, relaxation_phase = prewarm_damping((step-1)/args.hz,args.prewarm_damping,
                args.damping_hold_seconds,args.damping_release_seconds)
            if damping != authored_damping:
                damping_attr.Set(damping)
                authored_damping = damping
            before = time.perf_counter()
            simulation.simulate(1/args.hz,(step-1)/args.hz)
            simulation.fetch_results()
            solved = time.perf_counter()
            app.update()
            updated = time.perf_counter()
            solve_total += solved-before
            update_total += updated-solved
            timed_out = updated-start >= args.wall_limit
            capture_done = capture_start_step is not None and step-capture_start_step >= requested_steps
            capture_due = args.settle and capture_start_step is not None and (step-capture_start_step)%capture_stride == 0
            decay_capture_due=args.decay_test and step % (args.hz//args.decay_capture_fps)==0
            if step == 1 or step % sample_stride == 0 or step == total_steps or timed_out or capture_done or capture_due or decay_capture_due or updated-previous_sample >= 15:
                p = np.asarray(instancer.GetPositionsAttr().Get(),dtype=np.float32)
                v = np.asarray(instancer.GetVelocitiesAttr().Get(),dtype=np.float32)
                assert len(p) == report["particle_count"] and len(v) == len(p)
                assert np.isfinite(p).all() and np.isfinite(v).all()
                speed = np.linalg.norm(v,axis=1)
                tank_position=target_pose if action_plan and action_plan['case']['motion']['target']=='Tank' else origin
                out = ((np.abs(p[:,0]-tank_position[0]) > x/2) |
                       (np.abs(p[:,2]-tank_position[2]) > z/2) | (p[:,1] < tank_position[1]))
                over_rim = p[:,1] > origin[1]+height
                regional_levels = []
                contained = ~out & ~over_rim
                contained_speed = speed[contained]
                for positive_x, positive_z in ((False,False),(False,True),(True,False),(True,True)):
                    region = contained & (((p[:,0]>=origin[0]) == positive_x) & ((p[:,2]>=origin[2]) == positive_z))
                    regional_levels.append(float(np.quantile(p[region,1]-origin[1],.99)) if region.any() else None)
                row = dict(step=step,simulated_seconds=step/args.hz,
                    phase="settling" if capture_start_step is None else "settled_capture" if args.settle else "capacity_probe",
                    relaxation_phase=relaxation_phase,native_damping=float(damping_attr.Get()),
                    wall_seconds=time.perf_counter()-start,solve_fetch_seconds=solve_total,
                    app_update_seconds=update_total,max_speed_m_s=float(speed.max()),
                    rms_speed_m_s=float(np.sqrt(np.mean(speed**2))),
                    speed_p99_m_s=float(np.quantile(speed,.99)),
                    speed_cap_fraction=float(np.mean(speed>=4.2*.999)),
                    outside_side_or_floor_count=int(out.sum()),
                    over_rim_particle_count=int(over_rim.sum()),
                    contained_particle_count=int(contained.sum()),
                    contained_rms_speed_m_s=float(np.sqrt(np.mean(contained_speed**2))) if len(contained_speed) else None,
                    contained_speed_p99_m_s=float(np.quantile(contained_speed,.99)) if len(contained_speed) else None,
                    regional_level_p99_m=regional_levels,
                    bounds_m=[p.min(axis=0).tolist(),p.max(axis=0).tolist()])
                report["rows"].append(row)
                if args.decay_test:
                    from omni.physx import get_physx_interface
                    native_pose=get_physx_interface().get_rigidbody_transformation('/SurfaceStudy/Tank')
                    if not native_pose.get('ret_val') or np.linalg.norm(np.asarray(native_pose['position'])-origin)>1e-4:
                        raise RuntimeError('Frozen tank native pose mismatch')
                    row['phase']='decay_diagnostic'
                    row['wave_metrics']=wave_metrics(p,v,origin,[x,height,z])
                    if decay_capture_due:
                        filename=f'frame_{len(capture_frames):04d}.npz'
                        np.savez(args.output/'decay_capture'/filename,positions=p,velocities=v,simulated_seconds=step/args.hz)
                        capture_frames.append(dict(file=filename,recording_seconds=step/args.hz))
                if action_plan:
                    row['target_pose_isaac_m']=target_pose
                    row['target_velocity_isaac_m_s']=target_velocity
                    row['recording_seconds']=(step-capture_start_step)/args.hz if capture_start_step is not None else None
                    from omni.physx import get_physx_interface
                    native_pose=get_physx_interface().get_rigidbody_transformation(action_plan['target_path'])
                    if not native_pose.get('ret_val'):
                        raise RuntimeError('Native kinematic pose unavailable')
                    row['native_target_position_m']=list(native_pose['position'])
                    error=float(np.linalg.norm(np.asarray(native_pose['position'])-target_pose))
                    row['native_target_error_m']=error
                    if error>1e-4:
                        raise RuntimeError(f'Native kinematic target failed: {error} m')
                print('[probe-step] '+json.dumps(row),flush=True)
                write_report(report_path, report)
                previous_sample = time.perf_counter()
                if args.settle and (previous_sample-checkpoint_wall >= 120 or timed_out):
                    temporary = args.output/"prewarm_checkpoint.tmp.npz"
                    np.savez(temporary,positions=p,velocities=v,simulated_seconds=step/args.hz)
                    temporary.replace(args.output/"prewarm_checkpoint.npz")
                    checkpoint_wall = time.perf_counter()
                if args.settle and capture_start_step is None and relaxation_phase=="normal_validation" and gate.observe(row):
                    capture_start_step = step
                    report["settled_water"] = True
                    report["settling"]["passed_at_seconds"] = step/args.hz
                    report["status"] = "recording_after_settle"
                    np.savez(args.output/"settled_state.npz",positions=p,velocities=v,simulated_seconds=step/args.hz)
                    capture_due = True
                    print('[settled] '+json.dumps(row),flush=True)
                if capture_due:
                    filename = f"frame_{len(capture_frames):04d}.npz"
                    relative_time = (step-capture_start_step)/args.hz
                    np.savez(args.output/"settled_capture"/filename,positions=p,velocities=v,
                             simulated_seconds=step/args.hz,recording_seconds=relative_time)
                    capture_frames.append(dict(file=filename,recording_seconds=relative_time))
                    if action_plan:
                        capture_frames[-1]['target_pose_isaac_m']=target_pose
                    write_report(args.output/"settled_capture"/"manifest.json",
                        dict(product="surface_study_settled_native_snapshots",complete=False,
                             fps=30,particle_count=len(p),frames=capture_frames))
                if (not args.settle and (out.any() or (args.decay_test and over_rim.any()))) or row["speed_cap_fraction"] > .01 or (args.settle and capture_start_step is not None and not args.action_case and (out.any() or over_rim.any())):
                    report["status"] = "stopped_instability"
                    break
            if capture_done:
                report["status"] = "completed_decay_test" if args.decay_test else "completed_settled_capture" if args.settle else "completed_short_probe"
                break
            if args.settle and capture_start_step is None and step >= max_settle_steps:
                report["status"] = "not_settled_within_simulation_limit"
                break
            if timed_out:
                report["status"] = "stopped_wall_limit"
                break
        else:
            report["status"] = "not_settled_within_simulation_limit" if args.settle else "completed_short_probe"
        p = np.asarray(instancer.GetPositionsAttr().Get(),dtype=np.float32)
        v = np.asarray(instancer.GetVelocitiesAttr().Get(),dtype=np.float32)
        np.savez(args.output/"final_state.npz",positions=p,velocities=v,
                 simulated_seconds=step/args.hz)
        report["completed_steps"] = step
        report["simulated_seconds"] = step/args.hz
        report["loop_wall_seconds"] = time.perf_counter()-start
        if args.decay_test:
            report['recorded_frames']=len(capture_frames)
            write_report(args.output/'decay_capture/manifest.json',dict(complete=report['status']=='completed_decay_test',fps=args.decay_capture_fps,
                frames=capture_frames,source=report['initial_state'],note='Native diagnostic snapshots, no temporal interpolation'))
        if args.settle:
            report["recorded_frames"] = len(capture_frames)
            if action_plan:
                report['action_capture_outside_max']=max((r['outside_side_or_floor_count'] for r in report['rows'] if r['phase']=='settled_capture'),default=0)
            write_report(args.output/"settled_capture"/"manifest.json",
                dict(product="surface_study_settled_native_snapshots",
                     complete=report["status"]=="completed_settled_capture",fps=30,
                     particle_count=len(p),frames=capture_frames,action=action_plan,
                     note="Separate prewarm and recording clocks; arrays are native positions and velocities."))
        write_report(report_path,report)
        print('[probe-done] '+json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        write_report(report_path,report)
        raise
    finally:
        if attached:
            simulation.detach_stage()
        app.close()


if __name__ == "__main__":
    main()
