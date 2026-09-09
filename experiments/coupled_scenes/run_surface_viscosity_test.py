"""One-variable viscosity, friction or vorticity test against the v6 control."""
import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_surface_decay_compare import command, ROOT
sys.path.insert(0,str(ROOT))
from coupled_scene.surface_decay import decay_threshold_times


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--reference-report',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--friction-test',action='store_true',help='Zero contact friction instead of changing viscosity')
    mode.add_argument('--vorticity-test',action='store_true',help='Vorticity confinement 10 instead of changing viscosity')
    args=parser.parse_args()
    reference=json.loads(args.reference_report.read_text())
    assert reference['status']=='completed_decay_test' and reference['hz']==720 and reference['iterations']==64
    assert reference['simulated_seconds']==4 and reference['decay']['only_parameter_changed']=='iterations'
    args.output.mkdir(parents=True,exist_ok=False)
    simulation=args.output/'simulation'
    source=args.source_run/'settled_capture/frame_0180.npz'
    source_report=args.source_run/'probe_report.json'
    options=dict(friction_test=True) if args.friction_test else dict(viscosity=.0000017)
    if args.vorticity_test:options=dict(vorticity=10.)
    status=dict(phase='validating',started_utc=datetime.now(timezone.utc).isoformat(),
        reference_report=str(args.reference_report),reference_viscosity=.002,test_viscosity=.002 if (args.friction_test or args.vorticity_test) else .0000017,
        hz=720,iterations=64,seconds=4,source=str(source),
        controls='Only viscosity changes. Same native positions/velocities, fixed tank, no prewarm, no rendering.',
        reference_thresholds=decay_threshold_times(reference))
    if args.friction_test:
        status.update(reference_friction=.05,test_friction=0.,
            controls='Only particle/wall contact friction changes. Collisions retained. Same positions/velocities, 720 Hz, 64 iterations, viscosity .002, no prewarm/rendering.')
    if args.vorticity_test:
        status.update(reference_vorticity=.02,test_vorticity=10.,
            controls='Only vorticity confinement changes. Same positions/velocities, 720 Hz, 64 iterations, viscosity .002, friction .05, no prewarm/rendering.',
            interpretation='Added agitation is not necessarily preserved wave energy; compare coarse wave heights and residual motion.')
    def save():
        dest=args.output/'comparison_status.json';temp=dest.with_suffix('.tmp')
        temp.write_text(json.dumps(status,ensure_ascii=False,indent=2));temp.replace(dest)
    def run(cmd,log,progress=False):
        with log.open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                report_path=simulation/'probe_report.json'
                if progress and report_path.exists():
                    report=json.loads(report_path.read_text())
                    if report.get('rows'):
                        status['simulated_seconds']=report['rows'][-1]['simulated_seconds'];save()
                time.sleep(10)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
    save()
    try:
        run(command(args.layout,source,source_report,simulation,64,dry_run=True,**options),args.output/'validation.log')
        status['phase']='waiting_for_gpu';save()
        idle_count=0
        while idle_count<3:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True)
            values=[list(map(int,line.split(','))) for line in gpu.strip().splitlines()]
            idle_count=idle_count+1 if all(memory<1024 and usage<20 for memory,usage in values) else 0
            status['last_gpu']=values;save()
            if idle_count<3:time.sleep(10)
        status['phase']='simulating';save()
        run(command(args.layout,source,source_report,simulation,64,**options),args.output/'simulation.log',True)
        result=json.loads((simulation/'probe_report.json').read_text())
        assert result['status']=='completed_decay_test' and result['simulated_seconds']==4
        assert result['initial_state']['sha256']==reference['initial_state']['sha256']
        assert result['decay']['initial_metrics']==reference['decay']['initial_metrics']
        for key in ('hz','iterations','spacing_m','particle_count','native_offsets_m','gpu_buffers'):
            assert result[key]==reference[key],key
        if args.friction_test:
            assert result['decay']['only_parameter_changed']=='contact_friction'
            assert result['decay']['authored_wall_material']==dict(static_friction=0.,dynamic_friction=0.,restitution=0.)
            expected=dict(density=1000.,friction=0.,damping=.01,viscosity=.002,
                vorticityConfinement=.02,surfaceTension=.0074,cohesion=.01,adhesion=0.,cflCoefficient=1.)
            for key,value in expected.items():
                assert math.isclose(result['decay']['authored_material'][key],value,rel_tol=1e-6,abs_tol=1e-12),key
        if args.vorticity_test:
            assert result['decay']['only_parameter_changed']=='vorticity_confinement'
            expected=dict(density=1000.,friction=.05,damping=.01,viscosity=.002,
                vorticityConfinement=10.,surfaceTension=.0074,cohesion=.01,adhesion=0.,cflCoefficient=1.)
            for key,value in expected.items():
                assert math.isclose(result['decay']['authored_material'][key],value,rel_tol=1e-6,abs_tol=1e-12),key
            for key,value in dict(static_friction=.05,dynamic_friction=.05,restitution=0.).items():
                assert math.isclose(result['decay']['authored_wall_material'][key],value,rel_tol=1e-6,abs_tol=1e-12),key
        status.update(phase='complete',simulated_seconds=4,
            source_sha256=result['initial_state']['sha256'],wall_seconds=result['loop_wall_seconds'],
            test_thresholds=decay_threshold_times(result),final_metrics=result['rows'][-1]['wave_metrics'])
    except Exception as exc:
        status.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
