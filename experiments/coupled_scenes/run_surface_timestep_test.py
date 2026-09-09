"""Serial 360/1440 Hz decay tests; reuse completed 720 Hz original-viscosity control."""
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
    args=parser.parse_args()
    reference=json.loads(args.reference_report.read_text())
    assert reference['status']=='completed_decay_test' and reference['hz']==720 and reference['iterations']==64
    assert reference['simulated_seconds']==4 and reference['decay']['only_parameter_changed']=='iterations'
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.source_run/'settled_capture/frame_0180.npz'
    source_report=args.source_run/'probe_report.json'
    status=dict(phase='validating',started_utc=datetime.now(timezone.utc).isoformat(),
        reference_report=str(args.reference_report),viscosity=.002,iterations=64,seconds=4,
        source=str(source),test_hz=[360,1440],
        controls='Only timestep/frequency changes. Same native positions/velocities, fixed tank, no prewarm, no rendering.',
        cases={'720':dict(phase='complete_reused',thresholds=decay_threshold_times(reference))})
    def save():
        dest=args.output/'comparison_status.json';temp=dest.with_suffix('.tmp')
        temp.write_text(json.dumps(status,ensure_ascii=False,indent=2));temp.replace(dest)
    def run(cmd,log,progress=None):
        with log.open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if progress:progress()
                time.sleep(10)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
    def wait_idle():
        idle_count=0
        while idle_count<3:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True)
            values=[list(map(int,line.split(','))) for line in gpu.strip().splitlines()]
            idle_count=idle_count+1 if all(memory<1024 and usage<20 for memory,usage in values) else 0
            status.update(phase='waiting_for_gpu',last_gpu=values);save()
            if idle_count<3:time.sleep(10)
    save()
    try:
        for hz in (360,1440):
            run(command(args.layout,source,source_report,args.output/f'hz{hz}',64,
                        dry_run=True,hz=hz,timestep_test=True),args.output/f'validate_hz{hz}.log')
            status['cases'][str(hz)]=dict(phase='queued');save()
        print('Both input validations passed.',flush=True)
        for hz in (360,1440):
            wait_idle()
            dest=args.output/f'hz{hz}'
            state=status['cases'][str(hz)]
            state.update(phase='simulating',started_utc=datetime.now(timezone.utc).isoformat())
            status['phase']=f'hz{hz}';save()
            def progress():
                path=dest/'probe_report.json'
                if path.exists():
                    report=json.loads(path.read_text())
                    if report.get('rows'):
                        state['simulated_seconds']=report['rows'][-1]['simulated_seconds'];save()
            try:
                run(command(args.layout,source,source_report,dest,64,hz=hz,timestep_test=True),
                    args.output/f'hz{hz}.log',progress)
                result=json.loads((dest/'probe_report.json').read_text())
                assert result['status']=='completed_decay_test' and result['simulated_seconds']==4
                assert result['hz']==hz and result['completed_steps']==hz*4
                assert result['initial_state']['sha256']==reference['initial_state']['sha256']
                assert result['decay']['initial_metrics']==reference['decay']['initial_metrics']
                for key in ('iterations','spacing_m','particle_count','native_offsets_m','gpu_buffers'):
                    assert result[key]==reference[key],key
                expected=dict(density=1000.,friction=.05,damping=.01,viscosity=.002,
                    vorticityConfinement=.02,surfaceTension=.0074,cohesion=.01,adhesion=0.,cflCoefficient=1.)
                for key,value in expected.items():
                    assert math.isclose(result['decay']['authored_material'][key],value,rel_tol=1e-6,abs_tol=1e-12),key
                state.update(phase='complete',simulated_seconds=4,wall_seconds=result['loop_wall_seconds'],
                    thresholds=decay_threshold_times(result),final_metrics=result['rows'][-1]['wave_metrics'],
                    source_sha256=result['initial_state']['sha256'])
            except Exception as exc:
                state.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
                print(f'Failed {hz} Hz: {exc}',flush=True)
            save()
        status['phase']='complete' if all(status['cases'][str(h)]['phase']=='complete' for h in (360,1440)) else 'finished_with_failures'
    except Exception as exc:
        status.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
