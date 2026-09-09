"""WSL: wait for prior renders, then serial 720 Hz x 16/32/64 native decay tests."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from build_cabinet_liquid_surfaces import windows_path

ROOT=Path(__file__).resolve().parents[2]


def command(layout, source, report, output, iterations, dry_run=False, viscosity=None, hz=720, timestep_test=False, friction_test=False, vorticity=None, capture_fps=None):
    pairs=[('--layout',layout),('--initial-state',source),('--initial-report',report),('--output',output)]
    script=ROOT/'experiments/coupled_scenes/run_surface_study_probe.py'
    cmd='& "Y:\\isaacsim\\python.bat" -u "'+windows_path(script)+'"'
    cmd+=''.join(' '+key+' "'+windows_path(value)+'"' for key,value in pairs)
    cmd+=f' --decay-test --hz {hz} --iterations {iterations} --seconds 4 --wall-limit 5400'
    if timestep_test:cmd+=' --decay-timestep-test'
    if friction_test:cmd+=' --decay-friction-test'
    if viscosity is not None:cmd+=f' --decay-viscosity {viscosity}'
    if vorticity is not None:cmd+=f' --decay-vorticity {vorticity}'
    if capture_fps is not None:cmd+=f' --decay-capture-fps {capture_fps}'
    if dry_run:cmd+=' --dry-run'
    return ['powershell.exe','-NoProfile','-Command',cmd+'; exit $LASTEXITCODE']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--wait-for-status',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.source_run/'settled_capture/frame_0180.npz'
    report=args.source_run/'probe_report.json'
    summary=dict(product='shared_state_density_iteration_decay_comparison',phase='validating',
                 started_utc=datetime.now(timezone.utc).isoformat(),source=str(source),
                 hz=720,seconds=4,iterations=[16,32,64],cases={},
                 controls='Same positions AND velocities at slosh t=6. Fixed tank, no prewarm, no material changes, no surface reconstruction.')
    def save():
        target=args.output/'comparison_status.json';temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(summary,ensure_ascii=False,indent=2));temp.replace(target)
    def run(cmd,log,progress=None):
        with log.open('x',encoding='utf-8') as stream:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if progress:progress()
                time.sleep(10)
            if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,cmd)
    def wait_idle():
        idle_count=0
        while idle_count<3:
            prior=json.loads(args.wait_for_status.read_text())
            prior_done=prior['phase'] in ('complete','finished_with_failures')
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True)
            values=[list(map(int,line.split(','))) for line in gpu.strip().splitlines()]
            idle=prior_done and all(memory<1024 and usage<20 for memory,usage in values)
            idle_count=idle_count+1 if idle else 0
            summary.update(phase='waiting_for_gpu',prior_render_phase=prior['phase'],last_gpu=values)
            save()
            if idle_count<3:time.sleep(30)
    save()
    try:
        for iterations in (16,32,64):
            dest=args.output/f'iter{iterations}'
            run(command(args.layout,source,report,dest,iterations,True),args.output/f'validate_iter{iterations}.log')
            summary['cases'][str(iterations)]=dict(phase='queued')
            save()
        print('All three input validations passed; waiting for prior rendering and idle GPU.',flush=True)
        for iterations in (16,32,64):
            wait_idle()
            dest=args.output/f'iter{iterations}'
            state=summary['cases'][str(iterations)]
            state.update(phase='simulating',started_utc=datetime.now(timezone.utc).isoformat())
            summary['phase']=f'iter{iterations}';save()
            def progress():
                path=dest/'probe_report.json'
                if path.exists():
                    current=json.loads(path.read_text())
                    rows=current.get('rows',[])
                    state['simulated_seconds']=rows[-1]['simulated_seconds'] if rows else 0
                    save()
            try:
                run(command(args.layout,source,report,dest,iterations),args.output/f'iter{iterations}.log',progress)
                result=json.loads((dest/'probe_report.json').read_text())
                if result['status']!='completed_decay_test':raise RuntimeError(result['status'])
                state.update(phase='complete',simulated_seconds=result['simulated_seconds'],
                             wall_seconds=result['loop_wall_seconds'],source_sha256=result['initial_state']['sha256'],
                             final_metrics=result['rows'][-1]['wave_metrics'])
            except Exception as exc:
                state.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
            save()
        hashes={s['source_sha256'] for s in summary['cases'].values() if s['phase']=='complete'}
        assert len(hashes)<=1
        summary['phase']='complete' if all(s['phase']=='complete' for s in summary['cases'].values()) else 'finished_with_failures'
    except Exception as exc:
        summary.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
