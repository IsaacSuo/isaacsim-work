"""Serial official Water diagnostic with optional last-four-second inspection."""
import json
import argparse
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from run_pour_vorticity_compare import powershell
from build_cabinet_liquid_surfaces import windows_path
from run_active_pour_probe import atomic_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/coupled_scenes/active_piston_static_v7_official_water'
PREVIOUS=ROOT/'output/coupled_scenes/active_stirring_water_v6_speed2x/status.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--assisted-prewarm',action='store_true')
    parser.add_argument('--assets',type=Path,default=ROOT/'output/coupled_scenes/active_03_piston_push_assets_v2')
    parser.add_argument('--render-desktop',type=Path,help='After completed diagnostic, render 6-10s to this new desktop folder')
    args=parser.parse_args()
    meta=json.loads((args.assets/'assets.json').read_text(encoding='utf-8'))
    assert meta['case']['id']=='03_piston_push'
    out=args.output
    out.mkdir(exist_ok=False)
    state=dict(phase='waiting_for_previous_task',started_utc=datetime.now(timezone.utc).isoformat(),
               waiting_for_task=str(PREVIOUS),diagnostic_only=True,assisted_prewarm=args.assisted_prewarm,
               assets=str(args.assets),particle_count=meta['particle_count'],spacing_m=meta['particle_spacing_m'])
    def save():atomic_json(out/'status.json',state)
    save()
    try:
        while True:
            prior=json.loads(PREVIOUS.read_text(encoding='utf-8'))
            state['previous_task_phase']=prior['phase'];save()
            if prior['phase'] in ('complete','failed','finished_with_failures'):break
            time.sleep(10)
        state['phase']='waiting_for_gpu';save();idle=0
        while idle<3:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            values=[list(map(int,line.split(','))) for line in raw.strip().splitlines()]
            idle=idle+1 if all(m<1024 and u<20 for m,u in values) else 0
            state['gpu_idle_check']=values;save()
            if idle<3:time.sleep(10)
        command=powershell([r'Y:\isaacsim\python.bat',windows_path(ROOT/'experiments/coupled_scenes/run_active_pour_probe.py'),
            '--assets',windows_path(args.assets),
            '--output',windows_path(out/'simulation'),'--official-water-probe','--recycle-piston-prewarm']+
            (['--official-water-prewarm'] if args.assisted_prewarm else []))
        state.update(phase='simulating',command=command);save()
        with (out/'simulation.log').open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((out/'simulation/probe_report.json').read_text(encoding='utf-8'))
        if result.returncode or report['status']!='completed':
            raise RuntimeError(report.get('error',f"exit {result.returncode}, status {report['status']}"))
        state.update(phase='simulation_complete',physics_gate_passed=False,
            final_strict_still_water_passed=report['final_strict_still_water_passed'])
        save()
        if args.render_desktop:
            state['phase']='building_video';save()
            command=[sys.executable,str(ROOT/'experiments/coupled_scenes/run_active_pour_video.py'),
                '--assets',str(args.assets),'--design',str(ROOT/'output/coupled_scenes/active_drive_contact_v4/03_piston_push'),
                '--simulation',str(out/'simulation'),'--diagnostic-static-last4','--output',str(out/'video'),
                '--desktop',str(args.render_desktop),'--video-name',f'官方Water_{round(meta["particle_spacing_m"]*1000)}mm_静水最后4秒.mp4']
            with (out/'video_runner.log').open('x',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            video=json.loads((out/'video/video_status.json').read_text(encoding='utf-8'))
            if result.returncode or video['phase']!='complete':raise RuntimeError(video.get('error','Video did not complete'))
            state['video']=video['video']
        state['phase']='complete'
    except BaseException as exc:
        state.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:save()


if __name__=='__main__':main()
