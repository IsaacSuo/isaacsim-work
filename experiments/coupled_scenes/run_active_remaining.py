"""Serial accepted stirring/piston native tests, then matched videos.

One GPU task at a time. Per-case failures are reported, never rendered as success.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

from run_pour_vorticity_compare import powershell
from build_cabinet_liquid_surfaces import atomic_json,windows_path

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--desktop',type=Path,required=True)
    parser.add_argument('--designs',type=Path,default=ROOT/'output/coupled_scenes/active_drive_design_v3')
    parser.add_argument('--assets-version',default='v1')
    parser.add_argument('--stirring-prewarm-recycling',action='store_true')
    parser.add_argument('--piston-prewarm-recycling',action='store_true')
    parser.add_argument('--cases',nargs='+',choices=['02_stirring','03_piston_push'],default=['02_stirring','03_piston_push'])
    parser.add_argument('--wait-for-video',type=Path,help='Wait for an existing video task to reach complete/failed before using GPU')
    parser.add_argument('--wait-for-task',type=Path,help='Wait for an existing queue status to reach complete/failed before using GPU')
    parser.add_argument('--stirring-speed',type=float,help='Override stirring peak angular speed in rad/s')
    args=parser.parse_args()
    assert shutil.disk_usage(args.output.parent).free>120*1024**3
    args.output.mkdir(parents=True,exist_ok=False);args.desktop.mkdir(parents=True,exist_ok=False)
    status=dict(phase='starting',started_utc=datetime.now(timezone.utc).isoformat(),cases={})
    def save():
        atomic_json(args.output/'status.json',status);atomic_json(args.desktop/'任务状态.json',status)
    def idle():
        count=0
        while count<3:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
            values=[list(map(int,r.split(','))) for r in rows]
            count=count+1 if all(m<1024 and u<20 for m,u in values) else 0
            status['gpu']=values;save()
            if count<3:time.sleep(10)
    def run(cmd,log,progress):
        with log.open('x') as stream:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while p.poll() is None:
                progress();time.sleep(10)
            progress()
            if p.returncode:raise subprocess.CalledProcessError(p.returncode,cmd)
    cases=[(k,t) for k,t in [('02_stirring','玻璃碗搅拌_完整10秒'),('03_piston_push','活塞推水_完整10秒')] if k in args.cases]
    save()
    if args.wait_for_task:
        assert args.wait_for_task.is_file()
        status.update(phase='waiting_for_previous_task',waiting_for_task=str(args.wait_for_task));save()
        while True:
            previous=json.loads(args.wait_for_task.read_text(encoding='utf-8'))
            status['previous_task_phase']=previous['phase'];save()
            if previous['phase'] in ('complete','failed','finished_with_failures'):break
            time.sleep(10)
    if args.wait_for_video:
        assert args.wait_for_video.is_file()
        status.update(phase='waiting_for_previous_video',waiting_for_video=str(args.wait_for_video));save()
        while True:
            previous=json.loads(args.wait_for_video.read_text(encoding='utf-8'))
            status['previous_video_phase']=previous['phase'];save()
            if previous['phase'] in ('complete','failed'):break
            time.sleep(10)
    # Finish native checks for both cases before spending time rendering either.
    for key,title in cases:
        folder=args.output/key;folder.mkdir();sim=folder/'simulation'
        assets=ROOT/'output/coupled_scenes'/('active_'+key+'_assets_'+args.assets_version)
        state=dict(phase='waiting_for_gpu',title=title);status['cases'][key]=state;status['phase']=key;save()
        try:
            idle();state['phase']='simulating';save()
            def progress():
                path=sim/'probe_report.json'
                if path.exists():
                    d=json.loads(path.read_text());state['simulation']={k:d.get(k) for k in
                        ['status','last_simulated_seconds','last_action_seconds','recorded_frames','active_particle_count','error']}
                    save()
            cmd=powershell([r'Y:\isaacsim\python.bat',windows_path(ROOT/'experiments/coupled_scenes/run_active_pour_probe.py'),
                '--assets',windows_path(assets),'--output',windows_path(sim),
                '--restore-vorticity-after-settling','--relaxed-restored-readiness']+
                (['--stirring-speed',str(args.stirring_speed)] if key=='02_stirring' and args.stirring_speed is not None else [])+
                (['--recycle-stirring-prewarm'] if key=='02_stirring' and args.stirring_prewarm_recycling else [])+
                (['--recycle-piston-prewarm'] if key=='03_piston_push' and args.piston_prewarm_recycling else []))
            run(cmd,folder/'simulation.log',progress)
            d=json.loads((sim/'probe_report.json').read_text())
            assert d['status']=='completed',d.get('error')
            m=json.loads((sim/'capture/manifest.json').read_text());assert m['complete'] and len(m['frames'])==301
            state['phase']='awaiting_render';save()
        except Exception as exc:
            state.update(phase='failed',error=f'{type(exc).__name__}: {exc}');save()
    for key,title in cases:
        state=status['cases'][key]
        if state['phase']!='awaiting_render':continue
        folder=args.output/key;video=folder/'video';status['phase']=key
        try:
            state['phase']='building_video';save()
            def progress():
                path=video/'video_status.json'
                if path.exists():state['video']=json.loads(path.read_text());save()
            cmd=[sys.executable,str(ROOT/'experiments/coupled_scenes/run_active_pour_video.py'),
                '--assets',str(ROOT/'output/coupled_scenes'/('active_'+key+'_assets_'+args.assets_version)),
                '--design',str(args.designs/key),
                '--simulation',str(folder/'simulation'),'--output',str(video),
                '--desktop',str(args.desktop/key),'--video-name',title+'.mp4']
            run(cmd,folder/'video_runner.log',progress)
            assert state['video']['phase']=='complete'
            state['phase']='complete';save()
        except Exception as exc:
            state.update(phase='failed',error=f'{type(exc).__name__}: {exc}');save()
    status['phase']='complete' if all(s['phase']=='complete' for s in status['cases'].values()) else 'finished_with_failures'
    status['finished_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
