"""30 fps native recapture and matched render of vorticity 10 versus 0.02."""
import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

from run_surface_decay_compare import command, ROOT
from finish_surface_overnight import render_command
sys.path.insert(0,str(ROOT))
from coupled_scene.surface_decay import decay_threshold_times


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--desktop',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    args=parser.parse_args()
    if shutil.disk_usage(args.output.parent).free<60*1024**3:raise RuntimeError('Need 60 GiB free on output drive')
    args.output.mkdir(parents=True,exist_ok=False)
    args.desktop.mkdir(parents=True,exist_ok=False)
    source=args.source_run/'settled_capture/frame_0180.npz'
    report=args.source_run/'probe_report.json'
    original=json.loads(report.read_text())
    pose=original['action']['base_isaac']
    assert original['status']=='completed_settled_capture'
    status=dict(phase='validating',started_utc=datetime.now(timezone.utc).isoformat(),
        fps=30,seconds=4,resolution=[1280,960],samples=64,smoothing=25,cases={},
        note='Both arms restore the same native slosh t=6 state; no settling/interpolation; only vorticity differs.')
    (args.desktop/'说明.txt').write_text(
        '左：原版涡量补偿 0.02；右：涡量补偿 10。\n'
        '同一停止晃动快照恢复后，各续跑 4 秒，30 fps 原生快照，不做运动插帧。\n'
        '720 Hz、64 次迭代、粘性 0.002、摩擦 0.05；同一相机、25 次水面平滑。\n'
        '两组均重新补录；补偿增强运动不等于物理保真，需观察连贯波浪与杂乱搅动。\n',encoding='utf-8')
    def save():
        for dest in (args.output/'video_status.json',args.desktop/'任务状态.json'):
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(dest)
    def run(cmd,log,progress=None):
        with log.open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if progress:progress()
                time.sleep(10)
            if progress:progress()
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
    def wait_idle():
        count=0
        while count<3:
            values=[list(map(int,line.split(','))) for line in subprocess.check_output([
                'nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()]
            count=count+1 if all(m<1024 and u<20 for m,u in values) else 0
            status.update(phase='waiting_for_gpu',last_gpu=values);save()
            if count<3:time.sleep(10)
    def verify(video,width=1280,height=960):
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_frames,duration,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
        assert int(stream['nb_frames'])==120 and abs(float(stream['duration'])-4)<.001
        assert (stream['width'],stream['height'],stream['r_frame_rate'])==(width,height,'30/1')
        return stream
    def deliver(video):
        dest=args.desktop/video.name
        if dest.exists():raise FileExistsError(dest)
        shutil.copy2(video,dest)
        return str(dest)
    cases=[('vorticity10',10.,'涡量10_4秒'),('baseline',.02,'原版0.02_4秒')]
    save()
    try:
        for key,value,title in cases:
            run(command(args.layout,source,report,args.output/key/'simulation',64,dry_run=True,
                vorticity=value,capture_fps=30),args.output/f'validate_{key}.log')
            status['cases'][key]=dict(phase='queued',vorticity=value);save()
        for key,value,title in cases:
            case=args.output/key;case.mkdir()
            state=status['cases'][key]
            wait_idle()
            status['phase']=key;state['phase']='simulating';save()
            sim=case/'simulation'
            def sim_progress():
                path=sim/'probe_report.json'
                if path.exists():
                    data=json.loads(path.read_text())
                    if data['rows']:state['simulated_seconds']=data['rows'][-1]['simulated_seconds'];save()
            run(command(args.layout,source,report,sim,64,vorticity=value,capture_fps=30),case/'simulation.log',sim_progress)
            data=json.loads((sim/'probe_report.json').read_text())
            assert data['status']=='completed_decay_test' and data['simulated_seconds']==4
            assert data['decay']['authored_material']['vorticityConfinement']==value or abs(data['decay']['authored_material']['vorticityConfinement']-value)<1e-8
            state.update(source_sha256=data['initial_state']['sha256'],thresholds=decay_threshold_times(data))
            capture=sim/'decay_capture';manifest=json.loads((capture/'manifest.json').read_text())
            frames=manifest['frames']
            assert manifest['complete'] and manifest['fps']==30 and len(frames)==121
            assert all(abs(f['recording_seconds']-i/30)<1e-8 for i,f in enumerate(frames))
            state['phase']='reconstructing';save()
            surfaces=case/'surfaces';surfaces.mkdir()
            def build(frame):
                dest=surfaces/Path(frame['file']).stem
                with (surfaces/(dest.name+'.log')).open('x') as log:
                    subprocess.run([sys.executable,str(ROOT/'experiments/coupled_scenes/reconstruct_surface_snapshot.py'),
                        str(capture/frame['file']),str(dest),'--mesh-smoothing-iters','25'],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                return dict(surface=str(Path(dest.name)/'water.obj'),recording_seconds=frame['recording_seconds'],target_pose_isaac_m=pose)
            rebuilt=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for frame in pool.map(build,frames):
                    rebuilt.append(frame);state['reconstructed_frames']=len(rebuilt);save()
            (surfaces/'sequence.json').write_text(json.dumps(dict(complete=True,action=original['action'],frames=rebuilt),indent=2))
            wait_idle();status['phase']=key;state['phase']='rendering';save()
            renders=case/'renders'
            def render_progress():
                path=renders/'render_manifest.json'
                if path.exists():state['rendered_frames']=len(json.loads(path.read_text())['frames']);save()
            run(render_command(args.layout/'02_sloshing.blend',surfaces,renders),case/'render.log',render_progress)
            rendered=json.loads((renders/'render_manifest.json').read_text())
            assert rendered['complete'] and len(rendered['frames'])==121
            state['phase']='encoding';save()
            video=case/(title+'.mp4')
            run(['ffmpeg','-hide_banner','-loglevel','warning','-n','-framerate','30','-i',str(renders/'frame_%04d.png'),
                '-frames:v','120','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],case/'encode.log')
            state.update(phase='complete',verification=verify(video),video=str(video),desktop=deliver(video));save()
        assert len({s['source_sha256'] for s in status['cases'].values()})==1
        status['phase']='encoding_comparison';save()
        video=args.output/'并排对比_左原版_右涡量10.mp4'
        filter_graph=("[0:v]scale=960:720,drawtext=text='Baseline 0.02':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65[l];"
                      "[1:v]scale=960:720,drawtext=text='Vorticity 10':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65[r];[l][r]hstack=inputs=2[v]")
        run(['ffmpeg','-hide_banner','-loglevel','warning','-n','-i',status['cases']['baseline']['video'],
            '-i',status['cases']['vorticity10']['video'],'-filter_complex',filter_graph,'-map','[v]',
            '-frames:v','120','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],args.output/'compare_encode.log')
        status.update(phase='complete',comparison_verification=verify(video,1920,720),comparison_video=deliver(video))
    except Exception as exc:
        status.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
