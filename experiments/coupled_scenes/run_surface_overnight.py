"""Serial overnight first-group action previews; independent cases continue on failure."""
import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

from build_cabinet_liquid_surfaces import windows_path

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--desktop',type=Path,required=True)
    parser.add_argument('--initial-state',type=Path,required=True)
    parser.add_argument('--layouts',type=Path,required=True)
    args=parser.parse_args()
    if shutil.disk_usage(args.output.parent).free < 150*1024**3:
        raise RuntimeError('Need at least 150 GiB free for native arrays and meshes')
    args.output.mkdir(parents=True,exist_ok=False)
    args.desktop.mkdir(parents=True,exist_ok=False)
    (args.desktop/'说明.txt').write_text(
        '第一组液面动作串行样片\n'
        '02：容器晃动，10 秒；04：端部推波及反弹，10 秒。\n'
        '05：中央扰动后恢复，12 秒；03：同一次中央扰动的前 6 秒。\n'
        '03 是有限水槽里的传播观察，后半段可能含边界回波，不能整段当无限水域自由传播。\n'
        '05 不预先保证片段末尾完全恢复平静。\n'
        '4 mm 粒子，720 Hz，64 次迭代；25 次重建平滑；1280×960、30 fps。\n'
        '任务状态.json 标明进行中/成功/失败。失败不会被伪装成完成。\n',encoding='utf-8')
    cases=[('02_sloshing','02_容器晃动',10),('04_wave_reflection','04_推波与反弹',10),('05_surface_recovery','05_液面恢复',12)]
    summary=dict(product='surface_study_serial_overnight',started_utc=datetime.now(timezone.utc).isoformat(),
        phase='starting',settings=dict(spacing_mm=4,hz=720,iterations=64,smoothing=25,fps=30,resolution=[1280,960],samples=64),cases={})
    def save():
        for dest in (args.output/'overnight_status.json',args.desktop/'任务状态.json'):
            temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(dest)
    def run(command,log):
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def ps(arguments,log):run(['powershell.exe','-NoProfile','-Command',arguments+'; exit $LASTEXITCODE'],log)
    save()
    for case_id,title,duration in cases:
        case=args.output/case_id;case.mkdir()
        state=dict(phase='simulating',duration_s=duration)
        summary['cases'][case_id]=state
        summary['phase']=case_id;save()
        try:
            sim=case/'simulation';layout=args.layouts/case_id
            assisted=' --prewarm-damping 5' if case_id=='04_wave_reflection' else ''
            ps('& "Y:\\isaacsim\\python.bat" -u "'+windows_path(ROOT/'experiments/coupled_scenes/run_surface_study_probe.py')+
               '" --layout "'+windows_path(layout)+'" --output "'+windows_path(sim)+'" --initial-state "'+windows_path(args.initial_state)+
               f'" --action-case --settle --settle-min-seconds 1 --settle-max-seconds 8 --seconds {duration} --wall-limit 10800'+assisted,
               case/'simulation.log')
            report=json.loads((sim/'probe_report.json').read_text())
            if report['status']!='completed_settled_capture':raise RuntimeError('Simulation ended: '+report['status'])
            # Preserve the distinction between physically possible spill and hidden loss.
            state['maximum_outside_particles']=report.get('action_capture_outside_max',0)
            state['simulation_wall_seconds']=report['loop_wall_seconds']
            state['prewarm_seconds']=report['settling']['passed_at_seconds']
            capture=sim/'settled_capture'
            manifest=json.loads((capture/'manifest.json').read_text())
            frames=manifest['frames']
            if len(frames)!=duration*30+1:raise RuntimeError('Incomplete 30 Hz native capture')
            state['phase']='reconstructing';state['frames']=len(frames);save()
            surfaces=case/'surfaces';surfaces.mkdir()
            def build(frame):
                destination=surfaces/Path(frame['file']).stem
                run([sys.executable,str(ROOT/'experiments/coupled_scenes/reconstruct_surface_snapshot.py'),
                     str(capture/frame['file']),str(destination),'--mesh-smoothing-iters','25'],
                     surfaces/(Path(frame['file']).stem+'.log'))
                return dict(surface=str(Path(destination.name)/'water.obj'),recording_seconds=frame['recording_seconds'],
                            target_pose_isaac_m=frame['target_pose_isaac_m'])
            rebuilt=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for surface in pool.map(build,frames):
                    rebuilt.append(surface)
                    if len(rebuilt)%30==0:state['reconstructed_frames']=len(rebuilt);save()
            (surfaces/'sequence.json').write_text(json.dumps(dict(complete=True,action=manifest['action'],frames=rebuilt),indent=2))
            state['phase']='rendering';save()
            renders=case/'renders'
            ps('& "D:\\Program Files (x86)\\Blender\\blender.exe" --background --python-exit-code 1 --python "'+
               windows_path(ROOT/'experiments/coupled_scenes/render_surface_sequence.py')+'" -- --blend "'+
               windows_path(layout/(case_id+'.blend'))+'" --surfaces "'+windows_path(surfaces)+'" --output "'+windows_path(renders)+
               '" --samples 64 --width 1280 --height 960',case/'render.log')
            if len(list(renders.glob('frame_*.png')))!=len(frames):raise RuntimeError('Incomplete rendered sequence')
            def encode(name,seconds):
                video=case/(name+'.mp4')
                run(['ffmpeg','-hide_banner','-loglevel','warning','-n','-framerate','30','-i',str(renders/'frame_%04d.png'),
                     '-frames:v',str(seconds*30),'-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',
                     '-movflags','+faststart',str(video)],case/(name+'_encode.log'))
                info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                    'stream=width,height,nb_frames,duration,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
                assert int(info['nb_frames'])==seconds*30 and abs(float(info['duration'])-seconds)<.001
                dest=args.desktop/video.name
                if dest.exists():raise FileExistsError(dest)
                shutil.copy2(video,dest)
                return dict(video=str(dest),verification=info)
            state['outputs']=[encode(title,duration)]
            if case_id=='05_surface_recovery':
                state['outputs'].append(encode('03_中央波纹传播_前6秒',6))
            state['phase']='complete';save()
        except Exception as exc:
            state.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
            save()
            print(f'[case-failed] {case_id}: {exc}',flush=True)
    summary['phase']='complete' if all(c['phase']=='complete' for c in summary['cases'].values()) else 'finished_with_failures'
    summary['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
