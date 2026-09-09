"""WSL sequential pipeline: restore, revalidate, capture, reconstruct, render, MP4."""
import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
from pathlib import Path

from build_cabinet_liquid_surfaces import windows_path

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-state',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--blend',type=Path,required=True)
    parser.add_argument('--desktop-video',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    status_path=args.output/'pipeline_status.json'
    def status(phase,**extra):
        data=dict(phase=phase,**extra)
        status_path.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps(data),flush=True)
    def run(command,log):
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    try:
        status('simulating')
        sim=args.output/'simulation'
        # Explicitly quote Windows paths across the WSL/PowerShell boundary.
        command=['powershell.exe','-NoProfile','-Command',
            '& "Y:\\isaacsim\\python.bat" -u "'+windows_path(ROOT/'experiments/coupled_scenes/run_surface_study_probe.py')+
            '" --layout "'+windows_path(args.layout)+'" --output "'+windows_path(sim)+
            '" --initial-state "'+windows_path(args.initial_state)+
            '" --settle --settle-min-seconds 1 --settle-max-seconds 3 --seconds 1 --wall-limit 1800; exit $LASTEXITCODE']
        run(command,args.output/'simulation.log')
        report=json.loads((sim/'probe_report.json').read_text())
        if report['status']!='completed_settled_capture':
            raise RuntimeError(f'Revalidation/capture failed: {report["status"]}; no video made')
        capture=sim/'settled_capture'
        frames=json.loads((capture/'manifest.json').read_text())['frames']
        assert len(frames)==31 and abs(frames[-1]['recording_seconds']-1)<1e-8
        status('reconstructing',frames=len(frames))
        surfaces=args.output/'surfaces';surfaces.mkdir()
        def build(entry):
            destination=surfaces/Path(entry['file']).stem
            run([sys.executable,str(ROOT/'experiments/coupled_scenes/reconstruct_surface_snapshot.py'),
                 str(capture/entry['file']),str(destination),'--mesh-smoothing-iters','25'],
                 surfaces/(Path(entry['file']).stem+'.log'))
            return dict(surface=str(Path(destination.name)/'water.obj'),recording_seconds=entry['recording_seconds'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            surfaces_list=list(pool.map(build,frames))
        (surfaces/'sequence.json').write_text(json.dumps(dict(complete=True,frames=surfaces_list),indent=2))
        status('rendering',frames=len(frames))
        renders=args.output/'renders'
        run(['powershell.exe','-NoProfile','-Command',
             '& "D:\\Program Files (x86)\\Blender\\blender.exe" --background --python-exit-code 1 --python "'+
             windows_path(ROOT/'experiments/coupled_scenes/render_surface_sequence.py')+'" -- --blend "'+
             windows_path(args.blend)+'" --surfaces "'+windows_path(surfaces)+'" --output "'+windows_path(renders)+'"; exit $LASTEXITCODE'],
            args.output/'render.log')
        assert len(list(renders.glob('frame_*.png')))==31
        # The last snapshot is the t=1 endpoint; encode t=0..29/30 for exactly 1 s.
        video=args.output/'static_water_1s_smooth25.mp4'
        run(['ffmpeg','-hide_banner','-loglevel','warning','-n','-framerate','30','-i',str(renders/'frame_%04d.png'),
             '-frames:v','30','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
            args.output/'encode.log')
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,nb_frames,duration,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
        assert int(stream['nb_frames'])==30 and stream['r_frame_rate']=='30/1' and abs(float(stream['duration'])-1)<.001
        if args.desktop_video.exists():raise FileExistsError(args.desktop_video)
        shutil.copy2(video,args.desktop_video)
        status('complete',video=str(video),desktop=str(args.desktop_video),verification=stream)
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
