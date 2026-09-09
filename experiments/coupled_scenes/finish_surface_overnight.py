"""Render-only recovery of completed first-group caches; never runs physics/reconstruction."""
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from build_cabinet_liquid_surfaces import windows_path

ROOT = Path(__file__).resolve().parents[2]
CASES = [('02_sloshing', '02_容器晃动', 10),
         ('04_wave_reflection', '04_推波与反弹', 10),
         ('05_surface_recovery', '05_液面恢复', 12)]


def render_command(layout, surfaces, renders, *, resume=False, frame_limit=None):
    command = ('& "D:\\Program Files (x86)\\Blender\\blender.exe" --background --python-exit-code 1 --python "'
               + windows_path(ROOT / 'experiments/coupled_scenes/render_surface_sequence.py')
               + '" -- --blend "' + windows_path(layout)
               + '" --surfaces "' + windows_path(surfaces)
               + '" --output "' + windows_path(renders)
               + '" --samples 64 --width 1280 --height 960')
    if resume:
        command += ' --resume'
    if frame_limit is not None:
        command += f' --frame-limit {frame_limit}'
    return ['powershell.exe', '-NoProfile', '-Command', command + '; exit $LASTEXITCODE']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--desktop', type=Path, required=True)
    parser.add_argument('--layouts', type=Path, required=True)
    args = parser.parse_args()
    status = args.output / 'overnight_status.json'
    summary = json.loads(status.read_text())
    # Validate every saved frame before starting GPU work.
    for case_id, _, duration in CASES:
        case = args.output / case_id
        report = json.loads((case / 'simulation/probe_report.json').read_text())
        sequence = json.loads((case / 'surfaces/sequence.json').read_text())
        assert report['status'] == 'completed_settled_capture'
        assert sequence['complete'] and len(sequence['frames']) == duration * 30 + 1
        for i, frame in enumerate(sequence['frames']):
            assert abs(frame['recording_seconds'] - i / 30) < 1e-6
            assert (case / 'surfaces' / frame['surface']).is_file()
        assert (args.layouts / case_id / (case_id + '.blend')).is_file()
    retry_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    shutil.copy2(status, args.output / f'overnight_status_before_render_{retry_id}.json')
    summary.pop('finished_utc', None)
    summary['render_recovery_started_utc'] = datetime.now(timezone.utc).isoformat()

    def save():
        for dest in (status, args.desktop / '任务状态.json'):
            temp = dest.with_suffix('.tmp')
            temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(dest)

    def run(command, log, progress=None):
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if progress:
                    progress()
                time.sleep(10)
            if progress:
                progress()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)

    def verify_video(video, seconds):
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=width,height,nb_frames,duration,r_frame_rate', '-of', 'json', str(video)
        ], text=True))['streams'][0]
        assert int(info['nb_frames']) == seconds * 30
        assert abs(float(info['duration']) - seconds) < .001
        assert (info['width'], info['height'], info['r_frame_rate']) == (1280, 960, '30/1')
        return info

    for case_id, title, duration in CASES:
        case = args.output / case_id
        state = summary['cases'][case_id]
        if state['phase'] == 'complete':
            continue
        if 'error' in state:
            state.setdefault('previous_errors', []).append(state.pop('error'))
        summary['phase'] = case_id
        state.update(phase='rendering', reconstructed_frames=duration * 30 + 1)
        save()
        renders = case / 'renders'

        def progress():
            checkpoint = renders / 'render_manifest.json'
            if checkpoint.exists():
                data = json.loads(checkpoint.read_text())
                state['rendered_frames'] = len(data['frames'])
                save()

        try:
            run(render_command(args.layouts / case_id / (case_id + '.blend'),
                               case / 'surfaces', renders, resume=renders.exists()),
                case / f'render_recovery_{retry_id}.log', progress)
            rendered = json.loads((renders / 'render_manifest.json').read_text())
            assert rendered['complete'] and len(rendered['frames']) == duration * 30 + 1
            state['phase'] = 'encoding'
            save()
            outputs = [(title, duration)]
            if case_id == '05_surface_recovery':
                outputs.append(('03_中央波纹传播_前6秒', 6))
            state['outputs'] = []
            for name, seconds in outputs:
                video = case / (name + '.mp4')
                if not video.exists():
                    run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n', '-framerate', '30',
                         '-i', str(renders / 'frame_%04d.png'), '-frames:v', str(seconds * 30),
                         '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                         '-movflags', '+faststart', str(video)], case / f'{name}_encode_{retry_id}.log')
                info = verify_video(video, seconds)
                dest = args.desktop / video.name
                if dest.exists():
                    raise FileExistsError(f'Refusing to overwrite desktop video: {dest}')
                shutil.copy2(video, dest)
                state['outputs'].append(dict(video=str(dest), verification=info))
            state['phase'] = 'complete'
        except Exception as exc:
            state.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
            print(f'[case-failed] {case_id}: {exc}', flush=True)
        save()
    summary['phase'] = ('complete' if all(c['phase'] == 'complete' for c in summary['cases'].values())
                        else 'finished_with_failures')
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
