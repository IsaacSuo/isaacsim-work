"""WSL: v90 pour with vorticity 10, strict audits, and matched baseline render.

Only the new simulation's vorticity changes. Reuse validated baseline physics
and meshes, but render both arms with the current renderer and identical camera.
Never overwrite a run or bypass failed physics gates.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from build_cabinet_liquid_surfaces import windows_path, atomic_json

ROOT = Path(__file__).resolve().parents[2]
FRAMES = list(range(122, 199, 2))


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def powershell(command, environment=None):
    prefix = ''.join('$env:' + k + '=' + ps_quote(v) + '; ' for k, v in (environment or {}).items())
    return ['powershell.exe', '-NoProfile', '-Command',
            prefix + '& ' + ' '.join(map(ps_quote, command)) + '; exit $LASTEXITCODE']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--desktop', required=True, type=Path)
    args = parser.parse_args()
    baseline, out, desktop = args.baseline.resolve(), args.output.resolve(), args.desktop.resolve()
    original_event = read(baseline / 'glass_cabinet_pour.json')
    original_run = read(baseline / 'run_command.json')
    assert original_event['source']['vorticity_confinement'] == .02
    assert original_run['frames'] == 198
    assert read(baseline / 'run_complete.json')['valid']
    assert read(baseline / 'tet_deformation_trajectory.json')['valid']
    assert read(baseline / 'primary_fluid/manifest.json')['state'] == dict(
        complete=True, valid=True, completed_frames=198, expected_frames=198)
    mesh_config = read(baseline / 'liquid_surfaces_30fps/surface_manifest.json')
    assert mesh_config['state']['valid'] and mesh_config['state']['complete']
    assert mesh_config['configuration']['mesh_smoothing_iters'] == 25
    assert mesh_config['configuration']['selected_frames'] == FRAMES
    assert not out.exists() and not desktop.exists()
    assert shutil.disk_usage(out.parent).free > 20 * 1024**3
    out.mkdir(); desktop.mkdir()
    status = dict(phase='preparing', started_utc=datetime.now(timezone.utc).isoformat(),
                  baseline=str(baseline), only_physics_change={'vorticity_confinement': [.02, 10.]},
                  nominal_birth_liters=4.8384, frames=198, video_frames=39, fps=30)

    def save():
        atomic_json(out / 'comparison_status.json', status)
        atomic_json(desktop / '任务状态.json', status)

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

    def idle():
        status['phase'] = 'waiting_for_gpu'; save()
        count = 0
        while count < 3:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip().splitlines()
            values = [list(map(int, row.split(','))) for row in rows]
            count = count + 1 if all(m < 1024 and u < 20 for m, u in values) else 0
            status['last_gpu'] = values; save()
            if count < 3:
                time.sleep(10)

    def verify(video, width):
        stream = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,duration', '-of', 'json', str(video)], text=True))['streams'][0]
        assert int(stream['nb_frames']) == 39 and stream['r_frame_rate'] == '30/1'
        assert (stream['width'], stream['height']) == (width, 960)
        assert abs(float(stream['duration']) - 1.3) < .001
        shutil.copy2(video, desktop / video.name)
        return stream

    save()
    try:
        event = copy.deepcopy(original_event)
        event['source']['vorticity_confinement'] = 10.
        reverted = copy.deepcopy(event); reverted['source']['vorticity_confinement'] = .02
        assert reverted == original_event
        command = [value.replace(windows_path(baseline), windows_path(out)) for value in original_run['command']]
        assert command[command.index('--substeps') + 1] == '12'
        atomic_json(out / 'glass_cabinet_pour.json', event)
        shutil.copy2(baseline / 'bodies.json', out / 'bodies.json')
        updated_run = copy.deepcopy(original_run); updated_run['command'] = command
        updated_run['vorticity_comparison_baseline'] = str(baseline)
        atomic_json(out / 'run_command.json', updated_run)
        # Preserve the exact source identity for later checks, without touching it.
        status['baseline_config_sha256'] = hashlib.sha256((baseline / 'glass_cabinet_pour.json').read_bytes()).hexdigest()
        idle(); status['phase'] = 'simulating'; save()

        def progress():
            manifest = out / 'primary_fluid/manifest.json'
            if manifest.exists():
                status['simulation_state'] = read(manifest).get('state'); save()

        run(powershell(command), out / 'simulation_console.log', progress)
        status['phase'] = 'auditing'; save()
        result = read(out / 'run_complete.json')
        fluid = read(out / 'primary_fluid/manifest.json')
        tet = read(out / 'tet_deformation_trajectory.json')
        assert result['valid'] and fluid['state']['valid'] and fluid['state']['complete'] and tet['valid']
        assert fluid['state']['completed_frames'] == 198
        with (out / 'soft_body_bounce_hero.usda').open() as stream:
            authored = next(line for line in stream if 'physxPBDMaterial:vorticityConfinement =' in line)
        assert float(authored.split('=')[-1]) == 10.
        status['authored_vorticity'] = 10.
        # Existing v90 audit checks per-ID birth/removal accounting and native tet motion.
        shutil.copy2(baseline / 'audit_preview.py', out / 'audit_preview.py')
        run([sys.executable, str(out / 'audit_preview.py')], out / 'audit.log')
        audit = read(out / 'preview_audit.json')
        status['audit'] = {k: v for k, v in audit.items() if k not in ('bodies', 'body_interaction')}
        assert audit['identity_and_mass_accounting_valid']
        status['phase'] = 'reconstructing'; save()
        run([sys.executable, str(ROOT / 'experiments/coupled_scenes/build_cabinet_liquid_surfaces.py'),
             str(out / 'primary_fluid'), str(out / 'liquid_surfaces_30fps'), '--frames', *map(str, FRAMES),
             '--mesh-smoothing-iters', '25'], out / 'reconstruction.log')
        videos = {}
        for key, source, title in [('vorticity10', out, '倾倒_涡量10'), ('baseline', baseline, '倾倒_原版0.02')]:
            idle(); status['phase'] = 'rendering_' + key; save()
            renders = out / ('renders_' + key)
            env = dict(COUPLED_EVENT_CONFIG=windows_path(source / 'glass_cabinet_pour.json'),
                       COUPLED_FLUID_SURFACE_DIR=windows_path(source / 'liquid_surfaces_30fps/surface'),
                       COUPLED_HIDE_UPSTREAM='1', RENDER_SCENE_NAME='warehouse')
            render_args = [r'D:\Program Files (x86)\Blender\blender.exe', r'Y:\scenes\warehouse.blend',
                '--background', '--python-exit-code', '1', '--python',
                windows_path(ROOT / 'tools/blender/render_blender_soft_body_cache.py'), '--',
                windows_path(source / 'mixed_bodies.usdc'), windows_path(renders), '198', '64', '960',
                ','.join(map(str, FRAMES)), 'false', '1.527948021888733', '0.44044965505599976',
                '-8.983591079711914', '0.012403637170791626', '-0.3595503568649292', '-8.108591079711914',
                'false', '1.75', '0.8', '35,155,275', '0', '0', '0', 'silicone_cloudy']

            def render_progress():
                status[key + '_rendered_frames'] = len(list(renders.glob('preview_*.png'))); save()

            run(powershell(render_args, env), out / (key + '_render.log'), render_progress)
            assert read(renders / 'blender_render_report.json')['valid']
            assert [int(p.stem.split('_')[-1]) for p in sorted(renders.glob('preview_*.png'))] == FRAMES
            video = out / (title + '.mp4')
            run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n', '-framerate', '30', '-pattern_type',
                 'glob', '-i', str(renders / 'preview_*.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt',
                 'yuv420p', '-movflags', '+faststart', str(video)], out / (key + '_encode.log'))
            status[key + '_verification'] = verify(video, 960); videos[key] = str(video); save()
        status['phase'] = 'encoding_comparison'; save()
        comparison = out / '倾倒对比_左原版_右涡量10.mp4'
        graph = ("[0:v]drawtext=text='Baseline 0.02':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65[l];"
                 "[1:v]drawtext=text='Vorticity 10':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.65[r];[l][r]hstack[v]")
        run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n', '-i', videos['baseline'], '-i', videos['vorticity10'],
             '-filter_complex', graph, '-map', '[v]', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', str(comparison)], out / 'compare_encode.log')
        status['comparison_verification'] = verify(comparison, 1920)
        status['comparison_video'] = str(desktop / comparison.name)
        status['phase'] = 'complete'
    except Exception as exc:
        status.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    main()
