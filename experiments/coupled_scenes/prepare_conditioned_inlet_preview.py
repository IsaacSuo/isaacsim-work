"""Prepare the validated v90 inlet configuration; never launch simulation."""
import argparse
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]
PRESET = ROOT / 'configs/conditioned_inlet_v90.json'


def prepare(output, scenes_root, isaac_python, inlet_frames=60, post_frames=18):
    if inlet_frames < 1 or post_frames < 0:
        raise ValueError('inlet_frames must be positive and post_frames nonnegative')
    output = Path(output).resolve()
    replacements = {
        '${REPO}': str(ROOT),
        '${RUN_DIR}': str(output),
        '${SCENES_ROOT}': str(Path(scenes_root).resolve()),
        '${ISAAC_PYTHON}': str(Path(isaac_python).resolve()),
    }

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            for token, replacement in replacements.items():
                value = value.replace(token, replacement)
        return value

    preset = expand(json.loads(PRESET.read_text(encoding='utf-8')))
    event, bodies, run = preset['event'], preset['bodies'], preset['run']
    stop = event['source']['start_frame'] + inlet_frames - 1
    total = stop + post_frames
    event['source']['stop_frame'] = stop
    run['frames'] = total
    run['command'][run['command'].index('--frames') + 1] = str(total)
    run['suggested_capture']['physics_frame_stop'] = total
    run['render_environment'] = {
        'COUPLED_EVENT_CONFIG': str(output / 'glass_cabinet_pour.json'),
        'COUPLED_HIDE_UPSTREAM': '1',
        'RENDER_SCENE_NAME': 'warehouse',
    }
    run['notes'] = {
        'preset': str(PRESET),
        'nominal_birth_particles': 1260 * inlet_frames,
        'nominal_birth_litres': 1260 * inlet_frames * 0.004 ** 3 * 1000,
        'nominal_birth_litres_per_second': 4.8384,
        'validation': 'Local v90 validated 60 inlet frames plus 18 post frames only; longer runs require validation.',
        'assets': 'Scene, prebuilt collision USD, model and simulator must be provided on target; this tool does not install them.',
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in [('glass_cabinet_pour.json', event), ('bodies.json', bodies), ('run_command.json', run)]:
        (output / name).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scenes-root', required=True, type=Path)
    parser.add_argument('--isaac-python', required=True, type=Path)
    parser.add_argument('--inlet-frames', type=int, default=60)
    parser.add_argument('--post-frames', type=int, default=18)
    args = parser.parse_args()
    run = prepare(args.output, args.scenes_root, args.isaac_python, args.inlet_frames, args.post_frames)
    print(json.dumps(run['notes'], indent=2))
    print(shlex.join(run['command']))


if __name__ == '__main__':
    main()
