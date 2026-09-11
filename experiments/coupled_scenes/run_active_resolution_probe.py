"""Bounded piston 3 mm diagnostic: prepare on CPU, wait for GPU, no render."""
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

from run_pour_vorticity_compare import powershell
from build_cabinet_liquid_surfaces import atomic_json, windows_path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/coupled_scenes/active_piston_static_v5_3mm'
ASSETS=ROOT/'output/coupled_scenes/active_03_piston_push_assets_v3_3mm'
DESIGN=ROOT/'output/coupled_scenes/active_drive_contact_v4/03_piston_push'


def main():
    OUT.mkdir(exist_ok=False)
    state=dict(phase='preparing_assets',started_utc=datetime.now(timezone.utc).isoformat(),
               scope='3 mm piston, stationary 8s, no rendering; compare existing 4 mm pre-action cache')
    def save():atomic_json(OUT/'status.json',state)
    def run(command,log):
        with (OUT/log).open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{log}: exit {result.returncode}')
    save()
    try:
        run(powershell([r'D:\Program Files (x86)\Blender\blender.exe','--background','--factory-startup','--python',
            windows_path(ROOT/'experiments/coupled_scenes/prepare_active_apparatus.py'),'--',
            '--design',windows_path(DESIGN),'--output',windows_path(ASSETS),'--spacing','0.003']), 'prepare.log')
        meta=json.loads((ASSETS/'assets.json').read_text(encoding='utf-8'))
        state.update(phase='waiting_for_gpu',particle_count=meta['particle_count']);save()
        idle=0
        while idle<3:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            samples=[list(map(int,line.split(','))) for line in raw.strip().splitlines()]
            idle=idle+1 if all(m<1024 and u<20 for m,u in samples) else 0
            state['gpu_idle_check']=samples;save()
            if idle<3:time.sleep(10)
        state['phase']='simulating';save()
        command=powershell([r'Y:\isaacsim\python.bat',windows_path(ROOT/'experiments/coupled_scenes/run_active_pour_probe.py'),
            '--assets',windows_path(ASSETS),'--output',windows_path(OUT/'simulation'),
            '--stationary-resolution-probe','--recycle-piston-prewarm'])
        run(command,'simulation.log')
        report=json.loads((OUT/'simulation/probe_report.json').read_text())
        if report['status']!='completed':raise RuntimeError(report.get('error',report['status']))
        state.update(phase='complete',diagnostic_only=True,physics_gate_passed=False,
                     final_strict_still_water_passed=report['final_strict_still_water_passed'])
    except Exception as exc:
        state.update(phase='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__=='__main__':main()
