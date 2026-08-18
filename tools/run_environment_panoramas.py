"""Render static four-direction environment panoramas without running PhysX."""

import subprocess
import sys
from pathlib import Path


PREVIEW_RUNNER = Path(__file__).with_name("run_static_scene_previews.py")
ISAAC_PYTHON = Path(r"Y:\isaacsim\python.bat")


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [str(ISAAC_PYTHON), str(PREVIEW_RUNNER), "--panorama", *sys.argv[1:]],
            cwd=PREVIEW_RUNNER.parent.parent,
        )
    )
