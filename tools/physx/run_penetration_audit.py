"""Run the Blender penetration auditor with stable batch exit codes.

Exit codes:
  0: the audit is valid and passes every configured threshold
  1: Blender, the audit tool, or the output report is invalid
  2: the audit completed successfully and found a physical failure
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDITOR = ROOT / "tools" / "blender" / "audit_simulation_penetration.py"


def default_blender() -> Path:
    configured = os.environ.get("BLENDER_BIN")
    if configured:
        return Path(configured).expanduser()
    discovered = shutil.which("blender")
    if discovered:
        return Path(discovered)
    return Path(r"D:\Program Files (x86)\Blender\blender.exe")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("debug_usd", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--blender", type=Path, default=default_blender())
    parser.add_argument("--debug-report", type=Path, default=None)
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--max-inverted-tets", type=int, default=0)
    parser.add_argument("--max-non-manifold-edges", type=int, default=0)
    parser.add_argument("--support-y", type=float, default=None)
    parser.add_argument("--ground-tolerance-mm", type=float, default=2.0)
    return parser.parse_args()


def auditor_arguments(args) -> list[str]:
    command = [
        str(args.blender),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(AUDITOR),
        "--",
        str(args.debug_usd),
        str(args.output_json),
        "--penetration-tolerance-mm",
        str(args.penetration_tolerance_mm),
        "--max-inverted-tets",
        str(args.max_inverted_tets),
        "--max-non-manifold-edges",
        str(args.max_non_manifold_edges),
        "--ground-tolerance-mm",
        str(args.ground_tolerance_mm),
    ]
    if args.debug_report is not None:
        command.extend(("--debug-report", str(args.debug_report)))
    if args.support_y is not None:
        command.extend(("--support-y", str(args.support_y)))
    return command


def read_report(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def main() -> int:
    args = parse_args()
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        output_json.unlink()

    try:
        completed = subprocess.run(
            auditor_arguments(args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except OSError as error:
        print(f"[penetration-audit] unable to start Blender: {error}", file=sys.stderr)
        return 1

    report = read_report(output_json)
    if report is None or report.get("valid") is not True:
        if completed.stdout:
            print(completed.stdout.rstrip(), file=sys.stderr)
        print(
            f"[penetration-audit] invalid report={output_json} "
            f"blender_exit={completed.returncode}",
            file=sys.stderr,
        )
        return 1
    if report.get("passed") is not True:
        print(
            f"[penetration-audit] rejected failures={len(report.get('failures', []))} "
            f"report={output_json}"
        )
        return 2
    if completed.returncode != 0:
        print(
            f"[penetration-audit] Blender exited unexpectedly after a passing report: "
            f"{completed.returncode}",
            file=sys.stderr,
        )
        return 1
    print(f"[penetration-audit] passed report={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
