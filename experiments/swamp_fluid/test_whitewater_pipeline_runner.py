"""Failure-injection gates for fingerprinting, cache reuse and recovery."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


runner = Path(__file__).with_name("whitewater_pipeline_runner.py")


def run(configuration, *extra):
    return subprocess.run(
        [sys.executable, str(runner), str(configuration), *extra],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


with tempfile.TemporaryDirectory(prefix="wwv6_runner_") as temporary:
    root = Path(temporary)
    input_path = root / "input.json"
    input_path.write_text('{"value": 1}\n', encoding="utf-8")
    producer = root / "producer.py"
    producer.write_text(
        "import json, pathlib, sys\n"
        "path=pathlib.Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text(json.dumps({'product':'runner_fixture','valid':True}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    configuration = root / "pipeline.json"
    payload = {
        "schema": 1,
        "product": "whitewater_v6_pipeline",
        "name": "runner_success_fixture",
        "run_directory": str(root / "run"),
        "dependencies": [],
        "stages": [
            {
                "id": "build",
                "depends_on": [],
                "inputs": [str(input_path)],
                "producers": [str(producer)],
                "external": False,
                "action": {
                    "kind": "command",
                    "argv": [sys.executable, str(producer), "{stage_cache}/result.json"],
                    "success": {
                        "kind": "verify_json",
                        "path": "{stage_cache}/result.json",
                        "product": "runner_fixture",
                        "success_pointer": "/valid",
                    },
                },
            }
        ],
    }
    configuration.write_text(json.dumps(payload), encoding="utf-8")
    first = run(configuration)
    first_state = json.loads((root / "run" / "pipeline_state.json").read_text(encoding="utf-8"))
    second = run(configuration)
    second_state = json.loads((root / "run" / "pipeline_state.json").read_text(encoding="utf-8"))
    first_cache = Path(first_state["stages"][0]["cache_directory"])
    cache_reused = second_state["stages"][0]["cached"] is True

    producer.write_text(producer.read_text(encoding="utf-8") + "# fingerprint change\n", encoding="utf-8")
    third = run(configuration)
    third_state = json.loads((root / "run" / "pipeline_state.json").read_text(encoding="utf-8"))
    third_cache = Path(third_state["stages"][0]["cache_directory"])
    fingerprint_changed = third_cache != first_cache
    obsolete_written = (first_cache / "OBSOLETE.json").is_file()

    failing = root / "fail_once.py"
    failing.write_text(
        "import json, pathlib, sys\n"
        "root=pathlib.Path(sys.argv[1]); sentinel=root/'attempted'\n"
        "\nif not sentinel.exists():\n sentinel.write_text('failed',encoding='utf-8'); raise SystemExit(7)\n"
        "(root/'result.json').write_text(json.dumps({'product':'retry_fixture','valid':True}),encoding='utf-8')\n",
        encoding="utf-8",
    )
    retry_configuration = root / "retry_pipeline.json"
    retry_payload = {
        "schema": 1,
        "product": "whitewater_v6_pipeline",
        "name": "runner_retry_fixture",
        "run_directory": str(root / "retry_run"),
        "dependencies": [],
        "stages": [
            {
                "id": "fail_once",
                "depends_on": [],
                "inputs": [str(input_path)],
                "producers": [str(failing)],
                "external": False,
                "action": {
                    "kind": "command",
                    "argv": [sys.executable, str(failing), "{stage_cache}"],
                    "success": {
                        "kind": "verify_json",
                        "path": "{stage_cache}/result.json",
                        "product": "retry_fixture",
                        "success_pointer": "/valid",
                    },
                },
            }
        ],
    }
    retry_configuration.write_text(json.dumps(retry_payload), encoding="utf-8")
    failed = run(retry_configuration)
    refused = run(retry_configuration)
    recovered = run(retry_configuration, "--retry-failed")
    retry_state = json.loads((root / "retry_run" / "pipeline_state.json").read_text(encoding="utf-8"))

criteria = {
    "initial_command_stage_completes": first.returncode == 0,
    "identical_fingerprint_reuses_cache": second.returncode == 0 and cache_reused,
    "producer_change_creates_new_cache": third.returncode == 0 and fingerprint_changed,
    "replaced_cache_is_marked_obsolete": obsolete_written,
    "injected_failure_is_recorded": failed.returncode != 0,
    "failed_stage_requires_explicit_retry": refused.returncode != 0,
    "explicit_retry_recovers_same_fingerprint": recovered.returncode == 0 and retry_state["complete"] is True,
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_pipeline_runner",
    "valid": all(criteria.values()),
    "criteria": criteria,
}
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
