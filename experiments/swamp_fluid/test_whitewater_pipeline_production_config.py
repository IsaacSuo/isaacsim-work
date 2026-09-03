"""Gate direct execution of hash-locked production commands by the DAG runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_json(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


runner = Path(__file__).with_name("whitewater_pipeline_runner.py")
with tempfile.TemporaryDirectory(prefix="wwv6_production_runner_") as temporary:
    temporary = Path(temporary)
    producer = temporary / "surface_producer.py"
    producer.write_text(
        "import json,pathlib,sys\n"
        "out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)\n"
        "(out/'splashsurf_manifest.json').write_text(json.dumps({'schema':2,'state':{'complete':True}}),encoding='utf-8')\n",
        encoding="utf-8",
    )
    resolved_base = {
        "schema": 1,
        "product": "whitewater_v6_resolved_production_configuration",
        "name": "synthetic_artist_setup",
        "source_configuration": str(temporary / "source.production.json"),
        "source_configuration_sha256": "0" * 64,
        "paths": {},
        "physical": {},
        "splashsurf": {},
        "whitewater": {},
        "render": {},
        "manual_setup": {},
        "scene_contract": {},
        "flow_profile": {},
    }
    resolved = {
        **resolved_base,
        "configuration_sha256": sha256_json(resolved_base),
        "commands": {
            "splashsurf": [sys.executable, str(producer), "fixed-output-must-be-replaced"]
        },
    }
    resolved_path = temporary / "resolved.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    pipeline = {
        "schema": 1,
        "product": "whitewater_v6_pipeline",
        "name": "production_command_fixture",
        "run_directory": str(temporary / "run"),
        "production_configuration": str(resolved_path),
        "dependencies": [],
        "stages": [
            {
                "id": "surface_reconstruction",
                "depends_on": [],
                "inputs": [str(resolved_path)],
                "producers": [str(producer)],
                "external": False,
                "action": {
                    "kind": "production_command",
                    "command": "splashsurf",
                    "output_argument_index": 2,
                    "success": {
                        "kind": "verify_json",
                        "path": "{stage_cache}/splashsurf_manifest.json",
                        "success_pointer": "/state/complete",
                    },
                },
            }
        ],
    }
    pipeline_path = temporary / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(runner), str(pipeline_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    state = json.loads(
        (temporary / "run" / "pipeline_state.json").read_text(encoding="utf-8")
    )
    cache = Path(state["stages"][0]["cache_directory"])
    command_output_is_cache_owned = (cache / "splashsurf_manifest.json").is_file()
    fixed_output_was_not_used = not (temporary / "fixed-output-must-be-replaced").exists()

    tampered = json.loads(resolved_path.read_text(encoding="utf-8"))
    tampered["physical"] = {"particle_spacing_m": 999.0}
    tampered_path = temporary / "tampered_resolved.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    pipeline["production_configuration"] = str(tampered_path)
    tampered_pipeline = temporary / "tampered_pipeline.json"
    tampered_pipeline.write_text(json.dumps(pipeline), encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, str(runner), str(tampered_pipeline), "--dry-run"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

criteria = {
    "resolved_production_command_completes": completed.returncode == 0 and state["complete"],
    "output_is_redirected_to_fingerprinted_stage_cache": command_output_is_cache_owned,
    "fixed_resolver_output_is_not_mutated": fixed_output_was_not_used,
    "production_configuration_hash_is_in_state": (
        state["production_configuration"]["configuration_sha256"]
        == resolved["configuration_sha256"]
    ),
    "tampered_resolved_configuration_is_rejected": rejected.returncode != 0,
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_pipeline_production_configuration",
    "valid": all(criteria.values()),
    "criteria": criteria,
}
print(json.dumps(report, indent=2))
if not report["valid"]:
    print(completed.stdout)
    print(rejected.stdout)
    raise SystemExit(1)
