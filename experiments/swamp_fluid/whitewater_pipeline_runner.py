"""Run a versioned whitewater DAG with fingerprints, receipts and recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


RUNNER_SCHEMA = 1
RUNNER_PRODUCT = "whitewater_v6_pipeline"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def strict_keys(payload, allowed, required, label):
    unknown = set(payload) - set(allowed)
    missing = set(required) - set(payload)
    if unknown or missing:
        raise ValueError(
            f"{label} keys differ: missing={sorted(missing)} unknown={sorted(unknown)}"
        )


def resolve_path(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def json_pointer(payload, pointer):
    if pointer in ("", "/"):
        return payload
    value = payload
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def path_fingerprint(path):
    path = Path(path)
    if path.is_file():
        return {"path": str(path), "kind": "file", "sha256": sha256_file(path)}
    if path.is_dir():
        manifest_candidates = (
            path / "manifest.json",
            path / "audit_report.json",
            path / "render_manifest.json",
        )
        manifests = [candidate for candidate in manifest_candidates if candidate.is_file()]
        if len(manifests) != 1:
            raise ValueError(
                f"Directory inputs must contain exactly one canonical manifest: {path}"
            )
        return {
            "path": str(path),
            "kind": "directory_manifest",
            "manifest": str(manifests[0]),
            "sha256": sha256_file(manifests[0]),
        }
    raise FileNotFoundError(path)


def validate_action(action, base):
    kind = action["kind"]
    if kind == "verify_file":
        path = resolve_path(action["path"], base)
        if not path.is_file() or path.stat().st_size <= int(action.get("minimum_bytes", 1)):
            raise RuntimeError(f"Artifact file gate failed: {path}")
        return {"path": str(path), "sha256": sha256_file(path)}
    if kind == "verify_json":
        path = resolve_path(action["path"], base)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "product" in action and payload.get("product") != action["product"]:
            raise RuntimeError(f"Unexpected product in {path}")
        pointer = action.get("success_pointer")
        if pointer is not None and json_pointer(payload, pointer) is not True:
            raise RuntimeError(f"JSON success gate {pointer} is not true: {path}")
        return {"path": str(path), "sha256": sha256_file(path)}
    raise ValueError(f"Action {kind!r} does not support artifact validation")


def substitute(value, values):
    if isinstance(value, str):
        return value.format_map(values)
    return value


def validate_configuration(payload):
    strict_keys(
        payload,
        {
            "schema", "product", "name", "run_directory", "dependencies",
            "stages", "metadata", "production_configuration",
        },
        {"schema", "product", "name", "run_directory", "dependencies", "stages"},
        "pipeline",
    )
    if payload["schema"] != RUNNER_SCHEMA or payload["product"] != RUNNER_PRODUCT:
        raise ValueError("Unsupported pipeline schema or product")
    if not isinstance(payload["name"], str) or not payload["name"].strip():
        raise ValueError("Pipeline name must be non-empty")
    stage_ids = []
    for index, stage in enumerate(payload["stages"]):
        strict_keys(
            stage,
            {"id", "depends_on", "inputs", "producers", "external", "action"},
            {"id", "depends_on", "inputs", "producers", "external", "action"},
            f"stages[{index}]",
        )
        identifier = stage["id"]
        if not isinstance(identifier, str) or not identifier or identifier in stage_ids:
            raise ValueError("Stage IDs must be unique non-empty strings")
        stage_ids.append(identifier)
        if not isinstance(stage["depends_on"], list):
            raise ValueError(f"{identifier}: depends_on must be a list")
        action = stage["action"]
        if not isinstance(action, dict) or action.get("kind") not in {
            "verify_file", "verify_json", "command", "production_command"
        }:
            raise ValueError(f"{identifier}: unsupported action")
        if action["kind"] == "command" and not isinstance(action.get("argv"), list):
            raise ValueError(f"{identifier}: command argv must be a list")
        if action["kind"] == "production_command":
            if not isinstance(action.get("command"), str) or not action["command"]:
                raise ValueError(f"{identifier}: production command must be named")
            output_index = action.get("output_argument_index")
            if output_index is not None and (
                not isinstance(output_index, int) or output_index < 0
            ):
                raise ValueError(f"{identifier}: invalid output_argument_index")
    known = set(stage_ids)
    seen = set()
    for stage in payload["stages"]:
        dependencies = set(stage["depends_on"])
        if dependencies - known:
            raise ValueError(f"{stage['id']}: unknown dependencies")
        if dependencies - seen:
            raise ValueError(
                f"{stage['id']}: stages must be listed in topological order"
            )
        seen.add(stage["id"])
    return payload


def load_resolved_production_configuration(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != 1
        or payload.get("product")
        != "whitewater_v6_resolved_production_configuration"
    ):
        raise ValueError(f"Unsupported resolved production configuration: {path}")
    expected = payload.get("configuration_sha256")
    canonical_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"configuration_sha256", "commands"}
    }
    actual = sha256_json(canonical_payload)
    if expected != actual:
        raise ValueError(f"Resolved production configuration hash mismatch: {path}")
    commands = payload.get("commands")
    if not isinstance(commands, dict) or not commands:
        raise ValueError("Resolved production configuration has no commands")
    if any(not isinstance(argv, list) or not argv for argv in commands.values()):
        raise ValueError("Resolved production commands must be non-empty argv lists")
    return {
        "path": path,
        "file_sha256": sha256_file(path),
        "configuration_sha256": expected,
        "payload": payload,
    }


def main():
    args = parse_args()
    configuration_path = args.configuration.resolve()
    base = configuration_path.parent
    configuration = validate_configuration(
        json.loads(configuration_path.read_text(encoding="utf-8"))
    )
    production_configuration = None
    if configuration.get("production_configuration") is not None:
        production_configuration = load_resolved_production_configuration(
            resolve_path(configuration["production_configuration"], base)
        )
    if any(
        stage["action"]["kind"] == "production_command"
        for stage in configuration["stages"]
    ) and production_configuration is None:
        raise ValueError("production_command stages require production_configuration")
    dependencies = []
    for item in configuration["dependencies"]:
        strict_keys(
            item,
            {"id", "path", "sha256", "kind"},
            {"id", "path", "kind"},
            "dependency",
        )
        path = resolve_path(item["path"], base)
        kind = item["kind"]
        if kind == "executable":
            if not path.is_file():
                raise FileNotFoundError(path)
        elif kind == "file":
            if not path.is_file():
                raise FileNotFoundError(path)
        else:
            raise ValueError(f"Unsupported dependency kind: {kind}")
        actual_hash = sha256_file(path)
        if item.get("sha256") is not None and actual_hash != item["sha256"]:
            raise RuntimeError(f"Dependency hash mismatch: {path}")
        dependencies.append(
            {"id": item["id"], "path": str(path), "kind": kind, "sha256": actual_hash}
        )

    run_directory = resolve_path(configuration["run_directory"], base)
    state_path = run_directory / "pipeline_state.json"
    stage_root = run_directory / "stages"
    state = {
        "schema": 1,
        "product": "whitewater_v6_pipeline_state",
        "pipeline": configuration["name"],
        "configuration": str(configuration_path),
        "configuration_sha256": sha256_file(configuration_path),
        "updated_utc": utc_now(),
        "complete": False,
        "dry_run": args.dry_run,
        "dependencies": dependencies,
        "production_configuration": (
            {
                "path": str(production_configuration["path"]),
                "file_sha256": production_configuration["file_sha256"],
                "configuration_sha256": production_configuration[
                    "configuration_sha256"
                ],
            }
            if production_configuration is not None
            else None
        ),
        "stages": [],
    }
    if not args.dry_run:
        atomic_json(state_path, state)

    completed = {}
    for stage in configuration["stages"]:
        input_fingerprints = [
            path_fingerprint(resolve_path(path, base)) for path in stage["inputs"]
        ]
        producer_fingerprints = [
            path_fingerprint(resolve_path(path, base)) for path in stage["producers"]
        ]
        fingerprint_payload = {
            "runner_schema": RUNNER_SCHEMA,
            "stage": stage,
            "inputs": input_fingerprints,
            "producers": producer_fingerprints,
            "dependencies": [completed[name]["fingerprint"] for name in stage["depends_on"]],
            "production_configuration": (
                {
                    "file_sha256": production_configuration["file_sha256"],
                    "configuration_sha256": production_configuration[
                        "configuration_sha256"
                    ],
                }
                if stage["action"]["kind"] == "production_command"
                else None
            ),
        }
        fingerprint = sha256_json(fingerprint_payload)
        cache_directory = stage_root / stage["id"] / fingerprint[:20]
        receipt_path = cache_directory / "stage_receipt.json"
        stage_state = {
            "id": stage["id"],
            "fingerprint": fingerprint,
            "cache_directory": str(cache_directory),
            "status": "planned" if args.dry_run else "running",
            "started_utc": utc_now(),
            "cached": False,
        }
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("fingerprint") == fingerprint and receipt.get("status") == "complete":
                stage_state.update(status="complete", cached=True, receipt=str(receipt_path))
                completed[stage["id"]] = stage_state
                state["stages"].append(stage_state)
                continue
            if receipt.get("status") == "failed" and not args.retry_failed:
                raise RuntimeError(
                    f"Stage {stage['id']} previously failed; pass --retry-failed after review"
                )
        if args.dry_run:
            completed[stage["id"]] = stage_state
            state["stages"].append(stage_state)
            continue
        cache_directory.mkdir(parents=True, exist_ok=True)
        # Mark older owned cache generations obsolete, but never mutate external artifacts.
        stage_parent = cache_directory.parent
        if not stage["external"]:
            for older in stage_parent.iterdir():
                if older.is_dir() and older != cache_directory and (older / "stage_receipt.json").is_file():
                    obsolete_path = older / "OBSOLETE.json"
                    if not obsolete_path.exists():
                        atomic_json(
                            obsolete_path,
                            {
                                "schema": 1,
                                "status": "obsolete_fingerprint_replaced",
                                "replacement_fingerprint": fingerprint,
                                "replacement_cache": str(cache_directory),
                                "data_policy": "Retained; never deleted or overwritten by the runner.",
                            },
                        )
        action = stage["action"]
        receipt = {
            "schema": 1,
            "product": "whitewater_v6_stage_receipt",
            "stage": stage["id"],
            "fingerprint": fingerprint,
            "started_utc": stage_state["started_utc"],
            "status": "running",
            "input_fingerprints": input_fingerprints,
            "producer_fingerprints": producer_fingerprints,
        }
        atomic_json(receipt_path, receipt)
        try:
            if action["kind"] in {"verify_file", "verify_json"}:
                result = validate_action(action, base)
            else:
                values = {
                    "stage_cache": str(cache_directory),
                    "config_dir": str(base),
                    "python": sys.executable,
                }
                values.update(
                    {
                        f"{name}_cache": str(completed[name]["cache_directory"])
                        for name in completed
                    }
                )
                if action["kind"] == "production_command":
                    command_name = action["command"]
                    commands = production_configuration["payload"]["commands"]
                    if command_name not in commands:
                        raise ValueError(
                            f"Unknown resolved production command: {command_name}"
                        )
                    raw_argv = list(commands[command_name])
                    output_index = action.get("output_argument_index")
                    if output_index is not None:
                        if output_index >= len(raw_argv):
                            raise ValueError("output_argument_index exceeds command argv")
                        raw_argv[output_index] = "{stage_cache}"
                else:
                    raw_argv = action["argv"]
                argv = [substitute(value, values) for value in raw_argv]
                cwd = resolve_path(action.get("cwd", str(base)), base)
                process = subprocess.run(
                    argv,
                    cwd=cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                (cache_directory / "command.log").write_text(
                    process.stdout, encoding="utf-8"
                )
                if process.returncode:
                    raise RuntimeError(
                        f"Command returned {process.returncode}: {argv[0]}"
                    )
                success = action.get("success")
                result = (
                    validate_action(
                        {
                            **success,
                            "path": substitute(success["path"], values),
                        },
                        base,
                    )
                    if success
                    else {"command_log": str(cache_directory / "command.log")}
                )
            receipt.update(status="complete", completed_utc=utc_now(), result=result)
            atomic_json(receipt_path, receipt)
            stage_state.update(
                status="complete", receipt=str(receipt_path), result=result
            )
        except Exception as error:
            receipt.update(
                status="failed",
                failed_utc=utc_now(),
                error={"type": type(error).__name__, "message": str(error)},
            )
            atomic_json(receipt_path, receipt)
            stage_state.update(status="failed", receipt=str(receipt_path), error=receipt["error"])
            state["stages"].append(stage_state)
            state.update(updated_utc=utc_now(), failed_stage=stage["id"])
            atomic_json(state_path, state)
            raise
        completed[stage["id"]] = stage_state
        state["stages"].append(stage_state)
        state["updated_utc"] = utc_now()
        atomic_json(state_path, state)

    state.update(complete=True, completed_utc=utc_now(), updated_utc=utc_now())
    if not args.dry_run:
        atomic_json(state_path, state)
    print(json.dumps({"complete": True, "dry_run": args.dry_run, "stages": state["stages"]}, indent=2))


if __name__ == "__main__":
    main()
