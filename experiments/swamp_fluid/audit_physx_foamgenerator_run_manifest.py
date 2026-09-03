"""Re-hash and validate a PhysX/FoamGenerator closed-loop run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_file(record: dict) -> None:
    path = Path(record["path"])
    require(path.is_file(), f"Missing file: {path}")
    require(path.stat().st_size == record["bytes"], f"Size mismatch: {path}")
    require(sha256_file(path) == record["sha256"], f"SHA-256 mismatch: {path}")


def audit_file_set(record: dict) -> None:
    root = Path(record["root"])
    aggregate = hashlib.sha256()
    total_bytes = 0
    for item in record["files"]:
        path = root / item["relative_path"]
        require(path.is_file(), f"Missing file-set member: {path}")
        require(path.stat().st_size == item["bytes"], f"Size mismatch: {path}")
        require(sha256_file(path) == item["sha256"], f"SHA-256 mismatch: {path}")
        aggregate.update(item["relative_path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(item["bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(item["sha256"].encode("ascii"))
        aggregate.update(b"\n")
        total_bytes += item["bytes"]
    require(len(record["files"]) == record["file_count"], f"File count mismatch: {root}")
    require(total_bytes == record["total_bytes"], f"Byte count mismatch: {root}")
    require(aggregate.hexdigest() == record["inventory_sha256"], f"Inventory mismatch: {root}")
    absent = record.get("absent_zero_particle_files", [])
    for item in absent:
        require(item.get("expected_particles") == 0, f"Invalid absent-file contract: {root}")
        require(not (root / item["relative_path"]).exists(), f"Expected absent zero-particle file exists: {root / item['relative_path']}")
    if "logical_record_count" in record:
        require(
            record["file_count"] + len(absent) == record["logical_record_count"],
            f"Logical record count mismatch: {root}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()
    require(args.manifest.is_file(), f"Missing manifest: {args.manifest}")
    with args.manifest.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    require(manifest.get("schema") == "physx-foamgenerator-closed-loop-run/v1", "Wrong schema")
    require(manifest.get("valid") is True, "Manifest is not valid")
    require(manifest["execution_contract"]["finite_difference_velocity_used"] is False, "Finite differences declared")
    require(manifest["execution_contract"]["warp_device"].startswith("cuda:"), "CUDA not declared")
    for record in manifest["audit_results"].values():
        audit_file(record)
    for record in manifest["inputs"].values():
        if "files" in record:
            audit_file_set(record)
        else:
            audit_file(record)
    for record in manifest["outputs"].values():
        audit_file_set(record)
    for record in manifest["manifests"].values():
        audit_file(record)
    for record in manifest["producer_code"].values():
        audit_file(record)
    executable = manifest.get("software", {}).get("foamgenerator_executable")
    if executable is not None:
        audit_file(executable)
    report = {
        "schema": "physx-foamgenerator-closed-loop-run-audit/v1",
        "valid": True,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "input_file_count": sum(record.get("file_count", 1) for record in manifest["inputs"].values()),
        "output_file_count": sum(record["file_count"] for record in manifest["outputs"].values()),
        "producer_file_count": len(manifest["producer_code"]),
    }
    require(not args.output_report.exists(), f"Refusing to overwrite: {args.output_report}")
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(args.output_report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
