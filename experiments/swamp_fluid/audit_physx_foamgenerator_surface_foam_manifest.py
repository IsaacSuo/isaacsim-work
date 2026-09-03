"""Rehash every file and inventory in a surface-foam extension manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_file_record(record: dict, errors: list[str]) -> None:
    path = Path(record["path"])
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return
    if path.stat().st_size != record["bytes"]:
        errors.append(f"size mismatch: {path}")
    if sha256_file(path) != record["sha256"]:
        errors.append(f"SHA-256 mismatch: {path}")


def audit_file_set(record: dict, errors: list[str]) -> int:
    root = Path(record["root"])
    aggregate = hashlib.sha256()
    total = 0
    for item in record["files"]:
        path = root / item["relative_path"]
        if not path.is_file():
            errors.append(f"missing inventory member: {path}")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != item["bytes"]:
            errors.append(f"inventory size mismatch: {path}")
        if digest != item["sha256"]:
            errors.append(f"inventory SHA-256 mismatch: {path}")
        aggregate.update(item["relative_path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        total += size
    if len(record["files"]) != record["file_count"]:
        errors.append(f"inventory count mismatch: {root}")
    if total != record["total_bytes"]:
        errors.append(f"inventory byte mismatch: {root}")
    if aggregate.hexdigest() != record["inventory_sha256"]:
        errors.append(f"inventory aggregate mismatch: {root}")
    return len(record["files"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    individual = 0
    inventory_files = 0
    for section_name in ("audits", "product_manifests", "producer_code"):
        for record in manifest[section_name].values():
            audit_file_record(record, errors)
            individual += 1
    audit_file_record(manifest["inputs"]["root_closed_loop_manifest"], errors)
    individual += 1
    inventory_files += audit_file_set(
        manifest["inputs"]["splashsurf_surface_sequence"], errors
    )
    for record in manifest["products"].values():
        inventory_files += audit_file_set(record, errors)
    report = {
        "schema": "physx-foamgenerator-surface-foam-extension-audit/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": manifest.get("valid") is True and not errors,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "individual_files_audited": individual,
        "inventory_files_audited": inventory_files,
        "errors": errors,
    }
    output = args.report.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
