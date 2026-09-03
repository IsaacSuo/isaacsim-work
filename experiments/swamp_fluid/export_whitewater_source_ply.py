"""Export selected audited 120 Hz whitewater source samples as binary PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--source-samples", nargs="+", type=int, required=True)
    parser.add_argument("--sample-stride", type=int, default=4)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_binary_ply(path, positions):
    positions = np.ascontiguousarray(positions, dtype="<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment audited PhysX PBD source positions\n"
        f"element vertex {len(positions)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        stream.write(positions.tobytes(order="C"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


args = parse_args()
source_directory = args.source_directory.resolve()
output_directory = args.output_directory.resolve()
if args.sample_stride < 1:
    raise ValueError("sample-stride must be positive")
requested = sorted(set(args.source_samples))
if not requested or any(sample < 0 for sample in requested):
    raise ValueError("source-samples must contain non-negative indices")
if any(sample % args.sample_stride for sample in requested):
    raise ValueError("Every source sample must align to sample-stride")
if output_directory.exists() and any(output_directory.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")
output_directory.mkdir(parents=True, exist_ok=True)

manifest_path = source_directory / "manifest.json"
audit_path = source_directory / "audit_report.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("valid") is not True:
    raise ValueError("Source cache has not passed its independent audit")
if audit.get("manifest_sha256") != sha256_file(manifest_path):
    raise ValueError("Source audit does not reference the current manifest")
sample_by_index = {
    int(item["sample_index"]): item for item in manifest["samples"]
}
missing = sorted(set(requested) - set(sample_by_index))
if missing:
    raise ValueError(f"Source cache is missing samples: {missing}")

output_samples = []
for source_sample in requested:
    item = sample_by_index[source_sample]
    source_path = source_directory / item["file"]
    if sha256_file(source_path) != item["sha256"]:
        raise ValueError(f"Source sample hash mismatch: {source_path}")
    with np.load(source_path, allow_pickle=False) as cache:
        positions = np.asarray(cache["positions"], dtype=np.float32)
        simulation_time = float(cache["simulation_time"])
    if positions.shape != (manifest["particle_count"], 3):
        raise ValueError(f"Unexpected position shape in {source_path}")
    if not np.isfinite(positions).all():
        raise ValueError(f"Non-finite position in {source_path}")
    render_frame = source_sample // args.sample_stride
    output_path = output_directory / f"particles_{render_frame:04d}.ply"
    atomic_binary_ply(output_path, positions)
    output_samples.append(
        {
            "source_sample_index": source_sample,
            "render_frame": render_frame,
            "simulation_time": simulation_time,
            "file": output_path.name,
            "vertices": len(positions),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "source_file": item["file"],
            "source_sha256": item["sha256"],
        }
    )
    print(f"Exported {output_path.name}: {len(positions):,} vertices")

output_manifest = {
    "schema": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "producer": {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    },
    "source": {
        "directory": str(source_directory),
        "manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": sha256_file(audit_path),
    },
    "configuration": {
        "source_samples": requested,
        "sample_stride": args.sample_stride,
        "format": "binary_little_endian float32 xyz",
    },
    "samples": output_samples,
    "complete": True,
}
atomic_json(output_directory / "export_manifest.json", output_manifest)
print(json.dumps({"complete": True, "samples": len(output_samples)}, indent=2))
