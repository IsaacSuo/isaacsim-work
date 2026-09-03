"""Assemble consecutive v6 liquid-field segments without copying field data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("segment_directories", type=Path, nargs="+")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output: {args.output_directory}"
        )

    segments = []
    for directory in args.segment_directories:
        manifest_path = directory / "manifest.json"
        manifest = load_json(manifest_path)
        if manifest.get("product") != "whitewater_v6_liquid_fields":
            raise ValueError(f"Unexpected segment product: {manifest_path}")
        if not manifest.get("state", {}).get("complete"):
            raise ValueError(f"Incomplete segment: {manifest_path}")
        samples = manifest.get("samples", [])
        if not samples:
            raise ValueError(f"Empty segment: {manifest_path}")
        segments.append((directory.resolve(), manifest_path.resolve(), manifest))

    reference = segments[0][2]
    for _, manifest_path, manifest in segments[1:]:
        for key in ("grid", "source", "terrain", "field_contract"):
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"Segment {key} mismatch: {manifest_path}")

    ordered = sorted(
        segments,
        key=lambda item: int(item[2]["samples"][0]["source_sample_index"]),
    )
    source_indices = []
    for _, manifest_path, manifest in ordered:
        indices = [int(row["source_sample_index"]) for row in manifest["samples"]]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"Non-consecutive samples inside {manifest_path}")
        source_indices.extend(indices)
    if source_indices != list(range(source_indices[0], source_indices[-1] + 1)):
        raise ValueError("Segments overlap or leave a source-sample gap")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    static_source = ordered[0][0] / reference["static_fields"]["file"]
    static_output = args.output_directory / "grid_and_static_fields.npz"
    os.link(static_source, static_output)

    assembled_samples = []
    for output_frame, source_index in enumerate(source_indices):
        owner = next(
            item
            for item in ordered
            if int(item[2]["samples"][0]["source_sample_index"])
            <= source_index
            <= int(item[2]["samples"][-1]["source_sample_index"])
        )
        source_directory, _, manifest = owner
        source_row = next(
            row
            for row in manifest["samples"]
            if int(row["source_sample_index"]) == source_index
        )
        source_path = source_directory / source_row["file"]
        output_name = f"liquid_fields_{output_frame:06d}.npz"
        output_path = args.output_directory / output_name
        if sha256_file(source_path) != source_row["sha256"]:
            raise ValueError(f"Segment sample hash mismatch: {source_path}")
        os.link(source_path, output_path)
        row = dict(source_row)
        row["file"] = output_name
        row["output_frame"] = output_frame
        assembled_samples.append(row)

    configuration = dict(reference["configuration"])
    configuration["start_sample"] = source_indices[0]
    configuration["end_sample"] = source_indices[-1]
    configuration["assembled_from_segments"] = True
    manifest = {
        **reference,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "method": "hard-linked consecutive audited-grid segments",
            "segments": [
                {
                    "directory": str(directory),
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                }
                for directory, manifest_path, _ in ordered
            ],
        },
        "configuration": configuration,
        "static_fields": {
            "file": static_output.name,
            "sha256": sha256_file(static_output),
            "bytes": static_output.stat().st_size,
        },
        "samples": assembled_samples,
        "state": {
            "complete": True,
            "completed_samples": len(assembled_samples),
            "expected_samples": len(assembled_samples),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    atomic_json(args.output_directory / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "complete": True,
                "samples": len(assembled_samples),
                "source_range": [source_indices[0], source_indices[-1]],
                "hard_links": len(assembled_samples) + 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
