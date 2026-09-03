"""Build volume-conservative render payloads for hysteresis-pending liquid."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PENDING_STATE = np.uint8(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def proxy_dtype():
    return np.dtype(
        [
            ("particle_id", "<u8"),
            ("source_sample", "<i4"),
            ("position", "<f4", (3,)),
            ("velocity", "<f4", (3,)),
            ("orientation", "<f4", (3,)),
            ("shape_semiaxes", "<f4", (3,)),
            ("liquid_volume", "<f8"),
        ]
    )


def main():
    args = parse_args()
    domain_directory = args.domain_directory.resolve()
    output_directory = args.output_directory.resolve()
    domain_manifest_path = domain_directory / "domain_manifest.json"
    domain_manifest = json.loads(domain_manifest_path.read_text(encoding="utf-8"))
    if domain_manifest.get("schema") != 2 or not domain_manifest.get("valid"):
        raise RuntimeError("Pending proxies require a valid schema-2 domain")

    output_manifest_path = output_directory / "manifest.json"
    if output_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_manifest_path}")
    output_directory.mkdir(parents=True, exist_ok=True)

    source_directory = Path(domain_manifest["source"]["directory"])
    source_manifest_path = Path(domain_manifest["source"]["manifest"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_by_index = {
        int(sample["sample_index"]): sample for sample in source_manifest["samples"]
    }
    particle_spacing = float(domain_manifest["configuration"]["particle_spacing"])
    particle_volume = particle_spacing**3
    equivalent_radius = float((3.0 * particle_volume / (4.0 * np.pi)) ** (1.0 / 3.0))
    samples = domain_manifest["samples"]
    sample_times = np.asarray([float(sample["simulation_time"]) for sample in samples])
    positive_steps = np.diff(sample_times)
    reference_dt = float(np.median(positive_steps)) if len(positive_steps) else 0.0

    manifest = {
        "schema": 1,
        "product": "whitewater_v6_pending_liquid_proxy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "configuration": {
            "particle_spacing_m": particle_spacing,
            "liquid_volume_per_particle_m3": particle_volume,
            "equivalent_sphere_radius_m": equivalent_radius,
            "maximum_ballistic_stretch": 2.5,
            "shape_policy": (
                "velocity-aligned ellipsoid; semiaxis product is fixed to the "
                "equal-volume sphere, so visual stretching cannot create liquid"
            ),
            "ownership_policy": (
                "state 1 only; a pending proxy is mutually exclusive with core "
                "Splashsurf ownership and persistent secondary marker birth"
            ),
        },
        "inputs": {
            "domain_manifest": str(domain_manifest_path),
            "domain_manifest_sha256": sha256_file(domain_manifest_path),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
        },
        "record_contract": {
            "dtype": proxy_dtype().descr,
            "coordinates": "Isaac/USD world XYZ in metres",
            "particle_id": "implicit PhysX source array index",
            "liquid_volume": "owned physical liquid; additive exactly once",
        },
        "samples": [],
    }
    atomic_json(output_manifest_path, manifest)

    cumulative_rows = 0
    maximum_rows = 0
    for output_frame, sample in enumerate(samples):
        source_sample = int(sample["sample_index"])
        classification_path = domain_directory / sample["classification"]["file"]
        source_entry = source_by_index[source_sample]
        source_path = source_directory / source_entry["file"]
        if sha256_file(classification_path) != sample["classification"]["sha256"]:
            raise RuntimeError(f"Classification hash mismatch: {classification_path}")
        if sha256_file(source_path) != sample["source_sha256"]:
            raise RuntimeError(f"Source hash mismatch: {source_path}")

        with np.load(classification_path, allow_pickle=False) as cache:
            state = np.asarray(cache["particle_state"], dtype=np.uint8)
            handoff = np.asarray(cache["new_secondary_handoff"], dtype=np.uint8)
        pending_indices = np.flatnonzero(state == PENDING_STATE)
        if np.any(handoff[pending_indices]):
            raise RuntimeError("A pending proxy cannot be a confirmed handoff")
        with np.load(source_path, allow_pickle=False) as cache:
            positions = np.asarray(cache["positions"], dtype=np.float32)[pending_indices]
            velocities = np.asarray(cache["velocities"], dtype=np.float32)[pending_indices]

        records = np.empty(len(pending_indices), dtype=proxy_dtype())
        records["particle_id"] = pending_indices.astype(np.uint64)
        records["source_sample"] = source_sample
        records["position"] = positions
        records["velocity"] = velocities
        speed = np.linalg.norm(velocities.astype(np.float64), axis=1)
        orientation = np.zeros((len(records), 3), dtype=np.float64)
        orientation[:, 1] = 1.0
        moving = speed > 1.0e-8
        orientation[moving] = velocities[moving] / speed[moving, None]
        records["orientation"] = orientation.astype(np.float32)

        stretch = np.clip(
            1.0 + 0.5 * speed * reference_dt / equivalent_radius,
            1.0,
            manifest["configuration"]["maximum_ballistic_stretch"],
        )
        axes = np.empty((len(records), 3), dtype=np.float64)
        axes[:, 0] = equivalent_radius * np.power(stretch, 2.0 / 3.0)
        axes[:, 1] = equivalent_radius / np.power(stretch, 1.0 / 3.0)
        axes[:, 2] = axes[:, 1]
        records["shape_semiaxes"] = axes.astype(np.float32)
        records["liquid_volume"] = particle_volume

        output_path = output_directory / f"pending_liquid_proxy_{output_frame:06d}.npz"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        with output_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.int32(1),
                output_frame=np.int32(output_frame),
                source_sample_index=np.int32(source_sample),
                simulation_time=np.float64(sample["simulation_time"]),
                proxies=records,
            )
        frame_volume = float(records["liquid_volume"].sum(dtype=np.float64))
        entry = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "output_frame": output_frame,
            "source_sample_index": source_sample,
            "simulation_time": float(sample["simulation_time"]),
            "proxy_count": len(records),
            "owned_liquid_volume_m3": frame_volume,
            "particle_indices_sha256": hashlib.sha256(
                pending_indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "classification_file": sample["classification"]["file"],
            "classification_sha256": sample["classification"]["sha256"],
            "source_file": source_entry["file"],
            "source_sha256": sample["source_sha256"],
        }
        manifest["samples"].append(entry)
        cumulative_rows += len(records)
        maximum_rows = max(maximum_rows, len(records))
        atomic_json(output_manifest_path, manifest)
        print(
            f"[v6-pending-proxy] source={source_sample:04d} "
            f"count={len(records)} volume={frame_volume:.9e}",
            flush=True,
        )

    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["cumulative"] = {
        "sample_rows": cumulative_rows,
        "maximum_simultaneous_proxies": maximum_rows,
        "maximum_simultaneous_owned_liquid_volume_m3": maximum_rows
        * particle_volume,
    }
    atomic_json(output_manifest_path, manifest)
    print(json.dumps(manifest["cumulative"], indent=2))


if __name__ == "__main__":
    main()
