"""Build deterministic v6 whitewater marker-birth events from audited fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.marker_birth import (
    BIRTH_DTYPE,
    CHANNEL_NAMES,
    BirthChannel,
    GridEmissionReservoir,
    MarkerBirthModel,
    expected_marker_budget,
    nodal_control_volumes,
    realize_marker_births,
)
from whitewater.flow_profiles import FlowProfile, sha256_file as profile_sha256_file
from whitewater.domain_partition import (
    load_domain_partition_contract,
    load_particle_classification,
)
from whitewater.secondary_handoff import (
    EXTERNAL_SOURCE_NAMESPACE,
    handoff_spray_births,
)


SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("liquid_field_directory", type=Path)
    parser.add_argument("emission_field_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", type=int, default=284729)
    parser.add_argument("--flow-profile", type=Path)
    parser.add_argument("--domain-manifest", type=Path)
    parser.add_argument("--domain-body-id")
    parser.add_argument("--spray-rate-density", type=float)
    parser.add_argument("--air-rate-density", type=float)
    parser.add_argument("--churn-rate-density", type=float)
    parser.add_argument("--maximum-position-attempts", type=int, default=4)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def validated_input_manifests(liquid_directory, emission_directory):
    liquid_path = liquid_directory / "manifest.json"
    emission_path = emission_directory / "manifest.json"
    liquid = load_json(liquid_path)
    emission = load_json(emission_path)
    if liquid.get("product") != "whitewater_v6_liquid_fields":
        raise RuntimeError(f"Unexpected liquid product in {liquid_path}")
    if emission.get("product") != "whitewater_v6_emission_fields":
        raise RuntimeError(f"Unexpected emission product in {emission_path}")
    if liquid.get("grid") != emission.get("grid"):
        raise RuntimeError("Liquid and emission grids differ")
    liquid_samples = liquid.get("samples", [])
    emission_samples = emission.get("samples", [])
    if not liquid_samples or len(liquid_samples) != len(emission_samples):
        raise RuntimeError("Liquid and emission manifests have different sample counts")
    for liquid_sample, emission_sample in zip(liquid_samples, emission_samples):
        keys = ("output_frame", "source_sample_index", "simulation_time")
        for key in keys:
            if liquid_sample.get(key) != emission_sample.get(key):
                raise RuntimeError(f"Input sample mismatch for {key}")
        if not (liquid_directory / liquid_sample["file"]).is_file():
            raise FileNotFoundError(liquid_directory / liquid_sample["file"])
        if not (emission_directory / emission_sample["file"]).is_file():
            raise FileNotFoundError(emission_directory / emission_sample["file"])
    return liquid_path, emission_path, liquid, emission


def channel_statistics(records, channel, expected_total, realization):
    selected = records[records["channel"] == np.uint8(channel)]
    classes = {
        str(value): int(np.count_nonzero(selected["render_class"] == value))
        for value in range(3)
    }
    return {
        "expected_births": float(expected_total),
        "candidate_births": int(realization["candidate_count"]),
        "accepted_births": int(realization["accepted_count"]),
        "rejected_births": int(realization["rejected_count"]),
        "rejection_reasons": {
            "nonfinite_or_outside_grid": int(realization["rejected_nonfinite"]),
            "wrong_liquid_phase": int(realization["rejected_phase"]),
            "solid_clearance": int(realization["rejected_collision"]),
        },
        "phase_volume_m3": float(selected["phase_volume"].sum(dtype=np.float64)),
        "radius_quantiles_m": (
            np.quantile(
                selected["physical_radius"].astype(np.float64),
                (0.0, 0.5, 0.9, 0.99, 1.0),
            ).tolist()
            if len(selected)
            else [0.0] * 5
        ),
        "render_class_counts": classes,
    }


def main():
    args = parse_args()
    (
        liquid_manifest_path,
        emission_manifest_path,
        liquid_manifest,
        emission_manifest,
    ) = validated_input_manifests(
        args.liquid_field_directory, args.emission_field_directory
    )
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty marker directory: {args.output_directory}"
        )

    grid = liquid_manifest["grid"]
    spec = GridSpec(
        tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"])
    )
    domain_contract = None
    domain_source_manifest = None
    domain_source_directory = None
    declared_domain = liquid_manifest.get("domain_partition")
    domain_manifest_path = (
        args.domain_manifest.resolve()
        if args.domain_manifest is not None
        else (
            Path(declared_domain["manifest"]).resolve()
            if declared_domain is not None
            else None
        )
    )
    if args.domain_body_id is not None and domain_manifest_path is None:
        raise RuntimeError("--domain-body-id requires a domain manifest")
    if domain_manifest_path is not None:
        raw_domain = load_json(domain_manifest_path)
        domain_source_manifest_path = Path(raw_domain["source"]["manifest"])
        domain_source_manifest = load_json(domain_source_manifest_path)
        domain_contract = load_domain_partition_contract(
            domain_manifest_path,
            source_manifest_path=domain_source_manifest_path,
            source_manifest=domain_source_manifest,
            requested_spacing=raw_domain["configuration"]["domain_spacing"],
            body_id=(
                args.domain_body_id
                if args.domain_body_id is not None
                else declared_domain.get("body_id") if declared_domain else None
            ),
        )
        if (
            declared_domain is None
            or declared_domain.get("manifest_sha256")
            != domain_contract["manifest_sha256"]
            or declared_domain.get("body_id") != domain_contract["body"]["body_id"]
        ):
            raise RuntimeError("Liquid field and marker-birth domain ownership differ")
        if spec.cell_count >= (1 << 29):
            raise RuntimeError("Grid node IDs overlap the external particle namespace")
        domain_source_directory = Path(raw_domain["source"]["directory"])
        particle_spacing = float(raw_domain["configuration"]["particle_spacing"])
    sample_rate = float(
        liquid_manifest["configuration"]["whitewater_sample_rate_hz"]
    )
    sample_stride = int(liquid_manifest["configuration"].get("sample_stride", 1))
    dt = sample_stride / sample_rate
    profile = None
    profile_path = args.flow_profile.resolve() if args.flow_profile is not None else None
    if profile_path is not None:
        profile = FlowProfile.load(profile_path)
        emission_profile = emission_manifest.get("flow_profile", {}).get("resolved")
        if emission_profile is not None and profile.metadata() != emission_profile:
            raise RuntimeError("Marker-birth flow profile differs from emission profile")
    elif emission_manifest.get("flow_profile", {}).get("resolved") is not None:
        profile = FlowProfile.from_mapping(
            emission_manifest["flow_profile"]["resolved"]
        )
    profile_rates = profile.birth_rate_density if profile is not None else {
        "spray": 1.20e6,
        "entrained_air": 1.60e6,
        "churn": 5.00e6,
    }
    spray_rate_density = (
        profile_rates["spray"]
        if args.spray_rate_density is None
        else args.spray_rate_density
    )
    air_rate_density = (
        profile_rates["entrained_air"]
        if args.air_rate_density is None
        else args.air_rate_density
    )
    churn_rate_density = (
        profile_rates["churn"]
        if args.churn_rate_density is None
        else args.churn_rate_density
    )
    model = MarkerBirthModel(
        spacing=spec.spacing,
        spray_rate_density=spray_rate_density,
        entrained_air_rate_density=air_rate_density,
        churn_rate_density=churn_rate_density,
        maximum_position_attempts=args.maximum_position_attempts,
    )
    control_volumes = nodal_control_volumes(spec)
    reservoirs = {
        channel: GridEmissionReservoir(spec.cell_count, channel, args.seed)
        for channel in BirthChannel
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_manifest_path = args.output_directory / "manifest.json"
    marker_module_path = Path(__file__).parent / "whitewater" / "marker_birth.py"
    manifest = {
        "schema": SCHEMA,
        "product": "whitewater_v6_marker_births",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": spec.metadata(),
        "configuration": {
            "seed": args.seed,
            "sample_rate_hz": sample_rate,
            "sample_stride": sample_stride,
            "dt_seconds": dt,
            "control_volume_quadrature": "node_centered_trapezoidal",
            "birth_interval": "[simulation_time, simulation_time + dt]",
            "model": model.metadata(),
            "flow_profile": (
                {
                    "mode": "file" if profile_path is not None else "inherited_from_emission",
                    "path": str(profile_path) if profile_path is not None else None,
                    "sha256": (
                        profile_sha256_file(profile_path)
                        if profile_path is not None
                        else emission_manifest.get("flow_profile", {}).get("sha256")
                    ),
                    "resolved": profile.metadata() if profile is not None else None,
                }
            ),
            "external_liquid_handoff": (
                {
                    "enabled": True,
                    "domain_manifest": str(domain_contract["manifest_path"]),
                    "domain_manifest_sha256": domain_contract["manifest_sha256"],
                    "body_id": domain_contract["body"]["body_id"],
                    "particle_spacing": particle_spacing,
                    "phase_volume_per_particle_m3": particle_spacing**3,
                    "handoff_event": "new_secondary_handoff == 1",
                }
                if domain_contract is not None
                else {"enabled": False}
            ),
        },
        "inputs": {
            "liquid_manifest": str(liquid_manifest_path.resolve()),
            "liquid_manifest_sha256": sha256_file(liquid_manifest_path),
            "emission_manifest": str(emission_manifest_path.resolve()),
            "emission_manifest_sha256": sha256_file(emission_manifest_path),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "birth_module": str(marker_module_path.resolve()),
            "birth_module_sha256": sha256_file(marker_module_path),
        },
        "record_contract": {
            "dtype": BIRTH_DTYPE.descr,
            "id_layout": "channel[63:62] | source_node_id[61:32] | emission_index[31:0]",
            "position_space": "Isaac world XYZ metres",
            "physical_radius_units": "metres",
            "phase_volume_definition": "4/3*pi*physical_radius^3*representative_count",
            "external_source_namespace_bit": int(EXTERNAL_SOURCE_NAMESPACE),
            "external_handoff_semantics": (
                "one liquid spray birth per one-shot domain ownership transfer; "
                "source_node_id bit 29 distinguishes PhysX particle IDs from grid nodes"
            ),
        },
        "samples": [],
        "complete": False,
    }
    atomic_json(output_manifest_path, manifest)

    cumulative = {
        CHANNEL_NAMES[channel]: {
            "expected_births": 0.0,
            "candidate_births": 0,
            "accepted_births": 0,
            "rejected_births": 0,
            "phase_volume_m3": 0.0,
        }
        for channel in BirthChannel
    }
    cumulative_external = {
        "births": 0,
        "liquid_phase_volume_m3": 0.0,
        "source_particle_indices_sha256_by_sample": [],
    }
    paired_samples = zip(liquid_manifest["samples"], emission_manifest["samples"])
    for output_index, (liquid_sample, emission_sample) in enumerate(paired_samples):
        liquid_path = args.liquid_field_directory / liquid_sample["file"]
        emission_path = args.emission_field_directory / emission_sample["file"]
        with np.load(liquid_path) as cache:
            liquid_fields = {
                name: np.asarray(cache[name]).copy()
                for name in ("phi", "normal", "velocity", "collision_sdf")
            }
        channel_records = []
        frame_statistics = {}
        with np.load(emission_path) as cache:
            for channel in BirthChannel:
                name = CHANNEL_NAMES[channel]
                strength = np.asarray(cache[name])
                expected = expected_marker_budget(
                    strength,
                    control_volumes,
                    model.rate_density(channel),
                    dt,
                )
                expected_total = float(expected.sum(dtype=np.float64))
                reservoir_result = reservoirs[channel].advance(
                    expected,
                    float(liquid_sample["simulation_time"]),
                    dt,
                )
                records, realization = realize_marker_births(
                    reservoir_result,
                    channel,
                    int(liquid_sample["source_sample_index"]),
                    liquid_fields,
                    spec,
                    model,
                )
                channel_records.append(records)
                statistics = channel_statistics(
                    records, channel, expected_total, realization
                )
                frame_statistics[name] = statistics
                for key in cumulative[name]:
                    cumulative[name][key] += statistics[key]

        external_records = np.empty(0, dtype=BIRTH_DTYPE)
        external_indices = np.empty(0, dtype=np.int64)
        classification = None
        if domain_contract is not None:
            source_sample = int(liquid_sample["source_sample_index"])
            classification = load_particle_classification(
                domain_contract,
                source_sample,
                int(domain_source_manifest["particle_count"]),
            )
            if classification["classification_schema"] < 2:
                raise RuntimeError("Domain classification lacks one-shot handoff events")
            external_indices = classification["selected_new_handoff_indices"]
            source_row = {
                int(item["sample_index"]): item
                for item in domain_source_manifest["samples"]
            }[source_sample]
            source_path = domain_source_directory / source_row["file"]
            if sha256_file(source_path) != source_row["sha256"]:
                raise RuntimeError(f"External handoff source hash mismatch: {source_path}")
            if len(external_indices):
                with np.load(source_path, allow_pickle=False) as source:
                    source_positions = np.asarray(source["positions"], dtype=np.float32)
                    source_velocities = np.asarray(source["velocities"], dtype=np.float32)
                external_records = handoff_spray_births(
                    source_positions[external_indices],
                    source_velocities[external_indices],
                    external_indices,
                    source_sample=source_sample,
                    birth_time=float(liquid_sample["simulation_time"]),
                    particle_spacing=particle_spacing,
                    seed=args.seed,
                )
            channel_records.append(external_records)
            external_volume = float(
                external_records["phase_volume"].sum(dtype=np.float64)
            )
            cumulative_external["births"] += len(external_records)
            cumulative_external["liquid_phase_volume_m3"] += external_volume
            cumulative_external["source_particle_indices_sha256_by_sample"].append(
                {
                    "source_sample_index": source_sample,
                    "count": int(len(external_indices)),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(external_indices, dtype=np.int64).tobytes()
                    ).hexdigest(),
                }
            )

        records = (
            np.concatenate(channel_records)
            if any(len(value) for value in channel_records)
            else np.empty(0, dtype=BIRTH_DTYPE)
        )
        if len(records):
            records = records[np.argsort(records["id"])]
            if np.any(np.diff(records["id"]) == 0):
                raise RuntimeError("Duplicate marker ID generated within one sample")
        output_path = args.output_directory / f"marker_births_{output_index:06d}.npz"
        atomic_npz(
            output_path,
            schema=np.int32(SCHEMA),
            output_frame=np.int32(liquid_sample["output_frame"]),
            source_sample_index=np.int32(liquid_sample["source_sample_index"]),
            simulation_time=np.float64(liquid_sample["simulation_time"]),
            dt=np.float64(dt),
            births=records,
            external_source_particle_indices=external_indices,
        )
        sample_payload = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "output_frame": int(liquid_sample["output_frame"]),
            "source_sample_index": int(liquid_sample["source_sample_index"]),
            "simulation_time": float(liquid_sample["simulation_time"]),
            "birth_count": len(records),
            "first_birth_time": (
                float(records["birth_time"].min()) if len(records) else None
            ),
            "last_birth_time": (
                float(records["birth_time"].max()) if len(records) else None
            ),
            "channels": frame_statistics,
            "external_liquid_handoff": {
                "births": int(len(external_records)),
                "liquid_phase_volume_m3": float(
                    external_records["phase_volume"].sum(dtype=np.float64)
                ),
                "classification_file": (
                    classification["path"].name if classification is not None else None
                ),
                "classification_sha256": (
                    classification["sha256"] if classification is not None else None
                ),
            },
        }
        manifest["samples"].append(sample_payload)
        atomic_json(output_manifest_path, manifest)
        counts = " ".join(
            f"{name}={frame_statistics[name]['accepted_births']}"
            f"/{frame_statistics[name]['candidate_births']}"
            for name in ("spray", "entrained_air", "churn")
        )
        print(
            f"[v6-birth] source={liquid_sample['source_sample_index']:04d} "
            f"births={len(records)} external={len(external_records)} {counts}",
            flush=True,
        )

    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["cumulative"] = {
        "field_emission": cumulative,
        "external_liquid_handoff": cumulative_external,
    }
    atomic_json(output_manifest_path, manifest)
    print(
        json.dumps(
            {
                "complete": True,
                "samples": len(manifest["samples"]),
                "cumulative": manifest["cumulative"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
