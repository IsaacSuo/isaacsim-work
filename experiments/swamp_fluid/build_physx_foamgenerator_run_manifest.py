"""Build an audited provenance manifest for the PhysX/FoamGenerator closed loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physx_directory", type=Path)
    parser.add_argument("foamgenerator_directory", type=Path)
    parser.add_argument("render_cache_directory", type=Path)
    parser.add_argument("--foam-gate", required=True, type=Path)
    parser.add_argument("--bubbles-gate", required=True, type=Path)
    parser.add_argument("--spray-gate", required=True, type=Path)
    parser.add_argument("--baseline-external-directory", required=True, type=Path)
    parser.add_argument("--foamgenerator-executable", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def file_record(path: Path) -> dict:
    path = path.resolve()
    require(path.is_file(), f"Missing file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_set_record(root: Path, files: list[Path]) -> dict:
    root = root.resolve()
    unique = sorted({path.resolve() for path in files}, key=lambda p: p.as_posix())
    require(unique, f"Empty file set: {root}")
    items = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in unique:
        require(path.is_file(), f"Missing file-set member: {path}")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        items.append({"relative_path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        total_bytes += size
    return {
        "root": str(root),
        "file_count": len(items),
        "total_bytes": total_bytes,
        "inventory_sha256": aggregate.hexdigest(),
        "files": items,
    }


def listed_file_set(root: Path, records: list[dict]) -> dict:
    return file_set_record(root, [Path(record["file"]) for record in records])


def listed_birth_file_set(root: Path, records: list[dict]) -> dict:
    present = []
    absent_zero_particle_files = []
    root = root.resolve()
    for record in records:
        path = Path(record["file"]).resolve()
        if path.is_file():
            present.append(path)
            continue
        require(record.get("particles") == 0, f"Missing non-empty birth file: {path}")
        absent_zero_particle_files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "expected_particles": 0,
            }
        )
    result = file_set_record(root, present)
    result["logical_record_count"] = len(records)
    result["absent_zero_particle_files"] = absent_zero_particle_files
    return result


def validate_render_gate(path: Path, layer: str, camera: str) -> tuple[dict, Path]:
    manifest_path = path / "render_manifest.json"
    manifest = load_json(manifest_path)
    configuration = manifest.get("configuration", {})
    require(manifest.get("state", {}).get("complete") is True, f"Incomplete gate: {path}")
    require(configuration.get("secondary_particles", {}).get("layers") == [layer], f"Wrong gate layer: {path}")
    require(configuration.get("camera", {}).get("preset") == camera, f"Wrong gate camera: {path}")
    require(configuration.get("frame_indices") == [14, 18, 22, 26, 30, 36, 44], f"Wrong gate frames: {path}")
    return manifest, manifest_path


def main() -> None:
    args = parse_args()
    physx = args.physx_directory.resolve()
    foam = args.foamgenerator_directory.resolve()
    cache = args.render_cache_directory.resolve()
    gates = {
        "foam_overview": (args.foam_gate.resolve(), "foam", "overview"),
        "bubbles_underwater": (args.bubbles_gate.resolve(), "bubbles", "underwater"),
        "spray_overview": (args.spray_gate.resolve(), "spray", "overview"),
    }

    preflight_path = physx / "preflight_report.json"
    bridge_path = physx / "bridge_report.json"
    lifecycle_path = foam / "lifecycle_audit.json"
    external_path = foam / "external_motion_audit.json"
    dynamics_path = foam / "type_dynamics_control_audit.json"
    cache_manifest_path = cache / "secondary_manifest.json"
    cache_audit_path = cache / "audit_report.json"

    preflight = load_json(preflight_path)
    bridge = load_json(bridge_path)
    lifecycle = load_json(lifecycle_path)
    external = load_json(external_path)
    dynamics = load_json(dynamics_path)
    cache_manifest = load_json(cache_manifest_path)
    cache_audit = load_json(cache_audit_path)

    require(preflight.get("schema") == "physx-secondary-scene-bridge-preflight/v2", "Wrong preflight schema")
    require(preflight.get("valid") is True, "PhysX preflight is invalid")
    require(bridge.get("schema") == "physx-secondary-scene-bridge/v2", "Wrong bridge schema")
    require(bridge.get("valid") is True, "PhysX bridge is invalid")
    require(all(bridge.get("checks", {}).values()), "At least one PhysX bridge check failed")
    require(bridge.get("secondary_dynamics", {}).get("mode") == "foamgenerator-warp", "Wrong dynamics mode")
    require(bridge.get("secondary_dynamics", {}).get("warp_device", "").startswith("cuda:"), "Warp was not on CUDA")
    require(bridge.get("secondary_dynamics", {}).get("finite_difference_velocity_used") is False, "Finite-difference velocity used")
    require(lifecycle.get("valid") is True, "Lifecycle audit failed")
    require(external.get("valid") is True, "External-motion audit failed")
    require(external.get("positions_bit_exact") is True, "External positions are not bit-exact")
    require(external.get("velocities_bit_exact") is True, "External velocities are not bit-exact")
    require(dynamics.get("valid") is True, "Type-dynamics audit failed")
    require(dynamics.get("cuda_reproduction_bit_exact") is True, "CUDA reproduction is not bit-exact")
    require(dynamics.get("finite_difference_velocity_used") is False, "Dynamics audit found finite differences")
    require(cache_audit.get("valid") is True, "Render-cache audit failed")
    require(cache_manifest.get("configuration", {}).get("output_frame_range") == [14, 44], "Wrong render-cache range")

    gate_manifests = {}
    gate_manifest_paths = {}
    for name, (path, layer, camera) in gates.items():
        gate_manifests[name], gate_manifest_paths[name] = validate_render_gate(path, layer, camera)

    source_files = listed_file_set(Path(preflight["source_files"][0]["file"]).parent, preflight["source_files"])
    birth_files = listed_birth_file_set(Path(preflight["birth_files"][0]["file"]).parent, preflight["birth_files"])
    baseline_files = file_set_record(
        args.baseline_external_directory,
        sorted(args.baseline_external_directory.glob("external_*.bgeo")),
    )

    script_root = Path(__file__).resolve().parent
    scripts = [
        "run_physx_secondary_scene_bridge.py",
        "foamgenerator_warp_dynamics.py",
        "audit_foamgenerator_lifecycle_bgeo.py",
        "audit_foamgenerator_external_motion.py",
        "audit_physx_type_dynamics_control.py",
        "build_physx_foamgenerator_render_caches.py",
        "audit_physx_foamgenerator_render_caches.py",
        "render_splashsurf_sequence.py",
        Path(__file__).name,
    ]

    output_sets = {
        "physx_external_motion": file_set_record(physx / "external_motion", sorted((physx / "external_motion").glob("external_*.bgeo"))),
        "physx_dynamics_control": file_set_record(physx / "dynamics_control", sorted((physx / "dynamics_control").glob("control_*.npz"))),
        "foamgenerator_classification": file_set_record(foam, sorted(foam.glob("secondary_*.bgeo"))),
        "render_cache": file_set_record(cache, sorted(cache.glob("*/*.npz"))),
    }
    for name, (path, _layer, _camera) in gates.items():
        output_sets[name] = file_set_record(path, sorted(path.glob("frame_*.png")))

    manifest = {
        "schema": "physx-foamgenerator-closed-loop-run/v1",
        "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "scene": "swamp",
            "source_frame_range": [48, 180],
            "external_motion_frame_range": [49, 180],
            "render_frame_range": [14, 44],
            "source_fps": 120,
            "physics_fps": 240,
            "render_fps": 30,
        },
        "responsibility_boundary": {
            "physx": [
                "native primary-liquid positions and velocities",
                "secondary integration",
                "primary-secondary interaction",
                "all scene collisions",
            ],
            "foamgenerator": [
                "birth events",
                "stable ids",
                "chronological lifetime",
                "neighbor-count classification",
                "classification after PhysX readback",
            ],
            "warp_cuda": [
                "FoamGenerator-compatible neighborhood query",
                "CubicKernel local primary-liquid velocity",
                "type-dependent velocity control",
            ],
            "blender": ["render-only layer visibility", "render-only radius, shape, and opacity"],
        },
        "execution_contract": {
            "isaac_sim_renderer": "MinimalRendering",
            "isaac_sim_cpu_thread_limit": 2,
            "multi_gpu": False,
            "maximum_gpu_count": 1,
            "warp_device": bridge["secondary_dynamics"]["warp_device"],
            "finite_difference_velocity_used": False,
            "high_load_processes_concurrent": False,
        },
        "algorithms": bridge["secondary_dynamics"],
        "statistics": {
            "maximum_active_secondary_particles": bridge["maximum_active_secondary_particles"],
            "maximum_secondary_speed": bridge["maximum_secondary_speed"],
            "controlled_particle_states": bridge["controlled_particle_states"],
            "controlled_type_totals": bridge["controlled_type_totals"],
            "lifecycle_files_audited": lifecycle["files_audited"],
            "external_motion_particles_audited": external["particles_audited"],
            "type_dynamics_particle_states_audited": dynamics["particle_states_audited"],
            "render_cache_particle_states_audited": cache_audit["particle_states_audited"],
        },
        "audit_results": {
            "physx_preflight": file_record(preflight_path),
            "physx_bridge": file_record(bridge_path),
            "foamgenerator_lifecycle": file_record(lifecycle_path),
            "external_motion_authority": file_record(external_path),
            "type_dynamics_cuda": file_record(dynamics_path),
            "render_cache": file_record(cache_audit_path),
        },
        "inputs": {
            "scene_usd": file_record(Path(bridge["scene_usd"])),
            "configured_stage": file_record(Path(bridge["configured_stage"])),
            "primary_liquid_source": source_files,
            "foamgenerator_births": birth_files,
            "ballistic_baseline_for_comparison_only": baseline_files,
        },
        "outputs": output_sets,
        "manifests": {
            "render_cache": file_record(cache_manifest_path),
            **{name: file_record(path) for name, path in gate_manifest_paths.items()},
        },
        "producer_code": {name: file_record(script_root / name) for name in scripts},
        "software": {
            "isaac_sim": "6.0",
            "warp": "1.13.0",
            "cuda_toolkit": "12.9",
            "foamgenerator": "SPlisHSPlasH FoamGenerator 2.18.1",
            "blender": "5.0.1",
            "foamgenerator_executable": (
                file_record(args.foamgenerator_executable)
                if args.foamgenerator_executable is not None
                else None
            ),
        },
        "known_nonfatal_warnings": [
            {
                "producer": "Isaac Sim/Fabric",
                "message": "Unsupported type encountered during VtValue extraction",
                "impact": "Observed during the valid run; no failed bridge or readback audit.",
            },
            {
                "producer": "Isaac Sim/Fabric",
                "message": "Attempting to set an invalid data source to array attribute",
                "impact": "Observed during the valid run; no failed bridge or readback audit.",
            },
            {
                "producer": "Isaac Sim stage shutdown",
                "message": "Reference-count warning while closing the stage",
                "impact": "Shutdown-only warning; process exited normally and no residual process remained.",
            },
            {
                "producer": "Blender Cycles",
                "message": "HIPEW initialization failed",
                "impact": "HIP is unavailable; the render used the configured NVIDIA GPU path and completed.",
            },
        ],
        "visual_gate": {
            "dynamics": "passed",
            "foam_optics": "not-final-discrete-parcel-spheres",
            "bubble_optics": "not-final-open-free-surface-medium",
            "spray_optics": "not-final-single-scale-droplets",
            "water_surface_quadrilateral_holes_observed": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), f"Refusing to overwrite: {args.output}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(args.output)
    print(json.dumps({
        "valid": True,
        "output": str(args.output.resolve()),
        "output_file_sets": {name: value["file_count"] for name, value in output_sets.items()},
        "input_primary_files": source_files["file_count"],
        "input_birth_files": birth_files["file_count"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
