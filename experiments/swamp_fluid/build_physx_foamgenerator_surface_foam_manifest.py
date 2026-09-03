"""Build the versioned provenance manifest for the audited surface-foam extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict:
    path = path.resolve()
    require(path.is_file(), f"Missing file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def file_set(root: Path, files) -> dict:
    root = root.resolve()
    paths = sorted({Path(path).resolve() for path in files}, key=lambda path: path.as_posix())
    require(paths, f"Empty file set: {root}")
    aggregate = hashlib.sha256()
    records = []
    total = 0
    for path in paths:
        require(path.is_file(), f"Missing file-set member: {path}")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        records.append({"relative_path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
        total += size
    return {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": total,
        "inventory_sha256": aggregate.hexdigest(),
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_run_manifest", type=Path)
    parser.add_argument("binding_directory", type=Path)
    parser.add_argument("payload_directory", type=Path)
    parser.add_argument("membrane_directory", type=Path)
    parser.add_argument("render_directory", type=Path)
    parser.add_argument("surface_directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root_run_path = args.root_run_manifest.resolve()
    binding = args.binding_directory.resolve()
    payload = args.payload_directory.resolve()
    membrane = args.membrane_directory.resolve()
    render = args.render_directory.resolve()
    surface = args.surface_directory.resolve()
    root_run = load_json(root_run_path)
    binding_manifest_path = binding / "manifest.json"
    binding_audit_path = binding / "sequence_audit_v4_chatter.json"
    payload_manifest_path = payload / "manifest.json"
    membrane_manifest_path = membrane / "manifest.json"
    membrane_audit_path = membrane / "sequence_ab_audit_v2.json"
    render_manifest_path = render / "render_manifest.json"
    binding_manifest = load_json(binding_manifest_path)
    binding_audit = load_json(binding_audit_path)
    payload_manifest = load_json(payload_manifest_path)
    membrane_manifest = load_json(membrane_manifest_path)
    membrane_audit = load_json(membrane_audit_path)
    render_manifest = load_json(render_manifest_path)

    require(root_run.get("valid") is True, "Root closed-loop manifest is invalid")
    require(binding_manifest.get("complete") is True, "Binding is incomplete")
    require(binding_audit.get("valid") is True, "Binding sequence audit failed")
    require(binding_audit.get("structural_valid") is True, "Binding structural audit failed")
    require(payload_manifest.get("complete") is True, "Payload is incomplete")
    require(
        payload_manifest["summary"]["maximum_area_balance_residual_m2"] == 0.0,
        "Payload area ledger is not exact",
    )
    require(membrane_manifest.get("complete") is True, "Membrane is incomplete")
    configuration = membrane_manifest["configuration"]
    require(configuration.get("finite_difference_velocity_used") is False, "Finite-difference velocity used")
    require(configuration.get("temporal_hysteresis_enabled") is True, "Temporal hysteresis is disabled")
    require(configuration["eligibility_hysteresis"]["enabled"] is True, "Eligibility debounce is disabled")
    require(
        membrane_audit.get("valid") is True
        and membrane_audit.get("structural_valid") is True,
        "Membrane sequence audit failed",
    )
    require(
        membrane_audit["candidate"]["summary"][
            "total_one_frame_eligible_residual_chatter_events"
        ]
        == 0,
        "Connected/residual chatter is non-zero",
    )
    require(render_manifest.get("state", {}).get("complete") is True, "Render is incomplete")
    require(render_manifest["configuration"]["frame_indices"] == list(range(14, 45)), "Wrong render frames")
    require(
        Path(render_manifest["configuration"]["connected_foam_membrane"]["directory"]).resolve()
        == membrane,
        "Render used a different membrane product",
    )

    native_video = render / "surface_foam_full_native_30fps.mp4"
    slow_video = render / "surface_foam_full_slow_15fps.mp4"
    script_root = Path(__file__).resolve().parent
    producer_names = [
        "build_physx_foamgenerator_surface_binding.py",
        "audit_physx_foamgenerator_surface_binding_sequence.py",
        "build_physx_foamgenerator_surface_payload.py",
        "build_physx_foamgenerator_membrane.py",
        "audit_physx_foamgenerator_membrane_sequence.py",
        "render_splashsurf_sequence.py",
        "whitewater/mesh_surface_sampler.py",
        "whitewater/foam_render_payload.py",
        "whitewater/foam_membrane_surface.py",
        Path(__file__).name,
        "audit_physx_foamgenerator_surface_foam_manifest.py",
    ]
    products = {
        "surface_binding": file_set(binding, binding.glob("binding_*.npz")),
        "surface_payload": file_set(payload, payload.glob("surface_payload_*.npz")),
        "connected_membrane": file_set(membrane, membrane.glob("membrane_*.npz")),
        "render_frames": file_set(render, render.glob("frame_*.png")),
        "render_videos": file_set(render, (native_video, slow_video)),
    }
    inputs = {
        "root_closed_loop_manifest": file_record(root_run_path),
        "splashsurf_surface_sequence": file_set(
            surface,
            [surface / f"surface_{frame:04d}_clipped.obj" for frame in range(14, 45)],
        ),
    }
    manifest = {
        "schema": "physx-foamgenerator-surface-foam-extension/v2",
        "valid": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "scene": "swamp",
            "output_frames": [14, 44],
            "source_frames": [56, 176],
            "render_fps": 30,
            "product": "surface foam optical gate, not full whitewater final",
        },
        "contracts": {
            "height_field_used": False,
            "surface_binding": "exact closest oriented triangle on open Splashsurf mesh",
            "finite_difference_velocity_used": False,
            "temporal_advection": "audited FoamGenerator/PhysX native velocity",
            "film_area": "particle_spacing^2 * opacity * drainage coverage; exact ledger",
            "connected_residual_partition": "mutually exclusive and stable-ID debounced",
            "render_only_calibration": [
                "8 mm primary-liquid spacing support footprint",
                "initial coverage 0.72",
                "drainage time 2.5 seconds",
                "5 mm world-space cellular scattering",
            ],
        },
        "statistics": {
            "binding_states_audited": binding_audit["states_audited"],
            "binding_one_frame_chatter_events": binding_audit["temporal_gate"][
                "total_one_frame_bound_unbound_chatter_events"
            ],
            "payload_maximum_area_balance_residual_m2": payload_manifest["summary"][
                "maximum_area_balance_residual_m2"
            ],
            "membrane_maximum_area_residual_m2": membrane_manifest["summary"][
                "maximum_area_residual_m2"
            ],
            "eligible_residual_one_frame_chatter_events": 0,
            "mean_area_retained_within_10mm": membrane_audit["candidate"]["summary"][
                "mean_area_retained_within_10mm"
            ],
            "mean_area_supported_within_10mm": membrane_audit["candidate"]["summary"][
                "mean_area_supported_within_10mm"
            ],
        },
        "audits": {
            "surface_binding": file_record(binding_audit_path),
            "membrane_sequence_ab": file_record(membrane_audit_path),
            "render_manifest": file_record(render_manifest_path),
        },
        "product_manifests": {
            "surface_binding": file_record(binding_manifest_path),
            "surface_payload": file_record(payload_manifest_path),
            "membrane": file_record(membrane_manifest_path),
        },
        "inputs": inputs,
        "products": products,
        "producer_code": {
            name: file_record(script_root / name) for name in producer_names
        },
        "known_limits": [
            "FoamGenerator does not export physical gas radius or gas volume; support area and coverage are render-only calibration.",
            "Unbound foam IDs remain explicit in the binding cache and are not forced onto the free surface.",
            "Entrained bubbles are not rendered as submerged gas boundaries because the clipped Splashsurf sheet is not a closed water medium.",
            "Spray multi-scale optics and Plateau hero-cell geometry remain separate gates.",
            "Near-static raw surface-anchor normal/correction diagnostics exceed their advisory references; downstream membrane hysteresis and debounce are separately audited.",
        ],
    }
    output = args.output.resolve()
    require(not output.exists(), f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"valid": True, "output": str(output), "products": {key: value["file_count"] for key, value in products.items()}}, indent=2))


if __name__ == "__main__":
    main()
