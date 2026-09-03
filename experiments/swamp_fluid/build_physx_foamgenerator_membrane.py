"""Build exact-area connected foam membranes on retained Splashsurf triangles."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.foam_membrane_surface import (
    FoamMembraneModel,
    build_conservative_membrane,
    eligible_patch_mask,
    load_triangle_obj,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frame-list", nargs="+", required=True, type=int)
    parser.add_argument("--temporal-hysteresis", action="store_true")
    parser.add_argument("--temporal-hysteresis-radius", type=float, default=0.012)
    parser.add_argument("--temporal-preference-strength", type=float, default=1.0)
    parser.add_argument("--eligibility-hysteresis", action="store_true")
    parser.add_argument("--eligibility-enter-abs-up-normal", type=float, default=0.55)
    parser.add_argument("--eligibility-exit-abs-up-normal", type=float, default=0.45)
    parser.add_argument("--eligibility-switch-confirmation-frames", type=int, default=2)
    args = parser.parse_args()
    payload_directory = args.payload_directory.resolve()
    output_directory = args.output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    payload_manifest_path = payload_directory / "manifest.json"
    payload_manifest = json.loads(payload_manifest_path.read_text(encoding="utf-8"))
    if payload_manifest.get("complete") is not True:
        raise RuntimeError("Surface-payload manifest is incomplete")
    payload_samples = {
        int(sample["output_frame"]): sample for sample in payload_manifest["samples"]
    }
    frames = sorted(set(args.frame_list))
    model = FoamMembraneModel(
        temporal_hysteresis_radius_m=args.temporal_hysteresis_radius,
        temporal_preference_strength=args.temporal_preference_strength,
    )
    previous_temporal_state = None
    eligibility_states = {}
    if not (
        0.0 < args.eligibility_exit_abs_up_normal
        < args.eligibility_enter_abs_up_normal
        <= 1.0
    ):
        raise ValueError("Eligibility hysteresis requires 0 < exit < enter <= 1")
    if args.eligibility_switch_confirmation_frames < 1:
        raise ValueError("Eligibility switch confirmation must be at least one frame")
    manifest_path = output_directory / "manifest.json"
    manifest = {
        "schema": "physx-foamgenerator-membrane/v2",
        "product": "physx_foamgenerator_conservative_membrane",
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "frames": frames,
            "model": model.metadata(),
            "temporal_hysteresis_enabled": args.temporal_hysteresis,
            "temporal_advection": (
                "previous selected triangle centres advected by nearest source parcel's "
                "audited native velocity; no finite-difference velocity"
            ),
            "finite_difference_velocity_used": False,
            "eligibility_hysteresis": {
                "enabled": args.eligibility_hysteresis,
                "enter_abs_up_normal": args.eligibility_enter_abs_up_normal,
                "exit_abs_up_normal": args.eligibility_exit_abs_up_normal,
                "switch_confirmation_frames": args.eligibility_switch_confirmation_frames,
                "policy": (
                    "stable-ID Schmitt trigger changes only the mutually exclusive "
                    "connected/residual optical role; total film-area ledger is unchanged"
                ),
            },
            "geometry_authority": (
                "retained Splashsurf triangles selected by parcel density; exact ledger "
                "film area with similarity-scaled final residual triangle"
            ),
            "residual_policy": "non-top-facing, macro, hole-host, or unbound patches remain explicit residuals",
        },
        "inputs": {
            "payload_directory": str(payload_directory),
            "payload_manifest_sha256": sha256_file(payload_manifest_path),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "module_sha256": sha256_file(Path(__file__).parent / "whitewater" / "foam_membrane_surface.py"),
        },
        "samples": [],
    }
    atomic_json(manifest_path, manifest)
    try:
        for frame in frames:
            sample = payload_samples.get(frame)
            if sample is None:
                raise RuntimeError(f"Payload has no frame {frame}")
            payload_path = payload_directory / sample["file"]
            surface_path = Path(sample["surface"])
            with np.load(payload_path, allow_pickle=False) as cache:
                patches = np.asarray(cache["patches"]).copy()
                source_foam = np.asarray(cache["source_foam"]).copy()
            vertices, triangles = load_triangle_obj(surface_path)
            eligibility_mask = None
            if args.eligibility_hysteresis:
                core = (
                    (patches["film_area"] > 0.0)
                    & (patches["hole_fraction"] <= 0.0)
                    & (patches["render_class"] != 2)
                )
                abs_up = np.abs(patches["normal"][:, 1])
                eligibility_mask = np.zeros(len(patches), dtype=bool)
                current_ids = set(map(int, patches["source_foam_id"]))
                eligibility_states = {
                    marker_id: state
                    for marker_id, state in eligibility_states.items()
                    if marker_id in current_ids
                }
                for row, marker_id_value in enumerate(patches["source_foam_id"]):
                    marker_id = int(marker_id_value)
                    if not core[row]:
                        eligibility_states[marker_id] = {
                            "eligible": False,
                            "pending": None,
                            "count": 0,
                        }
                        continue
                    state = eligibility_states.get(marker_id)
                    if state is None:
                        state = {
                            "eligible": bool(abs_up[row] >= model.minimum_abs_up_normal),
                            "pending": None,
                            "count": 0,
                        }
                    desired = state["eligible"]
                    if state["eligible"]:
                        if abs_up[row] < args.eligibility_exit_abs_up_normal:
                            desired = False
                    elif abs_up[row] >= args.eligibility_enter_abs_up_normal:
                        desired = True
                    if desired == state["eligible"]:
                        state["pending"] = None
                        state["count"] = 0
                    else:
                        if state["pending"] == desired:
                            state["count"] += 1
                        else:
                            state["pending"] = desired
                            state["count"] = 1
                        if state["count"] >= args.eligibility_switch_confirmation_frames:
                            state["eligible"] = desired
                            state["pending"] = None
                            state["count"] = 0
                    eligibility_states[marker_id] = state
                    eligibility_mask[row] = state["eligible"]
            temporal_reference = None
            if args.temporal_hysteresis and previous_temporal_state is not None:
                source_step = int(sample["source_frame"]) - previous_temporal_state["source_frame"]
                if source_step <= 0:
                    raise RuntimeError("Temporal membrane frames must have increasing source frames")
                temporal_reference = (
                    previous_temporal_state["centres"]
                    + previous_temporal_state["velocity"] * (source_step / 120.0)
                )
            geometry, metrics = build_conservative_membrane(
                vertices,
                triangles,
                patches,
                model,
                temporal_reference_centres=temporal_reference,
                eligibility_mask=eligibility_mask,
            )
            mask = (
                eligible_patch_mask(patches, model=model)
                if eligibility_mask is None
                else eligibility_mask
            )
            eligible = patches[mask]
            residual = patches[~mask]
            metrics = {
                **metrics,
                "residual_film_area_m2": float(
                    residual["film_area"].sum(dtype=np.float64)
                ),
            }
            source_velocity = {
                int(marker_id): velocity
                for marker_id, velocity in zip(source_foam["id"], source_foam["velocity"])
            }
            nearest_ids = eligible["source_foam_id"][
                geometry["nearest_eligible_patch_rows"]
            ]
            triangle_advection_velocity = np.asarray(
                [source_velocity[int(marker_id)] for marker_id in nearest_ids],
                dtype=np.float32,
            )
            represented_centres = geometry["vertices"][geometry["triangles"]].mean(axis=1)
            previous_temporal_state = {
                "source_frame": int(sample["source_frame"]),
                "centres": represented_centres.astype(np.float64),
                "velocity": triangle_advection_velocity.astype(np.float64),
            }
            output_path = output_directory / f"membrane_{frame:04d}.npz"
            atomic_npz(
                output_path,
                schema=np.asarray("physx-foamgenerator-membrane-frame/v2"),
                output_frame=np.int32(frame),
                source_frame=np.int32(int(sample["source_frame"])),
                eligible_patches=eligible,
                residual_patches=residual,
                source_foam=source_foam,
                triangle_advection_velocity=triangle_advection_velocity,
                **geometry,
            )
            record = {
                "output_frame": frame,
                "source_frame": int(sample["source_frame"]),
                "file": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "payload": str(payload_path),
                "payload_sha256": sha256_file(payload_path),
                "surface": str(surface_path),
                "surface_sha256": sha256_file(surface_path),
                "metrics": metrics,
            }
            manifest["samples"].append(record)
            atomic_json(manifest_path, manifest)
            print(
                f"[foam-membrane] frame={frame:04d} eligible={len(eligible)} "
                f"residual={len(residual)} triangles={metrics['selected_triangle_count']} "
                f"area={metrics['represented_membrane_area_m2']:.6g}m2",
                flush=True,
            )
        manifest["complete"] = True
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = {
            "frames": len(manifest["samples"]),
            "maximum_selected_triangles": max(sample["metrics"]["selected_triangle_count"] for sample in manifest["samples"]),
            "maximum_area_residual_m2": max(abs(sample["metrics"]["area_residual_m2"]) for sample in manifest["samples"]),
            "maximum_residual_film_area_m2": max(sample["metrics"]["residual_film_area_m2"] for sample in manifest["samples"]),
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps({"valid": True, **manifest["summary"]}, indent=2))
    except Exception as exc:
        atomic_json(
            output_directory / "OBSOLETE.json",
            {
                "schema": 1,
                "obsolete": True,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reason": str(exc),
                "traceback": traceback.format_exc(),
                "completed_samples": len(manifest["samples"]),
            },
        )
        raise


if __name__ == "__main__":
    main()
