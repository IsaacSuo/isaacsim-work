"""Build persistent v6 marker trajectories and conservative terminal events."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec, load_collision_fields
from whitewater.marker_solver import (
    EVENT_DTYPE,
    SOLVER_DTYPE,
    SUPPORT_LOSS_DTYPE,
    MarkerSolverModel,
    TerminalEventKind,
    advance_marker_interval,
    births_to_solver_markers,
)
from whitewater.render_surface_support import (
    build_render_surface_support,
    load_render_surface_support_recipe,
)
from whitewater.point_collisions import SparsePointCollisionScene
from whitewater.scene_contract import SceneContract
from whitewater.state_machine import WhitewaterState


SCHEMA = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("liquid_field_directory", type=Path)
    parser.add_argument("marker_birth_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--render-surface-recipe-manifest",
        type=Path,
        required=True,
        help=(
            "Production Splashsurf manifest whose terrain shoreline and impact "
            "contract constrain renderable surface-bubble transitions."
        ),
    )
    parser.add_argument("--maximum-substeps", type=int, default=64)
    parser.add_argument(
        "--secondary-domain-padding-cells",
        type=float,
        help=(
            "Explicit ballistic marker envelope outside the carrier grid. "
            "Default derives a conservative whole-sequence travel bound."
        ),
    )
    parser.add_argument(
        "--secondary-scene-contract",
        type=Path,
        help=(
            "Optional scene contract override for sparse collision queries. "
            "Default uses the liquid-field scene contract."
        ),
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_liquid_fields(path, spec, support_recipe):
    with np.load(path) as cache:
        fields = {
            name: np.asarray(cache[name]).copy()
            for name in ("phi", "normal", "velocity")
        }
        load_collision_fields(cache, fields)
        support, support_metrics = build_render_surface_support(
            np.asarray(cache["fluid_mask"]), spec, support_recipe
        )
        fields["render_surface_support"] = support
        return fields, support_metrics


def load_source_snapshot(source_directory, sample):
    path = source_directory / sample["source_file"]
    with np.load(path, allow_pickle=False) as cache:
        return {name: np.asarray(cache[name]).copy() for name in cache.files}


def state_counts(markers):
    return {
        "spray": int(
            np.count_nonzero(markers["state"] == np.uint8(WhitewaterState.SPRAY))
        ),
        "entrained_bubble": int(
            np.count_nonzero(
                markers["state"] == np.uint8(WhitewaterState.ENTRAINED_BUBBLE)
            )
        ),
        "surface_bubble": int(
            np.count_nonzero(
                markers["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
            )
        ),
    }


def event_counts(events):
    return {
        "spray_reentry": int(
            np.count_nonzero(events["kind"] == np.uint8(TerminalEventKind.SPRAY_REENTRY))
        ),
        "bubble_burst": int(
            np.count_nonzero(events["kind"] == np.uint8(TerminalEventKind.BUBBLE_BURST))
        ),
        "domain_escape": int(
            np.count_nonzero(events["kind"] == np.uint8(TerminalEventKind.DOMAIN_ESCAPE))
        ),
    }


def main():
    args = parse_args()
    liquid_manifest_path = args.liquid_field_directory / "manifest.json"
    birth_manifest_path = args.marker_birth_directory / "manifest.json"
    liquid_manifest = load_json(liquid_manifest_path)
    birth_manifest = load_json(birth_manifest_path)
    if liquid_manifest.get("product") != "whitewater_v6_liquid_fields":
        raise RuntimeError("Unexpected liquid-field product")
    if birth_manifest.get("product") != "whitewater_v6_marker_births":
        raise RuntimeError("Unexpected marker-birth product")
    if not birth_manifest.get("complete"):
        raise RuntimeError("Marker-birth input is incomplete")
    if liquid_manifest.get("grid") != birth_manifest.get("grid"):
        raise RuntimeError("Liquid and marker-birth grids differ")
    liquid_samples = liquid_manifest.get("samples", [])
    birth_samples = birth_manifest.get("samples", [])
    if not liquid_samples or len(liquid_samples) != len(birth_samples):
        raise RuntimeError("Liquid and marker-birth sample counts differ")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty trajectory directory: {args.output_directory}"
        )

    grid = liquid_manifest["grid"]
    spec = GridSpec(
        tuple(grid["origin"]), float(grid["spacing"]), tuple(grid["shape"])
    )
    maximum_birth_speed = 0.0
    for birth_sample in birth_samples:
        with np.load(args.marker_birth_directory / birth_sample["file"]) as cache:
            births = np.asarray(cache["births"])
        if len(births):
            maximum_birth_speed = max(
                maximum_birth_speed,
                float(np.linalg.norm(births["velocity"], axis=1).max()),
            )
    sequence_start = float(liquid_samples[0]["simulation_time"])
    sequence_end = float(liquid_samples[-1]["simulation_time"]) + float(
        birth_manifest["configuration"]["dt_seconds"]
    )
    sequence_duration = sequence_end - sequence_start
    ballistic_travel_bound = (
        maximum_birth_speed * sequence_duration
        + 0.5 * 9.81 * sequence_duration**2
    )
    derived_padding_cells = float(
        np.ceil(ballistic_travel_bound / spec.spacing) + 2.0
    )
    secondary_padding_cells = (
        derived_padding_cells
        if args.secondary_domain_padding_cells is None
        else args.secondary_domain_padding_cells
    )
    model = MarkerSolverModel(
        spec.spacing,
        maximum_substeps=args.maximum_substeps,
        secondary_domain_padding_cells=secondary_padding_cells,
    )
    support_recipe = load_render_surface_support_recipe(
        args.render_surface_recipe_manifest
    )
    scene_contract_record = liquid_manifest.get("scene_contract", {})
    if args.secondary_scene_contract is not None:
        scene_contract_path = args.secondary_scene_contract.resolve()
        scene_contract = SceneContract.load(scene_contract_path)
    elif scene_contract_record.get("path"):
        scene_contract_path = Path(scene_contract_record["path"]).resolve()
        scene_contract = SceneContract.load(scene_contract_path)
    elif scene_contract_record.get("resolved"):
        scene_contract_path = None
        scene_contract = SceneContract.from_mapping(scene_contract_record["resolved"])
    else:
        raise RuntimeError("Liquid fields do not identify a secondary scene contract")
    point_collision_scene = SparsePointCollisionScene(scene_contract)
    source_directory = Path(liquid_manifest["source"]["directory"])
    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_directory / "manifest.json"
    solver_module = Path(__file__).parent / "whitewater" / "marker_solver.py"
    manifest = {
        "schema": SCHEMA,
        "product": "whitewater_v6_marker_trajectories",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": spec.metadata(),
        "configuration": {
            "model": model.metadata(),
            "secondary_domain_derivation": {
                "mode": (
                    "whole_sequence_ballistic_bound"
                    if args.secondary_domain_padding_cells is None
                    else "explicit_cells"
                ),
                "maximum_birth_speed_mps": maximum_birth_speed,
                "sequence_duration_seconds": sequence_duration,
                "ballistic_travel_bound_m": ballistic_travel_bound,
                "derived_padding_cells": derived_padding_cells,
                "selected_padding_cells": secondary_padding_cells,
                "grid_allocation": False,
            },
            "field_time_interpolation": "linear between adjacent 120 Hz snapshots",
            "last_interval_field_policy": "zero_order_hold",
            "birth_activation": "exact per-marker birth_time within interval",
            "render_surface_support": support_recipe.metadata(),
            "secondary_collision_scene": point_collision_scene.metadata(),
        },
        "inputs": {
            "liquid_manifest": str(liquid_manifest_path.resolve()),
            "liquid_manifest_sha256": sha256_file(liquid_manifest_path),
            "birth_manifest": str(birth_manifest_path.resolve()),
            "birth_manifest_sha256": sha256_file(birth_manifest_path),
            "secondary_scene_contract": (
                str(scene_contract_path) if scene_contract_path is not None else None
            ),
            "secondary_scene_contract_sha256": (
                sha256_file(scene_contract_path)
                if scene_contract_path is not None
                else scene_contract_record.get("sha256")
            ),
        },
        "producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
            "solver_module": str(solver_module.resolve()),
            "solver_module_sha256": sha256_file(solver_module),
            "render_surface_support_module": str(
                (Path(__file__).parent / "whitewater" / "render_surface_support.py").resolve()
            ),
            "render_surface_support_module_sha256": sha256_file(
                Path(__file__).parent / "whitewater" / "render_surface_support.py"
            ),
            "point_collision_module": str(
                (Path(__file__).parent / "whitewater" / "point_collisions.py").resolve()
            ),
            "point_collision_module_sha256": sha256_file(
                Path(__file__).parent / "whitewater" / "point_collisions.py"
            ),
        },
        "record_contract": {
            "marker_dtype": SOLVER_DTYPE.descr,
            "event_dtype": EVENT_DTYPE.descr,
            "surface_support_loss_dtype": SUPPORT_LOSS_DTYPE.descr,
            "snapshot_time": "end of each source interval",
            "terminal_event_semantics": {
                "1": "spray re-entered carrier liquid",
                "2": "surface bubble burst and gas escaped",
                "3": "marker escaped reconstruction domain",
            },
        },
        "samples": [],
        "complete": False,
    }
    atomic_json(manifest_path, manifest)

    markers = np.empty(0, dtype=SOLVER_DTYPE)
    cumulative_born_volume = {"liquid": 0.0, "gas": 0.0}
    cumulative_terminal_volume = {"liquid": 0.0, "gas": 0.0}
    cumulative_events = {"spray_reentry": 0, "bubble_burst": 0, "domain_escape": 0}
    current_fields, current_support_metrics = load_liquid_fields(
        args.liquid_field_directory / liquid_samples[0]["file"],
        spec,
        support_recipe,
    )
    current_snapshot = load_source_snapshot(source_directory, liquid_samples[0])
    for index, (liquid_sample, birth_sample) in enumerate(
        zip(liquid_samples, birth_samples)
    ):
        if int(liquid_sample["source_sample_index"]) != int(
            birth_sample["source_sample_index"]
        ):
            raise RuntimeError("Trajectory inputs have mismatched source samples")
        birth_path = args.marker_birth_directory / birth_sample["file"]
        with np.load(birth_path) as cache:
            births = np.asarray(cache["births"]).copy()
            dt = float(cache["dt"])
        new_markers = births_to_solver_markers(births)
        if len(new_markers):
            existing_ids = markers["id"] if len(markers) else np.empty(0, np.uint64)
            if np.intersect1d(existing_ids, new_markers["id"]).size:
                raise RuntimeError("A marker birth ID already exists in the active set")
            markers = np.concatenate((markers, new_markers))
            markers = markers[np.argsort(markers["id"])]
        liquid_volume = float(
            births["phase_volume"][births["phase"] == 1].sum(dtype=np.float64)
        )
        gas_volume = float(
            births["phase_volume"][births["phase"] == 2].sum(dtype=np.float64)
        )
        cumulative_born_volume["liquid"] += liquid_volume
        cumulative_born_volume["gas"] += gas_volume

        if index + 1 < len(liquid_samples):
            next_fields, next_support_metrics = load_liquid_fields(
                args.liquid_field_directory / liquid_samples[index + 1]["file"],
                spec,
                support_recipe,
            )
            next_snapshot = load_source_snapshot(
                source_directory, liquid_samples[index + 1]
            )
        else:
            next_fields = None
            next_support_metrics = current_support_metrics
            next_snapshot = None
        step_start = float(liquid_sample["simulation_time"])
        markers, events, support_losses, metrics = advance_marker_interval(
            markers,
            step_start,
            dt,
            current_fields,
            next_fields,
            spec,
            model,
            secondary_collision_query=lambda positions, alpha: point_collision_scene.query(
                positions,
                snapshot0=current_snapshot,
                snapshot1=next_snapshot,
                alpha=alpha,
            ),
        )
        counts = event_counts(events)
        for name, value in counts.items():
            cumulative_events[name] += value
        cumulative_terminal_volume["liquid"] += float(
            events["phase_volume"][events["phase"] == 1].sum(dtype=np.float64)
        )
        cumulative_terminal_volume["gas"] += float(
            events["phase_volume"][events["phase"] == 2].sum(dtype=np.float64)
        )
        output_path = args.output_directory / f"marker_state_{index:06d}.npz"
        atomic_npz(
            output_path,
            schema=np.int32(SCHEMA),
            output_frame=np.int32(liquid_sample["output_frame"]),
            source_sample_index=np.int32(liquid_sample["source_sample_index"]),
            interval_start=np.float64(step_start),
            interval_end=np.float64(step_start + dt),
            markers=markers,
            terminal_events=events,
            surface_support_losses=support_losses,
        )
        marker_volume = {
            "liquid": float(
                markers["phase_volume"][markers["phase"] == 1].sum(dtype=np.float64)
            ),
            "gas": float(
                markers["phase_volume"][markers["phase"] == 2].sum(dtype=np.float64)
            ),
        }
        sample_payload = {
            "file": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "output_frame": int(liquid_sample["output_frame"]),
            "source_sample_index": int(liquid_sample["source_sample_index"]),
            "interval_start": step_start,
            "interval_end": step_start + dt,
            "birth_count": len(births),
            "active_count": len(markers),
            "state_counts": state_counts(markers),
            "event_counts": counts,
            "surface_support_loss_count": len(support_losses),
            "active_phase_volume_m3": marker_volume,
            "solver_metrics": metrics,
            "render_surface_support_metrics": next_support_metrics,
        }
        manifest["samples"].append(sample_payload)
        atomic_json(manifest_path, manifest)
        states = sample_payload["state_counts"]
        print(
            f"[v6-trajectory] source={liquid_sample['source_sample_index']:04d} "
            f"born={len(births)} active={len(markers)} "
            f"spray={states['spray']} entrained={states['entrained_bubble']} "
            f"surface={states['surface_bubble']} events={len(events)} "
            f"substeps={metrics['substeps']} cfl={metrics['maximum_cfl']:.3f}",
            flush=True,
        )
        current_fields = next_fields if next_fields is not None else current_fields
        current_support_metrics = next_support_metrics
        current_snapshot = (
            next_snapshot if next_snapshot is not None else current_snapshot
        )

    active_volume = {
        "liquid": float(
            markers["phase_volume"][markers["phase"] == 1].sum(dtype=np.float64)
        ),
        "gas": float(
            markers["phase_volume"][markers["phase"] == 2].sum(dtype=np.float64)
        ),
    }
    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["cumulative"] = {
        "born_phase_volume_m3": cumulative_born_volume,
        "active_phase_volume_m3": active_volume,
        "terminal_phase_volume_m3": cumulative_terminal_volume,
        "terminal_event_counts": cumulative_events,
        "active_markers": len(markers),
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"complete": True, "samples": len(manifest["samples"]), "cumulative": manifest["cumulative"]}, indent=2))


if __name__ == "__main__":
    main()
