"""Strict artist-assisted production configuration for whitewater v6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .flow_profiles import FlowProfile
from .scene_contract import SceneContract


PRODUCT = "whitewater_v6_production_configuration"
SCHEMA = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict(payload, keys, label):
    if not isinstance(payload, dict) or set(payload) != set(keys):
        actual = set(payload) if isinstance(payload, dict) else set()
        raise ValueError(
            f"{label} keys differ: missing={sorted(set(keys)-actual)} "
            f"unknown={sorted(actual-set(keys))}"
        )


def positive(value, label, zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not zero):
        raise ValueError(f"{label} must be finite and {'non-negative' if zero else 'positive'}")
    return value


def integer(value, label, zero=False):
    number = positive(value, label, zero=zero)
    result = int(number)
    if float(result) != number:
        raise ValueError(f"{label} must be an integer")
    return result


def resolve(value, base, label, required=True):
    if value is None:
        if required:
            raise ValueError(f"{label} is required")
        return None
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    return path


@dataclass(frozen=True)
class ProductionConfig:
    path: Path
    name: str
    paths: dict
    physical: dict
    splashsurf: dict
    whitewater: dict
    render: dict
    manual_setup: dict
    scene_contract: SceneContract
    flow_profile: FlowProfile

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        base = path.parent
        payload = json.loads(path.read_text(encoding="utf-8"))
        strict(
            payload,
            (
                "schema", "product", "name", "paths", "physical",
                "splashsurf", "whitewater", "render", "manual_setup",
            ),
            "production configuration",
        )
        if payload["schema"] != SCHEMA or payload["product"] != PRODUCT:
            raise ValueError("Unsupported production configuration schema/product")
        if not isinstance(payload["name"], str) or not payload["name"].strip():
            raise ValueError("name must be non-empty")
        paths = payload["paths"]
        strict(
            paths,
            (
                "scene_contract", "flow_profile", "source_manifest",
                "source_directory", "domain_manifest", "particle_directory",
                "plateau_directory", "marker_directory", "pending_proxy_directory",
                "blender_scene", "hdri", "output_root", "isaac_python",
                "splashsurf_executable", "blender_executable", "ffmpeg_executable",
            ),
            "paths",
        )
        resolved = {
            name: resolve(value, base, f"paths.{name}")
            for name, value in paths.items()
            if name != "output_root"
        }
        if paths["output_root"] is None:
            raise ValueError("paths.output_root is required, but need not exist yet")
        resolved["output_root"] = resolve(
            paths["output_root"], base, "paths.output_root", required=False
        )
        for executable in (
            "isaac_python", "splashsurf_executable", "blender_executable", "ffmpeg_executable"
        ):
            if not resolved[executable].is_file():
                raise FileNotFoundError(resolved[executable])
        for directory in (
            "source_directory", "particle_directory", "plateau_directory",
            "marker_directory", "pending_proxy_directory",
        ):
            if not resolved[directory].is_dir():
                raise NotADirectoryError(resolved[directory])
        for filename in (
            "scene_contract", "flow_profile", "source_manifest", "domain_manifest",
            "blender_scene", "hdri",
        ):
            if not resolved[filename].is_file():
                raise FileNotFoundError(resolved[filename])

        scene = SceneContract.load(resolved["scene_contract"])
        profile = FlowProfile.load(resolved["flow_profile"])
        if scene.schema != 2 or scene.terrain is None:
            raise ValueError("Production configuration requires schema-2 open terrain")
        physical = payload["physical"]
        strict(physical, ("particle_spacing_m", "water_level_m"), "physical")
        physical = {
            "particle_spacing_m": positive(physical["particle_spacing_m"], "particle spacing"),
            "water_level_m": float(physical["water_level_m"]),
        }
        if not np.isfinite(physical["water_level_m"]):
            raise ValueError("water level must be finite")

        splash = payload["splashsurf"]
        strict(
            splash,
            (
                "workers", "threads_per_worker", "particle_radius_m",
                "smoothing_length", "cube_size", "surface_threshold",
                "mesh_smoothing_iterations", "normal_smoothing_iterations",
                "shoreline_mode", "shoreline_reference_ply",
                "minimum_particle_layers", "shoreline_erosion_cells",
                "terrain_raster_bounds_xz", "terrain_raster_axis_dtype",
            ),
            "splashsurf",
        )
        shoreline_mode = splash["shoreline_mode"]
        if shoreline_mode not in {"terrain", "reference-ply"}:
            raise ValueError(
                "Production shoreline_mode must be terrain or reference-ply; "
                "per-frame dynamic masks are experimental and temporally unstable"
            )
        reference = resolve(
            splash["shoreline_reference_ply"],
            base,
            "splashsurf.shoreline_reference_ply",
            required=shoreline_mode == "reference-ply",
        )
        if shoreline_mode != "reference-ply" and reference is not None:
            raise ValueError(
                "shoreline_reference_ply must be null unless shoreline_mode is reference-ply"
            )
        if reference is not None and not reference.is_file():
            raise FileNotFoundError(reference)
        bounds = splash["terrain_raster_bounds_xz"]
        if bounds is not None:
            bounds = tuple(map(float, bounds))
            if len(bounds) != 4 or not bounds[0] < bounds[1] or not bounds[2] < bounds[3]:
                raise ValueError("terrain_raster_bounds_xz must be [xmin,xmax,zmin,zmax]")
        splash = {
            **splash,
            "workers": integer(splash["workers"], "workers"),
            "threads_per_worker": integer(splash["threads_per_worker"], "threads"),
            "particle_radius_m": positive(splash["particle_radius_m"], "particle radius"),
            "smoothing_length": positive(splash["smoothing_length"], "smoothing length"),
            "cube_size": positive(splash["cube_size"], "cube size"),
            "surface_threshold": positive(splash["surface_threshold"], "surface threshold"),
            "mesh_smoothing_iterations": integer(splash["mesh_smoothing_iterations"], "mesh smoothing", zero=True),
            "normal_smoothing_iterations": integer(splash["normal_smoothing_iterations"], "normal smoothing", zero=True),
            "shoreline_reference_ply": str(reference) if reference else None,
            "minimum_particle_layers": integer(splash["minimum_particle_layers"], "minimum layers"),
            "shoreline_erosion_cells": integer(splash["shoreline_erosion_cells"], "erosion", zero=True),
            "terrain_raster_bounds_xz": bounds,
        }
        if splash["terrain_raster_axis_dtype"] not in {"float32", "float64"}:
            raise ValueError("terrain_raster_axis_dtype must be float32 or float64")
        if bounds is not None:
            for span, label in (
                (bounds[1] - bounds[0], "terrain raster X span"),
                (bounds[3] - bounds[2], "terrain raster Z span"),
            ):
                cells = span / physical["particle_spacing_m"]
                if not np.isclose(cells, round(cells), atol=1.0e-7, rtol=0.0):
                    raise ValueError(f"{label} must align to particle spacing")

        whitewater = payload["whitewater"]
        strict(
            whitewater,
            ("water_body_id", "source_samples", "water_body_seed_xz", "splash_retention_radius_m"),
            "whitewater",
        )
        source_samples = tuple(map(int, whitewater["source_samples"]))
        if (
            not source_samples
            or any(sample < 0 for sample in source_samples)
            or source_samples != tuple(sorted(set(source_samples)))
            or any(float(source) != sample for source, sample in zip(whitewater["source_samples"], source_samples))
        ):
            raise ValueError("whitewater.source_samples must be non-negative sorted unique integers")
        if not isinstance(whitewater["water_body_id"], str) or not whitewater["water_body_id"].strip():
            raise ValueError("whitewater.water_body_id must be non-empty")
        seed = tuple(map(float, whitewater["water_body_seed_xz"]))
        if len(seed) != 2 or not np.isfinite(seed).all():
            raise ValueError("water_body_seed_xz must contain two finite values")
        whitewater = {
            **whitewater,
            "water_body_id": whitewater["water_body_id"].strip(),
            "source_samples": source_samples,
            "water_body_seed_xz": seed,
            "splash_retention_radius_m": positive(
                whitewater["splash_retention_radius_m"], "splash retention radius"
            ),
        }

        render = payload["render"]
        strict(
            render,
            (
                "camera_preset", "camera_eye_xyz_m", "camera_target_xyz_m",
                "camera_lens_mm", "camera_sensor_width_mm", "camera_clip_start_m",
                "impactor_collider_id", "resolution", "samples", "fps",
                "hdri_strength", "micro_pixel_radius", "hero_pixel_radius",
            ),
            "render",
        )
        if render["camera_preset"] != "custom":
            raise ValueError("Production rendering requires an explicit custom camera")
        eye = tuple(map(float, render["camera_eye_xyz_m"]))
        target = tuple(map(float, render["camera_target_xyz_m"]))
        if len(eye) != 3 or len(target) != 3 or not np.isfinite(eye + target).all():
            raise ValueError("Camera eye and target must each contain three finite metres")
        if np.linalg.norm(np.asarray(eye) - np.asarray(target)) <= 1.0e-6:
            raise ValueError("Camera eye and target must differ")
        impactor_id = render["impactor_collider_id"]
        impactors = [
            collider for collider in scene.colliders
            if collider.identifier == impactor_id
        ]
        if len(impactors) != 1 or impactors[0].shape != "sphere":
            raise ValueError("Production renderer currently requires one selected sphere impactor")
        render = {
            **render,
            "camera_eye_xyz_m": eye,
            "camera_target_xyz_m": target,
            "camera_lens_mm": positive(render["camera_lens_mm"], "camera lens"),
            "camera_sensor_width_mm": positive(render["camera_sensor_width_mm"], "camera sensor width"),
            "camera_clip_start_m": positive(render["camera_clip_start_m"], "camera clip start"),
            "impactor_collider_id": impactor_id,
            "impactor_radius_m": impactors[0].parameters["radius"],
            "resolution": integer(render["resolution"], "render resolution"),
            "samples": integer(render["samples"], "render samples"),
            "fps": positive(render["fps"], "render fps"),
            "hdri_strength": positive(render["hdri_strength"], "HDRI strength", zero=True),
            "micro_pixel_radius": positive(render["micro_pixel_radius"], "micro pixel radius"),
            "hero_pixel_radius": positive(render["hero_pixel_radius"], "hero pixel radius"),
        }
        if render["micro_pixel_radius"] >= render["hero_pixel_radius"]:
            raise ValueError("micro pixel threshold must be below hero threshold")

        manual = payload["manual_setup"]
        strict(
            manual,
            ("terrain_selected", "water_body_domain_reviewed", "collider_roles_reviewed", "shoreline_reviewed", "camera_reviewed", "notes"),
            "manual_setup",
        )
        for flag in (
            "terrain_selected", "water_body_domain_reviewed", "collider_roles_reviewed", "shoreline_reviewed", "camera_reviewed"
        ):
            if not isinstance(manual[flag], bool) or not manual[flag]:
                raise ValueError(f"manual_setup.{flag} must be explicitly true")
        if not isinstance(manual["notes"], str):
            raise ValueError("manual_setup.notes must be a string")

        source_manifest = json.loads(resolved["source_manifest"].read_text(encoding="utf-8"))
        available = {int(row["sample_index"]) for row in source_manifest.get("samples", [])}
        missing = set(source_samples) - available
        if missing:
            raise ValueError(f"Configured source samples are missing: {sorted(missing)}")
        domain_manifest = json.loads(resolved["domain_manifest"].read_text(encoding="utf-8"))
        domain_spacing = float(domain_manifest["configuration"]["particle_spacing"])
        if not np.isclose(domain_spacing, physical["particle_spacing_m"], atol=1.0e-12, rtol=0.0):
            raise ValueError("Physical spacing differs from domain manifest")
        domain_source = domain_manifest.get("source", {})
        if Path(domain_source.get("manifest", "")).resolve() != resolved["source_manifest"]:
            raise ValueError("Domain manifest is bound to a different source manifest")
        if Path(domain_source.get("directory", "")).resolve() != resolved["source_directory"]:
            raise ValueError("Domain manifest is bound to a different source directory")
        if resolved["particle_directory"] != resolved["source_directory"]:
            raise ValueError("Domain-mode particle_directory must equal source_directory")
        body_ids = {str(body["body_id"]) for body in domain_manifest.get("bodies", [])}
        if whitewater["water_body_id"] not in body_ids:
            raise ValueError(
                f"Unknown water body {whitewater['water_body_id']!r}; available={sorted(body_ids)}"
            )
        classified_samples = {int(row["sample_index"]) for row in domain_manifest.get("samples", [])}
        unclassified = set(source_samples) - classified_samples
        if unclassified:
            raise ValueError(f"Configured source samples are not classified: {sorted(unclassified)}")
        return cls(
            path, payload["name"].strip(), resolved, physical, splash,
            whitewater, render, manual, scene, profile,
        )

    def resolved_metadata(self):
        input_paths = {
            name: {
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for name, path in self.paths.items()
            if name != "output_root"
        }
        payload = {
            "schema": SCHEMA,
            "product": "whitewater_v6_resolved_production_configuration",
            "name": self.name,
            "source_configuration": str(self.path),
            "source_configuration_sha256": sha256_file(self.path),
            "paths": {**input_paths, "output_root": str(self.paths["output_root"])},
            "physical": self.physical,
            "splashsurf": self.splashsurf,
            "whitewater": self.whitewater,
            "render": self.render,
            "manual_setup": self.manual_setup,
            "scene_contract": self.scene_contract.metadata(),
            "flow_profile": self.flow_profile.metadata(),
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        payload["configuration_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    def splashsurf_command(self, output_directory):
        seed_x, seed_z = self.whitewater["water_body_seed_xz"]
        values = [
            str(self.paths["isaac_python"]),
            str(Path(__file__).resolve().parents[1] / "build_splashsurf_sequence.py"),
            str(self.paths["particle_directory"]),
            str(self.paths["scene_contract"]),
            str(Path(output_directory).resolve()),
            "--water-level", str(self.physical["water_level_m"]),
            "--spacing", str(self.physical["particle_spacing_m"]),
            "--workers", str(self.splashsurf["workers"]),
            "--threads-per-worker", str(self.splashsurf["threads_per_worker"]),
            "--particle-radius", str(self.splashsurf["particle_radius_m"]),
            "--smoothing-length", str(self.splashsurf["smoothing_length"]),
            "--cube-size", str(self.splashsurf["cube_size"]),
            "--surface-threshold", str(self.splashsurf["surface_threshold"]),
            "--mesh-smoothing-iters", str(self.splashsurf["mesh_smoothing_iterations"]),
            "--normal-smoothing-iters", str(self.splashsurf["normal_smoothing_iterations"]),
            "--shoreline-mode", self.splashsurf["shoreline_mode"],
            "--minimum-layers", str(self.splashsurf["minimum_particle_layers"]),
            "--shoreline-erosion-cells", str(self.splashsurf["shoreline_erosion_cells"]),
            "--impact-x", str(seed_x), "--impact-z", str(seed_z),
            "--impact-radius", str(self.whitewater["splash_retention_radius_m"]),
            "--terrain-raster-axis-dtype", self.splashsurf["terrain_raster_axis_dtype"],
            "--terrain-source-mode", "scene-contract",
            "--splashsurf", str(self.paths["splashsurf_executable"]),
            "--domain-manifest", str(self.paths["domain_manifest"]),
            "--domain-body-id", self.whitewater["water_body_id"],
            "--frames", *map(str, self.whitewater["source_samples"]),
        ]
        if self.splashsurf["shoreline_reference_ply"]:
            values.extend(["--shoreline-particles-ply", self.splashsurf["shoreline_reference_ply"]])
        if self.splashsurf["terrain_raster_bounds_xz"]:
            values.extend(["--terrain-raster-bounds", *map(str, self.splashsurf["terrain_raster_bounds_xz"])])
        return values

    def render_command(self, surface_directory="{surface_reconstruction_cache}/surface", output_directory="{stage_cache}"):
        eye = self.render["camera_eye_xyz_m"]
        target = self.render["camera_target_xyz_m"]
        return [
            str(self.paths["blender_executable"]),
            str(self.paths["blender_scene"]),
            "--background", "--python",
            str(Path(__file__).resolve().parents[1] / "render_whitewater_v6_plateau_sequence.py"),
            "--",
            str(self.paths["plateau_directory"]),
            str(self.paths["marker_directory"]),
            str(self.paths["pending_proxy_directory"]),
            str(self.paths["source_directory"]),
            surface_directory,
            output_directory,
            "--samples", str(self.render["samples"]),
            "--resolution", str(self.render["resolution"]),
            "--fps", str(self.render["fps"]),
            "--source-samples", *map(str, self.whitewater["source_samples"]),
            "--camera", "custom",
            "--camera-eye", *map(str, eye),
            "--camera-target", *map(str, target),
            "--camera-lens-mm", str(self.render["camera_lens_mm"]),
            "--camera-sensor-width-mm", str(self.render["camera_sensor_width_mm"]),
            "--camera-clip-start-m", str(self.render["camera_clip_start_m"]),
            "--surface-index-mode", "source-sample",
            "--impactor-radius", str(self.render["impactor_radius_m"]),
            "--micro-pixel-radius", str(self.render["micro_pixel_radius"]),
            "--hero-pixel-radius", str(self.render["hero_pixel_radius"]),
            "--hdri", str(self.paths["hdri"]),
            "--hdri-strength", str(self.render["hdri_strength"]),
        ]
