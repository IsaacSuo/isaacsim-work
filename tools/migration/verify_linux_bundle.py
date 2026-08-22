#!/usr/bin/env python3
"""Validate the portable source/assets bundle before running expensive jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SCENES = (
    "alley",
    "apartment",
    "bedroom",
    "city",
    "classroom",
    "elevator",
    "factory",
    "garage",
    "graffiti_warehouse",
    "hospital",
    "mountain",
    "subway2",
    "swamp",
    "warehouse",
)
EXPECTED_HDRI = (
    "bambanani_sunset_8k.exr",
    "bryanston_park_sunrise_8k.exr",
    "charolettenbrunn_park_4k.hdr",
    "noon_grass_4k.hdr",
    "rogland_clear_night_8k.exr",
)


def resolve_command(value: str | None) -> Path | None:
    if not value:
        return None
    expanded = Path(value).expanduser()
    if expanded.is_file():
        return expanded
    found = shutil.which(value)
    return Path(found) if found else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_collisions(root: Path) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        groups.setdefault(relative.casefold(), []).append(relative)
    return [values for values in groups.values() if len(values) > 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also require ISAAC_PYTHON, Blender and FFmpeg executables.",
    )
    parser.add_argument(
        "--hashes",
        action="store_true",
        help="Verify BUNDLE_MANIFEST.sha256 (slow, reads every packaged file).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes_root = Path(os.environ.get("SCENES_ROOT", ROOT / "scenes")).expanduser()
    errors: list[str] = []
    warnings: list[str] = []

    required_source = (
        ROOT / "soft_body_bounce_hero.py",
        ROOT / "experiments/model_material/run_experiments.py",
        ROOT / "experiments/model_material/run_multi_object_videos.py",
        ROOT / "experiments/model_material/render_videos.py",
        ROOT / "configs/scene_experiments.json",
        ROOT / "configs/model_material_experiments.json",
        ROOT / "configs/multi_object_scene_experiments.json",
    )
    for path in required_source:
        if not path.is_file():
            errors.append(f"missing source/config: {path.relative_to(ROOT)}")

    for scene in EXPECTED_SCENES:
        for path in (
            scenes_root / f"{scene}.blend",
            scenes_root / scene / f"{scene}.usdc",
            scenes_root / scene / f"{scene}_sim.usda",
        ):
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing/empty scene asset: {path}")
    for name in EXPECTED_HDRI:
        path = scenes_root / "HDRI" / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing/empty HDRI: {path}")

    for directory, minimum in (
        (ROOT / "assets/archieved_models", 1),
        (ROOT / "assets/simulation_ready_models", 18),
        (ROOT / "output/scene_collision_assets", 1),
    ):
        count = sum(1 for path in directory.glob("*") if path.is_file()) if directory.is_dir() else 0
        if count < minimum:
            errors.append(f"asset directory too small: {directory.relative_to(ROOT)} ({count} files)")

    try:
        configured = json.loads((ROOT / "configs/scene_experiments.json").read_text(encoding="utf-8"))
        if set(configured) != set(EXPECTED_SCENES):
            errors.append("scene_experiments.json does not cover exactly the 14 expected scenes")
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"cannot parse scene config: {error}")

    collisions = case_collisions(ROOT)
    if collisions:
        errors.extend(f"case-only path collision: {', '.join(group)}" for group in collisions)

    if not (ROOT / ".git").is_dir():
        warnings.append(".git history is absent; experiments can run, but provenance is reduced")

    if args.runtime:
        runtime_values = {
            "ISAAC_PYTHON": os.environ.get("ISAAC_PYTHON"),
            "BLENDER_BIN": os.environ.get("BLENDER_BIN", "blender"),
            "FFMPEG_BIN": os.environ.get("FFMPEG_BIN", "ffmpeg"),
        }
        for name, value in runtime_values.items():
            executable = resolve_command(value)
            if executable is None:
                errors.append(f"runtime executable not found: {name}={value!r}")
            else:
                print(f"[runtime] {name}={executable}")

    if args.hashes:
        manifest = ROOT / "BUNDLE_MANIFEST.sha256"
        if not manifest.is_file():
            errors.append("BUNDLE_MANIFEST.sha256 is missing")
        else:
            for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                expected, relative = line.split("  ", 1)
                path = ROOT / Path(relative)
                if not path.is_file():
                    errors.append(f"manifest file missing: {relative}")
                elif sha256(path) != expected:
                    errors.append(f"hash mismatch: {relative}")
                if errors:
                    print(f"[hash-progress] line={line_number}", file=sys.stderr)

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(
        f"bundle_root={ROOT} scenes_root={scenes_root} "
        f"status={'FAIL' if errors else 'OK'} warnings={len(warnings)} errors={len(errors)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
