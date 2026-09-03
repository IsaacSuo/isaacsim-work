"""Typed NumPy wrapper for the native Isaac Sim Diffuse buffer bridge."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


_BIN = Path(__file__).resolve().parent / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import _physx_diffuse_bridge as _native


def abi_info() -> dict:
    return dict(_native.abi_info())


def probe(particle_set_path: str) -> dict:
    return dict(_native.probe(particle_set_path))


def particle_system_info(particle_system_path: str) -> dict:
    return dict(_native.particle_system_info(particle_system_path))


def set_full_diffuse_advection(particle_system_path: str, enabled: bool) -> None:
    _native.set_full_diffuse_advection(particle_system_path, bool(enabled))


def read_diffuse(particle_set_path: str) -> dict:
    raw = _native.read_diffuse(particle_set_path)
    count = int(raw["active_count"])
    position_lifetime = np.frombuffer(raw["position_lifetime"], dtype="<f4").reshape(count, 4)
    velocity = np.frombuffer(raw["velocity"], dtype="<f4").reshape(count, 4)
    return {
        "active_count": count,
        "max_count": int(raw["max_count"]),
        "position_lifetime": position_lifetime,
        "velocity": velocity,
    }


def read_particle_frame(particle_set_path: str) -> dict:
    raw = _native.read_particle_frame(particle_set_path)
    primary_count = int(raw["primary_active_count"])
    diffuse_count = int(raw["diffuse_active_count"])
    return {
        "primary_active_count": primary_count,
        "primary_max_count": int(raw["primary_max_count"]),
        "diffuse_active_count": diffuse_count,
        "diffuse_max_count": int(raw["diffuse_max_count"]),
        "primary_position_inv_mass": np.frombuffer(
            raw["primary_position_inv_mass"], dtype="<f4"
        ).reshape(primary_count, 4),
        "primary_velocity": np.frombuffer(raw["primary_velocity"], dtype="<f4").reshape(
            primary_count, 4
        ),
        "diffuse_position_lifetime": np.frombuffer(
            raw["diffuse_position_lifetime"], dtype="<f4"
        ).reshape(diffuse_count, 4),
        "diffuse_velocity": np.frombuffer(raw["diffuse_velocity"], dtype="<f4").reshape(
            diffuse_count, 4
        ),
    }
