"""Physical mapping from transported whitewater markers to surface kernels."""

from __future__ import annotations

import math

import numpy as np

from .state_machine import WhitewaterState


GAUSSIAN_CUTOFF_SIGMA = 3.0
GAUSSIAN_TRUNCATION = 1.0 - math.exp(-0.5 * GAUSSIAN_CUTOFF_SIGMA**2)
FOAM_BIRTH_FILM_THICKNESS = 0.00030
FOAM_DRAINED_FILM_THICKNESS = 0.00008
FOAM_DRAINAGE_TIME = 0.45
FOAM_TARGET_PEAK_TAU = 0.80
BUBBLE_TARGET_PEAK_TAU = 0.65
MAXIMUM_FOAM_ANISOTROPY = 2.5
FOAM_ANISOTROPY_SPEED = 0.25
FOAM_ANISOTROPY_AGE = 0.30
FOAM_RAFT_SIGMA_BIRTH = 0.006
FOAM_RAFT_SIGMA_MATURE = 0.012
FOAM_RAFT_COALESCENCE_TIME = 0.35


def film_thickness(age):
    """Exponential film drainage without deleting represented liquid volume."""
    age = np.maximum(np.asarray(age, dtype=np.float64), 0.0)
    return FOAM_DRAINED_FILM_THICKNESS + (
        FOAM_BIRTH_FILM_THICKNESS - FOAM_DRAINED_FILM_THICKNESS
    ) * np.exp(-age / FOAM_DRAINAGE_TIME)


def _directions_and_speed(velocity):
    tangent = np.asarray(velocity, dtype=np.float64)[:, (0, 2)]
    speed = np.linalg.norm(tangent, axis=1)
    direction = np.zeros_like(tangent)
    moving = speed > 1.0e-8
    direction[moving] = tangent[moving] / speed[moving, None]
    direction[~moving, 0] = 1.0
    return direction, speed


def _kernel_from_area(area, target_peak, anisotropy, minimum_sigma):
    area = np.asarray(area, dtype=np.float64)
    anisotropy = np.asarray(anisotropy, dtype=np.float64)
    desired_product = area / (
        2.0 * math.pi * target_peak * GAUSSIAN_TRUNCATION
    )
    base_sigma = np.maximum(np.sqrt(desired_product), minimum_sigma)
    anisotropy_root = np.sqrt(anisotropy)
    sigma_major = base_sigma * anisotropy_root
    sigma_minor = base_sigma / anisotropy_root
    peak = area / (
        2.0 * math.pi * sigma_major * sigma_minor * GAUSSIAN_TRUNCATION
    )
    return sigma_major, sigma_minor, peak


def surface_marker_kernels(snapshot, atlas_cell_size):
    """Return conservative Gaussian parameters for foam and surface bubbles."""
    state = np.asarray(snapshot["state"], dtype=np.uint8)
    foam_mask = state == np.uint8(WhitewaterState.SURFACE_FOAM)
    bubble_mask = state == np.uint8(WhitewaterState.SURFACE_BUBBLE)
    selected = foam_mask | bubble_mask
    selected_indices = np.flatnonzero(selected)
    count = len(selected_indices)
    if count == 0:
        return {
            "source_indices": selected_indices.astype(np.int64),
            "positions": np.empty((0, 3), dtype=np.float32),
            "directions": np.empty((0, 2), dtype=np.float32),
            "sigma_major": np.empty(0, dtype=np.float32),
            "sigma_minor": np.empty(0, dtype=np.float32),
            "peak_optical_depth": np.empty(0, dtype=np.float32),
            "age": np.empty(0, dtype=np.float32),
            "radius": np.empty(0, dtype=np.float32),
            "kind": np.empty(0, dtype=np.int32),
            "foam_area": 0.0,
            "bubble_area": 0.0,
        }

    positions = np.asarray(snapshot["position"][selected], dtype=np.float32)
    velocity = np.asarray(snapshot["velocity"][selected], dtype=np.float32)
    age = np.asarray(snapshot["state_age"][selected], dtype=np.float32)
    radius = np.asarray(snapshot["radius"][selected], dtype=np.float64)
    weight = np.asarray(
        snapshot["representative_weight"][selected], dtype=np.float64
    )
    liquid_volume = np.asarray(
        snapshot["liquid_volume"][selected], dtype=np.float64
    )
    directions, speed = _directions_and_speed(velocity)
    kind = bubble_mask[selected].astype(np.int32)

    area = np.empty(count, dtype=np.float64)
    anisotropy = np.ones(count, dtype=np.float64)
    selected_foam = kind == 0
    selected_bubble = ~selected_foam
    if np.any(selected_foam):
        thickness = film_thickness(age[selected_foam])
        area[selected_foam] = liquid_volume[selected_foam] / thickness
        speed_factor = 1.0 - np.exp(
            -speed[selected_foam] / FOAM_ANISOTROPY_SPEED
        )
        age_factor = 1.0 - np.exp(
            -np.maximum(age[selected_foam], 0.0) / FOAM_ANISOTROPY_AGE
        )
        anisotropy[selected_foam] = 1.0 + (
            MAXIMUM_FOAM_ANISOTROPY - 1.0
        ) * speed_factor * age_factor
    if np.any(selected_bubble):
        area[selected_bubble] = (
            math.pi * radius[selected_bubble] ** 2 * weight[selected_bubble]
        )

    target_peak = np.where(
        selected_foam, FOAM_TARGET_PEAK_TAU, BUBBLE_TARGET_PEAK_TAU
    )
    minimum_sigma = np.full(count, float(atlas_cell_size), dtype=np.float64)
    if np.any(selected_foam):
        foam_age = np.maximum(age[selected_foam].astype(np.float64), 0.0)
        minimum_sigma[selected_foam] = np.maximum(
            float(atlas_cell_size),
            FOAM_RAFT_SIGMA_BIRTH
            + (FOAM_RAFT_SIGMA_MATURE - FOAM_RAFT_SIGMA_BIRTH)
            * (1.0 - np.exp(-foam_age / FOAM_RAFT_COALESCENCE_TIME)),
        )
    sigma_major, sigma_minor, peak = _kernel_from_area(
        area, target_peak, anisotropy, minimum_sigma
    )
    return {
        "source_indices": selected_indices.astype(np.int64),
        "positions": positions,
        "directions": directions.astype(np.float32),
        "sigma_major": sigma_major.astype(np.float32),
        "sigma_minor": sigma_minor.astype(np.float32),
        "peak_optical_depth": peak.astype(np.float32),
        "age": age,
        "radius": radius.astype(np.float32),
        "kind": kind,
        "foam_area": float(np.sum(area[selected_foam], dtype=np.float64)),
        "bubble_area": float(np.sum(area[selected_bubble], dtype=np.float64)),
    }
