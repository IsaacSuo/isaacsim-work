"""Deterministic GPU validation for conservative foam-density deposition."""

from __future__ import annotations

import json
import math

import numpy as np

from whitewater.foam_density_warp import FoamDensityComputer


CELL_SIZE = 0.002
axis = np.arange(-0.16, 0.1601, CELL_SIZE, dtype=np.float32)
computer = FoamDensityComputer(
    axis,
    axis,
    maximum_markers=8,
    maximum_support=0.09,
    device="cuda:0",
    grid_dimension=64,
)


def compute(
    positions,
    directions,
    sigma_major,
    sigma_minor,
    peak,
    age=None,
    radius=None,
    kind=None,
):
    count = len(positions)
    return computer.compute(
        np.asarray(positions, dtype=np.float32).reshape(count, 3),
        np.asarray(directions, dtype=np.float32).reshape(count, 2),
        np.asarray(sigma_major, dtype=np.float32),
        np.asarray(sigma_minor, dtype=np.float32),
        np.asarray(peak, dtype=np.float32),
        np.zeros(count, dtype=np.float32)
        if age is None
        else np.asarray(age, dtype=np.float32),
        np.zeros(count, dtype=np.float32)
        if radius is None
        else np.asarray(radius, dtype=np.float32),
        np.zeros(count, dtype=np.int32)
        if kind is None
        else np.asarray(kind, dtype=np.int32),
    )


def truncated_gaussian_integral(peak, sigma_major, sigma_minor):
    return (
        peak
        * 2.0
        * math.pi
        * sigma_major
        * sigma_minor
        * (1.0 - math.exp(-4.5))
    )


def relative_error(actual, expected):
    return abs(actual - expected) / expected


empty = compute([], [], [], [], [])
isotropic = compute(
    [[0.0, 0.0, 0.0]],
    [[1.0, 0.0]],
    [0.018],
    [0.018],
    [0.8],
    age=[0.7],
)
direction = np.asarray([0.6, 0.8], dtype=np.float32)
anisotropic = compute(
    [[0.0, 0.0, 0.0]],
    [direction],
    [0.026],
    [0.009],
    [1.1],
    age=[0.35],
)
single_a = compute(
    [[-0.012, 0.0, 0.006]], [[1.0, 0.0]], [0.02], [0.012], [0.65]
)
single_b = compute(
    [[0.014, 0.0, -0.008]], [[0.0, 1.0]], [0.017], [0.011], [0.45]
)
overlap = compute(
    [[-0.012, 0.0, 0.006], [0.014, 0.0, -0.008]],
    [[1.0, 0.0], [0.0, 1.0]],
    [0.02, 0.017],
    [0.012, 0.011],
    [0.65, 0.45],
)
channels = compute(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    [0.018, 0.016, 0.016],
    [0.012, 0.010, 0.010],
    [0.7, 0.4, 0.9],
    age=[0.25, 0.0, 0.0],
    radius=[0.0, 0.001, 0.004],
    kind=[0, 1, 1],
)

cell_area = CELL_SIZE**2
isotropic_integral = float(np.sum(isotropic["foam_tau"], dtype=np.float64) * cell_area)
anisotropic_integral = float(
    np.sum(anisotropic["foam_tau"], dtype=np.float64) * cell_area
)
isotropic_expected = truncated_gaussian_integral(0.8, 0.018, 0.018)
anisotropic_expected = truncated_gaussian_integral(1.1, 0.026, 0.009)
overlap_residual = overlap["foam_tau"] - (
    single_a["foam_tau"] + single_b["foam_tau"]
)
coverage = -np.expm1(-overlap["foam_tau"].astype(np.float64))

orientation_tau = float(np.sum(anisotropic["foam_tau"], dtype=np.float64))
orientation = np.asarray(
    [
        np.sum(anisotropic["foam_orientation_xx"], dtype=np.float64),
        np.sum(anisotropic["foam_orientation_xz"], dtype=np.float64),
        np.sum(anisotropic["foam_orientation_zz"], dtype=np.float64),
    ]
) / orientation_tau
expected_orientation = np.asarray(
    [direction[0] ** 2, direction[0] * direction[1], direction[1] ** 2]
)

center = len(axis) // 2
bubble_tau_at_center = float(channels["bubble_tau"][center, center])
bubble_mean_radius_at_center = float(
    channels["bubble_radius_weighted"][center, center] / bubble_tau_at_center
)
expected_bubble_mean_radius = (0.4 * 0.001 + 0.9 * 0.004) / (0.4 + 0.9)

rejected_invalid_inputs = {}
invalid_cases = {
    "zero_sigma": dict(directions=[[1.0, 0.0]], major=[0.0], kind=[0]),
    "non_unit_direction": dict(
        directions=[[2.0, 0.0]], major=[0.01], kind=[0]
    ),
    "unknown_kind": dict(directions=[[1.0, 0.0]], major=[0.01], kind=[7]),
    "undersized_support": dict(
        directions=[[1.0, 0.0]], major=[0.031], kind=[0]
    ),
}
for name, values in invalid_cases.items():
    try:
        compute(
            [[0.0, 0.0, 0.0]],
            values["directions"],
            values["major"],
            [0.01],
            [0.5],
            kind=values["kind"],
        )
    except ValueError:
        rejected_invalid_inputs[name] = True
    else:
        rejected_invalid_inputs[name] = False

all_arrays = [array for result in (empty, isotropic, anisotropic, overlap, channels) for array in result.values()]
metrics = {
    "isotropic_integral_relative_error": relative_error(
        isotropic_integral, isotropic_expected
    ),
    "anisotropic_integral_relative_error": relative_error(
        anisotropic_integral, anisotropic_expected
    ),
    "overlap_maximum_absolute_residual": float(np.max(np.abs(overlap_residual))),
    "orientation_maximum_absolute_error": float(
        np.max(np.abs(orientation - expected_orientation))
    ),
    "bubble_mean_radius_at_center": bubble_mean_radius_at_center,
    "bubble_mean_radius_expected": expected_bubble_mean_radius,
    "maximum_coverage": float(np.max(coverage)),
    "empty_output_maximum": float(max(np.max(array) for array in empty.values())),
}
criteria = {
    "all_outputs_finite": all(np.isfinite(array).all() for array in all_arrays),
    "empty_markers_are_zero": metrics["empty_output_maximum"] == 0.0,
    "isotropic_area_is_conservative": metrics["isotropic_integral_relative_error"] < 0.03,
    "anisotropic_area_is_conservative": metrics["anisotropic_integral_relative_error"] < 0.03,
    "anisotropy_preserves_orientation": metrics["orientation_maximum_absolute_error"] < 1.0e-5,
    "overlapping_tau_is_linear": metrics["overlap_maximum_absolute_residual"] < 1.0e-6,
    "coverage_is_bounded": bool(np.all(coverage >= 0.0) and np.all(coverage < 1.0)),
    "foam_and_bubble_channels_are_separate": bool(
        np.max(channels["foam_tau"]) > 0.0
        and np.max(channels["bubble_tau"]) > 0.0
        and np.max(channels["foam_age_weighted"]) > 0.0
    ),
    "bubble_radius_weighting_is_correct": abs(
        bubble_mean_radius_at_center - expected_bubble_mean_radius
    ) < 1.0e-7,
    "invalid_inputs_are_rejected": all(rejected_invalid_inputs.values()),
}
report = {
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": metrics,
    "invalid_input_rejections": rejected_invalid_inputs,
}
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
