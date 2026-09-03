"""Deterministic, conservative primitives for unified whitewater states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class WhitewaterState(IntEnum):
    DEAD = 0
    SPRAY = 1
    SURFACE_FOAM = 2
    ENTRAINED_BUBBLE = 3
    SURFACE_BUBBLE = 4


class EmissionChannel(IntEnum):
    SPRAY = 1
    ENTRAINED_AIR = 2


@dataclass(frozen=True)
class FluidProperties:
    liquid_density: float = 998.2
    gas_density: float = 1.204
    dynamic_viscosity: float = 1.002e-3
    surface_tension: float = 0.0728
    gravity: float = 9.81


MARKER_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("state", "u1"),
        ("source_particle_id", "<i4"),
        ("birth_sample", "<i4"),
        ("birth_time", "<f8"),
        ("state_age", "<f4"),
        ("position", "<f4", (3,)),
        ("velocity", "<f4", (3,)),
        ("radius", "<f4"),
        ("liquid_volume", "<f4"),
        ("gas_volume", "<f4"),
        ("representative_weight", "<f4"),
        ("shape", "<f4", (3,)),
        ("random_key", "<u8"),
    ]
)


def splitmix64(values):
    """Counter-based hash with stable uint64 behavior on every platform."""
    values = np.asarray(values, dtype=np.uint64).copy()
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return values ^ (values >> np.uint64(31))


def counter_keys(source_ids, channel, counters, seed=0):
    source_ids = np.asarray(source_ids, dtype=np.uint64)
    counters = np.asarray(counters, dtype=np.uint64)
    mask = (1 << 64) - 1
    channel_key = np.uint64((int(channel) * 0xD6E8FEB86659FD93) & mask)
    seed_key = np.uint64(int(seed) & mask)
    mixed = (
        source_ids
        ^ channel_key
        ^ (counters * np.uint64(0xA0761D6478BD642F))
        ^ seed_key
    )
    return splitmix64(mixed)


def uniform01_from_keys(keys):
    keys = np.asarray(keys, dtype=np.uint64)
    return (keys >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


class DeterministicEmissionReservoir:
    """Integrate expected births without timestep-dependent Bernoulli noise."""

    def __init__(self, particle_count, channel, seed=0):
        self.particle_count = int(particle_count)
        self.channel = int(channel)
        self.seed = int(seed)
        ids = np.arange(self.particle_count, dtype=np.uint64)
        initial_keys = counter_keys(
            ids,
            self.channel,
            np.zeros(self.particle_count, dtype=np.uint64),
            self.seed,
        )
        self.accumulator = uniform01_from_keys(initial_keys)
        self.emission_count = np.zeros(self.particle_count, dtype=np.uint32)

    def advance(self, expected_births):
        expected_births = np.asarray(expected_births, dtype=np.float64)
        if expected_births.shape != (self.particle_count,):
            raise ValueError(f"Expected shape {(self.particle_count,)}, got {expected_births.shape}")
        if not np.isfinite(expected_births).all() or np.any(expected_births < 0.0):
            raise ValueError("Expected births must be finite and non-negative")
        accumulated = self.accumulator + expected_births
        births_per_source = np.floor(accumulated).astype(np.uint32)
        self.accumulator = accumulated - births_per_source
        active_sources = np.flatnonzero(births_per_source)
        if not len(active_sources):
            return {
                "source_particle_id": np.empty(0, dtype=np.int32),
                "source_emission_index": np.empty(0, dtype=np.uint32),
                "random_key": np.empty(0, dtype=np.uint64),
            }
        repeated_sources = np.repeat(active_sources, births_per_source[active_sources])
        offsets = np.concatenate(
            [np.arange(count, dtype=np.uint32) for count in births_per_source[active_sources]]
        )
        emission_indices = self.emission_count[repeated_sources] + offsets
        self.emission_count[active_sources] += births_per_source[active_sources]
        keys = counter_keys(
            repeated_sources.astype(np.uint64),
            self.channel,
            emission_indices.astype(np.uint64) + np.uint64(1),
            self.seed,
        )
        return {
            "source_particle_id": repeated_sources.astype(np.int32),
            "source_emission_index": emission_indices,
            "random_key": keys,
        }


def sphere_volume(radius):
    radius = np.asarray(radius, dtype=np.float64)
    return (4.0 / 3.0) * np.pi * radius**3


def radius_from_volume(volume):
    volume = np.asarray(volume, dtype=np.float64)
    if np.any(volume < 0.0):
        raise ValueError("Volume cannot be negative")
    return np.cbrt(volume * (3.0 / (4.0 * np.pi)))


def bubble_terminal_velocity(radius, properties=FluidProperties(), iterations=12):
    """Solve buoyancy/drag balance using the Schiller-Naumann drag law."""
    radius = np.asarray(radius, dtype=np.float64)
    if np.any(radius <= 0.0) or not np.isfinite(radius).all():
        raise ValueError("Bubble radii must be finite and positive")
    diameter = 2.0 * radius
    density_difference = properties.liquid_density - properties.gas_density
    velocity = np.maximum(
        2.0
        * density_difference
        * properties.gravity
        * radius**2
        / (9.0 * properties.dynamic_viscosity),
        1.0e-6,
    )
    for _ in range(iterations):
        reynolds = np.maximum(
            properties.liquid_density
            * velocity
            * diameter
            / properties.dynamic_viscosity,
            1.0e-8,
        )
        drag = np.where(
            reynolds < 1000.0,
            24.0 / reynolds * (1.0 + 0.15 * reynolds**0.687),
            0.44,
        )
        velocity = np.sqrt(
            4.0
            * density_difference
            * properties.gravity
            * diameter
            / (3.0 * drag * properties.liquid_density)
        )
    return np.minimum(velocity, 0.35)


def bubble_shape(radius, rise_velocity, properties=FluidProperties()):
    """Return volume-preserving oblate scales from Eotvos and Weber numbers."""
    radius = np.asarray(radius, dtype=np.float64)
    rise_velocity = np.asarray(rise_velocity, dtype=np.float64)
    diameter = 2.0 * radius
    density_difference = properties.liquid_density - properties.gas_density
    eotvos = (
        density_difference * properties.gravity * diameter**2 / properties.surface_tension
    )
    weber = (
        properties.liquid_density * rise_velocity**2 * diameter / properties.surface_tension
    )
    axis_ratio = 1.0 / np.sqrt(1.0 + 0.163 * eotvos**0.757 + 0.05 * weber)
    axis_ratio = np.clip(axis_ratio, 0.35, 1.0)
    lateral = axis_ratio ** (-1.0 / 3.0)
    vertical = axis_ratio ** (2.0 / 3.0)
    return np.column_stack((lateral, vertical, lateral)).astype(np.float32)


def make_markers(
    marker_ids,
    state,
    source_particle_ids,
    birth_sample,
    birth_time,
    positions,
    velocities,
    radii,
    representative_weights,
    random_keys,
):
    count = len(source_particle_ids)
    markers = np.zeros(count, dtype=MARKER_DTYPE)
    markers["id"] = np.asarray(marker_ids, dtype=np.uint64)
    markers["state"] = np.uint8(state)
    markers["source_particle_id"] = np.asarray(source_particle_ids, dtype=np.int32)
    markers["birth_sample"] = int(birth_sample)
    markers["birth_time"] = float(birth_time)
    markers["position"] = np.asarray(positions, dtype=np.float32)
    markers["velocity"] = np.asarray(velocities, dtype=np.float32)
    markers["radius"] = np.asarray(radii, dtype=np.float32)
    markers["representative_weight"] = np.asarray(representative_weights, dtype=np.float32)
    markers["random_key"] = np.asarray(random_keys, dtype=np.uint64)
    markers["shape"] = 1.0
    volumes = sphere_volume(markers["radius"]).astype(np.float32)
    if state in (WhitewaterState.ENTRAINED_BUBBLE, WhitewaterState.SURFACE_BUBBLE):
        markers["gas_volume"] = volumes * markers["representative_weight"]
        rise = bubble_terminal_velocity(markers["radius"])
        markers["shape"] = bubble_shape(markers["radius"], rise)
    elif state in (WhitewaterState.SPRAY, WhitewaterState.SURFACE_FOAM):
        markers["liquid_volume"] = volumes * markers["representative_weight"]
    else:
        raise ValueError(f"Cannot birth marker in state {state}")
    return markers


def transition_state(markers, selection, new_state):
    selection = np.asarray(selection, dtype=bool)
    before_gas = float(markers["gas_volume"][selection].sum(dtype=np.float64))
    before_liquid = float(markers["liquid_volume"][selection].sum(dtype=np.float64))
    markers["state"][selection] = np.uint8(new_state)
    markers["state_age"][selection] = 0.0
    after_gas = float(markers["gas_volume"][selection].sum(dtype=np.float64))
    after_liquid = float(markers["liquid_volume"][selection].sum(dtype=np.float64))
    if before_gas != after_gas or before_liquid != after_liquid:
        raise RuntimeError("State transition changed a conserved phase volume")


def advance_kinematics(markers, dt, gravity=9.81):
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    alive = markers["state"] != np.uint8(WhitewaterState.DEAD)
    spray = markers["state"] == np.uint8(WhitewaterState.SPRAY)
    bubbles = markers["state"] == np.uint8(WhitewaterState.ENTRAINED_BUBBLE)
    markers["velocity"][spray, 1] -= np.float32(gravity * dt)
    if np.any(bubbles):
        rise = bubble_terminal_velocity(markers["radius"][bubbles]).astype(np.float32)
        radius = markers["radius"][bubbles].astype(np.float64)
        response_time = np.maximum(
            2.0 * FluidProperties().liquid_density * radius**2
            / (9.0 * FluidProperties().dynamic_viscosity),
            1.0e-3,
        )
        relaxation = (1.0 - np.exp(-dt / response_time)).astype(np.float32)
        vertical = markers["velocity"][bubbles, 1]
        markers["velocity"][bubbles, 1] = vertical + (rise - vertical) * relaxation
        markers["shape"][bubbles] = bubble_shape(markers["radius"][bubbles], rise)
    markers["position"][alive] += markers["velocity"][alive] * np.float32(dt)
    markers["state_age"][alive] += np.float32(dt)
