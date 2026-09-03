"""Versioned, scene-independent flow profiles for whitewater emission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np


REQUIRED_THRESHOLDS = (
    "crest_curvature",
    "spray_outward_velocity",
    "spray_outward_acceleration",
    "gravity_facing_normal",
    "entrainment_convergence",
    "entrainment_rotation",
    "entrainment_inward_velocity",
    "churn_inward_acceleration",
    "churn_agitation",
)
CHANNELS = ("spray", "entrained_air", "churn")
EVENT_GATES = ("tagged_source_contact", "always")
CHURN_SOURCE_POLICIES = (
    "required_tagged_source",
    "optional_tagged_source",
    "global_near_surface",
    "disabled",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_keys(payload, expected, label):
    keys = set(payload)
    expected = set(expected)
    if keys != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - keys)} "
            f"unknown={sorted(keys - expected)}"
        )


def _positive(value, label, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        raise ValueError(f"{label} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return value


def _range(value, label):
    values = tuple(map(float, value))
    if len(values) != 2 or not np.isfinite(values).all() or values[1] <= values[0]:
        raise ValueError(f"{label} must contain two finite increasing values")
    return values


@dataclass(frozen=True)
class FlowProfile:
    name: str
    flow_family: str
    density_kg_m3: float
    surface_tension_n_m: float
    conditioning_sigma_cells: float
    surface_half_width_cells: float
    churn_depth_cells: float
    churn_source_influence_cells: float
    thresholds: object
    spray_impulse_weight: float
    entrainment_inward_weight: float
    channel_gains: object
    churn_source_policy: str
    event_gate: str
    attack_seconds: float
    decay_seconds: float
    baseline_quantile: float
    baseline_multiplier: float
    explicit_noise_floors: object
    birth_rate_density: object

    def __post_init__(self):
        if not self.name or not self.flow_family:
            raise ValueError("Flow profile name and family must be non-empty")
        for name in (
            "density_kg_m3",
            "surface_tension_n_m",
            "conditioning_sigma_cells",
            "surface_half_width_cells",
            "churn_depth_cells",
            "churn_source_influence_cells",
            "attack_seconds",
            "decay_seconds",
            "baseline_multiplier",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("spray_impulse_weight", "entrainment_inward_weight"):
            value = _positive(getattr(self, name), name, allow_zero=True)
            if value > 1.0:
                raise ValueError(f"{name} must lie within 0..1")
            object.__setattr__(self, name, value)
        if not 0.90 <= float(self.baseline_quantile) < 1.0:
            raise ValueError("baseline_quantile must lie within 0.90..1.0")
        if self.churn_source_policy not in CHURN_SOURCE_POLICIES:
            raise ValueError("Unsupported churn_source_policy")
        if self.event_gate not in EVENT_GATES:
            raise ValueError("Unsupported event_gate")
        thresholds = dict(self.thresholds)
        _strict_keys(thresholds, REQUIRED_THRESHOLDS, "thresholds")
        thresholds = {
            name: _range(thresholds[name], f"thresholds.{name}")
            for name in REQUIRED_THRESHOLDS
        }
        gains = dict(self.channel_gains)
        floors = dict(self.explicit_noise_floors)
        rates = dict(self.birth_rate_density)
        for values, label, allow_zero in (
            (gains, "channel_gains", True),
            (floors, "explicit_noise_floors", True),
            (rates, "birth_rate_density", False),
        ):
            _strict_keys(values, CHANNELS, label)
            for channel in CHANNELS:
                values[channel] = _positive(
                    values[channel], f"{label}.{channel}", allow_zero=allow_zero
                )
        if any(value > 1.0 for value in floors.values()):
            raise ValueError("Explicit noise floors must lie within 0..1")
        object.__setattr__(self, "thresholds", MappingProxyType(thresholds))
        object.__setattr__(self, "channel_gains", MappingProxyType(gains))
        object.__setattr__(self, "explicit_noise_floors", MappingProxyType(floors))
        object.__setattr__(self, "birth_rate_density", MappingProxyType(rates))

    @classmethod
    def from_mapping(cls, payload):
        _strict_keys(
            payload,
            ("schema", "product", "name", "flow_family", "physics", "emission", "event", "birth"),
            "flow profile",
        )
        if payload["schema"] != 1 or payload["product"] != "whitewater_v6_flow_profile":
            raise ValueError("Unsupported flow profile schema or product")
        physics = payload["physics"]
        emission = payload["emission"]
        event = payload["event"]
        birth = payload["birth"]
        _strict_keys(physics, ("density_kg_m3", "surface_tension_n_m"), "physics")
        _strict_keys(
            emission,
            (
                "conditioning_sigma_cells",
                "surface_half_width_cells",
                "churn_depth_cells",
                "churn_source_influence_cells",
                "thresholds",
                "spray_impulse_weight",
                "entrainment_inward_weight",
                "channel_gains",
                "churn_source_policy",
                "attack_seconds",
                "decay_seconds",
            ),
            "emission",
        )
        _strict_keys(
            event,
            ("gate", "baseline_quantile", "baseline_multiplier", "explicit_noise_floors"),
            "event",
        )
        _strict_keys(birth, ("rate_density_m3_s" ,), "birth")
        return cls(
            name=str(payload["name"]),
            flow_family=str(payload["flow_family"]),
            density_kg_m3=physics["density_kg_m3"],
            surface_tension_n_m=physics["surface_tension_n_m"],
            conditioning_sigma_cells=emission["conditioning_sigma_cells"],
            surface_half_width_cells=emission["surface_half_width_cells"],
            churn_depth_cells=emission["churn_depth_cells"],
            churn_source_influence_cells=emission["churn_source_influence_cells"],
            thresholds=emission["thresholds"],
            spray_impulse_weight=emission["spray_impulse_weight"],
            entrainment_inward_weight=emission["entrainment_inward_weight"],
            channel_gains=emission["channel_gains"],
            churn_source_policy=emission["churn_source_policy"],
            event_gate=event["gate"],
            attack_seconds=emission["attack_seconds"],
            decay_seconds=emission["decay_seconds"],
            baseline_quantile=event["baseline_quantile"],
            baseline_multiplier=event["baseline_multiplier"],
            explicit_noise_floors=event["explicit_noise_floors"],
            birth_rate_density=birth["rate_density_m3_s"],
        )

    @classmethod
    def load(cls, path):
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def metadata(self):
        return {
            "schema": 1,
            "product": "whitewater_v6_flow_profile",
            "name": self.name,
            "flow_family": self.flow_family,
            "physics": {
                "density_kg_m3": self.density_kg_m3,
                "surface_tension_n_m": self.surface_tension_n_m,
            },
            "emission": {
                "conditioning_sigma_cells": self.conditioning_sigma_cells,
                "surface_half_width_cells": self.surface_half_width_cells,
                "churn_depth_cells": self.churn_depth_cells,
                "churn_source_influence_cells": self.churn_source_influence_cells,
                "thresholds": {key: list(value) for key, value in self.thresholds.items()},
                "spray_impulse_weight": self.spray_impulse_weight,
                "entrainment_inward_weight": self.entrainment_inward_weight,
                "channel_gains": dict(self.channel_gains),
                "churn_source_policy": self.churn_source_policy,
                "attack_seconds": self.attack_seconds,
                "decay_seconds": self.decay_seconds,
            },
            "event": {
                "gate": self.event_gate,
                "baseline_quantile": self.baseline_quantile,
                "baseline_multiplier": self.baseline_multiplier,
                "explicit_noise_floors": dict(self.explicit_noise_floors),
            },
            "birth": {"rate_density_m3_s": dict(self.birth_rate_density)},
        }


def default_impact_profile():
    return FlowProfile.from_mapping(
        {
            "schema": 1,
            "product": "whitewater_v6_flow_profile",
            "name": "default_impact_compatibility",
            "flow_family": "rigid_body_impact",
            "physics": {"density_kg_m3": 997.0, "surface_tension_n_m": 0.072},
            "emission": {
                "conditioning_sigma_cells": 1.0,
                "surface_half_width_cells": 1.5,
                "churn_depth_cells": 4.0,
                "churn_source_influence_cells": 8.0,
                "thresholds": {
                    "crest_curvature": [0.05, 0.50],
                    "spray_outward_velocity": [0.08, 0.60],
                    "spray_outward_acceleration": [0.50, 4.00],
                    "gravity_facing_normal": [-0.10, 0.70],
                    "entrainment_convergence": [0.03, 0.25],
                    "entrainment_rotation": [0.04, 0.30],
                    "entrainment_inward_velocity": [0.05, 0.50],
                    "churn_inward_acceleration": [0.50, 8.00],
                    "churn_agitation": [0.03, 0.35]
                },
                "spray_impulse_weight": 0.60,
                "entrainment_inward_weight": 1.0,
                "channel_gains": {"spray": 1.0, "entrained_air": 1.0, "churn": 1.0},
                "churn_source_policy": "required_tagged_source",
                "attack_seconds": 0.018,
                "decay_seconds": 0.120
            },
            "event": {
                "gate": "tagged_source_contact",
                "baseline_quantile": 0.999,
                "baseline_multiplier": 1.10,
                "explicit_noise_floors": {"spray": 0.0, "entrained_air": 0.0, "churn": 0.0}
            },
            "birth": {
                "rate_density_m3_s": {"spray": 1.20e6, "entrained_air": 1.60e6, "churn": 5.00e6}
            }
        }
    )
