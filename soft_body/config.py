"""Pure configuration helpers for soft-body simulation bodies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


SUPPORTED_PHYSICS_KINDS = frozenset({"deformable", "rigid"})


def normalize_body_specs(
    bodies: Sequence[Mapping[str, Any]],
    defaults: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Validate and normalize JSON body records without mutating the input.

    The returned dictionaries use a single ``spawn`` tuple internally while the
    serialized experiment format keeps the more readable ``spawn_x``,
    ``drop_height`` and ``spawn_z`` fields.
    """
    if not bodies:
        raise ValueError("Body config must contain at least one body")

    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(bodies):
        spec = deepcopy(dict(source))
        for key in ("model", "model_height", "spawn_x", "drop_height", "spawn_z"):
            if key not in spec:
                raise ValueError(f"Body {index} is missing required field: {key}")

        physics_kind = str(spec.get("physics_kind", "deformable"))
        if physics_kind not in SUPPORTED_PHYSICS_KINDS:
            raise ValueError(
                f"Unsupported physics_kind for body {index}: {physics_kind}"
            )
        spec["physics_kind"] = physics_kind
        spec["spawn"] = (
            float(spec.pop("spawn_x")),
            float(spec.pop("drop_height")),
            float(spec.pop("spawn_z")),
        )
        spec["model_height"] = float(spec["model_height"])
        if spec["model_height"] <= 0.0:
            raise ValueError(f"Body {index} model_height must be positive")
        spec["model_yaw"] = float(spec.get("model_yaw", 0.0))
        for key, default in defaults.items():
            spec[key] = float(spec.get(key, default))

        if "collision_contact_offset" in spec:
            spec["collision_contact_offset"] = float(
                spec["collision_contact_offset"]
            )
        if "collision_rest_offset" in spec:
            spec["collision_rest_offset"] = float(spec["collision_rest_offset"])
        if (
            "collision_contact_offset" in spec
            and "collision_rest_offset" in spec
            and spec["collision_contact_offset"] < spec["collision_rest_offset"]
        ):
            raise ValueError(
                f"Body {index} collision_contact_offset must be >= collision_rest_offset"
            )
        normalized.append(spec)
    return normalized
