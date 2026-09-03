"""Synthetic gates for conservative multi-scale foam render primitives."""

from __future__ import annotations

import json

import numpy as np

from whitewater.foam_render_payload import (
    FoamRenderClass,
    FoamRenderModel,
    compose_foam_render_payload,
)
from whitewater.liquid_fields import GridSpec
from whitewater.surface_foam import FOAM_DTYPE, REPELLENT_DTYPE


spec = GridSpec((-0.05, -0.05, -0.05), 0.005, (21, 21, 21))
normal = np.zeros(spec.shape + (3,), dtype=np.float32)
normal[..., 1] = 1.0
foam = np.zeros(3, dtype=FOAM_DTYPE)
foam["id"] = (1, 2, 3)
foam["position"] = ((-0.0002, 0.0, 0.0), (0.0002, 0.0, 0.0), (0.02, 0.0, 0.0))
foam["film_area"] = (1.0e-7, 2.0e-6, 8.0e-5)
foam["support_area"] = (3.0e-7, 8.0e-6, 1.2e-4)
foam["principal_direction"] = (1.0, 0.0, 0.0)
foam["anisotropy"] = (1.0, 2.0, 4.0)
foam["random_key"] = (11, 22, 33)


empty_repellents = np.empty(0, dtype=REPELLENT_DTYPE)
patches, rings, quiet = compose_foam_render_payload(
    foam, empty_repellents, normal, spec
)
if len(patches) != 3 or len(rings):
    raise AssertionError("Quiet render payload has the wrong primitive counts")
if not np.array_equal(
    np.sort(patches["render_class"]),
    np.asarray(
        (
            FoamRenderClass.MICRO_DENSITY,
            FoamRenderClass.BUBBLE_CLUSTER,
            FoamRenderClass.MACRO_PATCH,
        ),
        dtype=np.uint8,
    ),
):
    raise AssertionError("Multi-scale render classification is wrong")
if abs(quiet["area_balance_residual_m2"]) > 1.0e-12:
    raise AssertionError("Quiet render payload does not conserve film area")


repellents = np.zeros(3, dtype=REPELLENT_DTYPE)
repellents["id"] = (101, 102, 103)
repellents["position"] = ((0.0, 0.0, 0.0), (0.0205, 0.0, 0.0), (0.045, 0.0, 0.0))
repellents["radius"] = (0.0008, 0.004, 0.001)
repellents["strength"] = (0.8, 0.9, 0.95)
repellents["lifetime"] = (1.0, 1.0, 1.0)
repellents["age"] = (0.2, 0.1, 0.0)
repellents["random_key"] = (77, 88, 99)
patches, rings, metrics = compose_foam_render_payload(
    foam, repellents, normal, spec, FoamRenderModel()
)
if metrics["affected_patches"] < 2 or len(rings) != 2:
    raise AssertionError("Repellents did not create patch holes and rim primitives")
if metrics["orphan_repellents"] != 1:
    raise AssertionError("An orphan repellent was not rejected explicitly")
if np.any(rings["host_patch_count"] < 1) or np.any(
    rings["enclosure_fraction"] < FoamRenderModel().minimum_enclosure_fraction
):
    raise AssertionError("A ring lacks an enclosed host-film cluster")
if not np.any(patches["hole_fraction"] > 0.0):
    raise AssertionError("Patch hole fractions remain zero")
if np.any(rings["outer_radius"] <= rings["inner_radius"]):
    raise AssertionError("Foam rings have non-positive width")
if np.any((rings["coverage"] <= 0.0) | (rings["coverage"] > 0.950001)):
    raise AssertionError("Foam ring coverage is invalid")
if abs(metrics["area_balance_residual_m2"]) > 2.0e-11:
    raise AssertionError("Repellent hole/rim transfer does not conserve film area")
if not np.allclose(np.linalg.norm(patches["normal"], axis=1), 1.0, atol=2.0e-6):
    raise AssertionError("Patch normals are not unit length")
if np.max(
    np.abs(
        np.sum(patches["normal"] * patches["principal_direction"], axis=1)
    )
) > 2.0e-6:
    raise AssertionError("Patch directions are not tangent")
repeat_patches, repeat_rings, _ = compose_foam_render_payload(
    foam, repellents, normal, spec, FoamRenderModel()
)
if not np.array_equal(patches, repeat_patches) or not np.array_equal(
    rings, repeat_rings
):
    raise AssertionError("Foam render payload is not deterministic")


report = {
    "valid": True,
    "patches": len(patches),
    "rings": len(rings),
    "affected_patches": metrics["affected_patches"],
    "source_film_area_m2": metrics["source_film_area_m2"],
    "patch_film_area_m2": metrics["patch_film_area_m2"],
    "ring_film_area_m2": metrics["ring_film_area_m2"],
    "area_balance_residual_m2": metrics["area_balance_residual_m2"],
    "render_class_counts": metrics["render_class_counts"],
}
print(json.dumps(report, indent=2))
