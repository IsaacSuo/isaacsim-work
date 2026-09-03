"""Audited, scene-adaptable whitewater simulation components."""

__all__ = [
    "SceneContract",
    "build_collider_fields",
    "open_terrain_collision_sdf",
    "PourSource",
    "MeasuredSurfaceContactTracker",
    "WarpFeatureComputer",
    "load_warp",
]


def __getattr__(name):
    if name in {"WarpFeatureComputer", "load_warp"}:
        from .features_warp import WarpFeatureComputer, load_warp

        return {"WarpFeatureComputer": WarpFeatureComputer, "load_warp": load_warp}[name]
    if name == "SceneContract":
        from .scene_contract import SceneContract

        return SceneContract
    if name == "build_collider_fields":
        from .collision_fields import build_collider_fields

        return build_collider_fields
    if name == "open_terrain_collision_sdf":
        from .terrain_fields import open_terrain_collision_sdf

        return open_terrain_collision_sdf
    if name == "PourSource":
        from .pour_source import PourSource

        return PourSource
    if name == "MeasuredSurfaceContactTracker":
        from .contact_episodes import MeasuredSurfaceContactTracker

        return MeasuredSurfaceContactTracker
    raise AttributeError(name)
