"""Generate the reproducible 14-scene mixed-material multi-object experiment set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "configs" / "multi_object_scene_experiments.json"
SCENES = (
    "alley",
    "apartment",
    "bedroom",
    "city",
    "classroom",
    "elevator",
    "factory",
    "garage",
    "graffiti_warehouse",
    "mountain",
    "subway2",
    "swamp",
    "warehouse",
    "hospital",
)

# Every archived shape participates through a generated, topology-capped simulation
# STL. This keeps the random pool broad without feeding multi-million-face scans
# directly to the PhysX deformable cooker.
SIMULATION_MODEL_POOL = (
    "banana.stl",
    "bear.stl",
    "bird.stl",
    "cap.stl",
    "carrot.stl",
    "cat.stl",
    "chair.stl",
    "coral.stl",
    "elephant.stl",
    "elephant_1.stl",
    "elephant_2.stl",
    "fish.stl",
    "fish_1.stl",
    "heart.stl",
    "rabbit.stl",
    "statue.stl",
    "tree.stl",
    "tree_1.stl",
)

MATERIAL_SLOTS = (
    ("silicone_cloudy", ("silicone_soft", "silicone_firm")),
    ("rough_white", ("hard_white",)),
    ("brushed_metal", ("hard_metal",)),
)

DROP_LAYOUT = (
    {"offset_x": -0.06, "offset_z": 0.03, "drop_offset": 0.0},
    {"offset_x": 0.05, "offset_z": -0.04, "drop_offset": 0.72},
    {"offset_x": 0.0, "offset_z": 0.05, "drop_offset": 1.44},
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def generate(seed: int):
    rng = random.Random(seed)
    model_deck = list(SIMULATION_MODEL_POOL)
    rng.shuffle(model_deck)
    experiments = []
    for scene_index, scene in enumerate(SCENES):
        # Cycle through a shuffled deck before reshuffling, which keeps selection
        # random but prevents a few models from dominating the 14-scene batch.
        if len(model_deck) < 3:
            model_deck = list(SIMULATION_MODEL_POOL)
            rng.shuffle(model_deck)
        selected_models = [model_deck.pop() for _ in range(3)]
        material_slots = list(MATERIAL_SLOTS)
        rng.shuffle(material_slots)
        bodies = []
        for body_index, (model, material_slot, layout) in enumerate(
            zip(selected_models, material_slots, DROP_LAYOUT)
        ):
            material_preset, profile_choices = material_slot
            profile = rng.choice(profile_choices)
            body = {
                "model": model,
                "model_height": round(rng.uniform(0.50, 0.60), 3),
                "model_yaw": round(rng.uniform(-55.0, 55.0), 1),
                **layout,
                "physics_profile": profile,
                "material_preset": material_preset,
                "collision_contact_offset": 0.03,
                "collision_rest_offset": 0.01,
            }
            bodies.append(body)
        experiments.append(
            {
                "id": f"{scene}_random_mixed_collision",
                "scene": scene,
                "camera_distance_scale": 0.88 if scene == "mountain" else 0.78,
                "bodies": bodies,
            }
        )
    return {
        "seed": seed,
        "selection_policy": "shuffled_deck_without_replacement_within_scene",
        "interbody_collision_layout": "three_near_coaxial_staggered_drops",
        "simulation_model_pool": list(SIMULATION_MODEL_POOL),
        "experiments": experiments,
    }


def main():
    args = parse_args()
    payload = generate(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} seed={args.seed} scenes={len(payload['experiments'])}")


if __name__ == "__main__":
    main()
