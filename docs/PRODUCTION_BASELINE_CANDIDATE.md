# Production baseline candidate

This document records the currently validated Isaac Sim/PhysX to Blender
production path while the implementation is still changing. It is a repeatable
regression candidate, not a frozen M0 milestone and not the public benchmark
format. Its configuration and acceptance results may evolve with the code.

## Currently validated scope

- Isaac Sim 6.0 creates fixed-topology animated USD caches at 60 FPS.
- Four physics profiles cover soft/firm silicone and hard white/metal bodies.
- Hard visual materials use PhysX rigid bodies; soft materials use volume
  deformables. Each rigid model has an explicit convex-decomposition or SDF
  collision policy recorded in `configs/multi_object_scene_experiments.json`.
- The current multi-object regression matrix contains exactly 14 scenes and three
  different models/materials per scene.
- Scene support is extracted from bounded real environment geometry. Exact
  reusable collision assets must pass `tools/audit/audit_scene_collision_assets.py`.
- Finite-support free falls and downhill motion on uneven terrain are validated
  by exact mesh contact and motion/visual checks, never by a single world-Y
  support threshold.
- Blender Cycles consumes the animated USD without rerunning physics and uses
  the original `.blend`, workspace HDRI, authored lights, validated camera, and
  per-body material preset.

## Authoritative configuration

- `configs/scene_experiments.json`: support height, spawn point, collision crop,
  authored environment prim, camera, lighting, and free-fall policy.
- `configs/model_material_experiments.json`: physics profiles and visual/physics
  compatibility.
- `configs/multi_object_scene_experiments.json`: reproducible seed, model selection,
  curated per-scene spacing/contact fixes, and rigid collision policies.
- `configs/production_environment.json`: validated Isaac Sim, PhysX, Blender,
  FFmpeg, renderer, timing, and encoding versions/settings.

`generate_multi_object_config.py --seed 20260820` must reproduce the checked-in
multi-object JSON data exactly. Curated production overrides are part of the
generator; they must not survive only as hand-edited JSON.

## Candidate regression gates

1. Python compilation succeeds for the production entrypoints.
2. The complete repository test suite passes.
3. The checked-in multi-object configuration is exactly reproducible.
4. All 14 exact collision assets pass geometry, winding, support-height, and
   crop-coverage checks.
5. A forced 14-scene run produces, for every scene:
   - a readable `run_complete.json`;
   - three configured bodies with matching rigid/deformable collision policy;
   - a valid fixed-topology Blender USD cache;
   - detected inter-body contact when required;
   - valid ground-penetration checks;
   - a Blender render report and non-empty H.264 video.
   A readable animation cache never overrides an invalid PhysX report; this gate is
   strict about physics validation rather than treating cache recovery as pass.
6. Windows and Linux use the same Git commit and environment-variable based
   executable/scene paths.
7. Record the Git commit used by every accepted run. Passing this candidate
   does not freeze the worktree and does not authorize M1 to begin.

Run the forced production regression with:

```powershell
Y:\isaacsim\python.bat experiments\model_material\run_multi_object_videos.py `
  --output Y:\isaacsim_work\output\multi_object_mixed_material_14 `
  --frames 300 --samples 16 --resolution 640 --stage all --force
```

Run the final machine-readable gate with:

```powershell
Y:\isaacsim\python.bat tools\audit\verify_production_baseline_candidate.py
```

## Promotion to M0

Promote a future commit to M0 only after the user explicitly decides that the
production interfaces are stable enough to freeze. At minimum, the benchmark
schema, truth-export contract, camera contract, configuration compatibility,
and supported physics families must have explicit versioned boundaries. Until
then, passing this document's gates means only “current regression candidate
passes,” never “M0 complete.”

Native PhysX contact-manifold truth, multi-view benchmark cameras, metric
depth/normal/mask passes, parameter sweeps, paired counterfactuals, future
splits, and a unified fluid schema remain outside this candidate.
