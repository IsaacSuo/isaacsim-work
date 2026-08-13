# Realistic Liquid Long Video Pipeline

## Locked first production profile

- 10 seconds
- 60 Hz physics
- 30 fps output (`capture_stride = 2`)
- 300 output frames, numbered `000000..000299`
- 1280×720 PathTracing
- 32 spp
- fixed `stream-front` camera
- six render segments of 50 frames

The first cached frame is the post-preroll state at `t=0`. Frame `n` is sampled at `n / 30` seconds. The final sample is at `9.9667 s`; 300 frames played at 30 fps produce a 10.000-second file. Do not append an off-cadence “last simulation frame.”

## Simulator integration

`physx_realistic_liquid.py` should import:

```python
from liquid_video_cache import SurfaceCacheTakeWriter
```

After loading `job.json`, create exactly one writer for a fresh take:

```python
writer = SurfaceCacheTakeWriter(
    take_dir,
    take_id=job["take_id"],
    config_hash_value=job["config_hash"],
    simulation_provenance_hash_value=job["simulation_provenance_hash"],
    expected_frames=job["video"]["output_frames"],
    physics_fps=job["physics"]["physics_fps"],
    output_fps=job["video"]["output_fps"],
    compressed=False,
)
```

At post-preroll `t=0`, and then after every two 60 Hz physics steps, capture exactly one output frame. The caller must supply the actual post-preroll PhysX step and simulation time; the writer rejects synthetic or off-cadence timing.

1. Wait for the native PhysX isosurface update to settle.
2. Read points, face counts, face indices, and normals if authored.
3. Call `writer.write_frame(...)` once.
4. Preserve the returned row only for logging; the writer atomically writes both cache and authoritative manifest.

Example:

```python
writer.write_frame(
    points=mesh_points,
    face_vertex_counts=face_counts,
    face_vertex_indices=face_indices,
    normals=mesh_normals,
    sim_step=actual_post_preroll_step,
    sim_time_seconds=actual_sim_time,
    frame_metadata={
        "active_particles": active_particle_count,
        "recycled_particles": recycled_count,
        "measured_particle_flux": measured_flux,
        "isosurface_parameters": isosurface_parameters,
    },
)
```

Export a render-only template to:

```text
takes/<take_id>/render_template.usda
```

It must preserve the basin, faucet, obstacle, lights, water material, and camera. The cache renderer hides `/World/WaterParticles` and `/World/ParticleSystem` and injects `/World/CachedLiquid`.

After exactly 300 frames and all physical/recycle/log acceptance checks pass:

```python
writer.finalize(completion_metadata=physical_acceptance_summary)
```

A physical take is deliberately non-resumable. If simulation fails before `simulation_complete.json`, create a new take and rerun it from `t=0`; never join two simulation takes as one continuous shot.

## WSL commands

Initialize a production job:

```bash
python3 /mnt/y/isaacsim_work/liquid_video_pipeline.py init \
  --job-id faucet_10s_v1
```

Run one continuous cache simulation take. The stage streams Isaac Sim output live and writes its accepted marker only after the simulation exit code, fatal-log scan, GPU peak, completion marker, and cache manifest all pass:

```bash
python3 /mnt/y/isaacsim_work/liquid_video_pipeline.py simulate \
  --job /mnt/y/isaacsim_work/output/long_video/faucet_10s_v1
```

Validate all 300 caches and array schemas:

```bash
python3 /mnt/y/isaacsim_work/liquid_video_pipeline.py validate-cache \
  --job /mnt/y/isaacsim_work/output/long_video/faucet_10s_v1 \
  --full
```

Render the first canonical 50-frame segment and encode frames `0..29` as the one-second pilot:

```bash
python3 /mnt/y/isaacsim_work/liquid_video_pipeline.py render-pilot \
  --job /mnt/y/isaacsim_work/output/long_video/faucet_10s_v1
```

The pilot deliberately renders `0..49` (the first canonical segment) but encodes only `0..29`; it does not create an overlapping `0..29` render manifest.

The full production render requires a literal confirmation flag:

```bash
python3 /mnt/y/isaacsim_work/liquid_video_pipeline.py render-all \
  --job /mnt/y/isaacsim_work/output/long_video/faucet_10s_v1 \
  --confirm-production 300-frames
```

Encode after a complete render:

```bash
python3 /mnt/y/isaacsim_work/liquid_video_pipeline.py encode \
  --job /mnt/y/isaacsim_work/output/long_video/faucet_10s_v1
```

Inspect progress at any time:

```bash
python3 /mnt/y/isaacsim_work/liquid_video_pipeline.py status \
  --job /mnt/y/isaacsim_work/output/long_video/faucet_10s_v1
```

## Recovery rules

- Simulation/cache generation: no state resume and no cross-take splice.
- Cache rendering: resume only when an accepted marker binds the current segment manifest and current render provenance; verified PNGs are hash-validated before they are skipped and are never overwritten.
- Render segments: six canonical, non-overlapping Isaac Sim processes of 50 frames. A stale or failed segment manifest is preserved with a unique `_rejected_<run-id>` suffix rather than silently reused.
- Segment acceptance: written only after renderer completion, fatal-log scan, GPU peak validation, completion-marker identity checks, and manifest hashing succeed.
- Encoding: repeatable after all PNGs pass strict signature/chunk CRC/zlib, dimensions, take ID, config hash, and provenance checks.
- Jobs must live under a Windows-mounted path such as `/mnt/y`; Kit/USD access through a WSL UNC path is rejected.

## Encoding acceptance

The encoder requires and verifies:

- H.264 / `libx264`
- CRF 18, preset `slow`
- `yuv420p`
- BT.709 color tags
- 1280×720
- 30 fps
- 300 decoded frames
- 10.000-second duration
- complete decode with zero ffmpeg errors
- representative frames and a 20-frame contact sheet

Consecutive identical PNG hashes are reported as suspicious duplicates but retained for review rather than silently modified.
