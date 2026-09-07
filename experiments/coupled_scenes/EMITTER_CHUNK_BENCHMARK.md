# Bounded particle-set append experiment

`chunked_density_sets` keeps the existing inlet birth schedule, positions,
velocities and global particle IDs. Consecutive births append to one density
set until the next complete birth would exceed `emission_chunk_particles`.
Then a new empty set is enabled. Completed sets are never read or rewritten
by emission again; PhysX continues to simulate them and output capture still
reads all active sets. This is **bounded copying**, not GPU zero-copy append.

Enable explicitly in an existing event JSON:

```json
{
  "particle_set_strategy": "chunked_density_sets",
  "emission_chunk_particles": 8192
}
```

The default capacity, when omitted for this strategy, is 16384. A birth larger
than the capacity is rejected before simulation; increase the capacity in
that case. No particle is dropped or delayed to satisfy the limit. All sets
belong to the same particle system with self-collision enabled and group 0.
Total particle memory and contact-buffer requirements are not reduced.

The cabinet builder exposes `--emission-chunk-particles 8192` together with
`--continuous-inlet-probe`, `--compact-impact-probe`, or `--server-deep-pour`.
The option does not change defaults. For the server preset specifically, it
also selects the substep `continuous_inlet` instead of the older frame-layer
inlet, so it is NOT a storage-only A/B against that older server preset.
Compare against an existing continuous-inlet configuration for a clean test.

## Local completed comparison

Both tests used the v70 continuous inlet: 4 mm spacing, 37,800 particles,
360 births at 720 Hz, 180 output frames including 120 dry prewarm frames,
64/64 particle/deformable iterations. Configurations differed only in
`particle_set_strategy` (both contain capacity 8192, ignored by shared mode).

- Baseline: `output/coupled_scenes/warehouse_glass_cabinet_pour_v72_profile_shared_37800p_r2`
- Chunked: `output/coupled_scenes/warehouse_glass_cabinet_pour_v73_chunk8192_density_37800p`

| Measurement | Shared set | Bounded chunks |
| --- | ---: | ---: |
| Particle sets | 1 | 5 |
| Cumulative old particle rows read during emission | 6,785,340 | 1,379,940 |
| Cumulative particle rows written during emission | 6,823,140 | 1,417,740 |
| Maximum old rows handled by one birth | 37,696 | 8,086 |
| Emission authoring time, seconds | 0.4463 | 0.3375 |
| Explicit activation app-update time, seconds | 2.4369 | 3.0156 |
| Dry interval, frame 1 to 120, seconds | 153.21 | 96.05 |
| Inlet interval, frame 120 to 150, seconds | 60.29 | 42.89 |
| Post-inlet interval, frame 150 to 180, seconds | 53.46 | 55.68 |
| Cache interval, frame 1 to 180, seconds | 266.96 | 194.62 |

Phase durations come from consecutive cache-file modification timestamps;
they include simulation and capture but exclude application startup and final
export. In-code timings are in `primary_fluid/manifest.json:emission_statistics`.
They measure Python/USD calls, not isolated GPU transfer or solver time.

The old-row traffic reduction is 79.66%. The direct authoring saving was only
0.109 seconds, and explicit app updates took 0.579 seconds longer. Although
the chunked run completed the cache interval faster, its dry prewarm was also
much faster; a single sequential A/B does not establish that chunking caused
the whole-run improvement. Do not advertise a 27% whole-run speedup or replace
the default based on this comparison. The demonstrated benefit is bounded
emission array traffic; server-scale throughput still needs profiling.

Both completed caches and run audits are valid. Both retained all 37,800
particles; every wet frame has the expected count, unique contiguous IDs and
finite positions/velocities. No inverted softbody tetrahedra, lateral/bottom
escapes or contact-buffer overflows were reported. Peak speed-cap fractions
were 0.1032% and 0.0873%. No render/visual-equivalence comparison was performed.

## Related mass correction

The installed `particleUtils.add_physx_particleset_pointinstancer` accepts
PER-PARTICLE mass and multiplies by particle count itself. Passing total batch
mass, as the older explicit-mass path did, multiplied mass twice. That path
now passes density times spacing cubed and verifies the authored USD total.
The shared-density and chunked-density comparison above clears explicit mass
and uses the same density-derived mass policy in both runs. Older explicit-mass
multi-set crashes cannot establish a particle-set-count limit.
