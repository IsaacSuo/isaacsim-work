# PhysX Diffuse readback audit

Audit date: 2026-09-01

Target runtime: Isaac Sim 6.0, `omni.physx` 110.1.13, Python 3.12.13.

## Required cache contract

For every simulated frame and particle set, the exporter must provide:

- `position_lifetime`: float32 `[N, 4]`, with xyz position and remaining lifetime;
- `velocity`: float32 `[N, 4]` (xyz used; w preserved for auditability);
- `active_count`: uint32 scalar;
- source particle-set path and frame/time metadata.

A USD/Fabric `points` array is only a display projection. It cannot pass this
contract because it drops lifetime, velocity, and the native active count.

## Evidence

1. Native PhysX 5.9 exposes all required data on
   `PxParticleAndDiffuseBuffer`: `getDiffusePositionLifeTime()`,
   `getDiffuseVelocities()`, and `getNbActiveDiffuseParticles()`.
2. The PhysX 5.9 OmniPVD schema declares `diffusePositionLifeTime` and
   `diffuseVelocities`, but it does not declare `nbActiveDiffuseParticles`.
3. In the same 5.9 source, `streamParticleAndDiffuseBufferAttributes()` writes
   only `maxDiffuseParticles` and the Diffuse parameter object. It does not
   write either Diffuse data array. This matches the v7/v8 recordings exactly.
4. Explicitly disabling `/physics/suppressReadback` in v8 did not change the
   recorded attributes, so suppression is not the cause.
5. The installed public Python stub and native Python binding contain no
   `get_physx_ptr`, `get_object_id`, particle-buffer view, or Diffuse-buffer
   accessor. `omni.physics.tensors` also has no particle view in this runtime.
6. NVIDIA's open integration source confirms the intended native architecture:
   an `ePTParticleSet` database record points at a
   `PxParticleAndDiffuseBuffer`; its display path downloads only
   `getDiffusePositionLifeTime()` into a pinned host buffer, while no Diffuse
   velocity download is performed there.

Version-specific PhysX sources:

- `physx/include/PxParticleBuffer.h` at tag
  `110.1-omni-and-physx-5.9.0`
- `physx/source/physx/src/omnipvd/OmniPvdTypes.h` at the same tag
- `physx/source/physx/src/omnipvd/OmniPvdPxSampler.cpp` at the same tag

Local runtime evidence:

- `Y:\isaacsim\extscache\omni.physx-110.1.13+110.1.2.wx64.r.cp312.u7f4\omni\physx\bindings\_physx.pyi`
- `Y:\isaacsim_work\output\mountain_waterfall_preview\physx_diffuse_probe_v6_private_fabric\probe_report.json`
- `Y:\isaacsim_work\output\mountain_waterfall_preview\physx_diffuse_probe_v8_omnipvd_readback\probe_report.json`
- `Y:\isaacsim_work\output\mountain_waterfall_preview\physx_diffuse_probe_v9_native_bridge\probe_report.json`

## Native bridge result

The exact native route was implemented and passed its low-load gate on
2026-09-01.  The bridge is in:

- `Y:\isaacsim_work\experiments\mountain_waterfall\native_bridge`

Its pinned contract is:

- `omni::physx::IPhysx` 5.0;
- `omni::physx::IPhysxPrivate` 2.0;
- OpenUSD 25.11 (`pxrInternal_v0_25_11__pxrReserved__`);
- PhysX tag `110.1-omni-and-physx-5.9.0`, commit
  `517a0073715120e114ee055b63b26c95e00d9039`.

The v9 micro gate used 2,044 primary particles, 30 steps at 240 Hz, and did
not load the Mountain scene.  At steps 7, 15, 22, and 30 it copied exactly
4,088 active native Diffuse records from CUDA.  Every checkpoint passed:

- `position_lifetime`: float32 `[4088, 4]`, all finite;
- `velocity`: float32 `[4088, 4]`, all finite and non-zero in xyz;
- native `active_count == max_count == 4088`;
- remaining lifetime stayed within the authored 1.0 second range.

The four NPZ samples were reopened in a separate Python process and validated
independently.  The run report is `status=passed`, `valid=true`, and no
`OBSOLETE.json` marker was created.

## PhysX-compatible render-label recovery

PhysX 5.9 already performs native Diffuse emission and three-way dynamics in
`physx/source/gpusimulationcontroller/src/CUDA/diffuseParticles.cu`:

- fewer than 4 primary neighbors: spray with air drag;
- 4 through 7 primary neighbors: foam advected by local fluid velocity;
- 8 or more primary neighbors: bubble with buoyancy and bubble drag.

The neighbor count is stored only in a temporary kernel buffer.  The public
Diffuse velocity buffer explicitly writes zero to its fourth component, so the
native class is not available after `fetchResults()`.  The bridge therefore
also reads the public primary position/inverse-mass and velocity buffers, and a
render-label recovery layer reproduces the source thresholds without replacing
PhysX dynamics.

The kernel radius is **not** `particleContactOffset`.  In
`PxgParticleSystemCore::updateParticleSystemData()` it is assigned as:

```text
particleContactDistance = 2 * particleContactOffset
```

The first v10 recovery used the unscaled offset.  Its native readback remains
valid, but all v10 label arrays are invalid; that output is preserved with an
`OBSOLETE.json` marker.  The corrected v11 gate is:

- `Y:\isaacsim_work\output\mountain_waterfall_preview\physx_diffuse_probe_v11_label_radius_fix`

v11 explicitly enabled PhysX's public `eFULL_DIFFUSE_ADVECTION` flag so both
the native kernel and recovery use the 27-cell full neighborhood.  It passed
all four checkpoints with 2,044 primary and 4,088 Diffuse particles.  Each NPZ
contains native primary/Diffuse arrays, recovered uint8 labels, uint8 neighbor
counts, the exact 0.0505 m kernel radius, and source paths.  Independent NPZ
reopening proved the `<4 / <8 / >=8` mapping and count conservation.

This short submerged-impact gate produced foam and bubbles but no detached
spray.  Spray boundary behavior is covered by deterministic unit tests; a real
spray population must still be validated in a later low-load pouring/impact
gate rather than manufactured by weakening this gate.

## Conclusion

The native data exists and is not restricted to the renderer. However, Isaac
Sim 6.0's supported Python surface does not expose the complete buffer, and
OmniPVD 5.9 registers but does not serialize the GPU arrays. The implemented
smallest exact solution is a native bridge, not a replacement whitewater
simulation:

1. acquire the public C++ `IPhysx` interface;
2. resolve the particle-set USD path as `ePTParticleSet`;
3. cast the returned pointer to `PxParticleAndDiffuseBuffer`;
4. read the active count after `fetchResults()`;
5. copy exactly `N` position/lifetime and velocity elements from CUDA device
   memory into Python-owned host buffers;
6. return owned arrays to Python and write versioned NPZ frames.

The bridge is compiled with MSVC 19.44 and Windows SDK 10.0.26100.0 against the
exact headers above.  It dynamically verifies the expected Carbonite interface
versions and OpenUSD constructor symbol instead of guessing complete object
layouts or calling C++ virtual interfaces through `ctypes`.
