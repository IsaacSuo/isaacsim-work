# Whitewater scene contract — phase 3 open-terrain adapter

The whitewater core consumes metre-normalized particle and field data. A scene
contract describes analytic colliders and their roles without embedding scene
names, Blender objects, water levels or impact coordinates in solver code.

## Invocation

`build_whitewater_v6_liquid_fields.py` accepts an optional
`--scene-contract PATH`. Omitting it constructs the historical single-sphere
schema-1 contract from the PhysX run report and emits `sphere_collision_sdf`
only as a read-only compatibility alias. Schema-1 contracts continue to use
the explicit `--terrain-heightfield` adapter.

Schema-2 contracts must own an audited open terrain asset and its selection
record. They omit `--terrain-heightfield`; supplying both inputs is rejected.

New contracts emit these canonical fields:

- `collision_sdf`: terrain plus every collider carrying the `solid` role;
- `collider_collision_sdf`: all analytic colliders carrying `solid`;
- `dynamic_collision_sdf`: source-driven colliders carrying `dynamic`;
- `churn_source_sdf`: only colliders explicitly carrying `churn_source`.
- `dynamic_churn_source_sdf`: the dynamic-only churn subset retained so the
  static/dynamic composition can be independently audited.

Keeping `churn_source` separate is required: a terrain wall is a solid but is
not automatically an impactor that should generate subsurface churn.

The checked schema-1 swamp migration contract is
`configs/swamp_sphere_impact.scene.json`. The schema-2 open-terrain contract is
`configs/swamp_sphere_impact_open_terrain.scene.json`.

## Coordinate and unit contract

- Axes are right-handed `xyz`.
- Grid fields used by the solver are in metres.
- `metres_per_unit` converts source particle positions, velocities, collider
  translations and collider dimensions into metres.
- Collider transforms must be rigid and right-handed. Scale and shear are
  rejected because applying a rigid analytic SDF through a non-rigid matrix
  would no longer produce metric distance.
- Both USD-style row-translation matrices and conventional column-translation
  matrices are supported when declared explicitly.

## Collider shapes

Phase 1 supports exact analytic SDFs for:

- sphere;
- oriented box;
- oriented capsule with local x/y/z axis;
- oriented infinite plane/solid half-space;
- closed, watertight triangle mesh with pinned SHA-256 provenance.

Triangle meshes are queried in bounded chunks through `trimesh`; the core sign
is converted to the project-wide negative-inside convention. The authored mesh
must already be watertight, consistently outward-wound and positive-volume.
Automatic topology repair and winding reversal are deliberately disabled.

Static analytic and mesh colliders are evaluated once and stored in
`grid_and_static_fields.npz`. Only source-driven colliders are rebuilt per
frame. Per-frame canonical unions still include both static and dynamic solids.

Collider IDs must be unique. Source-driven transforms declare the exact NPZ
field that supplies their 4×4 matrix. Missing fields, unknown schema keys,
invalid dimensions, duplicate roles and non-rigid transforms fail before an
output cache is created.

## Open terrain selection and distance

Open terrain is deliberately separate from a closed mesh collider. It has no
well-defined enclosed volume, so applying a watertight inside/outside test to
it would be incorrect.

The schema-2 terrain entry pins both:

- a selected, consistently wound open triangle mesh;
- a JSON selection record with independent SHA-256 provenance.

The selection record stores the source USD hash and prim path, coordinate
conversion, explicit source face indices, ROI, maximum liquid-support level,
connected-component policy, orientation convention and selected mesh hash.
Unknown or mismatched assets are rejected before field construction.

Selection never casts a first-hit ray over the full authored scene. The swamp
migration first identifies the historical 62 below-water support faces, then
expands only their connected component inside the local ROI. This produces a
112-face sheet containing both the basin floor and dry banks. The same sheet
therefore has enough coverage for collision and later shoreline consumers
without treating unrelated stacked terrain as the water floor.

The field uses oriented Euclidean closest-surface distance. Positive values
are on the explicitly authored fluid-facing side and negative values are on
the solid side. A query whose closest point lies on an authored open boundary
is marked invalid rather than silently extruding that edge into a false wall.
The liquid-field builder requires its entire solver grid to have valid terrain
coverage.

## Downstream compatibility

Emission, marker collision projection, surface foam and surface raft prefer
canonical role fields. They still recognize `sphere_collision_sdf` when
reading historical caches. A newly configured scene never emits that alias,
preventing it from becoming a hidden dependency again.

New-scene emission contact begins when `churn_source_sdf` intersects the
reconstructed liquid surface band. Historical caches retain their exact
sphere-centre/water-level contact gate.

## Audited regressions

The phase-1 regression uses audited PhysX source sample 120 and the exact
production grid and terrain provenance. It rebuilds the sample through:

1. the implicit legacy sphere adapter;
2. the explicit scene JSON;
3. the locked pre-migration production baseline.

Every physical array is compared bit-for-bit. The role-separated collider
fields must also match the old sphere SDF exactly. The report is written to:

`output/swamp_fluid_preview/whitewater_v6_scene_contract_phase1_regression_audit.json`

The phase-3 terrain regression compares the selected open sheet to the locked
production heightfield on all 1,669,995 grid nodes. The 62 support source-face
indices must match exactly; the expanded sheet must be one connected component
and cover the complete grid. Solid-side classification must agree everywhere.
Because the new field is Euclidean distance while the old field measured
vertical height difference, non-zero magnitudes on slopes are expected but
must remain sub-voxel. The report is:

`output/swamp_fluid_preview/whitewater_v6_open_terrain_swamp_supported_component_phase3/audit_report.json`

A real source sample is also rebuilt without `--terrain-heightfield` and
audited by the unchanged liquid-field audit:

`output/swamp_fluid_preview/whitewater_v6_open_terrain_phase3_integration_s32/audit_report.json`

## Deliberate current boundary

Open terrain is now a first-class scene-contract input for liquid collision,
but ROI and liquid-support seeds are still authored during adapter export; the
pipeline does not yet discover them automatically from arbitrary caches.
Splashsurf clipping and render-surface shoreline support still consume their
historical heightfield path and must next be migrated to the same terrain mesh
and selection hash. Animated/deforming terrain, moving domains and multiple
independent water bodies also remain subsequent phases.
