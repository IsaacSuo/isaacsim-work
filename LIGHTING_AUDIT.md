# Blender → Isaac Sim lighting audit

This audit separates three concepts that must not be conflated:

1. an emissive texture/color that only makes a surface look bright;
2. an emissive mesh that was intended to illuminate its surroundings;
3. an authored Blender `LIGHT` object.

The source-of-truth scan is generated directly from all 14 `.blend` files with
`tools/audit/audit_blend_lighting.py`. Its current output is
`output/material_audit/blend_lighting_audit.json`.

## Restore decisions

| Scene | Blender evidence | Isaac decision |
| --- | --- | --- |
| alley | No light objects or emissive materials; bright World background | Unified HDRI only |
| apartment | 32 materials use texture-driven Principled emission across most of the asset; the only Point light is far outside the experiment region | Treat emission as baked appearance; do not create hundreds of proxy lights |
| bedroom | Emission belongs to screens/appliances; the authored lamp is about 20 m from the experiment region | Do not invent a local lamp |
| city | Eleven strength-1 shop/sign materials in an outdoor daytime scene | Unified HDRI is the real illuminant; no sign proxies |
| classroom | One strength-2.6667 ceiling material split into 45 horizontal 2.4 m panels | Restore the four panels within 3 m of the experiment as downward RectLights at intensity 4200 each |
| elevator | Texture emission exists, but both authored 1000 W Area lights have `hide_render=true` in Blender | Restore emissive appearance only; do not enable lights Blender explicitly disabled |
| factory | No light objects or emissive materials; authored HDRI only | Unified HDRI only |
| garage | Eight strength-100 hanging lamps; two are within 8 m of the experiment | Restore two downward RectLights at intensity 40000 each |
| graffiti_warehouse | One strength-90 ceiling mesh split into many components | Restore the seven nearby ceiling components at intensity 8000 each |
| hospital | Fourteen render-enabled 1 m Area lights: thirteen at 300.5 W and one at 10 W; no emissive materials; all lights were omitted from exported USD | Restore all fourteen at their Blender-authored positions and directions. The calibrated Isaac values are 75000/2495.85 (10x over the original conservative conversion), with generic Key/Rim/Fill disabled |
| mountain | No light objects or emissive materials | Unified HDRI only |
| subway2 | One strength-15 train-light mesh split into 540 components | Restore the eight nearby lamp components at intensity 2500 each |
| swamp | No light objects or emissive materials | Unified HDRI only |
| warehouse | One strength-50 ceiling-light mesh with 40 valid horizontal components | Restore all 40 at intensity 16000 each |

## Factory material correction

The imported `factory.blend` assigns `Metallic=1, Roughness=1` to 43 of its
45 Principled materials, including physically dielectric surfaces. Isaac's
enclosed/backlit view therefore rendered brick, concrete, wood, ordinary
building panels, trash bags, and similar surfaces nearly black because they
had no diffuse response. The experiment config now forces only those known
dielectric materials to nonmetal. Water and genuinely metallic fixtures keep
their authored metallic values.

Blender Z-up coordinates for authored hospital lights are converted to the
Y-up Isaac stage with `(x, y, z) → (x, z, -y)`. The generated USD records each
restored source name under `authoredLight:sourceName`.

## Current visual validation

All 14 scenes and all 42 orbit images pass the structural check. The scenes
with restored local illumination currently have these three-view averages:

| Scene | Mean luma | Near-black pixels | White-clipped pixels |
| --- | ---: | ---: | ---: |
| classroom | 31.57 | 41.85% | 0.0000% |
| garage | 24.98 | 45.47% | 0.0000% |
| graffiti_warehouse | 63.94 | 25.00% | 0.0000% |
| hospital | 47.58 | 8.21% | 0.0000% |
| subway2 | 36.76 | 23.30% | 0.0000% |
| warehouse | 62.08 | 9.43% | 0.0308% |

These metrics are a guardrail, not an acceptance criterion by themselves.
The reviewed contact sheet is
`output/scene_previews/visual_audit/all_scenes_contact_sheet.jpg`.

Formal video rendering must remain disabled until the contact sheet is
accepted.
