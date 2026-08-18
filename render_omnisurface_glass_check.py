"""Test Isaac 6 OmniSurface_Glass on the cached soft body in a high-key studio."""

from pathlib import Path


source_path = Path(__file__).resolve().with_name("render_omniglass_normals_check.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace('mdl_name="OmniGlass.mdl", mtl_name="OmniGlass",', 'mdl_name="OmniSurfacePresets.mdl", mtl_name="OmniSurface_Glass",')
source = source.replace('prim_name="OmniGlassSiliconeCheck",', 'prim_name="OmniSurfaceGlassCheck",')
source = source.replace('material_path = Sdf.Path("/World/Looks/OmniGlassSiliconeCheck")', 'material_path = Sdf.Path("/World/Looks/OmniSurfaceGlassCheck")')
source = source.replace(
    'shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.97, 0.995, 1.0))',
    'shader.CreateInput("specular_transmission_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))',
)
source = source.replace(
    'shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.41)',
    'shader.CreateInput("specular_reflection_ior", Sdf.ValueTypeNames.Float).Set(1.41)',
)
source = source.replace(
    'shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.035)',
    'shader.CreateInput("specular_reflection_roughness", Sdf.ValueTypeNames.Float).Set(0.018)',
)
source = source.replace(
    'shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(2.0)',
    'shader.CreateInput("specular_transmission_weight", Sdf.ValueTypeNames.Float).Set(1.0)',
)
source = source.replace(
    '    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)\n',
    '    backdrop_shader = UsdShade.Shader.Get(stage, Sdf.Path("/World/Looks/WarmBackdrop/PreviewSurface"))\n'
    '    floor_shader = UsdShade.Shader.Get(stage, Sdf.Path("/World/Looks/WarmStudio/PreviewSurface"))\n'
    '    backdrop_shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.92, 0.94, 0.96))\n'
    '    floor_shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.66, 0.69, 0.72))\n'
    '    ambient = stage.GetPrimAtPath(Sdf.Path("/World/Lights/Ambient"))\n'
    '    ambient.GetAttribute("inputs:intensity").Set(2200.0)\n'
    '    ambient.GetAttribute("inputs:color").Set(Gf.Vec3f(0.9, 0.95, 1.0))\n'
    '    stage.GetPrimAtPath(Sdf.Path("/World/Lights/Key")).GetAttribute("inputs:intensity").Set(14000.0)\n'
    '    stage.GetPrimAtPath(Sdf.Path("/World/Lights/Rim")).GetAttribute("inputs:intensity").Set(12000.0)\n'
    '    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)\n',
)
source = source.replace('settings.set("/rtx/post/tonemap/exposure", 0.7)', 'settings.set("/rtx/post/tonemap/exposure", -0.15)')
source = source.replace('"omniglass_normals_check_f000060.png"', '"omnisurface_glass_check_f000060.png"')
if "OmniSurface_Glass" not in source or "omnisurface_glass_check" not in source:
    raise RuntimeError("Diagnostic source injection failed")
exec(compile(source, str(source_path), "exec"), {"__file__": str(source_path), "__name__": "__main__"})
