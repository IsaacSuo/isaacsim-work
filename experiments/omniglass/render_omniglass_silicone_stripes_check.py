"""Readable OmniGlass silicone preview with a refraction stripe target."""

from pathlib import Path


source_path = Path(__file__).resolve().with_name("render_omniglass_normals_check.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    'shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.97, 0.995, 1.0))',
    'shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.90, 0.97))',
)
source = source.replace(
    'shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.035)',
    'shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.07)',
)
source = source.replace(
    'shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)',
    'shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)',
)
stripe_code = '''
    stripe_materials = []
    for stripe_name, stripe_color in (("StripeWhite", (0.92, 0.94, 0.96)), ("StripeDark", (0.025, 0.035, 0.05))):
        stripe_material = UsdShade.Material.Define(stage, f"/World/Looks/{stripe_name}")
        stripe_shader = UsdShade.Shader.Define(stage, f"/World/Looks/{stripe_name}/PreviewSurface")
        stripe_shader.CreateIdAttr("UsdPreviewSurface")
        stripe_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*stripe_color))
        stripe_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        stripe_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        stripe_material.CreateSurfaceOutput().ConnectToSource(stripe_shader.ConnectableAPI(), "surface")
        stripe_materials.append(stripe_material)
    for stripe_index in range(13):
        stripe = UsdGeom.Cube.Define(stage, f"/World/OmniGlassStripes/S{stripe_index:02d}")
        stripe.CreateSizeAttr(1.0)
        stripe_xform = UsdGeom.Xformable(stripe)
        stripe_xform.AddTranslateOp().Set(Gf.Vec3d((stripe_index - 6) * 0.62, 2.7, -3.02))
        stripe_xform.AddScaleOp().Set(Gf.Vec3f(0.31, 3.8, 0.025))
        UsdShade.MaterialBindingAPI.Apply(stripe.GetPrim()).Bind(stripe_materials[stripe_index % 2])
'''
source = source.replace(
    '    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)\n',
    '    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)\n' + stripe_code,
)
source = source.replace(
    '    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)\n',
    '    backdrop_shader = UsdShade.Shader.Get(stage, Sdf.Path("/World/Looks/WarmBackdrop/PreviewSurface"))\n'
    '    floor_shader = UsdShade.Shader.Get(stage, Sdf.Path("/World/Looks/WarmStudio/PreviewSurface"))\n'
    '    backdrop_shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.78, 0.81, 0.84))\n'
    '    floor_shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.56, 0.59, 0.62))\n'
    '    ambient = stage.GetPrimAtPath(Sdf.Path("/World/Lights/Ambient"))\n'
    '    ambient.GetAttribute("inputs:intensity").Set(1700.0)\n'
    '    ambient.GetAttribute("inputs:color").Set(Gf.Vec3f(0.9, 0.95, 1.0))\n'
    '    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)\n',
)
source = source.replace('settings.set("/rtx/post/tonemap/exposure", 0.7)', 'settings.set("/rtx/post/tonemap/exposure", 0.0)')
source = source.replace('"omniglass_normals_check_f000060.png"', '"omniglass_silicone_stripes_f000060.png"')
if "OmniGlassStripes" not in source or "omniglass_silicone_stripes" not in source:
    raise RuntimeError("Diagnostic source injection failed")
exec(compile(source, str(source_path), "exec"), {"__file__": str(source_path), "__name__": "__main__"})
