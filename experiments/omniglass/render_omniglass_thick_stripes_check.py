"""Correct thick-volume OmniGlass with low absorption and a refraction target."""

from pathlib import Path


wrapper_path = Path(__file__).resolve().with_name("render_omniglass_silicone_stripes_check.py")
wrapper = wrapper_path.read_text(encoding="utf-8")
wrapper = wrapper.replace(
    '\'shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)\'',
    '\'shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)\'',
)
wrapper = wrapper.replace(
    'source = source_path.read_text(encoding="utf-8")\n',
    'source = source_path.read_text(encoding="utf-8")\n'
    'source = source.replace(\'shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(2.0)\', '
    '\'shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.005)\')\n',
)
wrapper = wrapper.replace("Gf.Vec3f(0.72, 0.90, 0.97)", "Gf.Vec3f(0.97, 0.99, 1.0)")
wrapper = wrapper.replace("Sdf.ValueTypeNames.Float).Set(0.07)", "Sdf.ValueTypeNames.Float).Set(0.04)")
wrapper = wrapper.replace(
    '"omniglass_silicone_stripes_f000060.png"',
    '"omniglass_thick_stripes_f000060.png"',
)
wrapper = wrapper.replace(
    'or "omniglass_silicone_stripes" not in source',
    'or "omniglass_thick_stripes" not in source',
)
if "Set(0.005)" not in wrapper or "omniglass_thick_stripes" not in wrapper:
    raise RuntimeError("Corrected thick-stripes wrapper injection failed")
exec(compile(wrapper, str(wrapper_path), "exec"), {"__file__": str(wrapper_path), "__name__": "__main__"})
