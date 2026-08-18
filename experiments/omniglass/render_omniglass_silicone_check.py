"""Visually readable transparent-silicone OmniGlass preview."""

from pathlib import Path


wrapper_path = Path(__file__).resolve().with_name("render_omniglass_highkey_check.py")
wrapper = wrapper_path.read_text(encoding="utf-8")
wrapper = wrapper.replace(
    'source = source_path.read_text(encoding="utf-8")\n',
    'source = source_path.read_text(encoding="utf-8")\n'
    'source = source.replace(\'shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)\', '
    '\'shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)\')\n',
)
wrapper = wrapper.replace(
    'Gf.Vec3f(1.0, 1.0, 1.0)',
    'Gf.Vec3f(0.82, 0.94, 0.985)',
)
wrapper = wrapper.replace(
    'Sdf.ValueTypeNames.Float).Set(0.018)',
    'Sdf.ValueTypeNames.Float).Set(0.08)',
)
wrapper = wrapper.replace(
    '"omniglass_highkey_check_f000060.png"',
    '"omniglass_silicone_check_f000060.png"',
)
wrapper = wrapper.replace(
    'or "omniglass_highkey_check" not in source',
    'or "omniglass_silicone_check" not in source',
)
if "thin_walled" not in wrapper or "omniglass_silicone_check" not in wrapper:
    raise RuntimeError("Diagnostic wrapper injection failed")
exec(compile(wrapper, str(wrapper_path), "exec"), {"__file__": str(wrapper_path), "__name__": "__main__"})
