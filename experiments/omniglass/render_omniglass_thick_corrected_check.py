"""Correct thick-volume OmniGlass test with low absorption and rebuilt normals."""

from pathlib import Path


wrapper_path = Path(__file__).resolve().with_name("render_omniglass_highkey_check.py")
wrapper = wrapper_path.read_text(encoding="utf-8")
wrapper = wrapper.replace(
    'shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(20.0)',
    'shader.CreateInput("depth", Sdf.ValueTypeNames.Float).Set(0.005)',
)
wrapper = wrapper.replace(
    'Sdf.ValueTypeNames.Float).Set(0.018)',
    'Sdf.ValueTypeNames.Float).Set(0.035)',
)
wrapper = wrapper.replace(
    '"omniglass_highkey_check_f000060.png"',
    '"omniglass_thick_corrected_f000060.png"',
)
wrapper = wrapper.replace(
    'or "omniglass_highkey_check" not in source',
    'or "omniglass_thick_corrected" not in source',
)
if "Set(0.005)" not in wrapper or "omniglass_thick_corrected" not in wrapper:
    raise RuntimeError("Corrected thick-volume wrapper injection failed")
exec(compile(wrapper, str(wrapper_path), "exec"), {"__file__": str(wrapper_path), "__name__": "__main__"})
