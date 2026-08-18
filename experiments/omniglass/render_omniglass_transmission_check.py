"""Diagnostic wrapper that explicitly enables RTX transmission on ordinary USD geometry."""

from pathlib import Path


source_path = Path(__file__).resolve().with_name("render_omniglass_normals_check.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    '    visual.CreateDoubleSidedAttr().Set(False)\n',
    '    visual.CreateDoubleSidedAttr().Set(False)\n'
    '    visual.GetPrim().CreateAttribute("omni:rtx:enableTransmission", Sdf.ValueTypeNames.Bool).Set(True)\n',
)
source = source.replace(
    '    sphere.CreateRadiusAttr().Set(0.42)\n',
    '    sphere.CreateRadiusAttr().Set(0.42)\n'
    '    sphere.GetPrim().CreateAttribute("omni:rtx:enableTransmission", Sdf.ValueTypeNames.Bool).Set(True)\n',
)
source = source.replace(
    '"omniglass_normals_check_f000060.png"',
    '"omniglass_transmission_check_f000060.png"',
)
if source.count("omni:rtx:enableTransmission") != 2:
    raise RuntimeError("Diagnostic source injection failed")
exec(compile(source, str(source_path), "exec"), {"__file__": str(source_path), "__name__": "__main__"})
