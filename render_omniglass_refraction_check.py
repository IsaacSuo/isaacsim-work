"""Diagnostic wrapper that explicitly enables the RTX material refraction setting."""

from pathlib import Path


source_path = Path(__file__).resolve().with_name("render_omniglass_normals_check.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    '    settings = carb.settings.get_settings()\n',
    '    settings = carb.settings.get_settings()\n'
    '    settings.set("/rtx/material/enableRefraction", True)\n',
)
source = source.replace(
    '"omniglass_normals_check_f000060.png"',
    '"omniglass_refraction_check_f000060.png"',
)
if source.count('/rtx/material/enableRefraction') != 1:
    raise RuntimeError("Diagnostic source injection failed")
exec(compile(source, str(source_path), "exec"), {"__file__": str(source_path), "__name__": "__main__"})
