"""High-key OmniGlass preview using visually robust thin-wall transmission."""

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
    '"omniglass_highkey_check_f000060.png"',
    '"omniglass_thinwall_check_f000060.png"',
)
if "thin_walled" not in wrapper or "omniglass_thinwall_check" not in wrapper:
    raise RuntimeError("Diagnostic source injection failed")
exec(compile(wrapper, str(wrapper_path), "exec"), {"__file__": str(wrapper_path), "__name__": "__main__"})
