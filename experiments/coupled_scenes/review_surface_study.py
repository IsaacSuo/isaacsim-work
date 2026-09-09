"""Assemble existing layout renders into a desktop review; no simulation."""
import argparse
import html
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    title_font = ImageFont.truetype(str(args.font), 30)
    body_font = ImageFont.truetype(str(args.font), 23)
    canvas = Image.new("RGB", (1600, 2100), "#172129")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 22), "液面场景 · 第一组", font=title_font, fill="white")
    draw.text((30, 70), "场景搭建预览：水体仅标示水位，尚未进行流体仿真", font=body_font, fill="#a8d4dc")
    cards = []
    notes = ["静水基准：观察水位与边缘", "水槽左右移动，随后停止", "中央压头下压、退出，观察向外传播", "端部推板前推、复位，观察壁面反弹", "同一次中央扰动，保留更长恢复时间"]
    cases = sorted(args.input.glob("*/scene_spec.json"))
    if len(cases) != 5:
        raise ValueError(f"Expected five cases, got {len(cases)}")
    for index, spec_path in enumerate(cases):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        case = spec["case"]
        destination = args.output / case["id"]
        destination.mkdir(exist_ok=True)
        for source in spec_path.parent.iterdir():
            if source.suffix in {".png", ".json", ".usda", ".blend"}:
                shutil.copy2(source, destination / source.name)
        selected = "action_peak.png" if case["motion"] else "view_00.png"
        with Image.open(destination / selected) as render:
            preview = render.convert("RGB").resize((760, 570), Image.Resampling.LANCZOS)
        x, y = 20 + (index % 2) * 800, 135 + (index // 2) * 650
        canvas.paste(preview, (x, y))
        draw.text((x, y + 578), f"{index+1:02d}  {case['title']}", font=title_font, fill="white")
        draw.text((x, y + 619), notes[index], font=body_font, fill="#bed0d7")
        folder = case["id"]
        links = ' '.join(f'<a href="{folder}/{p.name}">{p.stem}</a>' for p in sorted(destination.glob("*.png")))
        cards.append(f'<section><h2>{html.escape(case["title"])}</h2><a href="{folder}/{selected}"><img src="{folder}/{selected}"></a><p>{html.escape(case["observation"])}</p><p>{links}</p><p><a href="{folder}/{folder}.blend">打开 / 保存 Blender 场景</a> · <a href="{folder}/scene_spec.json">场景说明</a></p></section>')
    draw.text((830, 1490), "共同布置", font=title_font, fill="white")
    for index, line in enumerate(["置于现有 warehouse 环境内", "槽内：160 × 90 cm，槽高 28 cm", "参考水深：12 cm", "每个场景：8 个固定相机", "动作已设置，水体尚未仿真", "预览中的平水面不代表运动结果"]):
        draw.text((830, 1550 + index*45), line, font=body_font, fill="#bed0d7")
    canvas.save(args.output / "overview.png")
    document = '<!doctype html><meta charset="utf-8"><title>液面场景第一组</title><style>body{background:#172129;color:#e3edf0;font:18px sans-serif;max-width:1400px;margin:30px auto;padding:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:28px}img{width:100%}a{color:#85d9ed;margin-right:10px}section{background:#23313b;padding:18px}p{line-height:1.6}</style><h1>液面场景 · 第一组</h1><p>场景搭建预览。水体仅标示水位，尚未进行流体仿真；动作峰值图只显示装置的位置。</p><p>槽内 160 × 90 cm，槽高 28 cm，参考水深 12 cm。每个场景含 8 个固定相机；静水槽另提供四向预览。</p><main>' + ''.join(cards) + '</main>'
    (args.output / "review.html").write_text(document, encoding="utf-8")
    print(args.output / "overview.png")


if __name__ == "__main__":
    main()
