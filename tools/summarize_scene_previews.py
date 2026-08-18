"""Build a contact sheet and luminance statistics for scene orbit previews."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"Y:\isaacsim_work\output\scene_previews")
    parser.add_argument("--output", default=r"Y:\isaacsim_work\output\scene_previews\visual_audit")
    return parser.parse_args()


ARGS = parse_args()
ROOT = Path(ARGS.root)
OUTPUT = Path(ARGS.output)
NAMES = ("orbit_left.png", "orbit_center.png", "orbit_right.png")
LABELS = ("left", "center", "right")
THUMB = 300
LABEL_HEIGHT = 28


def load_font(size):
    for path in (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\segoeui.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def image_metrics(path):
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        luminance = rgb.convert("L")
        histogram = luminance.histogram()
        pixels = rgb.width * rgb.height
        mean = ImageStat.Stat(luminance).mean[0]
        return {
            "width": rgb.width,
            "height": rgb.height,
            "mean_luma": mean,
            "pure_black_pct": 100.0 * histogram[0] / pixels,
            "dark_pct": 100.0 * sum(histogram[:9]) / pixels,
            "near_black_pct": 100.0 * sum(histogram[:21]) / pixels,
            "white_clip_pct": 100.0 * sum(histogram[250:]) / pixels,
        }


def main():
    scenes = sorted(
        directory.name
        for directory in ROOT.iterdir()
        if directory.is_dir() and all((directory / name).is_file() for name in NAMES)
    )
    if not scenes:
        raise RuntimeError(f"No complete orbit preview directories found in {ROOT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    title_font = load_font(18)
    label_font = load_font(15)
    sheet = Image.new("RGB", (THUMB * 3, (THUMB + LABEL_HEIGHT) * len(scenes)), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    rows = []
    scene_results = []
    for row_index, scene in enumerate(scenes):
        report_path = ROOT / scene / "run_complete.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        scene_results.append(
            {
                "scene": scene,
                "valid": bool(report.get("valid")),
                "report": str(report_path),
                "error": report.get("error"),
            }
        )
        for column_index, (filename, view) in enumerate(zip(NAMES, LABELS)):
            path = ROOT / scene / filename
            metrics = image_metrics(path)
            rows.append({"scene": scene, "view": view, "path": str(path), **metrics})
            with Image.open(path) as source:
                thumbnail = ImageOps.fit(source.convert("RGB"), (THUMB, THUMB), method=Image.Resampling.LANCZOS)
            x = column_index * THUMB
            y = row_index * (THUMB + LABEL_HEIGHT)
            sheet.paste(thumbnail, (x, y + LABEL_HEIGHT))
            label = f"{scene} | {view} | L={metrics['mean_luma']:.1f} D={metrics['near_black_pct']:.1f}%"
            draw.text((x + 5, y + 4), label, fill=(240, 240, 240), font=label_font)

    sheet_path = OUTPUT / "all_scenes_contact_sheet.jpg"
    sheet.save(sheet_path, quality=92, subsampling=0)
    json_path = OUTPUT / "image_metrics.json"
    payload = {
        "valid": all(result["valid"] for result in scene_results),
        "scene_count": len(scenes),
        "image_count": len(rows),
        "scene_results": scene_results,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = OUTPUT / "image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"scenes={len(scenes)} images={len(rows)} sheet={sheet_path} metrics={json_path}")


if __name__ == "__main__":
    main()
