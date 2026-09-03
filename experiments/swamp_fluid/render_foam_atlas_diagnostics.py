"""Render a compact visual QA sheet for audited foam atlas fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas_directory", type=Path)
    parser.add_argument(
        "--source-samples", type=int, nargs="+", default=[56, 100, 140, 180]
    )
    parser.add_argument("--panel-width", type=int, default=376)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path(r"C:\Windows\Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def display_order(array):
    return np.flip(np.asarray(array).T, axis=0)


def blend(background, foreground, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    return background * (1.0 - alpha) + foreground * alpha


def foam_image(coverage):
    coverage = display_order(coverage)
    base = np.empty(coverage.shape + (3,), dtype=np.float32)
    base[:] = (15, 27, 30)
    foam = np.empty_like(base)
    foam[:] = (238, 245, 226)
    return blend(base, foam, np.sqrt(coverage))


def bubble_image(coverage):
    coverage = display_order(coverage)
    base = np.empty(coverage.shape + (3,), dtype=np.float32)
    base[:] = (10, 20, 31)
    bubble = np.empty_like(base)
    bubble[:] = (72, 205, 228)
    return blend(base, bubble, np.sqrt(coverage))


def age_image(age, coverage):
    normalized = np.clip(display_order(age) / 1.2, 0.0, 1.0)
    coverage = display_order(coverage)
    stops = np.asarray(
        [
            (23, 20, 48),
            (78, 43, 117),
            (191, 69, 78),
            (244, 166, 57),
            (245, 238, 180),
        ],
        dtype=np.float32,
    )
    position = normalized * (len(stops) - 1)
    low = np.floor(position).astype(np.int32)
    high = np.minimum(low + 1, len(stops) - 1)
    fraction = (position - low)[..., None]
    color = stops[low] * (1.0 - fraction) + stops[high] * fraction
    background = np.empty_like(color)
    background[:] = (14, 23, 28)
    return blend(background, color, np.sqrt(coverage))


def composite_image(frame):
    wet = display_order(frame["wet_mask"].astype(np.float32))
    foam = display_order(frame["foam_coverage"])
    bubbles = display_order(frame["bubble_coverage"])
    base = np.empty(wet.shape + (3,), dtype=np.float32)
    base[:] = (21, 25, 22)
    water = np.empty_like(base)
    water[:] = (24, 64, 70)
    result = blend(base, water, 0.58 * wet)
    foam_color = np.empty_like(base)
    foam_color[:] = (241, 246, 225)
    result = blend(result, foam_color, np.sqrt(foam))
    bubble_color = np.empty_like(base)
    bubble_color[:] = (64, 193, 225)
    return blend(result, bubble_color, 0.75 * np.sqrt(bubbles))


def to_image(rgb, width):
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def add_orientation_arrows(image, frame, original_shape):
    draw = ImageDraw.Draw(image)
    nx, nz = original_shape
    sx = image.width / nx
    sy = image.height / nz
    spacing = 40
    half_length = 7
    coverage = frame["foam_coverage"]
    xx = frame["foam_orientation_xx"]
    xz = frame["foam_orientation_xz"]
    zz = frame["foam_orientation_zz"]
    for ix in range(spacing // 2, nx, spacing):
        for iz in range(spacing // 2, nz, spacing):
            if coverage[ix, iz] < 0.08:
                continue
            anisotropy_strength = math.sqrt(
                max((float(xx[ix, iz]) - float(zz[ix, iz])) ** 2 + 4.0 * float(xz[ix, iz]) ** 2, 0.0)
            )
            if anisotropy_strength < 0.08:
                continue
            angle = 0.5 * math.atan2(
                2.0 * float(xz[ix, iz]),
                float(xx[ix, iz]) - float(zz[ix, iz]),
            )
            dx = math.cos(angle) * half_length
            dy = -math.sin(angle) * half_length
            px = (ix + 0.5) * sx
            py = (nz - iz - 0.5) * sy
            draw.line(
                (px - dx, py - dy, px + dx, py + dy),
                fill=(255, 113, 46),
                width=2,
            )
    return image


args = parse_args()
atlas_directory = args.atlas_directory.resolve()
manifest = json.loads((atlas_directory / "manifest.json").read_text(encoding="utf-8"))
audit = json.loads((atlas_directory / "audit_report.json").read_text(encoding="utf-8"))
if audit.get("valid") is not True:
    raise ValueError("Refusing to visualize an atlas without a valid audit")
if audit.get("manifest_sha256") != __import__("hashlib").sha256(
    (atlas_directory / "manifest.json").read_bytes()
).hexdigest():
    raise ValueError("Audit does not reference the current atlas manifest")

sample_items = {int(item["source_sample_index"]): item for item in manifest["samples"]}
missing = [sample for sample in args.source_samples if sample not in sample_items]
if missing:
    raise ValueError(f"Requested source samples are unavailable: {missing}")
output_path = (
    args.output.resolve()
    if args.output
    else atlas_directory / "diagnostics" / "foam_atlas_contact_sheet.png"
)
if output_path.exists():
    raise FileExistsError(f"Refusing to overwrite {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)

loaded = []
for sample in args.source_samples:
    item = sample_items[sample]
    with np.load(atlas_directory / item["file"], allow_pickle=False) as cache:
        frame = {name: np.asarray(cache[name]) for name in cache.files}
    loaded.append((sample, item, frame))

first_shape = loaded[0][2]["foam_tau"].shape
panel_height = int(round(first_shape[1] * args.panel_width / first_shape[0]))
label_height = 30
header_height = 58
column_gap = 6
row_gap = 8
columns = 4
rows = len(loaded)
sheet_width = columns * args.panel_width + (columns - 1) * column_gap
sheet_height = header_height + rows * (panel_height + label_height) + (rows - 1) * row_gap
sheet = Image.new("RGB", (sheet_width, sheet_height), (10, 13, 15))
draw = ImageDraw.Draw(sheet)
draw.text((12, 7), "Whitewater surface atlas — audited 4 mm fields", font=font(22, True), fill=(235, 240, 235))
draw.text(
    (12, 34),
    "Foam coverage | Surface-bubble coverage | Foam age | Combined + principal flow direction",
    font=font(13),
    fill=(163, 178, 175),
)

for row, (sample, item, frame) in enumerate(loaded):
    images = [
        to_image(foam_image(frame["foam_coverage"]), args.panel_width),
        to_image(bubble_image(frame["bubble_coverage"]), args.panel_width),
        to_image(age_image(frame["foam_mean_age"], frame["foam_coverage"]), args.panel_width),
        to_image(composite_image(frame), args.panel_width),
    ]
    images[3] = add_orientation_arrows(images[3], frame, first_shape)
    y = header_height + row * (panel_height + label_height + row_gap)
    for column, image in enumerate(images):
        x = column * (args.panel_width + column_gap)
        sheet.paste(image, (x, y))
    label = (
        f"sample {sample:03d}  t={item['simulation_time']:.3f}s  "
        f"foam={item['foam_markers']:,}  surface bubbles={item['bubble_markers']:,}  "
        f"max tau=({item['maximum_foam_tau']:.2f}, {item['maximum_bubble_tau']:.2f})"
    )
    draw.text((8, y + panel_height + 5), label, font=font(14), fill=(220, 226, 219))

sheet.save(output_path, format="PNG", optimize=True)
print(output_path)
