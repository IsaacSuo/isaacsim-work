"""Compare an image sequence against a static empty reference frame."""

import argparse
from pathlib import Path

from PIL import Image, ImageChops


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
parser.add_argument("--reference", type=int, default=1)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
parser.add_argument("--threshold", type=int, default=5)
args = parser.parse_args()

directory = Path(args.directory)
reference = Image.open(directory / f"frame_{args.reference:04d}.png").convert("RGB")
for frame in range(args.start, args.end + 1):
    image = Image.open(directory / f"frame_{frame:04d}.png").convert("RGB")
    histogram = ImageChops.difference(reference, image).histogram()
    changed = sum(count for value, count in enumerate(histogram) if value % 256 > args.threshold)
    energy = sum((value % 256) * count for value, count in enumerate(histogram))
    print(f"{frame:04d} changed_channels={changed} difference_energy={energy}")
