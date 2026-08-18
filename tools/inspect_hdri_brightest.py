"""Locate the brightest region of the workspace HDRI at a reduced resolution."""

from array import array
import json
from pathlib import Path
import sys

import bpy


path = Path(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else Path(
    r"Y:\scenes\HDRI\bryanston_park_sunrise_8k.exr"
)
image = bpy.data.images.load(str(path), check_existing=False)
image.scale(1024, 512)
pixels = array("f", [0.0]) * (1024 * 512 * 4)
image.pixels.foreach_get(pixels)

best = []
for index in range(1024 * 512):
    offset = index * 4
    r, g, b = pixels[offset], pixels[offset + 1], pixels[offset + 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if len(best) < 20:
        best.append((luminance, index))
        best.sort()
    elif luminance > best[0][0]:
        best[0] = (luminance, index)
        best.sort()

payload = [
    {
        "luminance": value,
        "x": index % 1024,
        "y": index // 1024,
        "u": (index % 1024 + 0.5) / 1024,
        "v": (index // 1024 + 0.5) / 512,
    }
    for value, index in reversed(best)
]
print("HDRI_BRIGHTEST=" + json.dumps({"path": str(path), "pixels": payload}))
