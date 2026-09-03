"""Validate artist setup and emit a hash-locked production plan."""

import argparse
import json
from pathlib import Path

from whitewater.production_config import ProductionConfig


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("configuration", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
configuration = ProductionConfig.load(args.configuration)
payload = configuration.resolved_metadata()
payload["commands"] = {
    "splashsurf": configuration.splashsurf_command(
        configuration.paths["output_root"] / "splashsurf"
    ),
    "render": configuration.render_command(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({"valid": True, "name": configuration.name, "output": str(args.output.resolve()), "stages": list(payload["commands"])}, indent=2))
