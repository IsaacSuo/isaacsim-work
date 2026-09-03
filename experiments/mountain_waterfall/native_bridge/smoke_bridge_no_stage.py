"""ABI smoke gate: resolve a missing particle set without creating a stage."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import omni.kit.app


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "bin"))

result = {"passed": False, "expected_error": None, "abi_info": None}
try:
    import _physx_diffuse_bridge

    result["abi_info"] = dict(_physx_diffuse_bridge.abi_info())
    _physx_diffuse_bridge.probe("/World/DoesNotExist")
except RuntimeError as error:
    result["expected_error"] = str(error)
    result["passed"] = (
        "No live PhysX particle-set buffer" in str(error)
        and result["abi_info"]["bridge_schema"] == "physx_diffuse_native_bridge/2"
        and result["abi_info"]["iphysx_major"] == 5
        and result["abi_info"]["iphysx_private_major"] == 2
    )

print("PHYSX_DIFFUSE_BRIDGE_SMOKE=" + json.dumps(result, sort_keys=True))
omni.kit.app.get_app().post_quit(0 if result["passed"] else 2)
