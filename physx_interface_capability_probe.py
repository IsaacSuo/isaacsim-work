"""List particle-related methods exposed by Isaac Sim's Python bindings."""

import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.physx
import omni.physx.bindings._physx as bindings

for label, obj in (
    ("physx_interface", omni.physx.get_physx_interface()),
    ("simulation_interface", omni.physx.get_physx_simulation_interface()),
    ("bindings", bindings),
):
    names = [
        name
        for name in dir(obj)
        if any(token in name.lower() for token in ("particle", "buffer", "active"))
    ]
    print(f"[capability] {label}: {names}")

simulation_app.close()
