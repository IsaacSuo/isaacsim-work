"""Print the loaded omni.physx Carbonite interfaces and terminate Kit."""

from __future__ import annotations

import json

import carb
import omni.kit.app


def main() -> None:
    rows = []
    for plugin in carb.get_framework().get_plugins():
        plugin_name = plugin.impl.name
        if "physx" not in plugin_name.lower():
            continue
        rows.append(
            {
                "plugin": plugin_name,
                "library": plugin.libPath,
                "interfaces": [
                    {
                        "name": interface.name,
                        "major": interface.version.major,
                        "minor": interface.version.minor,
                    }
                    for interface in plugin.interfaces
                ],
            }
        )
    print("PHYSX_PLUGIN_DESCRIPTORS=" + json.dumps(rows, sort_keys=True))
    omni.kit.app.get_app().post_quit(0)


main()
