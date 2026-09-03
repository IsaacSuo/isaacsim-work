"""Audited runtime discovery for Isaac/PhysX diffuse-particle feasibility gates."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from omni.physx.bindings import _physx
from pxr import PhysicsSchemaTools, Sdf, UsdGeom, UsdUtils


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def scalar_json_value(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return {"type": type(value).__name__, "length": len(value)}
    return str(value)


class PhysxDiffuseProbe:
    """Persist evidence without assuming that internal diffuse buffers are public."""

    def __init__(
        self,
        stage,
        scene_path,
        particle_path,
        output_path,
        configuration,
        enabled_requested,
    ):
        self.stage = stage
        self.scene_path = Sdf.Path(scene_path)
        self.particle_path = Sdf.Path(particle_path)
        self.output_path = Path(output_path)
        self.statistics_interface = None
        self.statistics_setup_error = None
        self.stage_id = None
        self.encoded_scene_path = None
        try:
            self.statistics_interface = _physx.acquire_physx_statistics_interface()
            stage_cache = UsdUtils.StageCache.Get()
            stage_cache_id = stage_cache.GetId(stage)
            if not stage_cache_id.IsValid():
                stage_cache_id = stage_cache.Insert(stage)
            self.stage_id = stage_cache_id.ToLongInt()
            self.encoded_scene_path = PhysicsSchemaTools.encodeSdfPath(
                self.scene_path
            )
        except Exception as exc:
            self.statistics_setup_error = f"{type(exc).__name__}: {exc}"
        simulation_interface = None
        simulation_interface_error = None
        try:
            from omni.physx import get_physx_simulation_interface

            simulation_interface = get_physx_simulation_interface()
        except Exception as exc:
            simulation_interface_error = f"{type(exc).__name__}: {exc}"
        self.report = {
            "schema": 1,
            "created_utc": utc_now_iso(),
            "purpose": (
                "Determine whether Isaac/PhysX diffuse particles are generated, "
                "observable, and exportable without conflating those claims."
            ),
            "configuration": configuration,
            "enabled_requested": bool(enabled_requested),
            "discovery": {
                "usd_schema_present": True,
                "statistics_setup_error": self.statistics_setup_error,
                "binding_symbols_containing_diffuse": sorted(
                    name for name in dir(_physx) if "diffuse" in name.lower()
                ),
                "simulation_interface_symbols_containing_diffuse": (
                    sorted(
                        name
                        for name in dir(simulation_interface)
                        if "diffuse" in name.lower()
                    )
                    if simulation_interface is not None
                    else []
                ),
                "simulation_interface_error": simulation_interface_error,
                "public_buffer_export_discovered": False,
            },
            "state": {
                "complete": False,
                "snapshots": 0,
                "updated_utc": utc_now_iso(),
            },
            "snapshots": [],
        }
        atomic_write_json(self.output_path, self.report)

    def _scene_statistics(self):
        if self.statistics_interface is None:
            return {
                "available": False,
                "error": self.statistics_setup_error or "interface unavailable",
            }
        stats = _physx.PhysicsSceneStats()
        try:
            success = self.statistics_interface.get_physx_scene_statistics(
                self.stage_id, self.encoded_scene_path[0], stats
            )
            if not success:
                return {"available": False, "error": "query returned false"}
            fields = {}
            for name in dir(stats):
                if name.startswith("gpu_mem") or name in {
                    "nb_active_dynamic_rigids",
                    "nb_active_kinematic_rigids",
                }:
                    try:
                        fields[name] = scalar_json_value(getattr(stats, name))
                    except Exception as exc:
                        fields[name] = f"{type(exc).__name__}: {exc}"
            return {"available": True, "fields": fields}
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def _usd_state(self):
        diffuse_attributes = []
        point_carriers = []
        for prim in self.stage.Traverse():
            for attribute in prim.GetAttributes():
                if "diffuse" in attribute.GetName().lower():
                    try:
                        value = scalar_json_value(attribute.Get())
                    except Exception as exc:
                        value = f"{type(exc).__name__}: {exc}"
                    diffuse_attributes.append(
                        {
                            "prim": str(prim.GetPath()),
                            "attribute": attribute.GetName(),
                            "value": value,
                        }
                    )
            carrier = None
            positions = None
            if prim.IsA(UsdGeom.PointInstancer):
                carrier = "PointInstancer"
                positions = UsdGeom.PointInstancer(prim).GetPositionsAttr().Get()
            elif prim.IsA(UsdGeom.Points):
                carrier = "Points"
                positions = UsdGeom.Points(prim).GetPointsAttr().Get()
            if carrier is not None:
                point_carriers.append(
                    {
                        "prim": str(prim.GetPath()),
                        "type": carrier,
                        "point_count": 0 if positions is None else len(positions),
                        "is_primary_particle_set": prim.GetPath() == self.particle_path,
                    }
                )
        return {
            "diffuse_attributes": diffuse_attributes,
            "point_carriers": point_carriers,
            "non_primary_point_count": sum(
                item["point_count"]
                for item in point_carriers
                if not item["is_primary_particle_set"]
            ),
        }

    def capture(self, physics_step, simulation_time, phase):
        snapshot = {
            "physics_step": int(physics_step),
            "simulation_time": float(simulation_time),
            "phase": str(phase),
            "usd": self._usd_state(),
            "scene_statistics": self._scene_statistics(),
        }
        self.report["snapshots"].append(snapshot)
        self.report["state"].update(
            {
                "snapshots": len(self.report["snapshots"]),
                "updated_utc": utc_now_iso(),
            }
        )
        atomic_write_json(self.output_path, self.report)
        return snapshot

    def complete(self):
        self.report["state"].update(
            {
                "complete": True,
                "completed_utc": utc_now_iso(),
                "updated_utc": utc_now_iso(),
            }
        )
        atomic_write_json(self.output_path, self.report)
        return self.report
