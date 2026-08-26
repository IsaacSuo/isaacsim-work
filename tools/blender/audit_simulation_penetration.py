"""Audit one exported deformable debug USD for material penetrations.

Run with Blender in background mode:
  blender --background --python audit_simulation_penetration.py -- DEBUG_USD OUTPUT_JSON [options]

Invoke Blender with ``--python-exit-code 1``. Direct Blender execution returns
0 for a passing audit and 1 for either a rejected sample or a tool error; the
JSON ``valid`` field distinguishes those cases. Use the lightweight PhysX-side
wrapper when stable 0/1/2 batch exit codes are required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soft_body.penetration import (  # noqa: E402
    exact_triangle_crossings,
    penetration_exceeds_tolerance,
)
from soft_body.tet_quality import compute_tet_surface_topology  # noqa: E402


BODY_NAME = re.compile(
    r"Body_(?P<index>\d+)_(?P<layer>Visual|CollisionTetSurface|SimulationTetSurface)(?:\.\d+)?$"
)


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("debug_usd", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--debug-report",
        type=Path,
        default=None,
        help="Tet quality JSON; defaults to DEBUG_USD with a .json suffix.",
    )
    parser.add_argument("--penetration-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--max-inverted-tets", type=int, default=0)
    parser.add_argument("--max-non-manifold-edges", type=int, default=0)
    parser.add_argument(
        "--support-y",
        type=float,
        default=None,
        help="Optional Isaac Y support plane; debug USD is converted to Blender Z-up.",
    )
    parser.add_argument("--ground-tolerance-mm", type=float, default=2.0)
    return parser.parse_args(raw)


def mesh_arrays(obj):
    matrix = obj.matrix_world
    points = np.asarray(
        [tuple(matrix @ vertex.co) for vertex in obj.data.vertices],
        dtype=np.float64,
    )
    triangles = []
    for polygon in obj.data.polygons:
        if len(polygon.vertices) != 3:
            raise ValueError(f"Audit mesh must be triangulated: {obj.name}")
        triangles.append(tuple(int(value) for value in polygon.vertices))
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(points) or not len(triangles):
        raise ValueError(f"Audit mesh has no usable surface: {obj.name}")
    referenced = np.unique(triangles.reshape(-1))
    return points, triangles, referenced


def make_bvh(points, triangles, *, epsilon=1.0e-7):
    return BVHTree.FromPolygons(
        [tuple(point) for point in points],
        [tuple(int(value) for value in face) for face in triangles],
        all_triangles=True,
        epsilon=epsilon,
    )


def signed_nearest_distances(query_points, target_bvh):
    signed = []
    for point in query_points:
        nearest, normal, _, distance = target_bvh.find_nearest(tuple(point))
        if nearest is None:
            raise RuntimeError("BVH nearest-surface query failed")
        delta = np.asarray(tuple(point), dtype=np.float64) - np.asarray(
            tuple(nearest), dtype=np.float64
        )
        normal_array = np.asarray(tuple(normal), dtype=np.float64)
        signed.append(
            -float(distance)
            if float(np.dot(delta, normal_array)) < 0.0
            else float(distance)
        )
    return np.asarray(signed, dtype=np.float64)


def compare_surfaces(first_name, first_data, second_name, second_data, tolerance_m):
    first_points, first_triangles, first_referenced = first_data
    second_points, second_triangles, second_referenced = second_data
    first_bvh = make_bvh(first_points, first_triangles)
    second_bvh = make_bvh(second_points, second_triangles)
    candidates = first_bvh.overlap(second_bvh)
    crossings = exact_triangle_crossings(
        first_points,
        first_triangles,
        second_points,
        second_triangles,
        candidates,
        epsilon=1.0e-8,
    )

    first_signed = signed_nearest_distances(
        first_points[first_referenced], second_bvh
    )
    second_signed = signed_nearest_distances(
        second_points[second_referenced], first_bvh
    )
    first_inside = -first_signed[first_signed < 0.0]
    second_inside = -second_signed[second_signed < 0.0]
    # The exact triangle test establishes that the surfaces really cross.
    # Nearest normals are used only to measure depth, never as a standalone
    # inside/outside classifier. Prefer the authored second layer relative to
    # the first reference surface, matching the validated Garage analysis.
    maximum_depth = (
        float(np.max(second_inside))
        if crossings["exact_crossing_triangle_pairs"] and len(second_inside)
        else 0.0
    )
    failed = penetration_exceeds_tolerance(maximum_depth, tolerance_m)
    return {
        "first": first_name,
        "second": second_name,
        **crossings,
        "minimum_first_vertex_to_second_surface_m": float(
            np.min(np.abs(first_signed))
        ),
        "minimum_second_vertex_to_first_surface_m": float(
            np.min(np.abs(second_signed))
        ),
        "first_vertices_inside_second": int(len(first_inside)),
        "second_vertices_inside_first": int(len(second_inside)),
        "maximum_penetration_depth_m": maximum_depth,
        "penetration_tolerance_m": float(tolerance_m),
        "containment_without_surface_crossing_checked": False,
        "passed": not failed,
    }


def load_tet_checks(debug_report_path, maximum_inverted):
    source_payload = json.loads(debug_report_path.read_text(encoding="utf-8"))
    payload = source_payload.get("deformable_collision_debug") or source_payload
    checks = []
    for mesh in payload.get("meshes") or []:
        quality = mesh.get("tet_quality")
        if quality is None:
            continue
        mesh_name = str(mesh.get("name") or "")
        if "CollisionTetSurface" in mesh_name:
            checks.append(
                {
                    "mesh": mesh_name,
                    "tetrahedron_count": int(
                        quality.get("tetrahedron_count", 0)
                    ),
                    "applicable": False,
                    "available": False,
                    "reason": "collision_tet_bind_mapping_not_used_for_inversion",
                    "passed": True,
                }
            )
            continue
        bind_mapping = quality.get("bind_mapping")
        mapping_trusted = (
            bool(bind_mapping.get("trusted"))
            if isinstance(bind_mapping, dict)
            else "SimulationTetSurface" in mesh_name
        )
        if not mapping_trusted:
            checks.append(
                {
                    "mesh": mesh_name,
                    "tetrahedron_count": int(
                        quality.get("tetrahedron_count", 0)
                    ),
                    "applicable": True,
                    "available": False,
                    "reason": (bind_mapping or {}).get(
                        "reason", "simulation_tet_bind_mapping_untrusted"
                    ),
                    "maximum_allowed_inverted_tets": int(maximum_inverted),
                    "passed": False,
                }
            )
            continue
        inverted = int(quality.get("inverted_from_bind_pose_count", 0))
        minimum_ratio = quality.get("minimum_signed_volume_ratio_to_bind")
        passed = inverted <= maximum_inverted and (
            minimum_ratio is None or float(minimum_ratio) >= 0.0
        )
        checks.append(
            {
                "mesh": mesh.get("name"),
                "tetrahedron_count": int(quality.get("tetrahedron_count", 0)),
                "applicable": True,
                "available": True,
                "bind_mapping_source": (
                    (bind_mapping or {}).get("source")
                    or "legacy_simulation_tet_deformable_pose"
                ),
                "inverted_from_bind_pose_count": inverted,
                "minimum_signed_volume_ratio_to_bind": minimum_ratio,
                "maximum_allowed_inverted_tets": int(maximum_inverted),
                "passed": bool(passed),
            }
        )
    if not checks:
        checks.append(
            {
                "mesh": "TetQualityRecords",
                "available": False,
                "maximum_allowed_inverted_tets": int(maximum_inverted),
                "passed": False,
            }
        )
    return payload, checks


def load_tet_topology_checks(payload, maximum_non_manifold):
    checks = []
    for mesh in payload.get("meshes") or []:
        mesh_name = str(mesh.get("name") or "")
        if not (
            "CollisionTetSurface" in mesh_name
            or "SimulationTetSurface" in mesh_name
        ):
            continue
        topology = mesh.get("tet_volume_topology")
        if not isinstance(topology, dict):
            checks.append(
                {
                    "mesh": mesh_name,
                    "available": False,
                    "reason": "tet_volume_connectivity_not_recorded",
                    "passed": False,
                }
            )
            continue
        non_manifold_edges = int(
            topology.get("non_manifold_boundary_edges", 0)
        )
        non_manifold_faces = int(
            topology.get("faces_with_more_than_two_incident_tets", 0)
        )
        checks.append(
            {
                "mesh": mesh_name,
                "available": True,
                **topology,
                "maximum_allowed_non_manifold_boundary_edges": int(
                    maximum_non_manifold
                ),
                "maximum_allowed_faces_with_more_than_two_incident_tets": int(
                    maximum_non_manifold
                ),
                "passed": bool(
                    non_manifold_edges <= maximum_non_manifold
                    and non_manifold_faces <= maximum_non_manifold
                ),
            }
        )
    if not checks:
        checks.append(
            {
                "mesh": "TetVolumeTopologyRecords",
                "available": False,
                "reason": "tet_volume_connectivity_not_recorded",
                "passed": False,
            }
        )
    return checks


def main():
    args = parse_args()
    debug_usd = args.debug_usd.resolve()
    output_json = args.output_json.resolve()
    debug_report = (
        args.debug_report.resolve()
        if args.debug_report is not None
        else debug_usd.with_suffix(".json")
    )
    if args.debug_report is None and not debug_report.is_file():
        legacy_report = debug_usd.parent / "run_complete.json"
        if legacy_report.is_file():
            debug_report = legacy_report
    if not debug_usd.is_file():
        raise FileNotFoundError(debug_usd)
    if not debug_report.is_file():
        raise FileNotFoundError(debug_report)
    if args.penetration_tolerance_mm < 0.0 or args.ground_tolerance_mm < 0.0:
        raise ValueError("Audit tolerances must be non-negative")
    if args.max_inverted_tets < 0 or args.max_non_manifold_edges < 0:
        raise ValueError("Audit count thresholds must be non-negative")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(
        filepath=str(debug_usd),
        import_cameras=False,
        import_curves=False,
        import_lights=False,
        import_materials=False,
        import_volumes=False,
    )
    bodies = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        match = BODY_NAME.search(obj.name)
        if match:
            bodies.setdefault(int(match.group("index")), {})[
                match.group("layer")
            ] = obj
    if len(bodies) < 2 or any("Visual" not in layers for layers in bodies.values()):
        raise RuntimeError(f"Expected at least two complete body visuals: {sorted(bodies)}")

    arrays = {
        (index, layer): mesh_arrays(obj)
        for index, layers in bodies.items()
        for layer, obj in layers.items()
    }
    tolerance_m = args.penetration_tolerance_mm / 1000.0
    surface_checks = []
    body_indices = sorted(bodies)
    for offset, first_index in enumerate(body_indices):
        for second_index in body_indices[offset + 1 :]:
            surface_checks.append(
                compare_surfaces(
                    f"Body_{first_index:02d}_Visual",
                    arrays[(first_index, "Visual")],
                    f"Body_{second_index:02d}_Visual",
                    arrays[(second_index, "Visual")],
                    tolerance_m,
                )
            )
            for deformable_index, other_index in (
                (first_index, second_index),
                (second_index, first_index),
            ):
                for layer in ("CollisionTetSurface", "SimulationTetSurface"):
                    key = (deformable_index, layer)
                    if key not in arrays:
                        continue
                    surface_checks.append(
                        compare_surfaces(
                            f"Body_{other_index:02d}_Visual",
                            arrays[(other_index, "Visual")],
                            f"Body_{deformable_index:02d}_{layer}",
                            arrays[key],
                            tolerance_m,
                        )
                    )

    surface_topology_diagnostics = []
    for (index, layer), (points, triangles, _) in sorted(arrays.items()):
        if layer == "Visual":
            continue
        topology = compute_tet_surface_topology(triangles, points=points)
        surface_topology_diagnostics.append(
            {
                "mesh": f"Body_{index:02d}_{layer}",
                **topology,
                "authoritative": False,
                "note": (
                    "Imported surface triangles cannot distinguish every "
                    "internal Tet face; pass/fail uses recorded volume connectivity."
                ),
            }
        )

    debug_payload, tet_checks = load_tet_checks(
        debug_report, args.max_inverted_tets
    )
    topology_checks = load_tet_topology_checks(
        debug_payload, args.max_non_manifold_edges
    )
    ground_checks = []
    if args.support_y is not None:
        tolerance = args.ground_tolerance_mm / 1000.0
        for index in body_indices:
            points, _, referenced = arrays[(index, "Visual")]
            minimum_y = float(np.min(points[referenced, 2]))
            depth = max(0.0, float(args.support_y) - minimum_y)
            ground_checks.append(
                {
                    "body": f"Body_{index:02d}_Visual",
                    "minimum_surface_y_m": minimum_y,
                    "support_y_m": float(args.support_y),
                    "penetration_depth_m": depth,
                    "penetration_tolerance_m": tolerance,
                    "passed": not penetration_exceeds_tolerance(depth, tolerance),
                }
            )

    failures = []
    for category, checks in (
        ("surface_penetration", surface_checks),
        ("tet_inversion", tet_checks),
        ("surface_topology", topology_checks),
        ("support_penetration", ground_checks),
    ):
        for check in checks:
            if not check["passed"]:
                failures.append(
                    {
                        "category": (
                            "audit_data_unavailable"
                            if check.get("available") is False
                            else category
                        ),
                        "subject": check.get("mesh")
                        or check.get("body")
                        or f"{check.get('first')} vs {check.get('second')}",
                    }
                )

    audit_data_available = not any(
        failure["category"] == "audit_data_unavailable" for failure in failures
    )

    result = {
        "schema_version": 2,
        "valid": audit_data_available,
        "passed": audit_data_available and not failures,
        "debug_usd": str(debug_usd),
        "debug_report": str(debug_report),
        "frame": debug_payload.get("frame"),
        "thresholds": {
            "penetration_tolerance_m": tolerance_m,
            "maximum_inverted_tets": args.max_inverted_tets,
            "maximum_non_manifold_edges": args.max_non_manifold_edges,
            "ground_tolerance_m": args.ground_tolerance_mm / 1000.0,
        },
        "bodies": {
            str(index): sorted(layers) for index, layers in sorted(bodies.items())
        },
        "surface_checks": surface_checks,
        "tet_checks": tet_checks,
        "topology_checks": topology_checks,
        "surface_topology_diagnostics": surface_topology_diagnostics,
        "ground_checks": ground_checks,
        "failures": failures,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[penetration-audit] valid={result['valid']} passed={result['passed']} "
        f"frame={result['frame']} failures={len(failures)} report={output_json}",
        flush=True,
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        if len(raw) >= 2:
            failure_path = Path(raw[1]).resolve()
            failure = {
                "schema_version": 2,
                "valid": False,
                "passed": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            print(
                f"[penetration-audit] valid=False report={failure_path} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
        raise
    if exit_code:
        raise RuntimeError("Penetration audit did not pass")
