"""Create topology-capped STL variants for repeatable PhysX deformable cooking.

Run inside Blender:
  blender --background --python prepare_simulation_stls.py -- SOURCE_DIR OUTPUT_DIR [TARGET_FACES]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy


argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) < 2:
    raise SystemExit("Expected SOURCE_DIR OUTPUT_DIR [TARGET_FACES]")

SOURCE_DIR = Path(argv[0]).resolve()
OUTPUT_DIR = Path(argv[1]).resolve()
TARGET_FACES = int(argv[2]) if len(argv) >= 3 else 50000
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def mesh_counts(obj):
    mesh = obj.data
    return len(mesh.vertices), len(mesh.polygons)


def process(source):
    clear_scene()
    bpy.ops.wm.stl_import(filepath=str(source))
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {source}")
    bpy.context.view_layer.objects.active = meshes[0]
    for obj in meshes:
        obj.select_set(True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    source_vertices, source_faces = mesh_counts(obj)
    if source_faces > TARGET_FACES:
        modifier = obj.modifiers.new(name="SimulationTopologyCap", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = max(0.001, TARGET_FACES / source_faces)
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1.0e-7)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    output_vertices, output_faces = mesh_counts(obj)
    output = OUTPUT_DIR / source.name
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=str(output), export_selected_objects=True)
    return {
        "name": source.name,
        "source": str(source),
        "output": str(output),
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "output_vertices": output_vertices,
        "output_faces": output_faces,
        "decimated": source_faces > TARGET_FACES,
        "valid": output.is_file() and output.stat().st_size > 84 and output_faces > 0,
    }


results = []
for source in sorted(SOURCE_DIR.glob("*.stl")):
    try:
        result = process(source)
        print(
            f"[simulation-stl] {source.name} faces={result['source_faces']}"
            f"->{result['output_faces']}"
        )
    except Exception as error:
        result = {
            "name": source.name,
            "source": str(source),
            "valid": False,
            "error": f"{type(error).__name__}: {error}",
        }
        print(f"[simulation-stl-failed] {source.name}: {result['error']}")
    results.append(result)

manifest = {
    "valid": bool(results) and all(row["valid"] for row in results),
    "target_faces": TARGET_FACES,
    "source_directory": str(SOURCE_DIR),
    "output_directory": str(OUTPUT_DIR),
    "models": results,
}
(OUTPUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
if not manifest["valid"]:
    raise SystemExit(1)
