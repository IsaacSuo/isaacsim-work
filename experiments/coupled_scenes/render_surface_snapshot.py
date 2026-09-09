"""Render native final-state water inside the authored tank (Blender)."""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.coupled_scenes.blender_coupled_event_overlay import _read_obj, _water_material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout",type=Path,required=True)
    parser.add_argument("--surface",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    provenance=json.loads((args.surface.parent/"snapshot_surface.json").read_text(encoding="utf-8"))
    assert provenance["complete"]
    args.output.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(args.layout/"01_still_water.blend"))
    scene=bpy.context.scene
    placeholder=bpy.data.objects.get("REFERENCE_WATER_NOT_SIMULATED")
    assert placeholder is not None
    bpy.data.objects.remove(placeholder,do_unlink=True)
    # The original label referred to a design water level, not this final state.
    label=bpy.data.objects.get("Reference label")
    if label:
        label.data.body="PHYSX WATER  /  t = 8.0 s  /  DIAGNOSTIC PREVIEW"
        label.data.size=.018
    vertices,faces=_read_obj(args.surface)
    mesh=bpy.data.meshes.new("NativeFinalWaterMesh")
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    obj=bpy.data.objects.new("NativeFinalWater_t8s",mesh)
    bpy.data.collections["SurfaceStudy"].objects.link(obj)
    obj.data.materials.append(_water_material())
    for poly in mesh.polygons:
        poly.use_smooth=True
    obj["source"]=str(args.surface)
    obj["mesh_geometry_smoothing_iterations"]=provenance["mesh_smoothing_iters"]
    scene["fluid_simulated"]=True
    scene["static_water_gate_passed"]=False
    scene["snapshot_seconds"]=8.
    # Keep original scene, floor grid and lighting. Higher quality only.
    scene.cycles.samples=96
    scene.cycles.use_denoising=True
    scene.cycles.max_bounces=12
    scene.cycles.transmission_bounces=12
    prefs=bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type="OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use=device.type!="CPU"
    scene.cycles.device="GPU"
    scene.render.resolution_x=1600
    scene.render.resolution_y=1200
    scene.render.resolution_percentage=100
    scene.camera=bpy.data.objects["View_00"]
    spec=json.loads((args.layout/"scene_spec.json").read_text(encoding="utf-8"))
    origin=spec["tank_origin_isaac"]
    centre=Vector((origin[0],-origin[2],origin[1]+.08))
    data=bpy.data.cameras.new("WaterSurfaceLowAngle")
    data.lens=55
    low=bpy.data.objects.new(data.name,data)
    bpy.data.collections["SurfaceStudy"].objects.link(low)
    low.location=centre+Vector((1.5,-2.1,1.1))
    low.rotation_euler=(centre-low.location).to_track_quat("-Z","Y").to_euler()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/"static_water_t8s.blend"))
    for name,cam in [("overview",bpy.data.objects["View_00"]),("surface_close",low)]:
        scene.camera=cam
        scene.render.filepath=str(args.output/(name+".png"))
        bpy.ops.render.render(write_still=True)
    print('[snapshot-render-complete]',flush=True)


if __name__ == "__main__":
    main()
