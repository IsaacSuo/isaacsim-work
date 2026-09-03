"""Blender render overlay for a coupled glass-cabinet pour event."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import bpy
from mathutils import Vector


def _isaac_to_blender(values) -> Vector:
    x, y, z = (float(value) for value in values)
    return Vector((x, -z, y))


def _principled_input(node, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def _set_principled(node, value, *names) -> None:
    socket = _principled_input(node, *names)
    if socket is not None:
        socket.default_value = value


def _glass_material():
    material = bpy.data.materials.new("CoupledCabinetGlass")
    material.use_nodes = True
    principled = next(
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )
    _set_principled(principled, (0.97, 0.995, 1.0, 1.0), "Base Color")
    _set_principled(principled, 0.035, "Roughness")
    _set_principled(principled, 1.46, "IOR")
    _set_principled(principled, 1.0, "Transmission Weight", "Transmission")
    _set_principled(principled, 0.02, "Coat Weight", "Coat")
    return material


def _water_material():
    material = bpy.data.materials.new("CoupledPrimaryWater")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    _set_principled(principled, (0.78, 0.93, 0.98, 1.0), "Base Color")
    _set_principled(principled, 0.055, "Roughness")
    _set_principled(principled, 1.333, "IOR")
    _set_principled(principled, 1.0, "Transmission Weight", "Transmission")
    _set_principled(principled, 0.035, "Coat Weight", "Coat")
    output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    absorption.name = "WaterThicknessAbsorption"
    absorption.inputs["Color"].default_value = (0.72, 0.90, 0.96, 1.0)
    absorption.inputs["Density"].default_value = 0.035
    material.node_tree.links.new(absorption.outputs["Volume"], output.inputs["Volume"])
    return material


def _add_cabinet_area_light(collection, centre, inner_size):
    """Add one fixed apparatus light that remains coherent across camera views."""
    floor_y = float(centre[1])
    light_position = (
        float(centre[0]) - 0.28 * float(inner_size[0]),
        floor_y + float(inner_size[1]) + 0.72,
        float(centre[2]) - 0.18 * float(inner_size[2]),
    )
    light_target = (
        float(centre[0]),
        floor_y + 0.35 * float(inner_size[1]),
        float(centre[2]),
    )
    data = bpy.data.lights.new("CoupledCabinetTopSoftbox", type="AREA")
    data.energy = float(os.environ.get("COUPLED_EVENT_LIGHT_POWER", "650"))
    data.color = (0.92, 0.97, 1.0)
    data.shape = "DISK"
    data.size = 0.85 * max(float(inner_size[0]), float(inner_size[2]))
    light = bpy.data.objects.new("CoupledCabinetTopSoftbox", data)
    collection.objects.link(light)
    light.location = _isaac_to_blender(light_position)
    direction = _isaac_to_blender(light_target) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return light


def _read_obj(path: Path):
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append(_isaac_to_blender(values[1:4]))
            elif line.startswith("f "):
                indices = [int(token.split("/", 1)[0]) - 1 for token in line.split()[1:]]
                if len(indices) >= 3:
                    faces.append(indices)
    if not vertices or not faces:
        raise RuntimeError(f"Liquid surface has no renderable mesh: {path}")
    return vertices, faces


class CoupledEventOverlay:
    def __init__(self, scene, configuration_path: Path, surface_directory: Path | None):
        self.scene = scene
        self.configuration_path = Path(configuration_path).resolve()
        payload = json.loads(self.configuration_path.read_text(encoding="utf-8"))
        if (
            payload.get("product") != "coupled_scene_pbd_pour_event"
            or payload.get("layout")
            not in {
                "glass_cabinet_pour",
                "glass_cabinet_compact_pour",
                "glass_cabinet_pool_drop",
            }
        ):
            raise ValueError(f"Unsupported coupled render event: {self.configuration_path}")
        self.configuration = payload
        self.collection = bpy.data.collections.new("CoupledGlassCabinetEvent")
        scene.collection.children.link(self.collection)
        self.glass_material = _glass_material()
        self.water_material = _water_material()
        self.panels = []
        cabinet = payload["cabinet"]
        centre = cabinet["centre"]
        inner = cabinet["inner_size"]
        thickness = float(cabinet.get("render_thickness", cabinet["wall_thickness"]))
        floor_y = float(centre[1])
        half_x = 0.5 * float(inner[0])
        half_z = 0.5 * float(inner[2])
        panel_specs = (
            (
                "Bottom",
                (centre[0], floor_y - 0.5 * thickness, centre[2]),
                (inner[0] + 2.0 * thickness, thickness, inner[2] + 2.0 * thickness),
            ),
            (
                "Left",
                (centre[0] - half_x - 0.5 * thickness, floor_y + 0.5 * inner[1], centre[2]),
                (thickness, inner[1], inner[2] + 2.0 * thickness),
            ),
            (
                "Right",
                (centre[0] + half_x + 0.5 * thickness, floor_y + 0.5 * inner[1], centre[2]),
                (thickness, inner[1], inner[2] + 2.0 * thickness),
            ),
            (
                "Front",
                (centre[0], floor_y + 0.5 * inner[1], centre[2] - half_z - 0.5 * thickness),
                (inner[0], inner[1], thickness),
            ),
            (
                "Back",
                (centre[0], floor_y + 0.5 * inner[1], centre[2] + half_z + 0.5 * thickness),
                (inner[0], inner[1], thickness),
            ),
        )
        for name, panel_centre, isaac_size in panel_specs:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=_isaac_to_blender(panel_centre))
            panel = bpy.context.object
            panel.name = f"CoupledCabinet_{name}"
            for owner in list(panel.users_collection):
                owner.objects.unlink(panel)
            self.collection.objects.link(panel)
            panel.dimensions = (
                float(isaac_size[0]),
                float(isaac_size[2]),
                float(isaac_size[1]),
            )
            bpy.context.view_layer.objects.active = panel
            panel.select_set(True)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bevel = panel.modifiers.new("GlassEdgeBevel", "BEVEL")
            bevel.width = min(0.004, 0.2 * thickness)
            bevel.segments = 2
            panel.data.materials.append(self.glass_material)
            panel.select_set(False)
            self.panels.append(panel)

        self.apparatus_light = _add_cabinet_area_light(
            self.collection, centre, inner
        )

        self.surface_directory = (
            Path(surface_directory).resolve() if surface_directory is not None else None
        )
        self.surface_paths = {}
        if self.surface_directory is not None:
            if not self.surface_directory.is_dir():
                raise FileNotFoundError(self.surface_directory)
            for path in self.surface_directory.glob("surface_*.obj"):
                match = re.fullmatch(r"surface_(\d+)\.obj", path.name)
                if match:
                    self.surface_paths[int(match.group(1))] = path
        self.fluid_mesh = bpy.data.meshes.new("CoupledPrimaryWaterMesh")
        self.fluid_object = bpy.data.objects.new("CoupledPrimaryWater", self.fluid_mesh)
        self.collection.objects.link(self.fluid_object)
        self.fluid_object.data.materials.append(self.water_material)
        self.loaded_frame = None
        self.loaded_geometry = {}
        self.update(scene.frame_current)
        bpy.app.handlers.frame_change_pre.append(self._frame_change_handler)

    def _frame_change_handler(self, scene, _depsgraph=None):
        self.update(scene.frame_current)

    def update(self, frame: int) -> None:
        frame = int(frame)
        if self.loaded_frame == frame:
            return
        self.loaded_frame = frame
        path = self.surface_paths.get(frame)
        if path is None:
            self.fluid_object.hide_render = True
            self.fluid_object.hide_viewport = True
            return
        vertices, faces = self.loaded_geometry.get(frame, (None, None))
        if vertices is None:
            vertices, faces = _read_obj(path)
            self.loaded_geometry[frame] = (vertices, faces)
        self.fluid_mesh.clear_geometry()
        self.fluid_mesh.from_pydata(vertices, [], faces)
        self.fluid_mesh.update()
        for polygon in self.fluid_mesh.polygons:
            polygon.use_smooth = True
        self.fluid_object.hide_render = False
        self.fluid_object.hide_viewport = False

    def report(self) -> dict:
        return {
            "valid": bool(self.panels) and bool(self.surface_paths),
            "configuration": str(self.configuration_path),
            "scene": self.configuration.get("scene"),
            "layout": self.configuration.get("layout"),
            "panel_objects": [panel.name for panel in self.panels],
            "panel_count": len(self.panels),
            "open_top": True,
            "surface_directory": (
                str(self.surface_directory) if self.surface_directory is not None else None
            ),
            "surface_frames": sorted(self.surface_paths),
            "fluid_object": self.fluid_object.name,
            "glass_material": self.glass_material.name,
            "water_material": self.water_material.name,
            "apparatus_light": self.apparatus_light.name,
            "apparatus_light_power_w": float(self.apparatus_light.data.energy),
            "coordinate_conversion": "Isaac (x,y,z) -> Blender (x,-z,y)",
        }


def create_overlay_from_environment(scene):
    import os

    configuration = os.environ.get("COUPLED_EVENT_CONFIG")
    if not configuration:
        return None
    surface_directory = os.environ.get("COUPLED_FLUID_SURFACE_DIR")
    return CoupledEventOverlay(
        scene,
        Path(configuration),
        Path(surface_directory) if surface_directory else None,
    )
