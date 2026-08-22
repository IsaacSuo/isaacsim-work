"""Import an Isaac soft-body USD cache into the current .blend and render it.

Run with Blender, for example:
  blender scene.blend --background --python this_script.py -- CACHE OUTPUT [FRAMES] [SAMPLES] [RESOLUTION] [SELECTED_FRAMES] [STOP_WHEN_OUT_OF_VIEW] [EYE_X EYE_Y EYE_Z TARGET_X TARGET_Y TARGET_Z] [AUTO_ORBIT] [AUTO_RADIUS] [AUTO_ELEVATION] [AUTO_AZIMUTHS] [TX TY TZ] [MATERIAL_PRESET]
"""

import json
import math
import os
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) < 2:
    raise SystemExit("Expected: CACHE_USD OUTPUT_DIRECTORY [FRAME_COUNT]")

CACHE_PATH = Path(argv[0]).resolve()
OUTPUT_DIR = Path(argv[1]).resolve()
REQUESTED_FRAMES = int(argv[2]) if len(argv) >= 3 else 60
CYCLES_SAMPLES = int(argv[3]) if len(argv) >= 4 else 32
RENDER_RESOLUTION = int(argv[4]) if len(argv) >= 5 else 640
SELECTED_FRAMES = (
    [int(value) for value in argv[5].split(",") if value]
    if len(argv) >= 6 and argv[5] and argv[5].strip().lower() not in {"none", "all", "-"}
    else None
)
STOP_WHEN_OUT_OF_VIEW = (
    len(argv) >= 7 and argv[6].strip().lower() in {"1", "true", "yes", "on"}
)
CAMERA_EYE = tuple(float(value) for value in argv[7:10]) if len(argv) >= 13 else (-2.5, 1.6, -3.5)
CAMERA_TARGET = tuple(float(value) for value in argv[10:13]) if len(argv) >= 13 else (0.0, 0.0, 0.0)
AUTO_CAMERA_ORBIT = len(argv) >= 14 and argv[13].strip().lower() in {"auto", "orbit", "true", "1"}
AUTO_CAMERA_RADIUS = float(argv[14]) if len(argv) >= 15 and argv[14] else None
AUTO_CAMERA_ELEVATION = float(argv[15]) if len(argv) >= 16 and argv[15] else None
AUTO_CAMERA_AZIMUTHS = (
    tuple(float(value) for value in argv[16].split(",") if value)
    if len(argv) >= 17 and argv[16]
    else (35.0, 155.0, 275.0)
)
CACHE_TRANSLATION = (
    Vector(tuple(float(value) for value in argv[17:20])) if len(argv) >= 20 else Vector((0.0, 0.0, 0.0))
)
MATERIAL_PRESETS = (
    [value.strip().lower() for value in argv[20].split(",") if value.strip()]
    if len(argv) >= 21
    else ["silicone_cloudy"]
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENES_ROOT = Path(os.environ.get("SCENES_ROOT", PROJECT_ROOT / "scenes")).expanduser()
if not SCENES_ROOT.is_dir() and os.name == "nt" and Path(r"Y:\scenes").is_dir():
    SCENES_ROOT = Path(r"Y:\scenes")


def configure_workspace_hdri(scene, scene_name):
    """Use one known workspace HDRI while preserving authored scene lights."""
    world = scene.world
    if scene_name == "apartment":
        if world is None:
            world = bpy.data.worlds.new("ApartmentBakedWorld")
            scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputWorld")
        background = nodes.new("ShaderNodeBackground")
        background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        background.inputs["Strength"].default_value = 0.0
        world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
        return {
            "applied": False,
            "policy": "apartment_baked_environment_plus_authored_sun",
            "path": None,
            "strength": 0.0,
            "rotation_degrees": 0.0,
            "authored_lights_preserved": False,
        }
    if scene_name == "mountain":
        hdri_name = "noon_grass_4k.hdr"
    elif scene_name == "apartment":
        hdri_name = os.environ.get("APARTMENT_HDRI_NAME", "charolettenbrunn_park_4k.hdr")
    else:
        hdri_name = "bryanston_park_sunrise_8k.exr"
    hdri_path = SCENES_ROOT / "HDRI" / hdri_name
    if not hdri_path.is_file():
        raise FileNotFoundError(hdri_path)
    if world is None:
        world = bpy.data.worlds.new("CyclesFallbackHDRIWorld")
        scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    # Keep already bright/open scenes at the established baseline and add only
    # modest fill where the validated camera lands in a genuinely dark area.
    # Indoor scenes without authored Blender lights still need the HDRI as
    # their only physically meaningful environment fill.
    hdri_strengths = {
        "alley": 1.25,
        "bedroom": 1.10,
        "classroom": 1.05,
        "elevator": 2.00,
        "factory": 1.00,
        "swamp": 0.95,
        "warehouse": 1.25,
    }
    hdri_strength = hdri_strengths.get(scene_name, 0.65)
    if scene_name == "apartment":
        hdri_strength = float(os.environ.get("APARTMENT_HDRI_STRENGTH", "0.65"))
    background.inputs["Strength"].default_value = hdri_strength
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    rotation_degrees = 0.0
    if scene_name in {"mountain", "apartment"}:
        if scene_name == "mountain":
            rotation_degrees = float(os.environ.get("MOUNTAIN_HDRI_ROTATION_DEGREES", "0"))
        else:
            rotation_degrees = float(os.environ.get("APARTMENT_HDRI_ROTATION_DEGREES", "-146"))
        texcoord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_degrees)
        world.node_tree.links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
        world.node_tree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
    return {
        "applied": True,
        "policy": "workspace_hdri_required",
        "path": str(hdri_path),
        "strength": float(background.inputs["Strength"].default_value),
        "rotation_degrees": rotation_degrees,
        "authored_lights_preserved": True,
    }


def tune_authored_scene_lights(scene, scene_name):
    """Apply restrained scene-specific boosts to real authored light objects."""
    energy_scales = {
        # This enclosed scene contains one point light and two area lights;
        # strengthening them is more plausible than relying on exterior fill.
        "elevator": 5.00,
    }
    scale = energy_scales.get(scene_name, 1.0)
    adjusted = []
    if scale != 1.0:
        adjusted_data = set()
        for obj in scene.objects:
            if obj.type != "LIGHT":
                continue
            data_key = obj.data.as_pointer()
            if data_key in adjusted_data:
                continue
            adjusted_data.add(data_key)
            previous_energy = float(obj.data.energy)
            obj.data.energy = previous_energy * scale
            adjusted.append(
                {
                    "name": obj.name,
                    "type": obj.data.type,
                    "previous_energy": previous_energy,
                    "new_energy": float(obj.data.energy),
                }
            )
    return {"applied": bool(adjusted), "scale": scale, "lights": adjusted}


def tune_authored_emissive_lights(scene_name):
    """Boost only materials verified to belong to authored room luminaires."""
    material_scales = {
        "bedroom": {"Light": 3.00},
        "classroom": {"ceiling": 2.00},
    }
    adjusted = []
    for material_name, scale in material_scales.get(scene_name, {}).items():
        material = bpy.data.materials.get(material_name)
        if material is None or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.bl_idname != "ShaderNodeBsdfPrincipled":
                continue
            emission_strength = node.inputs.get("Emission Strength")
            if emission_strength is None:
                continue
            previous_strength = float(emission_strength.default_value)
            emission_strength.default_value = previous_strength * scale
            adjusted.append(
                {
                    "material": material_name,
                    "previous_strength": previous_strength,
                    "new_strength": float(emission_strength.default_value),
                }
            )
    return {"applied": bool(adjusted), "materials": adjusted}


def configure_scene_shadow_key(scene, scene_name):
    """Restore a readable soft direct-light shadow for scenes with no usable authored key light.

    The direction follows the dominant sunrise region in the shared Bryanston Park
    HDRI.  A broad sun angle avoids introducing an unnaturally hard second shadow.
    This is render-only: it does not affect the imported animation or physics cache.
    """
    strengths = {
    }
    strength = strengths.get(scene_name)
    if strength is None:
        return {"applied": False}

    # Bright HDRI cluster: equirectangular u ~= 0.5835, v ~= 0.577.
    longitude = math.radians(-30.0)
    latitude = math.radians(13.8)
    to_sun = Vector(
        (
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        )
    )
    ray_direction = -to_sun

    light_data = bpy.data.lights.new(f"{scene_name}_HDRI_ShadowKey", type="SUN")
    light_data.energy = strength
    light_data.angle = math.radians(7.0)
    light_data.use_shadow = True
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)
    light.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()
    return {
        "applied": True,
        "type": "SUN",
        "energy": strength,
        "angle_degrees": 7.0,
        "hdri_sun_longitude_degrees": -30.0,
        "hdri_sun_latitude_degrees": 13.8,
        "ray_direction": list(ray_direction),
    }


def configure_apartment_baked_shadow_overlay(scene, soft_bodies):
    """Add a linked shadow catcher without relighting the baked apartment scene."""
    if CACHE_PATH.parent.name != "apartment" or os.environ.get("APARTMENT_SHADOW_CATCHER", "0") != "1":
        return {"applied": False}

    bpy.ops.mesh.primitive_plane_add(
        size=12.0,
        location=(-4.1, -6.4, -0.477),
    )
    catcher = bpy.context.object
    catcher.name = "ApartmentSoftBodyShadowCatcher"
    catcher.is_shadow_catcher = True

    catcher_material = bpy.data.materials.new("ApartmentShadowCatcherDiffuse")
    catcher_material.use_nodes = True
    catcher.data.materials.append(catcher_material)

    light_data = bpy.data.lights.new("ApartmentBakedShadowKey", type="SUN")
    light_data.energy = 1.0
    light_data.angle = math.radians(8.0)
    light_data.use_shadow = True
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)
    ray_direction = Vector((0.35, 0.25, -1.0)).normalized()
    light.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()

    receivers = bpy.data.collections.new("ApartmentShadowReceivers")
    blockers = bpy.data.collections.new("ApartmentShadowBlockers")
    receivers.objects.link(catcher)
    for soft_body in soft_bodies:
        blockers.objects.link(soft_body)
    light.light_linking.receiver_collection = receivers
    light.light_linking.blocker_collection = blockers
    return {
        "applied": True,
        "receiver": catcher.name,
        "receiver_location": list(catcher.location),
        "receiver_size": 12.0,
        "light": light.name,
        "light_energy": light_data.energy,
        "light_angle_degrees": 8.0,
        "ray_direction": list(ray_direction),
        "light_linked_receiver_only": True,
        "shadow_blocker_soft_body_only": True,
    }


def configure_apartment_shadow_receiver_material():
    """Make only the baked floor physically light-reactive so it can receive shadows."""
    if CACHE_PATH.parent.name != "apartment":
        return {"applied": False}
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hit, location, _normal, polygon_index, hit_object, _matrix = scene.ray_cast(
        depsgraph,
        Vector((-4.1, -6.4, 2.0)),
        Vector((0.0, 0.0, -1.0)),
    )
    if not hit or hit_object is None or hit_object.type != "MESH":
        raise RuntimeError("Could not identify the apartment floor below the soft-body landing point")
    polygon = hit_object.data.polygons[polygon_index]
    material = hit_object.material_slots[polygon.material_index].material
    if material is None or not material.use_nodes or material.node_tree is None:
        raise RuntimeError("Apartment floor hit material is missing or has no nodes")
    changed = []
    new_emission_strength = float(os.environ.get("APARTMENT_FLOOR_EMISSION_STRENGTH", "0.0"))
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        emission_strength = principled_input(node, "Emission Strength")
        if emission_strength is not None:
            changed.append(
                {
                    "node": node.name,
                    "previous_emission_strength": float(emission_strength.default_value),
                }
            )
            emission_strength.default_value = new_emission_strength
    if not changed:
        raise RuntimeError("Apartment floor Principled shader has no Emission Strength input")
    return {
        "applied": True,
        "object": hit_object.name,
        "ray_hit_location": list(location),
        "material": material.name,
        "changes": changed,
        "new_emission_strength": new_emission_strength,
    }


def configure_apartment_hdri_shadow_access(original_objects):
    """Let HDRI shadow rays pass through the baked shell while keeping it camera-visible."""
    if CACHE_PATH.parent.name != "apartment":
        return {"applied": False}
    shadow_passthrough = []
    disabled_lights = []
    for obj in original_objects:
        if obj.type == "MESH" and getattr(obj, "visible_shadow", False):
            obj.visible_shadow = False
            shadow_passthrough.append(obj.name)
        elif obj.type == "LIGHT" and not obj.hide_render:
            obj.hide_render = True
            disabled_lights.append(obj.name)
    return {
        "applied": True,
        "policy": "HDRI_only_for_baked_interior",
        "camera_visibility_preserved": True,
        "shadow_passthrough_mesh_count": len(shadow_passthrough),
        "disabled_authored_lights": disabled_lights,
    }


def configure_apartment_authored_sun(scene, original_objects, soft_bodies):
    """Recreate the apartment's directional light from its imported Sun marker.

    The source asset stores ``Sun`` as an EMPTY parented to ``Root`` rather than
    as a Blender light.  The viewport relationship line therefore records the
    intended direction: rays travel from the marker toward its parent's origin.
    """
    if CACHE_PATH.parent.name != "apartment":
        return {"applied": False}

    disabled_lights = []
    for obj in original_objects:
        if obj.type == "LIGHT" and not obj.hide_render:
            obj.hide_render = True
            disabled_lights.append(obj.name)

    marker = bpy.data.objects.get("Sun")
    if marker is None or marker.type != "EMPTY" or marker.parent is None:
        raise RuntimeError("Apartment Sun direction marker or its Root parent is missing")
    marker_location = marker.matrix_world.translation.copy()
    target = marker.parent.matrix_world.translation.copy()
    ray_direction = (target - marker_location).normalized()

    data = bpy.data.lights.new("ApartmentAuthoredSun", type="SUN")
    data.energy = float(os.environ.get("APARTMENT_SUN_ENERGY", "1.0"))
    data.angle = math.radians(float(os.environ.get("APARTMENT_SUN_ANGLE_DEGREES", "6.0")))
    data.color = (1.0, 0.88, 0.72)
    data.use_shadow = True
    light = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(light)
    light.location = marker_location
    light.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()

    # The environment already contains authored/baked occlusion.  Restrict this
    # reconstructed key light's blockers to the simulated body so the imported
    # room shell does not block the marker direction a second time.
    blockers = bpy.data.collections.new("ApartmentAuthoredSunBlockers")
    for soft_body in soft_bodies:
        blockers.objects.link(soft_body)
    light.light_linking.blocker_collection = blockers
    return {
        "applied": True,
        "type": "SUN",
        "source_marker": marker.name,
        "source_parent": marker.parent.name,
        "location": list(marker_location),
        "target": list(target),
        "ray_direction": list(ray_direction),
        "energy": data.energy,
        "angle_degrees": math.degrees(data.angle),
        "color": list(data.color),
        "disabled_authored_lights": disabled_lights,
        "shadow_blocker_soft_body_only": True,
    }


def isaac_to_blender(values):
    x, y, z = (float(value) for value in values)
    return Vector((x, -z, y))


def look_at(camera, eye_isaac, target_isaac):
    camera.location = isaac_to_blender(eye_isaac)
    direction = isaac_to_blender(target_isaac) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def principled_input(node, *names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def set_principled_value(principled, value, *names):
    socket = principled_input(principled, *names)
    if socket is not None:
        socket.default_value = value


def add_micro_bump(material, principled, *, scale, detail, roughness, strength, distance, name):
    nodes = material.node_tree.nodes
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = f"{name}MicroVariation"
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = roughness
    bump = nodes.new("ShaderNodeBump")
    bump.name = f"{name}MicroBump"
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    material.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    normal = principled_input(principled, "Normal")
    if normal is not None:
        material.node_tree.links.new(bump.outputs["Normal"], normal)


def create_silicone_material(scene_name):
    material = bpy.data.materials.new("TransparentSilicone_Cycles")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    set_principled_value(principled, (0.965, 0.985, 1.0, 1.0), "Base Color")
    set_principled_value(principled, 0.14, "Roughness")
    set_principled_value(principled, 1.41, "IOR")
    set_principled_value(principled, 1.0, "Transmission Weight", "Transmission")
    set_principled_value(principled, 0.025, "Coat Weight", "Coat")

    material_output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
    volume_absorption = nodes.new("ShaderNodeVolumeAbsorption")
    volume_absorption.name = "SiliconeThicknessAbsorption"
    volume_absorption.inputs["Color"].default_value = (0.82, 0.92, 1.0, 1.0)
    volume_absorption.inputs["Density"].default_value = 0.045
    volume_scatter = nodes.new("ShaderNodeVolumeScatter")
    volume_scatter.name = "SiliconeCloudiness"
    volume_scatter.inputs["Color"].default_value = (0.96, 0.985, 1.0, 1.0)
    volume_scatter.inputs["Density"].default_value = 0.9
    volume_scatter.inputs["Anisotropy"].default_value = 0.18
    volume_mix = nodes.new("ShaderNodeAddShader")
    volume_mix.name = "SiliconeVolume"
    material.node_tree.links.new(volume_absorption.outputs["Volume"], volume_mix.inputs[0])
    material.node_tree.links.new(volume_scatter.outputs["Volume"], volume_mix.inputs[1])
    material.node_tree.links.new(volume_mix.outputs[0], material_output.inputs["Volume"])

    shadow_transmission = None
    if scene_name in {"apartment", "mountain"}:
        surface = material_output.inputs["Surface"]
        original_surface = surface.links[0].from_socket
        material.node_tree.links.remove(surface.links[0])
        light_path = nodes.new("ShaderNodeLightPath")
        light_path.name = "SiliconeShadowRaySelector"
        transparent_shadow = nodes.new("ShaderNodeBsdfTransparent")
        transparent_shadow.name = "SiliconeClearShadowRay"
        transparent_shadow.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        diffuse_shadow = nodes.new("ShaderNodeBsdfDiffuse")
        diffuse_shadow.name = "SiliconeShadowBlocker"
        diffuse_shadow.inputs["Color"].default_value = (0.8, 0.88, 0.95, 1.0)
        shadow_value = float(os.environ.get("SILICONE_SHADOW_TRANSMISSION", "0.48"))
        shadow_attenuation = nodes.new("ShaderNodeMixShader")
        shadow_attenuation.name = "SiliconePartialShadowBlocker"
        shadow_attenuation.inputs[0].default_value = 1.0 - shadow_value
        material.node_tree.links.new(transparent_shadow.outputs[0], shadow_attenuation.inputs[1])
        material.node_tree.links.new(diffuse_shadow.outputs[0], shadow_attenuation.inputs[2])
        shadow_mix = nodes.new("ShaderNodeMixShader")
        shadow_mix.name = "SiliconeSurfaceWithShadowTransmission"
        material.node_tree.links.new(original_surface, shadow_mix.inputs[1])
        material.node_tree.links.new(shadow_attenuation.outputs[0], shadow_mix.inputs[2])
        material.node_tree.links.new(light_path.outputs["Is Shadow Ray"], shadow_mix.inputs[0])
        material.node_tree.links.new(shadow_mix.outputs[0], surface)
        shadow_transmission = [shadow_value, shadow_value, shadow_value, 1.0]

    add_micro_bump(
        material,
        principled,
        scale=7.0,
        detail=3.0,
        roughness=0.55,
        strength=0.018,
        distance=0.0004,
        name="Silicone",
    )
    return material, shadow_transmission


def create_rough_white_material():
    material = bpy.data.materials.new("RoughWhiteFilm_Hard")
    material.use_nodes = True
    principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    set_principled_value(principled, (0.82, 0.84, 0.86, 1.0), "Base Color")
    set_principled_value(principled, 0.72, "Roughness")
    set_principled_value(principled, 0.0, "Metallic")
    set_principled_value(principled, 1.46, "IOR")
    set_principled_value(principled, 0.0, "Transmission Weight", "Transmission")
    set_principled_value(principled, 0.04, "Coat Weight", "Coat")
    add_micro_bump(
        material,
        principled,
        scale=38.0,
        detail=5.0,
        roughness=0.7,
        strength=0.16,
        distance=0.0012,
        name="RoughWhiteFilm",
    )
    return material, None


def create_brushed_metal_material():
    material = bpy.data.materials.new("BrushedMetal_Hard")
    material.use_nodes = True
    principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    set_principled_value(principled, (0.42, 0.46, 0.52, 1.0), "Base Color")
    set_principled_value(principled, 0.24, "Roughness")
    set_principled_value(principled, 1.0, "Metallic")
    set_principled_value(principled, 0.36, "Anisotropic IOR Level", "Anisotropic")
    set_principled_value(principled, 0.0, "Transmission Weight", "Transmission")
    add_micro_bump(
        material,
        principled,
        scale=95.0,
        detail=2.0,
        roughness=0.45,
        strength=0.055,
        distance=0.00035,
        name="BrushedMetal",
    )
    return material, None


def create_soft_body_material(preset, scene_name):
    creators = {
        "silicone_cloudy": lambda: create_silicone_material(scene_name),
        "rough_white": create_rough_white_material,
        "brushed_metal": create_brushed_metal_material,
    }
    try:
        return creators[preset]()
    except KeyError as error:
        raise ValueError(f"Unknown material preset: {preset}; expected one of {sorted(creators)}") from error


if not CACHE_PATH.is_file():
    raise FileNotFoundError(CACHE_PATH)

before_objects = set(bpy.data.objects)
bpy.ops.wm.usd_import(
    filepath=str(CACHE_PATH),
    import_cameras=False,
    import_curves=False,
    import_lights=False,
    import_materials=False,
    import_volumes=False,
)
imported_objects = [obj for obj in bpy.data.objects if obj not in before_objects]
mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
if not mesh_objects:
    raise RuntimeError(f"Expected imported animated meshes, got: {[obj.name for obj in imported_objects]}")
soft_bodies = sorted(mesh_objects, key=lambda obj: obj.name)
if len(MATERIAL_PRESETS) not in {1, len(soft_bodies)}:
    raise RuntimeError(
        f"Received {len(MATERIAL_PRESETS)} material presets for {len(soft_bodies)} bodies"
    )
if len(MATERIAL_PRESETS) == 1:
    material_presets = MATERIAL_PRESETS * len(soft_bodies)
else:
    material_presets = MATERIAL_PRESETS
scene_name = CACHE_PATH.parent.name

body_materials = []
for body_index, (soft_body, material_preset) in enumerate(zip(soft_bodies, material_presets)):
    soft_body.name = f"IsaacSoftBodyCache_{body_index:02d}"
    soft_body.location += CACHE_TRANSLATION
    material, shadow_transmission = create_soft_body_material(material_preset, scene_name)
    soft_body.data.materials.clear()
    soft_body.data.materials.append(material)
    for polygon in soft_body.data.polygons:
        polygon.use_smooth = True
    body_materials.append(
        {
            "object": soft_body.name,
            "preset": material_preset,
            "material": material.name,
            "shadow_ray_transmission": shadow_transmission,
        }
    )

scene = bpy.context.scene
scene.render.engine = "CYCLES"
shadow_receiver_material = configure_apartment_shadow_receiver_material()
hdri_shadow_access = {"applied": False, "policy": "not_used_for_apartment"}
world_lighting = configure_workspace_hdri(scene, scene_name)
authored_light_tuning = tune_authored_scene_lights(scene, scene_name)
authored_emissive_tuning = tune_authored_emissive_lights(scene_name)
shadow_key = configure_scene_shadow_key(scene, scene_name)
baked_shadow_overlay = configure_apartment_baked_shadow_overlay(scene, soft_bodies)
apartment_authored_sun = configure_apartment_authored_sun(scene, before_objects, soft_bodies)
scene.cycles.samples = CYCLES_SAMPLES
scene.cycles.use_denoising = True
scene.cycles.device = "GPU"
scene.render.use_persistent_data = True
device_mode = "OPTIX"
try:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type != "CPU"
except Exception as error:
    scene.cycles.device = "CPU"
    device_mode = f"CPU fallback: {error}"

scene.render.resolution_x = RENDER_RESOLUTION
scene.render.resolution_y = RENDER_RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.render.use_file_extension = True
scene.render.filepath = str(OUTPUT_DIR / "frame_")
scene.render.fps = 60
scene.frame_start = 1
scene.frame_end = max(1, REQUESTED_FRAMES)

camera_data = bpy.data.cameras.new("SoftBodyRenderCamera")
camera = bpy.data.objects.new("SoftBodyRenderCamera", camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera.data.lens = 52.0
camera.data.sensor_width = 36.0
camera.data.dof.use_dof = False
look_at(camera, CAMERA_EYE, CAMERA_TARGET)

requested_frame_end = scene.frame_end
last_camera_visible_frame = None
if STOP_WHEN_OUT_OF_VIEW:
    for frame in range(scene.frame_start, requested_frame_end + 1):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        camera_coordinates = []
        for soft_body in soft_bodies:
            evaluated = soft_body.evaluated_get(depsgraph)
            camera_coordinates.extend(
                world_to_camera_view(scene, camera, evaluated.matrix_world @ Vector(corner))
                for corner in evaluated.bound_box
            )
        if any(
            coordinate.z > 0.0
            and 0.0 <= coordinate.x <= 1.0
            and 0.0 <= coordinate.y <= 1.0
            for coordinate in camera_coordinates
        ):
            last_camera_visible_frame = frame
    if last_camera_visible_frame is None:
        raise RuntimeError("Animated soft body is never inside the render camera frustum")
    scene.frame_end = last_camera_visible_frame
    print(
        f"[camera-trim] requested_end={requested_frame_end} "
        f"last_visible={last_camera_visible_frame}",
        flush=True,
    )

probe_frames = sorted({1, max(1, REQUESTED_FRAMES // 2), REQUESTED_FRAMES})
probe_bounds = []
for frame in probe_frames:
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    world_vertices = []
    for soft_body in soft_bodies:
        evaluated = soft_body.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        world_vertices.extend(
            evaluated.matrix_world @ vertex.co for vertex in evaluated_mesh.vertices
        )
        evaluated.to_mesh_clear()
    minimum = [min(vertex[axis] for vertex in world_vertices) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in world_vertices) for axis in range(3)]
    probe_bounds.append({"frame": frame, "minimum": minimum, "maximum": maximum})

if len({tuple(round(value, 5) for value in row["minimum"] + row["maximum"]) for row in probe_bounds}) < 2:
    raise RuntimeError(f"Imported USD mesh does not visibly animate across probe frames: {probe_bounds}")

if SELECTED_FRAMES:
    for stale_preview in OUTPUT_DIR.glob("preview_*.png"):
        stale_preview.unlink()
    camera_candidates = []
    for frame in SELECTED_FRAMES:
        if frame < scene.frame_start or frame > scene.frame_end:
            raise ValueError(f"Selected frame {frame} is outside {scene.frame_start}..{scene.frame_end}")
        scene.frame_set(frame)
        if AUTO_CAMERA_ORBIT:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            corners = []
            for soft_body in soft_bodies:
                evaluated = soft_body.evaluated_get(depsgraph)
                corners.extend(
                    evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box
                )
            minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
            maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
            target = (minimum + maximum) * 0.5
            radius = AUTO_CAMERA_RADIUS or max(4.5, (maximum - minimum).length * 5.0)
            elevation = AUTO_CAMERA_ELEVATION or max(1.8, radius * 0.42)
            for view_index, azimuth_degrees in enumerate(AUTO_CAMERA_AZIMUTHS):
                azimuth = math.radians(azimuth_degrees)
                camera.location = target + Vector(
                    (radius * math.cos(azimuth), radius * math.sin(azimuth), elevation)
                )
                camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
                camera.data.lens = 48.0
                output_path = OUTPUT_DIR / f"preview_{frame:04d}_view{view_index}.png"
                scene.render.filepath = str(output_path)
                bpy.ops.render.render(write_still=True)
                camera_candidates.append(
                    {
                        "frame": frame,
                        "view": view_index,
                        "azimuth_degrees": azimuth_degrees,
                        "eye_blender": list(camera.location),
                        "target_blender": list(target),
                    }
                )
        else:
            scene.render.filepath = str(OUTPUT_DIR / f"preview_{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
    rendered_frames = sorted(OUTPUT_DIR.glob("preview_*.png"))
    expected_render_count = len(SELECTED_FRAMES) * (len(AUTO_CAMERA_AZIMUTHS) if AUTO_CAMERA_ORBIT else 1)
else:
    camera_candidates = []
    bpy.ops.render.render(animation=True)
    rendered_frames = sorted(OUTPUT_DIR.glob("frame_*.png"))
    expected_render_count = scene.frame_end - scene.frame_start + 1
if len(rendered_frames) != expected_render_count:
    raise RuntimeError(f"Rendered {len(rendered_frames)} frames, expected {expected_render_count}")

report = {
    "valid": True,
    "source_blend": bpy.data.filepath,
    "cache_usd": str(CACHE_PATH),
    "cache_translation_blender": list(CACHE_TRANSLATION),
    "imported_mesh": soft_bodies[0].name,
    "imported_meshes": [soft_body.name for soft_body in soft_bodies],
    "body_count": len(soft_bodies),
    "material_preset": material_presets[0] if len(material_presets) == 1 else None,
    "material_presets": material_presets,
    "body_materials": body_materials,
    "material_name": body_materials[0]["material"],
    "vertex_count": sum(len(soft_body.data.vertices) for soft_body in soft_bodies),
    "polygon_count": sum(len(soft_body.data.polygons) for soft_body in soft_bodies),
    "modifiers": {
        soft_body.name: [modifier.type for modifier in soft_body.modifiers]
        for soft_body in soft_bodies
    },
    "frame_start": scene.frame_start,
    "frame_end": scene.frame_end,
    "requested_frame_end": requested_frame_end,
    "stop_when_out_of_view": STOP_WHEN_OUT_OF_VIEW,
    "last_camera_visible_frame": last_camera_visible_frame,
    "fps": scene.render.fps,
    "renderer": scene.render.engine,
    "cycles_samples": scene.cycles.samples,
    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
    "cycles_device": device_mode,
    "persistent_data": scene.render.use_persistent_data,
    "world_lighting": world_lighting,
    "authored_light_tuning": authored_light_tuning,
    "authored_emissive_tuning": authored_emissive_tuning,
    "shadow_key": shadow_key,
    "baked_shadow_overlay": baked_shadow_overlay,
    "apartment_authored_sun": apartment_authored_sun,
    "shadow_receiver_material": shadow_receiver_material,
    "hdri_shadow_access": hdri_shadow_access,
    "shadow_ray_transmission": [row["shadow_ray_transmission"] for row in body_materials],
    "camera_eye_isaac": CAMERA_EYE,
    "camera_target_isaac": CAMERA_TARGET,
    "auto_camera_orbit": AUTO_CAMERA_ORBIT,
    "auto_camera_radius": AUTO_CAMERA_RADIUS,
    "auto_camera_elevation": AUTO_CAMERA_ELEVATION,
    "camera_candidates": camera_candidates,
    "probe_bounds": probe_bounds,
    "rendered_frame_count": len(rendered_frames),
    "selected_frames": SELECTED_FRAMES,
    "output_directory": str(OUTPUT_DIR),
}
(OUTPUT_DIR / "blender_render_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
