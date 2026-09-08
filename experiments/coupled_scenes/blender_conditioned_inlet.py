"""Render the guide or hide upstream water at the physical outlet plane."""


def hide_water_above_outlet(material, outlet_y):
    """World-space shader visibility; keep full reconstructed mesh and physics.

    Isaac Y maps to Blender Z. Hide both the surface and absorption above the
    outlet for all ray types, not just the camera. No synthetic cap is added.
    """
    nodes=material.node_tree.nodes
    links=material.node_tree.links
    output=next(node for node in nodes if node.type=='OUTPUT_MATERIAL')
    original=output.inputs['Surface'].links[0].from_socket
    position=nodes.new('ShaderNodeNewGeometry')
    position.name='UpstreamWorldPosition'
    separate=nodes.new('ShaderNodeSeparateXYZ')
    links.new(position.outputs['Position'],separate.inputs['Vector'])
    above=nodes.new('ShaderNodeMath')
    above.name='AbovePhysicalOutlet'
    above.operation='GREATER_THAN'
    above.inputs[1].default_value=float(outlet_y)
    links.new(separate.outputs['Z'],above.inputs[0])
    transparent=nodes.new('ShaderNodeBsdfTransparent')
    mix=nodes.new('ShaderNodeMixShader')
    mix.name='HideUpstreamWaterSurface'
    links.new(above.outputs[0],mix.inputs[0])
    links.new(original,mix.inputs[1])
    links.new(transparent.outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],output.inputs['Surface'])
    density=nodes.new('ShaderNodeMath')
    density.name='DownstreamOnlyWaterAbsorption'
    density.operation='MULTIPLY_ADD'
    links.new(above.outputs[0],density.inputs[0])
    absorption=nodes.get('WaterThicknessAbsorption')
    if absorption is not None:
        strength=float(absorption.inputs['Density'].default_value)
        density.inputs[1].default_value=-strength
        density.inputs[2].default_value=strength
        links.new(density.outputs[0],absorption.inputs['Density'])
    material['hidden_above_isaac_y']=float(outlet_y)


def add_guide(scene, collection, config):
    import bpy
    from coupled_scene.conditioned_inlet import guide_panels
    material=bpy.data.materials.new('ConditionedInletShell')
    material.use_nodes=True
    shader=next(node for node in material.node_tree.nodes if node.type=='BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value=(.13,.15,.17,1)
    shader.inputs['Metallic'].default_value=.65
    shader.inputs['Roughness'].default_value=.3
    for name,centre,size in guide_panels(config):
        bpy.ops.mesh.primitive_cube_add(size=1.,location=(centre[0],-centre[2],centre[1]))
        obj=bpy.context.object
        obj.name='ConditionedInlet_'+name
        for owner in list(obj.users_collection): owner.objects.unlink(obj)
        collection.objects.link(obj)
        obj.dimensions=(size[0],size[2],size[1])
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        obj.data.materials.append(material)
        obj.select_set(False)
