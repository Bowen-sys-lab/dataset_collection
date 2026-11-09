"""Build editable, grounded fixtures in the current model02 working copy."""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path('D:/IsaacSim_Demo/factory/model02_thermal_v1')


def material(name, color, metal=0., roughness=.5, emission=0.):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = (*color, 1.)
    bsdf.inputs['Metallic'].default_value = metal
    bsdf.inputs['Roughness'].default_value = roughness
    if emission:
        bsdf.inputs['Emission Color'].default_value = (*color, 1.)
        bsdf.inputs['Emission Strength'].default_value = emission
    return mat


def attach(obj, parent):
    world = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = world


def empty(name, parent, collection):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        collection.objects.link(obj)
    if parent:
        attach(obj, parent)
    return obj


def build_lighting():
    assert Path(bpy.data.filepath) == ROOT/'model02_thermal_v1.blend'
    assert not bpy.data.objects.get('ROAD_001'), 'Already built; refusing duplicate lights'
    scene = bpy.context.scene
    collection = bpy.data.collections.get('Model02_WeakLight')
    if collection is None:
        collection = bpy.data.collections.new('Model02_WeakLight')
        scene.collection.children.link(collection)
    root = empty('IndustrialPark02', None, collection)
    background = empty('BackgroundOriginal', root, collection)
    empty('InspectionZone01', root, collection)
    light_root = empty('Lighting', root, collection)
    diagnostics = empty('Diagnostics', None, collection)
    for obj in list(scene.objects):
        if obj.name.startswith('11_29_gc') or obj.name == 'Cinema_4D_编辑器':
            attach(obj, background)
        elif obj.name == 'Model02_DiagnosticCamera':
            attach(obj, diagnostics)
    day_sun = bpy.data.objects.get('Model02_DiagnosticSun')
    if day_sun:
        bpy.data.objects.remove(day_sun, do_unlink=True)
    steel = material('LGT_PowderCoat_Graphite', (.065, .08, .095), .65, .36)
    metal = material('LGT_GalvanisedMetal', (.28, .31, .33), .8, .28)
    concrete = material('LGT_ConcreteFooting', (.32, .32, .3), 0., .82)
    lens = material('LGT_Diffuser_Warm', (1., .84, .63), 0., .25, 2.)
    records = []

    def part(obj, fixture, suffix, mat, ir, temperature):
        obj.name = fixture.name + '_' + suffix
        obj.data.name = obj.name + '_Mesh'
        # Primitive operators link into the active source collection; move only this new part.
        for old_collection in list(obj.users_collection):
            old_collection.objects.unlink(obj)
        collection.objects.link(obj)
        obj.data.materials.append(mat)
        attach(obj, fixture)
        obj['surface_id'] = 'LGT.' + fixture.name + '.' + suffix
        obj['component'] = suffix
        obj['ir_material'] = ir
        obj['temperature_target_id'] = obj.name.lower()
        obj['temperature_c'] = temperature
        obj['modeling_approximation'] = 'Designed fixture; dimensions are not source BIM measurements'
        return obj

    def box(fixture, suffix, position, dimensions, mat, ir, temperature, rotation=None):
        bpy.ops.mesh.primitive_cube_add(size=1., location=position)
        obj = bpy.context.object
        for v in obj.data.vertices:
            v.co = Vector((v.co.x*dimensions[0], v.co.y*dimensions[1], v.co.z*dimensions[2]))
        if rotation:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = rotation
            bpy.context.view_layer.update()
        mod = obj.modifiers.new('Edge_Radius', 'BEVEL')
        mod.width = min(.012, min(dimensions)*.18)
        mod.segments = 2
        return part(obj, fixture, suffix, mat, ir, temperature)

    def cylinder(fixture, suffix, position, radius, depth, mat, ir, temperature):
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=depth, location=position)
        obj = bpy.context.object
        for p in obj.data.polygons:
            p.use_smooth = len(p.vertices) == 4
        return part(obj, fixture, suffix, mat, ir, temperature)

    def luminaire(fixture, origin, target, watts, size, temperature_k):
        direction = (Vector(target)-Vector(origin)).normalized()
        rotation = direction.to_track_quat('-Z', 'Y')
        box(fixture, 'Housing', Vector(origin)-direction*.075,
            (size[0]+.1,size[1]+.08,.13), steel,
            'aircraft_aluminium_painted', 34., rotation)
        box(fixture, 'Diffuser', Vector(origin)+direction*.012,
            (size[0],size[1],.018), lens, 'glass_windshield', 42., rotation)
        data = bpy.data.lights.new(fixture.name+'_Emitter', 'AREA')
        data.energy = watts
        data.normalize = True
        data.shape = 'RECTANGLE'
        data.size, data.size_y = size
        data.spread = math.radians(115)
        data.use_temperature = True
        data.temperature = temperature_k
        data.use_shadow = True
        light = bpy.data.objects.new(fixture.name+'_Light', data)
        collection.objects.link(light)
        # Keep the analytic emitter just outside the opaque visible diffuser.
        light.location = Vector(origin) + direction*.03
        light.rotation_mode = 'QUATERNION'
        light.rotation_quaternion = rotation
        bpy.context.view_layer.update()
        attach(light, fixture)
        light['light_id'] = fixture.name
        records.append({'light_id':fixture.name,'blender_object':light.name,
            'type':'AREA','position_m':list(light.matrix_world.translation),'target_m':list(target),
            'blender_radiant_power_w':watts,'color_temperature_k':temperature_k,
            'size_m':list(size),'spread_degrees':115.,'normalize':True,
            'approximation':'Designed industrial luminaire; power is a simulation preset, not a photometric calibration'})

    dg = bpy.context.evaluated_depsgraph_get()
    roads = [('ROAD_001',679.,-812.),('ROAD_002',700.,-836.),
             ('ROAD_003',656.,-787.),('ROAD_004',719.,-857.)]
    for name,x,y in roads+[('AREA_001',704.,-803.)]:
        hit,location,normal,face,ground,matrix = scene.ray_cast(
            dg, Vector((x,y,30)), Vector((0,0,-1)), distance=40)
        assert hit and abs(location.z)<.5, f'{name}: not clear ground: {ground}'
        z = float(location.z)
        fixture = empty(name, light_root, collection)
        fixture.location = (x,y,z)
        fixture['ground_z_m'] = z
        fixture['ground_source_object'] = ground.name
        fixture['fixture_role'] = 'Equipment apron perimeter' if name.startswith('AREA') else 'Paved-road edge'
        bpy.context.view_layer.update()
        height = 7.2 if name.startswith('AREA') else 6.5
        box(fixture,'Footing',(x,y,z+.12),(.42,.42,.24),concrete,'concrete',23.)
        box(fixture,'BasePlate',(x,y,z+.255),(.3,.3,.03),metal,'aluminium_anodised',23.)
        cylinder(fixture,'Pole',(x,y,z+height/2+.27),.085,height,metal,'aluminium_anodised',23.)
        box(fixture,'ServiceDoor',(x-.087,y,z+.85),(.025,.12,.25),steel,'aircraft_aluminium_painted',23.)
        inward = Vector((.743,.669,0))
        arm_center = Vector((x,y,z+height+.25))+inward*.52
        rotation = inward.to_track_quat('X','Z')
        box(fixture,'Arm',arm_center,(1.18,.075,.075),metal,'aluminium_anodised',24.,rotation)
        head = Vector((x,y,z+height+.22))+inward*1.08
        if name.startswith('AREA'):
            aim = (704.,-795.,.2)
            power = 420.
        else:
            aim = (x+inward.x*4.,y+inward.y*4.,z)
            power = 310.
        luminaire(fixture,head,aim,power,(.56,.23),4000.)
    for name,center,normal in [
        ('ENTRY_001',(721.857,-825.613,4.45),(.669,-.743,0)),
        ('ENTRY_002',(705.826,-829.652,4.45),(-.743,-.669,0))]:
        n = Vector(normal).normalized()
        fixture = empty(name,light_root,collection)
        fixture.location = Vector(center)+n*.04
        fixture['fixture_role'] = 'Building GC021 entrance wall-pack'
        fixture['mount_source_object'] = '11_29_gc021'
        bpy.context.view_layer.update()
        rotation = n.to_track_quat('Z','Y')
        box(fixture,'MountPlate',Vector(center)+n*.035,(.18,.27,.06),metal,
            'aluminium_anodised',24.,rotation)
        origin = Vector(center)+n*.2
        target = origin+n*1.9-Vector((0,0,4.1))
        luminaire(fixture,origin,target,65.,(.32,.18),3500.)
    moon_data = bpy.data.lights.new('MoonLight_Data','SUN')
    moon_data.energy = .018
    moon_data.angle = math.radians(.52)
    moon_data.color = (.56,.69,1.)
    moon = bpy.data.objects.new('MoonLight',moon_data)
    collection.objects.link(moon)
    moon.rotation_euler = (math.radians(35),math.radians(-20),math.radians(-35))
    attach(moon,light_root)
    world = bpy.data.worlds.new('Model02_NightWorld')
    world.use_nodes = True
    background_node = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
    background_node.inputs['Color'].default_value = (.25,.36,.6,1.)
    background_node.inputs['Strength'].default_value = .008
    scene.world = world
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.
    scene.view_settings.gamma = 1.
    scene['lighting_preset'] = 'low_light'
    scene['lighting_notes'] = 'Sparse industrial night lighting; IR temperatures configured independently'
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1440
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    cam = scene.camera
    cam.data.type = 'PERSP'
    cam.data.lens = 32.
    cam.data.sensor_width = 36.
    cam.location = (646.,-867.,44.)
    cam.rotation_euler = (Vector((700.,-809.,3.))-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.clip_end = 10000.
    bpy.context.view_layer.update()
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            area.spaces.active.shading.type = 'MATERIAL'
            area.spaces.active.shading.use_scene_world = True
            area.spaces.active.shading.use_scene_lights = True
            area.tag_redraw()
    (ROOT/'lights.json').write_text(json.dumps({'preset':'low_light','lights':records,
        'world_strength':.008,'moon_energy':.018,'exposure':0.,
        'fixture_geometry':'Editable Mesh parts, actual area emitters, grounded footings and wall mounts',
        'remaining_validation':'Blender actual render then Isaac USD light/exposure verification'},
        ensure_ascii=False,indent=2),encoding='utf-8')
    return {'fixtures':len(records),'mesh_parts':sum(o.type=='MESH' for o in collection.objects),
            'lights':sum(o.type=='LIGHT' for o in collection.objects),'preset':'low_light'}


if __name__ == '__main__':
    result = build_lighting()
