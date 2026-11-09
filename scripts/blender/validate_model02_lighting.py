"""Read-only checks for the actual Blender weak-light asset, not a mock scene."""
import math


EXPECTED_FIXTURES = ('ROAD_001', 'ROAD_002', 'ROAD_003', 'ROAD_004',
                     'AREA_001', 'ENTRY_001', 'ENTRY_002')


def validate_lighting():
    import bpy
    from mathutils import Vector
    problems = []
    fixtures = []
    scene = bpy.context.scene
    for fixture_id in EXPECTED_FIXTURES:
        root = bpy.data.objects.get(fixture_id)
        if root is None:
            problems.append(f'Missing fixture {fixture_id}')
            continue
        children = list(root.children_recursive)
        lights = [obj for obj in children if obj.type == 'LIGHT']
        meshes = [obj for obj in children if obj.type == 'MESH']
        if len(lights) != 1 or not meshes:
            problems.append(f'{fixture_id}: needs one light and physical fixture parts')
        for obj in lights:
            if obj.data.energy <= 0 or not math.isfinite(obj.data.energy):
                problems.append(f'{fixture_id}: non-positive/non-finite power')
            if not obj.data.use_shadow:
                problems.append(f'{fixture_id}: shadows disabled')
            for suffix in ('Housing', 'Diffuser'):
                part = bpy.data.objects.get(fixture_id + '_' + suffix)
                if part and abs(part.matrix_world.to_quaternion().dot(
                        obj.matrix_world.to_quaternion())) < .99999:
                    problems.append(f'{fixture_id}: {suffix} is not aligned with the emitter')
        for obj in meshes:
            if not obj.get('surface_id') or not obj.get('ir_material'):
                problems.append(f'{obj.name}: no independent thermal identity')
        if fixture_id.startswith(('ROAD', 'AREA')):
            ground_z = root.get('ground_z_m')
            if ground_z is None or abs(float(root.location.z) - ground_z) > .02:
                problems.append(f'{fixture_id}: no verified ground mount')
        fixtures.append({'id': fixture_id, 'meshes': len(meshes), 'lights': len(lights),
                         'position': list(root.matrix_world.translation)})
    ids = [obj.get('surface_id') for obj in scene.objects if obj.get('surface_id')]
    if len(ids) != len(set(ids)):
        problems.append('Duplicate thermal surface IDs')
    if not scene.world or not scene.world.use_nodes:
        problems.append('No explicit world lighting')
    elif next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND').inputs['Strength'].default_value > .02:
        problems.append('World fill too bright for this weak-light preset')
    moon = bpy.data.objects.get('MoonLight')
    if not moon or moon.type != 'LIGHT' or moon.data.energy > .02:
        problems.append('Weak moon light missing or excessive')
    if bpy.data.objects.get('Model02_DiagnosticSun'):
        problems.append('Daylight diagnostic sun remains in weak-light asset')
    return {'result': 'PASS' if not problems else 'FAIL', 'problems': problems,
            'fixtures': fixtures, 'fixture_count': len(fixtures),
            'blender_filepath': bpy.data.filepath}
