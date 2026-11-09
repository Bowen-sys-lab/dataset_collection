"""Read-only model checks before export and after each refinement step."""


def validate_refinement():
    import bpy
    problems = []
    fixtures = [o for o in bpy.context.scene.objects if o.type == 'LIGHT' and o.get('light_id')]
    if len(fixtures) != 13:
        problems.append(f'Expected 13 fixture emitters, found {len(fixtures)}')
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.get('surface_id')]
    glass = [o for o in mesh_objects if o.get('component') == 'WindowGlass']
    frames = [o for o in mesh_objects if o.get('component') == 'WindowFrame']
    if not glass or len(frames) != 8:
        problems.append(f'Need 8 geometric window groups with independent glass/frame; glass={len(glass)}, frames={len(frames)}')
    doors = [o for o in mesh_objects if o.get('component') == 'DoorFrame']
    if len(doors) != 7:
        problems.append(f'Need 7 geometric door frames, found {len(doors)}')
    hot = [o for o in mesh_objects if o.get('component') == 'TankShell' and o.get('region') == 'Hotspot']
    transition = [o for o in mesh_objects if o.get('component') == 'TankShell' and o.get('region','').startswith('Transition')]
    if len(hot) != 1 or len(transition) < 5:
        problems.append('Need a local tank hotspot core and at least five transition regions')
    for obj in mesh_objects:
        if obj.get('source_object') in ['11_29_gc021','11_29_gc075','11_29_gc076','11_29_gc077','11_29_gc078']:
            for name in ['surface_id','temperature_target_id','ir_material','temperature_c']:
                if obj.get(name) is None:
                    problems.append(obj.name+': missing '+name)
    ids = [o['surface_id'] for o in mesh_objects]
    if len(ids) != len(set(ids)):
        problems.append('Duplicate surface IDs')
    return {'result':'FAIL' if problems else 'PASS','problems':problems,
            'fixture_emitters':len(fixtures),'glass_meshes':len(glass),'window_frames':len(frames),
            'door_frames':len(doors),'hotspot_meshes':len(hot),'transition_meshes':len(transition)}
