"""Refine measured model02 openings and tank surfaces, preserving source lineage."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('D:/IsaacSim_Demo/factory/model02_thermal_v2')


def clip_polygon(polygon, axis, plane, *, keep_above):
    output = []
    for previous,current in zip(polygon[-1:]+polygon[:-1],polygon):
        a,b = previous[axis]-plane,current[axis]-plane
        inside_a = a >= -1e-10 if keep_above else a <= 1e-10
        inside_b = b >= -1e-10 if keep_above else b <= 1e-10
        if inside_a != inside_b:
            output.append(previous+(current-previous)*(-a/(b-a)))
        if inside_b:
            output.append(current.copy())
    clean = []
    for vertex in output:
        if not clean or np.linalg.norm(vertex[:3]-clean[-1][:3]) > 1e-9:
            clean.append(vertex)
    if len(clean)>1 and np.linalg.norm(clean[0][:3]-clean[-1][:3])<1e-9:
        clean.pop()
    if len(clean)<3 or polygon_area(clean)<1e-10:
        return []
    return clean


def polygon_area(polygon):
    xyz = np.asarray(polygon)[:,:3]
    return sum(np.linalg.norm(np.cross(xyz[i]-xyz[0],xyz[i+1]-xyz[0]))/2
               for i in range(1,len(xyz)-1))


def split_by_planes(polygon, axis, planes):
    pieces = [polygon]
    for plane in sorted(set(planes)):
        result = []
        for piece in pieces:
            values = [p[axis] for p in piece]
            if min(values)+1e-9 < plane < max(values)-1e-9:
                for above in (False,True):
                    cut = clip_polygon(piece,axis,plane,keep_above=above)
                    if cut:
                        result.append(cut)
            else:
                result.append(piece)
        pieces = result
    return pieces


def build_geometry():
    import bpy
    from mathutils import Matrix, Vector
    assert Path(bpy.data.filepath) == ROOT/'model02_thermal_v2.blend'
    assert bpy.data.objects.get('BLD_GC021') is None, 'Refinement already exists'
    collection = bpy.data.collections.new('Model02_RefinedParts')
    bpy.context.scene.collection.children.link(collection)
    archive = bpy.data.collections.new('Model02_SourceArchive')
    bpy.context.scene.collection.children.link(archive)
    archive.hide_render = True
    archive.hide_viewport = True
    surfaces = []
    parts = []
    area_checks = []
    preservation_checks = []
    sources = {name:bpy.data.objects[name] for name in
               ['11_29_gc021','11_29_gc075','11_29_gc076','11_29_gc077','11_29_gc078']}

    def empty(name,parent):
        obj = bpy.data.objects.new(name,None)
        collection.objects.link(obj)
        obj.parent = parent
        return obj

    def assign(obj,source,component,region,temperature,material,face_ids,reconstruction=None):
        obj['surface_id'] = obj.name
        obj['source_object'] = source.name
        obj['component'] = component
        obj['region'] = region
        obj['temperature_target_id'] = obj.name.lower()
        obj['temperature_c'] = float(temperature)
        obj['ir_material'] = material
        obj['modeling_approximation'] = ('Prescribed surface temperature; library IR material is a proxy, not measured source BIM')
        record = {key:obj[key] for key in ['surface_id','source_object','component','region',
                  'temperature_target_id','temperature_c','ir_material','modeling_approximation']}
        record.update(blender_object=obj.name,source_mesh=source.data.name,
                      source_face_ids=sorted(set(face_ids)),rgb_materials=[m.name for m in obj.data.materials if m],
                      source_rgb_material_slots=[m.name if m else None for m in source.data.materials])
        if reconstruction:
            record['reconstruction'] = reconstruction
        surfaces.append(record)

    def source_records(source,face_ids):
        mesh = source.data
        transform = np.asarray(source.matrix_world,dtype=float)
        uv = mesh.uv_layers.active.data if mesh.uv_layers.active else None
        records = []
        for face_id in face_ids:
            p = mesh.polygons[face_id]
            polygon = []
            for li in p.loop_indices:
                xyz = np.asarray(mesh.vertices[mesh.loops[li].vertex_index].co,dtype=float)
                world = (transform@np.r_[xyz,1.])[:3]
                tex = tuple(uv[li].uv) if uv else (0.,0.)
                polygon.append(np.r_[world,tex,tuple(mesh.corner_normals[li].vector)])
            records.append({'face':face_id,'poly':polygon,'material':p.material_index,'smooth':p.use_smooth,
                            'vertex_ids':list(p.vertices)})
        return records

    def mesh_from_records(name,parent,source,records,component,region,temp,ir):
        mesh = bpy.data.meshes.new(name+'_Mesh')
        inverse = np.linalg.inv(np.asarray(source.matrix_world,dtype=float))
        vertices = []
        faces = []
        lookup = {}
        texcoords = []
        normals = []
        for record in records:
            face = []
            for j,vertex in enumerate(record['poly']):
                key = ('source',record['vertex_ids'][j]) if record.get('vertex_ids') is not None else tuple(np.round(vertex[:3],9))
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append((inverse@np.r_[vertex[:3],1.])[:3])
                face.append(lookup[key])
                texcoords.append(vertex[3:5])
                n = vertex[5:8]
                normals.append(n/max(np.linalg.norm(n),1e-15))
            faces.append(face)
        mesh.from_pydata(vertices,[],faces)
        used_materials = sorted({record['material'] for record in records})
        material_map = {original:new for new,original in enumerate(used_materials)}
        for index in used_materials:
            mesh.materials.append(source.data.materials[index])
        for p,record in zip(mesh.polygons,records):
            p.material_index = material_map[record['material']]
            p.use_smooth = record['smooth']
        if source.data.uv_layers:
            uv = mesh.uv_layers.new(name=source.data.uv_layers.active.name)
            uv.data.foreach_set('uv',np.asarray(texcoords,dtype='float32').reshape(-1))
        mesh.update()
        mesh.normals_split_custom_set(normals)
        obj = bpy.data.objects.new(name,mesh)
        collection.objects.link(obj)
        obj.parent = parent
        bpy.context.view_layer.update()
        obj.matrix_world = source.matrix_world.copy()
        assign(obj,source,component,region,temp,ir,[r['face'] for r in records])
        # Compare actual retained corners after creating the independent mesh.
        original_records = [r for r in records if r.get('vertex_ids') is not None]
        if original_records:
            vertex_error = 0.
            uv_error = 0.
            normal_error = 0.
            world = np.asarray(obj.matrix_world,dtype=float)
            for p,record in zip(mesh.polygons,records):
                if record.get('vertex_ids') is None:
                    continue
                for li,expected in zip(p.loop_indices,record['poly']):
                    actual = (world@np.r_[tuple(mesh.vertices[mesh.loops[li].vertex_index].co),1.])[:3]
                    vertex_error = max(vertex_error,float(np.max(np.abs(actual-expected[:3]))))
                    if source.data.uv_layers:
                        uv_error = max(uv_error,float(np.max(np.abs(np.asarray(mesh.uv_layers.active.data[li].uv)-expected[3:5]))))
                    normal_error = max(normal_error,float(np.max(np.abs(np.asarray(mesh.corner_normals[li].vector)-expected[5:8]))))
            preservation_checks.append({'object':name,'world_coordinate_max_error_m':vertex_error,
                'uv_max_error':uv_error,'normal_component_max_error':normal_error})
        return obj

    def material(name,color,source_image=None,glass=False):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        node = next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        node.inputs['Base Color'].default_value = (*color,1.)
        node.inputs['Roughness'].default_value = .22 if glass else .5
        if glass:
            node.inputs['IOR'].default_value = 1.45
            # No invented interior: tinted, reflective glazing stays optically opaque here.
            node.inputs['Transmission Weight'].default_value = 0.
        if source_image:
            tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
            tex.image = source_image
            mat.node_tree.links.new(tex.outputs['Color'],node.inputs['Base Color'])
        return mat

    frame_mat = material('GC021_WindowFrame_Painted',(.43,.45,.42))
    source = sources['11_29_gc021']
    images = {slot:next(n.image for n in source.data.materials[slot].node_tree.nodes if n.type=='TEX_IMAGE' and n.image)
              for slot in [2,3,4]}
    for slot in [2,4]:
        original = images[slot]
        converted = bpy.data.images.load(str(ROOT/f'source_texture_{slot}.png'),check_existing=True)
        converted.colorspace_settings.name = original.colorspace_settings.name
        converted.pack()
        images[slot] = converted
    window_mat = material('GC021_WindowGlass',(.08,.12,.14),images[3],glass=True)
    red_panel_mat = material('GC021_DoorPaintedPanel_Red',(.2,.04,.03),images[2])
    red_glass_mat = material('GC021_DoorGlazing',(.08,.12,.1),images[2],glass=True)
    blue_panel_mat = material('GC021_DoorPaintedPanel_Blue',(.12,.2,.22),images[4])

    def box_geometry(bounds,n,u,plane,depth_back,depth_front):
        x0,x1,z0,z1 = bounds
        points = []
        for depth in [depth_back,depth_front]:
            for x,z in [(x0,z0),(x1,z0),(x1,z1),(x0,z1)]:
                points.append(u*x+n*(plane+depth)+np.array([0.,0.,z]))
        return points,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]

    def make_boxes(name,parent,source,boxes,n,u,plane,affine,mat,component,region,temp,ir,ids,depth):
        positions = []
        faces = []
        for bounds in boxes:
            vertices,new_faces = box_geometry(bounds,n,u,plane,*depth)
            offset = len(positions)
            positions.extend(vertices)
            faces.extend(tuple(offset+i for i in face) for face in new_faces)
        center = np.mean(positions,axis=0)
        mesh = bpy.data.meshes.new(name+'_Mesh')
        mesh.from_pydata(np.asarray(positions)-center,[],faces)
        mesh.materials.append(mat)
        uv = mesh.uv_layers.new(name='UVChannel_1')
        for loop in mesh.loops:
            p = positions[loop.vertex_index]
            uv.data[loop.index].uv = np.array([p@u,p[2],1.])@affine
        mesh.update()
        obj = bpy.data.objects.new(name,mesh)
        collection.objects.link(obj)
        obj.parent = parent
        bpy.context.view_layer.update()
        obj.matrix_world = Matrix.Translation(Vector(center))
        if component.endswith('Frame'):
            bevel = obj.modifiers.new('Frame_edge_radius','BEVEL')
            bevel.width = .003
            bevel.segments = 2
        assign(obj,source,component,region,temp,ir,ids,
               'Measured source opening rectangle and UV crop; frame/depth are explicit modeling approximations')
        return obj

    def merge_intervals(intervals,minimum,maximum):
        merged = []
        for lo,hi in sorted((max(minimum,a),min(maximum,b)) for a,b in intervals if b>minimum and a<maximum):
            if merged and lo<=merged[-1][1]+1e-7:
                merged[-1][1] = max(merged[-1][1],hi)
            else:
                merged.append([lo,hi])
        return merged

    def gaps(intervals,lo,hi):
        cursor = lo
        out = []
        for a,b in intervals:
            if a>cursor+1e-5:
                out.append((cursor,a))
            cursor = max(cursor,b)
        if cursor<hi-1e-5:
            out.append((cursor,hi))
        return out

    building = empty('BLD_GC021',bpy.data.objects['InspectionZone01'])
    openings = json.loads((ROOT/'opening_analysis.json').read_text(encoding='utf-8'))['opening_components']
    window_index = 0
    door_index = 0
    replaced = set()
    for opening in openings:
        ids = opening['faces']
        records = source_records(source,ids)
        all_vertices = np.asarray([v for r in records for v in r['poly']])
        n = np.asarray(opening['normal'],dtype=float)
        n[2] = 0.;n /= np.linalg.norm(n)
        u = np.array([-n[1],n[0],0.])
        xs = all_vertices[:,:3]@u
        zs = all_vertices[:,2]
        x0,x1,z0,z1 = min(xs),max(xs),min(zs),max(zs)
        plane = float(np.mean(all_vertices[:,:3]@n))
        affine = np.linalg.lstsq(np.c_[xs,zs,np.ones(len(xs))],all_vertices[:,3:5],rcond=None)[0]
        width,height = x1-x0,z1-z0
        area = sum(polygon_area(r['poly']) for r in records)
        assert abs(area-width*height)/area<.001, 'Opening is not a measured rectangle'
        if opening['mat']==3:
            window_index += 1
            prefix = f'GC021_Window_{window_index:03d}'
            parent = empty(prefix,building)
            # Window source photo: six columns/tile and a transom at V ~= 0.70.
            uv_min,uv_max = min(all_vertices[:,3]),max(all_vertices[:,3])
            a = affine[0,0]
            b = affine[1,0]*(z0+z1)/2+affine[2,0]
            frame_width = min(.045,max(.02,abs(1/a)*.012))
            vertical = [(x0,x0+frame_width),(x1-frame_width,x1)]
            for k in range(math.floor(uv_min*6),math.ceil(uv_max*6)+1):
                x = (k/6-b)/a
                vertical.append((x-frame_width/2,x+frame_width/2))
            vertical = merge_intervals(vertical,x0,x1)
            z_cross = (.70-affine[0,1]*(x0+x1)/2-affine[2,1])/affine[1,1]
            horizontal = merge_intervals([(z0,z0+.03),(z1-.03,z1),(z_cross-.018,z_cross+.018)],z0,z1)
            xg,zg = gaps(vertical,x0,x1),gaps(horizontal,z0,z1)
            frame_boxes = [(a,b,z0,z1) for a,b in vertical]
            frame_boxes += [(a,b,c,d) for a,b in xg for c,d in horizontal]
            make_boxes(prefix+'_Frame',parent,source,frame_boxes,n,u,plane,affine,frame_mat,
                'WindowFrame','Frame',24.,'aircraft_aluminium_painted',ids,(-.018,.065))
            index = 0
            for a,b in xg:
                for c,d in zg:
                    index += 1
                    make_boxes(prefix+f'_Glass_{index:03d}',parent,source,[(a,b,c,d)],n,u,plane,affine,
                        window_mat,'WindowGlass','Pane',22.5+(window_index%3)*.15,'glass_windshield',ids,(-.018,0.))
        else:
            door_index += 1
            prefix = f'GC021_Door_{door_index:03d}'
            parent = empty(prefix,building)
            red = opening['mat']==2
            # Door photographs provide the panel/glazing layout; positions remain source-derived.
            cuts_x = [0.,.07,.46,.54,.93,1.] if red else [0.,.055,.47,.53,.945,1.]
            cuts_z = [0.,.055,.43,.49,.94,1.] if red else [0.,.04,.80,.86,.96,1.]
            groups = defaultdict(list)
            for i,(a,b) in enumerate(zip(cuts_x,cuts_x[1:])):
                for j,(c,d) in enumerate(zip(cuts_z,cuts_z[1:])):
                    kind = 'Frame' if i in (0,2,4) or j in (0,4) else ('Glass' if red and j==3 else 'Panel')
                    groups[kind].append((x0+a*width,x0+b*width,z0+c*height,z0+d*height))
            for kind,boxes in groups.items():
                mat = red_glass_mat if kind=='Glass' else red_panel_mat if red else blue_panel_mat
                make_boxes(prefix+'_'+kind,parent,source,boxes,n,u,plane,affine,mat,'Door'+kind,kind,
                    23. if kind=='Glass' else 24.5,'glass_windshield' if kind=='Glass' else 'aircraft_aluminium_painted',
                    ids,(-.035,0.) if kind=='Glass' else (-.04,.055 if kind=='Frame' else .015))
        replaced.update(ids)
        parts.append({'source_object':source.name,'source_face_ids':ids,'action':'replace',
            'reconstruction':prefix+'; measured rectangle and source UV grid; tiled frame/panel/glass surfaces'})
    records = source_records(source,[p.index for p in source.data.polygons if p.index not in replaced])
    groups = defaultdict(list)
    for record in records:
        p = source.data.polygons[record['face']]
        n = (source.matrix_world.to_3x3().inverted().transposed()@p.normal).normalized()
        center = np.mean(record['poly'],axis=0)[:3]
        if p.material_index==0 or (n.z>.65 and center[2]>4.5):
            component,region,temp = 'Roof','RoofCover',25.5
        elif n.z<-.65:
            component,region,temp = 'Soffit','Underside',23.5
        elif n.z>.65:
            component,region,temp = 'StepsAndPlinth','Horizontal',24.5
        else:
            direction = max({'Front':Vector((.669,-.743,0)), 'Rear':Vector((-.669,.743,0)),
                             'West':Vector((-.743,-.669,0)),'East':Vector((.743,.669,0))},
                            key=lambda k:n.dot({'Front':Vector((.669,-.743,0)), 'Rear':Vector((-.669,.743,0)),
                             'West':Vector((-.743,-.669,0)),'East':Vector((.743,.669,0))}[k]))
            component,region,temp = 'Wall',direction,26. if direction=='Front' else 23.5 if direction=='Rear' else 24.5
        groups[(component,region,temp)].append(record)
    for (component,region,temp),group in groups.items():
        mesh_from_records('GC021_'+component+'_'+region,building,source,group,component,region,temp,'etics_render')
        parts.append({'source_object':source.name,'source_face_ids':[r['face'] for r in group],'action':'split'})

    equipment = empty('Equipment',bpy.data.objects['InspectionZone01'])
    hotspot = np.array([693.672056966,-789.8881293,2.83407541097])
    for source_name in ['11_29_gc075','11_29_gc076','11_29_gc077','11_29_gc078']:
        source = sources[source_name]
        tank_id = source_name[-5:].upper()
        parent = empty('TANK_'+tank_id,equipment)
        shell_parent = empty('Shell_'+tank_id,parent)
        groups = defaultdict(list)
        source_area = 0.
        output_area = 0.
        for record in source_records(source,range(len(source.data.polygons))):
            polygon = record['poly']
            source_area += polygon_area(polygon)
            p = source.data.polygons[record['face']]
            n = (source.matrix_world.to_3x3().inverted().transposed()@p.normal).normalized()
            center = np.mean(polygon,axis=0)[:3]
            if p.material_index==2 and abs(n.z)<.2 and max(v[2] for v in polygon)>1.:
                component = 'TankShell'
                front = source_name.endswith('075') and n.y<-.7 and center[1]<-789.
                pieces = split_by_planes(polygon,0,np.arange(hotspot[0]-1.4,hotspot[0]+1.401,.14)) if front else [polygon]
                if front:
                    pieces = [q for piece in pieces for q in split_by_planes(piece,2,np.arange(hotspot[2]-1.68,hotspot[2]+1.681,.14))]
                for piece in pieces:
                    c = np.mean(piece,axis=0)[:3]
                    q = ((c[0]-hotspot[0])/.6)**2+((c[2]-hotspot[2])/.75)**2 if front else 100.
                    bands = [(.35,'Hotspot',78.),(.55,'Transition_001',72.),(.8,'Transition_002',66.),
                        (1.1,'Transition_003',60.),(1.5,'Transition_004',53.),(2.,'Transition_005',46.),
                        (2.6,'Transition_006',40.),(3.3,'Transition_007',35.),(4.,'Transition_008',32.)]
                    selected = next(((region,temp) for limit,region,temp in bands if q<=limit),None)
                    if selected:
                        region,temp = selected
                    else:
                        region = 'Normal_'+('North' if n.y>.5 else 'South' if n.y<-.5 else 'East' if n.x>0 else 'West')
                        temp = 30.
                    new_record = dict(record,poly=piece)
                    if front:
                        new_record['vertex_ids'] = None
                    groups[(component,region,temp,'aircraft_aluminium_painted')].append(new_record)
                    output_area += polygon_area(piece)
            else:
                if p.material_index in (5,6):
                    component,region,temp,ir = 'Foundation','Base',26.,'concrete'
                elif p.material_index==2:
                    component,region,temp,ir = ('TankRoof','Cap',29.,'aircraft_aluminium_painted') if n.z>.2 else ('BaseRing','Skirt',28.,'aircraft_aluminium_painted')
                elif p.material_index==3:
                    component,region,temp,ir = 'RailingAndSupports','PaintedBlue',27.5,'aircraft_aluminium_painted'
                elif p.material_index==4:
                    component,region,temp,ir = 'ExternalPipework','PaintedRed',30.8,'aircraft_aluminium_painted'
                else:
                    component,region,temp,ir = 'AccessAttachments',f'Material_{p.material_index:02d}',27.,'aircraft_aluminium_painted'
                groups[(component,region,temp,ir)].append(record)
                output_area += polygon_area(polygon)
        for (component,region,temp,ir),records in groups.items():
            mesh_from_records(tank_id+'_'+component+'_'+region,shell_parent if component=='TankShell' else parent,
                              source,records,component,region,temp,ir)
        assert abs(source_area-output_area)/source_area<1e-7
        area_checks.append({'source':source_name,'source_area_m2':source_area,'output_area_m2':output_area,
                            'relative_error':abs(source_area-output_area)/source_area})
        parts.append({'source_object':source.name,'source_face_ids':list(range(len(source.data.polygons))),
            'action':'split','evidence':'Original facets retained; local Cartesian subdivisions on existing source triangles, UV/normals interpolated'})

    for source in sources.values():
        for old_collection in list(source.users_collection):
            old_collection.objects.unlink(source)
        archive.objects.link(source)
        source.hide_render = True
        source.hide_set(True)
    snapshot = json.loads((ROOT/'source_snapshot.json').read_text(encoding='utf-8'))
    for obj in snapshot['objects']:
        if obj['name'] not in sources:
            parts.append({'source_object':obj['name'],'source_face_ids':list(range(obj['faces'])),'action':'keep'})
    (ROOT/'partitions.json').write_text(json.dumps({'parts':parts},ensure_ascii=False),encoding='utf-8')
    (ROOT/'surfaces.json').write_text(json.dumps({'surfaces':surfaces},ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'geometry_preservation.json').write_text(json.dumps({'tank_area':area_checks,'retained_mesh_checks':preservation_checks},indent=2),encoding='utf-8')
    (ROOT/'backlog.json').write_text(json.dumps({'preserved_background':[o for o in snapshot['objects'] if o['name'] not in sources],
        'remaining':'Other route-side pipework/equipment beyond the four tanks and GC021 retains original geometry; refine by source evidence in later passes'},ensure_ascii=False,indent=2),encoding='utf-8')
    bpy.context.view_layer.update()
    return {'refined_sources':list(sources),'surfaces':len(surfaces),'windows':window_index,'doors':door_index,
            'tank_area_checks':area_checks,'max_retained_coordinate_error_m':max(c['world_coordinate_max_error_m'] for c in preservation_checks),
            'max_retained_uv_error':max(c['uv_max_error'] for c in preservation_checks),
            'max_retained_normal_error':max(c['normal_component_max_error'] for c in preservation_checks)}
