"""Validate/package the Blender-first v3; deliberately no Isaac migration."""
import bpy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from collections import Counter,defaultdict
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT=Path('D:/IsaacSim_Demo')
BUILD=ROOT/'factory/model02_thermal_v3_build'
OUT=ROOT/'factory/model02_thermal_v3'
sys.path.insert(0,str(ROOT))
from scripts.blender.model02_route_refinement import scene_parts,ROUTE,BUILDING_BASIS,authored_world
from scripts.blender.model02_inventory import snapshot_scene


def world_vertices(obj):
    matrix=np.asarray(obj.matrix_world,dtype=float)
    local=np.asarray([v.co[:] for v in obj.data.vertices])
    return (np.c_[local,np.ones(len(local))]@matrix.T)[:,:3]


def area(obj):
    vs=world_vertices(obj);obj.data.calc_loop_triangles()
    idx=np.asarray([t.vertices[:] for t in obj.data.loop_triangles])
    p=vs[idx]
    return float(np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1).sum()/2)


def thermal_basis(o):
    component=o['component']
    if o.get('region','').startswith(('Hotspot','Transition')):
        return '故障注入条件；核心和过渡独立规定温度，非现场测量或传热求解'
    if component in ['InsulationCladding','InsulationLayer','PipeWall']:
        return '设计热介质80C、环境20C、保温30mm；正常外护约25.5C，破损露管72C，作为可解释静态工况估计'
    if component in ['TankShell','VesselShell','TankRoof','BaseRing','MetalCladding','ExternalPipework']:
        return '持续运行设备/介质温升、湿壁与顶盖差异及外部冷却的静态估计；没有实测工艺资料'
    if component in ['Wall','Roof','Soffit','WindowGlass','WindowFrame','DoorGlass','DoorPanel','DoorFrame']:
        return '入夜约2小时、外气20C、室内24C；围护结构余热和屋顶辐射散热的静态估计'
    if component in ['Diffuser','Housing']:
        return '运行中灯具外壳/透光罩温升假设；RGB功率和LWIR温度独立'
    return '干燥弱风夜间外气20C，非加热构件近环境；铺装/基础保留少量余热，非实测'


def configure_and_inventory():
    rows=[];parts=scene_parts()
    for o in parts:
        component=o['component'];ir=o['ir_material']
        if ir=='aircraft_aluminium_painted':ir='industrial_painted_steel'
        if component in ['WindowFrame','Housing']:ir='painted_aluminium'
        if component in ['DoorFrame','DoorPanel']:ir='industrial_painted_steel'
        if component in ['WindowGlass','DoorGlass','Diffuser']:ir='glass_windshield'
        o['ir_material']=ir;o['physical_material']=ir
        if component in ['WindowGlass','DoorGlass']:o['temperature_c']=21.
        elif component=='WindowFrame':o['temperature_c']=20.5
        elif component in ['DoorFrame','DoorPanel']:o['temperature_c']=21.5
        elif component in ['RailAndAccess','RailAndSupport','RailingAndSupports','AccessAttachments','AccessAttachment','Pole','Arm','ServiceDoor','MountPlate']:o['temperature_c']=20.5
        elif component in ['Footing','BasePlate','Foundation']:o['temperature_c']=22.
        o['temperature_basis']=thermal_basis(o)
        # Explicit region and temperature identity even for retained v2 panes/fixtures.
        o['thermal_scope']=o.get('thermal_scope','retained_refined_v2')
        if not o.get('region'):o['region']='Normal'
        c=world_vertices(o).mean(axis=0)
        path='/'+ '/'.join(reversed([o.name]+[p.name for p in parents(o)]))
        rows.append({'surface_id':o['surface_id'],'blender_object':o.name,'hierarchy':path,
            'component':component,'region':o['region'],'temperature_target_id':o['temperature_target_id'],
            'temperature_c':float(o['temperature_c']),'ir_material':ir,'physical_material':ir,
            'temperature_basis':o['temperature_basis'],'thermal_scope':o['thermal_scope'],
            'source_object':o.get('source_object','DesignedLightFixture'),
            'source_part':o.get('source_part',''),'source_face_ids':list(o.get('source_face_ids',[])),
            'rgb_materials':[m.name for m in o.data.materials if m],
            'center_m':c.tolist(),'faces':len(o.data.polygons)})
    ids=[r['surface_id'] for r in rows];targets=[r['temperature_target_id'] for r in rows]
    assert len(ids)==len(set(ids)) and len(targets)==len(set(targets))
    (OUT/'surfaces.json').write_text(json.dumps({'status':'Blender authored; USD paths/IR validation deferred until migration','surfaces':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['surface_id','blender_object','hierarchy','component','region','temperature_target_id','temperature_c','ir_material','source_object','thermal_scope','temperature_basis']
    with (OUT/'surface_temperatures.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    with (OUT/'thermal_targets_template.yaml').open('w',encoding='utf-8') as f:
        f.write('# Blender-first target template; bindings must be generated from actual USD after migration.\n')
        f.write('targets:\n')
        for r in rows:f.write(f"  {r['temperature_target_id']}: {{mode: fixed, temperature_c: {r['temperature_c']:.6f}}}\n")
    return rows


def parents(o):
    result=[];p=o.parent
    while p:result.append(p);p=p.parent
    return result


def verify_geometry(rows):
    state=json.loads((BUILD/'construction_state.json').read_text())
    selected={r['source'] for r in state['processed']};actual=defaultdict(float)
    for o in scene_parts():
        p=o
        while p.get('source_part'):
            name=p['source_part']
            if name in selected:
                actual[name]+=area(o);break
            p=bpy.data.objects[name]
    checks=[]
    for r in state['area_checks']:
        relative=abs(actual[r['source']]-r['source_m2'])/r['source_m2']
        checks.append(dict(r,actual_mesh_area_m2=actual[r['source']],actual_relative_error=relative))
        assert relative<3e-6,(r['source'],relative)
    before=json.loads((OUT/'source_snapshot_serialized_baseline.json').read_text(encoding='utf-8'))
    after=snapshot_scene()
    for record in after['objects']:record['matrix_world']=[list(row) for row in authored_world(bpy.data.objects[record['name']])]
    b={o['name']:o for o in before['objects']};a={o['name']:o for o in after['objects']}
    differences=[n for n in b if b[n]['geometry_uv_material_normal_sha256']!=a[n]['geometry_uv_material_normal_sha256'] or b[n]['matrix_world']!=a[n]['matrix_world']]
    assert not differences,differences
    mesh_problems=[]
    for o in scene_parts():
        if o.data.validate(clean_customdata=False):mesh_problems.append(o.name)
        assert np.isfinite(world_vertices(o)).all()
    assert not mesh_problems,mesh_problems[:10]
    wall_spans=[];tank_heights=[];pipe_lengths=[]
    for o in scene_parts():
        if not o.get('source_part'):continue
        v=world_vertices(o)
        if o['component'] in ['Wall','Roof']:
            span=np.ptp(v@BUILDING_BASIS.T,axis=0);wall_spans.append(float(span.max()))
        if o['component']=='TankShell' and str(o['region']).startswith('Normal'):
            tank_heights.append(float(np.ptp(v[:,2])))
        if o['component']=='ExternalPipework':pipe_lengths.append(float(np.linalg.norm(np.ptp(v,axis=0))))
    assert max(wall_spans)<=2.0001,max(wall_spans)
    assert max(tank_heights)<=1.0001,max(tank_heights)
    assert max(pipe_lengths)<=1.0001,max(pipe_lengths)
    report={'result':'PASS','source_meshes_unchanged':len(b),'source_faces':sum(o['faces'] for o in before['objects']),
        'source_fingerprint_differences':differences,'area_checks':checks,'invalid_meshes':mesh_problems,
        'max_wall_roof_zone_axis_span_m':max(wall_spans),'max_tank_normal_zone_height_m':max(tank_heights),
        'max_original_pipe_zone_bbox_diagonal_m':max(pipe_lengths),
        'visible_thermal_surfaces':len(rows),'components':dict(Counter(r['component'] for r in rows))}
    (OUT/'geometry_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def verify_independence(rows):
    report=[]
    initial={r['surface_id']:r['temperature_c'] for r in rows}
    for component in ['Wall','TankShell','PipeWall']:
        chosen=next(r for r in rows if r['component']==component and not r['region'].startswith(('Hotspot','Transition')))
        o=bpy.data.objects[chosen['blender_object']];old=o['temperature_c'];o['temperature_c']=float(old)+10.
        changed=[x['surface_id'] for x in scene_parts() if float(x['temperature_c'])!=initial[x['surface_id']]]
        assert changed==[chosen['surface_id']],changed
        o['temperature_c']=old
        report.append({'component':component,'changed_surface_only':changed[0],'delta_c':10.,'restored':True})
    (OUT/'temperature_independence.json').write_text(json.dumps({'result':'PASS_BLENDER_PROPERTIES','checks':report,'limitation':'This verifies independent authored configuration, not rendered LWIR response. Isaac validation is a later phase.'},ensure_ascii=False,indent=2),encoding='utf-8')


def check_routes():
    verts=[];tris=[];graph=bpy.context.evaluated_depsgraph_get();objects=0
    for o in bpy.context.scene.objects:
        if o.type!='MESH' or o.hide_render or any(c.hide_render for c in o.users_collection):continue
        evaluated=o.evaluated_get(graph);m=evaluated.to_mesh();m.calc_loop_triangles();offset=len(verts)
        verts.extend(evaluated.matrix_world@v.co for v in m.vertices)
        tris.extend(tuple(offset+i for i in t.vertices) for t in m.loop_triangles)
        evaluated.to_mesh_clear();objects+=1
    tree=BVHTree.FromPolygons(verts,tris,all_triangles=True)
    routes={'tank':ROUTE,'pipe':ROUTE[:3]+[(688.,-804.3,3.5)]+ROUTE[4:]}
    reports={}
    for name,route in routes.items():
        samples=[]
        for a,b in zip(route,route[1:]):
            a,b=Vector(a),Vector(b);n=math.ceil((b-a).length/.25)
            samples.extend(a.lerp(b,i/n) for i in range(n))
        samples.append(Vector(route[-1]));d=[tree.find_nearest(p)[3] for p in samples]
        lower=min(d)-.2-.125-.001
        reports[name]={'waypoints':route,'minimum_sampled_center_distance_m':min(d),'continuous_body_clearance_lower_bound_m':lower,'closest_route_point':list(samples[d.index(min(d))]),'sample_count':len(samples)}
        eye=Vector(route[3]);points=[]
        for o in scene_parts():
            if o['component']!=('TankShell' if name=='tank' else 'PipeWall') or o['region']!='Hotspot':continue
            v=world_vertices(o);o.data.calc_loop_triangles()
            for tri in o.data.loop_triangles:
                points.append(Vector(v[list(tri.vertices)].mean(axis=0)))
        visible=[]
        for point in points:
            direction=point-eye;distance=direction.length
            hit=tree.ray_cast(eye,direction.normalized(),distance+.01)[0]
            if hit is not None and (hit-point).length<.002:visible.append((distance,list(point)))
        assert visible,'No visible hotspot facet from '+name+' observation point'
        reports[name]['hotspot_line_of_sight']={'tested_facets':len(points),'visible_facets':len(visible),'nearest_visible_distance_m':min(x[0] for x in visible),'example_visible_facet_center_m':visible[0][1]}
    (OUT/'route_clearance.json').write_text(json.dumps({'scope':'All render-visible Blender evaluated Mesh, conservative0.2m body, maxstep0.25m; not human flight or physics certification','objects':objects,'triangles':len(tris),'routes':reports},indent=2),encoding='utf-8')
    return reports


def main():
    rows=configure_and_inventory();print('V3 properties configured',len(rows),flush=True)
    validation=verify_geometry(rows);print('V3 geometry PASS',flush=True)
    verify_independence(rows);routes=check_routes();print('V3 routes',routes,flush=True)
    task=[]
    for name,component in [('tank','TankShell'),('pipe','PipeWall')]:
        hot=[r for r in rows if r['component']==component and r['region']=='Hotspot']
        assert hot
        task.append({'task_id':'model02_v3_'+name,'target_surface_ids':[r['surface_id'] for r in hot],
            'target_center_m':np.mean([r['center_m'] for r in hot],axis=0).tolist(),
            'reference_waypoints_m':routes[name]['waypoints'],'observation_position_m':routes[name]['waypoints'][3],
            'hold_s':6.,'preferred_observation_distance_m':[6.,8.],
            'instruction':'沿园区道路进入，确认蓝色护栏储罐外观，用红外检查罐壁异常，安全悬停后按T确认。' if name=='tank' else '沿园区道路进入，检查较高设备组靠道路侧立管的保温破损区域，用红外定位异常，安全悬停后按T确认。',
            'status':'Blender task definition; actual USD paths/Isaac entry pending migration','manual_success':None})
    (OUT/'inspection_tasks.json').write_text(json.dumps({'tasks':task},ensure_ascii=False,indent=2),encoding='utf-8')
    for filename in ['construction_state.json','pipe_construction.json']:
        shutil.copyfile(BUILD/filename,OUT/filename)
    bpy.context.scene['lighting_preset']='low_light_v3'
    bpy.context.scene['thermal_condition']='Dry low-wind night, air20C, indoor24C, operating equipment, prescribed surface snapshot'
    bpy.context.scene['thermal_pipeline_status']='Blender-first; no Isaac migration or IR validation yet'
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'model02_thermal_v3.blend'))
    report={'result':'BLENDER_GEOMETRY_AND_CONFIGURATION_VALIDATED','surfaces':len(rows),'fixture_lights':sum(o.type=='LIGHT' and bool(o.get('light_id')) for o in bpy.context.scene.objects),
            'geometry':{k:v for k,v in validation.items() if k not in ['area_checks','components']},'isaac_migration':'deferred per latest user instruction'}
    (OUT/'delivery_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
