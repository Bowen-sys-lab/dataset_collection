"""Blender-first physical paving partitions; additive v4 release, v3 unchanged."""
import bpy
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import defaultdict,Counter
import numpy as np

ROOT=Path('D:/IsaacSim_Demo');OUT=ROOT/'factory/model02_thermal_v4'
sys.path.insert(0,str(ROOT))
from scripts.blender.model02_route_refinement import records,new_mesh,scene_parts,BUILDING_BASIS,clip_rectangle,ROUTE
from scripts.blender.model02_route_zones import split_grid,cell_in_corridor
from scripts.blender.model02_geometry import polygon_area,clip_polygon
from scripts.model02_paving import partition,temperature


def main():
    source=ROOT/'factory/model02_thermal_v3/model02_thermal_v3_configured.blend'
    assert Path(bpy.data.filepath)==source
    destination=OUT/'model02_thermal_v4.blend'
    assert not destination.exists()
    OUT.mkdir(exist_ok=True)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    originals=json.loads((source.parent/'surfaces.json').read_text(encoding='utf-8'))['surfaces']
    by_id={r['surface_id']:r for r in originals}
    candidates=[o for o in scene_parts() if o.get('component')=='PavingAndKerb' and o.get('thermal_scope')=='route_refined']
    floor=bpy.data.objects['11_29_gc051']
    candidates.append(floor)
    groups=defaultdict(list);provenance=defaultdict(set);replaced=[];source_area=0.
    representative={}
    basis=BUILDING_BASIS.copy();anchor=np.array([700.,-845.,0.])
    for source_obj in candidates:
        source_id=source_obj.get('surface_id','SOURCE_GC051')
        if source_id!='SOURCE_GC051':replaced.append(source_id)
        source_records=records(source_obj)
        if source_id=='SOURCE_GC051':
            bounded=[]
            for r in source_records:
                inside,outside=clip_rectangle(r['poly'])
                for p in outside:bounded.append(dict(r,poly=p,covered_background=True))
                if inside:
                    for key,p in split_grid(inside,np.eye(3),[2.,2.,2.]):
                        bounded.append(dict(r,poly=p,covered_background=not cell_in_corridor(key,ROUTE)))
            source_records=bounded
        for r in source_records:
            poly=np.asarray(r['poly']);source_area+=polygon_area(poly)
            n=np.cross(poly[1,:3]-poly[0,:3],poly[2,:3]-poly[0,:3]);n/=max(np.linalg.norm(n),1e-15)
            mat=source_obj.data.materials[r['material']].name
            kind='road' if source_id=='SOURCE_GC051' or mat.endswith('_39') else 'kerb'
            if r.get('covered_background'):
                group=('background',0,0,'base',0)
                groups[group].append(dict(r,material=0));provenance[group].add(source_id);representative[group]=source_obj
                continue
            # Source audit: slot39 is the ground with reversed winding; slot40
            # is only a vertical riser, slot38 is kerb. Author pedestrian pavers
            # on actual ground at the building approach and tank-area approach.
            if abs(n[2])>.95 and kind=='road':
                local=poly.copy();local[:,:3]=(local[:,:3]-anchor)@basis.T
                remaining=[list(local)];pavers=[]
                for bounds in ([(16.,34.,-9.,-3.),(20.,32.,34.,42.)] if source_id=='SOURCE_GC051' else []):
                    next_remaining=[]
                    for candidate in remaining:
                        inside=candidate
                        for axis,value,above in [(0,bounds[0],True),(0,bounds[1],False),(1,bounds[2],True),(1,bounds[3],False)]:
                            if not inside:break
                            outside=clip_polygon(inside,axis,value,keep_above=not above)
                            if outside:next_remaining.append(outside)
                            inside=clip_polygon(inside,axis,value,keep_above=above)
                        if inside:pavers.append(inside)
                    remaining=next_remaining
                for surface_kind,polygons in [('paving',pavers),('road',remaining)]:
                    for subpoly in polygons:
                        for key,piece in partition(subpoly,tile_size=.6 if surface_kind=='paving' else 4.,joint=.008 if surface_kind=='paving' else .01):
                            piece=np.asarray(piece);piece[:,:3]=piece[:,:3]@basis+anchor
                            # Ground source winding is inverted, but its visible
                            # side is upward. Correct new top normals explicitly.
                            piece[:,5:8]=[0.,0.,1.]
                            if n[2]<0:piece=piece[::-1]
                            group=(surface_kind,*key,source_obj.get('source_object',source_obj.name))
                            groups[group].append(dict(r,poly=piece,material=0,smooth=False))
                            provenance[group].add(source_id);representative[group]=source_obj
            else:
                center=poly[:,:3].mean(0);group=('kerb',int(math.floor(center[0])),int(math.floor(center[1])),mat,0)
                groups[group].append(dict(r,poly=poly,material=0))
                provenance[group].add(source_id);representative[group]=source_obj
        source_obj.hide_render=True;source_obj.hide_set(True)
    materials={}
    for kind,color,rough in [('paving',(.36,.35,.32,1),.85),('road',(.24,.23,.21,1),.9),('joint',(.15,.145,.13,1),.95)]:
        m=bpy.data.materials.new('V4_'+kind);m.use_nodes=True
        bsdf=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bsdf.inputs['Base Color'].default_value=color;bsdf.inputs['Roughness'].default_value=rough
        materials[kind]=m
    checks=[];created=[]
    for count,(key,items) in enumerate(sorted(groups.items())):
        kind=key[0];part=key[3] if kind in ('paving','road') else 'kerb'
        name='PAV4_'+ '_'.join(str(k).replace('-','m') for k in key)
        src=representative[key]
        # new_mesh uses a single source-local frame. Supply a dedicated one-slot
        # source so materials never depend on source slot ordering.
        class Source:
            pass
        frame=Source();frame.name=src.name;frame.matrix_world=src.matrix_world.copy()
        frame.data=Source();frame.data.materials=[materials['joint' if part=='joint' else kind] if kind in ('paving','road') else src.data.materials[0]]
        source_name=src.get('source_object',src.name)
        frame.get=lambda k,d=None:source_name if k=='source_object' else d
        center=np.concatenate([np.asarray(r['poly'])[:,:3] for r in items]).mean(0)
        temp=temperature(*center[:2],'joint' if part=='joint' else 'tile') if kind in ('paving','road') else temperature(*center[:2],'joint')-.3 if kind=='kerb' else 22.
        component=('PavingJoint' if part=='joint' else 'PavingBrick') if kind=='paving' else ('RoadJoint' if part=='joint' else 'ConcreteRoadSlab') if kind=='road' else 'KerbAndRiser' if kind=='kerb' else 'PreservedRoadBackground'
        ir='paving_cement_joint' if part=='joint' else 'paving_concrete' if kind=='paving' else 'concrete'
        obj=new_mesh(name,frame,items,component,str(part),temp,ir)
        obj['replaces_surface_ids']=json.dumps(sorted(provenance[key]));obj['paving_layout']='600mm paver +8mm filled joint;4m concrete road slabs +10mm control joint; source dimensions unavailable'
        obj['temperature_basis']='Estimated static dry night20C,2h after sunset: broad residual-heat field; grout cools somewhat faster. No random noise, measurements or conduction solution.'
        obj['temperature_estimate_range_c']=json.dumps([21.5,22.] if part=='joint' else [21.7,22.3] if kind!='kerb' else [21.1,21.7])
        created.append(obj);checks.extend(polygon_area(np.asarray(r['poly'])) for r in items)
        if count%500==0:print('[v4] surfaces '+str(count)+'/'+str(len(groups)),flush=True)
    actual_area=sum(checks)
    assert abs(actual_area-source_area)<max(1e-6,source_area*1e-8)
    assert any(o['component']=='PavingBrick' for o in created)
    rows=[]
    for o in scene_parts():
        sid=o['surface_id']
        if sid in by_id:
            rows.append(by_id[sid]);continue
        local=np.array([v.co[:] for v in o.data.vertices]);matrix=np.array(o.matrix_world)
        points=(np.c_[local,np.ones(len(local))]@matrix.T)[:,:3]
        parents=[];parent=o.parent
        while parent:parents.append(parent.name);parent=parent.parent
        rows.append({'surface_id':sid,'blender_object':o.name,'hierarchy':'/'+ '/'.join(reversed([o.name]+parents)),
            'component':o['component'],'region':o['region'],'temperature_target_id':o['temperature_target_id'],
            'temperature_c':o['temperature_c'],'ir_material':o['ir_material'],'physical_material':o['physical_material'],
            'temperature_basis':o['temperature_basis'],'temperature_estimate_range_c':json.loads(o['temperature_estimate_range_c']),
            'thermal_scope':'route_refined_v4','source_object':o['source_object'],'source_part':o['source_part'],
            'replaces_surface_ids':json.loads(o['replaces_surface_ids']),'rgb_materials':[m.name for m in o.data.materials],
            'center_m':points.mean(0).tolist(),'bounds_m':[points.min(0).tolist(),points.max(0).tolist()],
            'faces':len(o.data.polygons)})
    ids=[r['surface_id'] for r in rows];assert len(ids)==len(set(ids))
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1.
    bpy.context.scene['thermal_configuration_phase']='v4 geometric authoring; apply configuration before export'
    bpy.context.scene['v4_replaced_ground_surfaces']=json.dumps(replaced)
    bpy.context.scene['v4_paving_geometry']='Shared original surface, filled joints with no gaps/overlaid faces; seam recess not invented without source dimensions'
    bpy.context.scene['v4_uav_start_m']=json.dumps([700,-845,8.5])
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    (OUT/'surfaces.json').write_text(json.dumps({'status':'v4 Blender authored','surfaces':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    report={'result':'PASS_BLENDER_PAVING_AREA','source_blend_sha256':source_hash,'blend_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'superseded_surface_ids':replaced,'new_surfaces':len(created),'total_surfaces':len(rows),'source_area_m2':source_area,'partition_area_m2':actual_area,
        'component_counts':dict(Counter(o['component'] for o in created)),
        'replaced_original_nodes':['/World/LimeExp_02/node_1_29_gc051'],
        'geometric_design_assumptions':'Actual exposed road is sourceGC051 atZ0.01m; sourceGC010 atZ0m is below landscape. Designed two pedestrian patches on GC051 in factory-local coordinates [16,34]x[-9,-3] and [20,32]x[34,42],600mm pavers/8mm filled joints. Other scoped road4m slabs/10mm control joints. Unrefined original road background retained at22C assumed continuity. All original elevations retained; grout fills seams without gaps or overlay faces. No source brick dimensions available.',
        'temperature_condition':'Dry20C outdoor,24C indoor,wind1m/s,2h after sunset,operating. Static estimates, not field calibrated.'}
    (OUT/'paving_geometry_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='superseded_surface_ids'}),flush=True)


if __name__=='__main__':main()
