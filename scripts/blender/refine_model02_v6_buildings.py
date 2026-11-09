"""Full-site measured openings and continuously partitioned building components."""
import bpy,json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.model02_full_factory_authoring import inputs,source_records,create_surface,save_batch,OUT
from scripts.blender.model02_full_factory_common import partition_surface,surface_kind,material_texture,split_by_planes
from scripts.blender.model02_full_factory_openings import author_openings
from scripts.blender.model02_geometry import polygon_area
from scripts.blender.model02_route_refinement import BUILDING_BASIS


def author(source,registry):
    started=time.perf_counter();records,points,zero=source_records(source)
    source_area=sum(polygon_area(r['poly']) for r in records)
    rows,opaque,opening_report=author_openings(source,registry,records)
    grouped=defaultdict(list);lower,upper=points[:,2].min(),points[:,2].max()
    expanded=[]
    for r in opaque:
        if r.get('existing_glazed_door'):
            pieces=[r['poly']]
            for axis,planes in [(3,[.1,.46,.54,.9]),(4,[.42,.93])]:pieces=[q for p in pieces for q in split_by_planes(p,axis,planes)]
            for p in pieces:
                center=np.asarray(p).mean(0);glazing=.1<center[3]<.9 and not .46<center[3]<.54 and .42<center[4]<.93
                expanded.append(dict(r,poly=p,component_override='DoorGlass' if glazing else 'DoorPanel',material_override='glass_windshield' if glazing else 'industrial_painted_steel'))
        else:expanded.append(r)
    for r in expanded:
        p=np.asarray(r['poly']);n=np.cross(p[1,:3]-p[0,:3],p[2,:3]-p[0,:3]);n/=np.linalg.norm(n)
        kind,ir=surface_kind(material_texture(source.data.materials[r['material']]),'building',n,p[:,2].mean(),lower,upper)
        kind,ir=r.get('component_override',kind),r.get('material_override',ir)
        for cell,poly in partition_surface(r['poly'],BUILDING_BASIS,2.):
            grouped[(r['material'],r['island'],kind,ir,cell)].append(dict(r,poly=poly))
    for key,polys in sorted(grouped.items()):
        rows.append(create_surface(source,registry,('source',key),polys,key[2],key[3],key[-1],
            'Measured source geometry/UV preserved with continuous thermal cuts; material appearance identifies an optical proxy, not surveyed substrate or alloy'))
    remainder_area=sum(polygon_area(r['poly']) for r in opaque)
    authored_opaque_area=sum(polygon_area(r['poly']) for polys in grouped.values() for r in polys)
    clipping_loss=remainder_area-authored_opaque_area
    assert abs(clipping_loss)<max(.0001,source_area*1e-7),(source.name,clipping_loss)
    constructed=sum(o.get('source_area_m2',0.) for o in opening_report['openings'])
    error=source_area-remainder_area-constructed-opening_report['opaque_carrier_area_cut_m2']
    assert abs(error)<max(.0001,source_area*1e-7),(source.name,error)
    source.hide_render=True;source.hide_set(True)
    check={'source_part':source.name,'source_object':registry['source_object'],'source_faces':len(source.data.polygons),
        'source_area_m2':source_area,'retained_external_area_m2':remainder_area,'opening_source_area_m2':constructed,
        'source_area_accounting_error_m2':error,'submicron_clipping_area_loss_m2':clipping_loss,'source_zero_area_triangles':zero,'openings':opening_report,
        'new_surfaces':len(rows),'seconds':time.perf_counter()-started}
    return rows,check


def main():
    selected=inputs({'building'});zones=defaultdict(list)
    for o,r in selected:zones[r['zone_id']].append((o,r))
    for zone,items in sorted(zones.items()):
        label='buildings_'+zone
        if (OUT/'batches'/(label+'.json')).exists():continue
        rows=[];checks=[];replaced=[]
        for source,registry in items:
            added,check=author(source,registry);rows.extend(added);checks.append(check);replaced.append(source.name)
            print('[v6] building',source.name,len(added),'regions',len(check['openings']['openings']),'opening groups',flush=True)
        save_batch(label,rows,replaced,checks)


if __name__=='__main__':main()
