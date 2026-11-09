"""Whole-site shallow ground construction and distinct vegetation surfaces."""
import bpy,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
import json
TEXTURE_PARAMETERS=json.loads((ROOT/'factory/model02_thermal_v5/texture_parameters.json').read_text())
from scripts.blender.model02_full_factory_authoring import inputs,source_records,create_surface,save_batch,OUT,authored_world
from scripts.blender.model02_full_factory_common import partition_surface,surface_kind,material_texture,clean_mesh_polygon
from scripts.blender.model02_geometry import polygon_area
from scripts.blender.model02_route_refinement import BUILDING_BASIS
from scripts.model02_paving import partition
from scripts.model02_ground_relief import relief_pieces


def projected_area(polys):
    total=0.
    for r in polys:
        p=np.asarray(r['poly']).copy();p[:,2]=0.;total+=polygon_area(p)
    return total


def author(source,registry):
    started=time.perf_counter();records,points,zero=source_records(source);grouped=defaultdict(list)
    basis=BUILDING_BASIS;anchor=np.array([700.,-845.,0.]);rotation=np.asarray(authored_world(source))[:3,:3]
    background=registry['source_object']=='11_29_gc1750';lower,upper=points[:,2].min(),points[:,2].max()
    for r in records:
        p=np.asarray(r['poly']);n=np.cross(p[1,:3]-p[0,:3],p[2,:3]-p[0,:3]);n/=np.linalg.norm(n)
        texture=material_texture(source.data.materials[r['material']]);kind,ir=surface_kind(texture,'ground',n,p[:,2].mean(),lower,upper)
        if source.get('component') in ['PlatformDeck','RailAndSupport','RailAndAccess']:
            kind,ir='StructuralMetal',source.get('ir_material','industrial_painted_steel')
        if background:
            grouped[('Landscape','vegetation_leaf',r['material'],'source',(0,))].append(r);continue
        grass=kind=='Landscape'
        kerb=not grass and (registry['source_object']=='11_29_gc052' or source.get('component')=='PavingAndKerb' and not source.data.materials[r['material']].name.endswith('_39'))
        horizontal=abs(n[2])>.95
        paved=texture in ['11-29-gc019.jpg','map_127.jpg','map_1709.jpg'] and registry['source_object']!='11_29_gc051'
        road=source.get('component')=='PreservedRoadBackground' or (horizontal and not grass and not paved)
        road=road and kind not in ['MetalCladding','BareMetalSurface','StructuralMetal','WindowGlass','DoorPanel'] and texture!='11-29-gc012.jpg'
        if horizontal and not grass and (paved or road):
            local=p.copy();local[:,:3]=(local[:,:3]-anchor)@basis.T
            for key,piece in partition(list(local),.6 if paved else 4.,.008 if paved else .01):
                mode='paving' if paved else 'road';joint=key[2]=='joint'
                polys=[np.asarray(piece).copy()] if joint else relief_pieces(piece,mode)
                if joint:
                    for q in polys:q[:,2]-=.002 if paved else .001
                for q in polys:
                    clean=clean_mesh_polygon(list(q))
                    if not clean:continue
                    q=np.asarray(clean).copy()
                    rgb_kind='joint' if joint else 'brick' if paved else 'concrete'
                    q[:,3:5]=q[:,:2]/TEXTURE_PARAMETERS[rgb_kind]['texture_span_m']
                    q[:,:3]=q[:,:3]@basis+anchor
                    if n[2]<0:q=q[::-1].copy()
                    world_n=np.cross(q[1,:3]-q[0,:3],q[2,:3]-q[0,:3]);world_n/=max(np.linalg.norm(world_n),1e-12)
                    local_n=rotation.T@world_n;local_n/=max(np.linalg.norm(local_n),1e-12);q[:,5:8]=local_n
                    component=('PavingJoint' if joint else 'PavingBrick') if paved else ('RoadJoint' if joint else 'ConcreteRoadSlab')
                    if kerb:component='KerbJoint' if joint else 'KerbAndRiser'
                    material='paving_cement_joint' if joint else 'paving_concrete' if paved else 'concrete'
                    grouped[(component,material,0,'layout',key)].append(dict(r,poly=list(q),material=0,smooth=False))
        else:
            for key,piece in partition_surface(r['poly'],basis,8. if grass else 2.):
                q=np.asarray(piece).copy()
                if grass:q[:,3:5]=(q[:,:3]-anchor)@basis.T[:,:2]/TEXTURE_PARAMETERS['grass']['texture_span_m']
                component='KerbAndRiser' if kerb else kind
                grouped[(component,ir,r['material'],'source',key)].append(dict(r,poly=list(q)))
    rows=[];print('[v6] ground partition complete',source.name,len(grouped),'regions; building meshes',flush=True)
    for key,polys in sorted(grouped.items()):
        if rows and len(rows)%10000==0:print('[v6] ground meshes',source.name,len(rows),'/',len(grouped),flush=True)
        c,ir,slot,layout,cell=key
        if layout=='layout':
            name='V5_joint' if c in ['PavingJoint','RoadJoint','KerbJoint'] else 'V5_brick' if c=='PavingBrick' else 'V5_concrete'
            mats=[bpy.data.materials[name]]
        elif c=='Landscape' and not background:
            mats=[bpy.data.materials['V5_grass']];polys=[dict(r,material=0) for r in polys]
        else:mats=None
        assumption='Distant landscape outside factory: intentional low-detail optical/temperature proxy, not refined factory ground' if background else 'Measured ground boundary/elevation retained;600mm pavers/8mm filled joints or4m road slabs/10mm control joints;2mm/1mm recess and small edge chamfers are explicit unsurveyed construction assumptions. Grass uses soft normal detail, not added grain.'
        rows.append(create_surface(source,registry,key,polys,c,ir,cell,assumption,mats))
    before=projected_area(records);after=projected_area([r for p in grouped.values() for r in p])
    assert abs(before-after)<max(.0001,before*1e-7),(source.name,before,after)
    source.hide_render=True;source.hide_set(True)
    return rows,{'source_part':source.name,'source_object':registry['source_object'],'source_faces':len(source.data.polygons),
        'projected_area_before_m2':before,'projected_area_after_m2':after,'source_area_accounting_error_m2':before-after,
        'source_zero_area_triangles':zero,'new_surfaces':len(rows),'intentional_distant_background_proxy':background,'seconds':time.perf_counter()-started}


def main():
    zones=defaultdict(list)
    for source,registry in inputs({'ground'}):zones[registry['zone_id']].append((source,registry))
    for zone,items in sorted(zones.items()):
        label='ground_'+zone
        if (OUT/'batches'/(label+'.json')).exists():continue
        rows=[];checks=[];replaced=[]
        for source,registry in items:
            added,check=author(source,registry);rows.extend(added);checks.append(check);replaced.append(source.name)
            print('[v6] ground',source.name,len(added),'regions',flush=True)
        save_batch(label,rows,replaced,checks)


if __name__=='__main__':main()
