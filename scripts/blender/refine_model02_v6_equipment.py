"""Preserve measured equipment, connections and supports; separate thermal parts."""
import bpy,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.model02_full_factory_authoring import inputs,source_records,create_surface,save_batch,OUT
from scripts.blender.model02_full_factory_common import partition_surface,partition_axial,surface_kind,material_texture
from scripts.blender.model02_geometry import polygon_area


def island_frames(records,reviewed_vessel=False):
    groups=defaultdict(list)
    for r in records:groups[(r['material'],r['island'])].append(r)
    out={}
    for key,items in groups.items():
        p=np.unique(np.round(np.concatenate([r['poly'] for r in items])[:,:3],6),axis=0);center=p.mean(0)
        _,vectors=np.linalg.eigh((p-center).T@(p-center));axis=vectors[:,-1]
        # Stable principal frame of the measured connected part, not a fitted cylinder.
        if axis[np.argmax(abs(axis))]<0:axis=-axis
        other=np.array([0.,0.,1.]) if abs(axis[2])<.9 else np.array([1.,0.,0.])
        x=np.cross(other,axis);x/=np.linalg.norm(x);y=np.cross(axis,x);basis=np.array([x,y,axis])
        span=np.ptp((p-center)@basis.T,axis=0)
        world_span=np.ptp(p,axis=0)
        body=bool(reviewed_vessel and min(world_span[:2])>1. and world_span[2]>.4*max(world_span[:2]))
        if body:basis=np.eye(3);span=world_span
        out[key]=(center,basis,span,body)
    return out


def author(source,registry):
    started=time.perf_counter();records,points,zero=source_records(source)
    reviewed_vessel=registry['source_object'] in ['11_29_gc003','11_29_gc005','11_29_gc067']+['11_29_gc'+str(i).zfill(3) for i in range(79,89)]
    frames=island_frames(records,reviewed_vessel)
    grouped=defaultdict(list);before=sum(polygon_area(r['poly']) for r in records)
    lower,upper=points[:,2].min(),points[:,2].max();role=registry['role']
    for r in records:
        p=np.asarray(r['poly']);n=np.cross(p[1,:3]-p[0,:3],p[2,:3]-p[0,:3]);n/=np.linalg.norm(n)
        kind,ir=surface_kind(material_texture(source.data.materials[r['material']]),role,n,p[:,2].mean(),lower,upper)
        center,basis,span,body=frames[(r['material'],r['island'])]
        axial=body or role in ['equipment','pipework'] and span[2]>max(span[0],span[1])*.8
        if kind=='EquipmentShell':
            kind='EquipmentEndFace' if abs(n@basis[2])>.8 else 'EquipmentShell'
        if body:kind='TankHead' if abs(n[2])>.8 else 'TankShell'
        if axial:
            pieces=partition_axial(r['poly'],basis,center,1.,4 if role=='pipework' else 8)
        else:pieces=partition_surface(r['poly'],np.eye(3),1. if role=='pipework' else 2.)
        for cell,poly in pieces:grouped[(r['material'],r['island'],kind,ir,cell)].append(dict(r,poly=poly))
    rows=[]
    for key,polys in sorted(grouped.items()):
        row=create_surface(source,registry,key,polys,key[2],key[3],key[-1],
            'Original measured connected part and source normals retained; joints/ports/supports are existing geometry. Optical material is an explicit uncalibrated proxy; reviewed vertical vessel bodies use world-Z1m bands and8 sectors about measured centroid; no circular geometry fit is added')
        if row['component'] in ['TankShell','TankHead']:assert row['bounds_m'][1][2]-row['bounds_m'][0][2]<=1.00001
        rows.append(row)
    after=sum(r['area_m2'] for r in rows);assert abs(before-after)<max(1e-5,before*1e-7),(source.name,before,after)
    source.hide_render=True;source.hide_set(True)
    return rows,{'source_part':source.name,'source_object':registry['source_object'],'source_faces':len(source.data.polygons),
        'source_area_m2':before,'authored_area_m2':after,'source_area_accounting_error_m2':before-after,
        'connected_material_parts':len(frames),'source_zero_area_triangles':zero,'new_surfaces':len(rows),
        'seconds':time.perf_counter()-started,'reviewed_vertical_vessel_bodies':sum(v[3] for v in frames.values()),
        'construction':'No duplicate attachment rings; measured connections preserved; pipe sectors centred on cardinal transverse directions'}


def main():
    zones=defaultdict(list)
    for source,registry in inputs({'equipment','pipework','structure'}):zones[registry['zone_id']].append((source,registry))
    for zone,items in sorted(zones.items()):
        label='equipment_'+zone
        if (OUT/'batches'/(label+'.json')).exists():continue
        rows=[];checks=[];replaced=[]
        for source,registry in items:
            print('[v6] start',registry['role'],source.name,flush=True)
            added,check=author(source,registry);rows.extend(added);checks.append(check);replaced.append(source.name)
            print('[v6]',registry['role'],source.name,len(added),'regions',flush=True)
        save_batch(label,rows,replaced,checks)


if __name__=='__main__':main()
