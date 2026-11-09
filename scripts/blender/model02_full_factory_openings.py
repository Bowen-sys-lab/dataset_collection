"""Source-UV supported opening construction, with real frames and recessed panes."""
import bpy,bmesh,math
from collections import defaultdict
import numpy as np
from scripts.blender.model02_geometry import clip_polygon,polygon_area
from scripts.blender.model02_full_factory_common import material_texture,box_geometry,partition_surface,connected_opening_groups,opening_support_offset
from scripts.blender.model02_full_factory_authoring import create_surface,authored_world,OBJECT_INDEX


def merged(intervals,lo,hi):
    out=[]
    for a,b in sorted((max(lo,a),min(hi,b)) for a,b in intervals if b>lo and a<hi):
        if out and a<=out[-1][1]+1e-8:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out


def gaps(intervals,lo,hi):
    cursor=lo;out=[]
    for a,b in intervals:
        if a>cursor+1e-6:out.append((cursor,a))
        cursor=max(cursor,b)
    if hi>cursor+1e-6:out.append((cursor,hi))
    return out


def plain_material():
    m=bpy.data.materials.get('V6_OpeningFrame_PaintedMetal')
    if m:return m
    m=bpy.data.materials.new('V6_OpeningFrame_PaintedMetal');m.use_nodes=True
    node=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    node.inputs['Base Color'].default_value=(.25,.28,.26,1.);node.inputs['Roughness'].default_value=.45
    return m


def box_records(bounds,n,u,plane,back,front,source,affine,face_ids,bevel=0.):
    vertices,faces=box_geometry(bounds,n,u,plane,back,front);center=np.mean(vertices,axis=0)
    bm=bmesh.new();vs=[bm.verts.new(tuple(p-center)) for p in vertices]
    for indices in faces:bm.faces.new([vs[i] for i in indices])
    bm.normal_update()
    if bevel:bmesh.ops.bevel(bm,geom=list(bm.edges),offset=bevel,segments=2,affect='EDGES')
    bm.normal_update();items=[]
    rotation=np.asarray(authored_world(source))[:3,:3]
    for face in bm.faces:
        local_n=rotation.T@np.asarray(face.normal);local_n/=max(np.linalg.norm(local_n),1e-12)
        poly=[]
        for v in face.verts:
            p=np.asarray(v.co)+center;uv=np.array([p@u,p[2],1.])@affine
            poly.append(np.r_[p,uv,local_n])
        items.append({'poly':poly,'face':face_ids[0],'material':0,'island':face_ids[0],'smooth':bool(bevel)})
    bm.free();return items


def author_openings(source,registry,records):
    candidates=[];others=[];rows=[];cuts=[];reports=[]
    windows={'11-29-gc003.tif':(1,.62),'11-29-gc005.jpg':(6,.70),'11-29-gc033.jpg':(6,.65)}
    doors={'11-29-gc023.tif','11-29-gc034.tif','11-29-gc035.tif'}
    for r in records:
        texture=material_texture(source.data.materials[r['material']])
        if texture not in windows and texture not in doors:others.append(r);continue
        p=np.asarray(r['poly']);n=np.cross(p[1,:3]-p[0,:3],p[2,:3]-p[0,:3]);n/=np.linalg.norm(n)
        if abs(n[2])>.1:others.append(r);continue
        candidates.append(r)
    textures={i:material_texture(m) for i,m in enumerate(source.data.materials)}
    source_matrix=authored_world(source);source_center=np.mean(np.asarray([source_matrix@v.co for v in source.data.vertices]),axis=0)
    for number,items in enumerate(connected_opening_groups(candidates,textures)):
        key=(items[0]['material'],);p=np.concatenate([r['poly'] for r in items]);normals=[np.cross(np.asarray(r['poly'])[1,:3]-np.asarray(r['poly'])[0,:3],np.asarray(r['poly'])[2,:3]-np.asarray(r['poly'])[0,:3]) for r in items]
        n=sum((v if v@normals[0]>=0 else -v for v in normals),np.zeros(3))
        n[2]=0.;n/=np.linalg.norm(n);u=np.array([-n[1],n[0],0.])
        if n@(p[:,:3].mean(0)-source_center)<0:n=-n;u=-u
        x=p[:,:3]@u;z=p[:,2];x0,x1=float(x.min()),float(x.max());z0,z1=float(z.min()),float(z.max())
        area=sum(polygon_area(r['poly']) for r in items);width=x1-x0;height=z1-z0
        texture=material_texture(source.data.materials[key[0]]);is_window=texture in windows
        existing_door=registry['source_object']=='11_29_gc095' and texture=='11-29-gc034.tif'
        if width<.15 or height<.15 or (not is_window and (height<1.4 or width<.5)) or existing_door:
            for r in items:
                r=dict(r,component_override='WindowFrame' if is_window else 'DoorPanel',material_override='painted_aluminium' if is_window else 'industrial_painted_steel')
                if existing_door and width>.6 and height>1.4 and area>width*height*.7:r['existing_glazed_door']=True
                others.append(r)
            reports.append({'opening':number,'status':'existing_profile_or_hardware_retained','source_face_ids':sorted({r['face'] for r in items}),
                'width_m':width,'height_m':height,'basis':'Measured narrow/low profile, or source095 existing door body and modeled hardware; no duplicate generic frame or handle'})
            continue
        if width<.15 or height<.15 or abs(area-width*height)>max(area*.005,1e-5):
            others.extend(items);reports.append({'opening':number,'status':'source_shape_retained_nonrectangular','source_face_ids':sorted({r['face'] for r in items})});continue
        plane=float((p[:,:3]@n).mean());affine=np.linalg.lstsq(np.c_[x,z,np.ones(len(x))],p[:,3:5],rcond=None)[0]
        support=opening_support_offset(others,n,u,[0.,0.,1.],(x0,x1,z0,z1),plane)
        frame_back=min(-.018,support-.003) if support is not None else -.018
        frame_width=min(.035,width*.08,height*.08)
        vertical=[(x0,x0+frame_width),(x1-frame_width,x1)]
        horizontal=[(z0,z0+frame_width),(z1-frame_width,z1)]
        if is_window and abs(affine[0,0])>1e-8 and abs(affine[1,1])>1e-8:
            columns,transom=windows[texture];a=affine[0,0];b=affine[1,0]*(z0+z1)/2+affine[2,0]
            for k in range(math.floor(p[:,3].min()*columns),math.ceil(p[:,3].max()*columns)+1):
                v=(k/columns-b)/a;vertical.append((v-frame_width/2,v+frame_width/2))
            for k in range(math.floor(p[:,4].min()),math.ceil(p[:,4].max())+1):
                v=(k+transom-affine[0,1]*(x0+x1)/2-affine[2,1])/affine[1,1]
                horizontal.append((v-frame_width/2,v+frame_width/2))
        elif not is_window:
            vertical.append(((x0+x1)/2-.0125,(x0+x1)/2+.0125))
            if texture in ['11-29-gc023.tif','11-29-gc034.tif']:
                divider=z0+height*(.08 if texture=='11-29-gc023.tif' else .42)
                horizontal.append((divider-frame_width/2,divider+frame_width/2))
                if texture=='11-29-gc023.tif':
                    transom=z0+height*.82;horizontal.append((transom-frame_width/2,transom+frame_width/2))
        vertical=merged(vertical,x0,x1);horizontal=merged(horizontal,z0,z1)
        xg,zg=gaps(vertical,x0,x1),gaps(horizontal,z0,z1)
        frame_boxes=[(a,b,z0,z1) for a,b in vertical]+[(a,b,c,d) for a,b in xg for c,d in horizontal]
        panel_boxes=[(a,b,c,d) for a,b in xg for c,d in zg]
        assert abs(sum((b-a)*(d-c) for a,b,c,d in frame_boxes+panel_boxes)-width*height)<1e-6
        fid=sorted({r['face'] for r in items});basis=np.array([u,[0.,0.,1.],n])
        for kind,boxes,back,front in [('WindowFrame' if is_window else 'DoorFrame',frame_boxes,frame_back,.05),('WindowGlass' if is_window else 'DoorPanel',panel_boxes,-.021,-.003 if is_window else .015)]:
            for j,bounds in enumerate(boxes):
                actual_kind=kind
                if kind=='DoorPanel' and texture in ['11-29-gc023.tif','11-29-gc034.tif'] and (bounds[2]+bounds[3])/2>z0+height*(.08 if texture=='11-29-gc023.tif' else .42):actual_kind='DoorGlass'
                parts=defaultdict(list)
                generated=box_records(bounds,n,u,plane,back,-.003 if actual_kind=='DoorGlass' else front,source,affine,fid,.002 if kind.endswith('Frame') else 0.)
                for r in generated:
                    for cell,poly in partition_surface(r['poly'],basis,2.):parts[cell].append(dict(r,poly=poly))
                for cell,polys in sorted(parts.items()):
                    rgb=[plain_material()] if kind.endswith('Frame') else [source.data.materials[key[0]]]
                    result=create_surface(source,registry,('opening',number,actual_kind,j,cell),polys,actual_kind,
                        'glass_windshield' if actual_kind in ['WindowGlass','DoorGlass'] else 'painted_aluminium' if kind=='WindowFrame' else 'industrial_painted_steel',
                        (number,j,cell),'Source measured rectangle and UV layout;35mm nominal frame width/18mm glass thickness/53mm glass-front recess from frame front/2mm frame bevel are explicit construction assumptions',rgb)
                    result['source_face_ids']=fid;OBJECT_INDEX[result['object']]['source_face_ids']=fid;rows.append(result)
        # An actual projecting sill meets the lower frame; hardware meets door leaf.
        additions=[('WindowSill',(x0-.02,x1+.02,z0-.017,z0+.003),frame_back,.08)] if is_window else [
            ('DoorHandle',((x0+x1)/2-.09,(x0+x1)/2-.065,z0+min(1.,height*.5),z0+min(1.,height*.5)+.12),-.006,.06),
            ('DoorHandle',((x0+x1)/2+.065,(x0+x1)/2+.09,z0+min(1.,height*.5),z0+min(1.,height*.5)+.12),-.006,.06)]
        for j,(kind,bounds,back,front) in enumerate(additions):
            if kind=='DoorHandle':
                a,b,c,d=bounds
                generated=box_records((a,b,c,c+.02),n,u,plane,back,.045,source,affine,fid,.001)
                generated+=box_records((a,b,d-.02,d),n,u,plane,back,.045,source,affine,fid,.001)
                generated+=box_records((a,b,c+.01,d-.01),n,u,plane,.035,front,source,affine,fid,.001)
            else:generated=box_records(bounds,n,u,plane,back,front,source,affine,fid,.001)
            parts=defaultdict(list)
            for r in generated:
                for cell,poly in partition_surface(r['poly'],basis,2.):parts[cell].append(dict(r,poly=poly))
            for cell,polys in sorted(parts.items()):
                result=create_surface(source,registry,('hardware',number,kind,j,cell),polys,kind,'painted_aluminium' if is_window else 'industrial_painted_steel',cell,
                    'Explicit outer-profile assumption:20mm sill with80mm outward reach overlaps lower frame3mm; or25x120mm pull handle with two attached standoffs and clearance behind grip. Hidden profile cavities are not resolved; not surveyed hardware.',[plain_material()])
                result['source_face_ids']=fid;OBJECT_INDEX[result['object']]['source_face_ids']=fid;rows.append(result)
        cuts.append({'normal':n,'u':u,'plane':plane,'bounds':(x0,x1,z0,z1),'depth':max(.05,-frame_back+.005)})
        reports.append({'opening':number,'status':'constructed','texture':texture,'width_m':width,'height_m':height,
            'source_face_ids':fid,'source_area_m2':area,'surface_partition_area_m2':width*height,
            'rectangularisation_area_adjustment_m2':width*height-area,'normal':n.tolist(),'u':u.tolist(),
            'plane':plane,'bounds_xz':[x0,x1,z0,z1],'carrier_offset_m':support,'frame_rear_offset_m':frame_back,
            'support_basis':'Measured parallel source carrier overlap; rear extends3mm into carrier' if support is not None else 'No separate carrier found within100mm; frame depth is an explicit assumed profile in source opening plane'})
    # Remove any source opaque carrier surface behind the new recessed opening.
    kept=[];removed_area=0.;carrier_ids=set()
    for r in others:
        pieces=[np.asarray(r['poly']).copy()]
        for cut in cuts:
            n,u,plane=cut['normal'],cut['u'],cut['plane'];next_pieces=[]
            for p in pieces:
                if np.max(np.abs(p[:,:3]@n-plane))>cut['depth']:next_pieces.append(p);continue
                basis=np.array([u,[0.,0.,1.],n]);q=p.copy();q[:,:3]=q[:,:3]@basis.T
                inside=list(q);outside=[];x0,x1,z0,z1=cut['bounds']
                for axis,value,above in [(0,x0,True),(0,x1,False),(1,z0,True),(1,z1,False)]:
                    if not inside:break
                    part=clip_polygon(inside,axis,value,keep_above=not above)
                    if part:outside.append(part)
                    inside=clip_polygon(inside,axis,value,keep_above=above)
                if inside:removed_area+=polygon_area(inside);carrier_ids.add(r['face'])
                for q in outside:
                    q=np.asarray(q);q[:,:3]=q[:,:3]@basis;next_pieces.append(q)
            pieces=next_pieces
        kept.extend(dict(r,poly=list(p)) for p in pieces)
    return rows,kept,{'openings':reports,'opaque_carrier_area_cut_m2':removed_area,'opaque_carrier_source_face_ids':sorted(carrier_ids)}
