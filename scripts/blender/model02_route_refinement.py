"""Blender-first v3 construction and independent thermal surface zoning.

No USD/Isaac writes. Original geometry and the v2 work remain archived in .blend.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from scripts.blender.model02_geometry import clip_polygon,polygon_area,split_by_planes
from scripts.blender.model02_route_zones import split_grid,corridor_distance,angular_bin,cell_in_corridor

ROOT=Path('D:/IsaacSim_Demo/factory/model02_thermal_v3')
ROUTE=[(640,-855,15),(662,-832,15),(677,-816,15),(693.37,-797.25,3.5),(710,-807,15),(700,-840,15),(640,-855,15)]
PIPE_CENTER=np.array([682.28082,-799.35732])
BUILDING_BASIS=np.array([[.743294,.668965,0],[-.668965,.743294,0],[0,0,1.]])
BUILDING_BASIS[:2]/=np.linalg.norm(BUILDING_BASIS[0])


def authored_world(obj):
    """Hidden archive objects have unevaluated matrix_world after file reload."""
    if obj.parent is None:return obj.matrix_basis.copy()
    return authored_world(obj.parent)@obj.matrix_parent_inverse@obj.matrix_basis


def records(obj):
    m=obj.data;matrix=np.asarray(obj.matrix_world,dtype=float)
    uv=m.uv_layers.active.data if m.uv_layers.active else None
    result=[]
    m.calc_loop_triangles()
    for triangle in m.loop_triangles:
        face=m.polygons[triangle.polygon_index]
        poly=[]
        for li in triangle.loops:
            v=m.vertices[m.loops[li].vertex_index]
            p=(matrix@np.r_[v.co[:],1.])[:3]
            poly.append(np.r_[p,tuple(uv[li].uv) if uv else (0.,0.),tuple(m.corner_normals[li].vector)])
        result.append({'face':face.index,'poly':poly,'material':face.material_index,'smooth':face.use_smooth})
    return result


def scene_parts():
    import bpy
    return [o for o in bpy.context.scene.objects if o.type=='MESH' and o.get('surface_id') and not o.hide_render]


def setup():
    import bpy
    assert Path(bpy.data.filepath)==ROOT/'model02_thermal_v3.blend'
    assert bpy.data.collections.get('Model02_RouteZones') is None
    col=bpy.data.collections.new('Model02_RouteZones');bpy.context.scene.collection.children.link(col)
    archive=bpy.data.collections.new('Model02_V2Archive');bpy.context.scene.collection.children.link(archive)
    archive.hide_render=True;archive.hide_viewport=True
    parent=bpy.data.objects.new('RouteZone20m',None);col.objects.link(parent)
    parent.parent=bpy.data.objects['InspectionZone01']
    bpy.context.view_layer.update()
    baseline={'source_v2_parts':[o.name for o in scene_parts()],
              'source_objects':[o.name for o in bpy.context.scene.objects if o.name.startswith('11_29_gc')],
              'route':ROUTE,'corridor_radius_m':20.,'scope_boundary':'2m cells conservatively include boundary-crossing surfaces',
              'area_checks':[],'processed':[]}
    (ROOT/'construction_state.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    return {'ready':True}


def archive(obj):
    import bpy
    col=bpy.data.collections['Model02_V2Archive']
    for old in list(obj.users_collection):old.objects.unlink(obj)
    col.objects.link(obj);obj.hide_render=True;obj.hide_set(True)


def new_mesh(name,source,items,component,region,temp,material,parent=None,background=False):
    import bpy
    m=bpy.data.meshes.new(name+'_Mesh');inv=np.linalg.inv(np.asarray(source.matrix_world,dtype=float))
    verts=[];faces=[];uvs=[];normals=[];lookup={}
    slots=sorted({r['material'] for r in items});slot_map={n:i for i,n in enumerate(slots)}
    created=[]
    for r in items:
        # Clipping exactly at source corners can leave numerically coincident
        # corners. Deduplicate at the SAME tolerance as vertex welding before
        # passing topology to Blender's native custom-normal implementation.
        corners=[];seen=set()
        for v in r['poly']:
            key=tuple(np.round(v[:3],8))
            if key in seen:continue
            seen.add(key);corners.append((key,v))
        if len(corners)<3:continue
        face=[]
        for key,v in corners:
            if key not in lookup:
                lookup[key]=len(verts);verts.append((inv@np.r_[v[:3],1.])[:3])
            face.append(lookup[key]);uvs.append(v[3:5]);n=v[5:8];normals.append(n/max(np.linalg.norm(n),1e-12))
        faces.append(face);created.append(r)
    m.from_pydata(verts,[],faces)
    for slot in slots:m.materials.append(source.data.materials[slot])
    for p,r in zip(m.polygons,created):p.material_index=slot_map[r['material']];p.use_smooth=r['smooth']
    uv=m.uv_layers.new(name='UVMap');uv.data.foreach_set('uv',np.asarray(uvs,dtype='float32').reshape(-1))
    m.update()
    if m.validate(verbose=True,clean_customdata=False):
        raise ValueError('Invalid clipped topology before normal assignment: '+name)
    assert len(m.loops)==len(normals),(name,len(m.loops),len(normals))
    m.normals_split_custom_set(normals)
    o=bpy.data.objects.new(name,m);bpy.data.collections['Model02_RouteZones'].objects.link(o)
    o.parent=parent or bpy.data.objects['RouteZone20m'];o.matrix_world=source.matrix_world.copy()
    o['surface_id']=name;o['temperature_target_id']=name.lower();o['temperature_c']=float(temp)
    o['ir_material']=material;o['component']=component;o['region']=str(region)
    o['source_object']=source.get('source_object',source.name);o['source_part']=source.name
    o['thermal_scope']='background_retained' if background else 'route_refined'
    o['source_face_ids']=sorted({r['face'] for r in items})
    o['physical_material']=material;o['temperature_basis']='Estimated dry low-wind night operating snapshot; per-zone prescribed, not solved or measured'
    return o


def sector_split(poly,center,count):
    # Split through radial planes, not merely classify a long original face.
    pieces=[poly]
    for i in range(count//2):
        a=i*math.pi/(count//2)
        normal=np.array([-math.sin(a),math.cos(a),0.]);distance=np.dot(normal[:2],center)
        basis=np.array([normal,[math.cos(a),math.sin(a),0.],[0.,0.,1.]])
        transformed=[]
        for p in pieces:
            p=np.asarray(p).copy();p[:,:3]=p[:,:3]@basis.T;p[:,0]-=distance
            for cut in split_by_planes(list(p),0,[0.]):
                cut=np.asarray(cut);cut[:,0]+=distance;cut[:,:3]=cut[:,:3]@basis
                transformed.append(list(cut))
        pieces=transformed
    return pieces


def update_state(source,source_area,output_area,count,policy):
    path=ROOT/'construction_state.json';s=json.loads(path.read_text())
    relative=abs(source_area-output_area)/max(source_area,1e-12)
    assert relative<2e-6,(source,relative)
    s['area_checks'].append({'source':source,'source_m2':source_area,'output_m2':output_area,'relative_error':relative})
    s['processed'].append({'source':source,'zone_count':count,'policy':policy})
    path.write_text(json.dumps(s,indent=2),encoding='utf-8')


def refine_existing():
    import bpy
    selected=[o for o in scene_parts() if not str(o['surface_id']).startswith('LGT.')]
    processed=0;zones=0
    for source in selected:
        component=source.get('component','')
        if component not in ['Wall','Roof','TankShell','TankRoof','Soffit','StepsAndPlinth','ExternalPipework','BaseRing','RailingAndSupports','Foundation','AccessAttachments']:
            continue
        groups=defaultdict(list);rs=records(source);source_area=sum(polygon_area(r['poly']) for r in rs)
        center=np.array(source.matrix_world.translation[:2])
        if component.startswith('Tank'):
            original=bpy.data.objects[source['source_object']]
            original_matrix=authored_world(original)
            points=np.array([tuple(original_matrix@v.co) for v in original.data.vertices])
            center=(points.min(0)+points.max(0))[:2]/2
        for r in rs:
            if component=='TankShell' and source.get('region','').startswith(('Hotspot','Transition')):
                splits=split_grid(r['poly'],np.eye(3),[.16,.16,.16])
            elif component=='TankShell':
                splits=[]
                for p in sector_split(r['poly'],center,8):
                    for q in split_by_planes(p,2,np.arange(0,15,1.)):
                        c=np.asarray(q)[:,:3].mean(0)
                        splits.append(((angular_bin(c[0]-center[0],c[1]-center[1],8),int(math.floor(c[2]))),q))
            else:
                step=.5 if component=='ExternalPipework' else (1. if component in ['BaseRing','RailingAndSupports','Foundation','AccessAttachments'] else 2.)
                splits=split_grid(r['poly'],BUILDING_BASIS if component in ['Wall','Roof','Soffit','StepsAndPlinth'] else np.eye(3),[step]*3)
            for key,p in splits:
                if component=='ExternalPipework':
                    xyz=np.asarray(p)[:,:3];normal=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
                    axis=int(np.argmax(np.abs(normal)));key=(*key,axis,int(normal[axis]>0))
                groups[key].append(dict(r,poly=p))
        for key,items in sorted(groups.items()):
            suffix='_'.join(('m'+str(-int(k))) if k<0 else str(int(k)) for k in key)
            c=np.mean(np.concatenate([r['poly'] for r in items])[:,:3],axis=0)
            temp=float(source['temperature_c']);ir=source['ir_material']
            if component=='Wall':temp={'Front':21.6,'Rear':20.7,'East':21.,'West':21.2}.get(source.get('region'),21.2)-.05*c[2]
            elif component in ['Roof','TankRoof']:temp=19.5 if component=='Roof' else 26.
            elif component=='Soffit':temp=21.2
            elif component=='StepsAndPlinth':temp=22.
            elif component in ['RailingAndSupports','AccessAttachments']:temp=20.5
            elif component=='Foundation':temp=22.
            elif component=='BaseRing':temp=25.
            elif component=='TankShell' and source.get('region','').startswith('Normal'):temp=30.-.08*c[2]
            if ir=='aircraft_aluminium_painted':ir='industrial_painted_steel'
            new_mesh(source.name+'_Z_'+suffix,source,items,component,source.get('region',''),temp,ir,parent=source.parent)
        out=sum(polygon_area(r['poly']) for items in groups.values() for r in items)
        update_state(source.name,source_area,out,len(groups),'Existing v2 surface split, original archived')
        archive(source);processed+=1;zones+=len(groups)
    bpy.context.view_layer.update()
    return {'refined_v2_parts':processed,'new_zones':zones}


def clip_rectangle(poly):
    inside=poly;outside=[]
    for axis,plane,above in [(0,620.,True),(0,732.,False),(1,-876.,True),(1,-776.,False)]:
        if not inside:break
        cut=clip_polygon(inside,axis,plane,keep_above=not above)
        if cut:outside.append(cut)
        inside=clip_polygon(inside,axis,plane,keep_above=above)
    return inside,outside


def source_class(name,slot,center,normal):
    if name=='11_29_gc010':
        return ('Landscape',20.,'vegetation_leaf') if slot in [2,7,14] else ('PavingAndKerb',22.,'concrete')
    if name=='11_29_gc090':return 'ConcreteContainment',22.,'concrete'
    if name=='11_29_gc092':return 'PlatformDeck',21.,'industrial_painted_steel'
    if name=='11_29_gc093':return 'RailAndSupport',20.5,'industrial_painted_steel'
    if slot in [7,8]:return 'Foundation',22.,'concrete'
    if slot==5:return 'RailAndAccess',20.5,'industrial_painted_steel'
    if slot==6:return 'ExternalPipework',35.,'industrial_painted_steel'
    if slot==2:return 'MetalCladding',27.,'bare_aluminium'
    if slot==1:return ('TankRoof',27.,'industrial_painted_steel') if normal[2]>.25 else ('VesselShell',32.-.08*center[2],'industrial_painted_steel')
    return 'AccessAttachment',22.,'industrial_painted_steel'


def refine_source(name):
    import bpy
    source=bpy.data.objects[name];rs=records(source);groups=defaultdict(list)
    # Three source vessels share one mesh: use measured column centers per welded body.
    vessel_centers=np.array([[682.6,-795.1],[674.4,-786.1],[665.,-777.1]])
    for r in rs:
        xyz=np.asarray(r['poly'])[:,:3];c=xyz.mean(0)
        normal=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]);normal/=max(np.linalg.norm(normal),1e-12)
        component,temp,ir=source_class(name,r['material'],c,normal)
        if name=='11_29_gc089':
            # Largest body facets split in height and radial angle. Other parts retain construction geometry.
            if component=='VesselShell':
                center=vessel_centers[np.argmin(np.linalg.norm(vessel_centers-c[:2],axis=1))]
                pieces=[p for q in sector_split(r['poly'],center,8) for p in split_by_planes(q,2,np.arange(0,12,1.))]
                splits=[((int(np.argmin(np.linalg.norm(vessel_centers-c[:2],axis=1))),angular_bin(np.mean(p,axis=0)[0]-center[0],np.mean(p,axis=0)[1]-center[1],8),int(math.floor(np.mean(p,axis=0)[2]))),p) for p in pieces]
            else:
                splits=split_grid(r['poly'],np.eye(3),[.5,.5,.5] if component=='ExternalPipework' else [1.,1.,1.])
        else:
            inside,outside=clip_rectangle(r['poly'])
            for p in outside:groups[('Background',r['material'])].append(dict(r,poly=p))
            splits=split_grid(inside,np.eye(3),[2.,2.,2.]) if inside else []
        for key,p in splits:
            c=np.mean(p,axis=0)[:3]
            in_scope=name=='11_29_gc089' or cell_in_corridor(key,ROUTE)
            tag=(component,r['material'],*key) if in_scope else ('Background',r['material'])
            # Divide the pipe surfaces by actual face orientation as well as 1m grid.
            if component=='ExternalPipework':tag=(*tag,int(np.argmax(np.abs(normal))),int(normal[np.argmax(np.abs(normal))]>0))
            groups[tag].append(dict(r,poly=p))
    count=0
    for key,items in sorted(groups.items(),key=lambda x:str(x[0])):
        background=key[0]=='Background';c=np.mean(np.concatenate([r['poly'] for r in items])[:,:3],axis=0)
        sample=items[0];poly=np.asarray(sample['poly']);n=np.cross(poly[1,:3]-poly[0,:3],poly[2,:3]-poly[0,:3]);n/=max(np.linalg.norm(n),1e-12)
        component,temp,ir=source_class(name,sample['material'],c,n)
        suffix='_'.join(str(k).replace('-','m') for k in key)
        new_mesh('GC'+name.split('gc')[-1]+'_'+suffix,source,items,component,'RetainedBackground' if background else 'Normal',temp,ir,background=background)
        count+=1
    update_state(name,sum(polygon_area(r['poly']) for r in rs),sum(polygon_area(r['poly']) for items in groups.values() for r in items),count,'Whole equipment cluster' if name=='11_29_gc089' else 'Local network cells plus retained outside fragments')
    archive(source);bpy.context.view_layer.update()
    return {'source':name,'output_parts':count}


def add_pipe_failure():
    """Measured original 80mm pipe; add 30mm insulation and real cutaway."""
    import bpy
    from mathutils import Matrix
    col=bpy.data.collections['Model02_RouteZones']
    parent=bpy.data.objects.new('PIPE_GC089_InsulationFault',None);col.objects.link(parent)
    parent.parent=bpy.data.objects['RouteZone20m']
    materials={}
    for name,color,metal in [('V3_AluminiumCladding',(.48,.5,.51),.65),('V3_MineralWool',(.48,.44,.27),0.)]:
        m=bpy.data.materials.new(name);m.use_nodes=True;n=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED');n.inputs['Base Color'].default_value=(*color,1.);n.inputs['Metallic'].default_value=metal;n.inputs['Roughness'].default_value=.5;materials[name]=m
    # Front opening looks toward the independent safe observation point.
    facing=math.atan2(-804.3-PIPE_CENTER[1],688.-PIPE_CENTER[0])
    bands=np.unique(np.r_[.35,1.,2.,2.8,np.arange(2.9,4.31,.15),5.,6.,7.,8.,8.15])
    angles=np.arange(49)*2*math.pi/48
    count=0;geometry_groups={}
    def accumulate(key,verts,faces,mat,component,region,temp,ir):
        if key not in geometry_groups:geometry_groups[key]=[[],[],mat,component,region,temp,ir]
        g=geometry_groups[key];offset=len(g[0]);g[0].extend(verts);g[1].extend([[offset+i for i in f] for f in faces])
    for za,zb in zip(bands,bands[1:]):
        for a,b in zip(angles,angles[1:]):
            mid=(a+b)/2;z=(za+zb)/2;delta=abs(math.atan2(math.sin(mid-facing),math.cos(mid-facing)))
            missing=3.25<=z<=3.85 and delta<math.pi/3
            # Annular end faces at each cut are authored only at physical exposed boundaries.
            if missing:continue
            quadrant=angular_bin(math.cos(mid),math.sin(mid),4)
            verts=[];faces=[]
            def ringpoint(r,t,h):return [PIPE_CENTER[0]+r*math.cos(t),PIPE_CENTER[1]+r*math.sin(t),h]
            for r in [.072,.070]:
                verts.extend([ringpoint(r,a,za),ringpoint(r,b,za),ringpoint(r,b,zb),ringpoint(r,a,zb)])
            faces=[[0,1,2,3],[7,6,5,4]]
            if abs(za-.35)<1e-5 or (abs(za-3.8)<1e-5 and delta<math.pi/3):faces.append([0,4,5,1])
            if abs(zb-8.15)<1e-5 or (abs(zb-3.2)<1e-5 and delta<math.pi/3):faces.append([3,2,6,7])
            # Group angular facets into four independent circumference zones.
            name=f'PIPE089_Cladding_H{za:.2f}_Q{quadrant}'.replace('.','p')
            accumulate(name,verts,faces,materials['V3_AluminiumCladding'],'InsulationCladding','Normal',25.5,'bare_aluminium')
            wool=[]
            for r in [.070,.04]:wool.extend([ringpoint(r,a,za),ringpoint(r,b,za),ringpoint(r,b,zb),ringpoint(r,a,zb)])
            # Wool surface is hidden below jacket, with cut ends visible at missing patch.
            wf=[[0,1,2,3],[7,6,5,4]]
            # Only real end and cut faces, no coincident internal segment caps.
            if abs(za-.35)<1e-5 or (abs(za-3.8)<1e-5 and delta<math.pi/3):wf.append([0,4,5,1])
            if abs(zb-8.15)<1e-5 or (abs(zb-3.2)<1e-5 and delta<math.pi/3):wf.append([3,2,6,7])
            for angle,face in [(a,[0,3,7,4]),(b,[1,5,6,2])]:
                if 3.25<=z<=3.85 and abs(abs(math.atan2(math.sin(angle-facing),math.cos(angle-facing)))-math.pi/3)<2*math.pi/48:wf.append(face)
            temp=31. if abs(z-3.55)<.65 and delta<1.5 else 25.5
            accumulate(name.replace('Cladding','MineralWool'),wool,wf,materials['V3_MineralWool'],'InsulationLayer','CutEdge' if temp>26 else 'Normal',temp,'mineral_wool')
    for name,g in geometry_groups.items():
        make_constructed(name,parent,*g);count+=1
    # Assign the existing steel pipe below the genuine opening; split it before temperature binding.
    selected=[o for o in scene_parts() if o.get('source_object')=='11_29_gc089' and o.get('component')=='ExternalPipework']
    cut_count=0
    for source in selected:
        rs=records(source);groups=defaultdict(list);touched=False
        for r in rs:
            p=np.asarray(r['poly']);c=p[:,:3].mean(0)
            if np.linalg.norm(c[:2]-PIPE_CENTER)<.09:
                touched=True
                for key,q in split_grid(r['poly'],np.eye(3),[.16,.16,.15]):
                    h=np.mean(q,axis=0)[2]
                    region='Hotspot' if 3.25<=h<=3.85 else ('Transition' if 2.8<h<4.3 else 'CoveredSteel')
                    groups[(region,*key)].append(dict(r,poly=q))
            else:groups[('Other',)].append(r)
        if not touched:continue
        for key,items in groups.items():
            region=key[0];h=np.mean(np.concatenate([r['poly'] for r in items]),axis=0)[2]
            temp=72. if region=='Hotspot' else (max(40.,72.-45.*max(0.,abs(h-3.55)-.3)) if region=='Transition' else (70. if region=='CoveredSteel' else 35.))
            suffix='_'.join(str(k).replace('-','m') for k in key)
            o=new_mesh(source.name+'_PipeDetail_'+suffix,source,items,'PipeWall',region,temp,'industrial_painted_steel',parent=parent)
            o['fault_id']='PIPE089_INSULATION_001' if region in ['Hotspot','Transition'] else ''
            cut_count+=1
        archive(source)
    assert cut_count>0,'Existing measured steel pipe was not found'
    bpy.context.view_layer.update()
    return {'new_insulation_meshes':count,'steel_zones':cut_count,'center_xy':PIPE_CENTER.tolist(),'steel_radius_m':.04,'insulation_outer_radius_m':.07,'outer_radius_m':.072,'fluid_temperature_assumption_c':80.}


def make_constructed(name,parent,verts,faces,material,component,region,temp,ir):
    import bpy
    from mathutils import Matrix
    center=np.mean(verts,axis=0);m=bpy.data.meshes.new(name+'_Mesh')
    m.from_pydata(np.asarray(verts)-center,[],faces);m.materials.append(material);m.update()
    o=bpy.data.objects.new(name,m);bpy.data.collections['Model02_RouteZones'].objects.link(o)
    o.parent=parent;o.matrix_world=Matrix.Translation(center)
    for k,v in {'surface_id':name,'temperature_target_id':name.lower(),'component':component,'region':region,'temperature_c':float(temp),'ir_material':ir,'physical_material':ir,'source_object':'11_29_gc089','thermal_scope':'route_refined','modeling_approximation':'Designed 30mm mineral wool/jacket around measured existing 80mm pipe; not source BIM','temperature_basis':'Prescribed estimated operating snapshot, fluid80C, air20C, no coupled solve'}.items():o[k]=v
    return o
