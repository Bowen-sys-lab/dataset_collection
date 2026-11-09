"""Geometry-preserving primitives shared by full-site component authoring."""
import math
import numpy as np
from collections import defaultdict
from scripts.blender.model02_geometry import polygon_area


def native_area_rounding_bound(local_vertices,world_matrix,triangle_indices):
    p=np.asarray(local_vertices,dtype='float32');matrix=np.asarray(world_matrix,dtype=float)
    error=.5*np.linalg.norm(np.max(np.spacing(abs(p)),axis=0))*np.linalg.norm(matrix[:3,:3],ord=2)
    world=(np.c_[p,np.ones(len(p))]@matrix.T)[:,:3];triangles=world[np.asarray(triangle_indices)]
    edges=np.linalg.norm(triangles[:,1]-triangles[:,0],axis=1)+np.linalg.norm(triangles[:,2]-triangles[:,0],axis=1)
    return float(error*edges.sum()+2.*len(triangles)*error**2)


def opening_support_offset(records,normal,u,v,bounds,plane):
    from scripts.blender.model02_geometry import clip_polygon
    basis=np.asarray([u,v,normal]);offsets=[];x0,x1,y0,y1=bounds
    for record in records:
        p=np.asarray(record['poly'])[:,:3]@basis.T;offset=p[:,2]-plane
        if np.ptp(offset)>.002 or offset.mean()>.002 or offset.mean()<-.1:continue
        inside=list(p)
        for axis,value,above in [(0,x0,True),(0,x1,False),(1,y0,True),(1,y1,False)]:
            if not inside:break
            inside=clip_polygon(inside,axis,value,keep_above=above)
        if inside and polygon_area(inside)>1e-6:offsets.append(float(offset.mean()))
    return max(offsets) if offsets else None


def split_by_planes(poly,axis,planes):
    """Exact sign tests avoid extrapolating near-coplanar source edges."""
    pieces=[poly]
    for plane in planes:
        result=[]
        for p in pieces:
            values=[v[axis]-plane for v in p]
            if not min(values)<0.<max(values):result.append(p);continue
            for above in [False,True]:
                cut=[]
                for a,b in zip(p[-1:]+p[:-1],p):
                    da,db=a[axis]-plane,b[axis]-plane;ia=da>=0. if above else da<=0.;ib=db>=0. if above else db<=0.
                    if ia!=ib:
                        point=a+(b-a)*(-da/(db-da));point[axis]=plane;cut.append(point)
                    if ib:cut.append(b.copy())
                cut=clean_mesh_polygon(cut)
                if cut and polygon_area(cut)>1e-12:result.append(cut)
        pieces=result
    return pieces


def clean_mesh_polygon(poly):
    # Blender stores vertex coordinates as float32. Sub-micron clipping crumbs
    # cannot represent physical construction and may collapse into invalid faces.
    out=[]
    for p in poly:
        if not out or np.linalg.norm(p[:3]-out[-1][:3])>1e-7:out.append(p)
    if len(out)>1 and np.linalg.norm(out[0][:3]-out[-1][:3])<=1e-7:out.pop()
    return out if len(out)>=3 else []


def connected_opening_groups(records,textures):
    roots=list(range(len(records)));edges=defaultdict(list);normals=[]
    def find(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    for i,r in enumerate(records):
        p=np.asarray(r['poly']);n=np.cross(p[1,:3]-p[0,:3],p[2,:3]-p[0,:3]);n/=np.linalg.norm(n);normals.append(n)
        vertices=[tuple(np.round(v[:3],5)) for v in p]
        for a,b in zip(vertices,vertices[1:]+vertices[:1]):
            key=(textures[r['material']],*sorted([a,b]))
            for j in edges[key]:
                q=np.asarray(records[j]['poly'])
                if abs(n@normals[j])>.99999 and np.max(np.abs((q[:,:3]-p[0,:3])@n))<.002:
                    x,y=find(i),find(j)
                    if x!=y:roots[max(x,y)]=min(x,y)
            edges[key].append(i)
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[find(i)].append(r)
    return [groups[k] for k in sorted(groups)]


def partition_surface(poly,basis,pitch):
    basis=np.asarray(basis);p=np.asarray(poly,dtype=float).copy();p[:,:3]=p[:,:3]@basis.T
    pieces=[list(p)]
    for axis in range(3):
        lo,hi=float(p[:,axis].min()),float(p[:,axis].max())
        planes=np.arange(math.floor(lo/pitch)+1,math.ceil(hi/pitch))*pitch
        pieces=[piece for old in pieces for piece in split_by_planes(old,axis,planes)]
    out=[]
    for piece in pieces:
        piece=np.asarray(piece);key=tuple(np.floor(piece[:,:3].mean(0)/pitch+1e-9).astype(int))
        piece[:,:3]=piece[:,:3]@basis
        clean=clean_mesh_polygon(list(piece))
        if clean:out.append((key,clean))
    return out


def box_geometry(bounds,n,u,plane,back,front):
    x0,x1,z0,z1=bounds;n=np.asarray(n);u=np.asarray(u)
    vertices=[u*x+n*(plane+d)+np.array([0.,0.,z]) for d in [back,front]
              for x,z in [(x0,z0),(x1,z0),(x1,z1),(x0,z1)]]
    return vertices,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]


def section_segment(poly,z):
    points=[]
    for a,b in zip(poly[-1:]+poly[:-1],poly):
        if (a[2]-z)*(b[2]-z)<0:
            points.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
    return points if len(points)==2 else []


def partition_axial(poly,basis,center,pitch,sectors):
    """Split only existing shell faces; never add caps at a thermal boundary."""
    basis=np.asarray(basis);center=np.asarray(center);p=np.asarray(poly).copy()
    p[:,:3]=(p[:,:3]-center)@basis.T;pieces=[list(p)]
    lo,hi=p[:,2].min(),p[:,2].max()
    pieces=[q for old in pieces for q in split_by_planes(old,2,np.arange(math.floor(lo/pitch)+1,math.ceil(hi/pitch))*pitch)]
    for i in range(sectors//2):
        offset=math.pi/4 if sectors==4 else 0.
        angle=i*2*math.pi/sectors+offset;c,s=math.cos(angle),math.sin(angle)
        rot=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
        result=[]
        for old in pieces:
            q=np.asarray(old).copy();q[:,:3]=q[:,:3]@rot.T
            for cut in split_by_planes(list(q),0,[0.]):
                cut=np.asarray(cut);cut[:,:3]=cut[:,:3]@rot;result.append(list(cut))
        pieces=result
    out=[]
    for p in pieces:
        p=np.asarray(p);mean=p[:,:3].mean(0)
        key=(int(math.floor(mean[2]/pitch+1e-9)),int(math.floor((math.atan2(mean[1],mean[0])+math.pi+(math.pi/4 if sectors==4 else 0.))*sectors/(2*math.pi)))%sectors)
        p[:,:3]=p[:,:3]@basis+center;clean=clean_mesh_polygon(list(p))
        if clean:out.append((key,clean))
    return out


def material_texture(material):
    if not material or not material.use_nodes:return ''
    for n in material.node_tree.nodes:
        if n.type=='TEX_IMAGE' and n.image:
            value=n.image.filepath.replace('\\','/').split('/')[-1].lower()
            for key in ['127','1709','1974','124']:
                if value.startswith('map') and key in value:return 'map_'+key+'.jpg'
            if material.name.startswith('V5_grass'):return '11-29-gc006.jpg'
            return value
    return ''


def surface_kind(texture,role,normal,z,lower_z,upper_z):
    if texture in ['11-29-gc003.tif','11-29-gc005.jpg','11-29-gc033.jpg']:return 'WindowGlass','glass_windshield'
    if texture in ['11-29-gc023.tif','11-29-gc034.tif','11-29-gc035.tif']:return 'DoorPanel','industrial_painted_steel'
    if texture in ['11-29-gc006.jpg','11-29-gc007.jpg','map_1974.jpg']:return 'Landscape','vegetation_leaf'
    building_component='Roof' if normal[2]>.55 and z>(lower_z+upper_z)/2 else 'Soffit' if normal[2]<-.55 else 'Wall'
    if texture in ['11-29-gc013.jpg','11-29-gc020.jpg','11-29-gc031.jpg','11-29-gc032.jpg']:
        return (building_component if role=='building' else 'BareMetalSurface'),'bare_aluminium'
    if texture in ['11-29-gc008.jpg','11-29-gc014.jpg','11-29-gc015.jpg','11-29-gc016.jpg','11-29-gc028.jpg','11-29-gc029.jpg','11-29-gc030.jpg']:
        return (building_component if role=='building' else 'MetalCladding'),'industrial_painted_steel'
    if texture in ['11-29-gc009.jpg','11-29-gc010.jpg','11-29-gc022.jpg','11-29-gc038.jpg','11-29-gc012.jpg','map_127.jpg']:
        return (building_component if role=='building' else 'Foundation'),'concrete'
    if texture in ['11-29-gc019.jpg','map_1709.jpg']:return 'PavingAndKerb','paving_concrete'
    if role=='ground':return 'GroundStructure','concrete'
    if role=='building':return building_component,'etics_render'
    if role=='pipework':return 'PipeWall','industrial_painted_steel'
    if role=='structure':return 'StructuralMetal','industrial_painted_steel'
    if z<lower_z+.45 and abs(normal[2])>.6:return 'EquipmentBase','industrial_painted_steel'
    return 'EquipmentShell','industrial_painted_steel'
