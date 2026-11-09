"""Pure geometric operations shared by the v3 Blender zoning and checks."""
import math
import numpy as np
from scripts.blender.model02_geometry import split_by_planes


def corridor_distance(point,route):
    p=np.asarray(point,dtype=float)[:2]
    best=float('inf')
    for a,b in zip(route,route[1:]):
        a,b=np.asarray(a,dtype=float)[:2],np.asarray(b,dtype=float)[:2]
        d=b-a
        t=np.clip(np.dot(p-a,d)/np.dot(d,d),0.,1.)
        best=min(best,float(np.linalg.norm(p-a-t*d)))
    return best


def angular_bin(x,y,count):
    return int(math.floor((math.atan2(y,x)+math.pi)*count/(2*math.pi)))%count


def cell_in_corridor(key,route,radius=20.,step=2.):
    center=np.asarray(key[:2],dtype=float)*step+step/2
    return corridor_distance(center,route)<=radius+step/math.sqrt(2.)


def split_grid(polygon,basis,steps):
    """Split in an orthonormal frame, interpolating all other per-corner data."""
    matrix=np.asarray(basis,dtype=float)
    a=np.asarray(polygon,dtype=float).copy()
    a[:,:3]=a[:,:3]@matrix.T
    pieces=[list(a)]
    for axis,step in enumerate(steps):
        lo,hi=a[:,axis].min(),a[:,axis].max()
        planes=np.arange(math.floor(lo/step)+1,math.ceil(hi/step))*step
        pieces=[part for p in pieces for part in split_by_planes(p,axis,planes)]
    result=[]
    for part in pieces:
        part=np.asarray(part)
        key=tuple(np.floor(part[:,:3].mean(axis=0)/np.asarray(steps)+1e-8).astype(int))
        part[:,:3]=part[:,:3]@matrix
        result.append((key,list(part)))
    return result
