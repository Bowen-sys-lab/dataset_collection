"""Shallow filled ground joints; dimensions are explicit simulation assumptions."""
import math
import numpy as np
from scripts.blender.model02_geometry import split_by_planes

LAYOUT={'paving':(.6,.008,.003,.002),'road':(4.,.01,.005,.001)}


def height(x,y,kind):
    tile,joint,bevel,depth=LAYOUT[kind];pitch=tile+joint
    distances=[]
    for value in (x,y):
        u=value%pitch
        distances.append(min(u-joint/2,pitch-joint/2-u))
    return -depth*(1.-float(np.clip(min(distances)/bevel,0.,1.)))


def relief_pieces(poly,kind):
    tile,joint,bevel,depth=LAYOUT[kind];pitch=tile+joint
    pieces=[poly]
    for axis in (0,1):
        lower=min(p[axis] for p in poly);upper=max(p[axis] for p in poly)
        planes=[i*pitch+offset for i in range(math.floor(lower/pitch),math.floor(upper/pitch)+1)
                for offset in (joint/2+bevel,pitch-joint/2-bevel)]
        pieces=[p for old in pieces for p in split_by_planes(old,axis,planes)]
    for p in pieces:
        for v in p:v[2]+=height(v[0],v[1],kind)
    return pieces
