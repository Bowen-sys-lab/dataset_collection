"""One-time authored paving layout and estimated night snapshot (not a solver)."""
import math
import numpy as np
from scripts.blender.model02_geometry import clip_polygon, polygon_area


def partition(poly, tile_size=.6, joint=.008):
    """Disjoint brick quadrants and filled joints, preserving interpolated UVs.

    X/Y are paving-local coordinates; all remaining attributes are interpolated.
    Cell pitch includes the joint. No height offset or overlaid thermal decals.
    """
    pitch=tile_size+joint
    xyz=np.asarray(poly)
    lower=np.floor(xyz[:,:2].min(0)/pitch).astype(int)
    upper=np.floor((xyz[:,:2].max(0)-1e-10)/pitch).astype(int)
    result=[]
    for i in range(lower[0],upper[0]+1):
        start=i*pitch
        xs=[start,start+joint/2,start+joint/2+tile_size/2,start+pitch-joint/2,start+pitch]
        for xi in range(4):
            clipped=clip_polygon(poly,0,xs[xi],keep_above=True)
            clipped=clip_polygon(clipped,0,xs[xi+1],keep_above=False) if clipped else []
            if not clipped:continue
            # Long thin diagonal source faces otherwise scan their entire empty
            # bounding rectangle for every X strip. Clip X once, then visit only
            # Y cells touched by that strip; geometry and target keys are identical.
            first=int(math.floor(min(p[1] for p in clipped)/pitch))
            last=int(math.floor((max(p[1] for p in clipped)-1e-10)/pitch))
            for j in range(first,last+1):
                start=j*pitch
                ys=[start,start+joint/2,start+joint/2+tile_size/2,start+pitch-joint/2,start+pitch]
                for yi in range(4):
                    piece=clip_polygon(clipped,1,ys[yi],keep_above=True)
                    piece=clip_polygon(piece,1,ys[yi+1],keep_above=False) if piece else []
                    if len(piece)<3 or polygon_area(piece)<=1e-12:continue
                    tile=xi in (1,2) and yi in (1,2)
                    key=(i,j,'tile' if tile else 'joint',(xi-1)*2+yi-1 if tile else 0)
                    result.append((key,piece))
    return result


def temperature(x,y,part):
    # Broad spatial field encodes an assumed residual-heat pattern, not noise or
    # a temperature derived from RGB colour. All output is editable fixed targets.
    residual=.18*math.sin((x-700)/24)+.09*math.cos((y+830)/19)
    return float(22.+residual if part=='tile' else 21.7+.65*residual)
