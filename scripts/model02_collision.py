"""Numerically local static collider, independent of thermal surface partitioning."""
import numpy as np


def consolidate_collision(points,triangles):
    points=np.asarray(points,dtype=np.float64);triangles=np.asarray(triangles,dtype=np.int32)
    origin=(points.min(0)+points.max(0))/2
    local=points-origin
    _,first,inverse=np.unique(np.round(local,6),axis=0,return_index=True,return_inverse=True)
    welded=local[first];faces=inverse[triangles]
    t=welded[faces];twice_area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    keep=twice_area>2e-12
    error=float(np.max(np.abs(welded[inverse]-local)))
    if error>1e-6:raise ValueError('Collider welding exceeds1um tolerance')
    return origin,welded,faces[keep],{'input_vertices':len(points),'welded_vertices':len(welded),
        'triangles':int(keep.sum()),'discarded_degenerate_triangles':int((~keep).sum()),
        'maximum_weld_coordinate_error_m':error,'local_origin_m':origin.tolist()}
