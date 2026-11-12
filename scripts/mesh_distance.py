"""Small geometry helpers shared by collision checks."""
import math
import numpy as np


def point_to_mesh_distances(points, vertices, triangles, *, body_radius_m=0.0):
    """Return unsigned point-to-triangle distances minus an optional radius."""
    points = np.asarray(points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    indices = np.asarray(triangles)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError('points must be a finite Nx3 array')
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('vertices must be a nonempty finite Nx3 array')
    if (indices.ndim != 2 or indices.shape[1:] != (3,) or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(vertices))):
        raise ValueError('triangles must be nonempty integer Mx3 vertex indices')
    if not math.isfinite(body_radius_m) or body_radius_m < 0:
        raise ValueError('body radius must be finite and nonnegative')
    faces = vertices[indices]
    output = np.empty(len(points), dtype=float)
    for point_index, point in enumerate(points):
        best = float('inf')
        for offset in range(0, len(faces), 8192):
            face = faces[offset:offset + 8192]
            a, b, c = face[:, 0], face[:, 1], face[:, 2]
            ab, ac = b - a, c - a
            normal = np.cross(ab, ac)
            normal_squared = np.einsum('ij,ij->i', normal, normal)
            nondegenerate = normal_squared > np.finfo(float).tiny
            ap = point - a
            numerator = np.einsum('ij,ij->i', ap, normal)
            factor = np.divide(numerator, normal_squared, out=np.zeros_like(numerator), where=nondegenerate)
            projection = point - factor[:, None] * normal
            inside = nondegenerate.copy()
            for start, end in ((a, b), (b, c), (c, a)):
                side = np.einsum('ij,ij->i', np.cross(end - start, projection - start), normal)
                inside &= side >= 0.0
                edge = end - start
                edge_squared = np.einsum('ij,ij->i', edge, edge)
                along = np.divide(np.einsum('ij,ij->i', point - start, edge), edge_squared,
                                  out=np.zeros(len(face)), where=edge_squared > 0)
                difference = point - (start + np.clip(along, 0.0, 1.0)[:, None] * edge)
                best = min(best, float(np.min(np.einsum('ij,ij->i', difference, difference))))
            if np.any(inside):
                best = min(best, float(np.min(numerator[inside] ** 2 / normal_squared[inside])))
        output[point_index] = math.sqrt(max(0.0, best)) - body_radius_m
    return output
