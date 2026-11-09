"""Conservative continuous sphere checks against static triangle surfaces.

Distance to a fixed surface is 1-Lipschitz along a segment. Endpoint distances
therefore bound every point between them. Subdivide only intervals whose bound
cannot prove clearance. An unresolved interval never becomes CLEAR_SURFACES.
This unsigned surface test does not establish whether a starting point is inside
a closed solid or whether the supplied geometry covers the entire scene.
"""
import math
import numpy as np
from scripts.evaluate_factory_inspection import point_to_mesh_distances


def sweep_surface_segment(start, end, triangles, *, radius, spatial_tolerance=1e-3):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    faces = np.asarray(triangles, dtype=float)
    if (start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all()
        or not np.isfinite(end).all()):
        raise ValueError('Segment endpoints must be finite three-dimensional points')
    if faces.ndim != 3 or faces.shape[1:] != (3, 3) or not len(faces) or not np.isfinite(faces).all():
        raise ValueError('Surface geometry must be a nonempty finite Mx3x3 triangle array')
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Sphere radius must be finite and positive')
    if not math.isfinite(spatial_tolerance) or spatial_tolerance <= 0:
        raise ValueError('Spatial tolerance must be finite and positive')
    epsilon = 1e-9
    try:
        with np.errstate(over='raise', invalid='raise'):
            low = np.minimum(start, end) - radius - epsilon
            high = np.maximum(start, end) + radius + epsilon
            vector = end - start
            length = float(np.linalg.norm(vector))
    except FloatingPointError as error:
        raise ValueError('Segment arithmetic exceeds supported numeric range') from error
    if not math.isfinite(length):
        raise ValueError('Segment length exceeds supported numeric range')
    candidate_indices = np.flatnonzero(np.all(faces.max(axis=1) >= low, axis=1)
                                      & np.all(faces.min(axis=1) <= high, axis=1))
    base = {'status': 'CLEAR_SURFACES', 'distance_queries': 0, 'contact_fraction_interval': None,
            'scope': 'Static supplied surfaces only; start containment and geometry coverage require separate checks'}
    if not len(candidate_indices):
        return base
    candidates = faces[candidate_indices]
    vertices = candidates.reshape(-1, 3)
    indices = np.arange(len(vertices)).reshape(-1, 3)

    def distance(fraction):
        base['distance_queries'] += 1
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                value = float(point_to_mesh_distances((start + fraction * vector)[None, :], vertices, indices)[0])
        except FloatingPointError as error:
            raise ValueError('Surface distance arithmetic exceeds supported numeric range') from error
        if not math.isfinite(value):
            raise ValueError('Surface distance is nonfinite')
        return value

    def search(left, right, left_distance, right_distance, depth):
        if left_distance <= radius + epsilon:
            return {'status': 'CONTACT', 'contact_fraction_interval': [left, left]}
        interval_length = (right - left) * length
        lower_bound = (left_distance + right_distance - interval_length) / 2
        if lower_bound > radius + epsilon:
            return None
        if interval_length <= spatial_tolerance or depth >= 60:
            return {'status': 'CONTACT' if right_distance <= radius + epsilon else 'INDETERMINATE',
                    'contact_fraction_interval': [left, right]}
        middle = (left + right) / 2
        middle_distance = distance(middle)
        earlier = search(left, middle, left_distance, middle_distance, depth + 1)
        return earlier if earlier is not None else search(middle, right, middle_distance, right_distance, depth + 1)

    hit = search(0., 1., distance(0.), distance(1.), 0)
    if hit is None:
        return base
    base.update(hit)
    if base['status'] == 'CONTACT':
        fraction = base['contact_fraction_interval'][1]
        point = start + fraction * vector
        distances = [float(point_to_mesh_distances(point[None, :], face, np.array([[0, 1, 2]]))[0])
                     for face in candidates]
        nearest = int(np.argmin(distances))
        base.update(contact_triangle_index=int(candidate_indices[nearest]),
                    sphere_surface_overlap_m=max(0., radius - distances[nearest]))
    return base
