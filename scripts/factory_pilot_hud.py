"""CPU-only collection guidance; projection never certifies image visibility."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import numpy as np


@dataclass
class EquipmentClearance:
    vertices: np.ndarray
    triangles: np.ndarray
    body_radius_m: float

    @classmethod
    def load(cls, root, inspection):
        root = Path(root).resolve()
        relative = Path(inspection['geometry_file'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError('Pilot geometry must remain inside the project')
        if hashlib.sha256(path.read_bytes()).hexdigest() != inspection['geometry_sha256']:
            raise ValueError('Pilot equipment geometry hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            vertices = np.asarray(data['equipment_vertices'], dtype=float)
            triangles = np.asarray(data['equipment_triangles'], dtype=np.int64)
        radius = float(inspection['inspection_requirements']['body_envelope_radius_m'])
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('Pilot body radius must be finite and non-negative')
        return cls(vertices, triangles, radius)

    def distance(self, position):
        from scripts.evaluate_factory_inspection import point_to_mesh_distances
        return float(point_to_mesh_distances(np.asarray(position).reshape(1,3),
                     self.vertices, self.triangles, body_radius_m=self.body_radius_m)[0])


def projected_inside(target, matrix, width, height, focal, haperture, vaperture):
    """Frustum projection only: no geometry occlusion or AOV age assertion."""
    values = [focal, haperture, vaperture, width, height]
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
        return False
    matrix = np.asarray(matrix, dtype=float)
    cam = (np.asarray(target, dtype=float) - matrix[3,:3]) @ matrix[:3,:3].T
    depth = -float(cam[2])
    if not np.isfinite(cam).all() or depth <= 1e-6:
        return False
    u = focal / haperture * width * cam[0] / depth + width / 2
    v = height / 2 - focal / vaperture * height * cam[1] / depth
    return bool(0 <= u < width and 0 <= v < height)


class ObservationTimer:
    def __init__(self, distance_range=(6.,8.), hold_s=3., max_speed=.3, max_gap=.25):
        self.lo, self.hi = distance_range
        self.required, self.max_speed, self.max_gap = hold_s, max_speed, max_gap
        self.last_time = self.last_position = None
        self.last_eligible = False
        self.hold = 0.
        self.last_speed = None

    def update(self, sim_time, position, clearance, in_frame):
        position = np.asarray(position, dtype=float)
        valid = (position.shape == (3,) and np.isfinite(position).all()
                 and math.isfinite(float(sim_time)))
        dt = float(sim_time) - self.last_time if valid and self.last_time is not None else None
        speed = (float(np.linalg.norm(position-self.last_position)/dt)
                 if dt is not None and dt > 1e-6 else None)
        in_range = (clearance is not None and math.isfinite(float(clearance))
                    and self.lo <= clearance <= self.hi)
        if dt == 0 and valid and self.last_position is not None and np.array_equal(position, self.last_position):
            if not (in_range and in_frame):
                self.hold = 0.
                self.last_eligible = False
            return {'hold_s': self.hold, 'hold_complete': self.hold + 1e-9 >= self.required,
                    'speed_m_s': self.last_speed, 'in_range': bool(in_range),
                    'slow_enough': self.last_speed is not None and self.last_speed <= self.max_speed + 1e-9,
                    'projected_inside': bool(in_frame), 'actual_visibility_verified': False}
        slow = speed is not None and math.isfinite(speed) and speed <= self.max_speed + 1e-9
        observed = dt is not None and 1e-6 < dt <= self.max_gap + 1e-9
        eligible = bool(valid and in_range and slow and in_frame and observed)
        if eligible and self.last_eligible:
            self.hold += dt
        elif dt == 0 and self.last_eligible and valid and in_range and in_frame:
            pass  # Re-reading a paused state cannot add simulation time.
        else:
            self.hold = 0.
        self.last_eligible = eligible
        self.last_time = float(sim_time) if valid else None
        self.last_position = position.copy() if valid else None
        self.last_speed = speed
        return {'hold_s': self.hold, 'hold_complete': self.hold + 1e-9 >= self.required,
                'speed_m_s': speed, 'in_range': bool(in_range), 'slow_enough': bool(slow),
                'projected_inside': bool(in_frame), 'actual_visibility_verified': False}
