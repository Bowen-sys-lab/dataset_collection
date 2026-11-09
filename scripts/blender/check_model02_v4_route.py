"""Static reference-route QA; human route and mileage remain unrestricted."""
import json
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=Path('D:/IsaacSim_Demo');OUT=ROOT/'factory/model02_thermal_v4'


def main():
    parameters=json.loads((OUT/'mission_and_camera.json').read_text())
    d=np.load(ROOT/'factory/model02_thermal_v3/scene_geometry.npz');origin=np.array([700.,-800.,0.])
    b=BVHTree.FromPolygons((d['vertices']-origin).tolist(),d['triangles'].tolist(),all_triangles=True)
    p=np.array(parameters['reference_waypoints_m']);samples=np.concatenate([a+(c-a)*np.linspace(0,1,int(np.ceil(np.linalg.norm(c-a)/.25))+1)[:,None] for a,c in zip(p[:-1],p[1:])])
    clearance=[b.find_nearest(Vector(q-origin))[3]-.2-.125-.001 for q in samples]
    agl=[b.ray_cast(Vector(q-origin),Vector((0,0,-1)),1000)[3] for q in samples]
    tank=np.load(ROOT/'factory/model02_thermal_v3/tank_inspection_geometry.npz')
    equipment=BVHTree.FromPolygons((tank['equipment_vertices']-origin).tolist(),tank['equipment_triangles'].tolist(),all_triangles=True)
    distances=[equipment.find_nearest(Vector(q-origin))[3]-.2 for q in samples]
    assert min(clearance)>0 and min(agl)>=2.5 and min(distances)>=5.
    report={'result':'PASS_REFERENCE_ROUTE_STATIC_GEOMETRY','route_length_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
        'minimum_body_clearance_lower_bound_m':min(clearance),'minimum_agl_m':min(agl),'minimum_tank_equipment_clearance_m':min(distances),
        'body_radius_m':.2,'maximum_sample_step_m':.25,'subtracted_interval_bound_m':.125,'subtracted_numeric_allowance_m':.001,
        'start_position_m':parameters['start_position_m'],'start_clearance_m':b.find_nearest(Vector(np.array(parameters['start_position_m'])-origin))[3]-.2,
        'height_above_nearby_roof_m':parameters['start_position_m'][2]-6.36610282696,
        'source':'Complete actual v3 visible geometry; v4 partitions preserve the same surfaces/elevations. Not a dynamics/human-flight certification.',
        'manual_route_required':False,'mileage_required':False}
    (OUT/'route_geometry_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
