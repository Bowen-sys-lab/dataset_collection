"""Measure the reference route against the scanned factory triangle geometry."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from scripts.scene_config import load_scene_config
    from scripts.evaluate_factory_inspection import point_to_mesh_distances
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/factory_inspection_scene.yaml')
    parser.add_argument('--geometry',type=Path,default=ROOT/'validation/factory-inspection/factory_geometry.npz')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    config=load_scene_config(args.config)
    data=config.task['environment_evidence']['factory_inspection']
    waypoints=np.asarray(data['reference_waypoints_m'])
    samples=[]
    for a,b in zip(waypoints,waypoints[1:]):
        count=int(np.ceil(np.linalg.norm(b-a)/.5))
        samples.extend(a+(b-a)*t/count for t in range(count))
    samples=np.asarray([*samples,waypoints[-1]])
    geometry=np.load(args.geometry,allow_pickle=False)
    triangles=geometry['triangles'].astype('float64')
    lo=triangles.min(1);hi=triangles.max(1)
    # Per-sample exact triangle distances, conservatively retaining every
    # triangle whose AABB can fall within 10m of that sample. No centroid cull.
    clearances=[];ground=[]
    for position in samples:
        near=np.all(lo[:,:2]<=position[:2]+10,axis=1)&np.all(hi[:,:2]>=position[:2]-10,axis=1)
        tri=triangles[near]
        distance=point_to_mesh_distances(position[None,:],tri.reshape(-1,3),
            np.arange(len(tri)*3).reshape(-1,3),body_radius_m=.2)[0]
        clearances.append(float(distance))
        xy=tri[:,:,:2];a=xy[:,0];ab=xy[:,1]-a;ac=xy[:,2]-a;ap=position[:2]-a
        det=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
        valid=abs(det)>1e-10
        u=np.divide(ap[:,0]*ac[:,1]-ap[:,1]*ac[:,0],det,out=np.full(len(det),-1.),where=valid)
        v=np.divide(ab[:,0]*ap[:,1]-ab[:,1]*ap[:,0],det,out=np.full(len(det),-1.),where=valid)
        z=tri[:,0,2]+u*(tri[:,1,2]-tri[:,0,2])+v*(tri[:,2,2]-tri[:,0,2])
        below=valid&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(z<position[2])
        ground.append(float(z[below].max()) if below.any() else None)
    agl=[float(p[2]-z) if z is not None else None for p,z in zip(samples,ground)]
    waypoint_metrics=[]
    for name,position in zip(data['reference_waypoint_names'],waypoints):
        index=int(np.argmin(np.linalg.norm(samples-position,axis=1)))
        waypoint_metrics.append({'name':name,'position_m':position.tolist(),
            'highest_surface_below_z_m':ground[index],
            'height_above_surface_m':agl[index],
            'all_geometry_clearance_m':clearances[index]})
    with np.load(ROOT/data['geometry_file'],allow_pickle=False) as equipment:
        observation_distance=float(point_to_mesh_distances(
            np.asarray(data['observation_position'])[None,:],
            equipment['equipment_vertices'],equipment['equipment_triangles'],body_radius_m=.2)[0])
    delta=np.diff(waypoints,axis=0)
    flight=data['flight_settings']
    flight_time=float(np.maximum(np.linalg.norm(delta[:,:2],axis=1)/flight['horizontal_speed_m_s'],
        abs(delta[:,2])/flight['vertical_speed_m_s']).sum())
    report={'reference_path_length_m':float(np.linalg.norm(np.diff(waypoints,axis=0),axis=1).sum()),
        'sample_spacing_max_m':.5,'samples':len(samples),'body_envelope_radius_m':.2,
        'minimum_all_geometry_clearance_m':min(clearances),
        'minimum_agl_m':min(x for x in agl if x is not None),
        'ground_world_z_range_m':[min(x for x in ground if x is not None),max(x for x in ground if x is not None)],
        'minimum_clearance_position_m':samples[int(np.argmin(clearances))].tolist(),
        'observation_equipment_clearance_m':observation_distance,
        'waypoints':waypoint_metrics,
        'estimated_flight_time_s':flight_time,
        'estimated_time_with_hold_s':flight_time+data['inspection_requirements']['hold_duration_s'],
        'geometry_sha256':hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        'stage_sha256':hashlib.sha256(config.scene.stage.read_bytes()).hexdigest(),
        'limitation':'0.5m discrete exact triangle-distance samples; no automatic avoidance or continuous swept collision certificate'}
    report['result']='PASS' if 180<=report['reference_path_length_m']<=220 and min(clearances)>=1 and report['minimum_agl_m']>=2.5 and 6<=observation_distance<=8 else 'FAIL'
    (args.out/'route_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(args.out/'route_samples.npz',positions=samples,clearances=clearances,ground_z=ground)
    print(json.dumps(report),flush=True)
    return 0 if report['result']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
