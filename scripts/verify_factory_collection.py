"""Read actual composed v7m USD; export measurement geometry and spawn evidence.

No Kit, no USD authoring. Surface checks are sampled, not solid/collision proofs.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
USD_LIB=Path('D:/isaacsim/extscache/omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312')
sys.path.insert(0,str(USD_LIB))
from scripts.factory_collection_config import PLAN, READINESS, resolve_selection, digest, runtime_dependency_hashes
from scripts.evaluate_factory_inspection import point_to_mesh_distances


def mesh_triangles(prim, cache):
    from pxr import UsdGeom
    mesh=UsdGeom.Mesh(prim)
    p=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    if not len(p): return np.empty((0,3,3))
    p=(np.c_[p,np.ones(len(p))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:,:3]
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=int)
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int)
    if np.all(counts==3): return p[indices.reshape(-1,3)]
    triangles=[]; offset=0
    for count in counts:
        face=indices[offset:offset+count]; offset+=count
        for i in range(1,count-1): triangles.append(p[[face[0],face[i],face[i+1]]])
    return np.asarray(triangles).reshape(-1,3,3)


def distance(point, triangles):
    return float(point_to_mesh_distances(np.asarray(point)[None,:], triangles.reshape(-1,3),
        np.arange(triangles.size//3).reshape(-1,3), body_radius_m=.2)[0])


def main():
    import yaml
    from pxr import Usd, UsdGeom
    out=READINESS.parent; out.mkdir(parents=True,exist_ok=True)
    stage_path=ROOT/'factory/model02_thermal_v7m/model02_v7m_flight.usda'
    print('Opening composed v7m stage (read-only)...',flush=True)
    stage=Usd.Stage.Open(str(stage_path))
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-9: raise ValueError('audit assumes stage meters')
    cache=UsdGeom.XformCache(); bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
    plan=yaml.safe_load(PLAN.read_text(encoding='utf-8'))
    report={'inputs':{},'targets':{},'starts':{},'combinations':{},
            'scope':'actual visible composed USD mesh surface samples; no whole-scene continuous collision certification'}
    probes={g['group']:g['position_m'] for g in plan['start_groups']}
    for target in ('tank','pipe'):
        selected=resolve_selection(target,1,'A',require_ready=False)
        inspection=selected['task']['environment_evidence']['factory_inspection']
        equipment=stage.GetPrimAtPath(inspection['equipment_prim_path'])
        paths=inspection.get('hotspot_prim_paths',[inspection['hotspot_prim_path']])
        if not equipment or any(not stage.GetPrimAtPath(p) for p in paths):
            raise ValueError(f'{target}: equipment or hotspot missing in composed v7m stage')
        triangles=np.concatenate([mesh_triangles(p,cache) for p in Usd.PrimRange(equipment) if p.IsA(UsdGeom.Mesh)])
        path=out/f'{target}_v7m_geometry.npz'
        np.savez_compressed(path,equipment_vertices=triangles.reshape(-1,3),
                            equipment_triangles=np.arange(triangles.size//3).reshape(-1,3))
        hot=np.concatenate([mesh_triangles(stage.GetPrimAtPath(p),cache) for p in paths])
        areas=np.linalg.norm(np.cross(hot[:,1]-hot[:,0],hot[:,2]-hot[:,0]),axis=1)/2
        center=np.average(hot.mean(1),weights=areas,axis=0)
        # The recorded task center covers a core patch, possibly a subset of all visual facets.
        declared=np.asarray(inspection['hotspot_position_m'])
        nearest=distance(declared,hot)+.2
        if nearest>.05: raise ValueError(f'{target} declared target not on actual hotspot: {nearest}')
        observation=inspection['observation_position']
        probes[f'{target}_observation']=observation
        report['targets'][target]={'geometry_file':path.relative_to(ROOT).as_posix(),
            'geometry_sha256':digest(path),'triangles':len(triangles),
            'hotspot_paths_verified':len(paths),'hotspot_centroid_m':center.tolist(),
            'declared_target_surface_error_m':nearest,
            'observation_equipment_clearance_m':distance(observation,triangles)}
    near={name:[] for name in probes}
    positions=np.asarray(list(probes.values()))
    print('Checking spawn and observation surfaces against actual composed meshes...',flush=True)
    meshes=0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()).startswith('/World/Crazyflie'): continue
        if UsdGeom.Imageable(prim).ComputeVisibility()=='invisible': continue
        bound=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if bound.IsEmpty(): continue
        lo=np.asarray(bound.GetMin()); hi=np.asarray(bound.GetMax())
        # Retain all meshes whose XY extent covers any probe within 3m; no centroid cull.
        candidates=np.all(lo[:2]<=positions[:,:2]+3,axis=1)&np.all(hi[:2]>=positions[:,:2]-3,axis=1)
        if not candidates.any(): continue
        triangles=mesh_triangles(prim,cache); meshes+=1
        for i,name in enumerate(probes):
            if candidates[i]: near[name].append(triangles)
    for name, position in probes.items():
        tri=np.concatenate(near[name]); p=np.asarray(position)
        lo=tri.min(1); hi=tri.max(1)
        keep=np.all(lo<=p+3,axis=1)&np.all(hi>=p-3,axis=1)
        clearance=min(2.8,distance(p,tri[keep])) if keep.any() else 2.8
        a=tri[:,0]; ab=tri[:,1]-a; ac=tri[:,2]-a; ap=p-a
        det=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]; valid=np.abs(det)>1e-10
        u=np.divide(ap[:,0]*ac[:,1]-ap[:,1]*ac[:,0],det,out=np.full(len(det),-1.),where=valid)
        v=np.divide(ab[:,0]*ap[:,1]-ab[:,1]*ap[:,0],det,out=np.full(len(det),-1.),where=valid)
        z=a[:,2]+u*ab[:,2]+v*ac[:,2]
        below=valid&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(z<p[2]-1e-7)
        agl=float(p[2]-z[below].max()) if below.any() else None
        report['starts'][name]={'position_m':position,'surface_clearance_lower_bound_m':clearance,
            'height_above_surface_m':agl,'passed':clearance>=.5 and agl is not None and agl>=2.5}
    for target in ('tank','pipe'):
        good=report['starts'][target+'_observation']['passed'] and 6<=report['targets'][target]['observation_equipment_clearance_m']<=8
        for group in plan['start_groups']:
            report['combinations'][f'{target}:{group["group"]}']={
                'enabled':False,
                'geometry_passed':bool(good and report['starts'][group['group']]['passed']),
                'basis':'spawn/observation mesh-surface clearance and AGL; camera view checked separately',
                'continuous_route_collision_verified':False}
    inputs=[PLAN,ROOT/'configs/model02_v7m_bc_pilot_scene.yaml',
            ROOT/'configs/model02_v7m_bc_pilot_task.yaml',ROOT/'configs/model02_v3_pipe_task.yaml']
    inputs += [Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_relative_to(ROOT)]
    for path in inputs: report['inputs'][path.relative_to(ROOT).as_posix()]=digest(path)
    report['runtime_dependencies']=runtime_dependency_hashes()
    report['inputs'].update(report['runtime_dependencies'])
    report['source_inputs']=dict(report['inputs'])
    native_path=out/'native_checks.json'
    if native_path.exists():
        native=json.loads(native_path.read_text(encoding='utf-8'))
        if native.get('source_inputs')==report['source_inputs']:
            proof_files=native.get('evidence_hashes',{})
            valid=all((ROOT/p).is_file() and digest(ROOT/p)==h for p,h in proof_files.items())
            for target in ('tank','pipe'):
                for group in plan['start_groups']:
                    key=f'{target}:{group["group"]}'
                    report['combinations'][key]['enabled']=bool(valid and report['combinations'][key]['geometry_passed']
                        and native['targets'].get(target,{}).get('passed')
                        and native['starts'].get(group['group'],{}).get('passed'))
            report['inputs'].update(proof_files)
            report['inputs'][native_path.relative_to(ROOT).as_posix()]=digest(native_path)
    report['meshes_examined_near_probes']=meshes
    READINESS.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'targets':report['targets'],'starts':report['starts'],'combinations':report['combinations']},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
