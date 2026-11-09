"""Fail-closed checks for diagnostic views of physically hidden factory targets."""
import math
import json
import hashlib
from pathlib import Path
import numpy as np

TARGET_MARKERS={'tank':'gc075','pipe':'gc089'}


def _labels(labels):
    if not isinstance(labels,dict) or not labels:raise ValueError('Missing renderer labels')
    result={}
    for key,value in labels.items():
        if not isinstance(value,str) or not value.startswith('/'):raise ValueError('Invalid prim label')
        result[int(key)]=value
    return result


def _plane(ids,labels):
    plane=np.asarray(ids);mapping=_labels(labels)
    if plane.ndim!=2 or not plane.size or not np.issubdtype(plane.dtype,np.integer):
        raise ValueError('Expected a nonempty integer instance plane')
    # Actual world geometry is a positive render sanity check; empty/sky frames cannot prove hiding.
    world=[key for key,path in mapping.items() if path.startswith('/World/') and key>1]
    if np.isin(plane,world).sum()<20:raise ValueError('No populated world geometry in instance plane')
    unknown=set(int(v) for v in np.unique(plane) if v>1)-set(mapping)
    if unknown:raise ValueError('Instance pixels lack prim labels')
    return plane,mapping


def target_pixel_counts(instance_ids,labels):
    plane=np.asarray(instance_ids);mapping=_labels(labels)
    if plane.ndim!=2 or not plane.size or not np.issubdtype(plane.dtype,np.integer):
        raise ValueError('Expected a nonempty integer instance plane')
    return {target:int(np.isin(plane,[key for key,path in mapping.items() if marker in path.lower()]).sum())
            for target,marker in TARGET_MARKERS.items()}


def validate_views(views,expected_position,*,expected_yaw_degrees=0.):
    """Three populated views, both full authored source groups hidden, real foreground blockers.

    Caller binds files/hashes, source geometry and native body/camera readbacks. Source-group
    markers deliberately include normal equipment surfaces rather than only hotspot paths.
    """
    try:
        if not isinstance(views,list) or len(views)!=3:return False
        by_kind={v['kind']:v for v in views}
        if set(by_kind)!={'initial','aim_tank','aim_pipe'}:return False
        expected=np.asarray(expected_position,dtype=float)
        yaw_expected=float(expected_yaw_degrees)
        if expected.shape!=(3,) or not np.isfinite(expected).all() or not math.isfinite(yaw_expected):return False
        for kind,view in by_kind.items():
            position=np.asarray(view['body_position_m'],dtype=float)
            if position.shape!=(3,) or not np.isfinite(position).all() or np.linalg.norm(position-expected)>.25:return False
            matrix=np.asarray(view['body_world_matrix'],dtype=float)
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all():return False
            if not np.allclose(matrix[:,3],[0.,0.,0.,1.],rtol=0.,atol=1e-6) or np.linalg.norm(matrix[3,:3]-position)>1e-4:return False
            scale=np.linalg.norm(matrix[:3,:3],axis=1)
            if np.any(scale<=1e-12):return False
            rotation=matrix[:3,:3]/scale[:,None]
            if not np.allclose(rotation@rotation.T,np.eye(3),rtol=0.,atol=1e-5) or np.linalg.det(rotation)<.999:return False
            if kind=='initial':
                actual=math.degrees(math.atan2(rotation[0,1],rotation[0,0]));delta=(actual-yaw_expected+180.)%360.-180.
                pitch=math.degrees(math.asin(float(np.clip(-rotation[0,2],-1.,1.))))
                roll=math.degrees(math.atan2(rotation[1,2],rotation[2,2]))
                if abs(delta)>1. or abs(pitch)>1. or abs(roll)>1.:return False
            for stream in ('rgb','ir'):
                plane,_=_plane(view[stream+'_ids'],view[stream+'_labels'])
                if any(target_pixel_counts(plane,view[stream+'_labels']).values()):return False
            if kind=='initial':continue
            projected=float(view['projected_target_pixels'])
            if not math.isfinite(projected) or projected<20:return False
            blockers=view['ray_blockers']
            if not isinstance(blockers,list) or len(blockers)<3:return False
            for hit in blockers:
                path=hit['path'];distance=float(hit['distance_m']);target_distance=float(hit['target_distance_m'])
                if not isinstance(path,str) or not path.startswith('/World/'):return False
                if any(marker in path.lower() for marker in TARGET_MARKERS.values()):return False
                if not math.isfinite(distance) or not math.isfinite(target_distance) or not 0<distance<target_distance-.2:return False
        return True
    except (ValueError,TypeError,KeyError,OverflowError):return False


def validate_file_evidence(report_path,row,root,evidence_hashes,*,expected_source_inputs):
    """Load only hash-bound native artifacts inside the project and recheck their content."""
    try:
        from PIL import Image
        root=Path(root).resolve()
        def bound(value):
            if not isinstance(value,str) or not value:return None
            path=(root/value).resolve()
            if not path.is_relative_to(root) or not path.is_file():return None
            key=path.relative_to(root).as_posix()
            if key not in evidence_hashes or hashlib.sha256(path.read_bytes()).hexdigest()!=evidence_hashes[key]:return None
            return path
        path=bound(report_path)
        if path is None:return False
        report=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(expected_source_inputs,dict) or not expected_source_inputs or report.get('source_inputs')!=expected_source_inputs:return False
        if (report.get('schema_version')!=1 or report.get('scene')!='factory'
            or report.get('group')!=row['start_group'] or report.get('start_version')!=row['start_version']
            or report.get('source_manifest_sha256')!=row['manifest_sha256']):return False
        views=[]
        for original in report['views']:
            view=dict(original)
            for stream in ('rgb','ir'):
                paths={kind:bound(view[stream+'_'+kind]) for kind in ('ids','labels','image')}
                if any(p is None for p in paths.values()):return False
                view[stream+'_ids']=np.load(paths['ids'],allow_pickle=False)
                view[stream+'_labels']=json.loads(paths['labels'].read_text(encoding='utf-8'))
                with Image.open(paths['image']) as picture:
                    picture.load()
                    if picture.size!=tuple(view[stream+'_ids'].shape[::-1]):return False
            views.append(view)
        return validate_views(views,row['candidate_position_m'],expected_yaw_degrees=row['candidate_yaw_degrees'])
    except (ValueError,TypeError,KeyError,OverflowError,OSError):return False


def validate_route_file_evidence(report_path,row,root,evidence_hashes,hotspot_paths,*,expected_source_inputs):
    """Bound keyframes must show a hidden view, equipment emergence and actual hotspot at endpoint."""
    try:
        from PIL import Image
        root=Path(root).resolve()
        def bound(value):
            if not isinstance(value,str) or not value:raise ValueError('Missing evidence path')
            path=(root/value).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise ValueError('Evidence outside root or missing')
            key=path.relative_to(root).as_posix()
            if key not in evidence_hashes or hashlib.sha256(path.read_bytes()).hexdigest()!=evidence_hashes[key]:
                raise ValueError('Unbound or changed keyframe evidence')
            return path
        report=json.loads(bound(report_path).read_text(encoding='utf-8'))
        if not isinstance(expected_source_inputs,dict) or not expected_source_inputs or report.get('source_inputs')!=expected_source_inputs:return False
        if (report.get('schema_version')!=1 or report.get('scene')!='factory'
            or report.get('group')!=row['start_group'] or report.get('start_version')!=row['start_version']
            or report.get('target')!=row['target'] or report.get('source_manifest_sha256')!=row['manifest_sha256']
            or report.get('reference_waypoints')!=row['reference_waypoints']):return False
        if not isinstance(hotspot_paths,list) or not hotspot_paths or any(not isinstance(p,str) or not p.startswith('/') for p in hotspot_paths):return False
        views=report['views']
        if not isinstance(views,list) or len(views)!=3 or {v['kind'] for v in views}!={'before_corner','appearance','end'}:return False
        waypoints=np.asarray(row['reference_waypoints'],dtype=float)
        if waypoints.ndim!=2 or waypoints.shape[1]!=3 or len(waypoints)<2 or not np.isfinite(waypoints).all():return False
        edge_lengths=np.linalg.norm(np.diff(waypoints,axis=0),axis=1)
        cumulative=np.r_[0.,np.cumsum(edge_lengths)];progress={}
        for view in views:
            point=np.asarray(view['body_position_m'],dtype=float)
            if point.shape!=(3,) or not np.isfinite(point).all():return False
            edge=np.diff(waypoints,axis=0);denom=np.einsum('ij,ij->i',edge,edge)
            along=np.divide(np.einsum('ij,ij->i',point-waypoints[:-1],edge),denom,out=np.zeros(len(edge)),where=denom>1e-12)
            nearest=waypoints[:-1]+np.clip(along,0.,1.)[:,None]*edge
            errors=np.linalg.norm(point-nearest,axis=1);segment=int(np.argmin(errors))
            if errors[segment]>.25:return False
            progress[view['kind']]=float(cumulative[segment]+np.clip(along[segment],0.,1.)*edge_lengths[segment])
            if view['kind']=='end' and np.linalg.norm(point-waypoints[-1])>.25:return False
            if view['kind']=='before_corner':
                area=float(view['projected_target_pixels'])
                if not math.isfinite(area) or area<20:return False
            for stream in ('rgb','ir'):
                ids=np.load(bound(view[stream+'_ids']),allow_pickle=False)
                labels=json.loads(bound(view[stream+'_labels']).read_text(encoding='utf-8'))
                plane,mapping=_plane(ids,labels)
                with Image.open(bound(view[stream+'_image'])) as picture:
                    picture.load()
                    if picture.size!=tuple(plane.shape[::-1]):return False
                count=target_pixel_counts(plane,labels)[row['target']]
                if view['kind']=='before_corner' and count!=0:return False
                if view['kind'] in ('appearance','end') and count<=0:return False
                if view['kind']=='end':
                    ids_hot=[key for key,path in mapping.items() if path in hotspot_paths]
                    if int(np.isin(plane,ids_hot).sum())<20:return False
        return progress['before_corner']<=progress['appearance']+1e-6 and progress['appearance']<=progress['end']+1e-6
    except (ValueError,TypeError,KeyError,OverflowError,OSError):return False
