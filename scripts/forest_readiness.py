"""Consume current, target-specific forest geometry, native views and replay proofs."""
import json
import math
from pathlib import Path

import numpy as np
import yaml

from scripts.forest_collection_config import VERSION,target_config_path
from scripts.forest_target_evidence import target_surface_evidence


def validate_forest_evidence(row,report,root,native,expected_inputs):
    """Called after the common gate verified all file hashes in `native`."""
    def file(key):
        if key not in native:raise ValueError('Unbound forest evidence')
        path=(root/key).resolve()
        if not path.is_relative_to(root):raise ValueError('Forest proof outside project')
        return path
    def read(key):return json.loads(file(key).read_text(encoding='utf-8'))
    try:
        target=yaml.safe_load(target_config_path(row).read_text(encoding='utf-8'))['target']
        if report.get('forest_target_prim_path')!=target['prim_path']:return False
        geometry=read(report['forest_geometry_path']);audit=read(report['forest_route_audit_path'])
        if audit['geometry_sha256']!=geometry['geometry_sha256'] or native[report['forest_geometry_cache']]!=geometry['geometry_sha256']:return False
        if native[report['forest_geometry_script']]!=audit['geometry_script_sha256']:return False
        if audit['source_inputs']!=geometry['source_inputs'] or not audit['source_inputs']:return False
        for name,digest in audit['source_inputs'].items():
            rel=Path(name).resolve().relative_to(root).as_posix()
            if expected_inputs.get(rel)!=digest:return False
        if (float(audit['sample_spacing_max_m'])>.5 or float(audit['body_radius_m'])!=.2
            or float(audit['required_net_clearance_m'])<1.):return False
        routes=[r for r in audit['routes'] if r['group']==row['start_group'] and r['target']==row['target']]
        if len(routes)!=1:return False
        route=routes[0];waypoints=row['reference_waypoints']
        if (route.get('passed') is not True or route['waypoints']!=waypoints
            or waypoints[0]!=row['candidate_position_m'] or float(route['minimum_clearance_lower_bound_m'])<1.
            or not math.isclose(float(route['length_m']),float(row['reference_path_length_m']),abs_tol=1e-6)):return False
        samples=np.load(file(report['forest_route_samples']),allow_pickle=False)
        with samples:
            positions=np.asarray(samples['positions']);gaps=np.asarray(samples['clearance'])
            if (positions.shape!=(route['samples'],3) or gaps.shape!=(len(positions),) or len(positions)<2
                or not np.isfinite(positions).all() or not np.isfinite(gaps).all() or gaps.min()<1.
                or not np.allclose(positions[0],waypoints[0]) or not np.allclose(positions[-1],waypoints[-1])
                or np.linalg.norm(np.diff(positions,axis=0),axis=1).max()>.5+1e-8):return False
        views=read(report['forest_views_path'])
        if (views['version']!=VERSION or views['source_manifest_sha256']!=row['manifest_sha256']
            or views['source_inputs']!=expected_inputs):return False
        start=views['starts'][row['start_group']]
        if math.dist(start['body_position_m'],row['candidate_position_m'])>.25:return False
        end=views['endpoints'][row['target']]
        if (math.dist(end['body_position_m'],waypoints[-1])>.25 or end['target_prim_path']!=target['prim_path']
            or end['pair_validation']['time_error_s']>1e-8 or end['pair_validation']['camera_pose_max_error']>1e-5):return False
        for view in (start,end):
            for field in ('rgb_image','ir_image'):file(view[field])
        rgb=np.load(file(end['rgb_ids']),allow_pickle=False)
        rgb_labels=read(end['rgb_labels']);prefix=target['prim_path']
        if rgb.ndim!=2:return False
        rgb_ids=[int(k) for k,v in rgb_labels.items() if v==prefix or v.startswith(prefix+'/')]
        if int(np.isin(rgb,rgb_ids).sum())<20:return False
        evidence=target_surface_evidence(np.load(file(end['ir_ids']),allow_pickle=False),read(end['ir_labels']),
            np.load(file(end['truth_temperature']),allow_pickle=False),prefix,target['expected_temperatures_c'])
        if evidence['expected_temperature_pixels']<20:return False
        replay=read(report['forest_replay_proof'])
        if (replay['source_manifest_sha256']!=row['manifest_sha256'] or replay['source_inputs']!=expected_inputs
            or replay['target_prim_path']!=prefix or replay['trajectory_source']!='synthetic_control_probe'
            or replay['result']!='PASS' or replay['frame_count']<2):return False
        for key in ('recording_metadata','recording_trajectory','pipeline_status','replay_metadata','independent_validation'):
            file(replay[key])
        return True
    except (ValueError,KeyError,TypeError,OSError,AttributeError,IndexError):return False
