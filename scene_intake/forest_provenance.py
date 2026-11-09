"""Forest replay preflight built on the project's original recording contracts."""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import numpy as np
from scripts.scene_config import load_scene_config
from scripts.trajectory_recorder import read_trajectory,trajectory_provenance


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(value,field):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError(f'missing or invalid recorded identity: {field}')
    return value


def _file(path):
    return None if path is None else dict(path=str(Path(path).resolve()),sha256=_sha(path))


def validate_recording_source(config,trajectory,*,thermal_inputs=None):
    """Fail before Kit starts. Display-only changes are recorded, not disguised.

    Historical labels come from run_metadata, including a missing task. This
    validates recorded root-stage and thermal identity; it cannot invent an
    asset-tree snapshot that an old recording never stored.
    """
    from scripts.runtime_parameter_provenance import thermal_input_snapshot,validate_thermal_inputs
    trajectory=Path(trajectory).resolve()
    prov=trajectory_provenance(trajectory)
    if not prov.get('source_run_metadata_path'):raise ValueError('recorded run_metadata.json is required')
    stage_hash=_hash(prov.get('recording_stage_sha256'),'stage')
    if _sha(config.scene.stage)!=stage_hash:raise ValueError('recorded source USD differs from replay stage')
    source_path=prov.get('recording_scene_config_path')
    if not source_path or not Path(source_path).is_file():raise ValueError('recorded scene configuration is unavailable')
    if _sha(source_path)!=_hash(prov.get('recording_scene_config_sha256'),'scene configuration'):
        raise ValueError('recorded scene configuration changed')
    source=load_scene_config(source_path,include_task=False)
    if source.scene.stage.resolve()!=config.scene.stage.resolve():raise ValueError('replay uses a different stage path')
    for key in ('prim_path','body_path','camera_path'):
        if getattr(source.drone,key)!=getattr(config.drone,key):raise ValueError(f'replay camera rig mismatch: {key}')
    for field,path in [('recording_sensor_config_sha256',config.sensors.sensor_config),
                       ('recording_thermal_config_sha256',config.sensors.thermal_config)]:
        if _sha(path)!=_hash(prov.get(field),field):raise ValueError(f'replay input mismatch: {field}')
    for key in ('align_thermal_to_rgb','flat_field_enabled','sensor_seed'):
        if getattr(source.sensors,key)!=getattr(config.sensors,key):raise ValueError(f'replay sensor setting mismatch: {key}')
    recorded_inputs=prov.get('thermal_input_snapshot')
    validate_thermal_inputs(source,recorded_inputs)
    current=thermal_input_snapshot(config) if thermal_inputs is None else thermal_inputs
    def physical(snapshot,display):
        snapshot=deepcopy(snapshot)
        if display is not None:snapshot['file_sha256'].pop(str(Path(display).resolve()),None)
        return snapshot
    if physical(recorded_inputs,source.sensors.display_config)!=physical(current,config.sensors.display_config):
        raise ValueError('replay thermal/material/weather inputs differ from recording')
    records=read_trajectory(trajectory);previous=None
    for record in records:
        t=float(record['simulation_time'])
        pose=np.asarray(record.get('rgb_camera_world_matrix'),dtype=float)
        if not np.isfinite(t) or t<0 or (previous is not None and t<=previous):raise ValueError('recorded camera times must be finite and strictly increasing')
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or abs(np.linalg.det(pose))<1e-12:
            raise ValueError('recorded RGB camera pose is invalid')
        if type(record.get('frame_id')) is not int:raise ValueError('recorded frame ID must be an integer')
        previous=t
    old_display=_file(source.sensors.display_config);new_display=_file(config.sensors.display_config)
    return dict(recording=prov,
                validation=dict(status='PASS',scope='recorded_root_stage_camera_and_thermal_inputs',
                                task_metadata_status=prov['task_metadata_status'],
                                asset_tree_identity='not certified unless recorded separately'),
                render_overrides=dict(display=dict(recorded=old_display,current=new_display,changed=old_display!=new_display)))
