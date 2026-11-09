"""Headless three-scene synthetic obstacle/hold/undo/resume diagnostic; never teacher data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scene', choices=('city','factory','forest'), required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False})
    controller=None
    report={'result':'FAIL','trajectory_source':'synthetic_collision_undo_diagnostic',
            'independent_manual_flights':0}
    try:
        import numpy as np
        import omni.usd
        import omni.timeline
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        from scripts.manual_collision_undo import prepare_guard,RecordingRecovery
        from scripts.drone_keyboard_controller import DroneKeyboardController
        from scripts.trajectory_recorder import TrajectoryRecorder,trajectory_state_from_world_transform
        from scripts.drone_control_math import orientation_wxyz_from_rpy
        from scripts.scene_config import load_scene_config
        from scripts.recording_undo import active_branch
        from scripts.replay_factory_inspection import write_terminal_replay_source
        config=load_scene_config(ROOT/'configs'/dict(city='city_200m_v3_scene.yaml',factory='model02_v7m_bc_pilot_scene.yaml',forest='forest_pilot_two_target_scene.yaml')[args.scene],include_task=False)
        report['scene']=args.scene
        context=omni.usd.get_context()
        assert context.open_stage(str(config.scene.stage))
        for _ in range(8):app.update()
        stage=context.get_stage()
        source_hashes={layer.realPath:hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                       for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
        guard,guard_metadata=prepare_guard(stage,config.drone.prim_path,config.drone.body_path,
            exclude_paths=config.scene.collision_exclude_paths)
        report['guard']=guard_metadata
        print('[probe] collision meshes:',guard_metadata['mesh_count'],'radius:',guard.radius,flush=True)
        # Known diagnostic wall, authored only into the runtime session.
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            wall=UsdGeom.Cube.Define(stage,'/World/CollisionUndoDiagnosticWall')
            wall.CreateSizeAttr(1.)
            wall.AddTranslateOp().Set(Gf.Vec3d(9980.,10000.,50.))
            wall.AddScaleOp().Set(Gf.Vec3f(.2,6.,16.))
            UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
        controller=DroneKeyboardController(prim_path=config.drone.prim_path,minimum_height=1.5).start()
        controller.initial_pose=((10000.,10000.,50.),orientation_wxyz_from_rpy(0.,0.,math.pi))
        controller._has_control_focus=lambda:True
        timeline=omni.timeline.get_timeline_interface()
        from scripts.start_crazyflie_dual_camera_view import configure_recording_timeline
        configure_recording_timeline(timeline)
        sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        r=TrajectoryRecorder.create(args.out/'raw',{
            'trajectory_source':report['trajectory_source'],'sample_period_s':1.,
            'scene_config_path':str(config.source_path),'scene_config_sha256':config.sha256,
            'stage_path':str(config.scene.stage),'stage_sha256':sha(config.scene.stage),
            'sensor_config_sha256':sha(config.sensors.sensor_config),
            'thermal_config_sha256':sha(config.sensors.thermal_config),
            'collision_guard':guard_metadata},run_id=f'synthetic-{args.scene}-collision-undo')
        recovery=RecordingRecovery(r,timeline.get_current_time,guard)
        controller.collision_guard=guard
        controller.collision_callback=recovery.collision
        controller.undo_callback=recovery.prepare_undo
        controller.undo_committed_callback=recovery.commit_undo
        timeline.play()
        for _ in range(180):
            app.update()
            if controller._ready:break
        assert controller._ready and controller._running,controller.diagnostics()
        # Test the current scene's real starts, plus real ground geometry.
        from scripts.collection_episode import load_episodes
        if args.scene == 'city':
            starts=json.loads((ROOT/'configs/collection_episodes/city_v2.json').read_text(encoding='utf-8'))['episodes']
            table=json.loads((ROOT/'configs/city_200m_v2_task_table.json').read_text(encoding='utf-8'))['tasks']
            import yaml
            positions={r['episode']:yaml.safe_load(Path(r['file']).read_text(encoding='utf-8'))['start_position'] for r in table}
        else:
            positions={r['episode']:r['candidate_position_m'] for r in load_episodes() if r['scene']==args.scene}
        report['spawn_queries']={name:guard(position,position) for name,position in positions.items()}
        assert all(hit is None for hit in report['spawn_queries'].values()),report['spawn_queries']
        position=next(iter(positions.values()))
        ground=guard(position,(position[0],position[1],position[2]-200.))
        assert ground and ground['object_prim_path']!='/World/CollisionUndoDiagnosticWall',ground
        report['actual_scene_ground_sweep']=ground
        next_sample=-math.inf
        camera=config.drone.camera_path
        def update_and_record():
            nonlocal next_sample
            app.update()
            assert controller._running,controller.diagnostics()
            stamp=float(timeline.get_current_time())
            if recovery.can_sample(stamp) and stamp>=next_sample-1e-8:
                matrix=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.body_path)).ComputeLocalToWorldTransform(0)).tolist()
                camera_matrix=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(camera)).ComputeLocalToWorldTransform(0)).tolist()
                row=trajectory_state_from_world_transform(controller.state_snapshot(),matrix)
                r.append(dict(row,frame_id=len(r.records),simulation_time=stamp,
                    rgb_camera_world_matrix=camera_matrix,thermal_camera_world_matrix=camera_matrix))
                next_sample=stamp+1.
        update_and_record()
        controller.keys.press('W')
        for _ in range(1600):
            update_and_record()
            if controller.collision_held:break
        assert controller.collision_held,'Wall crossing did not hold'
        assert r.metadata['collection_edit_events'][-1]['detail']['object_prim_path']=='/World/CollisionUndoDiagnosticWall'
        held=controller.position
        for _ in range(30):update_and_record()
        assert controller.position==held and not controller.keys.pressed
        assert not controller._mark_task_success()
        report['held_position']=held
        assert controller.undo_collision()
        restored=controller.position
        assert restored[0]>held[0]+3.,(restored,held)
        report['restored_position']=restored
        assert not controller.keys.pressed
        controller.keys.press('S')
        resume_start=float(timeline.get_current_time())
        for _ in range(480):
            update_and_record()
            if timeline.get_current_time()-resume_start>=2.1:break
        controller._clear_motion()
        assert controller.position[0]>restored[0]+1.,(controller.position,restored)
        report['resumed_position']=controller.position
        # A second collision/undo exercises an already corrected branch.
        controller.keys.press('W')
        for _ in range(1600):
            update_and_record()
            if controller.collision_held:break
        assert controller.collision_held
        assert controller.undo_collision()
        for _ in range(80):update_and_record()
        original=r.close()
        branch,_=active_branch(r.records,r.metadata)
        assert all(b['simulation_time']>a['simulation_time'] for a,b in zip(branch,branch[1:]))
        assert all(math.dist(a['position'],b['position'])<=3.5 for a,b in zip(branch,branch[1:]))
        report.update(raw_records=len(r.records),corrected_records=len(branch),edit_events=r.metadata['collection_edit_events'])
        # Sparse rendering input, explicitly synthetic and without a fabricated human T label.
        selected=[branch[0],branch[len(branch)//2],branch[-1]]
        for index,row in enumerate(selected):
            row['source_frame_id']=row['frame_id'];row['frame_id']=index
        derived=write_terminal_replay_source(args.out/'render_input',original,selected)
        meta_path=derived.parent/'run_metadata.json'
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        meta.update(collection_branch_applied=True,source_derivation_schema_version=1,
                    derivation='synthetic collision/undo diagnostic; not human teacher data')
        meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        report['render_input']=str(derived)
        report['source_layers_unchanged']=all(sha(path)==digest for path,digest in source_hashes.items())
        assert report['source_layers_unchanged']
        from scripts.recording_quality import recording_quality
        report['technical_quality']=recording_quality(r.records,sample_period_s=1.,metadata=r.metadata)
        assert report['technical_quality']['result']=='PASS',report['technical_quality']
        report['result']='PASS'
    except Exception:
        report['error']=traceback.format_exc()
        print(report['error'],flush=True)
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if controller:controller.stop()
        app.close()
    return 0 if report['result']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
