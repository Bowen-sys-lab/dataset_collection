import unittest
from scripts import collection_episode as episodes


class EpisodeTests(unittest.TestCase):
    def test_synthetic_probes_use_separate_source_and_storage(self):
        row=episodes.resolve_episode('factory','F01')
        manual=episodes.recording_identity(row,'stamp',0)
        probe=episodes.recording_identity(row,'stamp',20)
        self.assertEqual(manual['trajectory_source'],'manual_driving')
        self.assertEqual(probe['trajectory_source'],'synthetic_control_probe')
        self.assertIn('diagnostics',str(probe['root']))
        self.assertNotEqual(manual['root'],probe['root'])
        for frames in (-1,121,1.5):
            with self.assertRaises(ValueError):episodes.recording_identity(row,'stamp',frames)
    def test_twenty_unique_english_episodes_with_group_splits(self):
        rows = episodes.load_episodes()
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row['episode'] for row in rows}), 20)
        for scene in ('factory','forest'):
            selected = [row for row in rows if row['scene'] == scene]
            self.assertEqual([sum(row['split']==split for row in selected)
                              for split in ('train','validation','test')], [6,2,2])
            for group in ('A','B','D','E','F'):
                pair = [row for row in selected if row['start_group']==group]
                self.assertEqual(len(pair), 2)
                self.assertEqual(pair[0]['candidate_position_m'],pair[1]['candidate_position_m'])
                self.assertEqual(pair[0]['candidate_yaw_degrees'],pair[1]['candidate_yaw_degrees'])
            self.assertTrue(all(row['language']=='en' and row['instruction'].isascii() for row in selected))

    def test_unknown_and_wrong_scene_are_rejected(self):
        for scene,identifier in [('city','F01'),('forest','F01'),('factory','../../F01')]:
            with self.assertRaises(ValueError):episodes.resolve_episode(scene,identifier)

    def test_unverified_drafts_cannot_launch_but_can_be_inspected(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder, patch.object(episodes,'READINESS',Path(folder)/'absent.json'):
            row = episodes.resolve_episode('factory','F01')
            self.assertFalse(row['enabled'])
            with self.assertRaisesRegex(ValueError,'readiness'):
                episodes.resolve_episode('factory','F01',require_ready=True)

    def test_resolved_task_removes_old_start_notes_and_sets_actual_instruction(self):
        for scene,identifier in [('factory','F01'),('forest','R01')]:
            row = episodes.resolve_episode(scene,identifier)
            task = episodes.task_snapshot(row)
            self.assertEqual(task['instruction'],row['instruction'])
            self.assertEqual(task['start_position'],row['candidate_position_m'])
            self.assertNotIn('50',task['notes'])
            self.assertEqual(task['environment_evidence']['collection_episode']['split'],row['split'])

    def test_runtime_snapshot_has_absolute_paths_and_no_preflight_writes(self):
        from pathlib import Path
        import tempfile
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            location=Path(folder)/'runtime'
            for scene,key in [('factory','F01'),('forest','R01')]:
                row=episodes.resolve_episode(scene,key)
                config,task=episodes.runtime_snapshot(row,location)
                data=yaml.safe_load(config)
                self.assertTrue(Path(data['scene']['stage']).is_absolute())
                self.assertEqual(data['task']['file'],str(location/'task.yaml'))
                self.assertEqual(yaml.safe_load(task)['instruction'],row['instruction'])
            self.assertFalse(location.exists())

    def test_factory_episode_has_no_contradictory_legacy_split_or_variant(self):
        row=episodes.resolve_episode('factory','F09')
        task=episodes.task_snapshot(row)
        inherited=task['environment_evidence'].get('bc_pilot')
        self.assertTrue(inherited is None or all(inherited.get(key)==row[key]
            for key in ('start_group','split','instruction_variant')))

    def test_arbitrary_hashed_files_are_not_native_readiness(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        key=episodes.MANIFEST.relative_to(episodes.ROOT).as_posix()
        digest=episodes.sha256(episodes.MANIFEST)
        with tempfile.TemporaryDirectory() as folder:
            proof=Path(folder)/'proof.json'
            proof.write_text(json.dumps({'episodes':{'F07':{'enabled':True,
                'manifest_sha256':digest,'input_sha256':{key:digest},
                'native_evidence_sha256':{key:digest}}}}))
            with patch.object(episodes,'READINESS',proof):
                self.assertFalse(episodes.resolve_episode('factory','F07')['enabled'])

    def test_native_readiness_accepts_bound_report_and_rejects_changed_inputs(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=episodes.ROOT/'validation') as folder:
            root=Path(folder);source=root/'input.txt';source.write_text('checked')
            report=root/'native.json';proof=root/'proof.json'
            key=source.relative_to(episodes.ROOT).as_posix()
            report_key=report.relative_to(episodes.ROOT).as_posix()
            with patch.object(episodes,'READINESS',proof), patch.object(episodes,'required_input_paths',return_value={key}):
                row=episodes.resolve_episode('factory','F01')
                inputs={key:episodes.sha256(source)}
                result=dict(schema_version=1,result='PASS',scene='factory',episodes=['F01'],
                    source_manifest_sha256=row['manifest_sha256'],source_inputs=inputs,
                    spawn_passed=True,route_clearance_passed=True,view_passed=True,long_replay_passed=True,
                    position_m=row['candidate_position_m'],yaw_degrees=row['candidate_yaw_degrees'])
                report.write_text(json.dumps(result))
                proof.write_text(json.dumps({'episodes':{'F01':dict(enabled=True,
                    manifest_sha256=row['manifest_sha256'],input_sha256=inputs,native_report=report_key,
                    native_evidence_sha256={report_key:episodes.sha256(report)})}}))
                self.assertTrue(episodes.resolve_episode('factory','F01',require_ready=True)['enabled'])
                source.write_text('changed after check')
                with self.assertRaisesRegex(ValueError,'readiness'):
                    episodes.resolve_episode('factory','F01',require_ready=True)


if __name__=='__main__':unittest.main()
