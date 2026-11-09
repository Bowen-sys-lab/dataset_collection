import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts import collection_episode as episodes


class OccludedReadinessTests(unittest.TestCase):
    def test_native_hash_map_cannot_override_a_stale_input_digest(self):
        with tempfile.TemporaryDirectory(dir=episodes.ROOT/'validation') as folder:
            root=Path(folder);source=root/'camera_helper.py';source.write_text('old camera')
            key=source.relative_to(episodes.ROOT).as_posix();inputs={key:episodes.sha256(source)}
            row=dict(episode='F01',scene='factory',start_group='A',start_version='pilot30_20261031',
                manifest_sha256=episodes.sha256(episodes.MANIFEST),candidate_position_m=[1.,2.,5.],candidate_yaw_degrees=0.)
            report=dict(schema_version=1,result='PASS',scene='factory',episodes=['F01'],source_inputs=inputs,
                source_manifest_sha256=row['manifest_sha256'],position_m=row['candidate_position_m'],yaw_degrees=0.,
                spawn_passed=True,route_clearance_passed=True,view_passed=True,long_replay_passed=True)
            path=root/'native.json';path.write_text(json.dumps(report));n=path.relative_to(episodes.ROOT).as_posix()
            source.write_text('new camera')
            evidence=dict(input_sha256=inputs,native_report=n,native_evidence_sha256={n:episodes.sha256(path),key:episodes.sha256(source)})
            with patch.object(episodes,'required_input_paths',return_value={key}):
                self.assertFalse(episodes.valid_readiness(row,evidence))

    def test_new_version_cannot_reuse_a_generic_native_pass(self):
        with tempfile.TemporaryDirectory(dir=episodes.ROOT/'validation') as folder:
            root=Path(folder);source=root/'source.txt';source.write_text('input')
            key=source.relative_to(episodes.ROOT).as_posix()
            row=dict(episode='F05',scene='factory',start_group='D',start_version='factory_interior_occluded_v1',
                manifest_sha256=episodes.sha256(episodes.MANIFEST),candidate_position_m=[1.,2.,5.],candidate_yaw_degrees=0.)
            inputs={key:episodes.sha256(source)}
            report=dict(schema_version=1,result='PASS',scene='factory',episodes=['F05'],source_inputs=inputs,
                source_manifest_sha256=row['manifest_sha256'],position_m=row['candidate_position_m'],yaw_degrees=0.,
                spawn_passed=True,route_clearance_passed=True,view_passed=True,long_replay_passed=True)
            path=root/'native.json';path.write_text(json.dumps(report));native_key=path.relative_to(episodes.ROOT).as_posix()
            evidence=dict(input_sha256=inputs,native_report=native_key,native_evidence_sha256={native_key:episodes.sha256(path)})
            with patch.object(episodes,'required_input_paths',return_value={key}):
                self.assertFalse(episodes.valid_readiness(row,evidence))
                report.update(initial_occlusion_passed=True,occlusion_evidence_path='hidden.json')
                path.write_text(json.dumps(report));evidence['native_evidence_sha256'][native_key]=episodes.sha256(path)
                with patch('scripts.factory_occlusion.validate_file_evidence',return_value=True):
                    self.assertFalse(episodes.valid_readiness(row,evidence),'Hidden starts still need keyframe emergence/end proof')

    def test_legacy_start_version_still_accepts_its_original_requirements(self):
        with tempfile.TemporaryDirectory(dir=episodes.ROOT/'validation') as folder:
            root=Path(folder);source=root/'source';source.write_text('a');key=source.relative_to(episodes.ROOT).as_posix()
            row=dict(episode='F01',scene='factory',start_group='A',start_version='pilot30_20261031',
                manifest_sha256=episodes.sha256(episodes.MANIFEST),candidate_position_m=[1.,2.,5.],candidate_yaw_degrees=0.)
            inputs={key:episodes.sha256(source)}
            report=dict(schema_version=1,result='PASS',scene='factory',episodes=['F01'],source_inputs=inputs,
                source_manifest_sha256=row['manifest_sha256'],position_m=row['candidate_position_m'],yaw_degrees=0.,
                spawn_passed=True,route_clearance_passed=True,view_passed=True,long_replay_passed=True)
            path=root/'native.json';path.write_text(json.dumps(report));n=path.relative_to(episodes.ROOT).as_posix()
            evidence=dict(input_sha256=inputs,native_report=n,native_evidence_sha256={n:episodes.sha256(path)})
            with patch.object(episodes,'required_input_paths',return_value={key}):
                self.assertTrue(episodes.valid_readiness(row,evidence))


if __name__=='__main__':unittest.main()
