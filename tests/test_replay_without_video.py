"""Exercise real orchestration with native rendering replaced at its process boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.collection_shell import powershell_executable
from scripts import process_collection_episode as episode

ROOT = Path(__file__).resolve().parents[1]


class EpisodeNoVideoTests(unittest.TestCase):
    def exercise(self, backend, fail_validation=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'recorded';run.mkdir();out=root/'processed'
            (run/'run_metadata.json').write_text(json.dumps({'scene_config_path':str(root/'scene.yaml')}))
            calls=[]
            def logged(command,log):
                calls.append([str(x) for x in command])
                self.assertFalse(any('generate_replay_video' in str(x) for x in command),
                                 'Default replay must not invoke the encoder, even without FFmpeg')
                if Path(log).name=='replay.log':
                    (out/'replay').mkdir()
                    (out/'replay/replay_metadata.json').write_text('{"result":"PASS"}')
                if Path(log).name=='validation.log' and fail_validation:
                    raise RuntimeError('invalid arrays')
                if Path(log).name=='index.log':
                    (out/'dataset').mkdir(exist_ok=True)
                    (out/'dataset/episode.json').write_text('{}')
            def factory(command,**kwargs):
                calls.append([str(x) for x in command]);out.mkdir()
                return subprocess.CompletedProcess(command,0)
            with patch.object(episode,'recorded_backend',return_value=backend), \
                 patch.object(episode,'validate_recorded_scene_assets'), \
                 patch.object(episode,'recorded_target_snapshot',return_value=root/'target.yaml'), \
                 patch.object(episode,'prepare_source',return_value=run/'trajectory.jsonl'), \
                 patch.object(episode,'run_logged',side_effect=logged), \
                 patch.object(episode.subprocess,'run',side_effect=factory), \
                 patch.object(sys,'argv',['process','--run',str(run),'--out',str(out)]):
                if fail_validation:
                    with self.assertRaisesRegex(RuntimeError,'invalid arrays'):episode.main()
                else:self.assertEqual(episode.main(),0)
            status=json.loads((out/'pipeline_status.json').read_text())
            self.assertEqual(status['status'],'FAILED' if fail_validation else 'COMPLETE')
            self.assertIsNone(status.get('video'))
            if not fail_validation:
                skipped=[s for s in status['stages'] if s['name']=='video']
                self.assertEqual(skipped,[dict(name='video',status='SKIPPED',reason='manual_only')])
                self.assertTrue((out/'dataset/episode.json').is_file())
            self.assertFalse((out/'video').exists())
            return calls

    def test_forest_completes_and_indexes_without_an_encoder(self):self.exercise('forest')
    def test_factory_completes_and_indexes_without_a_video(self):self.exercise('factory')
    def test_failed_forest_validation_still_stops_indexing(self):
        calls=self.exercise('forest',True)
        self.assertFalse(any('scripts.episode_dataset' in c for c in calls))


class PowerShellNoVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);scripts=self.root/'scripts';scripts.mkdir()
        self.out=self.root/'processed';self.trace=self.root/'calls.jsonl'
        self.run=self.root/'recorded';self.run.mkdir()
        (self.run/'trajectory.jsonl').write_text('{}\n')
        configs=self.root/'configs';configs.mkdir()
        for name in ['scene.yaml','model02_weaklight_scene.yaml','model02_refined_scene.yaml',
                     'model02_v3_tank_scene.yaml','model02_v4_tank_scene.yaml']:
            (configs/name).write_text('schema_version: 1\n')
        (self.run/'run_metadata.json').write_text(json.dumps({'scene_config_path':str(configs/'scene.yaml')}))
        self.python=self.root/'fake_python.cmd'
        stub=self.root/'fake_python.py'
        self.python.write_text(f'@echo off\n"{sys.executable}" "{stub}" %*\nexit /b %errorlevel%\n')
        stub.write_text('''import json,os,sys
from pathlib import Path
a=sys.argv[1:]
if a and a[0]=='-B':a=a[1:]
name=Path(a[0]).name
with open(os.environ['TEST_TRACE'],'a') as f:f.write(json.dumps(a)+'\\n')
def value(flag):return Path(a[a.index(flag)+1])
if name=='generate_replay_video.py':
 if os.environ.get('TEST_VIDEO_CREATE')=='1':
  d=value('--out');d.mkdir(parents=True);(d/'paired_replay.mp4').write_bytes(b'test video')
 code=int(os.environ.get('TEST_VIDEO_CODE','19'))
 if code:print(os.environ.get('TEST_VIDEO_ERROR','RuntimeError: encoded timestamps do not match replay-relative simulation times'),file=sys.stderr)
 sys.exit(code)
if name=='verify_factory_replay_video.py':
 if os.environ.get('TEST_VERIFY_FAIL')=='1':sys.exit(8)
 p=value('--out');p.write_text('{"result":"PASS"}');sys.exit(0)
if os.environ.get('TEST_VALIDATION_FAIL')=='1' and (name=='validate_replay_outputs.py' or '--validate-only' in a):sys.exit(7)
if '--out' in a:value('--out').mkdir(parents=True,exist_ok=True)
sys.exit(0)
''')
        names=['process.ps1','process_factory_inspection.ps1','process_manual_preview.ps1',
               'process_model02.ps1','process_model02_refined.ps1','process_model02_v3.ps1',
               'process_model02_v4.ps1','video.ps1']
        for name in names:
            text=(ROOT/'scripts'/name).read_text(encoding='utf-8')
            text=text.replace("'D:\\isaacsim\\python.bat'", "'"+str(self.python)+"'")
            (scripts/name).write_text(text,encoding='utf-8-sig')
        (scripts/'invoke_logged_powershell.ps1').write_text('''param($ShellPath,$EncodedCommand,$StdoutPath,$StderrPath)
$out = $env:TEST_OUTPUT
$replay = Join-Path $out 'replay'
New-Item -ItemType Directory -Path $replay -Force | Out-Null
if ($env:TEST_MISSING_FINAL -ne '1') { [IO.File]::WriteAllText((Join-Path $replay 'replay_metadata.json'), '{"result":"PASS"}') }
[IO.File]::WriteAllText((Join-Path $replay 'frame_000000.json'), '{}')
[IO.File]::WriteAllText($StdoutPath,'render complete')
[IO.File]::WriteAllText($StderrPath,'')
return 0
''',encoding='utf-8-sig')
        self.env=dict(os.environ,TEST_OUTPUT=str(self.out),TEST_TRACE=str(self.trace))
        self.env['PATH']=str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32')

    def invoke(self,script,extra=(),**env):
        command=[str(powershell_executable()),'-NoProfile','-ExecutionPolicy','Bypass','-File',
                 str(self.root/'scripts'/script),*extra]
        return subprocess.run(command,env=dict(self.env,**env),capture_output=True,text=True,
                              encoding='utf-8',errors='replace',timeout=45)

    def calls(self):return [json.loads(x) for x in self.trace.read_text().splitlines()] if self.trace.exists() else []

    def test_generic_replay_does_not_need_ffmpeg_and_retains_index(self):
        r=self.invoke('process.ps1',['-Run',str(self.run),'-Out',str(self.out)])
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertFalse(any('video' in Path(c[0]).name for c in self.calls()))
        status=json.loads((self.out/'pipeline_status.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(status['status'],'COMPLETE')
        stage=next(s for s in status['stages'] if s['name']=='video')
        self.assertEqual((stage['status'],stage['reason']),('SKIPPED','manual_only'))
        index=json.loads((self.out/'dataset/index.json').read_text(encoding='utf-8-sig'))
        self.assertIsNone(index['artifacts']['video'])
        self.assertFalse((self.out/'video').exists())

    def test_factory_compatibility_wrappers_keep_evaluation_without_video(self):
        for name in ['process_factory_inspection.ps1','process_manual_preview.ps1','process_model02.ps1',
                     'process_model02_refined.ps1','process_model02_v3.ps1','process_model02_v4.ps1']:
            with self.subTest(name=name):
                self.out=self.root/name.replace('.ps1','');self.env['TEST_OUTPUT']=str(self.out)
                r=self.invoke(name,['-Run',str(self.run),'-Out',str(self.out)])
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                self.assertFalse((self.out/'video').exists())
        calls=self.calls()
        self.assertFalse(any('video' in Path(c[0]).name for c in calls))
        self.assertTrue(any('--manual-summary' in c for c in calls))
        self.assertTrue(any(Path(c[0]).name=='evaluate_factory_inspection.py' for c in calls))

    def test_failed_validation_and_missing_final_metadata_remain_fatal(self):
        for env in [{'TEST_VALIDATION_FAIL':'1'},{'TEST_MISSING_FINAL':'1'}]:
            with self.subTest(env=env):
                out=self.root/next(iter(env));self.env['TEST_OUTPUT']=str(out)
                r=self.invoke('process.ps1',['-Run',str(self.run),'-Out',str(out)],**env)
                self.assertNotEqual(r.returncode,0)
                self.assertFalse((out/'dataset/index.json').exists())
                self.assertEqual(json.loads((out/'pipeline_status.json').read_text(encoding='utf-8-sig'))['status'],'FAILED')

    def video_fixture(self,factory):
        replay=self.root/'replay';replay.mkdir()
        (replay/'replay_metadata.json').write_text(json.dumps({'pipeline':'factory_inspection_offline_v1' if factory else 'generic'}))
        return replay

    def test_standalone_factory_video_performs_its_dedicated_verification(self):
        replay=self.video_fixture(True)
        r=self.invoke('video.ps1',['-Replay',str(replay),'-Out',str(self.out),'-Ffmpeg','custom-encoder.exe'],
                      TEST_VIDEO_CREATE='1',TEST_VIDEO_CODE='0')
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        calls=self.calls();self.assertEqual([Path(c[0]).name for c in calls],
            ['generate_replay_video.py','verify_factory_replay_video.py'])
        self.assertIn('--ffmpeg',calls[-1]);self.assertIn('custom-encoder.exe',calls[-1])

    def test_factory_fallback_requires_a_video_and_successful_timestamp_check(self):
        replay=self.video_fixture(True)
        r=self.invoke('video.ps1',['-Replay',str(replay),'-Out',str(self.out)],TEST_VIDEO_CREATE='1',TEST_VIDEO_CODE='1')
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertTrue((self.out/'video_validation.json').is_file())

    def test_failed_factory_timestamp_check_is_fatal(self):
        replay=self.video_fixture(True)
        r=self.invoke('video.ps1',['-Replay',str(replay),'-Out',str(self.out)],
                      TEST_VIDEO_CREATE='1',TEST_VIDEO_CODE='0',TEST_VERIFY_FAIL='1')
        self.assertNotEqual(r.returncode,0)

    def test_factory_fallback_does_not_hide_non_timing_generation_failure(self):
        replay=self.video_fixture(True)
        r=self.invoke('video.ps1',['-Replay',str(replay),'-Out',str(self.out)],
                      TEST_VIDEO_CREATE='1',TEST_VIDEO_CODE='1',TEST_VIDEO_ERROR='RuntimeError: source data changed during video generation')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(len(self.calls()),1)

    def test_generic_video_failure_is_not_treated_as_factory_rounding(self):
        replay=self.video_fixture(False)
        r=self.invoke('video.ps1',['-Replay',str(replay),'-Out',str(self.out)],TEST_VIDEO_CREATE='1',TEST_VIDEO_CODE='1')
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(len(self.calls()),1)


class StandaloneVideoNativeTests(unittest.TestCase):
    def test_real_generic_and_factory_mp4_preserve_source_files(self):
        from tests.test_replay_video import ReplayVideoTests
        from scripts.generate_replay_video import find_ffmpeg
        import hashlib
        fixture=ReplayVideoTests('test_real_mp4_has_one_encoded_frame_per_pair_and_matching_timestamps')
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        for factory in (False,True):
            with self.subTest(factory=factory):
                metadata=fixture.replay/'replay_metadata.json'
                report=json.loads(metadata.read_text())
                report['pipeline']='factory_inspection_offline_v1' if factory else 'generic'
                metadata.write_text(json.dumps(report))
                before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.replay.iterdir() if p.is_file()}
                out=fixture.root/('factory-video' if factory else 'generic-video')
                result=subprocess.run([str(powershell_executable()),'-NoProfile','-ExecutionPolicy','Bypass',
                    '-File',str(ROOT/'scripts/video.ps1'),'-Replay',str(fixture.replay),'-Out',str(out),
                    '-Ffmpeg',str(find_ffmpeg(None))],capture_output=True,text=True,
                    encoding='utf-8',errors='replace',timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(json.loads((out/'video_metadata.json').read_text())['encoded_frame_count'],3)
                self.assertTrue((out/'paired_replay.mp4').is_file())
                if factory:self.assertEqual(json.loads((out/'video_validation.json').read_text())['result'],'PASS')
                self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.replay.iterdir() if p.is_file()})


if __name__=='__main__':unittest.main()
