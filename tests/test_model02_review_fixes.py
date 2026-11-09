import json,shutil,sys,tempfile,types,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import yaml
from contextlib import redirect_stderr,redirect_stdout
import io

from scripts import validate_model02_v3_isaac as validator

ROOT=Path(__file__).resolve().parents[1]


class ReviewFixTests(unittest.TestCase):
    def test_task_truth_follows_independent_core_edits(self):
        from scripts.model02_migration import core_temperature_truth
        data={'condition':{'air_temperature_c':20},'targets':{'a':{'mode':'fixed','temperature_c':78},'b':{'mode':'fixed','temperature_c':78}},
              'surfaces':{'A':{'target':'a'},'B':{'target':'b'}}}
        entry={'target_surface_ids':['A','B']}
        self.assertEqual(core_temperature_truth(entry,data)['hotspot_temperature_c'],78)
        data['targets']['a']['temperature_c']=90
        result=core_temperature_truth(entry,data)
        self.assertIsNone(result['hotspot_temperature_c'])
        self.assertEqual(result['hotspot_temperature_range_c'],[78,90])
        self.assertEqual(result['hotspot_temperatures_c'],{'A':90,'B':78})
        data['targets']={'a':{'mode':'ambient'},'b':{'mode':'offset','offset_c':3}}
        self.assertEqual(core_temperature_truth(entry,data)['hotspot_temperatures_c'],{'A':20,'B':23})

    def test_inplace_runtime_material_weather_and_temperature_edits_cannot_replay_old_record(self):
        from scripts.runtime_parameter_provenance import thermal_input_snapshot,validate_thermal_inputs
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            root=Path(folder);materials=root/'materials';shutil.copytree(ROOT/'factory/model02_thermal_v3/configuration/materials',materials)
            weather=root/'weather.csv';weather.write_bytes((ROOT/'factory/model02_thermal_v3/night_weather.csv').read_bytes())
            env=yaml.safe_load((ROOT/'configs/model02_v3_night_environment.yaml').read_text(encoding='utf-8'))
            env['scene']['weather_file']=str(weather);environment=root/'env.yaml';environment.write_text(yaml.safe_dump(env),encoding='utf-8')
            thermal=root/'thermal.yaml';thermal.write_text('temperature_c: 20\n')
            sensor=root/'sensor.yaml';sensor.write_bytes((ROOT/'configs/model02_v3_tank_scene.yaml').read_bytes())
            config=SimpleNamespace(scene=SimpleNamespace(thermal_scene=environment),
                sensors=SimpleNamespace(material_directory=materials,thermal_config=thermal,sensor_config=sensor))
            snapshot=thermal_input_snapshot(config);validate_thermal_inputs(config,snapshot)
            with self.assertRaises(ValueError):validate_thermal_inputs(config,None)
            for path in [thermal,sensor,environment,weather,materials/'industrial_painted_steel.yaml']:
                original=path.read_bytes();path.write_bytes(original+b'\n# changed after recording\n')
                with self.assertRaises(ValueError):validate_thermal_inputs(config,snapshot)
                path.write_bytes(original)
            self.assertEqual(thermal_input_snapshot(config),snapshot)
            legacy=SimpleNamespace(sensors=SimpleNamespace(material_directory=None))
            validate_thermal_inputs(legacy,None)

    def test_startup_failure_still_writes_report_and_returns_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            output=Path(folder)/'run'
            fake=types.ModuleType('isaacsim')
            fake.SimulationApp=lambda settings:(_ for _ in ()).throw(RuntimeError('startup failed'))
            with patch.dict(sys.modules,{'isaacsim':fake}),patch.object(sys,'argv',['validate','--out',str(output)]),redirect_stderr(io.StringIO()),redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main(),1)
            self.assertEqual(json.loads((output/'summary.json').read_text())['result'],'FAIL')

    def test_failed_check_closes_app_and_returns_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            output=Path(folder)/'run';closed=[]
            fake=types.ModuleType('isaacsim');fake.SimulationApp=lambda settings:SimpleNamespace(close=lambda:closed.append(True))
            with patch.dict(sys.modules,{'isaacsim':fake,'omni':None}),patch.object(sys,'argv',['validate','--out',str(output)]),redirect_stderr(io.StringIO()),redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main(),1)
            self.assertEqual(closed,[True]);self.assertEqual(json.loads((output/'summary.json').read_text())['result'],'FAIL')


if __name__=='__main__':unittest.main()
