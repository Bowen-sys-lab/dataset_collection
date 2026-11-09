import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts import replay_factory_inspection as replay


class ReplayRepairTests(unittest.TestCase):
    def test_memory_budget_is_explicit_and_does_not_change_resolution_or_temperatures(self):
        self.assertTrue(hasattr(replay,'configure_replay_memory'))
        class Settings:
            def __init__(self):self.values={}
            def set(self,key,value):self.values[key]=value
            def get(self,key):return self.values.get(key)
        settings=Settings();policy=replay.configure_replay_memory(settings,.35)
        self.assertTrue(settings.get('/rtx-transient/resourcemanager/texturestreaming/enabled'))
        self.assertEqual(settings.get('/rtx-transient/resourcemanager/texturestreaming/memoryBudget'),.35)
        self.assertEqual(len(settings.values),2)
        self.assertEqual(policy['texture_budget_fraction'],.35)
        for bad in (0.,1.1,float('nan')):
            with self.assertRaises(ValueError):replay.configure_replay_memory(settings,bad)

    def test_hotspot_only_labels_preserve_actual_visibility_evidence(self):
        self.assertTrue(hasattr(replay,'hotspot_labels'))
        labels={1:'/Hot/A',2:'/Background',3:'/Hot/B'}
        filtered=replay.hotspot_labels(labels,['/Hot/A','/Hot/B'])
        self.assertEqual(set(filtered),{'1','3'})
        ids=np.array([[1,2,3]])
        self.assertEqual(replay.hotspot_evidence(ids,filtered,'/Hot/A',1,target_paths=['/Hot/A','/Hot/B']),
                         replay.hotspot_evidence(ids,labels,'/Hot/A',1,target_paths=['/Hot/A','/Hot/B']))

    def test_order_manifest_is_chronological_and_refuses_reversed_input(self):
        self.assertTrue(hasattr(replay,'write_frame_order'))
        rows=[{'frame_id':i,'simulation_time':.016+i*.1} for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path=replay.write_frame_order(Path(directory),rows)
            with path.open(encoding='utf-8-sig') as stream:saved=list(csv.DictReader(stream))
            self.assertEqual([r['rgb_file'] for r in saved],[f'frame_{i:06d}_rgb.png' for i in range(3)])
            self.assertEqual([float(r['time_from_start_s']) for r in saved],[0.,.1,.2])
            with self.assertRaises(ValueError):replay.write_frame_order(Path(directory),list(reversed(rows)))

if __name__=='__main__':unittest.main()
