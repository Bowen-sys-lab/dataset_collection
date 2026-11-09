import unittest
from scripts.collection_episode import load_episodes,task_snapshot,resolve_episode


class InteriorManifestTests(unittest.TestCase):
    def test_only_last_six_factory_rows_are_new_occluded_version(self):
        rows=load_episodes();self.assertEqual(len(rows),20)
        factory=[r for r in rows if r['scene']=='factory'];self.assertEqual(len(factory),10)
        expected={'A':([700.,-930.,8.5],65.07960786001456),'B':([620.,-890.,10.],54.4053115009569)}
        for row in factory[:4]:
            point,yaw=expected[row['start_group']]
            self.assertEqual(row['candidate_position_m'],point);self.assertAlmostEqual(row['candidate_yaw_degrees'],yaw)
            self.assertEqual(row['start_version'],'pilot30_20261031')
        for row in factory[4:]:
            self.assertEqual(row['start_version'],'factory_interior_occluded_v1')
            self.assertEqual(row['initial_visibility'],'physically_occluded')
            self.assertIn('hidden',row['instruction'].lower())
            self.assertTrue(row['instruction'].isascii())
            self.assertEqual(row['length_policy']['enforced'],False)
        self.assertEqual([r['episode'] for r in rows if r['start_version']=='factory_interior_occluded_v1'],
                         ['F05','F06','F07','F08','F09','F10'])

    def test_internal_pairs_preserve_same_spawn_and_splits(self):
        rows=load_episodes()
        for group,split,variant in [('D','train',3),('E','validation',1),('F','test',2)]:
            pair=[r for r in rows if r['scene']=='factory' and r['start_group']==group]
            self.assertEqual({r['target'] for r in pair},{'tank','pipe'})
            self.assertEqual(pair[0]['candidate_position_m'],pair[1]['candidate_position_m'])
            self.assertEqual(pair[0]['candidate_yaw_degrees'],pair[1]['candidate_yaw_degrees'])
            for row in pair:
                self.assertEqual(row['split'],split);self.assertEqual(row['instruction_variant'],variant)
                self.assertTrue(100<=row['reference_path_length_m']<=200)

    def test_internal_snapshot_records_the_visibility_contract(self):
        row=resolve_episode('factory','F05');task=task_snapshot(row)
        context=task['environment_evidence']['collection_episode']
        self.assertEqual(context['start_version'],'factory_interior_occluded_v1')
        self.assertIn('physically occluded',context['initial_visibility'])
        self.assertNotIn('reference_waypoints',context)


if __name__=='__main__':unittest.main()
