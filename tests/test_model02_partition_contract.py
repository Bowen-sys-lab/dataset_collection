import unittest

from scripts.blender.model02_inventory import validate_partition_plan


class PartitionContractTests(unittest.TestCase):
    def test_missing_duplicate_and_unknown_source_faces_are_reported(self):
        snapshot = {'objects':[{'name':'tank','faces':4}]}
        plan = {'parts':[{'source_object':'tank','source_face_ids':[0,1], 'action':'split'},
                         {'source_object':'tank','source_face_ids':[1,2], 'action':'split'},
                         {'source_object':'unknown','source_face_ids':[0], 'action':'keep'}]}
        report = validate_partition_plan(snapshot,plan)
        self.assertEqual(report['missing_faces'],[['tank',3]])
        self.assertEqual(report['duplicate_faces'],[['tank',1]])
        self.assertEqual(report['unknown_sources'],['unknown'])
        self.assertFalse(report['complete'])

    def test_reconstruction_requires_evidence_and_accounts_for_source_once(self):
        snapshot = {'objects':[{'name':'wall','faces':2}]}
        plan = {'parts':[{'source_object':'wall','source_face_ids':[0,1],
                         'action':'replace','reconstruction':'window glass and frame from source UV layout'}]}
        self.assertTrue(validate_partition_plan(snapshot,plan)['complete'])
        del plan['parts'][0]['reconstruction']
        self.assertFalse(validate_partition_plan(snapshot,plan)['complete'])


if __name__ == '__main__':
    unittest.main()
