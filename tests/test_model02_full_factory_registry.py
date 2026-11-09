import copy,importlib.util,unittest


class RegistryTests(unittest.TestCase):
    def test_retained_background_binding_cannot_hide_unprocessed_raw_source(self):
        m=self.module();_,r=self.fixture()
        for s in r['sources']:s['render_visible_original']=True
        batch={'replaced_parts':['A'],'surfaces':[{'source_part':'A','source_object':'A'}]}
        with self.assertRaisesRegex(ValueError,'raw'):m.validate_authoring_coverage(r,[batch],{'B'})
        batch['replaced_parts'].append('B');batch['surfaces'].append({'source_part':'B','source_object':'B'})
        self.assertEqual(m.validate_authoring_coverage(r,[batch],set())['raw_sources'],2)

    def test_duplicate_replacement_is_not_complete_coverage(self):
        m=self.module();_,r=self.fixture()
        batch={'replaced_parts':['A','B'],'surfaces':[{'source_part':s,'source_object':s} for s in ['A','B']]}
        with self.assertRaisesRegex(ValueError,'duplicate'):m.validate_authoring_coverage(r,[batch,batch],set())

    def test_validated_source_requires_actual_surface_outputs(self):
        m=self.module();inventory,registry=self.fixture()
        for r in registry['sources']:r['status']='validated'
        with self.assertRaisesRegex(ValueError,'output'):m.validate_registry(inventory,registry)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.model02_full_factory_registry'))
        from scripts import model02_full_factory_registry
        return model02_full_factory_registry

    def fixture(self):
        inventory={'objects':[{'name':'A','faces':3},{'name':'B','faces':2}]}
        rows=[{'source_object':name,'source_geometry_sha256':'a'*64,'zone_id':'Z00',
               'role':'building','classification_evidence':['geometry:A.png'],'component_rules':['wall_roof'],
               'face_accounting':[{'start':0,'stop':count,'action':'retained'}],'status':'classified'}
              for name,count in [('A',3),('B',2)]]
        return inventory,{'sources':rows}

    def test_every_source_is_accounted_once(self):
        m=self.module();inventory,registry=self.fixture()
        result=m.validate_registry(inventory,registry)
        self.assertEqual(result['source_objects'],2);self.assertEqual(result['source_faces'],5)
        registry['sources'].append(copy.deepcopy(registry['sources'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):m.validate_registry(inventory,registry)

    def test_missing_source_and_missing_or_overlapping_faces_are_rejected(self):
        m=self.module()
        for alteration in ['missing_source','missing_face','duplicate_face']:
            inventory,registry=self.fixture()
            if alteration=='missing_source':registry['sources'].pop()
            elif alteration=='missing_face':registry['sources'][0]['face_accounting'][0]['stop']=2
            else:registry['sources'][0]['face_accounting'].append({'start':1,'stop':2,'action':'retained'})
            with self.assertRaises(ValueError):m.validate_registry(inventory,registry)

    def test_geometric_duplicates_cannot_share_surface_or_thermal_identity(self):
        m=self.module();inventory,registry=self.fixture()
        registry['sources'][0]['outputs']=[{'surface_id':'sA','target':'tA'}]
        registry['sources'][1]['outputs']=[{'surface_id':'sB','target':'tA'}]
        with self.assertRaisesRegex(ValueError,'target'):m.validate_registry(inventory,registry)

    def test_unresolved_classification_is_never_marked_complete(self):
        m=self.module();inventory,registry=self.fixture()
        registry['sources'][0].update(role='unclassified',classification_evidence=[],status='unclassified')
        self.assertFalse(m.validate_registry(inventory,registry)['complete'])
        registry['sources'][0]['status']='validated'
        with self.assertRaisesRegex(ValueError,'classification'):m.validate_registry(inventory,registry)
