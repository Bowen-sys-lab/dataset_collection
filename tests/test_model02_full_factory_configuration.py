import copy,unittest,tempfile,json
from pathlib import Path
from scripts import model02_v6_configuration as v6


class ConfigurationTests(unittest.TestCase):
    def test_export_parts_preserve_ids_and_full_source_face_provenance(self):
        rows=[{'surface_id':'s'+str(i),'object':'o'+str(i),'source_part':'A','source_object':'A','source_face_ids':[i]} for i in range(5)]
        batch={'label':'equipment_Z11','surfaces':rows,'replaced_parts':['A'],'geometry_checks':[{'source_part':'A'}]}
        parts=v6.export_parts(batch,2)
        self.assertEqual([len(p['surfaces']) for p in parts],[2,2,1])
        self.assertEqual([r['surface_id'] for p in parts for r in p['surfaces']],[r['surface_id'] for r in rows])
        self.assertTrue(all(p['source_face_coverage']['A']==list(range(5)) for p in parts))

    def test_editing_a_source_file_preserves_other_source_targets(self):
        root=Path(__file__).resolve().parents[1]/'validation/full-factory/test_tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            folder=Path(folder);index=folder/'thermal_authoring.yaml';index.write_text(json.dumps({'source_files':['a.yaml','b.yaml']}))
            for name,t in [('a',22.),('b',23.)]:(folder/(name+'.yaml')).write_text(json.dumps({'targets':{name:{'temperature_c':t}},'surfaces':{name:{'object':name,'target':name,'material':'concrete'}}}))
            before=v6.load_authoring(index);a=json.loads((folder/'a.yaml').read_text());a['targets']['a']['temperature_c']=35.;(folder/'a.yaml').write_text(json.dumps(a))
            after=v6.load_authoring(index);self.assertEqual(after['targets']['a']['temperature_c'],35.);self.assertEqual(before['targets']['b'],after['targets']['b'])
            b=json.loads((folder/'b.yaml').read_text());b['targets']=a['targets'];(folder/'b.yaml').write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'duplicate'):v6.load_authoring(index)

    def test_retained_targets_keep_values_and_actual_legacy_export_bounds(self):
        row={'surface_id':'a','blender_object':'ObjectA','temperature_target_id':'target_a','ir_material':'concrete','temperature_c':22.5}
        exported=[{'surface_id':'a','bounds':[[1.,2.,3.],[4.,5.,6.]],'area_m2':9.}]
        result=v6.retained_surfaces([row],exported,set())[0]
        self.assertEqual(result['temperature_c'],22.5)
        self.assertEqual(result['target'],'target_a')
        self.assertEqual(result['bounds_m'],exported[0]['bounds'])
        self.assertEqual(v6.retained_surfaces([row],exported,{'a'}),[])

    def data(self):
        rows=[{'surface_id':'wall_a','object':'WallA','target':'wall_a','material':'concrete','assumption':'estimated concrete proxy'},
              {'surface_id':'wall_b','object':'WallB','target':'wall_b','material':'concrete','assumption':'estimated concrete proxy'}]
        authoring={'materials':{'concrete':{'basis':'uncalibrated concrete library proxy'}},
            'targets':{'wall_a':{'temperature_c':22.},'wall_b':{'temperature_c':23.}},
            'surfaces':{r['surface_id']:{k:r[k] for k in ['object','target','material']} for r in rows}}
        return authoring,rows

    def test_missing_surface_rejected(self):
        a,r=self.data();del a['surfaces']['wall_b']
        with self.assertRaises(ValueError):v6.compile_configuration(a,r)

    def test_shared_target_rejected(self):
        a,r=self.data();a['surfaces']['wall_b']['target']='wall_a'
        with self.assertRaises(ValueError):v6.compile_configuration(a,r)

    def test_colour_is_not_a_temperature_input_and_edit_is_local(self):
        a,r=self.data();before=v6.compile_configuration(a,r)
        r[0]['rgb_colour']=[1.,0.,0.]
        self.assertEqual(before,v6.compile_configuration(a,r))
        a['targets']['wall_a']['temperature_c']=35.
        after=v6.compile_configuration(a,r)
        self.assertEqual(after['objects']['WallB'],before['objects']['WallB'])
        self.assertEqual(after['objects']['WallA']['temperature_c'],35.)

    def test_proxy_requires_basis(self):
        a,r=self.data();a['materials']['concrete']['basis']=''
        with self.assertRaises(ValueError):v6.compile_configuration(a,r)


if __name__=='__main__':unittest.main()
