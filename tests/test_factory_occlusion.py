import copy
import unittest
import numpy as np
from scripts.factory_occlusion import target_pixel_counts, validate_views, validate_file_evidence, validate_route_file_evidence
SOURCE_INPUTS={'scene.usda':'source-v1'}


class OcclusionTests(unittest.TestCase):
    def views(self):
        plane=np.full((8,8),10,dtype=np.uint32)
        matrix=np.eye(4);matrix[3,:3]=[1.,2.,5.]
        return [dict(kind=kind,body_position_m=[1.,2.,5.],rgb_ids=plane.copy(),ir_ids=plane.copy(),
            body_world_matrix=matrix.tolist(),
            rgb_labels={'10':'/World/Buildings/Wall'},ir_labels={'10':'/World/Buildings/Wall'},
            projected_target_pixels=100,ray_blockers=[dict(path='/World/Buildings/Wall',distance_m=2.,target_distance_m=20.)]*3)
            for kind in ('initial','aim_tank','aim_pipe')]

    def test_normal_equipment_surfaces_are_counted_not_only_hotspot(self):
        counts=target_pixel_counts(np.array([[7,8,9,10]],dtype=np.uint32),
            {'7':'/World/TANK_GC075/Shell','8':'/World/GC089_NormalPipe','9':'/World/PIPE_GC089/Hotspot','10':'/World/Building'})
        self.assertEqual(counts,{'tank':1,'pipe':2})

    def test_real_hidden_views_pass(self):
        self.assertTrue(validate_views(self.views(),[1.,2.,5.]))

    def test_initial_actual_matrix_must_match_yaw_and_level_pose(self):
        views=self.views();views[0]['body_world_matrix'][:2]=[[0.,1.,0.,0.],[-1.,0.,0.,0.]]
        views[0]['body_yaw_degrees']=0.  # A copied scalar must not override the measured matrix.
        self.assertFalse(validate_views(views,[1.,2.,5.]))
        views=self.views();angle=np.radians(10)
        views[0]['body_world_matrix'][1][:3]=[0.,float(np.cos(angle)),float(np.sin(angle))]
        views[0]['body_world_matrix'][2][:3]=[0.,-float(np.sin(angle)),float(np.cos(angle))]
        self.assertFalse(validate_views(views,[1.,2.,5.]))

    def test_visible_normal_target_in_either_stream_fails(self):
        for stream in ('rgb','ir'):
            views=self.views();views[0][stream+'_ids'][0,0]=12
            views[0][stream+'_labels']['12']='/World/GC089_NormalPipe'
            self.assertFalse(validate_views(views,[1.,2.,5.]))

    def test_empty_uninitialized_or_unlabeled_planes_fail(self):
        for value in (np.zeros((8,8),dtype=np.uint32),np.empty((0,0),dtype=np.uint32),np.ones((8,8),dtype=float)):
            views=self.views();views[0]['rgb_ids']=value
            self.assertFalse(validate_views(views,[1.,2.,5.]))
        views=self.views();views[0]['ir_labels']={}
        self.assertFalse(validate_views(views,[1.,2.,5.]))

    def test_missing_aim_views_and_offscreen_target_fail(self):
        self.assertFalse(validate_views(self.views()[:1],[1.,2.,5.]))
        views=self.views();views[1]['projected_target_pixels']=0
        self.assertFalse(validate_views(views,[1.,2.,5.]))

    def test_missing_false_or_target_as_blocker_fails(self):
        for change in ({'ray_blockers':[]},{'ray_blockers':[dict(path='/World/Wall',distance_m=22.,target_distance_m=20.)]*3},
                       {'ray_blockers':[dict(path='/World/TANK_GC075',distance_m=2.,target_distance_m=20.)]*3}):
            views=self.views();views[1].update(change)
            self.assertFalse(validate_views(views,[1.,2.,5.]))

    def test_wrong_spawn_or_nonfinite_metrics_fail(self):
        views=self.views();views[2]['body_position_m']=[8.,2.,5.]
        self.assertFalse(validate_views(views,[1.,2.,5.]))
        views=self.views();views[1]['ray_blockers'][0]=dict(path='/World/Wall',distance_m=float('nan'),target_distance_m=20.)
        self.assertFalse(validate_views(views,[1.,2.,5.]))

    def test_file_evidence_requires_all_bound_images_masks_labels_and_matching_group(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from PIL import Image
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);views=self.views();hashes={}
            for i,view in enumerate(views):
                for stream in ('rgb','ir'):
                    ids=root/f'{i}_{stream}.npy';np.save(ids,view.pop(stream+'_ids'))
                    labels=root/f'{i}_{stream}.json';labels.write_text(json.dumps(view.pop(stream+'_labels')))
                    picture=root/f'{i}_{stream}.png';Image.new('RGB',(8,8)).save(picture)
                    for key,path in (('ids',ids),('labels',labels),('image',picture)):
                        view[stream+'_'+key]=path.name;hashes[path.name]=sha(path)
            report=root/'proof.json';row={'start_group':'D','start_version':'factory_interior_occluded_v1',
                'manifest_sha256':'manifest','candidate_position_m':[1.,2.,5.],'candidate_yaw_degrees':0.}
            data={'schema_version':1,'scene':'factory','group':'D','start_version':row['start_version'],
                'source_manifest_sha256':'manifest','source_inputs':SOURCE_INPUTS,'views':views}
            report.write_text(json.dumps(data));hashes[report.name]=sha(report)
            self.assertTrue(validate_file_evidence(report.name,row,root,hashes,expected_source_inputs=SOURCE_INPUTS))
            self.assertFalse(validate_file_evidence(report.name,row,root,hashes,expected_source_inputs={'scene.usda':'source-v2'}))
            del data['source_inputs'];report.write_text(json.dumps(data));hashes[report.name]=sha(report)
            self.assertFalse(validate_file_evidence(report.name,row,root,hashes,expected_source_inputs=SOURCE_INPUTS))
            data['source_inputs']=SOURCE_INPUTS;report.write_text(json.dumps(data));hashes[report.name]=sha(report)
            missing=dict(hashes);missing.pop('0_rgb.png')
            self.assertFalse(validate_file_evidence(report.name,row,root,missing,expected_source_inputs=SOURCE_INPUTS))
            self.assertFalse(validate_file_evidence(report.name,{**row,'start_group':'E'},root,hashes,expected_source_inputs=SOURCE_INPUTS))
            (root/'0_rgb.npy').write_bytes(b'changed')
            self.assertFalse(validate_file_evidence(report.name,row,root,hashes,expected_source_inputs=SOURCE_INPUTS))

    def test_route_views_require_hidden_before_visible_appearance_and_real_hotspot_at_end(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from PIL import Image
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);hashes={};views=[]
            points=[[1.,2.,5.],[1.,8.,5.],[1.,15.,3.5]]
            for i,kind in enumerate(('before_corner','appearance','end')):
                view=dict(kind=kind,body_position_m=points[i],projected_target_pixels=100)
                for stream in ('rgb','ir'):
                    mask=np.full((8,8),10,dtype=np.uint32)
                    labels={'10':'/World/Wall','20':'/World/TANK_GC075/Shell','21':'/World/TANK_GC075/Hotspot'}
                    if i==1:mask[:2,:2]=20
                    if i==2:mask[:5,:4]=21
                    ids=root/f'{i}_{stream}.npy';np.save(ids,mask)
                    lab=root/f'{i}_{stream}.json';lab.write_text(json.dumps(labels))
                    image=root/f'{i}_{stream}.png';Image.new('RGB',(8,8)).save(image)
                    for key,path in (('ids',ids),('labels',lab),('image',image)):
                        hashes[path.name]=sha(path);view[stream+'_'+key]=path.name
                views.append(view)
            row=dict(scene='factory',target='tank',start_group='D',start_version='factory_interior_occluded_v1',manifest_sha256='manifest',reference_waypoints=points)
            data=dict(schema_version=1,scene='factory',group='D',start_version=row['start_version'],source_manifest_sha256='manifest',source_inputs=SOURCE_INPUTS,target='tank',reference_waypoints=points,views=views)
            report=root/'route.json';report.write_text(json.dumps(data));hashes[report.name]=sha(report)
            targets=['/World/TANK_GC075/Hotspot']
            self.assertTrue(validate_route_file_evidence(report.name,row,root,hashes,targets,expected_source_inputs=SOURCE_INPUTS))
            self.assertFalse(validate_route_file_evidence(report.name,row,root,hashes,targets,expected_source_inputs={'scene.usda':'source-v2'}))
            self.assertFalse(validate_route_file_evidence(report.name,{**row,'target':'pipe'},root,hashes,targets,expected_source_inputs=SOURCE_INPUTS))
            self.assertFalse(validate_route_file_evidence(report.name,row,root,hashes,['/World/OtherHotspot'],expected_source_inputs=SOURCE_INPUTS))
            data['views'][0]['body_position_m']=[1.,10.,5.-3./7.]
            report.write_text(json.dumps(data));hashes[report.name]=sha(report)
            self.assertFalse(validate_route_file_evidence(report.name,row,root,hashes,targets,expected_source_inputs=SOURCE_INPUTS))
            data['views'][0]['body_position_m']=points[0]
            data['views'][1]['body_position_m']=[100.,100.,5.]
            report.write_text(json.dumps(data));hashes[report.name]=sha(report)
            self.assertFalse(validate_route_file_evidence(report.name,row,root,hashes,targets,expected_source_inputs=SOURCE_INPUTS))


if __name__=='__main__':unittest.main()
