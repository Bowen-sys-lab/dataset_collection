"""Forest pairs must reject mismatches and preserve the IR pass's measurements."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scene_intake import merge_replay_passes as merge
from scripts.frame_metadata import FrameMetadataWriter, load_frame_metadata


def make_pass(path, *, hot=False):
    path.mkdir()
    matrix = np.eye(4).tolist()
    meta = dict(frame_id=0, trajectory_frame_id=5, simulation_time_s=1.0,
                timeline_time_before=1.0, timeline_time_after=1.0,
                camera_world_matrix=matrix, thermal_camera_world_matrix=matrix,
                rgb_resolution=[4, 4], thermal_resolution=[2, 2],
                rgb_intrinsics_px=dict(fx_px=2., fy_px=2., cx_px=2., cy_px=2., width=4, height=4),
                thermal_intrinsics_px=dict(fx_px=1., fy_px=1., cx_px=1., cy_px=1., width=2, height=2),
                sensor_config_sha256='a'*64, thermal_config_sha256='b'*64,
                thermal_environment_config_sha256='c'*64, display_config_sha256='d'*64,
                stage_path='ir.usda' if hot else 'rgb.usda', source_stage_sha256=('e' if hot else 'f')*64,
                display_range_diagnostics=dict(above_range_pixels=1 if hot else 0,
                                               display_temperature_min_c=10., display_temperature_max_c=150.),
                display_profile=dict(temperature_min_c=10., temperature_max_c=150.),
                thermal_coverage=dict(complete_for_this_view=True, unmapped_pixels=0),
                strict_sync_verified=False)
    writer = FrameMetadataWriter(path, meta)
    writer.write(0, meta)
    rgb = np.full((4, 4, 4), 24, dtype=np.uint8); rgb[..., 3] = 255
    t = np.full((2, 2), 290., dtype=np.float32)
    if hot: t[0, 0] = 923.15
    planes = dict(rgb=rgb, apparent_t=t, radiance=t/10,
                  dn16=np.array([[65535 if hot else 300, 300], [300, 300]], dtype=np.uint16),
                  display8=np.array([[255 if hot else 10, 10], [10, 10]], dtype=np.uint8))
    for name, pixels in planes.items(): np.save(path/f'frame_000000_{name}.npy', pixels)
    Image.fromarray(rgb).save(path/'frame_000000_rgb.png')
    Image.fromarray(planes['display8']).save(path/'frame_000000_thermal.png')


class ForestMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rgb = self.root/'rgb'; self.ir = self.root/'ir'; self.out = self.root/'out'
        make_pass(self.rgb); make_pass(self.ir, hot=True)

    def run_merge(self):
        self.assertTrue(hasattr(merge, 'merge_runs'), 'missing validated merge_runs API')
        return merge.merge_runs(self.rgb, self.ir, self.out)

    def edit_frame(self, directory, key, value):
        p=directory/'frame_000000.json'; m=json.loads(p.read_text()); m[key]=value
        p.write_text(json.dumps(m))

    def test_ir_statistics_and_raw_planes_survive_merge(self):
        self.run_merge()
        m=load_frame_metadata(self.out/'frame_000000.json')
        self.assertEqual(m['display_range_diagnostics']['above_range_pixels'], 1)
        self.assertEqual(m['quality']['dn_saturated_pixels'], 1)
        self.assertFalse(m['strict_sync_verified'])
        for name, source in [('rgb.npy',self.rgb),('dn16.npy',self.ir),('apparent_t.npy',self.ir)]:
            self.assertEqual((self.out/f'frame_000000_{name}').read_bytes(),
                             (source/f'frame_000000_{name}').read_bytes())
        self.assertEqual(np.asarray(Image.open(self.out/'quality/frame_000000_saturation.png'))[0,0],255)

    def test_missing_plane_rejected_before_output_exists(self):
        (self.ir/'frame_000000_dn16.npy').unlink()
        with self.assertRaises(ValueError): self.run_merge()
        self.assertFalse(self.out.exists())

    def test_scalar_thermal_planes_reject_three_channels_and_float16(self):
        for suffix in ('apparent_t','radiance'):
            p=self.ir/f'frame_000000_{suffix}.npy';original=np.load(p)
            for malformed in (np.repeat(original[...,None],3,axis=2),original.astype(np.float16)):
                with self.subTest(suffix=suffix,shape=malformed.shape,dtype=str(malformed.dtype)):
                    np.save(p,malformed)
                    with self.assertRaises(ValueError):self.run_merge()
                    self.assertFalse(self.out.exists())
            np.save(p,original)

    def test_material_library_identity_must_match_when_recorded(self):
        self.edit_frame(self.rgb,'thermal_configuration',{'material_library_sha256':'a'*64})
        self.edit_frame(self.ir,'thermal_configuration',{'material_library_sha256':'b'*64})
        with self.assertRaises(ValueError):self.run_merge()
        self.assertFalse(self.out.exists())

    def test_pose_time_config_and_intrinsics_mismatch_rejected(self):
        cases=[('camera_world_matrix',[[1,0,0,0],[0,1,0,0],[0,0,1,0],[3,0,0,1]]),
               ('simulation_time_s',1.1),('timeline_time_before',float('nan')),('sensor_config_sha256','z'*64),
               ('thermal_intrinsics_px',dict(fx_px=9.,fy_px=1.,cx_px=1.,cy_px=1.,width=2,height=2))]
        p=self.ir/'frame_000000.json'; original=p.read_text()
        for key,value in cases:
            with self.subTest(key=key):
                p.write_text(original); self.edit_frame(self.ir,key,value)
                with self.assertRaises(ValueError): self.run_merge()
                self.assertFalse(self.out.exists())

    def test_nonfinite_plane_and_corrupt_metadata_rejected(self):
        p=self.ir/'frame_000000_apparent_t.npy'; a=np.load(p); a[0,0]=np.nan; np.save(p,a)
        with self.assertRaises(ValueError): self.run_merge()
        self.assertFalse(self.out.exists())
        a[0,0]=923.15;np.save(p,a)
        (self.ir/'common_metadata.json').write_text('{}')
        with self.assertRaises(ValueError): self.run_merge()
        self.assertFalse(self.out.exists())


if __name__=='__main__': unittest.main()
