import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import contextlib
import json
import tempfile
from pathlib import Path
from scripts import preview_scheduler


class OnDemandIRTests(unittest.TestCase):
    def test_frozen_attachment_warms_before_validation_and_restores_patches(self):
        module=self.module()
        self.assertTrue(hasattr(module,'frozen_initialization'))
        calls=[]
        step=lambda **kw:calls.append(kw)
        orchestrator=SimpleNamespace(step=step)
        class Reader:
            def attach(self):raise RuntimeError('empty')
            def detach(self):calls.append('detach')
        readers=SimpleNamespace(AovReader=Reader)
        with module.frozen_initialization(orchestrator,readers,SimpleNamespace(update=lambda:None)):
            orchestrator.step(rt_subframes=1)
            with self.assertRaises(RuntimeError):readers.AovReader().attach()
        self.assertEqual(calls[0]['delta_time'],0.)
        self.assertTrue(calls[0]['pause_timeline'])
        self.assertIn('detach',calls)
        self.assertIs(readers.AovReader,Reader)
        self.assertIs(orchestrator.step,step)

    def test_inspection_history_survives_Stop_close(self):
        module=self.module();self.assertTrue(hasattr(module,'bind_inspection_history'))
        from scripts.trajectory_recorder import TrajectoryRecorder
        with tempfile.TemporaryDirectory() as folder:
            r=TrajectoryRecorder.create(folder,{'trajectory_source':'synthetic_control_probe'})
            events=[];module.bind_inspection_history(r,events)
            events.append({'status':'COMPLETED'})
            r.close()
            self.assertEqual(json.loads((r.run_dir/'run_metadata.json').read_text())['preview_inspection_events'],events)

    def test_RGB_cruise_establishes_thermal_clock_before_first_request(self):
        from scripts.start_crazyflie_dual_camera_view import DualCameraPreview
        p=DualCameraPreview(None,None,None,None,None,width=160,height=128,cleanup=None,metadata={},show_ui=False)
        self.assertTrue(hasattr(p,'sample_cruise_clock'))
        p.sample_cruise_clock(.02);p.sample_cruise_clock(20.)
        self.assertAlmostEqual(p.clock.sample(30.)[0],29.98)
    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('scripts.on_demand_ir'),'on-demand lifecycle missing')
        from scripts import on_demand_ir
        return on_demand_ir

    def fixture(self,fail=False):
        calls=[]
        class Timeline:
            playing=True
            def is_playing(self):return self.playing
            def pause(self):self.playing=False;calls.append('pause')
            def play(self):self.playing=True;calls.append('play')
            def commit(self):pass
            def get_current_time(self):return 1.25
        class Product:
            def destroy(self):calls.append('destroy')
        class IR:
            _render_product=None
            def open(self,**kwargs):
                self._render_product=Product();calls.append(('open',kwargs))
            def close(self):calls.append('close')
        class Preview:
            thermal=IR()
            def inspection_initialization(self,app):return contextlib.nullcontext()
            def capture(self,**kwargs):
                self.assert_paused=not timeline.is_playing()
                calls.append('capture')
                if fail:raise RuntimeError('capture failed')
                return None,object(),{'simulation_time_s':1.25,'thermal_coverage':{'complete_for_this_view':True}}
            def publish_thermal(self,*args):calls.append('publish')
        timeline=Timeline();p=Preview()
        app=SimpleNamespace(update=lambda:calls.append('update'))
        controller=SimpleNamespace(_suspend=lambda:calls.append('clear_keys'))
        return p,timeline,app,controller,calls

    def test_snapshot_is_frozen_and_thermal_resources_released_before_resume(self):
        module=self.module();p,t,a,c,calls=self.fixture()
        event=module.capture_inspection(p,t,a,c)
        self.assertTrue(p.assert_paused)
        self.assertIn(('open',{'settle_frames':4,'rt_subframes':1}),calls)
        self.assertLess(calls.index('destroy'),calls.index('play'))
        self.assertIsNone(p.thermal._render_product)
        self.assertEqual(event['simulation_time_before'],event['simulation_time_after'])
        self.assertTrue(event['thermal_resources_released'])

    def test_failed_request_still_releases_resources_and_restores_timeline(self):
        module=self.module();p,t,a,c,calls=self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError,'capture failed'):module.capture_inspection(p,t,a,c)
        self.assertTrue(t.playing)
        self.assertIn('destroy',calls)
        self.assertIsNone(p.thermal._render_product)

    def test_capture_preserves_existing_paused_state(self):
        module=self.module();p,t,a,c,calls=self.fixture();t.playing=False
        module.capture_inspection(p,t,a,c)
        self.assertFalse(t.playing)
        self.assertNotIn('play',calls)

    def test_launcher_defaults_to_on_demand_instead_of_periodic_thermal(self):
        from scripts.factory_bc_pilot import build_launch
        launch=build_launch('factory-bc-pilot-demand-contract')
        args=launch['preview_args']
        self.assertEqual(args[args.index('--live-refresh')+1],'on-demand')

    def test_H_is_focus_gated_and_key_repeat_does_not_queue_multiple_requests(self):
        from scripts.drone_keyboard_controller import DroneKeyboardController
        events=SimpleNamespace(KEY_PRESS=1,KEY_RELEASE=2)
        fake=SimpleNamespace(KeyboardEventType=events)
        with patch.dict(sys.modules,{'carb':SimpleNamespace(input=fake),'carb.input':fake}):
            c=DroneKeyboardController();c._running=c._ready=True
            c._playing=lambda:True;c._has_control_focus=lambda:False
            calls=[];c.inspection_callback=lambda:calls.append('H')
            press=SimpleNamespace(input=SimpleNamespace(name='H'),type=1)
            release=SimpleNamespace(input=SimpleNamespace(name='H'),type=2)
            c._on_keyboard_event(press);self.assertEqual(calls,[])
            c._has_control_focus=lambda:True
            c._on_keyboard_event(press);c._on_keyboard_event(press)
            self.assertEqual(calls,['H'])
            c._on_keyboard_event(release);c._on_keyboard_event(press)
            self.assertEqual(calls,['H','H'])
            self.assertNotIn('H',c.keys.pressed)

if __name__=='__main__':unittest.main()
