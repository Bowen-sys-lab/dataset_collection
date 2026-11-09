import unittest
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from scripts import drone_keyboard_controller as bridge
from scripts import start_crazyflie_controller as launcher
from scripts.start_crazyflie_controller import (
    DEFAULT_MINIMUM_HEIGHT,
    DEFAULT_PRIM_PATH,
    start_options,
)


class CrazyflieLauncherTests(unittest.TestCase):
    def setUp(self):
        self.prim = SimpleNamespace(IsValid=lambda: True, HasAPI=lambda api: True)
        self.stage = SimpleNamespace(GetPrimAtPath=lambda path: self.prim)
        omni = ModuleType("omni")
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: self.stage)
        omni.usd = usd
        pxr = ModuleType("pxr")
        pxr.UsdPhysics = SimpleNamespace(ArticulationRootAPI=object())
        self.modules = patch.dict(sys.modules, {"omni": omni, "omni.usd": usd, "pxr": pxr})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(bridge.stop)

    def test_restart_releases_old_subscription_before_reloading_module(self):
        # Keep the real controller stop/start lifecycle; substitute only Kit
        # startup and reload, which would recreate the class under that patch.
        active = set()

        class Subscription:
            def reset(self):
                active.discard(self)

        def kit_start(controller):
            token = Subscription()
            active.add(token)
            controller._update_sub = token
            return controller

        def reload_module(module):
            self.assertEqual(active, set(), "Old callback still active at reload")
            # Reloading the real module also clears its module-level registry.
            module._active_controller = None
            return module

        with patch.object(bridge.DroneKeyboardController, "start", kit_start), patch.object(
            launcher.importlib, "reload", side_effect=reload_module
        ):
            first = launcher.start_crazyflie()
            second = launcher.start_crazyflie()
            self.assertIsNot(first, second)
            self.assertEqual(len(active), 1)
            self.assertEqual(second.prim_path, "/World/Crazyflie")
            launcher.stop_controller()
            launcher.stop_controller()
            self.assertEqual(active, set())

    def test_missing_stage_reports_error_before_stopping_existing_controller(self):
        self.stage = None
        with patch.object(bridge, "stop") as stop:
            with self.assertRaisesRegex(RuntimeError, "No USD stage"):
                launcher.start_crazyflie()
            stop.assert_not_called()

    def test_non_articulation_is_rejected_before_controller_start(self):
        self.prim.HasAPI = lambda api: False
        with patch.object(bridge, "start") as start:
            with self.assertRaisesRegex(RuntimeError, "ArticulationRootAPI"):
                launcher.start_crazyflie()
            start.assert_not_called()

    def test_launcher_targets_imported_crazyflie_articulation(self):
        self.assertEqual(DEFAULT_PRIM_PATH, "/World/Crazyflie")

    def test_start_options_are_explicit_and_overridable(self):
        self.assertEqual(
            start_options(),
            {"prim_path": "/World/Crazyflie", "minimum_height": DEFAULT_MINIMUM_HEIGHT},
        )
        self.assertEqual(
            start_options(prim_path="/World/OtherDrone", minimum_height=0.8),
            {"prim_path": "/World/OtherDrone", "minimum_height": 0.8},
        )

    def test_start_options_forward_the_task_sentence_to_the_control_window(self):
        options = start_options(task_instruction='请在学校校园寻找 Spot')
        self.assertEqual(options.get('task_instruction'), '请在学校校园寻找 Spot')


if __name__ == "__main__":
    unittest.main()
