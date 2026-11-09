"""One-command launcher for the Crazyflie keyboard controller.

Run this file from Isaac Sim's Script Editor after opening a stage that
contains ``/World/Crazyflie``.  The launcher only wires the scene to the
existing reusable controller; movement math remains in the controller module.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any


DEFAULT_PRIM_PATH = "/World/Crazyflie"
DEFAULT_MINIMUM_HEIGHT = 0.5
PROJECT_ROOT = r"D:\IsaacSim_Demo"


def start_options(
    *,
    prim_path: str = DEFAULT_PRIM_PATH,
    minimum_height: float = DEFAULT_MINIMUM_HEIGHT,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    """Return the explicit options passed to the reusable controller."""
    options = {"prim_path": prim_path, "minimum_height": minimum_height}
    if task_instruction is not None:
        options['task_instruction'] = task_instruction
    return options


def _ensure_project_importable() -> None:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def start_crazyflie(
    *,
    prim_path: str = DEFAULT_PRIM_PATH,
    minimum_height: float = DEFAULT_MINIMUM_HEIGHT,
    motion_config=None,
    attitude_config=None,
    task_instruction: str | None = None,
):
    """Start one controller for the Crazyflie in the current Isaac stage."""
    import omni.usd

    _ensure_project_importable()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open. Open lesson_crazyflie.usd first.")

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(
            f"Cannot find Crazyflie prim {prim_path}. "
            "Check the Stage path or pass prim_path=... ."
        )
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(
            f"Prim {prim_path} does not have ArticulationRootAPI. "
            "Select the asset's Articulation Root path."
        )

    # Stop the old module-level controller before reload.  Reloading first
    # would replace the registry and leave the old frame callback subscribed.
    old_controller_module = sys.modules.get("scripts.drone_keyboard_controller")
    if old_controller_module is not None:
        old_controller_module.stop()

    import scripts.drone_keyboard_controller as drone_controller
    import scripts.drone_control_math as control_math
    import scripts.drone_physics_runtime as physics_runtime
    importlib.invalidate_caches()
    importlib.reload(control_math)
    importlib.reload(physics_runtime)
    drone_controller = importlib.reload(drone_controller)
    options = start_options(prim_path=prim_path, minimum_height=minimum_height,
                            task_instruction=task_instruction)
    if motion_config is not None:
        options['config'] = motion_config
    if attitude_config is not None:
        options['attitude_config'] = attitude_config
    controller = drone_controller.start(**options)
    print(f"[crazyflie] controller started for {prim_path}")
    print("[crazyflie] Play, click Drone Control. W/S/A/D/R/F/Q/E + I/K/J/L; X = freeze.")
    return controller


def stop_controller() -> None:
    """Stop the active controller without changing the stage."""
    _ensure_project_importable()
    import scripts.drone_keyboard_controller as drone_controller

    drone_controller.stop()
    print("[crazyflie] controller stopped")


if __name__ == "__main__":
    controller = start_crazyflie()
