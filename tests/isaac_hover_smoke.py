"""Run with D:\\isaacsim\\python.bat, not normal Python.

Opens the saved scene in a separate headless process; never saves it.
Keyboard input is injected for deterministic physics verification (not UI QA).
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaacsim import SimulationApp

app = SimulationApp({'headless': True})
try:
    import omni.usd
    import omni.timeline
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate('isaacsim.core.experimental.prims', True)
    from isaacsim.core.simulation_manager import SimulationManager
    context = omni.usd.get_context()
    scene = str(Path(__file__).resolve().parents[1] / 'lesson_crazyflie.usd')
    if not context.open_stage(scene):
        raise RuntimeError('Could not open saved scene')
    for _ in range(30):
        app.update()
    from scripts.start_crazyflie_controller import start_crazyflie
    from scripts.drone_control_math import rpy_from_orientation_wxyz
    controller = start_crazyflie()
    # Headless automation has no physical keyboard focus.
    controller._has_control_focus = lambda: True
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(120):
        app.update()
        if controller._ready:
            break
    assert controller._ready, controller.diagnostics()
    dt = SimulationManager.get_physics_dt()

    def hold_test():
        target = controller.position[2]
        worst = 0.
        def sample(step, total):
            nonlocal worst
            state = controller.diagnostics()
            assert state['ready'] and state['running'], state
            worst = max(worst, abs(state['physics']['position'][2] - target))
        SimulationManager.step(steps=math.ceil(30/dt), callback=sample)
        assert worst <= .01, worst
        print(f'HOVER_30S_MAX_ERROR={worst:.8f}', flush=True)

    hold_test()
    controller.keys.press('R')
    SimulationManager.step(steps=math.ceil(1/dt))
    controller.keys.release('R')
    SimulationManager.step(steps=math.ceil(2/dt))
    hold_test()
    controller.keys.press('F')
    SimulationManager.step(steps=math.ceil(1/dt))
    controller.keys.release('F')
    SimulationManager.step(steps=math.ceil(2/dt))
    hold_test()
    for key, axis, sign in [('I', 1, 1), ('K', 1, -1), ('J', 0, -1), ('L', 0, 1)]:
        controller.keys.press(key)
        SimulationManager.step(steps=math.ceil(2/dt))
        state = controller.diagnostics()
        assert state['rpy_degrees'][axis]*sign > 19., state
        actual_rpy = rpy_from_orientation_wxyz(state['physics']['orientation'])
        assert math.degrees(actual_rpy[axis])*sign > 19., state
        controller.keys.release(key)
        SimulationManager.step(steps=math.ceil(3/dt))
        assert abs(controller.diagnostics()['rpy_degrees'][axis]) < .01
    # Turn left 90 degrees, then forward must move along the new heading.
    controller.keys.press('Q')
    SimulationManager.step(steps=math.ceil(2/dt))
    controller.keys.release('Q')
    SimulationManager.step(steps=math.ceil(1/dt))
    initial = controller.position
    heading = controller.yaw
    controller.keys.press('W')
    controller.keys.press('I')
    SimulationManager.step(steps=math.ceil(1/dt))
    controller.emergency_stop()
    frozen = controller.diagnostics()['physics']
    SimulationManager.step(steps=math.ceil(1/dt))
    current = controller.diagnostics()['physics']
    for field in ('position', 'orientation'):
        assert max(abs(a-b) for a,b in zip(frozen[field], current[field])) < 1e-5
    dx, dy = controller.position[0]-initial[0], controller.position[1]-initial[1]
    assert dx*math.cos(heading) + dy*math.sin(heading) > .5
    assert abs(dx*math.sin(heading) - dy*math.cos(heading)) < .01
    assert abs(controller.position[2]-initial[2]) < .01
    controller._emergency_held = False
    # Pause clears live input via the real timeline subscription.
    controller.keys.press('R')
    timeline.pause()
    app.update()
    assert not controller.keys.pressed
    timeline.play()
    app.update()
    timeline.stop()
    app.update()
    timeline.play()
    for _ in range(120):
        app.update()
        if controller._ready:
            break
    assert controller._ready, controller.diagnostics()
    hold_test()
    previous = controller
    controller = start_crazyflie()
    controller._has_control_focus = lambda: True
    for _ in range(10):
        app.update()
    assert not previous._running and not previous._callback_ids
    assert controller._ready, controller.diagnostics()
    hold_test()
    controller.stop()
    print('ISAAC_HOVER_SMOKE_PASS', flush=True)
finally:
    app.close()
