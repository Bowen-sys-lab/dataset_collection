"""Pure math for smooth keyboard-driven drone motion."""

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class AttitudeConfig:
    """Tilt limits in radians, radians/second and radians/second squared."""

    max_angle: float = math.radians(20)
    max_rate: float = math.radians(45)
    acceleration: float = math.radians(90)

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.max_angle, self.max_rate, self.acceleration)):
            raise ValueError("Attitude limits must be finite and positive")
        if self.max_angle >= math.pi / 2:
            raise ValueError("Tilt limit must be below 90 degrees")


def attitude_command_from_keys(keys: Iterable[str]) -> tuple[float, float]:
    pressed = {k.upper() for k in keys}
    # +X forward, +Y left, +Z up: positive roll lowers the right side;
    # positive pitch lowers the nose.
    return (float('L' in pressed) - float('J' in pressed),
            float('I' in pressed) - float('K' in pressed))


def orientation_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> tuple:
    """Rz(yaw) Ry(pitch) Rx(roll), in Isaac's WXYZ convention."""
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cy*cp*cr + sy*sp*sr, cy*cp*sr - sy*sp*cr,
            cy*sp*cr + sy*cp*sr, sy*cp*cr - cy*sp*sr)


def rpy_from_orientation_wxyz(q: Iterable[float]) -> tuple:
    w, x, y, z = q
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Orientation must be a finite nonzero quaternion")
    w, x, y, z = (v/norm for v in (w, x, y, z))
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def step_tilt(angle: float, rate: float, command: float, dt: float,
              config: AttitudeConfig = AttitudeConfig()) -> tuple[float, float]:
    """Approach a bounded tilt target; no input means return to level.

    Small internal steps limit frame-rate dependence. The braking-speed
    envelope slows the axis before it reaches the target.
    """
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and non-negative")
    target = max(-1., min(1., command)) * config.max_angle
    count = max(1, math.ceil(dt * 240))
    h = dt / count
    for _ in range(count):
        error = target - angle
        braking_speed = math.sqrt((config.acceleration*h)**2 + 2*config.acceleration*abs(error)) - config.acceleration*h
        desired = math.copysign(min(config.max_rate, braking_speed), error)
        rate = _move_towards(rate, desired, config.acceleration*h)
        next_angle = angle + rate*h
        if error * (target-next_angle) < 0 or (abs(error) < 1e-7 and abs(rate) < 1e-5):
            next_angle, rate = target, 0.
        # Allow a previously tilted model to return from outside the limit,
        # but never move farther outside the allowed range.
        if abs(angle) <= config.max_angle and abs(next_angle) > config.max_angle:
            next_angle, rate = math.copysign(config.max_angle, next_angle), 0.
        angle = next_angle
    return angle, rate


@dataclass(frozen=True)
class MotionConfig:
    max_horizontal_speed: float = 1.0
    max_vertical_speed: float = 0.5
    max_yaw_speed: float = math.radians(45.0)
    horizontal_acceleration: float = 2.0
    vertical_acceleration: float = 1.0
    yaw_acceleration: float = math.radians(90.0)


@dataclass(frozen=True)
class BodyCommand:
    forward: float = 0.0
    right: float = 0.0
    vertical: float = 0.0
    yaw: float = 0.0


def body_command_from_keys(keys: Iterable[str]) -> BodyCommand:
    pressed = {key.upper() for key in keys}
    forward = float("W" in pressed) - float("S" in pressed)
    # This imported model's local +Y points left, so -Y is the user's right.
    right = float("A" in pressed) - float("D" in pressed)
    vertical = float("R" in pressed) - float("F" in pressed)
    # The model's +Z rotation is counter-clockwise in XY: Q is left, E is right.
    yaw = float("Q" in pressed) - float("E" in pressed)
    forward, right = clamp_horizontal(forward, right)
    return BodyCommand(forward, right, vertical, yaw)


def world_heading_from_yaw(yaw: float) -> tuple[float, float]:
    """Return the unit XY vector that points along the drone nose."""
    return math.cos(yaw), math.sin(yaw)


def root_orientation_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Return a world-Z yaw quaternion in Isaac articulation's WXYZ order."""
    return math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)


def target_velocity_from_command(
    command: BodyCommand,
    config: MotionConfig = MotionConfig(),
) -> tuple[float, float, float, float]:
    """Convert a normalized body command into target speeds."""
    return (
        command.forward * config.max_horizontal_speed,
        command.right * config.max_horizontal_speed,
        command.vertical * config.max_vertical_speed,
        command.yaw * config.max_yaw_speed,
    )


def clamp_horizontal(forward: float, right: float) -> tuple[float, float]:
    length = math.hypot(forward, right)
    if length <= 1.0 or length == 0.0:
        return forward, right
    return forward / length, right / length


def _move_towards(current: float, target: float, maximum_delta: float) -> float:
    difference = target - current
    if abs(difference) <= maximum_delta:
        return target
    return current + math.copysign(maximum_delta, difference)


def smooth_velocity(
    current: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    dt: float,
    config: MotionConfig = MotionConfig(),
) -> tuple[float, float, float, float]:
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    return (
        _move_towards(current[0], target[0], config.horizontal_acceleration * dt),
        _move_towards(current[1], target[1], config.horizontal_acceleration * dt),
        _move_towards(current[2], target[2], config.vertical_acceleration * dt),
        _move_towards(current[3], target[3], config.yaw_acceleration * dt),
    )


def integrate_pose(
    position: tuple[float, float, float],
    yaw: float,
    velocity: tuple[float, float, float, float],
    dt: float,
    minimum_height: float,
) -> tuple[tuple[float, float, float], float]:
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    x, y, z = position
    forward, right, vertical, yaw_rate = velocity
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = forward * cos_yaw - right * sin_yaw
    world_y = forward * sin_yaw + right * cos_yaw
    next_position = (x + world_x * dt, y + world_y * dt, max(minimum_height, z + vertical * dt))
    return next_position, yaw + yaw_rate * dt
