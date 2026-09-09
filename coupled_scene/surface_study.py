"""Portable layout/motion specification; no fluid solver or particle creation."""
from __future__ import annotations

import math


def motion_state(motion, seconds):
    """Return local Y-up displacement and analytic velocity in metres/seconds.

    Evaluate this function at EVERY solver substep on the server. The 60 Hz
    tables in the handoff are inspection samples, not a solver timestep.
    """
    position, velocity = [0.0] * 3, [0.0] * 3
    if motion is None:
        return position, velocity
    start, stop = motion["start_s"], motion["stop_s"]
    if stop <= start:
        raise ValueError("Motion stop must follow start")
    if seconds <= start or seconds >= stop:
        return position, velocity
    duration = stop - start
    elapsed = seconds - start
    u = elapsed / duration
    envelope = 16 * u**2 * (1 - u)**2
    derivative = 32 * u * (1 - u) * (1 - 2 * u) / duration
    if motion["kind"] == "windowed_sine":
        omega = 2 * math.pi * motion["frequency_hz"]
        value = envelope * math.sin(omega * elapsed)
        speed = derivative * math.sin(omega * elapsed) + envelope * omega * math.cos(omega * elapsed)
    elif motion["kind"] == "pulse":
        value, speed = envelope, derivative
    else:
        raise ValueError(motion["kind"])
    axis, amplitude = motion["axis"], motion["amplitude_m"]
    position[axis], velocity[axis] = amplitude * value, amplitude * speed
    return position, velocity


def collision_panels(tank):
    """Boxes share the visible inner planes, with all thickness outward."""
    x, h, z = tank["inner_size_m"]
    t = tank["proxy_thickness_m"]
    return [
        ("Floor", [0, -t / 2, 0], [x + 2*t, t, z + 2*t]),
        ("Left", [-(x+t)/2, h/2, 0], [t, h, z+2*t]),
        ("Right", [(x+t)/2, h/2, 0], [t, h, z+2*t]),
        ("Front", [0, h/2, -(z+t)/2], [x, h, t]),
        ("Back", [0, h/2, (z+t)/2], [x, h, t]),
    ]


def actuator_spec(kind, tank):
    depth = tank["reference_water_depth_m"]
    if kind == "Plunger":
        return {"shape": "Cylinder", "centre": [0, depth+0.094, 0], "radius": 0.045, "height": 0.028}
    if kind == "Paddle":
        return {"shape": "Cube", "centre": [-0.69, 0.092, 0], "size": [0.018, 0.17, 0.82]}
    if kind is None:
        return None
    raise ValueError(kind)
