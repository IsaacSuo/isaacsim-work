"""Conservative project-local still-water readiness check, no solver dependency."""
from collections import deque


def prewarm_damping(seconds, assisted_damping, hold_seconds=1.5,
                    release_seconds=.5, runtime_damping=.01):
    """Off-camera damping, smoothly restored before physical validation.

    None preserves the unassisted baseline. Return (coefficient, phase).
    This changes a native material coefficient, never particle velocities.
    """
    if assisted_damping is None:
        return runtime_damping, "normal_validation"
    if hold_seconds < 0 or release_seconds <= 0 or assisted_damping < runtime_damping:
        raise ValueError("Invalid assisted damping schedule")
    if seconds < hold_seconds:
        return assisted_damping, "assisted_relaxation"
    u = (seconds-hold_seconds)/release_seconds
    if u < 1:
        blend = u*u*(3-2*u)
        return assisted_damping+(runtime_damping-assisted_damping)*blend, "damping_release"
    return runtime_damping, "normal_validation"


class SettlingGate:
    """Require a continuous low-motion window; never alter particle state."""

    policy_id = "surface_still_v2_relaxed_speed"

    def __init__(self, minimum_seconds=2., window_seconds=.5):
        if minimum_seconds < 0 or window_seconds <= 0:
            raise ValueError("Invalid settling times")
        self.minimum_seconds = minimum_seconds
        self.window_seconds = window_seconds
        self.rows = deque()
        self.limits = dict(rms_speed_m_s=.035, speed_p99_m_s=.09,
                           max_speed_m_s=.5, regional_level_drift_m=.002)

    def observe(self, row):
        t = row["simulated_seconds"]
        if self.rows and t <= self.rows[-1]["simulated_seconds"]:
            raise ValueError("Settling sample times must increase")
        self.rows.append(row)
        cutoff = t-self.window_seconds
        while len(self.rows) > 1 and self.rows[1]["simulated_seconds"] <= cutoff+1e-9:
            self.rows.popleft()
        if t < self.minimum_seconds or t-self.rows[0]["simulated_seconds"] < self.window_seconds-1e-9:
            return False
        for sample in self.rows:
            if sample["outside_side_or_floor_count"] or sample["over_rim_particle_count"]:
                return False
            for field in ("rms_speed_m_s", "speed_p99_m_s", "max_speed_m_s"):
                if not 0 <= sample[field] <= self.limits[field]:
                    return False
            if len(sample["regional_level_p99_m"]) != 4:
                return False
        for i in range(4):
            levels = [sample["regional_level_p99_m"][i] for sample in self.rows]
            if not all(isinstance(v, (int, float)) and abs(v) < 1e6 for v in levels):
                return False
            if max(levels)-min(levels) > self.limits["regional_level_drift_m"]:
                return False
        return True
