"""Shared defaults for coupled-scene water; explicit experiment values win."""
import math

DEFAULT_VORTICITY_CONFINEMENT = 10.0
LEGACY_DECAY_VORTICITY_CONFINEMENT = 0.02


def water_vorticity(source):
    value = float(source.get('vorticity_confinement', DEFAULT_VORTICITY_CONFINEMENT))
    if not math.isfinite(value) or value < 0:
        raise ValueError('Vorticity confinement must be finite and nonnegative')
    return value


def surface_vorticity(*, decay_test=False, decay_override=None):
    # The v6-v11 diagnostic arms retain their historical baseline, not the
    # evolving production default. A/B overrides remain explicit.
    if decay_override is not None and not decay_test:
        raise ValueError('Decay override requires a decay diagnostic')
    value = (LEGACY_DECAY_VORTICITY_CONFINEMENT if decay_test
             else DEFAULT_VORTICITY_CONFINEMENT) if decay_override is None else decay_override
    return water_vorticity({'vorticity_confinement': value})
