"""CPU diagnostics for shared-state wave-decay experiments (not native density)."""
import numpy as np


def decay_threshold_times(report):
    """Common 10 Hz samples only; require no later observed rebound and >=0.5 s follow-up."""
    rows=[(0.,report['decay']['initial_metrics']['height_p95_p05_mm'])]
    rows += [(r['simulated_seconds'],r['wave_metrics']['height_p95_p05_mm'])
             for r in report['rows'] if r['step'] % (report['hz']//10)==0]
    end=rows[-1][0]
    thresholds={}
    for value in (10,5,2,1,.5):
        first=next((t for t,h in rows if h<=value),None)
        sustained=next((t for i,(t,h) in enumerate(rows)
            if end-t>=.5-1e-8 and all(height<=value for _,height in rows[i:])),None)
        thresholds[str(value)]=dict(first_below_seconds=first,confirmed_below_seconds=sustained,
            confirmation='not_confirmed_within_recording' if sustained is None else 'no_later_observed_rebound_and_at_least_0.5s_followup')
    return dict(observed_seconds=end,sample_interval_seconds=.1,thresholds_mm=thresholds,
                metric='5 cm column p99 height, interior p95-p05; not exact crest-to-trough wave amplitude')


def wave_metrics(positions, velocities, origin, size):
    """5 cm columns; p99 heights reject isolated spray; exclude edge columns for wave amplitude."""
    nx, nz = round(size[0] / .05), round(size[2] / .05)
    ix = np.clip(((positions[:, 0] - origin[0] + size[0]/2) / size[0] * nx).astype(int), 0, nx-1)
    iz = np.clip(((positions[:, 2] - origin[2] + size[2]/2) / size[2] * nz).astype(int), 0, nz-1)
    index = ix * nz + iz
    counts = np.bincount(index, minlength=nx*nz)
    if np.any(counts == 0):
        raise ValueError('Empty height column; wave/volume proxy invalid')
    means = np.column_stack([np.bincount(index, weights=velocities[:, k], minlength=nx*nz)/counts for k in range(3)])
    mean_v2 = float(np.mean(np.sum(velocities.astype(np.float64)**2, axis=1)))
    coherent_v2 = float(np.sum(counts * np.sum(means**2, axis=1))/len(positions))
    order = np.argsort(index)
    heights = np.array([np.quantile(column, .99) - origin[1]
                        for column in np.split(positions[order, 1], np.cumsum(counts)[:-1])]).reshape(nx, nz)
    interior = heights[1:-1, 1:-1]
    return dict(height_std_mm=float(np.std(interior)*1000),
                height_p95_p05_mm=float(np.ptp(np.quantile(interior, [.05, .95]))*1000),
                coarse_flow_rms_m_s=coherent_v2**.5,
                within_column_residual_rms_m_s=max(0., mean_v2-coherent_v2)**.5,
                kinetic_energy_per_mass_j_kg=.5*mean_v2,
                gravitational_energy_per_mass_j_kg=9.81*float(np.mean(positions[:, 1].astype(np.float64)-origin[1])),
                mean_column_height_m=float(heights.mean()),
                height_volume_proxy_m3=float(heights.mean()*size[0]*size[2]),
                note='Residual includes real small-scale flow, not pure noise. Height volume is a proxy, not native density or certified volume.')
