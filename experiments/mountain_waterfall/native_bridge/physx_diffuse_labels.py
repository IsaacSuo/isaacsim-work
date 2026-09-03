"""PhysX 5.9-compatible render-label recovery for native Diffuse particles.

PhysX classifies Diffuse dynamics from the number of primary-fluid neighbors:
fewer than 4 is spray, 4..7 is foam, and 8 or more is bubble.  The public
buffer does not preserve that temporary neighbor count, so this module
reconstructs the same rule from a fetched primary/Diffuse frame.
"""

from __future__ import annotations

from itertools import product

import numpy as np


LABEL_SPRAY = np.uint8(0)
LABEL_FOAM = np.uint8(1)
LABEL_BUBBLE = np.uint8(2)
LABEL_NAMES = ("spray", "foam", "bubble")
_NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def recover_labels(
    primary_positions: np.ndarray,
    diffuse_positions: np.ndarray,
    particle_contact_distance: float,
    *,
    maximum_neighbors: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(labels, neighbor_counts)` using PhysX 5.9's thresholds.

    This is the CPU reference implementation used by the micro gate.  It uses
    a radius-sized uniform grid rather than an O(N*M) distance matrix.  With
    full Diffuse advection enabled, the 27-cell search represents the native
    kernel's full-neighborhood rule.  Counts are capped at PhysX's limit of 16.
    """

    primary = np.asarray(primary_positions, dtype=np.float32)
    diffuse = np.asarray(diffuse_positions, dtype=np.float32)
    if primary.ndim != 2 or primary.shape[1] != 3:
        raise ValueError(f"primary_positions must have shape [N,3], got {primary.shape}")
    if diffuse.ndim != 2 or diffuse.shape[1] != 3:
        raise ValueError(f"diffuse_positions must have shape [M,3], got {diffuse.shape}")
    if not np.isfinite(primary).all() or not np.isfinite(diffuse).all():
        raise ValueError("particle positions must be finite")
    radius = float(particle_contact_distance)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("particle_contact_distance must be finite and positive")
    if not 8 <= maximum_neighbors <= 255:
        raise ValueError("maximum_neighbors must be in [8,255]")

    primary_cells = np.floor(primary / radius).astype(np.int64)
    cell_members: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(primary_cells):
        cell_members.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(index)

    radius_squared = np.float32(radius * radius)
    counts = np.zeros(len(diffuse), dtype=np.uint8)
    diffuse_cells = np.floor(diffuse / radius).astype(np.int64)
    for diffuse_index, base_cell in enumerate(diffuse_cells):
        candidates: list[int] = []
        bx, by, bz = map(int, base_cell)
        for ox, oy, oz in _NEIGHBOR_OFFSETS:
            members = cell_members.get((bx + ox, by + oy, bz + oz))
            if members:
                candidates.extend(members)
        if not candidates:
            continue
        delta = primary[np.asarray(candidates, dtype=np.intp)] - diffuse[diffuse_index]
        count = int(np.count_nonzero(np.einsum("ij,ij->i", delta, delta) < radius_squared))
        counts[diffuse_index] = min(count, maximum_neighbors)

    labels = np.empty(len(diffuse), dtype=np.uint8)
    labels[counts < 4] = LABEL_SPRAY
    labels[(counts >= 4) & (counts < 8)] = LABEL_FOAM
    labels[counts >= 8] = LABEL_BUBBLE
    return labels, counts


def label_histogram(labels: np.ndarray) -> dict[str, int]:
    values = np.asarray(labels, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > LABEL_BUBBLE):
        raise ValueError("labels must be a one-dimensional array containing only 0, 1, or 2")
    return {
        name: int(np.count_nonzero(values == index))
        for index, name in enumerate(LABEL_NAMES)
    }
