"""Sparse, outlier-resistant water-body partitioning and domain construction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_indices(indices):
    values = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_domain_partition_contract(
    manifest_path,
    *,
    source_manifest_path,
    source_manifest,
    requested_spacing,
    body_id=None,
):
    """Validate a domain partition and resolve one explicit water body.

    This loader deliberately has no SciPy or mesh dependencies so the same
    provenance contract can be consumed by field builders, Splashsurf and
    Blender-side tooling.
    """

    manifest_path = Path(manifest_path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") not in {1, 2}
        or manifest.get("product") != "whitewater_v6_domain_partition"
        or manifest.get("valid") is not True
    ):
        raise ValueError("Domain partition is unsupported or has not passed audit")
    source = manifest.get("source", {})
    if source.get("manifest_sha256") != sha256_file(source_manifest_path):
        raise ValueError("Domain partition does not own the current source manifest")
    expected_particle_ids = source_manifest.get("particle_ids")
    if source.get("particle_id_contract") != expected_particle_ids:
        raise ValueError("Domain/source particle ID contracts differ")
    particle_count = int(source_manifest["particle_count"])

    bodies = manifest.get("bodies", [])
    if not bodies:
        raise ValueError("Domain partition contains no water bodies")
    body_ids = [item.get("body_id") for item in bodies]
    if len(set(body_ids)) != len(body_ids) or any(value is None for value in body_ids):
        raise ValueError("Domain body IDs are missing or duplicated")
    if body_id is None:
        if len(bodies) != 1:
            raise ValueError("--domain-body-id is required for a multi-water partition")
        body_index = 0
    else:
        if body_id not in body_ids:
            raise ValueError(f"Unknown domain body ID {body_id!r}")
        body_index = body_ids.index(body_id)
    body = bodies[body_index]
    domain = body.get("domain", {})
    origin = np.asarray(domain.get("origin"), dtype=np.float64)
    maximum = np.asarray(domain.get("maximum"), dtype=np.float64)
    shape = np.asarray(domain.get("shape"), dtype=np.int64)
    spacing = float(domain.get("spacing", np.nan))
    if (
        origin.shape != (3,)
        or maximum.shape != (3,)
        or shape.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(maximum).all()
        or np.any(shape < 3)
        or not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.allclose(
            maximum,
            origin + spacing * (shape - 1),
            rtol=0.0,
            atol=1.0e-12,
        )
        or int(np.prod(shape, dtype=np.int64)) != int(domain.get("cell_count", -1))
    ):
        raise ValueError("Selected body has an invalid domain")
    if not np.isclose(spacing, float(requested_spacing), rtol=0.0, atol=1.0e-12):
        raise ValueError("--voxel-size must exactly match the body domain spacing")

    membership_metadata = manifest.get("membership", {})
    membership_path = manifest_path.parent / str(membership_metadata.get("file", ""))
    if (
        not membership_path.is_file()
        or sha256_file(membership_path) != membership_metadata.get("sha256")
    ):
        raise ValueError("Domain membership file is missing or has the wrong hash")
    with np.load(membership_path, allow_pickle=False) as cache:
        required = {
            "schema",
            "particle_body_id",
            "reference_sample_index",
            "particle_id_contract_sha256",
        }
        if not required.issubset(cache.files) or int(cache["schema"]) != 1:
            raise ValueError("Unsupported domain membership payload")
        membership = np.asarray(cache["particle_body_id"], dtype=np.int32)
        particle_id_hash = str(np.asarray(cache["particle_id_contract_sha256"]).item())
        reference_sample_index = int(cache["reference_sample_index"])
    if membership.shape != (particle_count,):
        raise ValueError("Domain membership particle count changed")
    if particle_id_hash != expected_particle_ids.get("sha256"):
        raise ValueError("Membership particle ID contract changed")
    if np.any(membership < -1) or np.any(membership >= len(bodies)):
        raise ValueError("Membership contains an invalid body index")
    for index, item in enumerate(bodies):
        indices = np.flatnonzero(membership == index)
        if (
            len(indices) != int(item.get("particle_count", -1))
            or sha256_indices(indices) != item.get("particle_indices_sha256")
        ):
            raise ValueError(f"Stable membership mismatch for {item['body_id']}")
    sample_rows = {int(item["sample_index"]): item for item in manifest.get("samples", [])}
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "membership_path": membership_path,
        "membership_sha256": membership_metadata["sha256"],
        "membership": membership,
        "reference_sample_index": reference_sample_index,
        "body": body,
        "body_index": body_index,
        "body_particle_indices": np.flatnonzero(membership == body_index),
        "sample_rows": sample_rows,
    }


def load_particle_classification(contract, sample_index, expected_particle_count):
    """Load and fully reconcile one per-frame core/secondary ledger."""

    sample_index = int(sample_index)
    sample = contract["sample_rows"].get(sample_index)
    if sample is None:
        raise ValueError(f"Domain partition does not cover source sample {sample_index}")
    metadata = sample.get("classification")
    if not metadata:
        raise ValueError(
            "Domain partition predates consumable particle classifications"
        )
    path = contract["manifest_path"].parent / metadata["file"]
    if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
        raise ValueError(f"Particle classification hash mismatch for sample {sample_index}")
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "schema",
            "source_sample_index",
            "particle_state",
            "outside_body_domain",
            "outside_combined_domain",
            "particle_id_contract_sha256",
        }
        if not required.issubset(cache.files):
            raise ValueError("Particle classification is incomplete")
        classification_schema = int(cache["schema"])
        if classification_schema not in {1, 2} or int(cache["source_sample_index"]) != sample_index:
            raise ValueError("Particle classification metadata changed")
        state = np.asarray(cache["particle_state"], dtype=np.uint8)
        outside_body = np.asarray(cache["outside_body_domain"], dtype=np.uint8)
        outside_combined = np.asarray(cache["outside_combined_domain"], dtype=np.uint8)
        particle_id_hash = str(np.asarray(cache["particle_id_contract_sha256"]).item())
        if classification_schema == 2:
            lifecycle_required = {
                "instantaneous_detached",
                "new_secondary_handoff",
                "pending_return_to_core",
            }
            if not lifecycle_required.issubset(cache.files):
                raise ValueError("Lifecycle classification arrays are incomplete")
            instantaneous_detached = np.asarray(
                cache["instantaneous_detached"], dtype=np.uint8
            )
            new_handoff = np.asarray(cache["new_secondary_handoff"], dtype=np.uint8)
            pending_return = np.asarray(cache["pending_return_to_core"], dtype=np.uint8)
        else:
            instantaneous_detached = (state == 1).astype(np.uint8)
            new_handoff = np.zeros_like(state, dtype=np.uint8)
            pending_return = np.zeros_like(state, dtype=np.uint8)
    expected_shape = (int(expected_particle_count),)
    if any(
        value.shape != expected_shape
        for value in (
            state,
            outside_body,
            outside_combined,
            instantaneous_detached,
            new_handoff,
            pending_return,
        )
    ):
        raise ValueError("Particle classification count changed")
    maximum_state = 3 if classification_schema == 2 else 2
    if (
        np.any(state > maximum_state)
        or any(
            np.any(value > 1)
            for value in (
                outside_body,
                outside_combined,
                instantaneous_detached,
                new_handoff,
                pending_return,
            )
        )
    ):
        raise ValueError("Particle classification contains invalid codes")
    expected_id_hash = contract["manifest"]["source"]["particle_id_contract"]["sha256"]
    if particle_id_hash != expected_id_hash:
        raise ValueError("Particle classification ID contract changed")
    membership = contract["membership"]
    if not np.array_equal(state == 2, membership == -1):
        raise ValueError("Unassigned state disagrees with stable body membership")
    if classification_schema == 2:
        if np.any((new_handoff != 0) & (state != 3)):
            raise ValueError("New handoff event does not establish secondary ownership")
        if np.any((pending_return != 0) & (state != 0)):
            raise ValueError("Pending return event does not restore core ownership")
        if np.any((state == 1) & (instantaneous_detached == 0) & (outside_body == 0)):
            raise ValueError("Pending proxy has no detached/outside cause")

    index_sets = {
        "core": np.flatnonzero(state == 0),
        "unassigned": np.flatnonzero(state == 2),
        "secondary_candidate": np.flatnonzero((state != 0) | (outside_body != 0)),
        "outside_combined_domain": np.flatnonzero(outside_combined != 0),
    }
    if classification_schema == 2:
        index_sets.update(
            {
                "instantaneous_detached": np.flatnonzero(
                    instantaneous_detached != 0
                ),
                "pending_detached": np.flatnonzero(state == 1),
                "secondary_owned": np.flatnonzero(state == 3),
                "new_secondary_handoff": np.flatnonzero(new_handoff != 0),
                "pending_return_to_core": np.flatnonzero(pending_return != 0),
            }
        )
        count_keys = {
            "core": "core_particles",
            "instantaneous_detached": "instantaneous_detached_particles",
            "pending_detached": "pending_detached_particles",
            "unassigned": "unassigned_particles",
            "secondary_owned": "secondary_owned_particles",
            "new_secondary_handoff": "new_secondary_handoff_particles",
            "pending_return_to_core": "pending_return_to_core_particles",
            "secondary_candidate": "secondary_candidate_particles",
        }
    else:
        index_sets["detached"] = np.flatnonzero(state == 1)
        count_keys = {
            "core": "core_particles",
            "detached": "detached_particles",
            "unassigned": "unassigned_particles",
            "secondary_candidate": "secondary_candidate_particles",
        }
    for label, key in count_keys.items():
        indices = index_sets[label]
        if (
            len(indices) != int(metadata.get(key, -1))
            or sha256_indices(indices) != metadata.get(f"{key[:-1]}_indices_sha256")
        ):
            raise ValueError(f"Particle classification ledger mismatch: {label}")
    outside_indices = index_sets["outside_combined_domain"]
    if (
        len(outside_indices) != int(sample.get("outside_combined_domain_particles", -1))
        or sha256_indices(outside_indices)
        != sample.get("outside_combined_domain_particle_indices_sha256")
    ):
        raise ValueError("Combined-domain outside ledger mismatch")
    body_reports = {item["body_id"]: item for item in sample.get("bodies", [])}
    for body_index, body in enumerate(contract["manifest"]["bodies"]):
        body_report = body_reports.get(body["body_id"])
        if body_report is None:
            raise ValueError(f"Missing classification ledger for {body['body_id']}")
        body_indices = np.flatnonzero(membership == body_index)
        detached = body_indices[instantaneous_detached[body_indices] != 0]
        outside = body_indices[outside_body[body_indices] != 0]
        core_outside = body_indices[
            (instantaneous_detached[body_indices] == 0)
            & (outside_body[body_indices] != 0)
        ]
        for label, indices, count_key, hash_key in (
            (
                "detached",
                detached,
                "detached_particle_count",
                "detached_particle_indices_sha256",
            ),
            (
                "outside body domain",
                outside,
                "outside_body_domain_particles",
                "outside_body_domain_particle_indices_sha256",
            ),
            (
                "core outside body domain",
                core_outside,
                "core_outside_body_domain_particles",
                "core_outside_body_domain_particle_indices_sha256",
            ),
        ):
            if (
                len(indices) != int(body_report.get(count_key, -1))
                or sha256_indices(indices) != body_report.get(hash_key)
            ):
                raise ValueError(
                    f"{body['body_id']} {label} classification ledger mismatch"
                )

    selected = contract["body_particle_indices"]
    selected_state = state[selected]
    selected_outside = outside_body[selected] != 0
    return {
        "path": path,
        "sha256": metadata["sha256"],
        "state": state,
        "classification_schema": classification_schema,
        "instantaneous_detached": instantaneous_detached,
        "new_secondary_handoff": new_handoff,
        "pending_return_to_core": pending_return,
        "outside_body_domain": outside_body,
        "outside_combined_domain": outside_combined,
        "selected_core_indices": selected[(selected_state == 0) & ~selected_outside],
        "selected_detached_indices": selected[
            instantaneous_detached[selected] != 0
        ],
        "selected_pending_indices": selected[selected_state == 1],
        "selected_owned_indices": selected[selected_state == 3],
        "selected_new_handoff_indices": selected[new_handoff[selected] != 0],
        "selected_pending_return_indices": selected[
            pending_return[selected] != 0
        ],
        "selected_outside_indices": selected[selected_outside],
        "selected_secondary_indices": selected[
            (selected_state != 0) | selected_outside
        ],
        "sample": sample,
    }


def _validated_positions(positions):
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    if len(positions) == 0 or not np.isfinite(positions).all():
        raise ValueError("positions must be non-empty and finite")
    return positions


def sparse_particle_components(positions, cell_size, connectivity=26):
    """Label particle components without allocating a dense world-sized grid."""

    positions = _validated_positions(positions)
    cell_size = float(cell_size)
    if not np.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("cell_size must be finite and positive")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be 6, 18 or 26")
    coordinates = np.floor(positions / cell_size).astype(np.int64)
    voxels, inverse, voxel_particle_counts = np.unique(
        coordinates, axis=0, return_inverse=True, return_counts=True
    )
    parent = np.arange(len(voxels), dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    voxel_lookup = {tuple(map(int, coordinate)): index for index, coordinate in enumerate(voxels)}
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan == 3:
                    continue
                # Only half the symmetric neighbours are required.
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                offsets.append((dx, dy, dz))
    for index, coordinate in enumerate(voxels):
        base = tuple(map(int, coordinate))
        for offset in offsets:
            neighbour = (
                base[0] + offset[0],
                base[1] + offset[1],
                base[2] + offset[2],
            )
            other = voxel_lookup.get(neighbour)
            if other is not None:
                union(index, other)
    roots = np.asarray([find(index) for index in range(len(voxels))], dtype=np.int64)
    unique_roots, voxel_labels = np.unique(roots, return_inverse=True)
    particle_labels = voxel_labels[inverse]
    component_particle_counts = np.bincount(
        particle_labels, minlength=len(unique_roots)
    ).astype(np.int64)
    component_voxel_counts = np.bincount(
        voxel_labels, minlength=len(unique_roots)
    ).astype(np.int64)
    return {
        "particle_labels": particle_labels.astype(np.int32),
        "component_particle_counts": component_particle_counts,
        "component_voxel_counts": component_voxel_counts,
        "occupied_voxel_count": int(len(voxels)),
        "cell_size": cell_size,
        "connectivity": connectivity,
        "voxel_particle_counts": voxel_particle_counts,
    }


def reference_water_bodies(
    positions,
    cell_size,
    minimum_body_particles=1024,
    minimum_body_fraction=0.01,
    connectivity=26,
):
    """Return deterministic stable-ID masks for significant reference bodies."""

    positions = _validated_positions(positions)
    components = sparse_particle_components(positions, cell_size, connectivity)
    labels = components["particle_labels"]
    counts = components["component_particle_counts"]
    threshold = max(
        int(minimum_body_particles), int(math.ceil(len(positions) * minimum_body_fraction))
    )
    candidates = np.flatnonzero(counts >= threshold)
    if not len(candidates):
        candidates = np.asarray([int(np.argmax(counts))], dtype=np.int64)
    descriptions = []
    for label in candidates:
        indices = np.flatnonzero(labels == label)
        centroid = positions[indices].mean(axis=0)
        descriptions.append((int(label), indices, centroid))
    descriptions.sort(
        key=lambda item: (
            -len(item[1]),
            float(item[2][0]),
            float(item[2][1]),
            float(item[2][2]),
        )
    )
    bodies = []
    assigned = np.zeros(len(positions), dtype=bool)
    for body_index, (_label, indices, centroid) in enumerate(descriptions):
        assigned[indices] = True
        bodies.append(
            {
                "body_id": f"water_body_{body_index:03d}",
                "particle_indices": indices,
                "particle_count": int(len(indices)),
                "particle_indices_sha256": sha256_indices(indices),
                "reference_centroid": centroid.astype(float).tolist(),
                "reference_bounds_minimum": positions[indices].min(axis=0).astype(float).tolist(),
                "reference_bounds_maximum": positions[indices].max(axis=0).astype(float).tolist(),
            }
        )
    unassigned = np.flatnonzero(~assigned)
    metadata = {
        "particle_count": int(len(positions)),
        "body_count": int(len(bodies)),
        "assigned_particles": int(np.count_nonzero(assigned)),
        "unassigned_particles": int(len(unassigned)),
        "unassigned_particle_indices_sha256": sha256_indices(unassigned),
        "minimum_body_particle_threshold": threshold,
        "occupied_voxel_count": components["occupied_voxel_count"],
    }
    return bodies, metadata


def body_frame_core(
    positions,
    stable_indices,
    cell_size,
    minimum_fragment_particles=128,
    minimum_fragment_fraction=0.002,
    connectivity=26,
):
    """Keep meaningful connected fragments and audit detached stable IDs."""

    positions = _validated_positions(positions)
    stable_indices = np.asarray(stable_indices, dtype=np.int64)
    if stable_indices.ndim != 1 or np.any(stable_indices < 0) or np.any(
        stable_indices >= len(positions)
    ):
        raise ValueError("stable_indices are invalid")
    body_positions = positions[stable_indices]
    components = sparse_particle_components(body_positions, cell_size, connectivity)
    labels = components["particle_labels"]
    counts = components["component_particle_counts"]
    threshold = max(
        int(minimum_fragment_particles),
        int(math.ceil(len(stable_indices) * minimum_fragment_fraction)),
    )
    kept_labels = np.flatnonzero(counts >= threshold)
    largest = int(np.argmax(counts))
    if largest not in kept_labels:
        kept_labels = np.append(kept_labels, largest)
    core_local = np.isin(labels, kept_labels)
    core_indices = stable_indices[core_local]
    detached_indices = stable_indices[~core_local]
    return {
        "core_indices": core_indices,
        "detached_indices": detached_indices,
        "core_particle_count": int(len(core_indices)),
        "detached_particle_count": int(len(detached_indices)),
        "detached_particle_indices_sha256": sha256_indices(detached_indices),
        "kept_fragment_count": int(len(kept_labels)),
        "all_fragment_count": int(len(counts)),
        "minimum_fragment_particle_threshold": threshold,
        "bounds_minimum": positions[core_indices].min(axis=0).astype(float).tolist(),
        "bounds_maximum": positions[core_indices].max(axis=0).astype(float).tolist(),
    }


def aligned_domain_bounds(minimum, maximum, spacing, padding):
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    spacing = float(spacing)
    padding = np.broadcast_to(np.asarray(padding, dtype=np.float64), (3,))
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.isfinite(minimum).all()
        or not np.isfinite(maximum).all()
        or np.any(maximum <= minimum)
        or not np.isfinite(spacing)
        or spacing <= 0.0
        or np.any(padding < 0.0)
    ):
        raise ValueError("Invalid domain bounds, spacing or padding")
    origin = np.floor((minimum - padding) / spacing) * spacing
    top = np.ceil((maximum + padding) / spacing) * spacing
    shape = np.rint((top - origin) / spacing).astype(np.int64) + 1
    return {
        "origin": origin.astype(float).tolist(),
        "maximum": top.astype(float).tolist(),
        "spacing": spacing,
        "shape": shape.astype(int).tolist(),
        "cell_count": int(np.prod(shape, dtype=np.int64)),
        "padding": padding.astype(float).tolist(),
    }
