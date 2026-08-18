"""Report cached mesh properties that matter for thick transmissive materials."""

from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from fixed_topology_video import job_path, load_frame, load_job, load_topology
from liquid_video_cache import read_jsonl


job_dir = Path(sys.argv[1]).resolve()
job = load_job(job_dir)
topology = load_topology(job_path(job_dir, job, "topology"))
triangles = np.asarray(topology["face_vertex_indices"], dtype=np.int64).reshape(-1, 3)
rows = read_jsonl(job_path(job_dir, job, "simulation_manifest"))

for index in (0, 60, 120):
    row = rows[index]
    frame = load_frame(
        job_path(job_dir, job, "cache_dir") / row["cache_file"],
        expected_sha256=row["cache_sha256"],
        expected_vertex_count=int(topology["metadata"]["vertex_count"]),
    )
    points = np.asarray(frame["points"], dtype=np.float64)
    cross = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    signed_volume = np.einsum(
        "ij,ij->i",
        points[triangles[:, 0]],
        np.cross(points[triangles[:, 1]], points[triangles[:, 2]]),
    ).sum() / 6.0
    area = np.linalg.norm(cross, axis=1) * 0.5
    print(
        f"frame={index} points={len(points)} triangles={len(triangles)} "
        f"signed_volume={signed_volume:.9g} degenerate={(area < 1e-12).sum()} "
        f"min_area={area.min():.9g}"
    )

edges = np.sort(
    np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    ),
    axis=1,
)
_, counts = np.unique(edges, axis=0, return_counts=True)
values, frequencies = np.unique(counts, return_counts=True)
print(f"boundary_edges={(counts == 1).sum()}")
print(f"nonmanifold_edges={(counts > 2).sum()}")
print(f"edge_count_hist={dict(zip(values.tolist(), frequencies.tolist()))}")
