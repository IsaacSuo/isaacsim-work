"""Identity-stable contact episodes against an independently measured surface."""

from __future__ import annotations

import math

import numpy as np


class MeasuredSurfaceContactTracker:
    """Track cumulative normal deflection without relying on a rigid-contact API."""

    def __init__(
        self,
        points,
        normals,
        aabb_minimum,
        aabb_maximum,
        initial_particle_count,
        maximum_contact_distance=0.09,
        episode_entry_distance=0.11,
        minimum_incoming_normal_speed=0.20,
        minimum_cumulative_outward_change=0.30,
        maximum_episode_gap_steps=3,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 3
            or self.normals.shape != self.points.shape
            or len(self.points) == 0
            or not np.isfinite(self.points).all()
            or not np.isfinite(self.normals).all()
        ):
            raise ValueError("Measured surface points and normals must be finite Nx3 arrays")
        normal_lengths = np.linalg.norm(self.normals, axis=1)
        if np.min(normal_lengths) <= 1.0e-9:
            raise ValueError("Measured surface normals must be non-zero")
        self.normals = self.normals / normal_lengths[:, None]
        self.aabb_minimum = np.asarray(aabb_minimum, dtype=np.float64)
        self.aabb_maximum = np.asarray(aabb_maximum, dtype=np.float64)
        if (
            self.aabb_minimum.shape != (3,)
            or self.aabb_maximum.shape != (3,)
            or not np.all(self.aabb_maximum > self.aabb_minimum)
        ):
            raise ValueError("Measured surface AABB must contain finite increasing bounds")
        if not isinstance(initial_particle_count, int) or initial_particle_count < 0:
            raise ValueError("initial_particle_count must be a non-negative integer")
        values = (
            maximum_contact_distance,
            episode_entry_distance,
            minimum_incoming_normal_speed,
            minimum_cumulative_outward_change,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("Contact thresholds must be finite and positive")
        if episode_entry_distance < maximum_contact_distance:
            raise ValueError("Episode entry distance must cover the contact distance")
        if not isinstance(maximum_episode_gap_steps, int) or maximum_episode_gap_steps < 0:
            raise ValueError("maximum_episode_gap_steps must be a non-negative integer")
        self.initial_particle_count = initial_particle_count
        self.maximum_contact_distance = float(maximum_contact_distance)
        self.episode_entry_distance = float(episode_entry_distance)
        self.minimum_incoming_normal_speed = float(minimum_incoming_normal_speed)
        self.minimum_cumulative_outward_change = float(
            minimum_cumulative_outward_change
        )
        self.maximum_episode_gap_steps = maximum_episode_gap_steps
        self.previous_ids = None
        self.previous_velocities = None
        self.episodes = {}
        self.aabb_particle_ids = set()
        self.near_surface_particle_ids = set()
        self.contact_particle_ids = set()
        self.contact_steps = []
        self.minimum_sample_distance = math.inf
        self.maximum_cumulative_outward_change = 0.0

    def update(self, physics_step, particle_ids, positions, velocities):
        if not isinstance(physics_step, int) or physics_step < 0:
            raise ValueError("physics_step must be a non-negative integer")
        particle_ids = np.asarray(particle_ids, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        if (
            particle_ids.ndim != 1
            or positions.shape != (len(particle_ids), 3)
            or velocities.shape != positions.shape
            or not np.isfinite(positions).all()
            or not np.isfinite(velocities).all()
        ):
            raise ValueError("Particle contact state arrays have invalid shape or values")
        if self.previous_ids is None:
            self.previous_ids = particle_ids.copy()
            self.previous_velocities = velocities.copy()
            return 0
        previous_count = len(self.previous_ids)
        if len(particle_ids) < previous_count or not np.array_equal(
            particle_ids[:previous_count], self.previous_ids
        ):
            raise ValueError("Particle identities are not append-only during contact audit")

        existing = np.arange(len(particle_ids)) < previous_count
        emitted = particle_ids >= self.initial_particle_count
        inside = np.all(positions >= self.aabb_minimum, axis=1) & np.all(
            positions <= self.aabb_maximum, axis=1
        )
        candidate_indices = np.flatnonzero(existing & emitted & inside)
        self.aabb_particle_ids.update(map(int, particle_ids[candidate_indices]))
        seen_episode_ids = set()
        new_contact_ids = []
        step_minimum_distance = math.inf
        step_maximum_incoming = 0.0
        step_maximum_change = 0.0
        if len(candidate_indices):
            candidate_positions = positions[candidate_indices]
            offsets = candidate_positions[:, None, :] - self.points[None, :, :]
            distance_squared = np.sum(offsets * offsets, axis=2)
            nearest_indices = np.argmin(distance_squared, axis=1)
            nearest_distance = np.sqrt(
                distance_squared[np.arange(len(candidate_indices)), nearest_indices]
            )
            near_rows = np.flatnonzero(nearest_distance <= self.episode_entry_distance)
            self.near_surface_particle_ids.update(
                map(int, particle_ids[candidate_indices[near_rows]])
            )
            if len(near_rows):
                self.minimum_sample_distance = min(
                    self.minimum_sample_distance, float(np.min(nearest_distance[near_rows]))
                )
            for row in near_rows:
                state_index = int(candidate_indices[row])
                identifier = int(particle_ids[state_index])
                nearest_index = int(nearest_indices[row])
                nearest_normal = self.normals[nearest_index]
                previous_velocity = self.previous_velocities[state_index]
                incoming_speed = -float(np.dot(previous_velocity, nearest_normal))
                episode = self.episodes.get(identifier)
                if (
                    episode is None
                    or physics_step - episode["last_seen_step"]
                    > self.maximum_episode_gap_steps
                ):
                    if incoming_speed < self.minimum_incoming_normal_speed:
                        continue
                    episode = {
                        "normal": nearest_normal.copy(),
                        "entry_normal_velocity": float(
                            np.dot(previous_velocity, nearest_normal)
                        ),
                        "maximum_incoming_speed": incoming_speed,
                        "last_seen_step": physics_step,
                    }
                    self.episodes[identifier] = episode
                else:
                    episode["last_seen_step"] = physics_step
                    episode["maximum_incoming_speed"] = max(
                        episode["maximum_incoming_speed"], incoming_speed
                    )
                seen_episode_ids.add(identifier)
                current_normal_velocity = float(
                    np.dot(velocities[state_index], episode["normal"])
                )
                cumulative_change = (
                    current_normal_velocity - episode["entry_normal_velocity"]
                )
                self.maximum_cumulative_outward_change = max(
                    self.maximum_cumulative_outward_change, cumulative_change
                )
                if (
                    nearest_distance[row] <= self.maximum_contact_distance
                    and cumulative_change >= self.minimum_cumulative_outward_change
                    and identifier not in self.contact_particle_ids
                ):
                    self.contact_particle_ids.add(identifier)
                    new_contact_ids.append(identifier)
                    step_minimum_distance = min(
                        step_minimum_distance, float(nearest_distance[row])
                    )
                    step_maximum_incoming = max(
                        step_maximum_incoming, episode["maximum_incoming_speed"]
                    )
                    step_maximum_change = max(step_maximum_change, cumulative_change)

        expired = [
            identifier
            for identifier, episode in self.episodes.items()
            if identifier not in seen_episode_ids
            and physics_step - episode["last_seen_step"] > self.maximum_episode_gap_steps
        ]
        for identifier in expired:
            del self.episodes[identifier]
        if new_contact_ids:
            self.contact_steps.append(
                {
                    "physics_step": physics_step,
                    "new_unique_contact_particles": len(new_contact_ids),
                    "minimum_sample_distance_m": step_minimum_distance,
                    "maximum_incoming_normal_speed_m_s": step_maximum_incoming,
                    "maximum_cumulative_outward_change_m_s": step_maximum_change,
                }
            )
        self.previous_ids = particle_ids.copy()
        self.previous_velocities = velocities.copy()
        return len(new_contact_ids)

    def metrics(self):
        near_count = len(self.near_surface_particle_ids)
        contact_count = len(self.contact_particle_ids)
        return {
            "aabb_unique_particles": len(self.aabb_particle_ids),
            "near_surface_unique_particles": near_count,
            "contact_unique_particles": contact_count,
            "contact_fraction_of_near_surface": (
                contact_count / near_count if near_count else 0.0
            ),
            "minimum_sample_distance_m": (
                self.minimum_sample_distance
                if math.isfinite(self.minimum_sample_distance)
                else None
            ),
            "maximum_cumulative_outward_change_m_s": (
                self.maximum_cumulative_outward_change
            ),
            "contact_steps": list(self.contact_steps),
            "thresholds": {
                "maximum_contact_distance_m": self.maximum_contact_distance,
                "episode_entry_distance_m": self.episode_entry_distance,
                "minimum_incoming_normal_speed_m_s": self.minimum_incoming_normal_speed,
                "minimum_cumulative_outward_change_m_s": (
                    self.minimum_cumulative_outward_change
                ),
                "maximum_episode_gap_steps": self.maximum_episode_gap_steps,
            },
        }
