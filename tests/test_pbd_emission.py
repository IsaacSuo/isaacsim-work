"""Emission must not overwrite already simulated particle state."""

import unittest
from types import SimpleNamespace

import numpy as np

from coupled_scene.pbd_pour_event import PbdPourEvent, _emission_chunks


class Attribute:
    def __init__(self, value):
        self.value = value
        self.writes = 0
        self.reads = 0

    def Get(self):
        self.reads += 1
        return self.value

    def Set(self, value):
        self.value = np.array(value, copy=True)
        self.writes += 1


class Instancer:
    def __init__(self, positions, velocities):
        self.positions = Attribute(np.array(positions, dtype=np.float32))
        self.velocities = Attribute(np.array(velocities, dtype=np.float32))
        self.proto_indices = Attribute(np.zeros(len(positions), dtype=np.int32))

    def GetPositionsAttr(self):
        return self.positions

    def GetVelocitiesAttr(self):
        return self.velocities

    def GetProtoIndicesAttr(self):
        return self.proto_indices


class EmissionTests(unittest.TestCase):
    def event(self):
        event = PbdPourEvent.__new__(PbdPourEvent)
        event.particle_set_strategy = "preauthored_density_sets"
        event.active_instancers = []
        event.active_id_chunks = []
        event.enabled_batches = 0
        event.emission_statistics = dict(
            activation_count=0, old_particle_rows_read=0,
            particle_rows_written=0, authoring_seconds=0.0,
            activation_update_seconds=0.0,
            maximum_old_particle_rows_per_activation=0,
            attribute_read_seconds=0.0, array_build_seconds=0.0,
            attribute_write_seconds=0.0,
        )
        event.batch_records = [
            dict(birth_frame=121, birth_substep=i, ids=np.array([i]),
                 enabled=Attribute(False),
                 instancer=Instancer([[i, 0, 0]], [[0, -0.96, 0]]))
            for i in range(3)
        ]
        return event

    def test_activation_preserves_native_motion_and_ids(self):
        event = self.event()
        updates = []
        app = SimpleNamespace(update=lambda: updates.append(True))
        event.enable_due(120, 11, app)
        self.assertFalse(updates)
        event.enable_due(121, 0, app)
        # Model the state PhysX produced before the second birth. Never
        # restore the first batch's initial conditions while emitting more.
        old = event.active_instancers[0]
        old.positions.value[:] = [2, 3, 4]
        old.velocities.value[:] = [1, -2, 3]
        event.enable_due(121, 1, app)
        event.enable_due(121, 1, app)  # duplicate callback must not emit twice
        positions, velocities, ids = event.state_arrays()
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_array_equal(positions[0], [2, 3, 4])
        np.testing.assert_array_equal(velocities[0], [1, -2, 3])
        self.assertEqual(old.positions.writes + old.velocities.writes, 0)
        self.assertFalse(event.batch_records[2]["enabled"].value)
        self.assertEqual(len(updates), 2)
        self.assertEqual(event.emission_statistics["activation_count"], 2)
        self.assertEqual(event.emission_statistics["old_particle_rows_read"], 0)
        self.assertEqual(event.emission_statistics["particle_rows_written"], 0)

    def test_missed_birth_is_an_error(self):
        event = self.event()
        with self.assertRaisesRegex(RuntimeError, "Missed PBD emitter activation"):
            event.enable_due(121, 1, SimpleNamespace(update=lambda: None))

    def test_chunk_plan_preserves_births_and_bounds_size(self):
        records = [dict(positions=np.zeros((n, 3)), birth_substep=i)
                   for i, n in enumerate([2, 3, 1, 4, 2])]
        chunks = _emission_chunks(records, 5)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        flattened = [record for chunk in chunks for record in chunk]
        self.assertTrue(all(a is b for a, b in zip(records, flattened)))
        for capacity in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                _emission_chunks(records, capacity)
        with self.assertRaisesRegex(ValueError, "exceeding chunk capacity"):
            _emission_chunks(records, 3)

    def test_chunk_append_never_reads_or_rewrites_sealed_chunks(self):
        event = self.event()
        event.particle_set_strategy = "chunked_density_sets"
        event.particle_mass_attr = None
        array_type = SimpleNamespace(FromNumpy=lambda value: np.array(value, copy=True))
        event._Vt = SimpleNamespace(Vec3fArray=array_type, IntArray=array_type)
        for i, record in enumerate(event.batch_records):
            record["positions"] = np.array([[i, 0, 0]], dtype=np.float32)
            record["velocities"] = np.array([[0, -0.96, 0]], dtype=np.float32)
        for records in _emission_chunks(event.batch_records, 2):
            instancer = Instancer([], [])
            enabled = Attribute(False)
            for i, record in enumerate(records):
                record.update(instancer=instancer, enabled=enabled, starts_chunk=i == 0)
        updates = []
        app = SimpleNamespace(update=lambda: updates.append(True))
        event.enable_due(121, 0, app)
        old = event.active_instancers[0]
        old.positions.value[:] = [2, 3, 4]
        old.velocities.value[:] = [1, -2, 3]
        event.enable_due(121, 1, app)
        np.testing.assert_array_equal(old.positions.value[0], [2, 3, 4])
        np.testing.assert_array_equal(old.velocities.value[0], [1, -2, 3])
        accesses = [(attr.reads, attr.writes) for attr in
                    (old.positions, old.velocities, old.proto_indices)]
        event.enable_due(121, 2, app)
        event.enable_due(121, 2, app)
        self.assertEqual(accesses, [(attr.reads, attr.writes) for attr in
                                   (old.positions, old.velocities, old.proto_indices)])
        self.assertEqual(len(event.active_instancers), 2)
        self.assertEqual(len(updates), 3)
        positions, velocities, ids = event.state_arrays()
        np.testing.assert_array_equal(ids, [0, 1, 2])
        np.testing.assert_array_equal(positions[0], [2, 3, 4])
        np.testing.assert_array_equal(velocities[0], [1, -2, 3])
        self.assertEqual(event.emission_statistics["old_particle_rows_read"], 1)
        self.assertEqual(event.emission_statistics["particle_rows_written"], 4)
        self.assertEqual(event.emission_statistics["maximum_old_particle_rows_per_activation"], 1)


if __name__ == "__main__":
    unittest.main()
