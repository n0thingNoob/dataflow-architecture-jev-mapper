"""Unit tests for dependency-ready scheduling without the real TT-Sim library."""
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import simulator


class FakeWormhole:
    delays = {(1, 1): 1, (2, 1): 3, (3, 1): 1}

    def __init__(self, library):
        self.core = (1, 1)
        self.memory = {}
        self.active = {}
        self.elapsed = {}

        class Clock:
            def __init__(inner, outer):
                inner.outer = outer

            def libttsim_clock(inner, count):
                for core in list(inner.outer.active):
                    if inner.outer.active[core]:
                        inner.outer.elapsed[core] = inner.outer.elapsed.get(core, 0) + count

            def libttsim_exit(inner):
                pass

        self.lib = Clock(self)

    def initialize(self):
        pass

    def write(self, address, data):
        self.memory[(self.core, address)] = bytes(data)

    def write32(self, address, value):
        if address == simulator.RESET_REGISTER:
            if value == simulator.ALL_RESET:
                self.active[self.core] = False
                self.elapsed[self.core] = 0
            elif value == simulator.ALL_RESET & ~simulator.BRISC_RESET:
                self.active[self.core] = True
                self.elapsed[self.core] = 0
            return
        self.write(address, struct.pack("<I", value))

    def read(self, address, size):
        if address == simulator.DONE and self.active.get(self.core, False):
            if self.elapsed.get(self.core, 0) >= self.delays[self.core]:
                return struct.pack("<I", 1)
        if address == simulator.OUTPUT and self.elapsed.get(self.core, 0) >= self.delays.get(self.core, 10**9):
            return bytes(size)
        return self.memory.get((self.core, address), bytes(size))


class SchedulerTests(unittest.TestCase):
    def test_dependency_consumer_starts_before_unrelated_slow_branch_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "libttsim.so"
            library.write_bytes(b"fake-library")

            firmware = b"kernel"
            firmware_hash = hashlib.sha256(firmware).hexdigest()
            for name in ["a.bin", "b.bin", "c.bin"]:
                (root / name).write_bytes(firmware)

            manifest = {
                "mapping_hash": "mapping",
                "execution_policy": "exclusive_cores_dependency_barrier",
                "max_clock_steps": 20,
                "inputs": {"x": [0], "y": [0]},
                "outputs": ["c", "b"],
                "operations": [
                    {
                        "id": "A",
                        "region": "ra",
                        "core": [1, 1],
                        "inputs": ["x"],
                        "output": "a",
                        "elements": 1,
                        "firmware": "a.bin",
                        "firmware_sha256": firmware_hash,
                    },
                    {
                        "id": "B",
                        "region": "rb",
                        "core": [2, 1],
                        "inputs": ["y"],
                        "output": "b",
                        "elements": 1,
                        "firmware": "b.bin",
                        "firmware_sha256": firmware_hash,
                    },
                    {
                        "id": "C",
                        "region": "rc",
                        "core": [3, 1],
                        "inputs": ["a"],
                        "output": "c",
                        "elements": 1,
                        "firmware": "c.bin",
                        "firmware_sha256": firmware_hash,
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            with patch("simulator.Wormhole", FakeWormhole):
                result = simulator.execute(library, manifest_path)

            trace = {entry["op"]: entry for entry in result["trace"]}
            self.assertEqual(result["waves"], [["A", "B"], ["C"]])
            self.assertEqual(trace["A"]["completion_observed_api_step"], 1)
            self.assertEqual(trace["C"]["start_api_step"], 1)
            self.assertEqual(trace["C"]["completion_observed_api_step"], 2)
            self.assertEqual(trace["B"]["completion_observed_api_step"], 3)
            self.assertLess(
                trace["C"]["start_api_step"],
                trace["B"]["completion_observed_api_step"],
            )


if __name__ == "__main__":
    unittest.main()
