"""Unit tests for the optional TT-Metal Tensix placement backend."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from candidate_generator import generate_candidates
from specs import Architecture, Program, read_yaml
from tt_metal_probe import TTMetalProbeBackend, logical_core_to_coord

ROOT = Path(__file__).resolve().parents[1]


class TTMetalProbeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workdir = self.root / "trial"
        self.workdir.mkdir()

        self.tt_metal_home = self.root / "tt-metal"
        descriptor = (
            self.tt_metal_home
            / "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
        )
        descriptor.parent.mkdir(parents=True)
        descriptor.write_text("arch: wormhole\n")

        self.probe = self.root / "spatial_tensix_probe"
        self.probe.write_text("fake")
        self.library = self.root / "libttsim.so"
        self.library.write_bytes(b"fake-ttsim")

        self.arch = Architecture.model_validate(
            read_yaml(ROOT / "examples/wormhole_tensix_probe.yaml")
        )
        self.program = Program.model_validate(
            read_yaml(ROOT / "examples/bf16_tile_add.yaml")
        )
        self.mapping = generate_candidates(self.arch, self.program, limit=1)[0]

    def backend(self):
        return TTMetalProbeBackend(
            self.tt_metal_home,
            self.probe,
            self.library,
            timeout_seconds=5,
        )

    def test_logical_core_maps_row_major(self):
        self.assertEqual(logical_core_to_coord(self.arch, 0), (0, 0))
        self.assertEqual(logical_core_to_coord(self.arch, 3), (1, 1))
        with self.assertRaises(ValueError):
            logical_core_to_coord(self.arch, 4)

    def test_backend_invokes_requested_core_and_returns_correctness(self):
        self.mapping.regions[0].placement = [3]

        def fake_run(command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps({"passed": True, "core": [1, 1], "elements": 1024})
            )
            self.assertEqual(kwargs["env"]["TT_METAL_SLOW_DISPATCH_MODE"], "1")
            self.assertEqual(
                kwargs["env"]["TT_METAL_HOME"], str(self.tt_metal_home)
            )
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_probe.subprocess.run", side_effect=fake_run) as run:
            report = self.backend().run(
                self.arch, self.program, self.mapping, self.workdir
            )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.correctness, "passed")
        self.assertIsNone(report.objective)
        self.assertEqual(report.extensions["logical_core"], 3)
        self.assertEqual(report.extensions["physical_core"], [1, 1])
        command = run.call_args.args[0]
        self.assertEqual(
            command[1:5], ["--core-x", "1", "--core-y", "1"]
        )
        simulator_dir = self.workdir / "tt_metal_simulator"
        self.assertTrue((simulator_dir / "libttsim.so").is_file())
        self.assertTrue((simulator_dir / "soc_descriptor.yaml").is_file())

    def test_unsupported_program_never_invokes_probe(self):
        bad = self.program.model_copy(deep=True)
        bad.ops[0].op = "relu"
        bad.ops[0].inputs = ["a"]
        bad.edges = []
        mapping = generate_candidates(self.arch, bad, limit=1)[0]
        with patch("tt_metal_probe.subprocess.run") as run:
            report = self.backend().run(self.arch, bad, mapping, self.workdir)
        run.assert_not_called()
        self.assertEqual(report.status, "unsupported")
        self.assertEqual(report.extensions["error_code"], "UNSUPPORTED_PROGRAM")

    def test_result_core_mismatch_is_rejected(self):
        def fake_run(command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps({"passed": True, "core": [9, 9], "elements": 1024})
            )
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_probe.subprocess.run", side_effect=fake_run):
            report = self.backend().run(
                self.arch, self.program, self.mapping, self.workdir
            )
        self.assertEqual(report.status, "error")
        self.assertEqual(
            report.extensions["error_code"], "PROBE_RESULT_MISMATCH"
        )

    def test_missing_tt_metal_environment_is_reported(self):
        missing = self.root / "missing-tt-metal"
        backend = TTMetalProbeBackend(
            missing, self.probe, self.library, timeout_seconds=5
        )
        report = backend.run(
            self.arch, self.program, self.mapping, self.workdir
        )
        self.assertEqual(report.status, "error")
        self.assertEqual(report.extensions["error_code"], "PROBE_ENVIRONMENT")


if __name__ == "__main__":
    unittest.main()
