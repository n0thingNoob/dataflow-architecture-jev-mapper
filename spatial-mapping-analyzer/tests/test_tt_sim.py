"""Real integration tests run when the pinned Wormhole library has been built."""
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyzer.passthrough import PassthroughAnalyzer
from backends.tt_sim import DEFAULT_LIBRARY, TTSimBackend
from backends.tt_sim_dummy import generate_dummy
from specs.io import read_yaml
from specs.models import Architecture, Program

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("TT_SIM_TEST_LIBRARY", str(DEFAULT_LIBRARY))).resolve()


class DummyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/matmul_relu_matmul.yaml"))
        self.mapping = PassthroughAnalyzer().propose_mapping(self.arch, self.program)

    def test_dummy_files_contain_inputs_not_the_computed_output(self):
        manifest = generate_dummy(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(struct.unpack("<4I", (self.workdir / "dummy_data.bin").read_bytes()), (3, 3, 0xFFFFFFFF, 0))
        self.assertEqual(manifest["expected_result"], 6)
        self.assertEqual(len((self.workdir / "dummy.bin").read_bytes()), 32)
        self.assertFalse(manifest["program_dag_executed"])

    def test_timeout_is_recorded_without_a_synthetic_success(self):
        library = self.workdir / "fake.so"
        library.write_bytes(b"not used: subprocess is mocked for timeout test")
        with patch("backends.tt_sim.subprocess.run", side_effect=subprocess.TimeoutExpired("runner", 1)):
            report = TTSimBackend(library, 1).run(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(report.status, "error")
        self.assertEqual(report.extensions["error_code"], "SIMULATOR_TIMEOUT")
        self.assertIsNone(report.objective)
        self.assertTrue((self.workdir / "invocation.json").is_file())
        self.assertTrue((self.workdir / "stderr.log").is_file())

    def test_nonzero_exit_and_missing_result_are_errors(self):
        library = self.workdir / "fake.so"
        library.write_bytes(b"boundary test")
        for returncode, expected in [(7, "SIMULATOR_FAILED"), (0, "INVALID_SIMULATOR_RESULT")]:
            with self.subTest(returncode=returncode):
                with patch("backends.tt_sim.subprocess.run", return_value=subprocess.CompletedProcess([], returncode)):
                    report = TTSimBackend(library).run(self.arch, self.program, self.mapping, self.workdir)
                self.assertEqual(report.extensions["error_code"], expected)
                self.assertIsNone(report.objective)

    def test_invalid_timeout_rejected(self):
        for value in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                TTSimBackend(timeout_seconds=value)

    @unittest.skipUnless(LIBRARY.is_file(), "Build the pinned Wormhole libttsim.so to run real integration")
    def test_real_library_executes_changed_dummy_inputs(self):
        # One grouped region with two allocated cores changes the dummy sum to 3.
        self.mapping.regions[0].ops = [op.id for op in self.program.ops]
        self.mapping.regions[0].cores = 2
        self.mapping.regions = self.mapping.regions[:1]
        report = TTSimBackend(LIBRARY).run(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(report.status, "ok", report.message)
        execution = report.extensions["execution"]
        self.assertEqual(execution["inputs"], {"lhs": 1, "rhs": 2})
        self.assertEqual(execution["actual_result"], 3)
        self.assertEqual(execution["dummy_correctness"], "passed")

    @unittest.skipUnless(LIBRARY.is_file(), "Build the pinned Wormhole libttsim.so to run real integration")
    def test_real_cli_three_trials_and_feedback(self):
        output = self.workdir / "run"
        process = subprocess.run(
            [sys.executable, str(ROOT / "run_analyzer.py"), "--arch", str(ROOT / "examples/wormhole.yaml"),
             "--program", str(ROOT / "examples/matmul_relu_matmul.yaml"), "--backend", "tt-sim",
             "--tt-sim-library", str(LIBRARY), "--iterations", "3", "--output", str(output)],
            text=True, capture_output=True, timeout=60, cwd=self.workdir)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["best_trial_id"], "trial_0000")
        library_hash = hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
        for i in range(3):
            trial_dir = output / f"trial_{i:04d}"
            trial = json.loads((trial_dir / "trial.json").read_text())
            report = trial["report"]
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["correctness"], "not_checked")
            self.assertEqual(report["objective"]["source"], "synthetic")
            self.assertIsNone(trial["measured_cost"])
            self.assertIsNone(report["metrics"]["total_cycles"]["value"])
            self.assertEqual(len(trial["feedback_trial_ids"]), i)
            execution = report["extensions"]["execution"]
            self.assertEqual(execution["actual_result"], 6)
            self.assertEqual(execution["completion_flag"], 1)
            self.assertEqual(execution["library_sha256"], library_hash)
            self.assertGreater(execution["clock_steps_to_completion"], 0)
            for name in ["dummy.json", "dummy.bin", "dummy_data.bin", "invocation.json", "stdout.log", "stderr.log", "runner_result.json"]:
                self.assertTrue((trial_dir / name).is_file(), name)
        self.assertTrue((output / "best_mapping.yaml").is_file())

    @unittest.skipUnless(LIBRARY.is_file(), "Build the pinned Wormhole libttsim.so to run real integration")
    def test_real_simulator_error_is_isolated_from_parent(self):
        manifest = generate_dummy(self.arch, self.program, self.mapping, self.workdir)
        # A real illegal instruction makes upstream libttsim terminate the child.
        firmware = bytes(32)
        (self.workdir / "dummy.bin").write_bytes(firmware)
        manifest["firmware_sha256"] = hashlib.sha256(firmware).hexdigest()
        (self.workdir / "dummy.json").write_text(json.dumps(manifest))
        process = subprocess.run(
            [sys.executable, str(ROOT / "backends/tt_sim_runner.py"), "--library", str(LIBRARY),
             "--manifest", str(self.workdir / "dummy.json"), "--result", str(self.workdir / "runner_result.json")],
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(process.returncode, 0)
        self.assertTrue(process.stdout or process.stderr)
        self.assertFalse((self.workdir / "runner_result.json").exists())


if __name__ == "__main__":
    unittest.main()
