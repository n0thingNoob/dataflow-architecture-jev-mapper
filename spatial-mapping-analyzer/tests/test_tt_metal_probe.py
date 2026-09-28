"""Unit tests for the optional TT-Metal Tensix placement backend."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from candidate_generator import generate_candidates
from specs import Architecture, Program, read_yaml
from tt_metal_probe import (
    TTMetalProbeBackend,
    logical_core_id_to_tt_metal_logical_core,
)

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

    def backend(self, runtime="ttsim"):
        return TTMetalProbeBackend(
            self.tt_metal_home,
            self.probe,
            self.library,
            timeout_seconds=5,
            runtime=runtime,
            tt_metal_revision=(
                "038c8bbd192aa56a8ffaf6f7010f46d0b99eeca0"
                if runtime == "device"
                else None
            ),
        )

    def test_search_contract_only_varies_placement(self):
        backend = self.backend()
        self.assertFalse(backend.search_capabilities.topological_order)
        self.assertFalse(backend.search_capabilities.execution_policy)
        self.assertTrue(backend.search_capabilities.placement)

        first = backend.candidate_execution_signature(self.arch, self.program, self.mapping)
        moved = self.mapping.model_copy(deep=True)
        moved.regions[0].placement = [1]
        second = backend.candidate_execution_signature(self.arch, self.program, moved)
        self.assertNotEqual(first, second)

    def test_logical_core_maps_row_major(self):
        self.assertEqual(logical_core_id_to_tt_metal_logical_core(self.arch, 0), (0, 0))
        self.assertEqual(logical_core_id_to_tt_metal_logical_core(self.arch, 3), (1, 1))
        with self.assertRaises(ValueError):
            logical_core_id_to_tt_metal_logical_core(self.arch, 4)

    def test_backend_invokes_requested_core_and_returns_correctness(self):
        self.mapping.regions[0].placement = [3]

        def fake_run(command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "tt_metal_logical_core": [1, 1],
                        "worker_core": [7, 4],
                        "elements": 1024,
                    }
                )
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
        self.assertEqual(report.extensions["mapping_logical_core_id"], 3)
        self.assertEqual(report.extensions["tt_metal_logical_core"], [1, 1])
        self.assertEqual(report.extensions["worker_core"], [7, 4])
        self.assertEqual(
            report.observed_execution.value["worker_core"], [7, 4]
        )
        command = run.call_args.args[0]
        self.assertEqual(
            command[1:5], ["--core-x", "1", "--core-y", "1"]
        )
        simulator_dir = self.workdir / "tt_metal_simulator"
        self.assertTrue((simulator_dir / "libttsim.so").is_file())
        self.assertTrue((simulator_dir / "soc_descriptor.yaml").is_file())

    def test_device_runtime_exposes_measured_profiler_objective(self):
        result = {
            "passed": True,
            "tt_metal_logical_core": [0, 0],
            "worker_core": [1, 1],
            "elements": 1024,
            "measurement_source": "tt_metal_device_profiler",
            "device_kernel_duration_ns": 321,
        }

        def fake_run(command, **kwargs):
            profiler_dir = Path(kwargs["env"]["TT_METAL_PROFILER_DIR"]) / ".logs"
            profiler_dir.mkdir()
            (profiler_dir / "cpp_device_perf_report.csv").write_text(
                "DEVICE KERNEL DURATION [ns]\n321\n"
            )
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(result))
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_probe.subprocess.run", side_effect=fake_run):
            report = self.backend("device").run(
                self.arch, self.program, self.mapping, self.workdir
            )
        self.assertEqual(report.status, "ok")
        self.assertEqual(report.objective.value, 321)
        self.assertEqual(report.objective.source, "measured")
        self.assertEqual(report.metrics["latency"].value, 321)
        self.assertEqual(
            report.measurement_context.measurement_version,
            "tt-metal-device-kernel-duration-v2",
        )
        self.assertEqual(
            report.extensions["measurement_source"],
            "tt_metal_device_profiler",
        )

    def test_device_runtime_rejects_profiler_csv_mismatch(self):
        result = {
            "passed": True,
            "tt_metal_logical_core": [0, 0],
            "worker_core": [1, 1],
            "elements": 1024,
            "measurement_source": "tt_metal_device_profiler",
            "device_kernel_duration_ns": 321,
        }

        def fake_run(command, **kwargs):
            profiler_dir = Path(kwargs["env"]["TT_METAL_PROFILER_DIR"]) / ".logs"
            profiler_dir.mkdir()
            (profiler_dir / "cpp_device_perf_report.csv").write_text(
                "DEVICE KERNEL DURATION [ns]\n300\n"
            )
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(result))
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_probe.subprocess.run", side_effect=fake_run):
            report = self.backend("device").run(
                self.arch, self.program, self.mapping, self.workdir
            )
        self.assertEqual(report.status, "error")
        self.assertEqual(
            report.extensions["error_code"],
            "PROFILER_CROSS_CHECK_FAILED",
        )

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
                json.dumps(
                    {
                        "passed": True,
                        "tt_metal_logical_core": [9, 9],
                        "worker_core": [7, 4],
                        "elements": 1024,
                    }
                )
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

    @unittest.skipUnless(
        os.environ.get("TT_METAL_HOME") and os.environ.get("SPATIAL_TENSIX_PROBE_BINARY"),
        "Set TT_METAL_HOME and SPATIAL_TENSIX_PROBE_BINARY for real Tensix integration",
    )
    def test_real_tensix_probe_when_environment_is_available(self):
        tt_metal_home = Path(os.environ["TT_METAL_HOME"]).resolve()
        probe = Path(os.environ["SPATIAL_TENSIX_PROBE_BINARY"]).resolve()
        library = Path(
            os.environ.get(
                "TT_SIM_TEST_LIBRARY",
                str(ROOT.parent / "third_party/ttsim/src/_out/release_wh/libttsim.so"),
            )
        ).resolve()
        mapping = generate_candidates(self.arch, self.program, limit=1)[0]
        report = TTMetalProbeBackend(
            tt_metal_home, probe, library, timeout_seconds=120
        ).run(self.arch, self.program, mapping, self.workdir)
        self.assertEqual(report.status, "ok", report.message)
        self.assertEqual(report.correctness, "passed")
        self.assertEqual(report.extensions["tt_metal_logical_core"], [0, 0])
        self.assertIsNotNone(report.extensions["worker_core"])


if __name__ == "__main__":
    unittest.main()
