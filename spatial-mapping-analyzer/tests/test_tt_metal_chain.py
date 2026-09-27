"""Unit tests for the two-core TT-Metal Tensix chain backend."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from candidate_generator import generate_candidates
from specs import Architecture, Program, read_yaml
from tt_metal_chain import TTMetalChainBackend, check_chain_supported

ROOT = Path(__file__).resolve().parents[1]


class TTMetalChainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workdir = self.root / "trial"
        self.workdir.mkdir()

        self.tt_metal_home = self.root / "tt-metal"
        descriptor = self.tt_metal_home / "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
        descriptor.parent.mkdir(parents=True)
        descriptor.write_text("arch: wormhole\n")

        self.probe = self.root / "spatial_tensix_chain_probe"
        self.probe.write_text("fake")
        self.library = self.root / "libttsim.so"
        self.library.write_bytes(b"fake-ttsim")

        self.arch = Architecture.model_validate(
            read_yaml(ROOT / "examples/wormhole_tensix_probe.yaml")
        )
        self.program = Program.model_validate(
            read_yaml(ROOT / "examples/bf16_two_add_chain.yaml")
        )
        self.mapping = generate_candidates(self.arch, self.program, limit=1)[0]

    def backend(self, runtime="ttsim"):
        return TTMetalChainBackend(
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

    def run_with_result(self, result, runtime="ttsim"):
        def fake_run(command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(result))
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_probe.subprocess.run", side_effect=fake_run) as run:
            report = self.backend(runtime).run(
                self.arch, self.program, self.mapping, self.workdir
            )
        return report, run

    def test_search_contract_only_varies_effective_placement(self):
        backend = self.backend()
        self.assertFalse(backend.search_capabilities.topological_order)
        self.assertFalse(backend.search_capabilities.execution_policy)
        self.assertTrue(backend.search_capabilities.placement)

        first = backend.candidate_execution_signature(self.arch, self.program, self.mapping)
        moved = self.mapping.model_copy(deep=True)
        moved.regions[0].placement = [1]
        moved.regions[1].placement = [2]
        second = backend.candidate_execution_signature(self.arch, self.program, moved)
        self.assertNotEqual(first, second)

    def test_chain_contract_extracts_two_distinct_stages(self):
        first, second, logical, coords = check_chain_supported(
            self.arch, self.program, self.mapping
        )
        self.assertEqual([first.id, second.id], ["Add0", "Add1"])
        self.assertEqual(logical, [0, 1])
        self.assertEqual(coords, [(0, 0), (1, 0)])

    def test_backend_invokes_both_mapped_cores(self):
        report, run = self.run_with_result(
            {
                "passed": True,
                "producer_tt_metal_logical_core": [0, 0],
                "consumer_tt_metal_logical_core": [1, 0],
                "producer_worker_core": [1, 1],
                "consumer_worker_core": [2, 1],
                "intermediate_transport": "noc_direct",
                "intermediate_returned_to_host": False,
                "elements": 1024,
            }
        )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.correctness, "passed")
        self.assertIsNone(report.objective)
        self.assertEqual(
            report.extensions["mapping_logical_core_ids"], [0, 1]
        )
        self.assertEqual(
            report.extensions["tt_metal_logical_cores"], [[0, 0], [1, 0]]
        )
        self.assertEqual(
            report.extensions["worker_cores"], [[1, 1], [2, 1]]
        )
        self.assertEqual(
            report.observed_execution.value["worker_cores"],
            [[1, 1], [2, 1]],
        )
        self.assertEqual(report.extensions["intermediate_transport"], "noc_direct")
        self.assertFalse(report.extensions["intermediate_returned_to_host"])

        command = run.call_args.args[0]
        self.assertIn("--producer-x", command)
        self.assertIn("--consumer-x", command)
        self.assertIn("--kernel-root", command)

    def test_device_runtime_requires_explicit_tt_metal_revision(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            TTMetalChainBackend(
                self.tt_metal_home,
                self.probe,
                self.library,
                timeout_seconds=5,
                runtime="device",
            )

    def test_device_runtime_exposes_measured_profiler_objective(self):
        result = {
            "passed": True,
            "producer_tt_metal_logical_core": [0, 0],
            "consumer_tt_metal_logical_core": [1, 0],
            "producer_worker_core": [1, 1],
            "consumer_worker_core": [2, 1],
            "intermediate_transport": "noc_direct",
            "intermediate_returned_to_host": False,
            "elements": 1024,
            "measurement_source": "tt_metal_device_profiler",
            "device_kernel_duration_ns": 1234,
        }

        def fake_run(command, **kwargs):
            self.assertEqual(kwargs["env"]["SPATIAL_MEASURE_DEVICE"], "1")
            self.assertEqual(kwargs["env"]["TT_METAL_DEVICE_PROFILER"], "1")
            self.assertNotIn("TT_METAL_SIMULATOR", kwargs["env"])
            profiler_dir = Path(kwargs["env"]["TT_METAL_PROFILER_DIR"]) / ".logs"
            profiler_dir.mkdir()
            (profiler_dir / "cpp_device_perf_report.csv").write_text(
                "DEVICE KERNEL DURATION [ns]\n1234\n"
            )
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(json.dumps(result))
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_probe.subprocess.run", side_effect=fake_run):
            report = self.backend("device").run(
                self.arch, self.program, self.mapping, self.workdir
            )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.objective.name, "device_kernel_duration")
        self.assertEqual(report.objective.value, 1234)
        self.assertEqual(report.objective.unit, "ns")
        self.assertEqual(report.objective.source, "measured")
        self.assertEqual(report.metrics["latency"].value, 1234)
        self.assertEqual(
            report.measurement_context.measurement_version,
            "tt-metal-device-kernel-duration-v2",
        )
        self.assertEqual(
            report.measurement_context.implementation_revision,
            "038c8bbd192aa56a8ffaf6f7010f46d0b99eeca0",
        )
        self.assertIn(
            "custom_kernel_bundle_sha256",
            report.measurement_context.artifacts,
        )
        self.assertIn(
            "profiler_report_sha256",
            report.measurement_context.artifacts,
        )
        self.assertEqual(
            report.measurement_context.configuration["cross_check"],
            "cpp_device_perf_report.csv",
        )
        self.assertEqual(
            report.extensions["measurement_source"],
            "tt_metal_device_profiler",
        )
        self.assertFalse((self.workdir / "tt_metal_simulator").exists())

    def test_device_runtime_rejects_profiler_csv_mismatch(self):
        result = {
            "passed": True,
            "producer_tt_metal_logical_core": [0, 0],
            "consumer_tt_metal_logical_core": [1, 0],
            "producer_worker_core": [1, 1],
            "consumer_worker_core": [2, 1],
            "intermediate_transport": "noc_direct",
            "intermediate_returned_to_host": False,
            "elements": 1024,
            "measurement_source": "tt_metal_device_profiler",
            "device_kernel_duration_ns": 1234,
        }

        def fake_run(command, **kwargs):
            profiler_dir = Path(kwargs["env"]["TT_METAL_PROFILER_DIR"]) / ".logs"
            profiler_dir.mkdir()
            (profiler_dir / "cpp_device_perf_report.csv").write_text(
                "DEVICE KERNEL DURATION [ns]\n1200\n"
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

    def test_device_runtime_rejects_missing_profiler_measurement(self):
        report, _ = self.run_with_result(
            {
                "passed": True,
                "producer_tt_metal_logical_core": [0, 0],
                "consumer_tt_metal_logical_core": [1, 0],
                "producer_worker_core": [1, 1],
                "consumer_worker_core": [2, 1],
                "intermediate_transport": "noc_direct",
                "intermediate_returned_to_host": False,
                "elements": 1024,
            },
            runtime="device",
        )
        self.assertEqual(report.status, "error")
        self.assertEqual(
            report.extensions["error_code"],
            "PROBE_RESULT_MISMATCH",
        )

    def test_same_core_mapping_is_rejected(self):
        self.mapping.regions[1].placement = [0]
        with self.assertRaises(ValueError):
            check_chain_supported(self.arch, self.program, self.mapping)

    def test_wrong_dag_is_rejected(self):
        bad = self.program.model_copy(deep=True)
        bad.ops[1].inputs = ["a", "c"]
        bad.edges = []
        with self.assertRaises(ValueError):
            check_chain_supported(self.arch, bad, self.mapping)

    def test_probe_result_must_prove_direct_noc_transport(self):
        report, _ = self.run_with_result(
            {
                "passed": True,
                "producer_tt_metal_logical_core": [0, 0],
                "consumer_tt_metal_logical_core": [1, 0],
                "producer_worker_core": [1, 1],
                "consumer_worker_core": [2, 1],
                "intermediate_transport": "host",
                "intermediate_returned_to_host": True,
                "elements": 1024,
            }
        )
        self.assertEqual(report.status, "error")
        self.assertEqual(
            report.extensions["error_code"], "PROBE_RESULT_MISMATCH"
        )


if __name__ == "__main__":
    unittest.main()
