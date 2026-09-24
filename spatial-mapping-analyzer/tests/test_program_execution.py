"""Numerical and dependency E2E tests against the real upstream simulator."""
import json
import subprocess
import unittest
from unittest.mock import patch

from analyzer.passthrough import PassthroughAnalyzer
from backends.program_workload import check_supported, input_values, reference
from backends.tt_sim_program import TTSimProgramBackend
from pipeline import run
from specs.io import read_yaml
from specs.models import Architecture, Program
from tests.support import LIBRARY, ROOT, AnalyzerTestCase
from validator.checks import validate_inputs, validate_mapping

DEPENDENCY = "exclusive_cores_dependency_barrier"


class ProgramExecutionTests(AnalyzerTestCase):
    def setUp(self):
        super().setUp()
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole_brisc.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/scalar_diamond.yaml"))
        self.inputs = {"a": [-3], "b": [1], "c": [5]}
        self.mapping = PassthroughAnalyzer(DEPENDENCY).propose_mapping(self.arch, self.program)

    def test_scalar_contract_and_dependency_checks(self):
        self.assertEqual(validate_inputs(self.arch, self.program), [])
        self.assertEqual(validate_mapping(self.arch, self.program, self.mapping), [])
        self.assertEqual(reference(self.program, self.inputs), {"p": [-2], "left": [3], "right": [0], "y": [3]})
        self.mapping.regions.reverse()
        self.assertIn("DEPENDENCY_ORDER", {e["code"] for e in validate_mapping(self.arch, self.program, self.mapping)})
        self.program.tensors["a"].shape = [1]
        self.assertIn("OP_SIGNATURE", {e["code"] for e in validate_inputs(self.arch, self.program)})

    def test_input_fixture_and_unsupported_workload_rejected(self):
        for fixture in [{"a": [1]}, {**self.inputs, "a": [True]}, {**self.inputs, "a": [2**31]}, {**self.inputs, "a": []}]:
            with self.subTest(fixture=fixture), self.assertRaises(ValueError):
                input_values(self.program, fixture)
        for tensor in self.program.tensors.values():
            tensor.dtype = "float32"
        mapping = PassthroughAnalyzer().propose_mapping(self.arch, self.program)
        with patch("backends.tt_sim.subprocess.run") as execute:
            report = TTSimProgramBackend(LIBRARY).run(self.arch, self.program, mapping, self.workdir)
        execute.assert_not_called()
        self.assertEqual(report.status, "unsupported")
        self.assertIsNone(report.objective)

    def test_resource_limits_and_timeout(self):
        self.mapping.regions[0].cores = 2
        with self.assertRaises(ValueError):
            check_supported(self.arch, self.program, self.mapping)
        self.mapping.regions[0].cores = 1
        self.arch.local_memory_bytes_per_core = 4096
        mapping = PassthroughAnalyzer().propose_mapping(self.arch, self.program)
        with self.assertRaises(ValueError):
            check_supported(self.arch, self.program, mapping)
        self.arch.local_memory_bytes_per_core = 32768
        fake = self.workdir / "fake.so"
        fake.write_bytes(b"subprocess is mocked")
        with patch("backends.tt_sim.subprocess.run", side_effect=subprocess.TimeoutExpired("runner", 1)):
            report = TTSimProgramBackend(fake, inputs=self.inputs).run(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(report.extensions["error_code"], "SIMULATOR_TIMEOUT")
        self.assertFalse(report.extensions["program_dag_executed"])
        self.assertIsNone(report.objective)

    @unittest.skipUnless(LIBRARY.is_file(), "Build Wormhole TT-Sim for real program integration")
    def test_diamond_parallel_and_serial_execution(self):
        executions = []
        for policy in [DEPENDENCY, "exclusive_cores_tensor_barrier"]:
            directory = self.workdir / policy
            summary = run(self.arch, self.program, PassthroughAnalyzer(policy),
                          TTSimProgramBackend(LIBRARY, inputs=self.inputs), 1, directory)
            self.assertEqual(summary["status"], "ok")
            report = json.loads((directory / "trial_0000/report.json").read_text())
            self.assertEqual(report["correctness"], "passed")
            execution = report["extensions"]["execution"]
            self.assertEqual(execution["outputs"], {"y": [3]})
            executions.append(execution)
        parallel, serial = executions
        self.assertEqual(parallel["waves"], [["Producer"], ["Left", "Right"], ["Join"]])
        trace = {t["op"]: t for t in parallel["trace"]}
        left, right = trace["Left"], trace["Right"]
        self.assertNotEqual(left["core"], right["core"])
        self.assertLess(max(left["start_api_step"], right["start_api_step"]),
                        min(left["completion_observed_api_step"], right["completion_observed_api_step"]))
        self.assertGreaterEqual(trace["Join"]["start_api_step"],
                                max(left["completion_observed_api_step"], right["completion_observed_api_step"]))
        self.assertTrue(all(len(wave) == 1 for wave in serial["waves"]))
        self.assertLess(parallel["api_steps"], serial["api_steps"])  # diagnostic only, not hardware speedup

    @unittest.skipUnless(LIBRARY.is_file(), "Build Wormhole TT-Sim for real program integration")
    def test_changed_inputs_and_int32_overflow(self):
        for inputs, expected in [({"a": [3], "b": [1], "c": [2]}, 10),
                                 ({"a": [2**31-1], "b": [1], "c": [-1]}, 2**31-1)]:
            with self.subTest(inputs=inputs):
                report = TTSimProgramBackend(LIBRARY, inputs=inputs).run(self.arch, self.program, self.mapping, self.workdir)
                self.assertEqual(report.correctness, "passed", report.message)
                self.assertEqual(report.extensions["execution"]["outputs"], {"y": [expected]})

    @unittest.skipUnless(LIBRARY.is_file(), "Build Wormhole TT-Sim for real program integration")
    def test_rectangular_gemm_known_result(self):
        program = Program.model_validate(read_yaml(ROOT / "examples/gemm_relu_gemm.yaml"))
        shapes = {"a": [2, 3], "b": [3, 2], "hidden": [2, 2], "activated": [2, 2], "c": [2, 1], "y": [2, 1]}
        for name, shape in shapes.items():
            program.tensors[name].shape = shape
        inputs = {"a": [-1, -2, 0, 3, -2, 1], "b": [1, 2, 3, 4, 5, 6], "c": [-1, 2]}
        mapping = PassthroughAnalyzer().propose_mapping(self.arch, program)
        report = TTSimProgramBackend(LIBRARY, inputs=inputs).run(self.arch, program, mapping, self.workdir)
        self.assertEqual(report.correctness, "passed", report.message)
        self.assertEqual(report.extensions["execution"]["tensors"],
                         {"hidden": [-7, -10, 2, 4], "activated": [0, 0, 2, 4], "y": [0, 6]})

    @unittest.skipUnless(LIBRARY.is_file(), "Build Wormhole TT-Sim for real program integration")
    def test_tensor_cli_full_loop(self):
        process = self.cli("--arch", str(ROOT / "examples/wormhole_brisc.yaml"),
                           "--program", str(ROOT / "examples/gemm_relu_gemm.yaml"), "--backend", "tt-sim-program",
                           "--tt-sim-library", str(LIBRARY), "--iterations", "2", "--seed", "7")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        history = [json.loads(line) for line in (self.output / "history.jsonl").read_text().splitlines()]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["feedback_trial_ids"], ["trial_0000"])
        for trial in history:
            report = trial["report"]
            self.assertEqual(report["correctness"], "passed")
            self.assertTrue(report["extensions"]["program_dag_executed"])
            self.assertEqual(len(report["extensions"]["execution"]["outputs"]["y"]), 1024)
            self.assertEqual(report["objective"]["source"], "synthetic")
            self.assertIsNone(trial["measured_cost"])
            self.assertIsNone(report["metrics"]["total_cycles"]["value"])
            directory = self.output / trial["trial_id"]
            for name in ["inputs.json", "reference.json", "correctness.json", "program_execution.json", "op_0000.bin", "invocation.json"]:
                self.assertTrue((directory / name).is_file())
        self.assertTrue((self.output / "best_mapping.yaml").is_file())

    @unittest.skipUnless(LIBRARY.is_file(), "Build Wormhole TT-Sim for real program integration")
    def test_wrong_device_output_cannot_win(self):
        from backends import program_workload
        compile_real = program_workload.compile_kernel

        def wrong_kernel(op, shapes, count):
            # Deliberately lower scalar add to relu; reference remains independent.
            return compile_real("relu" if op == "add" else op, shapes, count)

        with patch("backends.program_workload.compile_kernel", side_effect=wrong_kernel):
            report = TTSimProgramBackend(LIBRARY, inputs=self.inputs).run(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(report.correctness, "failed")
        self.assertEqual(report.status, "error")
        self.assertIsNone(report.objective)
