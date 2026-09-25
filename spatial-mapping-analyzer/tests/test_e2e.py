"""Numerical and dependency E2E tests against the real upstream simulator."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyzer import PassthroughAnalyzer
from pipeline import run
from report import Objective, Report
from specs import Architecture, Program, fingerprint, read_yaml
from tt_sim import DEFAULT_LIBRARY, TTSimBackend
from validator import validate_inputs, validate_mapping
from workload import check_supported, input_values, reference

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("TT_SIM_TEST_LIBRARY", str(DEFAULT_LIBRARY))).resolve()


class E2ETests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workdir = Path(temp.name)
        self.output = self.workdir / "run"
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/scalar_diamond.yaml"))
        self.inputs = {"a": [-3], "b": [1], "c": [5]}
        self.mapping = PassthroughAnalyzer(parallel=True).propose_mapping(self.arch, self.program)

    def cli(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / "run_analyzer.py"),
                               "--program", str(ROOT / "examples/scalar_diamond.yaml"),
                               "--output", str(self.output), *extra],
                              cwd=self.workdir, capture_output=True, text=True, timeout=60)

    def test_invalid_mapping_never_executes(self):
        class Never:
            name = "test"

            def run(inner, *args):
                self.fail("Invalid mapping reached simulator")

        raw = self.mapping.model_dump()
        raw["regions"][1]["ops"] = ["Producer"]
        for index, proposal in enumerate([raw, {"regions": "invalid"}]):
            class Invalid:
                def propose_mapping(inner, *args):
                    return proposal
            directory = self.workdir / str(index)
            summary = run(self.arch, self.program, Invalid(), Never(), 1, directory)
            self.assertEqual(summary["status"], "no_valid_result")
            self.assertEqual(summary["trials"][0]["status"], "skipped")
            self.assertFalse((directory / "best_mapping.yaml").exists())

    def test_feedback_failure_recovery_and_stable_ranking(self):
        observed = []
        before = self.program.model_dump()

        class Spy(PassthroughAnalyzer):
            def propose_mapping(inner, arch, program, feedback=None):
                observed.append(len(feedback))
                mapping = super().propose_mapping(arch, program, feedback)
                program.id = "only-the-copy-changes"
                return mapping

        class Scores:
            name = "test"
            values = iter([None, 3, 1, 1])

            def run(inner, arch, program, mapping, directory):
                value = next(inner.values)
                if value is None:
                    raise RuntimeError("Injected failure")
                return Report(backend=inner.name, backend_version="test", status="ok",
                              mapping_hash=fingerprint(mapping),
                              objective=Objective(name="test", value=value, unit="test", source="synthetic"))

        summary = run(self.arch, self.program, Spy(), Scores(), 4, self.output)
        self.assertEqual(observed, [0, 1, 2, 3])
        self.assertEqual(self.program.model_dump(), before)
        self.assertEqual(summary["trials"][0]["status"], "error")
        self.assertEqual(summary["best_trial_id"], "trial_0002")

    def test_schemas_and_cli_errors(self):
        destination = self.workdir / "schemas"
        result = subprocess.run([sys.executable, str(ROOT / "export_schemas.py"), "--output", str(destination)],
                                capture_output=True, text=True, timeout=30, cwd=self.workdir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.stem for p in destination.iterdir()}, {n + ".schema" for n in ["arch", "program", "mapping", "report"]})
        result = self.cli("--iterations", "0")
        self.assertEqual(json.loads(result.stderr)["errors"][0]["code"], "ITERATIONS")
        result = self.cli("--tt-sim-library", str(self.workdir / "missing.so"))
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.output / "best_mapping.yaml").exists())
        history = (self.output / "history.jsonl").read_bytes()
        self.assertEqual(self.cli().returncode, 2)  # Existing output is not overwritten.
        self.assertEqual((self.output / "history.jsonl").read_bytes(), history)

    def test_scalar_contract_and_dependency_checks(self):
        self.assertEqual(validate_inputs(self.arch, self.program), [])
        self.assertEqual(validate_mapping(self.arch, self.program, self.mapping), [])
        self.assertEqual(reference(self.program, self.inputs), {"p": [-2], "left": [3], "right": [0], "y": [3]})
        self.mapping.regions.reverse()
        self.assertIn("DEPENDENCY_ORDER", {e["code"] for e in validate_mapping(self.arch, self.program, self.mapping)})
        self.program.tensors["a"].shape = [1]
        self.assertIn("OP_SIGNATURE", {e["code"] for e in validate_inputs(self.arch, self.program)})
        self.program.ops[0].inputs[0] = "y"
        self.assertIn("PROGRAM_CYCLE", {e["code"] for e in validate_inputs(self.arch, self.program)})

    def test_input_fixture_and_unsupported_workload_rejected(self):
        for fixture in [{"a": [1]}, {**self.inputs, "a": [True]}, {**self.inputs, "a": [2**31]}, {**self.inputs, "a": []}]:
            with self.subTest(fixture=fixture), self.assertRaises(ValueError):
                input_values(self.program, fixture)
        for tensor in self.program.tensors.values():
            tensor.dtype = "float32"
        mapping = PassthroughAnalyzer().propose_mapping(self.arch, self.program)
        with patch("tt_sim.subprocess.run") as execute:
            report = TTSimBackend(LIBRARY).run(self.arch, self.program, mapping, self.workdir)
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
        with patch("tt_sim.subprocess.run", side_effect=subprocess.TimeoutExpired("runner", 1)):
            report = TTSimBackend(fake, inputs=self.inputs).run(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(report.extensions["error_code"], "SIMULATOR_TIMEOUT")
        self.assertFalse(report.extensions["program_dag_executed"])
        self.assertIsNone(report.objective)

    @unittest.skipUnless(LIBRARY.is_file(), "Build Wormhole TT-Sim for real program integration")
    def test_diamond_parallel_and_serial_execution(self):
        executions = []
        for policy in [True, False]:
            directory = self.workdir / str(policy)
            summary = run(self.arch, self.program, PassthroughAnalyzer(policy),
                          TTSimBackend(LIBRARY, inputs=self.inputs), 1, directory)
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
                report = TTSimBackend(LIBRARY, inputs=inputs).run(self.arch, self.program, self.mapping, self.workdir)
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
        report = TTSimBackend(LIBRARY, inputs=inputs).run(self.arch, program, mapping, self.workdir)
        self.assertEqual(report.correctness, "passed", report.message)
        self.assertEqual(report.extensions["execution"]["tensors"],
                         {"hidden": [-7, -10, 2, 4], "activated": [0, 0, 2, 4], "y": [0, 6]})

    @unittest.skipUnless(LIBRARY.is_file(), "Build Wormhole TT-Sim for real program integration")
    def test_tensor_cli_full_loop(self):
        process = self.cli("--arch", str(ROOT / "examples/wormhole.yaml"),
                           "--program", str(ROOT / "examples/gemm_relu_gemm.yaml"), "--tt-sim-library", str(LIBRARY), "--iterations", "2", "--seed", "7")
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
        import workload
        compile_real = workload.compile_kernel

        def wrong_kernel(op, shapes, count):
            # Deliberately lower scalar add to relu; reference remains independent.
            return compile_real("relu" if op == "add" else op, shapes, count)

        with patch("workload.compile_kernel", side_effect=wrong_kernel):
            report = TTSimBackend(LIBRARY, inputs=self.inputs).run(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(report.correctness, "failed")
        self.assertEqual(report.status, "error")
        self.assertIsNone(report.objective)
        with patch("workload.compile_kernel", return_value=bytes(32)):
            report = TTSimBackend(LIBRARY, inputs=self.inputs).run(self.arch, self.program, self.mapping, self.workdir)
        self.assertEqual(report.extensions["error_code"], "SIMULATOR_FAILED")
        self.assertIsNone(report.objective)  # A fatal simulator exit stays inside the child.
