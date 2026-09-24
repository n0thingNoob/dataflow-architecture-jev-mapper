import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from analyzer.passthrough import PassthroughAnalyzer
from backends.base import Report
from backends.mock import MockBackend
from export_schemas import MODELS, schema_for
from mapping_ir.models import Mapping
from pipeline import run
from specs.io import read_yaml
from specs.models import Architecture, Program
from validator.checks import validate_inputs, validate_mapping

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "run"
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/matmul_relu_matmul.yaml"))
        self.mapping = PassthroughAnalyzer().propose_mapping(self.arch, self.program)

    def cli(self, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "run_analyzer.py"),
             "--arch", str(ROOT / "examples/wormhole.yaml"),
             "--program", str(ROOT / "examples/matmul_relu_matmul.yaml"),
             "--iterations", "3", "--output", str(self.output), *extra],
            cwd=self.temp.name, text=True, capture_output=True, timeout=30)

    def test_cli_full_loop_and_all_artifacts(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("synthetic", result.stdout)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["best_trial_id"], "trial_0000")
        self.assertEqual(len(summary["trials"]), 3)
        history = [json.loads(line) for line in (self.output / "history.jsonl").read_text().splitlines()]
        for i, trial in enumerate(history):
            directory = self.output / trial["trial_id"]
            self.assertEqual(trial["feedback_trial_ids"], [f"trial_{j:04d}" for j in range(i)])
            self.assertIsNone(trial["measured_cost"])
            for file in ["arch.yaml", "program.yaml", "mapping.yaml", "validation.json", "report.json", "trial.json"]:
                self.assertTrue((directory / file).is_file(), file)
            report = Report.model_validate_json((directory / "report.json").read_text())
            self.assertEqual(report.correctness, "not_checked")
            self.assertEqual(report.objective.source, "synthetic")
            self.assertTrue(all(m.value is None and m.status == "unsupported" for m in report.metrics.values()))
        best = Mapping.model_validate(read_yaml(self.output / "best_mapping.yaml"))
        self.assertEqual(best, self.mapping)

    def test_feedback_reaches_analyzer_and_inputs_are_preserved(self):
        observed = []
        before = deepcopy(self.program.model_dump())

        class Spy(PassthroughAnalyzer):
            def propose_mapping(inner, architecture, program, feedback=None):
                observed.append(deepcopy(feedback))
                return super().propose_mapping(architecture, program, feedback)

        run(self.arch, self.program, Spy(), MockBackend(), 3, self.output)
        self.assertEqual([len(x) for x in observed], [0, 1, 2])
        self.assertEqual(observed[2][0]["report"]["backend"], "mock")
        self.assertEqual(self.program.model_dump(), before)

    def test_invalid_mapping_never_calls_backend(self):
        class Invalid:
            def propose_mapping(inner, *args):
                raw = self.mapping.model_dump()
                raw["regions"][1]["ops"] = ["Matmul0"]
                return raw

        class MustNotRun(MockBackend):
            def run(inner, *args):
                self.fail("Backend was called for an invalid mapping")

        summary = run(self.arch, self.program, Invalid(), MustNotRun(), 2, self.output)
        self.assertIsNone(summary["best_trial_id"])
        self.assertFalse((self.output / "best_mapping.yaml").exists())
        validation = json.loads((self.output / "trial_0000/validation.json").read_text())
        self.assertIn("OP_COVERAGE", [e["code"] for e in validation["errors"]])
        Report.model_validate_json((self.output / "trial_0000/report.json").read_text())

    def test_malformed_mapping_recorded(self):
        class Invalid:
            def propose_mapping(inner, *args):
                return {"regions": "not-a-list"}
        summary = run(self.arch, self.program, Invalid(), MockBackend(), 1, self.output)
        self.assertEqual(summary["status"], "no_valid_result")
        validation = json.loads((self.output / "trial_0000/validation.json").read_text())
        self.assertEqual(validation["errors"][0]["code"], "SCHEMA_ERROR")

    def test_backend_failure_recorded_and_next_trial_runs(self):
        class Flaky(MockBackend):
            calls = 0

            def run(inner, *args):
                inner.calls += 1
                if inner.calls == 1:
                    raise RuntimeError("deliberate test failure")
                return super().run(*args)

        summary = run(self.arch, self.program, PassthroughAnalyzer(), Flaky(), 2, self.output)
        self.assertEqual(summary["trials"][0]["status"], "error")
        self.assertEqual(summary["best_trial_id"], "trial_0001")

    def test_tt_sim_explicitly_unsupported(self):
        result = self.cli("--backend", "tt-sim")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("no simulator was invoked", result.stdout)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertTrue(all(t["status"] == "unsupported" for t in summary["trials"]))
        self.assertFalse((self.output / "best_mapping.yaml").exists())

    def test_existing_output_refused(self):
        self.assertEqual(self.cli().returncode, 0)
        before = (self.output / "history.jsonl").read_bytes()
        self.assertEqual(self.cli().returncode, 2)
        self.assertEqual(before, (self.output / "history.jsonl").read_bytes())

    def test_invalid_iterations_and_bad_yaml_are_clean_errors(self):
        result = self.cli("--iterations", "0")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["errors"][0]["code"], "ITERATIONS")
        malformed = Path(self.temp.name) / "bad.yaml"
        malformed.write_text("ops: [")
        result = self.cli("--program", str(malformed))
        self.assertEqual(result.returncode, 2)
        self.assertIn("errors", json.loads(result.stderr))
        self.assertFalse(self.output.exists())

    def test_semantic_validation(self):
        cases = []
        bad = self.mapping.model_copy(deep=True)
        bad.regions[0].cores = 9
        cases.append((bad, "CORE_CAPACITY"))
        bad = self.mapping.model_copy(deep=True)
        bad.regions.reverse()
        cases.append((bad, "DEPENDENCY_ORDER"))
        bad = self.mapping.model_copy(deep=True)
        bad.program_hash = "wrong"
        cases.append((bad, "INPUT_MISMATCH"))
        raw = self.mapping.model_dump()
        raw["regions"][0]["fusions"] = [{"kind": "matmul_relu", "ops": ["Matmul0", "ReLU1"]}]
        cases.append((Mapping.model_validate(raw), "UNSUPPORTED_FUSION"))
        for mapping, code in cases:
            with self.subTest(code=code):
                self.assertIn(code, [e["code"] for e in validate_mapping(self.arch, self.program, mapping)])

    def test_shape_and_edge_validation(self):
        bad = self.program.model_copy(deep=True)
        bad.edges = []
        self.assertIn("EDGE_MISMATCH", [e["code"] for e in validate_inputs(self.arch, bad)])
        bad = self.program.model_copy(deep=True)
        bad.tensors["w0"].shape = [32, 64]
        self.assertIn("OP_SIGNATURE", [e["code"] for e in validate_inputs(self.arch, bad)])

    def test_program_cycle_rejected_and_unsorted_dag_supported(self):
        bad = self.program.model_copy(deep=True)
        bad.ops[0].inputs[0] = "y"
        self.assertIn("PROGRAM_CYCLE", [e["code"] for e in validate_inputs(self.arch, bad)])
        reordered = self.program.model_copy(deep=True)
        reordered.ops.reverse()
        self.assertEqual(validate_inputs(self.arch, reordered), [])
        proposal = PassthroughAnalyzer().propose_mapping(self.arch, reordered)
        self.assertEqual([r.ops[0] for r in proposal.regions], ["Matmul0", "ReLU1", "Matmul2"])

    def test_best_selection_uses_cost_and_stable_ties(self):
        class Scores(MockBackend):
            values = iter([3, 1, 1, 2])

            def run(inner, *args):
                report = super().run(*args)
                report.objective.value = next(inner.values)
                return report

        summary = run(self.arch, self.program, PassthroughAnalyzer(), Scores(), 4, self.output)
        self.assertEqual(summary["best_trial_id"], "trial_0001")

    def test_mismatched_report_cannot_win(self):
        class Wrong(MockBackend):
            def run(inner, *args):
                report = super().run(*args)
                report.mapping_hash = "another-mapping"
                return report
        summary = run(self.arch, self.program, PassthroughAnalyzer(), Wrong(), 1, self.output)
        self.assertEqual(summary["status"], "no_valid_result")
        self.assertEqual(summary["trials"][0]["status"], "error")

    def test_add_contract_and_exported_schemas(self):
        program = Program.model_validate({
            "id": "add", "tensors": {t: {"shape": [2, 2], "dtype": "float32"} for t in ["a", "b", "c"]},
            "inputs": ["a", "b"], "outputs": ["c"],
            "ops": [{"id": "Add", "op": "add", "inputs": ["a", "b"], "output": "c"}], "edges": []})
        self.assertEqual(validate_inputs(self.arch, program), [])
        for name, model in MODELS.items():
            stored = json.loads((ROOT / "specs/schemas" / f"{name}.schema.json").read_text())
            self.assertEqual(stored, schema_for(model), f"Regenerate {name} schema")


if __name__ == "__main__":
    unittest.main()
