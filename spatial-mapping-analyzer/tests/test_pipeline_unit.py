"""Focused unit tests for pipeline error isolation and ranking."""
import tempfile
import unittest
from pathlib import Path

from analyzer import PassthroughAnalyzer
from pipeline import _execute, _propose, run
from report import Objective, Report
from specs import Architecture, Program, fingerprint, read_yaml

ROOT = Path(__file__).resolve().parents[1]


class PipelineUnitTests(unittest.TestCase):
    def setUp(self):
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/scalar_diamond.yaml"))
        self.mapping = PassthroughAnalyzer().propose_mapping(self.arch, self.program)

    def test_propose_isolates_analyzer_exception(self):
        class Broken:
            def propose_mapping(self, *args):
                raise RuntimeError("boom")

        raw, mapping, errors = _propose(self.arch, self.program, Broken(), [])
        self.assertIsNone(raw)
        self.assertIsNone(mapping)
        self.assertEqual(errors[0]["code"], "ANALYZER_ERROR")

    def test_propose_preserves_invalid_raw_mapping_for_review(self):
        class Invalid:
            def propose_mapping(self, *args):
                return {"schema_version": "0.1", "regions": "bad"}

        raw, mapping, errors = _propose(self.arch, self.program, Invalid(), [])
        self.assertEqual(raw["regions"], "bad")
        self.assertIsNone(mapping)
        self.assertEqual(errors[0]["code"], "SCHEMA_ERROR")

    def test_execute_skips_invalid_mapping_without_calling_backend(self):
        outer = self

        class Backend:
            name = "never"

            def run(self, *args):
                outer.fail("backend should not execute")

        report = _execute(
            self.arch,
            self.program,
            self.mapping,
            Backend(),
            Path("."),
            [{"code": "bad"}],
            None,
        )
        self.assertEqual(report.status, "skipped")

    def test_execute_rejects_backend_identity_mismatch(self):
        class Backend:
            name = "expected"

            def run(inner, arch, program, mapping, directory):
                return Report(
                    backend="wrong",
                    backend_version="1",
                    status="ok",
                    mapping_hash=fingerprint(mapping),
                )

        report = _execute(
            self.arch, self.program, self.mapping, Backend(), Path("."), [], None
        )
        self.assertEqual(report.status, "error")
        self.assertIn("identity", report.message)

    def test_execute_rejects_objective_definition_change(self):
        class Backend:
            name = "test"

            def run(inner, arch, program, mapping, directory):
                return Report(
                    backend="test",
                    backend_version="1",
                    status="ok",
                    mapping_hash=fingerprint(mapping),
                    objective=Objective(
                        name="other", value=1, unit="cycles", source="synthetic"
                    ),
                )

        expected_key = ("test", "1", "latency", "cycles", "synthetic")
        report = _execute(
            self.arch,
            self.program,
            self.mapping,
            Backend(),
            Path("."),
            [],
            expected_key,
        )
        self.assertEqual(report.status, "error")
        self.assertIn("different objective", report.message)

    def test_success_without_objective_is_still_a_successful_run(self):
        class Backend:
            name = "correctness-only"

            def run(inner, arch, program, mapping, directory):
                return Report(
                    backend=inner.name,
                    backend_version="1",
                    status="ok",
                    correctness="passed",
                    mapping_hash=fingerprint(mapping),
                )

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            summary = run(
                self.arch, self.program, PassthroughAnalyzer(), Backend(), 2, output
            )
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["successful_trial_ids"], ["trial_0000", "trial_0001"])
            self.assertIsNone(summary["best_trial_id"])
            self.assertFalse((output / "best_mapping.yaml").exists())

    def test_run_records_measured_cost_and_selects_lowest(self):
        class Backend:
            name = "measured-test"

            def __init__(self):
                self.values = iter([9, 4, 7])

            def run(inner, arch, program, mapping, directory):
                return Report(
                    backend=inner.name,
                    backend_version="1",
                    status="ok",
                    mapping_hash=fingerprint(mapping),
                    objective=Objective(
                        name="latency",
                        value=next(inner.values),
                        unit="cycles",
                        source="measured",
                    ),
                )

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            summary = run(
                self.arch, self.program, PassthroughAnalyzer(), Backend(), 3, output
            )
            self.assertEqual(summary["best_trial_id"], "trial_0001")
            self.assertEqual(len(summary["successful_trial_ids"]), 3)
            self.assertTrue((output / "best_mapping.yaml").exists())

    def test_run_rejects_non_positive_iterations(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run(
                    self.arch,
                    self.program,
                    PassthroughAnalyzer(),
                    object(),
                    0,
                    Path(temp) / "run",
                )


if __name__ == "__main__":
    unittest.main()
