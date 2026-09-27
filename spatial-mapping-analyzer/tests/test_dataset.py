"""Tests for measured mapping dataset persistence and export."""
import json
import tempfile
import unittest
from pathlib import Path

from analyzer import PassthroughAnalyzer
from dataset import MeasuredMappingRecord
from export_dataset import export_dataset
from pipeline import run
from report import Objective, Report
from specs import Architecture, Program, fingerprint, read_yaml

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.arch = Architecture.model_validate(
            read_yaml(ROOT / "examples/wormhole.yaml")
        )
        self.program = Program.model_validate(
            read_yaml(ROOT / "examples/scalar_diamond.yaml")
        )

    class MeasuredBackend:
        name = "measured-test"

        def __init__(self):
            self.values = iter([12, 11, 13])

        def execution_signature(self, architecture, program, mapping):
            return ["mapping", fingerprint(mapping)]

        def run(self, architecture, program, mapping, directory):
            return Report(
                backend=self.name,
                backend_version="1",
                status="ok",
                correctness="passed",
                mapping_hash=fingerprint(mapping),
                objective=Objective(
                    name="latency",
                    value=next(self.values),
                    unit="ns",
                    source="measured",
                ),
            )

    class CorrectnessBackend:
        name = "correctness-test"

        def execution_signature(self, architecture, program, mapping):
            return ["mapping", fingerprint(mapping)]

        def run(self, architecture, program, mapping, directory):
            return Report(
                backend=self.name,
                backend_version="1",
                status="ok",
                correctness="passed",
                mapping_hash=fingerprint(mapping),
            )

    def test_export_preserves_measurement_observations_and_deduplicates_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            measured_run = root / "measured"
            summary = run(
                self.arch,
                self.program,
                PassthroughAnalyzer(),
                self.MeasuredBackend(),
                3,
                measured_run,
            )
            output = root / "dataset.jsonl"
            count = export_dataset([measured_run, measured_run], output)

            self.assertEqual(count, 3)
            rows = [
                MeasuredMappingRecord.model_validate_json(line)
                for line in output.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row.run_id for row in rows}, {summary["run_id"]})
            self.assertEqual(
                {row.trial_id for row in rows},
                {"trial_0000", "trial_0001", "trial_0002"},
            )
            self.assertEqual(len({row.sample_id for row in rows}), 3)
            self.assertEqual(
                {row.measurement.value for row in rows},
                {11.0, 12.0, 13.0},
            )
            self.assertTrue(
                all(row.measurement.source == "measured" for row in rows)
            )

            history = [
                json.loads(line)
                for line in (measured_run / "history.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(item["execution_signature"] for item in history))
            self.assertTrue(
                all(item["run_id"] == summary["run_id"] for item in history)
            )

    def test_export_ignores_correctness_only_trials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            correctness_run = root / "correctness"
            run(
                self.arch,
                self.program,
                PassthroughAnalyzer(),
                self.CorrectnessBackend(),
                2,
                correctness_run,
            )

            output = root / "dataset.jsonl"
            count = export_dataset([correctness_run], output)
            self.assertEqual(count, 0)
            self.assertEqual(output.read_text(), "")

    def test_measured_trial_without_execution_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            measured_run = root / "measured"
            run(
                self.arch,
                self.program,
                PassthroughAnalyzer(),
                self.MeasuredBackend(),
                1,
                measured_run,
            )
            history_path = measured_run / "history.jsonl"
            raw = json.loads(history_path.read_text())
            raw["execution_signature"] = None
            history_path.write_text(json.dumps(raw) + "\n")

            with self.assertRaisesRegex(ValueError, "execution_signature"):
                export_dataset([measured_run], root / "dataset.jsonl")


if __name__ == "__main__":
    unittest.main()
