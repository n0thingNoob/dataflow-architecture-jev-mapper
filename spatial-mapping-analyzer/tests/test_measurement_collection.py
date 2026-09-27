"""Tests for randomized repeated measurement collection."""
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from backend_contract import BackendCapabilities, ObservedExecutionIdentity
from measurement_collection import (
    collect_measurements,
    randomized_schedule,
)
from report import MeasurementContext, Objective, Report
from specs import Architecture, Program, fingerprint, read_yaml

ROOT = Path(__file__).resolve().parents[1]


class FakeMeasuredBackend:
    name = "fake-measured"
    backend_version = "v1"
    search_capabilities = BackendCapabilities(
        topological_order=False,
        execution_policy=False,
        placement=True,
    )

    def __init__(self):
        self.counts = defaultdict(int)

    def candidate_execution_signature(self, architecture, program, mapping):
        return tuple(region.placement[0] for region in mapping.regions)

    def run(self, architecture, program, mapping, directory):
        placement = tuple(
            region.placement[0] for region in mapping.regions
        )
        repeat = self.counts[placement]
        self.counts[placement] += 1
        noise = (2, 0, 1)[repeat]
        value = 100 + 10 * placement[0] + noise
        return Report(
            backend=self.name,
            backend_version=self.backend_version,
            status="ok",
            correctness="passed",
            mapping_hash=fingerprint(mapping),
            objective=Objective(
                name="latency",
                value=value,
                unit="ns",
                source="measured",
            ),
            observed_execution=ObservedExecutionIdentity(
                kind="fake-placement-v1",
                value={"placement": list(placement)},
            ),
            measurement_context=MeasurementContext(
                measurement_version="fake-measurement-v1",
                runtime="fake-device",
                source="fake-profiler",
                analysis="latency",
                implementation_revision="revision-1",
                executable_sha256="deadbeef",
            ),
        )


class MeasurementCollectionTests(unittest.TestCase):
    def setUp(self):
        self.arch = Architecture.model_validate(
            read_yaml(ROOT / "examples/wormhole_tensix_probe.yaml")
        )
        self.program = Program.model_validate(
            read_yaml(ROOT / "examples/bf16_two_add_chain.yaml")
        )

    def test_randomized_schedule_is_seeded_and_balanced(self):
        from candidate_generator import generate_candidates

        backend = FakeMeasuredBackend()
        candidates = generate_candidates(
            self.arch,
            self.program,
            limit=3,
            capabilities=backend.search_capabilities,
            candidate_execution_signature=backend.candidate_execution_signature,
        )
        first = randomized_schedule(candidates, 2, 17)
        second = randomized_schedule(candidates, 2, 17)
        self.assertEqual(
            [fingerprint(item) for item in first],
            [fingerprint(item) for item in second],
        )
        counts = defaultdict(int)
        for mapping in first:
            counts[fingerprint(mapping)] += 1
        self.assertEqual(set(counts.values()), {2})

    def test_collection_preserves_raw_observations_and_aggregates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "measurements"
            result = collect_measurements(
                self.arch,
                self.program,
                FakeMeasuredBackend(),
                candidate_limit=2,
                repeats=3,
                seed=7,
                output=output,
            )

            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["trial_count"], 6)
            self.assertEqual(len(result["aggregates"]), 2)
            self.assertFalse((output / "best_mapping.yaml").exists())

            dataset_lines = (
                output / "measured_mappings.jsonl"
            ).read_text().splitlines()
            self.assertEqual(len(dataset_lines), 6)

            history = [
                json.loads(line)
                for line in (output / "history.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(history), 6)
            self.assertTrue(
                all(item["report"]["objective"] is not None for item in history)
            )

            for aggregate in result["aggregates"]:
                self.assertEqual(aggregate["count"], 3)
                self.assertEqual(aggregate["mad"], 1.0)
                self.assertEqual(
                    len(aggregate["observation_ids"]),
                    3,
                )
                self.assertEqual(len(aggregate["mapping_hashes"]), 1)

    def test_collection_requires_repeated_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "at least two repeats"):
                collect_measurements(
                    self.arch,
                    self.program,
                    FakeMeasuredBackend(),
                    candidate_limit=1,
                    repeats=1,
                    seed=0,
                    output=Path(temp) / "measurements",
                )


if __name__ == "__main__":
    unittest.main()
