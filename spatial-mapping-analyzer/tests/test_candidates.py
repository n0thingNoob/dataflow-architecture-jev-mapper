"""Unit tests for executable mapping candidate generation."""
import unittest
from pathlib import Path

from analyzer import EnumeratingAnalyzer
from candidate_generator import generate_candidates
from specs import Architecture, Program, fingerprint, read_yaml
from validator import validate_mapping

ROOT = Path(__file__).resolve().parents[1]


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/scalar_diamond.yaml"))

    def test_candidates_are_distinct_valid_and_explicitly_placed(self):
        candidates = generate_candidates(self.arch, self.program, limit=12)
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len({fingerprint(m) for m in candidates}), 12)
        self.assertEqual(
            {m.execution_policy for m in candidates},
            {"exclusive_cores_tensor_barrier", "exclusive_cores_dependency_barrier"},
        )
        for mapping in candidates:
            self.assertEqual(validate_mapping(self.arch, self.program, mapping), [])
            placed = [core for region in mapping.regions for core in region.placement]
            self.assertEqual(len(placed), len(set(placed)))

    def test_enumerating_analyzer_advances_with_feedback(self):
        analyzer = EnumeratingAnalyzer(candidate_limit=8)
        first = analyzer.propose_mapping(self.arch, self.program, [])
        second = analyzer.propose_mapping(self.arch, self.program, [{"trial_id": "trial_0000"}])
        self.assertNotEqual(fingerprint(first), fingerprint(second))

    def test_invalid_placements_are_rejected(self):
        mapping = generate_candidates(self.arch, self.program, limit=1)[0]

        mapping.regions[1].placement = list(mapping.regions[0].placement)
        codes = {item["code"] for item in validate_mapping(self.arch, self.program, mapping)}
        self.assertIn("PLACEMENT_OVERLAP", codes)

        mapping = generate_candidates(self.arch, self.program, limit=1)[0]
        mapping.regions[0].placement = None
        codes = {item["code"] for item in validate_mapping(self.arch, self.program, mapping)}
        self.assertIn("PARTIAL_PLACEMENT", codes)

        mapping = generate_candidates(self.arch, self.program, limit=1)[0]
        mapping.regions[0].placement = [self.arch.available_cores]
        codes = {item["code"] for item in validate_mapping(self.arch, self.program, mapping)}
        self.assertIn("PLACEMENT_RANGE", codes)


if __name__ == "__main__":
    unittest.main()
