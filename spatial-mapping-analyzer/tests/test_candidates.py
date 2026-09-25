"""Unit tests for executable mapping candidate generation."""
import unittest
from pathlib import Path

from analyzer import EnumeratingAnalyzer, PassthroughAnalyzer
from candidate_generator import _placement_rotations, _topological_orders, generate_candidates
from specs import Architecture, Program, fingerprint, read_yaml
from validator import validate_mapping

ROOT = Path(__file__).resolve().parents[1]


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/scalar_diamond.yaml"))

    def test_topological_orders_include_both_diamond_branch_orders(self):
        orders = _topological_orders(self.program, limit=8)
        self.assertIn(("Producer", "Left", "Right", "Join"), orders)
        self.assertIn(("Producer", "Right", "Left", "Join"), orders)
        self.assertTrue(all(order[0] == "Producer" and order[-1] == "Join" for order in orders))

    def test_placement_rotations_are_deterministic(self):
        self.assertEqual(
            list(_placement_rotations(4, 3)),
            [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)],
        )

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

    def test_candidate_generation_rejects_invalid_limits_and_insufficient_cores(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            generate_candidates(self.arch, self.program, limit=0)

        small_arch = self.arch.model_copy(deep=True)
        small_arch.available_cores = len(self.program.ops) - 1
        small_arch.grid.cols = small_arch.available_cores
        with self.assertRaisesRegex(ValueError, "one available core per op"):
            generate_candidates(small_arch, self.program)

    def test_passthrough_is_stable_and_parallel_flag_changes_policy(self):
        serial = PassthroughAnalyzer().propose_mapping(self.arch, self.program)
        parallel = PassthroughAnalyzer(parallel=True).propose_mapping(self.arch, self.program)
        self.assertEqual(
            [r.ops for r in serial.regions],
            [[op.id] for op in self.program.ordered_ops()],
        )
        self.assertEqual(serial.execution_policy, "exclusive_cores_tensor_barrier")
        self.assertEqual(parallel.execution_policy, "exclusive_cores_dependency_barrier")
        self.assertTrue(all(region.placement is None for region in serial.regions))

    def test_enumerating_analyzer_advances_and_wraps_with_feedback(self):
        analyzer = EnumeratingAnalyzer(candidate_limit=3)
        mappings = [
            analyzer.propose_mapping(self.arch, self.program, [{"trial_id": str(i)} for i in range(n)])
            for n in range(4)
        ]
        hashes = [fingerprint(mapping) for mapping in mappings]
        self.assertEqual(len(set(hashes[:3])), 3)
        self.assertEqual(hashes[0], hashes[3])

    def test_enumerating_analyzer_rejects_invalid_limit(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            EnumeratingAnalyzer(0)

    def test_invalid_placements_are_rejected(self):
        mapping = generate_candidates(self.arch, self.program, limit=1)[0]
        mapping.regions[1].placement = list(mapping.regions[0].placement)
        self.assertIn(
            "PLACEMENT_OVERLAP",
            {item["code"] for item in validate_mapping(self.arch, self.program, mapping)},
        )

        mapping = generate_candidates(self.arch, self.program, limit=1)[0]
        mapping.regions[0].placement = None
        self.assertIn(
            "PARTIAL_PLACEMENT",
            {item["code"] for item in validate_mapping(self.arch, self.program, mapping)},
        )

        mapping = generate_candidates(self.arch, self.program, limit=1)[0]
        mapping.regions[0].placement = [self.arch.available_cores]
        self.assertIn(
            "PLACEMENT_RANGE",
            {item["code"] for item in validate_mapping(self.arch, self.program, mapping)},
        )

        mapping = generate_candidates(self.arch, self.program, limit=1)[0]
        mapping.regions[0].cores = 2
        self.assertIn(
            "PLACEMENT_SIZE",
            {item["code"] for item in validate_mapping(self.arch, self.program, mapping)},
        )


if __name__ == "__main__":
    unittest.main()
