"""Unit tests for executable mapping candidate generation."""
import unittest
from pathlib import Path

from pydantic import ValidationError

from analyzer import EnumeratingAnalyzer, PassthroughAnalyzer
from backend_contract import BackendCapabilities
from candidate_generator import (
    _placement_rotations,
    _topological_orders,
    generate_candidates,
)
from mapping_ir import Mapping
from specs import Architecture, Program, fingerprint, read_yaml
from validator import validate_mapping
from workload import check_supported

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

    def test_capabilities_limit_search_to_effective_dimensions(self):
        capabilities = BackendCapabilities(
            topological_order=False,
            execution_policy=False,
            placement=True,
        )
        candidates = generate_candidates(
            self.arch, self.program, limit=4, capabilities=capabilities
        )
        stable_order = [op.id for op in self.program.ordered_ops()]
        self.assertEqual(len(candidates), 4)
        self.assertTrue(
            all(
                [region.ops[0] for region in mapping.regions] == stable_order
                for mapping in candidates
            )
        )
        self.assertEqual(
            {mapping.execution_policy for mapping in candidates},
            {"exclusive_cores_tensor_barrier"},
        )
        placements = {
            tuple(region.placement[0] for region in mapping.regions)
            for mapping in candidates
        }
        self.assertEqual(len(placements), 4)

    def test_execution_signature_deduplicates_equivalent_ir_mappings(self):
        def signature(architecture, program, mapping):
            return tuple(
                (region.ops[0], region.placement[0])
                for region in mapping.regions
            )

        candidates = generate_candidates(
            self.arch,
            self.program,
            limit=6,
            candidate_key=signature,
        )
        signatures = [
            signature(self.arch, self.program, mapping)
            for mapping in candidates
        ]
        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(set(signatures)), 6)
        self.assertEqual(
            {mapping.execution_policy for mapping in candidates},
            {"exclusive_cores_tensor_barrier"},
        )

    def test_topological_order_does_not_change_op_to_core_assignment(self):
        capabilities = BackendCapabilities(
            topological_order=True,
            execution_policy=False,
            placement=True,
        )
        candidates = generate_candidates(
            self.arch, self.program, limit=8, capabilities=capabilities
        )
        by_order = {}
        for mapping in candidates:
            order = tuple(region.ops[0] for region in mapping.regions)
            placement = {
                region.ops[0]: region.placement[0]
                for region in mapping.regions
            }
            by_order.setdefault(order, placement)

        self.assertGreaterEqual(len(by_order), 2)
        placements = list(by_order.values())
        self.assertEqual(placements[0], placements[1])

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

    def test_placement_schema_rejects_bool_float_and_string(self):
        raw = generate_candidates(self.arch, self.program, limit=1)[0].model_dump()
        for bad in [False, 0.0, "0"]:
            with self.subTest(bad=bad):
                candidate = {**raw, "regions": [dict(region) for region in raw["regions"]]}
                candidate["regions"][0]["placement"] = [bad]
                with self.assertRaises(ValidationError):
                    Mapping.model_validate(candidate)

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

    def test_backend_rejects_architecture_larger_than_harness_capacity(self):
        large_arch = self.arch.model_copy(deep=True)
        large_arch.grid.cols = 9
        large_arch.available_cores = 9
        mapping = PassthroughAnalyzer().propose_mapping(large_arch, self.program)
        with self.assertRaisesRegex(ValueError, "at most 8 logical cores"):
            check_supported(large_arch, self.program, mapping)


if __name__ == "__main__":
    unittest.main()
