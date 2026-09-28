"""Tests for topology-aware heuristic scoring and evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dataset_split import build_split_manifest
from evaluate_scorer import evaluate
from evaluate_ttsim_profile import evaluate_profile_run
from mapping_ir import Mapping, Region
from scorer import HeuristicScorer
from specs import Architecture, Program, fingerprint, read_yaml

ROOT = Path(__file__).resolve().parents[1]


def explicit_mapping(arch, program, placements):
    return Mapping(
        program_id=program.id,
        program_hash=fingerprint(program),
        architecture_hash=fingerprint(arch),
        execution_policy="exclusive_cores_tensor_barrier",
        regions=[
            Region(
                id=f"region_{index}",
                ops=[op.id],
                cores=1,
                placement=[placements[index]],
            )
            for index, op in enumerate(program.ordered_ops())
        ],
    )


def split_record(program_group_id, execution_group_id):
    return SimpleNamespace(
        program_group_id=program_group_id,
        execution_group_id=execution_group_id,
    )


class FakeMeasurementContext:
    def model_dump(self, mode=None):
        return {
            "measurement_version": "test-v1",
            "runtime": "device",
            "source": "test-profiler",
            "analysis": "duration",
            "implementation_revision": "rev",
            "executable_sha256": "deadbeef",
            "artifacts": {},
            "configuration": {},
        }


def evaluation_record(
    arch,
    program,
    mapping,
    execution_group_id,
    value,
    trial_id,
):
    return SimpleNamespace(
        program_group_id="program_test",
        execution_group_id=execution_group_id,
        architecture=arch,
        program=program,
        mapping=mapping,
        mapping_hash=fingerprint(mapping),
        measurement=SimpleNamespace(
            name="device_kernel_duration",
            unit="ns",
            source="measured",
            value=float(value),
        ),
        measurement_context=FakeMeasurementContext(),
        trial_id=trial_id,
    )


class HeuristicBaselineTests(unittest.TestCase):
    def setUp(self):
        self.arch = Architecture.model_validate(
            read_yaml(ROOT / "examples/wormhole_tensix_probe.yaml")
        )
        self.program = Program.model_validate(
            read_yaml(ROOT / "examples/bf16_two_add_chain.yaml")
        )

    def test_chain_prefers_adjacent_placement(self):
        adjacent = explicit_mapping(self.arch, self.program, [0, 1])
        diagonal = explicit_mapping(self.arch, self.program, [0, 3])
        scores = HeuristicScorer().score_candidates(
            self.arch, self.program, [adjacent, diagonal]
        )
        self.assertLess(scores[0], scores[1])

    def test_breakdown_exposes_diagnostics_without_ranking_endpoint_traffic(self):
        mapping = explicit_mapping(self.arch, self.program, [0, 3])
        scorer = HeuristicScorer()
        breakdown = scorer.score_breakdown(
            self.arch, self.program, mapping
        )
        self.assertEqual(
            breakdown["score"],
            breakdown["critical_path_weighted_hops"]
            + breakdown["network_byte_hops"]
            + breakdown["max_link_load_ratio"],
        )
        self.assertGreater(breakdown["max_core_traffic_ratio"], 0)
        self.assertIsNone(breakdown["estimated_critical_path_cycles"])

    def test_tensor_size_weights_communication_cost(self):
        program = Program.model_validate(
            {
                "schema_version": "0.1",
                "id": "weighted_chain",
                "tensors": {
                    "a": {"shape": [32, 32], "dtype": "bfloat16"},
                    "b": {"shape": [32, 32], "dtype": "bfloat16"},
                    "small": {"shape": [1], "dtype": "bfloat16"},
                    "large": {"shape": [32, 32], "dtype": "bfloat16"},
                    "y": {"shape": [32, 32], "dtype": "bfloat16"},
                },
                "inputs": ["a", "b"],
                "outputs": ["y"],
                "ops": [
                    {"id": "A", "op": "add", "inputs": ["a", "b"], "output": "small"},
                    {"id": "B", "op": "add", "inputs": ["small", "a"], "output": "large"},
                    {"id": "C", "op": "add", "inputs": ["large", "b"], "output": "y"},
                ],
                "edges": [
                    {"source": "A", "target": "B", "tensor": "small", "input_index": 0},
                    {"source": "B", "target": "C", "tensor": "large", "input_index": 0},
                ],
            }
        )
        small_far = explicit_mapping(self.arch, program, [0, 3, 2])
        large_far = explicit_mapping(self.arch, program, [0, 1, 2])
        scores = HeuristicScorer().score_candidates(
            self.arch, program, [small_far, large_far]
        )
        self.assertLess(scores[0], scores[1])

    def test_link_congestion_breaks_equal_total_hop_tie(self):
        arch = self.arch.model_copy(deep=True)
        arch.grid.rows = 3
        arch.grid.cols = 3
        arch.available_cores = 9
        program = Program.model_validate(
            {
                "schema_version": "0.1",
                "id": "diamond",
                "tensors": {
                    name: {"shape": [32, 32], "dtype": "bfloat16"}
                    for name in ["a", "p", "l", "r", "y"]
                },
                "inputs": ["a"],
                "outputs": ["y"],
                "ops": [
                    {"id": "P", "op": "relu", "inputs": ["a"], "output": "p"},
                    {"id": "L", "op": "relu", "inputs": ["p"], "output": "l"},
                    {"id": "R", "op": "relu", "inputs": ["p"], "output": "r"},
                    {"id": "J", "op": "add", "inputs": ["l", "r"], "output": "y"},
                ],
                "edges": [
                    {"source": "P", "target": "L", "tensor": "p", "input_index": 0},
                    {"source": "P", "target": "R", "tensor": "p", "input_index": 0},
                    {"source": "L", "target": "J", "tensor": "l", "input_index": 0},
                    {"source": "R", "target": "J", "tensor": "r", "input_index": 1},
                ],
            }
        )
        low_congestion = explicit_mapping(arch, program, [0, 2, 6, 8])
        high_congestion = explicit_mapping(arch, program, [0, 1, 3, 8])
        scorer = HeuristicScorer()
        low = scorer.score_breakdown(arch, program, low_congestion)
        high = scorer.score_breakdown(arch, program, high_congestion)
        self.assertEqual(
            low["network_byte_hops"],
            high["network_byte_hops"],
        )
        self.assertLess(
            low["max_link_load_ratio"],
            high["max_link_load_ratio"],
        )
        self.assertLess(low["score"], high["score"])

    def test_unsupported_network_topology_is_rejected(self):
        arch = self.arch.model_copy(deep=True)
        arch.network_topology = "host-mediated"
        mapping = explicit_mapping(arch, self.program, [0, 1])
        with self.assertRaisesRegex(ValueError, "supported logical mesh"):
            HeuristicScorer().score_breakdown(
                arch, self.program, mapping
            )

    def test_bandwidth_and_latency_enable_cycle_diagnostic(self):
        arch = self.arch.model_copy(deep=True)
        arch.bandwidth_bytes_per_cycle = 32.0
        arch.link_latency_cycles = 2
        mapping = explicit_mapping(arch, self.program, [0, 3])
        breakdown = HeuristicScorer().score_breakdown(
            arch, self.program, mapping
        )
        self.assertGreater(
            breakdown["estimated_critical_path_cycles"], 0
        )

    def test_balanced_split_is_deterministic_and_nonempty_when_possible(self):
        records = [
            split_record(f"program_{index}", f"execution_{index}")
            for index in range(10)
        ]
        first = build_split_manifest(records)
        second = build_split_manifest(list(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(first["counts"]["train"], 7)
        self.assertGreater(first["counts"]["validation"], 0)
        self.assertGreater(first["counts"]["test"], 0)

    def test_balanced_split_handles_tiny_datasets(self):
        one = build_split_manifest(
            [split_record("program_a", "execution_a")]
        )
        self.assertEqual(one["counts"], {
            "train": 1,
            "validation": 0,
            "test": 0,
        })
        two = build_split_manifest(
            [
                split_record("program_a", "execution_a"),
                split_record("program_b", "execution_b"),
            ]
        )
        self.assertEqual(two["counts"], {
            "train": 1,
            "validation": 0,
            "test": 1,
        })

    def test_split_rejects_execution_group_leakage(self):
        records = [
            split_record("program_a", "execution_shared"),
            split_record("program_b", "execution_shared"),
        ]
        with self.assertRaisesRegex(ValueError, "leak"):
            build_split_manifest(records)

    def test_evaluation_rejects_stale_split_manifest(self):
        mapping = explicit_mapping(self.arch, self.program, [0, 1])
        records = [
            evaluation_record(
                self.arch, self.program, mapping,
                "execution_a", 100, "trial_a",
            )
        ]
        manifest = {
            "groups": {"program_other": "test"},
        }
        with self.assertRaisesRegex(ValueError, "does not cover"):
            evaluate(
                records,
                HeuristicScorer(),
                split_manifest=manifest,
                split="test",
            )

    def test_evaluation_rejects_ambiguous_execution_group_mapping(self):
        first = explicit_mapping(self.arch, self.program, [0, 1])
        second = explicit_mapping(self.arch, self.program, [0, 2])
        records = [
            evaluation_record(
                self.arch, self.program, first,
                "execution_shared", 100, "trial_a",
            ),
            evaluation_record(
                self.arch, self.program, second,
                "execution_shared", 101, "trial_b",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "multiple Mapping IR"):
            evaluate(records, HeuristicScorer())

    def test_ttsim_profile_evaluation_aggregates_repeats(self):
        adjacent_a = explicit_mapping(self.arch, self.program, [0, 1])
        adjacent_b = explicit_mapping(self.arch, self.program, [0, 2])
        diagonal = explicit_mapping(self.arch, self.program, [0, 3])
        mappings = [
            (adjacent_a, [100, 102]),
            (adjacent_b, [120, 122]),
            (diagonal, [90, 92]),
        ]
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            trials = []
            index = 0
            for mapping, values in mappings:
                for value in values:
                    trials.append({
                        "architecture": self.arch.model_dump(mode="json"),
                        "program": self.program.model_dump(mode="json"),
                        "mapping": mapping.model_dump(mode="json"),
                        "requested_execution_signature": ["test", fingerprint(mapping)],
                        "report": {
                            "status": "ok",
                            "mapping_hash": fingerprint(mapping),
                            "objective": {
                                "name": "ttsim_profile_kernel_duration",
                                "value": float(value),
                                "unit": "ns",
                                "source": "estimated",
                            },
                        },
                        "trial_id": f"trial_{index:04d}",
                    })
                    index += 1
            (run_dir / "history.jsonl").write_text(
                "".join(json.dumps(trial) + "\n" for trial in reversed(trials))
            )
            result = evaluate_profile_run(run_dir)

        self.assertEqual(result["source_kind"], "simulator_estimate")
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["repeat_counts"], [2])
        self.assertEqual(result["top_score_tie_count"], 2)
        self.assertEqual(result["oracle_profile_ns"], 91.0)
        self.assertEqual(result["top_tie_best_profile_ns"], 101.0)
        self.assertEqual(result["top_tie_worst_profile_ns"], 121.0)

    def test_ttsim_profile_evaluation_rejects_non_estimated_objective(self):
        mapping = explicit_mapping(self.arch, self.program, [0, 1])
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            trial = {
                "architecture": self.arch.model_dump(mode="json"),
                "program": self.program.model_dump(mode="json"),
                "mapping": mapping.model_dump(mode="json"),
                "report": {
                    "status": "ok",
                    "mapping_hash": fingerprint(mapping),
                    "objective": {
                        "name": "device_kernel_duration",
                        "value": 100.0,
                        "unit": "ns",
                        "source": "measured",
                    },
                },
            }
            (run_dir / "history.jsonl").write_text(json.dumps(trial) + "\n")
            with self.assertRaisesRegex(ValueError, "not a TT-Sim profiler estimate"):
                evaluate_profile_run(run_dir)

    def test_evaluation_reports_tie_regret_range_and_is_order_independent(self):
        adjacent_a = explicit_mapping(self.arch, self.program, [0, 1])
        adjacent_b = explicit_mapping(self.arch, self.program, [0, 2])
        diagonal = explicit_mapping(self.arch, self.program, [0, 3])
        records = [
            evaluation_record(
                self.arch, self.program, adjacent_a,
                "execution_a", 100, "trial_a0",
            ),
            evaluation_record(
                self.arch, self.program, adjacent_a,
                "execution_a", 102, "trial_a1",
            ),
            evaluation_record(
                self.arch, self.program, adjacent_b,
                "execution_b", 120, "trial_b0",
            ),
            evaluation_record(
                self.arch, self.program, diagonal,
                "execution_c", 90, "trial_c0",
            ),
        ]
        result = evaluate(records, HeuristicScorer())
        reversed_result = evaluate(
            list(reversed(records)),
            HeuristicScorer(),
        )
        self.assertEqual(result, reversed_result)
        case = result["cases"][0]
        self.assertEqual(case["top_score_tie_count"], 2)
        self.assertEqual(case["oracle_latency"], 90.0)
        self.assertEqual(case["top_tie_best_latency"], 101.0)
        self.assertEqual(case["top_tie_worst_latency"], 120.0)
        self.assertLess(
            case["top_tie_regret_min"],
            case["top_tie_regret_max"],
        )


if __name__ == "__main__":
    unittest.main()
