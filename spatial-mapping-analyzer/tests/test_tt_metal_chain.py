"""Unit tests for the two-core TT-Metal Tensix chain backend."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from candidate_generator import generate_candidates
from specs import Architecture, Program, read_yaml
from tt_metal_chain import TTMetalChainBackend, check_chain_supported

ROOT = Path(__file__).resolve().parents[1]


class TTMetalChainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workdir = self.root / "trial"
        self.workdir.mkdir()

        self.tt_metal_home = self.root / "tt-metal"
        descriptor = self.tt_metal_home / "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
        descriptor.parent.mkdir(parents=True)
        descriptor.write_text("arch: wormhole\n")

        self.probe = self.root / "spatial_tensix_chain_probe"
        self.probe.write_text("fake")
        self.library = self.root / "libttsim.so"
        self.library.write_bytes(b"fake-ttsim")

        self.arch = Architecture.model_validate(
            read_yaml(ROOT / "examples/wormhole_tensix_probe.yaml")
        )
        self.program = Program.model_validate(
            read_yaml(ROOT / "examples/bf16_two_add_chain.yaml")
        )
        self.mapping = generate_candidates(self.arch, self.program, limit=1)[0]

    def backend(self):
        return TTMetalChainBackend(
            self.tt_metal_home,
            self.probe,
            self.library,
            timeout_seconds=5,
        )

    def test_chain_contract_extracts_two_distinct_stages(self):
        first, second, logical, coords = check_chain_supported(
            self.arch, self.program, self.mapping
        )
        self.assertEqual([first.id, second.id], ["Add0", "Add1"])
        self.assertEqual(logical, [0, 1])
        self.assertEqual(coords, [(0, 0), (1, 0)])

    def test_backend_invokes_both_mapped_cores(self):
        def fake_run(command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "producer_core": [0, 0],
                        "consumer_core": [1, 0],
                        "intermediate_transport": "noc_direct",
                        "intermediate_returned_to_host": False,
                        "elements": 1024,
                    }
                )
            )
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_chain.subprocess.run", side_effect=fake_run) as run:
            report = self.backend().run(
                self.arch, self.program, self.mapping, self.workdir
            )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.correctness, "passed")
        self.assertIsNone(report.objective)
        self.assertEqual(report.extensions["logical_cores"], [0, 1])
        self.assertEqual(report.extensions["physical_cores"], [[0, 0], [1, 0]])
        self.assertEqual(report.extensions["intermediate_transport"], "noc_direct")
        self.assertFalse(report.extensions["intermediate_returned_to_host"])

        command = run.call_args.args[0]
        self.assertIn("--producer-x", command)
        self.assertIn("--consumer-x", command)
        self.assertIn("--kernel-root", command)

    def test_same_core_mapping_is_rejected(self):
        self.mapping.regions[1].placement = [0]
        with self.assertRaises(ValueError):
            check_chain_supported(self.arch, self.program, self.mapping)

    def test_wrong_dag_is_rejected(self):
        bad = self.program.model_copy(deep=True)
        bad.ops[1].inputs = ["a", "c"]
        bad.edges = []
        with self.assertRaises(ValueError):
            check_chain_supported(self.arch, bad, self.mapping)

    def test_probe_result_must_prove_direct_noc_transport(self):
        def fake_run(command, **kwargs):
            result_path = Path(command[command.index("--result") + 1])
            result_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "producer_core": [0, 0],
                        "consumer_core": [1, 0],
                        "intermediate_transport": "host",
                        "intermediate_returned_to_host": True,
                        "elements": 1024,
                    }
                )
            )
            return subprocess.CompletedProcess(command, 0)

        with patch("tt_metal_chain.subprocess.run", side_effect=fake_run):
            report = self.backend().run(
                self.arch, self.program, self.mapping, self.workdir
            )
        self.assertEqual(report.status, "error")
        self.assertEqual(
            report.extensions["error_code"], "PROBE_RESULT_MISMATCH"
        )


if __name__ == "__main__":
    unittest.main()
