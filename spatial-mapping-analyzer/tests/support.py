import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analyzer.passthrough import PassthroughAnalyzer
from backends.tt_sim import DEFAULT_LIBRARY
from specs.io import read_yaml
from specs.models import Architecture, Program

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("TT_SIM_TEST_LIBRARY", str(DEFAULT_LIBRARY))).resolve()


class AnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        self.output = self.workdir / "run"
        self.arch = Architecture.model_validate(read_yaml(ROOT / "examples/wormhole.yaml"))
        self.program = Program.model_validate(read_yaml(ROOT / "examples/matmul_relu_matmul.yaml"))
        self.mapping = PassthroughAnalyzer().propose_mapping(self.arch, self.program)

    def cli(self, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "run_analyzer.py"),
             "--arch", str(ROOT / "examples/wormhole.yaml"),
             "--program", str(ROOT / "examples/matmul_relu_matmul.yaml"),
             "--iterations", "3", "--output", str(self.output), *extra],
            cwd=self.workdir, text=True, capture_output=True, timeout=60)
