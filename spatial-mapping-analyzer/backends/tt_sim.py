"""Real libttsim execution of a fixed dummy workload; DAG lowering is deferred."""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from backends.base import Metric, Objective, Report
from backends.tt_sim_dummy import WORKLOAD, generate_dummy
from specs.io import fingerprint, write_json

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "third_party/ttsim/src/_out/release_wh/libttsim.so"


class TTSimBackend:
    name = "tt-sim"

    def __init__(self, library=None, timeout_seconds=30.0):
        self.library = Path(library).resolve() if library is not None else DEFAULT_LIBRARY
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("TT-Sim timeout must be a finite positive number")
        self.timeout_seconds = timeout_seconds

    def run(self, architecture, program, mapping, workdir):
        workdir = workdir.resolve()
        manifest = generate_dummy(architecture, program, mapping, workdir)
        metadata = {
            "execution_kind": WORKLOAD, "program_dag_executed": False,
            "mapping_lowered": False, "dummy_correctness": "not_checked",
            "library": str(self.library), "manifest": "dummy.json",
            "result_file": "runner_result.json", "stdout": "stdout.log", "stderr": "stderr.log",
        }

        def failure(code, message, status="error"):
            return Report(backend=self.name, backend_version="0.2-dummy", status=status,
                          mapping_hash=fingerprint(mapping), message=message,
                          extensions={**metadata, "error_code": code})

        if architecture.extensions.get("target_family") != "wormhole":
            return failure("UNSUPPORTED_ARCHITECTURE", "Dummy runner requires target_family: wormhole", "unsupported")
        if not self.library.is_file():
            return failure("MISSING_SIMULATOR", "Wormhole libttsim.so is missing. Initialize the submodule, then run "
                           "python make.py src/_out/release_wh/libttsim.so in third_party/ttsim, "
                           "or supply --tt-sim-library. No mock fallback occurred.")
        metadata["library_sha256"] = hashlib.sha256(self.library.read_bytes()).hexdigest()
        command = [sys.executable, str(Path(__file__).with_name("tt_sim_runner.py").resolve()),
                   "--library", str(self.library), "--manifest", str(workdir / "dummy.json"),
                   "--result", str(workdir / "runner_result.json")]
        write_json(workdir / "invocation.json", {"argv": command, "cwd": str(workdir),
                   "timeout_seconds": self.timeout_seconds, "library_sha256": metadata["library_sha256"]})
        try:
            with (workdir / "stdout.log").open("w") as stdout, (workdir / "stderr.log").open("w") as stderr:
                process = subprocess.run(command, cwd=workdir, stdout=stdout, stderr=stderr,
                                         timeout=self.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return failure("SIMULATOR_TIMEOUT", f"TT-Sim process exceeded {self.timeout_seconds:g} seconds")
        except OSError as exc:
            return failure("SIMULATOR_LAUNCH_FAILED", str(exc))
        metadata["exit_code"] = process.returncode
        if process.returncode != 0:
            return failure("SIMULATOR_FAILED", f"TT-Sim runner exited with {process.returncode}; see stdout.log and stderr.log")
        try:
            result = json.loads((workdir / "runner_result.json").read_text())
            expected = {
                "status": "ok", "workload": WORKLOAD, "mapping_hash": fingerprint(mapping),
                "firmware_sha256": manifest["firmware_sha256"], "data_sha256": manifest["data_sha256"],
                "library_sha256": metadata["library_sha256"], "inputs": manifest["inputs"],
                "actual_result": manifest["expected_result"], "expected_result": manifest["expected_result"],
                "completion_flag": 1, "dummy_correctness": "passed", "program_dag_executed": False,
            }
            if any(result.get(key) != value for key, value in expected.items()):
                raise ValueError("Runner result failed identity or dummy-output checks")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            return failure("INVALID_SIMULATOR_RESULT", str(exc))
        metadata.update({"dummy_correctness": "passed", "execution": result})
        return Report(
            backend=self.name, backend_version="0.2-dummy", status="ok",
            mapping_hash=fingerprint(mapping),
            # The program DAG has not executed. Only the separate dummy was checked.
            correctness="not_checked",
            objective=Objective(name="passthrough_cost", value=1, unit="arbitrary", source="synthetic"),
            metrics={name: Metric(unit=unit, reason="Dummy execution does not measure mapped-DAG hardware performance")
                     for name, unit in [("total_cycles", "cycles"), ("latency", "ns"),
                                        ("core_utilization", "fraction"), ("stall_cycles", "cycles"),
                                        ("communication", "bytes"), ("buffer_usage", "bytes")]},
            message="Real TT-Sim BRISC dummy passed. Program DAG lowering is pending; selection cost remains a constant synthetic placeholder.",
            extensions=metadata,
        )
