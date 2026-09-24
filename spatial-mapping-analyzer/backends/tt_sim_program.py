"""Real int32 DAG execution with CPU reference verification, no performance claim."""
import hashlib
import json
import subprocess

from backends.base import Objective, Report, unsupported_metrics
from backends.program_workload import check_supported, input_values, lower, reference
from backends.tt_sim import TTSimBackend
from specs.io import fingerprint, write_json


class TTSimProgramBackend(TTSimBackend):
    name = "tt-sim-program"
    runner_module = "backends.tt_sim_program_runner"
    manifest_filename = "program_execution.json"

    def __init__(self, library=None, timeout_seconds=30, inputs=None, seed=0):
        super().__init__(library, timeout_seconds)
        self.inputs, self.seed = inputs, seed

    def run(self, architecture, program, mapping, workdir):
        workdir = workdir.resolve()
        identity = dict(backend=self.name, backend_version="brisc-int32-v1", mapping_hash=fingerprint(mapping))
        scope = {"program_dag_executed": False, "mapping_lowered": False,
                 "compute_path": "BRISC RV32IM; no Tensix FPU/SFPU", "transfers": "host-mediated PCI/BAR"}

        def failure(code, message, status="error"):
            return Report(**identity, status=status, message=message, extensions={**scope, "error_code": code})

        try:
            check_supported(architecture, program, mapping)
        except ValueError as exc:
            return failure("UNSUPPORTED_PROGRAM", str(exc), "unsupported")
        inputs = input_values(program, self.inputs, self.seed)
        lower(program, mapping, inputs, workdir)
        expected = reference(program, inputs)
        write_json(workdir / "inputs.json", inputs)
        write_json(workdir / "reference.json", expected)  # never passed to the runner/device
        if not self.library.is_file():
            return failure("LIBRARY_MISSING", f"Build Wormhole TT-Sim first: {self.library}")
        library_hash = hashlib.sha256(self.library.read_bytes()).hexdigest()
        try:
            process = self._invoke(workdir, library_hash)
            if process.returncode:
                return failure("SIMULATOR_FAILED", f"Runner exited with {process.returncode}; see stderr.log")
            result = json.loads((workdir / "runner_result.json").read_text())
            manifest_hash = hashlib.sha256((workdir / self.manifest_filename).read_bytes()).hexdigest()
            if (not isinstance(result, dict) or result.get("status") != "ok"
                    or result.get("mapping_hash") != identity["mapping_hash"]
                    or result.get("manifest_sha256") != manifest_hash
                    or result.get("library_sha256") != library_hash):
                return failure("INVALID_SIMULATOR_RESULT", "Runner result provenance mismatch")
        except subprocess.TimeoutExpired:
            return failure("SIMULATOR_TIMEOUT", f"Runner exceeded {self.timeout_seconds:g} seconds")
        except (OSError, ValueError) as exc:
            return failure("SIMULATOR_RESULT_ERROR", str(exc))
        all_values = {**inputs, **expected}
        passed = result.get("tensors") == expected and result.get("outputs") == {t: all_values[t] for t in program.outputs}
        scope.update(program_dag_executed=True, mapping_lowered=True, execution=result)
        write_json(workdir / "correctness.json", {"passed": passed, "comparison": "exact int32, all intermediate tensors and outputs"})
        return Report(**identity, status="ok" if passed else "error", correctness="passed" if passed else "failed",
                      objective=Objective(name="passthrough_cost", value=1, unit="arbitrary", source="synthetic") if passed else None,
                      metrics=unsupported_metrics("BRISC correctness harness does not measure accelerator performance"),
                      message="DAG matches CPU reference" if passed else "DAG differs from CPU reference",
                      extensions=scope)
