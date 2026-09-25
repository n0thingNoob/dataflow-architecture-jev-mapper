"""TT-Metal backend for a two-core producer-consumer Tensix chain."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

from report import Report, unsupported_metrics
from specs import fingerprint, write_json
from tt_metal_probe import TTMetalProbeBackend, logical_core_to_coord
from validator import validate_inputs, validate_mapping


def check_chain_supported(architecture, program, mapping):
    errors = validate_inputs(architecture, program) or validate_mapping(
        architecture, program, mapping
    )
    if errors:
        raise ValueError(str(errors))
    if architecture.extensions.get("target_family") != "wormhole":
        raise ValueError("Tensix chain currently supports Wormhole only")
    if len(program.ops) != 2 or any(op.op != "add" for op in program.ops):
        raise ValueError("Tensix chain supports exactly two add ops")

    ordered = program.ordered_ops()
    first, second = ordered
    if first.output not in second.inputs:
        raise ValueError("Second add must consume the first add output")
    if len(mapping.regions) != 2:
        raise ValueError("Tensix chain requires two regions")

    region_by_op = {
        op_id: region
        for region in mapping.regions
        for op_id in region.ops
    }
    if set(region_by_op) != {first.id, second.id}:
        raise ValueError("Each add op must be mapped to its own region")

    placements = []
    for op in (first, second):
        region = region_by_op[op.id]
        if region.cores != 1 or region.placement is None or len(region.placement) != 1:
            raise ValueError("Each chain stage requires one explicitly placed core")
        placements.append(region.placement[0])
    if placements[0] == placements[1]:
        raise ValueError("Producer and consumer must use different cores")

    tensor_names = set(program.inputs + [op.output for op in program.ops])
    tensors = [program.tensors[name] for name in tensor_names]
    if any(t.dtype != "bfloat16" or t.shape != [32, 32] for t in tensors):
        raise ValueError("Tensix chain supports 32x32 BF16 tiles only")

    return (
        first,
        second,
        placements,
        [
            logical_core_to_coord(architecture, placements[0]),
            logical_core_to_coord(architecture, placements[1]),
        ],
    )


class TTMetalChainBackend(TTMetalProbeBackend):
    name = "tt-metal-tensix-chain"

    def _failure(self, mapping, code, message, status="error"):
        return Report(
            backend=self.name,
            backend_version="bf16-add-chain-v1",
            status=status,
            mapping_hash=fingerprint(mapping),
            message=message,
            extensions={
                "error_code": code,
                "compute_path": "Two-stage Tensix add chain via TT-Metal",
            },
        )

    def run(self, architecture, program, mapping, workdir):
        workdir = workdir.resolve()
        try:
            first, second, logical_cores, coords = check_chain_supported(
                architecture, program, mapping
            )
        except ValueError as exc:
            return self._failure(
                mapping, "UNSUPPORTED_PROGRAM", str(exc), status="unsupported"
            )

        try:
            simulator_library = self._prepare_simulator_directory(workdir)
        except (OSError, FileNotFoundError) as exc:
            return self._failure(mapping, "PROBE_ENVIRONMENT", str(exc))

        result_path = workdir / "tensix_chain_result.json"
        kernel_root = Path(__file__).resolve().parent / "tensix_probe" / "kernels"
        command = [
            str(self.probe_binary),
            "--producer-x",
            str(coords[0][0]),
            "--producer-y",
            str(coords[0][1]),
            "--consumer-x",
            str(coords[1][0]),
            "--consumer-y",
            str(coords[1][1]),
            "--kernel-root",
            str(kernel_root),
            "--result",
            str(result_path),
        ]
        env = os.environ.copy()
        env.update(
            {
                "TT_METAL_HOME": str(self.tt_metal_home),
                "TT_METAL_SIMULATOR": str(simulator_library),
                "TT_METAL_SIMULATOR_HOME": str(simulator_library.parent),
                "TT_METAL_SLOW_DISPATCH_MODE": "1",
                "TT_METAL_FORCE_JIT_COMPILE": "1",
                "TT_METAL_DISABLE_SFPLOADMACRO": "1",
            }
        )
        write_json(
            workdir / "tensix_chain_invocation.json",
            {
                "argv": command,
                "cwd": str(self.tt_metal_home),
                "logical_cores": logical_cores,
                "physical_cores": [list(coord) for coord in coords],
                "stages": [first.id, second.id],
                "ttsim_sha256": hashlib.sha256(
                    simulator_library.read_bytes()
                ).hexdigest(),
            },
        )

        try:
            with (workdir / "tensix_chain_stdout.log").open("w") as stdout, (
                workdir / "tensix_chain_stderr.log"
            ).open("w") as stderr:
                process = subprocess.run(
                    command,
                    cwd=self.tt_metal_home,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            if process.returncode != 0:
                return self._failure(
                    mapping,
                    "PROBE_FAILED",
                    f"Tensix chain probe exited with {process.returncode}",
                )
            result = json.loads(result_path.read_text())
        except subprocess.TimeoutExpired:
            return self._failure(
                mapping,
                "PROBE_TIMEOUT",
                f"Tensix chain probe exceeded {self.timeout_seconds:g} seconds",
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self._failure(mapping, "PROBE_RESULT_ERROR", str(exc))

        expected_producer = list(coords[0])
        expected_consumer = list(coords[1])
        if (
            not isinstance(result, dict)
            or result.get("producer_core") != expected_producer
            or result.get("consumer_core") != expected_consumer
            or result.get("elements") != 1024
            or result.get("intermediate_transport") != "noc_direct"
            or result.get("intermediate_returned_to_host") is not False
            or type(result.get("passed")) is not bool
        ):
            return self._failure(
                mapping,
                "PROBE_RESULT_MISMATCH",
                "Chain result does not match requested mapping",
            )

        passed = result["passed"]
        return Report(
            backend=self.name,
            backend_version="bf16-add-chain-v1",
            status="ok" if passed else "error",
            mapping_hash=fingerprint(mapping),
            correctness="passed" if passed else "failed",
            objective=None,
            metrics=unsupported_metrics(
                "Two-core chain validates mapped Tensix dataflow but exposes no timing objective"
            ),
            message=(
                f"{first.id}@{tuple(coords[0])} -> {second.id}@{tuple(coords[1])} executed with direct NoC intermediate"
                if passed
                else "Two-core Tensix chain result mismatch"
            ),
            extensions={
                "compute_path": "Two Tensix UNPACK/MATH/PACK stages via TT-Metal",
                "intermediate_transport": "noc_direct",
                "intermediate_returned_to_host": False,
                "stage_ops": [first.id, second.id],
                "logical_cores": logical_cores,
                "physical_cores": [list(coord) for coord in coords],
                "probe_result": result,
            },
        )
