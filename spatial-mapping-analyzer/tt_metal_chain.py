"""TT-Metal backend for a two-core producer-consumer Tensix chain."""
import csv
import hashlib
from pathlib import Path

from backend_contract import ObservedExecutionIdentity
from report import MeasurementContext, Metric, Objective, Report, unsupported_metrics
from specs import fingerprint
from tt_metal_probe import (
    TTMetalProbeBackend,
    logical_core_id_to_tt_metal_logical_core,
)
from validator import validate_inputs, validate_mapping


def kernel_bundle_sha256(kernel_root):
    digest = hashlib.sha256()
    for path in sorted(kernel_root.glob("*.cpp")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def profiler_csv_durations(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        field = "DEVICE KERNEL DURATION [ns]"
        if field not in (reader.fieldnames or []):
            raise ValueError(f"Profiler report is missing {field}")
        values = []
        for row in reader:
            raw = (row.get(field) or "").strip()
            if raw:
                values.append(int(raw))
    if not values:
        raise ValueError("Profiler report contains no device kernel duration")
    return values


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

    first, second = program.ordered_ops()
    if first.output not in second.inputs:
        raise ValueError("Second add must consume the first add output")
    if len(mapping.regions) != 2:
        raise ValueError("Tensix chain requires two regions")

    region_by_op = {
        op_id: region for region in mapping.regions for op_id in region.ops
    }
    if set(region_by_op) != {first.id, second.id}:
        raise ValueError("Each add op must be mapped to its own region")

    mapping_logical_core_ids = []
    for op in (first, second):
        region = region_by_op[op.id]
        if (
            region.cores != 1
            or region.placement is None
            or len(region.placement) != 1
        ):
            raise ValueError(
                "Each chain stage requires one explicitly placed core"
            )
        mapping_logical_core_ids.append(region.placement[0])
    if mapping_logical_core_ids[0] == mapping_logical_core_ids[1]:
        raise ValueError("Producer and consumer must use different cores")

    tensor_names = set(program.inputs + [op.output for op in program.ops])
    tensors = [program.tensors[name] for name in tensor_names]
    if any(t.dtype != "bfloat16" or t.shape != [32, 32] for t in tensors):
        raise ValueError("Tensix chain supports 32x32 BF16 tiles only")

    tt_metal_logical_cores = [
        logical_core_id_to_tt_metal_logical_core(architecture, core)
        for core in mapping_logical_core_ids
    ]
    return first, second, mapping_logical_core_ids, tt_metal_logical_cores


class TTMetalChainBackend(TTMetalProbeBackend):
    name = "tt-metal-tensix-chain"
    backend_version = "bf16-add-chain-v2"
    measurement_version = "tt-metal-device-kernel-duration-v2"
    compute_path = "Two-stage Tensix add chain via TT-Metal"
    probe_label = "Tensix chain probe"
    def candidate_execution_signature(self, architecture, program, mapping):
        first, second, _, tt_metal_logical_cores = check_chain_supported(
            architecture, program, mapping
        )
        return (
            self.name,
            first.id,
            second.id,
            tuple(tt_metal_logical_cores),
        )

    def run(self, architecture, program, mapping, workdir):
        workdir = workdir.resolve()
        try:
            (
                first,
                second,
                mapping_logical_core_ids,
                tt_metal_logical_cores,
            ) = check_chain_supported(architecture, program, mapping)
        except ValueError as exc:
            return self._failure(
                mapping, "UNSUPPORTED_PROGRAM", str(exc), status="unsupported"
            )

        result_path = workdir / "tensix_chain_result.json"
        kernel_root = Path(__file__).resolve().parent / "tensix_probe" / "kernels"
        custom_kernel_bundle_sha256 = kernel_bundle_sha256(kernel_root)
        command = [
            str(self.probe_binary),
            "--producer-x",
            str(tt_metal_logical_cores[0][0]),
            "--producer-y",
            str(tt_metal_logical_cores[0][1]),
            "--consumer-x",
            str(tt_metal_logical_cores[1][0]),
            "--consumer-y",
            str(tt_metal_logical_cores[1][1]),
            "--kernel-root",
            str(kernel_root),
            "--result",
            str(result_path),
        ]
        result, provenance, failure = self._run_probe(
            mapping,
            workdir,
            command,
            result_path,
            "tensix_chain",
            {
                "mapping_logical_core_ids": mapping_logical_core_ids,
                "tt_metal_logical_cores": [
                    list(coord) for coord in tt_metal_logical_cores
                ],
                "stages": [first.id, second.id],
                "custom_kernel_bundle_sha256": custom_kernel_bundle_sha256,
            },
        )
        if failure:
            return failure

        expected_tt_metal_cores = [
            list(coord) for coord in tt_metal_logical_cores
        ]
        producer_worker = result.get("producer_worker_core")
        consumer_worker = result.get("consumer_worker_core")
        if (
            not isinstance(result, dict)
            or result.get("producer_tt_metal_logical_core")
            != expected_tt_metal_cores[0]
            or result.get("consumer_tt_metal_logical_core")
            != expected_tt_metal_cores[1]
            or not isinstance(producer_worker, list)
            or len(producer_worker) != 2
            or not isinstance(consumer_worker, list)
            or len(consumer_worker) != 2
            or producer_worker == consumer_worker
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

        observed_execution = ObservedExecutionIdentity(
            kind="tt-metal-worker-core-chain-v1",
            value={
                "stages": [first.id, second.id],
                "mapping_logical_core_ids": mapping_logical_core_ids,
                "tt_metal_logical_cores": expected_tt_metal_cores,
                "worker_cores": [producer_worker, consumer_worker],
                "intermediate_transport": "noc_direct",
            },
        )

        passed = result["passed"]
        objective = None
        measurement_context = None
        metrics = unsupported_metrics("This runtime does not expose this metric")
        if self.runtime == "device":
            duration = result.get("device_kernel_duration_ns")
            if (
                result.get("measurement_source")
                != "tt_metal_device_profiler"
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or duration <= 0
            ):
                return self._failure(
                    mapping,
                    "PROBE_RESULT_MISMATCH",
                    "Device result is missing a valid TT-Metal profiler duration",
                )

            profiler_report = Path(provenance["profiler_report"])
            try:
                csv_durations = profiler_csv_durations(profiler_report)
            except (OSError, ValueError) as exc:
                return self._failure(
                    mapping,
                    "PROFILER_CROSS_CHECK_FAILED",
                    str(exc),
                )
            if set(csv_durations) != {int(duration)}:
                return self._failure(
                    mapping,
                    "PROFILER_CROSS_CHECK_FAILED",
                    "Profiler API duration does not match cpp_device_perf_report.csv",
                )
            objective = Objective(
                name="device_kernel_duration",
                value=float(duration),
                unit="ns",
                source="measured",
            )
            measurement_context = MeasurementContext(
                measurement_version=self.measurement_version,
                runtime="device",
                source="tt_metal_device_profiler",
                analysis="DEVICE KERNEL DURATION [ns]",
                implementation_revision=provenance["tt_metal_revision"],
                executable_sha256=provenance["probe_binary_sha256"],
                artifacts={
                    "custom_kernel_bundle_sha256": custom_kernel_bundle_sha256,
                },
                configuration={
                    **provenance["profiler_configuration"],
                    "cross_check": "cpp_device_perf_report.csv",
                },
            )
            metrics["latency"] = Metric(
                value=float(duration),
                unit="ns",
                status="available",
            )

        return Report(
            backend=self.name,
            backend_version=self.backend_version,
            status="ok" if passed else "error",
            mapping_hash=fingerprint(mapping),
            correctness="passed" if passed else "failed",
            objective=objective if passed else None,
            observed_execution=observed_execution,
            measurement_context=measurement_context if passed else None,
            metrics=metrics,
            message=(
                f"{first.id}@{tuple(tt_metal_logical_cores[0])} -> "
                f"{second.id}@{tuple(tt_metal_logical_cores[1])} "
                "executed with direct NoC intermediate"
                if passed
                else "Two-core Tensix chain result mismatch"
            ),
            extensions={
                "compute_path": "Two Tensix UNPACK/MATH/PACK stages via TT-Metal",
                "runtime": self.runtime,
                "measurement_source": (
                    "tt_metal_device_profiler"
                    if objective is not None
                    else "unavailable"
                ),
                "intermediate_transport": "noc_direct",
                "intermediate_returned_to_host": False,
                "stage_ops": [first.id, second.id],
                "mapping_logical_core_ids": mapping_logical_core_ids,
                "tt_metal_logical_cores": expected_tt_metal_cores,
                "worker_cores": [producer_worker, consumer_worker],
                "runtime_provenance": provenance,
                "probe_result": result,
            },
        )
