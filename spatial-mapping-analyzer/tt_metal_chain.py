"""TT-Metal backend for a two-core producer-consumer Tensix chain."""
from pathlib import Path

from candidate_generator import MappingSearchCapabilities
from report import Report, unsupported_metrics
from specs import fingerprint
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

    first, second = program.ordered_ops()
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

    coords = [logical_core_to_coord(architecture, core) for core in placements]
    return first, second, placements, coords


class TTMetalChainBackend(TTMetalProbeBackend):
    name = "tt-metal-tensix-chain"
    backend_version = "bf16-add-chain-v1"
    compute_path = "Two-stage Tensix add chain via TT-Metal"
    probe_label = "Tensix chain probe"
    search_capabilities = MappingSearchCapabilities(
        topological_order=False,
        execution_policy=False,
        placement=True,
    )

    def execution_signature(self, architecture, program, mapping):
        first, second, _, coords = check_chain_supported(
            architecture, program, mapping
        )
        return self.name, first.id, second.id, tuple(coords)

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
        result, failure = self._run_probe(
            mapping,
            workdir,
            command,
            result_path,
            "tensix_chain",
            {
                "logical_cores": logical_cores,
                "physical_cores": [list(coord) for coord in coords],
                "stages": [first.id, second.id],
            },
        )
        if failure:
            return failure

        if (
            not isinstance(result, dict)
            or result.get("producer_core") != list(coords[0])
            or result.get("consumer_core") != list(coords[1])
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
            backend_version=self.backend_version,
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
