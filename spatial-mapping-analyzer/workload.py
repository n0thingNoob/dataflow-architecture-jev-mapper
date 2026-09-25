"""Bounded int32 workload lowering and an independent Python reference."""
import hashlib
import math
import random

from kernels import DONE, MAX_ELEMENTS, compile_kernel
from specs import fingerprint, write_json
from validator import validate_inputs, validate_mapping

# Physical Wormhole Tensix coordinates exposed by this correctness harness.
# Column 5 is a DRAM column and is intentionally skipped.
CORES = [(x, 1) for x in (1, 2, 3, 4, 6, 7, 8, 9)]


def check_supported(architecture, program, mapping):
    errors = validate_inputs(architecture, program) or validate_mapping(architecture, program, mapping)
    if errors:
        raise ValueError(str(errors))
    if architecture.extensions.get("target_family") != "wormhole":
        raise ValueError("Program backend requires Wormhole")
    if architecture.available_cores > len(CORES):
        raise ValueError(
            f"BRISC correctness backend exposes at most {len(CORES)} logical cores"
        )
    if len(mapping.regions) > len(CORES) or any(len(r.ops) != 1 or r.cores != 1 for r in mapping.regions):
        raise ValueError("Program backend supports at most eight regions, one op and one core per region")
    placed = [core for region in mapping.regions for core in (region.placement or [])]
    if any(core >= len(CORES) for core in placed):
        raise ValueError("Mapping placement exceeds BRISC correctness backend core capacity")
    if any(t.dtype != "int32" or len(t.shape) > 2 or any(d > 32 for d in t.shape)
           or math.prod(t.shape) > MAX_ELEMENTS for t in program.tensors.values()):
        raise ValueError("Program backend supports int32 scalars/vectors/matrices with dimensions <= 32")
    if architecture.local_memory_bytes_per_core is not None and architecture.local_memory_bytes_per_core < DONE + 4:
        raise ValueError(f"Program backend requires at least {DONE + 4} bytes per core")


def input_values(program, supplied=None, seed=0):
    rng = random.Random(seed)
    values = supplied if supplied is not None else {
        name: [rng.randint(-2, 2) for _ in range(math.prod(program.tensors[name].shape))]
        for name in program.inputs}
    if not isinstance(values, dict) or set(values) != set(program.inputs):
        raise ValueError("Input fixture must contain exactly the program input names")
    for name, data in values.items():
        if (not isinstance(data, list) or len(data) != math.prod(program.tensors[name].shape)
                or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in data)):
            raise ValueError(f"Input {name} must be a flat int32 array matching its shape ([] has one value)")
    return values


def int32(value):
    return (value + 2**31) % 2**32 - 2**31


def reference(program, inputs):
    values = {name: list(data) for name, data in inputs.items()}
    for op in program.ordered_ops():
        a = values[op.inputs[0]]
        if op.op == "relu":
            output = [max(0, v) for v in a]
        elif op.op == "add":
            output = [int32(x + y) for x, y in zip(a, values[op.inputs[1]])]
        else:
            b = values[op.inputs[1]]
            m, k = program.tensors[op.inputs[0]].shape
            _, n = program.tensors[op.inputs[1]].shape
            output = [int32(sum(a[i*k+t] * b[t*n+j] for t in range(k)))
                      for i in range(m) for j in range(n)]
        values[op.output] = output
    return {op.output: values[op.output] for op in program.ops}


def lower(program, mapping, inputs, directory):
    operations = []
    nodes = {op.id: op for op in program.ops}
    for index, region in enumerate(mapping.regions):
        op = nodes[region.ops[0]]
        count = math.prod(program.tensors[op.output].shape)
        firmware = compile_kernel(op.op, [program.tensors[t].shape for t in op.inputs], count)
        filename = f"op_{index:04d}.bin"
        (directory / filename).write_bytes(firmware)
        logical_core = region.placement[0] if region.placement else index
        operations.append({"id": op.id, "region": region.id, "core": CORES[logical_core],
                           "inputs": op.inputs, "output": op.output, "elements": count,
                           "firmware": filename, "firmware_sha256": hashlib.sha256(firmware).hexdigest()})
    manifest = {"schema_version": "0.1", "mapping_hash": fingerprint(mapping),
                "program_hash": fingerprint(program), "execution_policy": mapping.execution_policy,
                "operations": operations, "inputs": inputs, "outputs": program.outputs,
                "max_clock_steps": 1_000_000}
    write_json(directory / "program_execution.json", manifest)
    return manifest
