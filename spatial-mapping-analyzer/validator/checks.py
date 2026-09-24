from collections import Counter
from graphlib import CycleError

from specs.io import fingerprint


def error(code, path, message):
    return {"code": code, "path": path, "message": message}


def schema_errors(exc):
    return [error("SCHEMA_ERROR", ".".join(map(str, e["loc"])), e["msg"])
            for e in exc.errors()]


class InvalidInput(ValueError):
    def __init__(self, errors):
        super().__init__("Invalid architecture, program or run options")
        self.errors = errors


def validate_inputs(architecture, program):
    errors = []
    if architecture.available_cores > architecture.grid.rows * architecture.grid.cols:
        errors.append(error("CORE_CAPACITY", "arch.available_cores", "Exceeds grid capacity"))
    ids = [op.id for op in program.ops]
    outputs = [op.output for op in program.ops]
    if len(ids) != len(set(ids)):
        errors.append(error("DUPLICATE_OP", "program.ops", "Op IDs must be unique"))
    if len(outputs) != len(set(outputs)) or set(outputs) & set(program.inputs):
        errors.append(error("MULTIPLE_PRODUCERS", "program.ops", "Tensor must have exactly one producer"))
    if len(program.inputs) != len(set(program.inputs)) or len(program.outputs) != len(set(program.outputs)):
        errors.append(error("DUPLICATE_IO", "program", "Input/output lists must not contain duplicates"))
    referenced = set(program.inputs + program.outputs + outputs)
    referenced.update(t for op in program.ops for t in op.inputs)
    if referenced - program.tensors.keys():
        errors.append(error("UNKNOWN_TENSOR", "program.tensors", "Missing referenced tensor definitions"))
    available = set(program.inputs + outputs)
    if any(t not in available for op in program.ops for t in op.inputs) or set(program.outputs) - available:
        errors.append(error("MISSING_PRODUCER", "program", "Tensor is neither a program input nor an op output"))
    if errors:
        return errors
    producers = {op.output: op.id for op in program.ops}
    expected = Counter((producers[t], op.id, t, i)
                       for op in program.ops for i, t in enumerate(op.inputs) if t in producers)
    actual = Counter((e.source, e.target, e.tensor, e.input_index) for e in program.edges)
    if actual != expected:
        errors.append(error("EDGE_MISMATCH", "program.edges", "Edges must exactly match tensor producer/consumer ports"))
    try:
        program.ordered_ops()
    except CycleError:
        errors.append(error("PROGRAM_CYCLE", "program.ops", "Program must be a DAG"))
    for op in program.ops:
        tensors = [program.tensors[t] for t in op.inputs]
        out = program.tensors[op.output]
        shapes = [t.shape for t in tensors]
        valid = False
        if op.op == "matmul" and len(shapes) == 2 and all(len(s) == 2 for s in shapes):
            a, b = shapes
            valid = a[1] == b[0] and out.shape == [a[0], b[1]]
        elif op.op == "relu":
            valid = len(shapes) == 1 and shapes[0] == out.shape
        elif op.op == "add":
            valid = len(shapes) == 2 and shapes[0] == shapes[1] == out.shape
        if not valid or any(t.dtype != out.dtype for t in tensors):
            errors.append(error("OP_SIGNATURE", f"program.ops.{op.id}", "Unsupported arity, shape or dtype; add does not broadcast"))
    return errors


def validate_mapping(architecture, program, mapping):
    errors = []
    if (mapping.program_id != program.id or mapping.program_hash != fingerprint(program)
            or mapping.architecture_hash != fingerprint(architecture)):
        errors.append(error("INPUT_MISMATCH", "mapping", "Mapping does not reference these exact inputs"))
    expected = Counter(op.id for op in program.ops)
    actual = Counter(op for r in mapping.regions for op in r.ops)
    if expected != actual:
        errors.append(error("OP_COVERAGE", "mapping.regions", "Every program op must appear exactly once; unknown ops are forbidden"))
    names = [r.id for r in mapping.regions]
    if len(names) != len(set(names)):
        errors.append(error("DUPLICATE_REGION", "mapping.regions", "Region IDs must be unique"))
    if sum(r.cores for r in mapping.regions) > architecture.available_cores:
        errors.append(error("CORE_CAPACITY", "mapping.regions", "Exclusive core allocations exceed available cores"))
    # Fusion remains a declared extension point, never silently accepted as implemented.
    if any(r.fusions for r in mapping.regions):
        errors.append(error("UNSUPPORTED_FUSION", "mapping.regions", "Passthrough scaffold has no fusion lowering"))
    if expected != actual or len(names) != len(set(names)):
        return errors
    positions = {op: (i, j) for i, r in enumerate(mapping.regions) for j, op in enumerate(r.ops)}
    for edge in program.edges:
        if positions[edge.source] >= positions[edge.target]:
            errors.append(error("DEPENDENCY_ORDER", "mapping.regions", f"{edge.source} must precede {edge.target}; regions are ordered tensor barriers"))
    return errors
