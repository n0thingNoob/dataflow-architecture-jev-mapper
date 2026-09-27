"""Deterministic, bounded mapping candidates supported by a backend."""
from backend_contract import BackendCapabilities
from mapping_ir import Mapping, Region
from specs import fingerprint

POLICIES = (
    "exclusive_cores_tensor_barrier",
    "exclusive_cores_dependency_barrier",
)


def _topological_orders(program, limit):
    """Return up to limit deterministic topological orders without exploring unboundedly."""
    predecessors = {op.id: set() for op in program.ops}
    for edge in program.edges:
        predecessors[edge.target].add(edge.source)

    orders = []

    def visit(prefix, remaining):
        if len(orders) >= limit:
            return
        if not remaining:
            orders.append(tuple(prefix))
            return
        ready = sorted(
            op_id
            for op_id in remaining
            if all(parent not in remaining for parent in predecessors[op_id])
        )
        for op_id in ready:
            visit(prefix + [op_id], remaining - {op_id})
            if len(orders) >= limit:
                return

    visit([], {op.id for op in program.ops})
    return orders


def _placement_rotations(available_cores, op_count):
    """Generate simple one-op/one-core placements over logical core IDs."""
    for offset in range(available_cores):
        yield tuple((offset + index) % available_cores for index in range(op_count))


def generate_candidates(
    architecture,
    program,
    limit=16,
    capabilities=None,
    candidate_key=None,
):
    """Generate distinct mappings in dimensions the selected backend can execute."""
    if limit < 1:
        raise ValueError("Candidate limit must be positive")
    if len(program.ops) > architecture.available_cores:
        raise ValueError("Current candidate generator requires one available core per op")

    capabilities = capabilities or BackendCapabilities()
    orders = (
        _topological_orders(program, limit)
        if capabilities.topological_order
        else [tuple(op.id for op in program.ordered_ops())]
    )
    if not orders:
        raise ValueError("Program has no legal topological order")

    placements = (
        _placement_rotations(architecture.available_cores, len(program.ops))
        if capabilities.placement
        else [tuple(range(len(program.ops)))]
    )
    policies = POLICIES if capabilities.execution_policy else POLICIES[:1]

    candidates = []
    seen = set()
    canonical_ops = tuple(op.id for op in program.ordered_ops())
    for placement in placements:
        placement_by_op = dict(zip(canonical_ops, placement))
        for order in orders:
            for policy in policies:
                mapping = Mapping(
                    program_id=program.id,
                    program_hash=fingerprint(program),
                    architecture_hash=fingerprint(architecture),
                    execution_policy=policy,
                    regions=[
                        Region(
                            id=f"region_{op_id}",
                            ops=[op_id],
                            cores=1,
                            placement=[placement_by_op[op_id]],
                        )
                        for op_id in order
                    ],
                )
                key = (
                    candidate_key(architecture, program, mapping)
                    if candidate_key is not None
                    else fingerprint(mapping)
                )
                try:
                    duplicate = key in seen
                except TypeError as exc:
                    raise ValueError(
                        "Candidate execution signature must be hashable"
                    ) from exc
                if duplicate:
                    continue
                seen.add(key)
                candidates.append(mapping)
                if len(candidates) >= limit:
                    return candidates
    return candidates
