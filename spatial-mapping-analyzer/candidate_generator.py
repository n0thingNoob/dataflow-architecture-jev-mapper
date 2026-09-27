"""Deterministic, bounded mapping candidates supported by the current TT-Sim path."""

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
            op_id for op_id in remaining
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


def generate_candidates(architecture, program, limit=16):
    """Generate distinct mappings that the current BRISC backend can actually execute.

    The current backend supports one op per region and one core per op. Candidate
    diversity therefore comes from legal topological order, execution policy and
    explicit logical-core placement. Fusion and multi-core regions stay out of this
    layer until the backend can execute them faithfully.
    """
    if limit < 1:
        raise ValueError("Candidate limit must be positive")
    if len(program.ops) > architecture.available_cores:
        raise ValueError("Current candidate generator requires one available core per op")

    orders = _topological_orders(program, limit)
    if not orders:
        raise ValueError("Program has no legal topological order")

    candidates = []
    seen = set()
    for placement in _placement_rotations(architecture.available_cores, len(program.ops)):
        for order in orders:
            for policy in POLICIES:
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
                            placement=[core_id],
                        )
                        for op_id, core_id in zip(order, placement)
                    ],
                )
                key = fingerprint(mapping)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(mapping)
                if len(candidates) >= limit:
                    return candidates
    return candidates
