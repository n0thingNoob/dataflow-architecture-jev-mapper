"""Deterministic topology-aware scoring for executable mapping candidates."""
from collections import defaultdict
from math import prod

DTYPE_BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "int32": 4,
}
SUPPORTED_LOGICAL_MESH_TOPOLOGIES = {"mesh", "wormhole-noc"}


def _validate_architecture(architecture):
    if architecture.network_topology not in SUPPORTED_LOGICAL_MESH_TOPOLOGIES:
        raise ValueError(
            "Topology-aware heuristic requires a supported logical mesh network"
        )
    grid_capacity = architecture.grid.rows * architecture.grid.cols
    if architecture.available_cores > grid_capacity:
        raise ValueError("Architecture exposes more cores than its logical grid")


def _tensor_bytes(tensor):
    return prod(tensor.shape or [1]) * DTYPE_BYTES[tensor.dtype]


def _core_coord(architecture, core):
    if core < 0 or core >= architecture.available_cores:
        raise ValueError("Mapping placement is outside available logical cores")
    x = core % architecture.grid.cols
    y = core // architecture.grid.cols
    if y >= architecture.grid.rows:
        raise ValueError("Mapping placement is outside the logical grid")
    return x, y


def _xy_route(architecture, source, target):
    """Return directed logical links for deterministic X-then-Y mesh routing."""
    x, y = _core_coord(architecture, source)
    target_x, target_y = _core_coord(architecture, target)
    links = []

    while x != target_x:
        next_x = x + (1 if target_x > x else -1)
        links.append(((x, y), (next_x, y)))
        x = next_x
    while y != target_y:
        next_y = y + (1 if target_y > y else -1)
        links.append(((x, y), (x, next_y)))
        y = next_y
    return links


def _core_by_op(mapping):
    result = {}
    for region in mapping.regions:
        if region.placement is None or len(region.placement) != 1:
            raise ValueError(
                "Heuristic scorer requires one explicit core per region"
            )
        for op_id in region.ops:
            if op_id in result:
                raise ValueError("Operation appears in more than one region")
            result[op_id] = region.placement[0]
    return result


class HeuristicScorer:
    """Topology-aware communication baseline.

    The ranking score is dimensionless and lower is better. It combines only
    placement-sensitive structural terms:
    - critical-path tensor-weighted hop cost,
    - multicast-aware total network byte-hops,
    - peak directed-link load.

    Peak endpoint traffic remains available as a diagnostic, but is deliberately
    excluded from the score because it is often placement-invariant for the
    current injective one-op-per-core search space.

    If architecture bandwidth and link latency are both available, an additional
    cycle-like critical-path estimate is reported for diagnostics. It does not
    affect ranking.
    """

    name = "heuristic-topology-v2"

    def score_breakdown(self, architecture, program, mapping):
        _validate_architecture(architecture)
        core_by_op = _core_by_op(mapping)
        bandwidth = architecture.bandwidth_bytes_per_cycle
        link_latency = architecture.link_latency_cycles
        use_cycle_estimate = bandwidth is not None and link_latency is not None

        edge_data = []
        total_edge_bytes = 0
        for edge in program.edges:
            try:
                source_core = core_by_op[edge.source]
                target_core = core_by_op[edge.target]
                tensor = program.tensors[edge.tensor]
            except KeyError as exc:
                raise ValueError(
                    "Heuristic scorer cannot resolve program edge placement"
                ) from exc

            size_bytes = _tensor_bytes(tensor)
            links = _xy_route(architecture, source_core, target_core)
            edge_data.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "tensor": edge.tensor,
                    "bytes": size_bytes,
                    "hops": len(links),
                    "links": links,
                }
            )
            total_edge_bytes += size_bytes

        if not edge_data:
            return {
                "score": 0.0,
                "critical_path_weighted_hops": 0.0,
                "network_byte_hops": 0.0,
                "max_link_load_ratio": 0.0,
                "max_core_traffic_ratio": 0.0,
                "estimated_critical_path_cycles": (
                    0.0 if use_cycle_estimate else None
                ),
            }

        normalizer = float(total_edge_bytes)
        incoming = defaultdict(list)
        link_load = defaultdict(int)
        core_traffic = defaultdict(int)
        multicast_links = set()
        multicast_injections = set()

        for item in edge_data:
            weighted_hops = item["bytes"] * item["hops"] / normalizer
            incoming[item["target"]].append(
                (item["source"], weighted_hops, item)
            )

            source_core = core_by_op[item["source"]]
            target_core = core_by_op[item["target"]]
            injection_key = (item["source"], item["tensor"])
            if (
                architecture.multicast is not True
                or injection_key not in multicast_injections
            ):
                core_traffic[source_core] += item["bytes"]
                multicast_injections.add(injection_key)
            core_traffic[target_core] += item["bytes"]

            for link in item["links"]:
                multicast_key = (item["source"], item["tensor"], link)
                if (
                    architecture.multicast is True
                    and multicast_key in multicast_links
                ):
                    continue
                link_load[link] += item["bytes"]
                multicast_links.add(multicast_key)

        path_cost = {}
        estimated_path_cycles = {}
        for op in program.ordered_ops():
            choices = incoming.get(op.id, [])
            path_cost[op.id] = max(
                (
                    path_cost.get(source, 0.0) + edge_cost
                    for source, edge_cost, _ in choices
                ),
                default=0.0,
            )
            if use_cycle_estimate:
                estimated_path_cycles[op.id] = max(
                    (
                        estimated_path_cycles.get(source, 0.0)
                        + item["hops"] * link_latency
                        + item["bytes"] / bandwidth
                        for source, _, item in choices
                    ),
                    default=0.0,
                )

        critical_path = max(path_cost.values(), default=0.0)
        network_byte_hops = sum(link_load.values()) / normalizer
        max_link_load_ratio = max(link_load.values(), default=0) / normalizer
        max_core_traffic_ratio = max(
            core_traffic.values(), default=0
        ) / normalizer
        score = critical_path + network_byte_hops + max_link_load_ratio

        return {
            "score": float(score),
            "critical_path_weighted_hops": float(critical_path),
            "network_byte_hops": float(network_byte_hops),
            "max_link_load_ratio": float(max_link_load_ratio),
            "max_core_traffic_ratio": float(max_core_traffic_ratio),
            "estimated_critical_path_cycles": (
                float(max(estimated_path_cycles.values(), default=0.0))
                if use_cycle_estimate
                else None
            ),
        }

    def score_candidates(self, architecture, program, candidates):
        return [
            self.score_breakdown(architecture, program, mapping)["score"]
            for mapping in candidates
        ]
