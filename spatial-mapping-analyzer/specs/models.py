from graphlib import TopologicalSorter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Versioned(Contract):
    schema_version: Literal["0.1"] = "0.1"
    extensions: dict[str, Any] = Field(default_factory=dict)


class Grid(Contract):
    rows: PositiveInt
    cols: PositiveInt


class Architecture(Versioned):
    id: Identifier
    grid: Grid
    available_cores: PositiveInt
    local_memory_bytes_per_core: PositiveInt | None = None
    compute_units: list[Identifier] = Field(default_factory=list)
    network_topology: Identifier = "unspecified"
    multicast: bool | None = None
    bandwidth_bytes_per_cycle: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    link_latency_cycles: PositiveInt | None = None


class Tensor(Contract):
    shape: list[PositiveInt]  # [] is a scalar; no implicit [1] conversion.
    dtype: Literal["float32", "bfloat16", "int32"]


class Op(Contract):
    id: Identifier
    op: Literal["matmul", "relu", "add"]
    inputs: Annotated[list[Identifier], Field(min_length=1)]
    output: Identifier


class Edge(Contract):
    source: Identifier
    target: Identifier
    tensor: Identifier
    input_index: Annotated[int, Field(strict=True, ge=0)]


class Program(Versioned):
    id: Identifier
    tensors: dict[Identifier, Tensor]
    inputs: Annotated[list[Identifier], Field(min_length=1)]
    outputs: Annotated[list[Identifier], Field(min_length=1)]
    ops: Annotated[list[Op], Field(min_length=1)]
    edges: list[Edge]

    def ordered_ops(self) -> list[Op]:
        """Stable dependency order; raises CycleError for a cyclic program."""
        producers = {op.output: op.id for op in self.ops}
        dependencies = {op.id: list(dict.fromkeys(producers[t] for t in op.inputs if t in producers))
                        for op in self.ops}
        nodes = {op.id: op for op in self.ops}
        return [nodes[name] for name in TopologicalSorter(dependencies).static_order()]
