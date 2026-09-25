from typing import Annotated, Literal

from pydantic import Field

from specs import Contract, Identifier, PositiveInt, Versioned

LogicalCoreId = Annotated[int, Field(strict=True, ge=0)]


class Fusion(Contract):
    kind: Literal["matmul_relu"]
    ops: Annotated[list[Identifier], Field(min_length=2, max_length=2)]


class Region(Contract):
    id: Identifier
    ops: Annotated[list[Identifier], Field(min_length=1)]
    cores: PositiveInt
    placement: list[LogicalCoreId] | None = None
    fusions: list[Fusion] = Field(default_factory=list)


class Mapping(Versioned):
    program_id: Identifier
    program_hash: Identifier
    architecture_hash: Identifier
    execution_policy: Literal["exclusive_cores_tensor_barrier", "exclusive_cores_dependency_barrier"] = "exclusive_cores_tensor_barrier"
    regions: Annotated[list[Region], Field(min_length=1)]
