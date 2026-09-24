from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from mapping_ir.models import Mapping
from specs.models import Architecture, Contract, Identifier, Program, Versioned


class Objective(Contract):
    name: Identifier
    value: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    unit: Identifier
    source: Literal["synthetic", "measured", "estimated"]


class Metric(Contract):
    value: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    unit: Identifier
    status: Literal["unsupported", "available"] = "unsupported"
    reason: str | None = None

    @model_validator(mode="after")
    def consistent_availability(self):
        if (self.status == "available") != (self.value is not None):
            raise ValueError("Available metrics require a value; unsupported metrics require null")
        return self


def unsupported_metrics(reason: str) -> dict[str, Metric]:
    units = {"total_cycles": "cycles", "latency": "ns", "core_utilization": "fraction",
             "stall_cycles": "cycles", "communication": "bytes", "buffer_usage": "bytes"}
    return {name: Metric(unit=unit, reason=reason) for name, unit in units.items()}


class Report(Versioned):
    backend: Identifier
    backend_version: Identifier
    status: Literal["ok", "unsupported", "error", "skipped"]
    mapping_hash: Identifier
    correctness: Literal["not_checked", "passed", "failed"] = "not_checked"
    objective: Objective | None = None
    metrics: dict[str, Metric] = Field(default_factory=dict)
    message: str | None = None

    @model_validator(mode="after")
    def consistent_objective(self):
        if self.objective is not None and (self.status != "ok" or self.correctness == "failed"):
            raise ValueError("Failed, skipped or unsupported reports cannot carry a ranking objective")
        return self

    def objective_key(self) -> tuple | None:
        """Only objectives with the same identity and units may be compared."""
        if self.objective is None:
            return None
        cost = self.objective
        return self.backend, self.backend_version, cost.name, cost.unit, cost.source


class Backend(Protocol):
    name: str

    def run(self, architecture: Architecture, program: Program, mapping: Mapping,
            workdir: Path) -> Report: ...
