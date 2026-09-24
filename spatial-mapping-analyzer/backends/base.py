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


class Backend(Protocol):
    name: str

    def run(self, architecture: Architecture, program: Program, mapping: Mapping,
            workdir: Path) -> Report: ...
