from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from backend_contract import ObservedExecutionIdentity
from specs import Contract, Identifier, Versioned


class Objective(Contract):
    name: Identifier
    value: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    unit: Identifier
    source: Literal["synthetic", "measured", "estimated"]


class MeasurementContext(Contract):
    measurement_version: Identifier
    runtime: Identifier
    source: Identifier
    analysis: Identifier
    implementation_revision: Identifier
    executable_sha256: Identifier
    artifacts: dict[str, Identifier] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)


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
    observed_execution: ObservedExecutionIdentity | None = None
    measurement_context: MeasurementContext | None = None
    metrics: dict[str, Metric] = Field(default_factory=dict)
    message: str | None = None

    @model_validator(mode="after")
    def consistent_objective(self):
        if self.objective is not None and (
            self.status != "ok" or self.correctness == "failed"
        ):
            raise ValueError(
                "Failed, skipped or unsupported reports cannot carry a ranking objective"
            )
        if self.objective is not None and self.objective.source == "measured":
            if self.observed_execution is None or self.measurement_context is None:
                raise ValueError(
                    "Measured objectives require observed execution and measurement context"
                )
        elif self.measurement_context is not None:
            raise ValueError(
                "Measurement context is only valid with a measured objective"
            )
        return self

    def objective_key(self) -> tuple | None:
        """Only objectives with the same execution/measurement definition may compare."""
        if self.objective is None:
            return None
        cost = self.objective
        key = (
            self.backend,
            self.backend_version,
            cost.name,
            cost.unit,
            cost.source,
        )
        if cost.source == "measured":
            return (*key, self.measurement_context.measurement_version)
        return key

