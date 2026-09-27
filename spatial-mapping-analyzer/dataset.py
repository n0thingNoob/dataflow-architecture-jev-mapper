"""Validated training records derived only from measured successful trials."""
import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from backend_contract import ObservedExecutionIdentity
from mapping_ir import Mapping
from report import MeasurementContext
from specs import Architecture, Contract, Identifier, Program, Versioned, fingerprint


def content_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class RequestedExecutionIdentity(Contract):
    backend: Identifier
    backend_version: Identifier
    signature: Any

    @field_validator("signature")
    @classmethod
    def json_safe_signature(cls, value):
        if value is None:
            raise ValueError("Requested execution signature is required")
        json.dumps(value, sort_keys=True, allow_nan=False)
        return value


class MeasuredObjective(Contract):
    name: Identifier
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    unit: Identifier
    source: Literal["measured"] = "measured"


def make_observation_id(run_id, trial_id):
    return "observation_" + content_hash(
        {"run_id": run_id, "trial_id": trial_id}
    )


def make_program_group_id(program):
    return "program_" + fingerprint(program)


def make_execution_group_id(
    program,
    architecture,
    requested_execution,
    observed_execution,
):
    return "execution_" + content_hash(
        {
            "program_hash": fingerprint(program),
            "architecture_hash": fingerprint(architecture),
            "requested_execution": requested_execution,
            "observed_execution": observed_execution,
        }
    )


class MeasuredMappingRecord(Versioned):
    observation_id: Identifier
    content_hash: Identifier
    run_id: Identifier
    trial_id: Identifier
    program_group_id: Identifier
    execution_group_id: Identifier
    architecture: Architecture
    program: Program
    mapping: Mapping
    mapping_hash: Identifier
    requested_execution: RequestedExecutionIdentity
    observed_execution: ObservedExecutionIdentity
    measurement_context: MeasurementContext
    measurement: MeasuredObjective

    def content_payload(self):
        return self.model_dump(exclude={"content_hash"})

    @model_validator(mode="after")
    def consistent_identity(self):
        if self.mapping.program_hash != fingerprint(self.program):
            raise ValueError("Mapping program hash does not match dataset program")
        if self.mapping.architecture_hash != fingerprint(self.architecture):
            raise ValueError(
                "Mapping architecture hash does not match dataset architecture"
            )
        if self.mapping_hash != fingerprint(self.mapping):
            raise ValueError("Dataset mapping hash does not match mapping")
        if self.observation_id != make_observation_id(
            self.run_id, self.trial_id
        ):
            raise ValueError("Observation ID does not match run/trial identity")
        if self.program_group_id != make_program_group_id(self.program):
            raise ValueError("Program group ID does not match program")
        expected_execution_group = make_execution_group_id(
            self.program,
            self.architecture,
            self.requested_execution.model_dump(),
            self.observed_execution.model_dump(),
        )
        if self.execution_group_id != expected_execution_group:
            raise ValueError(
                "Execution group ID does not match effective execution"
            )
        expected_content_hash = "content_" + content_hash(
            self.content_payload()
        )
        if self.content_hash != expected_content_hash:
            raise ValueError("Content hash does not match record contents")
        return self
