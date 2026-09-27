"""Validated training records derived only from measured successful trials."""
import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from mapping_ir import Mapping
from specs import Architecture, Contract, Identifier, Program, Versioned, fingerprint


def content_hash(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ExecutionIdentity(Contract):
    backend: Identifier
    backend_version: Identifier
    signature: Any

    @field_validator("signature")
    @classmethod
    def json_safe_signature(cls, value):
        if value is None:
            raise ValueError("Execution signature is required")
        json.dumps(value, sort_keys=True, allow_nan=False)
        return value


class MeasuredObjective(Contract):
    name: Identifier
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    unit: Identifier
    source: Literal["measured"] = "measured"


class MeasuredMappingRecord(Versioned):
    sample_id: Identifier
    run_id: Identifier
    trial_id: Identifier
    architecture: Architecture
    program: Program
    mapping: Mapping
    mapping_hash: Identifier
    execution: ExecutionIdentity
    measurement: MeasuredObjective

    @model_validator(mode="after")
    def consistent_identity(self):
        if self.mapping.program_hash != fingerprint(self.program):
            raise ValueError("Mapping program hash does not match dataset program")
        if self.mapping.architecture_hash != fingerprint(self.architecture):
            raise ValueError("Mapping architecture hash does not match dataset architecture")
        if self.mapping_hash != fingerprint(self.mapping):
            raise ValueError("Dataset mapping hash does not match mapping")
        return self


def make_sample_id(
    run_id,
    trial_id,
    mapping_hash,
    execution,
    measurement,
):
    return "sample_" + content_hash(
        {
            "run_id": run_id,
            "trial_id": trial_id,
            "mapping_hash": mapping_hash,
            "execution": execution,
            "measurement": measurement,
        }
    )
