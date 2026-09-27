"""Shared contracts between mapping search, execution backends, and reports."""
import json
from dataclasses import dataclass
from typing import Any, Hashable, Protocol

from pydantic import field_validator

from specs import Contract, Identifier


@dataclass(frozen=True)
class BackendCapabilities:
    """Mapping dimensions that can change execution for a backend."""

    topological_order: bool = True
    execution_policy: bool = True
    placement: bool = True


class BackendProtocol(Protocol):
    name: str
    backend_version: str
    search_capabilities: BackendCapabilities

    def candidate_execution_signature(
        self, architecture, program, mapping
    ) -> Hashable:
        ...

    def run(self, architecture, program, mapping, workdir):
        ...


class ObservedExecutionIdentity(Contract):
    """Backend-observed execution identity after lowering/runtime mapping."""

    kind: Identifier
    value: Any

    @field_validator("value")
    @classmethod
    def json_safe_value(cls, value):
        if value is None:
            raise ValueError("Observed execution identity is required")
        json.dumps(value, sort_keys=True, allow_nan=False)
        return value


def json_safe_signature(value):
    """Return a canonical JSON-compatible representation of a signature."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
