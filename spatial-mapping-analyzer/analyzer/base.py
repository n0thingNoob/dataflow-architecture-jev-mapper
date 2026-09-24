from typing import Protocol

from mapping_ir.models import Mapping
from specs.models import Architecture, Program


class Analyzer(Protocol):
    def propose_mapping(
        self, architecture: Architecture, program: Program,
        feedback: list[dict] | None = None,
    ) -> Mapping: ...
