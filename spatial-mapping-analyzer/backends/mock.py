from backends.base import Objective, Report, unsupported_metrics
from specs.io import fingerprint


class MockBackend:
    name = "mock"

    def run(self, architecture, program, mapping, workdir):
        return Report(
            backend=self.name, backend_version="0.1", status="ok",
            mapping_hash=fingerprint(mapping),
            objective=Objective(name="mock_cost", value=1, unit="arbitrary", source="synthetic"),
            metrics=unsupported_metrics("Mock backend does not execute kernels or model performance"),
            message="Constant passthrough score. No numerical execution or performance prediction.",
        )
