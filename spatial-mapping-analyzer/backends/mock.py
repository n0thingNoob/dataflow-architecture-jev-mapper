from backends.base import Metric, Objective, Report
from specs.io import fingerprint


class MockBackend:
    name = "mock"

    def run(self, architecture, program, mapping, workdir):
        return Report(
            backend=self.name, backend_version="0.1", status="ok",
            mapping_hash=fingerprint(mapping),
            objective=Objective(name="mock_cost", value=1, unit="arbitrary", source="synthetic"),
            metrics={name: Metric(unit=unit, reason="Mock backend does not execute kernels or model performance")
                     for name, unit in [("total_cycles", "cycles"), ("latency", "ns"),
                                        ("core_utilization", "fraction"), ("stall_cycles", "cycles"),
                                        ("communication", "bytes"), ("buffer_usage", "bytes")]},
            message="Constant passthrough score. No numerical execution or performance prediction.",
        )
