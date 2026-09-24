from backends.base import Report
from specs.io import fingerprint


class TTSimBackend:
    name = "tt-sim"

    def run(self, architecture, program, mapping, workdir):
        return Report(
            backend=self.name, backend_version="0.1", status="unsupported",
            mapping_hash=fingerprint(mapping),
            message="TT-Sim lowering/runner is not implemented. No process was launched; no mock fallback occurred.",
        )
