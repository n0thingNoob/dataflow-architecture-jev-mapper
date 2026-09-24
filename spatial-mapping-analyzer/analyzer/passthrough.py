from mapping_ir.models import Mapping, Region
from specs.io import fingerprint


class PassthroughAnalyzer:
    """Identity policy: one op per region, one exclusive core per region.

    Feedback is deliberately ignored until a search/learned policy is added.
    Repeated trials produce identical mappings; no optimization is claimed.
    """

    def __init__(self, execution_policy="exclusive_cores_tensor_barrier"):
        self.execution_policy = execution_policy

    def propose_mapping(self, architecture, program, feedback=None):
        return Mapping(
            program_id=program.id,
            program_hash=fingerprint(program),
            architecture_hash=fingerprint(architecture),
            execution_policy=self.execution_policy,
            regions=[Region(id=f"region_{i}", ops=[op.id], cores=1)
                     for i, op in enumerate(program.ordered_ops())],
        )
