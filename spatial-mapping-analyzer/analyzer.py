from candidate_generator import generate_candidates
from mapping_ir import Mapping, Region
from specs import fingerprint


class PassthroughAnalyzer:
    """Identity policy: one op per region, one exclusive core per region."""

    def __init__(self, parallel=False):
        self.execution_policy = ("exclusive_cores_dependency_barrier" if parallel
                                 else "exclusive_cores_tensor_barrier")

    def propose_mapping(self, architecture, program, feedback=None):
        return Mapping(
            program_id=program.id,
            program_hash=fingerprint(program),
            architecture_hash=fingerprint(architecture),
            execution_policy=self.execution_policy,
            regions=[Region(id=f"region_{i}", ops=[op.id], cores=1)
                     for i, op in enumerate(program.ordered_ops())],
        )


class EnumeratingAnalyzer:
    """Walk a deterministic list of executable mapping candidates.

    This is intentionally not an optimizer. It exists to exercise multiple real
    mappings and collect simulator feedback before a learned policy is introduced.
    """

    def __init__(self, candidate_limit=16):
        if candidate_limit < 1:
            raise ValueError("Candidate limit must be positive")
        self.candidate_limit = candidate_limit

    def propose_mapping(self, architecture, program, feedback=None):
        feedback = feedback or []
        candidates = generate_candidates(
            architecture, program, limit=self.candidate_limit
        )
        return candidates[len(feedback) % len(candidates)]
