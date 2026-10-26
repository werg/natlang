---
args: { request: SubstituteEvaluationRequest, requirements: StageResultFunctionalRequirement, evidence: EvidencePage[] }
returns: StageResultSubstituteCandidate
---
Search the supplied supplier evidence for alternatives that could serve the confirmed function, including a different mechanism only when constraints permit. Bind every candidate to a supplier item and revision. Record the source property evidence for each candidate; do not copy a specification from a prior revision or invent catalog values.
