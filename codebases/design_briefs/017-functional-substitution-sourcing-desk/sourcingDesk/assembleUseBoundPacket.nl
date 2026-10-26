---
args: { request: SubstituteEvaluationRequest, candidates: StageResultSubstituteCandidate, findings: StageResultCompatibilityFinding, qualification: StageResultQualificationTask, approval: QualificationGuardrail, adapter: QualificationGuardrail }
returns: StageResultSubstitutionPacket
---
Package only candidates that passed every essential property and have current scoped technical and commercial approval. Bind approval to one machine/use, candidate revision, environment, adapter, verified total cost, lead time, and production-lot rule. Explicitly state exclusions so approval cannot propagate to a different machine or revision.
