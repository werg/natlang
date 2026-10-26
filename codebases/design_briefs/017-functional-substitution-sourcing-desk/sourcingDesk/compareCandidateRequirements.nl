---
args: { request: SubstituteEvaluationRequest, requirements: StageResultFunctionalRequirement, candidates: StageResultSubstituteCandidate, properties: CandidatePropertyEvidence[], conversion: QualificationGuardrail }
returns: StageResultCompatibilityFinding
---
For each candidate compare every essential requirement against documented evidence for the same revision. Bind matrix = scoreInteractingCandidateProperties(requirements.accepted, candidates.accepted, properties, conversion). One critical failure rejects that candidate. An unknown critical property stays unknown; do not average a fail against other passing properties.
