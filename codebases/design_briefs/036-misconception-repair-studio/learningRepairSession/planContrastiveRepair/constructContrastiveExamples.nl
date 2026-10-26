---
args: { explanation: LearningIntervention[], diagnostics: UnbiasedDiagnostic[], input: LearnerSessionInput }
returns: LearningIntervention[]
---
Build one example that exposes the discriminating concept and one close contrast that changes that concept while keeping surface difficulty stable. Preserve the independently validated answer space and avoid reproducing only the original numbers or wording.
