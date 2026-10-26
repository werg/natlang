---
args: { diagnostics: UnbiasedDiagnostic[], input: LearnerSessionInput }
returns: UnbiasedDiagnostic[]
---
1. Bind presentation = transformPresentationForAccess(diagnostics, input). Apply only documented accommodation requirements.
2. Bind invariant = auditCompetencyInvariant(diagnostics, presentation, input). Confirm the target competency, cognitive demand, and accepted answer space are unchanged.
3. Return the accessible diagnostic with its accommodations visible to authorized staff, not as a student-facing label.
