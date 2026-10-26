---
args: { candidate: ProgressFinding, decisions: DecisionsOnboardingRevision, evidence: EvidencePage[] }
returns: DraftOnboardingRevision
---
1. Construct an evidence-backed OnboardingRevision record from established atoms, using the fields declared for OnboardingRevision in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: When dependencies fail or objectives change, construct feasible alternatives and disclose scope, timing, and effort consequences. Preserve accepted work, obtain approval for material changes, and replay end-to-end outcomes before declaring onboarding complete.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
