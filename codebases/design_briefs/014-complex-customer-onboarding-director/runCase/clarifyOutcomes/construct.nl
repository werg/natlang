---
args: { candidate: Onboarding, decisions: DecisionsOnboardingOutcome, evidence: EvidencePage[] }
returns: DraftOnboardingOutcome
---
1. Construct an evidence-backed OnboardingOutcome record from established atoms, using the fields declared for OnboardingOutcome in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Convert broad goals into observable outcomes and confirm what onboarding does not include. Distinguish an executive aspiration from an agreed deliverable, and identify decisions requiring customer authority before technical work begins.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
