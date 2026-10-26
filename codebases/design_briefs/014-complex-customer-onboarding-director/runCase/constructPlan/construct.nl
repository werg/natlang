---
args: { candidate: EnvironmentConstraint, decisions: DecisionsOnboardingTask, evidence: EvidencePage[] }
returns: DraftOnboardingTask
---
1. Construct an evidence-backed OnboardingTask record from established atoms, using the fields declared for OnboardingTask in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Decompose outcomes into configuration, data, training, and acceptance tasks. Order dependencies, allocate ownership explicitly, and create reversible pilots before broad changes; identify shared prerequisites that would otherwise generate duplicated requests.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
