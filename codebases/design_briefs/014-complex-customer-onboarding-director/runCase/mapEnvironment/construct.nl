---
args: { candidate: OnboardingOutcome, decisions: DecisionsEnvironmentConstraint, evidence: EvidencePage[] }
returns: DraftEnvironmentConstraint
---
1. Construct an evidence-backed EnvironmentConstraint record from established atoms, using the fields declared for EnvironmentConstraint in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconcile customer architecture notes, sample data, security requirements, and available integration capabilities. Separate verified compatibility from assumptions and formulate narrow discovery tasks for unknown interfaces rather than promising unsupported integrations.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
