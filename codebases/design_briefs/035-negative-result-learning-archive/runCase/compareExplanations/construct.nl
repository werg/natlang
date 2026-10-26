---
args: { candidate: NegativeFinding, decisions: DecisionsFailureExplanation, evidence: EvidencePage[] }
returns: DraftFailureExplanation
---
1. Construct an evidence-backed FailureExplanation record from established atoms, using the fields declared for FailureExplanation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Construct competing explanations grounded in observed deviations and results. Separate demonstrated causes from plausible hypotheses, seek contradictory observations, and identify shared mechanisms across attempts without assuming all similar outcomes have one cause.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
