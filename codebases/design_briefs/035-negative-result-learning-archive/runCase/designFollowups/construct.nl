---
args: { candidate: FailureExplanation, decisions: DecisionsDiscriminatingExperiment, evidence: EvidencePage[] }
returns: DraftDiscriminatingExperiment
---
1. Construct an evidence-backed DiscriminatingExperiment record from established atoms, using the fields declared for DiscriminatingExperiment in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Propose bounded low-risk follow-ups that distinguish rival explanations with minimal changes. Specify what each outcome would mean before execution and avoid repeating the same invalid procedure or adding so many changes that the result becomes uninterpretable.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
