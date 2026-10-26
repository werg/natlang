---
args: { candidate: ReplicationDependency, decisions: DecisionsMethodQuestion, evidence: EvidencePage[] }
returns: DraftMethodQuestion
---
1. Construct an evidence-backed MethodQuestion record from established atoms, using the fields declared for MethodQuestion in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Enumerate plausible interpretations of missing details and assess whether each changes the result. Seek author clarification or documented evidence for consequential choices; retain sensitivity branches where ambiguity cannot be resolved.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
