---
args: { candidate: ProblemStudy, decisions: DecisionsProblemEpisode, evidence: EvidencePage[] }
returns: DraftProblemEpisode
---
1. Construct an evidence-backed ProblemEpisode record from established atoms, using the fields declared for ProblemEpisode in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read enough surrounding conversation to identify the attempted job, the obstacle, and its consequence. Separate a product complaint from a requested feature, and split independently resolvable problems while retaining links to the original interaction.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
