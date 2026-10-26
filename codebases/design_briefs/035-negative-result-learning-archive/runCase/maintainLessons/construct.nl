---
args: { candidate: DiscriminatingExperiment, decisions: DecisionsResearchLesson, evidence: EvidencePage[] }
returns: DraftResearchLesson
---
1. Construct an evidence-backed ResearchLesson record from established atoms, using the fields declared for ResearchLesson in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Publish a scoped lesson with conditions under which it applies and links to contradictory or superseding evidence. Update recommendations when new attempts arrive, preserving the historical knowledge state and uncertainty rather than rewriting failures into hindsight certainty.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
