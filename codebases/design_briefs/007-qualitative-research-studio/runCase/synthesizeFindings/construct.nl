---
args: { candidate: CodingDecision, decisions: DecisionsResearchFinding, evidence: EvidencePage[] }
returns: DraftResearchFinding
---
1. Construct an evidence-backed ResearchFinding record from established atoms, using the fields declared for ResearchFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Build findings around supported mechanisms and recurring patterns, actively search for counterexamples, and distinguish an absent observation from evidence of absence. Return unresolved questions and a traceable path from each conclusion to the coded passages.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
