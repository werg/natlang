---
args: { candidate: TermSense, decisions: DecisionsConceptAlignment, evidence: EvidencePage[] }
returns: DraftConceptAlignment
---
1. Construct an evidence-backed ConceptAlignment record from established atoms, using the fields declared for ConceptAlignment in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare operational meanings and counterexamples for candidate terms. Establish equivalence only when obligations and consequences align; represent partial overlap explicitly and retain organization-specific variants where unification would lose behavior.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
