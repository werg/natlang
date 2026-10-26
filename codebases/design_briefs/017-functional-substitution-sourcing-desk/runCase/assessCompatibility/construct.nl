---
args: { candidate: SubstituteCandidate, decisions: DecisionsCompatibilityFinding, evidence: EvidencePage[] }
returns: DraftCompatibilityFinding
---
1. Construct an evidence-backed CompatibilityFinding record from established atoms, using the fields declared for CompatibilityFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare every essential requirement against documented candidate evidence and approved conversion rules. Check interacting properties and interface tolerances; preserve unknowns explicitly and reject candidates whose apparent equivalence depends on an unsupported assumption.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
