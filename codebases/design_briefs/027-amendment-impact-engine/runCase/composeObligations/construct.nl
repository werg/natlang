---
args: { candidate: AmendmentScope, decisions: DecisionsAmendedObligation, evidence: EvidencePage[] }
returns: DraftAmendedObligation
---
1. Construct an evidence-backed AmendedObligation record from established atoms, using the fields declared for AmendedObligation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Construct the amended obligation by applying the specific operation to its target. Preserve untouched exceptions, prerequisites, definitions, and cross-references; compare the result against counterexamples that would expose accidental broadening or deletion.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
