---
args: { candidate: Counteroffer, decisions: DecisionsSourcingAgreement, evidence: EvidencePage[] }
returns: DraftSourcingAgreement
---
1. Construct an evidence-backed SourcingAgreement record from established atoms, using the fields declared for SourcingAgreement in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret responses as acceptance, rejection, or qualified counteroffer at the term level. Rebuild package feasibility after each material change, identify unresolved conditions, and prepare a final agreement only when all linked commitments are mutually consistent.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
