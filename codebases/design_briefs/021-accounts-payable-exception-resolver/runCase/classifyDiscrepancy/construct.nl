---
args: { candidate: InvoiceMatch, decisions: DecisionsPayableDiscrepancy, evidence: EvidencePage[] }
returns: DraftPayableDiscrepancy
---
1. Construct an evidence-backed PayableDiscrepancy record from established atoms, using the fields declared for PayableDiscrepancy in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret differences using contract terms and receiving evidence. Distinguish partial delivery, unauthorized substitution, freight charges, and genuine duplication; a repeated invoice number is a signal to investigate rather than a sufficient payment decision.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
