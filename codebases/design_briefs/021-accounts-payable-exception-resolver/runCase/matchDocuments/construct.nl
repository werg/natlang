---
args: { candidate: PayableCase, decisions: DecisionsInvoiceMatch, evidence: EvidencePage[] }
returns: DraftInvoiceMatch
---
1. Construct an evidence-backed InvoiceMatch record from established atoms, using the fields declared for InvoiceMatch in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Align descriptions, units, supplier references, and delivery context across invoice and order lines. Permit many-to-many matches for split shipments, but preserve ambiguous allocations and do not force every invoice line onto the nearest description.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
