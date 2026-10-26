---
args: { candidate: RepairPlan, decisions: DecisionsCustodyEvent, evidence: EvidencePage[] }
returns: DraftCustodyEvent
---
1. Construct an evidence-backed CustodyEvent record from established atoms, using the fields declared for CustodyEvent in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Track collection, receipt, repair transfer, and return as distinct custody events. Compare condition at each handoff and isolate new discrepancies rather than attributing all damage to the customer or overwriting the original inspection.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
