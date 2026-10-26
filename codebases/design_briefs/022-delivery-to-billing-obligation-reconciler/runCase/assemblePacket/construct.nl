---
args: { candidate: BillableAllocation, decisions: DecisionsBillingPacket, evidence: EvidencePage[] }
returns: DraftBillingPacket
---
1. Construct an evidence-backed BillingPacket record from established atoms, using the fields declared for BillingPacket in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Assemble a customer-readable explanation and internal approval packet linking each amount to satisfied conditions. Recheck late amendments and acceptance changes before release, then reconcile the actual invoice identifier back to consumed obligations.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
