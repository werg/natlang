---
args: { candidate: BillingCondition, decisions: DecisionsDeliveryEvidence, evidence: EvidencePage[] }
returns: DraftDeliveryEvidence
---
1. Construct an evidence-backed DeliveryEvidence record from established atoms, using the fields declared for DeliveryEvidence in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Relate project artifacts, service logs, and customer messages to the promised deliverables. Identify partial or alternative delivery and distinguish a draft shared for comment from an accepted contractual handover.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
