---
args: { candidate: CustodyEvent, decisions: DecisionsResaleDisclosure, evidence: EvidencePage[] }
returns: DraftResaleDisclosure
---
1. Construct an evidence-backed ResaleDisclosure record from established atoms, using the fields declared for ResaleDisclosure in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Produce a truthful resale or disposal packet from verified work and remaining defects. Reconcile grade terminology with marketplace rules, remove prior-owner data, and hold listing until required functional checks and ownership transfer are complete.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
