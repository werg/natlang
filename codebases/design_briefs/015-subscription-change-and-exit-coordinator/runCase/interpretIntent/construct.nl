---
args: { candidate: SubscriptionCase, decisions: DecisionsChangeIntent, evidence: EvidencePage[] }
returns: DraftChangeIntent
---
1. Construct an evidence-backed ChangeIntent record from established atoms, using the fields declared for ChangeIntent in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read the request in conversational context and identify the intended account scope. Distinguish cancelling an add-on from closing the account, and ask a focused question only when ambiguity would change access, billing, or data retention.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
