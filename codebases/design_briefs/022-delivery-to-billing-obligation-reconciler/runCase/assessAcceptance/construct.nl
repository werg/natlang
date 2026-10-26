---
args: { candidate: DeliveryEvidence, decisions: DecisionsAcceptanceFinding, evidence: EvidencePage[] }
returns: DraftAcceptanceFinding
---
1. Construct an evidence-backed AcceptanceFinding record from established atoms, using the fields declared for AcceptanceFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret customer responses under the agreement’s acceptance process. Separate acknowledgment of receipt from acceptance, verify the respondent’s authority, and apply deemed-acceptance timing only when all supplied contractual prerequisites are established.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
