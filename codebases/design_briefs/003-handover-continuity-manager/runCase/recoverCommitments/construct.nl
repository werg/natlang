---
args: { candidate: Handover, decisions: DecisionsCommitment, evidence: EvidencePage[] }
returns: DraftCommitment
---
1. Construct an evidence-backed Commitment record from established atoms, using the fields declared for Commitment in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read threads with their replies and amendments. Distinguish a tentative intention from a promise, identify the beneficiary and deliverable, and retain conditional commitments whose activation event has not yet occurred.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
