---
args: { candidate: ClauseNegotiation, decisions: DecisionsNegotiationPosition, evidence: EvidencePage[] }
returns: DraftNegotiationPosition
---
1. Construct an evidence-backed NegotiationPosition record from established atoms, using the fields declared for NegotiationPosition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Translate stakeholder goals and the approved playbook into issue-level positions. Distinguish mandatory limits from preferences, identify dependencies between issues, and preserve disagreements that require a mandate decision before drafting a proposal.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
