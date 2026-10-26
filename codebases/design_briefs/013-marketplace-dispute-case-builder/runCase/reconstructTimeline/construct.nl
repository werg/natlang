---
args: { candidate: DisputeCase, decisions: DecisionsDisputeEvent, evidence: EvidencePage[] }
returns: DraftDisputeEvent
---
1. Construct an evidence-backed DisputeEvent record from established atoms, using the fields declared for DisputeEvent in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Align listings, messages, shipping records, and submissions without equating timestamps with event occurrence. Attribute disputed statements to their speakers, preserve edited listings, and mark contradictions rather than manufacturing a single settled narrative.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
