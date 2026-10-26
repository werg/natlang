---
args: { candidate: DisputeEvent, decisions: DecisionsDisputedQuestion, evidence: EvidencePage[] }
returns: DraftDisputedQuestion
---
1. Construct an evidence-backed DisputedQuestion record from established atoms, using the fields declared for DisputedQuestion in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Determine which factual questions would change the rule-based outcome. Separate immaterial disagreements from decisive ones, identify the applicable evidence burden, and formulate neutral questions that either party can answer with concrete records.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
