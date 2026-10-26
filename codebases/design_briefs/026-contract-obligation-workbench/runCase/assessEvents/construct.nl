---
args: { candidate: ObligationRelation, decisions: DecisionsTriggerFinding, evidence: EvidencePage[] }
returns: DraftTriggerFinding
---
1. Construct an evidence-backed TriggerFinding record from established atoms, using the fields declared for TriggerFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare incoming operational events with contractual triggers and conditions. Distinguish an allegation that a trigger occurred from established evidence, and request the smallest missing fact when activation would materially change a duty or deadline.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
