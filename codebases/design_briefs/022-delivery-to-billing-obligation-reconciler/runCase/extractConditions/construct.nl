---
args: { candidate: BillingCase, decisions: DecisionsBillingCondition, evidence: EvidencePage[] }
returns: DraftBillingCondition
---
1. Construct an evidence-backed BillingCondition record from established atoms, using the fields declared for BillingCondition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read contract schedules and amendments to identify triggers, acceptance requirements, partial billing rules, and exclusions. Distinguish work performed from the specific event authorizing billing and retain conditions that apply only to particular deliverables.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
