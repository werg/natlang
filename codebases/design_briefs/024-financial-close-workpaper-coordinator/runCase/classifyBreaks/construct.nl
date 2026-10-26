---
args: { candidate: ReconciliationScope, decisions: DecisionsReconciliationBreak, evidence: EvidencePage[] }
returns: DraftReconciliationBreak
---
1. Construct an evidence-backed ReconciliationBreak record from established atoms, using the fields declared for ReconciliationBreak in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Consume exact reconciliation differences and inspect supporting narratives. Distinguish timing, mapping, missing postings, and unsupported explanations; group related breaks only when they share a demonstrated mechanism rather than the same amount.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
