---
args: { candidate: CloseWorkpaper, decisions: DecisionsReconciliationScope, evidence: EvidencePage[] }
returns: DraftReconciliationScope
---
1. Construct an evidence-backed ReconciliationScope record from established atoms, using the fields declared for ReconciliationScope in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret the purpose and supplied close policy for each workpaper. Identify the assertion to establish, relevant systems, exclusions, and escalation thresholds; keep documentary preparation separate from decisions reserved for qualified accounting reviewers.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
