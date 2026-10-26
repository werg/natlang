---
args: { candidate: ReturnCase, decisions: DecisionsConditionAssessment, evidence: EvidencePage[] }
returns: DraftConditionAssessment
---
1. Construct an evidence-backed ConditionAssessment record from established atoms, using the fields declared for ConditionAssessment in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Reconcile customer descriptions, inspection notes, and supplied images or structured image annotations. Distinguish cosmetic damage, functional symptoms, and unverified allegations; preserve uncertainty and request evidence that changes routing rather than generic photo uploads.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
