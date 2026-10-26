---
args: { userRequest: CommissionRequest; record: ArchiveCommission; sourceEvidence: EvidenceItem[] }
returns: ArchiveCommissionBinding
---
Build the active commission request from the ArchiveCommission's top-level domain fields, not record.fields: copy researchQuestion, requesterId, researcherId, collectionIds, and effortBudget exactly. Retain the user's deliverable and exclusions from userRequest. Do not substitute a record ID for the archive collection set or lower the effort budget. Match every record.evidenceIds and record.sourceRefs entry to sourceEvidence by exact ID and version; report missing source references as blockers rather than inferring their contents. Return the bound CommissionRequest, the exact evidence IDs, and those blockers.
