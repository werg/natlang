---
args: { prior: EvidenceMapCheckpoint|null, request: EvidenceMapRequest, corpus: ResearchEvidencePack, review: ReviewFrame, eligibility: EligibilitySnapshot, lineage: StudyLineage, observations: ObservationLedger, rescored: ObservationLedger, measures: MeasurementAlignment, claims: EvidenceClaimGraph, retractions: ReviewUpdateLedger, rescreen: ReviewUpdateLedger }
returns: EvidenceMapCheckpoint
---
Preserve the review question, cutoff, frozen eligibility version and study lineage editions. Keep older screening observations when criteria change; append the new rescreened ledger and identify which study IDs changed eligibility. A report retraction invalidates only claims linked to that report lineage; preserve independent support and opposing evidence. Record reviewer questions for every reopened claim. Duplicate source and retraction receipt replay is a no-op.
