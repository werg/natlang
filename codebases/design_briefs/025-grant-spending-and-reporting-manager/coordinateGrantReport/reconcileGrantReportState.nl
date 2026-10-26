---
args: { prior: GrantCheckpoint|null, request: GrantCycleRequest, ledger: GrantEvidenceLedger, conditions: GrantConditionSet, allocations: GrantAllocationResult|null, activities: GrantActivityRegister|null, disclosure: ParticipantDisclosureReview, corrected: GrantCorrectionPlan, submission: GrantSubmissionPlan|null }
returns: GrantCheckpoint
---
Bind the checkpoint to the award and reporting period. Preserve exact allocation slices and their original expense amounts; shared activity is attributed once by its documented basis. A revised award condition reopens only expenses in its effective period. Keep permitted participant disclosure separate from delivery evidence. Preserve every funder submission and receipt; a correction must link to the old submission and may not replace it. Record dispatch as complete only from a matching funder receipt. Identical evidence replay is a no-op.
