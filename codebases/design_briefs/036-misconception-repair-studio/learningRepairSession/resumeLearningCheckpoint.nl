---
args: { caseId: ID, previous: LearnerCheckpoint | null }
returns: LearnerCheckpoint
---
Restore the last committed task boundary from previous when it belongs to caseId. Preserve completed task IDs and prior hint exposure, but do not replay hidden diagnostic labels or disclose a private hypothesis in a learner-facing prompt. If no prior checkpoint exists, initialize empty completed IDs and zero handoff attempts for this case.
