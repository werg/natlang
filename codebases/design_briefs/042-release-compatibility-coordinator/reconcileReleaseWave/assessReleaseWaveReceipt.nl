---
args: { request: CompatibilityReleaseRequest, plan: ReleaseTransitionProposal, wave: CoordinatedChangeWave, receipt: ReleaseWaveReceipt, book: ReleaseRolloutBook }
returns: ReleaseWaveAssessment
---
1. Bind consumers = qualifyConsumerWave(plan, wave, receipt) and rollback = auditCandidateWriteRollback(plan.rollback, receipt). Consume both child results; carry all consumer IDs and candidate fields they leave unresolved.
2. Bind versions exactly: receipt.releaseId, baselineRevision, and candidateRevision must match the request; book.candidateRevision must match request.candidateRevision; and the live active revision must equal receipt.baselineRevision.
3. A completed wave advances only when plan.readiness.status is "ready", consumer acceptance and replay evidence are complete, rollback.safe is true, and receipt.evidenceRefs exist. Partial outcomes remain partial even when most consumers accepted.
4. Return advanceAllowed, rollbackSafe, unresolved consumer IDs, and every version, data, or acceptance blocker.
