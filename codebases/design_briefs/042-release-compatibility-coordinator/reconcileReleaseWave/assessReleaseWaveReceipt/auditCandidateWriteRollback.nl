---
args: { plan: DataRollbackPlan, receipt: ReleaseWaveReceipt }
returns: { safe: boolean, unresolvedFields: ID[], reasons: string[] }
---
Compare every candidateWriteFields entry with the actual baseline-readable field list and baselineReadbackEvidence. If the baseline cannot represent a candidate write, require the planned reconciliation step to have an actual receipt reference. A declared rollback procedure is not a successful rollback rehearsal. Return every field whose value could be lost or misread.
