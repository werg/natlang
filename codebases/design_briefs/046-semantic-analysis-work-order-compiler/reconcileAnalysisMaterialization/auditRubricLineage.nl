---
args: { request: SemanticWorkOrderRequest, proposal: SemanticWorkOrderProposal, receipt: AnalysisMaterializationReceipt, reclassification: RubricReclassificationPlan | null }
returns: RubricLineageAssessment
---
1. Require receipt.rubricVersion to equal request.rubricVersion for a current result. If it differs, this output cannot replace the current result.
2. When request.priorRun exists and the rubric or input evidence changed, use reclassification: reuse only unaffected judgments with the same prior row and rubric basis; reclassify every affected row under the new rubric before publication.
3. If the new rubric run has not been completed, set reclassificationRequired true, preserve the old result under its old rubric, and return only the judgments still eligible for reuse. Never silently apply a changed semantic rubric to prior materialized rows.
