---
args: { graph: AnalysisGraphPlan, analysis: SemanticWorkOrderRequest }
returns: AnalysisExecutionAudit
---
Reconcile every eligible input row to included, excluded, inaccessible, missing, or unresolved disposition. Compare expected and materialized join counts at each declared grain. Keep the exact rubric version attached to cached semantic judgments. If the rubric changed, do not relabel the old run in place; create a bounded affected-row reclassification plan. If only grouping changed, reuse qualified row judgments and recalculate only the aggregation.
When current and prior rubric versions differ, bind invalidation = pinRubricVersionForReclassification(analysis.priorRun, analysis.rubricVersion, analysis.changedEvidenceIds). Preserve all prior classifications under their original version.
