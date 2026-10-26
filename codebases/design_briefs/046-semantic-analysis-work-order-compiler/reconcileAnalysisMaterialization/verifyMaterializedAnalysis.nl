---
args: { request: SemanticWorkOrderRequest, proposal: SemanticWorkOrderProposal, receipt: AnalysisMaterializationReceipt, expectedGraphDigest: string, reclassification: RubricReclassificationPlan | null }
returns: AnalysisMaterializationAssessment
---
1. Bind coverage = auditRowCoverage(request, proposal, receipt, expectedGraphDigest), rubric = auditRubricLineage(request, proposal, receipt, reclassification), and disclosure = discloseToAuthorizedAudience(request, receipt, proposal).
2. Publish only when row coverage is complete, graph grain and schema version match, rubric is current with all required reclassification complete, and disclosure.status is allowed. Any inaccessible rows stay excluded.
3. Return status "unknown" when evidence or authorization is indeterminate, "needs-reclassification" when the rubric version or affected row labels changed, and "held" for denied disclosure or a material coverage/grain defect.
4. Preserve exact protected row IDs and give the caller every gap to repair; never create a typed answer from a held artifact.
