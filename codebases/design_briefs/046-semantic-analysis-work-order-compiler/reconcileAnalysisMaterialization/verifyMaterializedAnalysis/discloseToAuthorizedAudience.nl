---
args: { request: SemanticWorkOrderRequest, receipt: AnalysisMaterializationReceipt, proposal: SemanticWorkOrderProposal }
returns: AnalysisDisclosureAssessment
---
Require audienceAuthorization "authorized" and an outputAudienceId bound to the request's authorized requester or declared audience. Check every included record class against request.accessRules for that requester. If any included class is denied, block visibility; if authorization or record class is unknown, hold the output. Excluded counts may be shown only when the policy explicitly permits disclosure. Keep the analysis artifact stored but hidden when these checks fail.
