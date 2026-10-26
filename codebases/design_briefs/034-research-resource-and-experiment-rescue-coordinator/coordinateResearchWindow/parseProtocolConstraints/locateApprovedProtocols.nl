---
args: { request: ResearchWindowRequest, snapshot: ResearchOperationsSnapshot }
returns: ProtocolReview
---
Match requested experiment IDs from the scoped case records to all protocol candidates in the snapshot. Preserve approved, superseded, draft, and unavailable versions with exact source references. Do not accept an unapproved protocol as a mandate. Report every request experiment with no approved protocol in missingProtocolIds.
