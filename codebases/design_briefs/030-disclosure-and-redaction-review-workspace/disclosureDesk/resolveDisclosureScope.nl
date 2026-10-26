---
args: { request: DisclosureReviewRequest, cases: DisclosureCase[], policyCandidates: PolicyCandidate[] }
returns: DisclosureScopeDecision
---
Match exact recipient, purpose, document set, date window, and authority. Apply the narrowest permitted topic and exclude privileged or unrelated categories. If purpose or recipient authority is not established, withhold the affected material and return the missing authority evidence.
