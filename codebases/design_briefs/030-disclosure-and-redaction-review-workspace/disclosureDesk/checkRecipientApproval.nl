---
args: { request: DisclosureReviewRequest, approvals: RecipientApproval[] }
returns: string[]
---
Match approval to the current recipient, purpose, document set, and artifact hash. A recipient change invalidates prior delivery authorization; require fresh authority and approval for the new recipient before dispatch.
