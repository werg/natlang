---
args: { request: AmendmentReviewRequest, agreements: AmendmentCase[] }
returns: GoalContract
---
Match the exact base agreement, amendment IDs, executed edition hash, effective date and review owner. Reject a baseline derived only from a draft upload; keep executed text and proposed change as distinct records.
