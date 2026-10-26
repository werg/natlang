---
args: {"submission": "ExpenseSubmission", "decisions": "ExpenseLineDecision[]"}
returns: "string | null"
---
Determine the claimant and preparer identities before reading approval delegation. Exclude both from approving their own case, including a manager claiming their own expenses. Resolve the actual entity/category/amount-specific independent authority from the supplied policy. Return that authority or null with a review blocker; hierarchy or seniority alone is not delegated approval.
