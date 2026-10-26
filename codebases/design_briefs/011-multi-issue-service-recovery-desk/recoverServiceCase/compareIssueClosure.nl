---
description: Compare each issue's promised outcome with exact receipts and accepted alternatives.
args:{scope:GoalContract;policy:PolicySelection;issues:ServiceIssue[];actions:RecoveryAction[];receipts:EvidencePage[];choices:EvidencePage[]}
returns:StageResultRecoveryClosure
---
Bind comparisons = `compareAgreedIssueOutcomes(issues, actions, receipts, choices)`. A refund request, shipping label or dispatch notice is not fulfillment. Close only when the exact remedy receipt exists or the customer accepted an authorized alternative. Report partial completion per issue and retain earlier promise dependencies and open obligations. Never reset case history when one issue closes.
