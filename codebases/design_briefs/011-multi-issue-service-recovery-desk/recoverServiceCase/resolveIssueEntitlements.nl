---
description: Match each issue to its purchase-date policy and scoped exception.
args:{scope:GoalContract;policy:PolicySelection;issues:ServiceIssue[];purchaseRecords:EvidencePage[];exceptions:EvidencePage[]}
returns:StageResultEntitlementFinding
---
For each issue bind its original purchase date to the effective policy edition and purchased service. Bind matches = `matchIssuePurchaseTerms(issues, purchaseRecords, exceptions)`. A supervisor exception applies only to the named customer, issue, amount and expiry. Separate mandatory remedies from discretionary gestures. If a decisive purchase fact or exception authority is missing, keep the finding unresolved and ask only for that fact.
