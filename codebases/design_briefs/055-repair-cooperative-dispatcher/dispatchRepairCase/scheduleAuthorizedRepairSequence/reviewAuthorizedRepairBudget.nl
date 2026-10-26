---
args: { assignments: RepairerMatch[], parts: PartCompatibilityReview, budget: BudgetAuthorization }
returns: RepairBudgetReview
---
Compare the proposed task and compatible part costs with the current customer-authorized scope and limit. A higher budget, different scope, or part substitution requires renewed approval before scheduling that work; retain already approved unaffected tasks.
