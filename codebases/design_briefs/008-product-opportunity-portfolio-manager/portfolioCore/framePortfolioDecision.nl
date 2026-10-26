---
args: { request: PortfolioDecisionRequest, rules: PolicyCandidate[] }
returns: GoalContract
---
Constrain the optimization to the named strategy, planning horizon, and sponsor-authorized priorities. Keep unprovided strategic weights, budget scope, or capacity assumptions unresolved instead of inventing them.
