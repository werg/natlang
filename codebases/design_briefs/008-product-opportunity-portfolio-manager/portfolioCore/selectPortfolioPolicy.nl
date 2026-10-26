---
args: { rules: PolicyCandidate[], strategyId: string, horizon: string }
returns: PolicySelection
---
Select only priorities and constraints applicable to the named strategy and planning horizon. Preserve unresolved sponsor authority, capacity limits, and benefit weights instead of assigning default values.
