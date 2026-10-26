---
args: { policies: PolicyCandidate[], orderIds: string[] }
returns: PolicySelection
---
Select policy candidates whose remit covers each purchased service and original order date. Preserve multiple editions for multi-order cases and return unresolved precedence questions rather than choosing the newest policy by default.
