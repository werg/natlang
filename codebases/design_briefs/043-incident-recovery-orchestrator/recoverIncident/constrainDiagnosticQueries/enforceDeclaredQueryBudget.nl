---
args: { allowlist: string[], policy: string[] }
returns: DiagnosticBudgetDecision
---
Bind each candidate to an exact named operation and its declared rate or resource limit. Reject broad query scopes and any attempt to extend credentials. If a measurable budget is absent, block the diagnostic rather than inventing one.
