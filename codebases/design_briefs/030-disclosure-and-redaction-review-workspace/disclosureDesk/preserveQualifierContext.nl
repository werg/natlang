---
args: { selections: SpanSelection[], plan: RedactionPlan, inspections: DocumentInspection[] }
returns: RedactionPlan
---
For every removed phrase, test whether the surrounding qualifier changes the allowed meaning. Keep the smallest non-sensitive context needed for truthful interpretation; if preserving it would disclose restricted content, withhold the claim and explain the limitation instead of inventing a substitute meaning.
