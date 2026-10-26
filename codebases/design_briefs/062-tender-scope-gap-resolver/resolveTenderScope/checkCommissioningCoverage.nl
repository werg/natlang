---
args: { amended: ClarificationResult, commissioning: ID[] }
returns: ID[]
---
Return every mandatory commissioning requirement without current explicit inclusion or accepted clarification. A bid missing commissioning is functionally incomplete even if its total is lower.
