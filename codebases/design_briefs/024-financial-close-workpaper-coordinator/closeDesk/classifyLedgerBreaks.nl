---
args: { scope: CloseAssertionScope; evidence: EvidencePage[] }
returns: CloseBreakLedger
---
For every in-scope unmatched source line, compare transaction identity, posting date, account mapping and supported explanation evidence. Classify a difference as timing only when the same underlying item and later settlement are evidenced; classify mapping only with a documented mapping rule; use missing-posting only when the source journal is absent; otherwise preserve it as unsupported or disputed. Never net distinct lines merely because their amounts cancel. Keep the exact currency, scale and residual amount from the source records; report each unresolved fact and the evidence refs actually inspected.