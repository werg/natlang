---
args: {"component": "ReceiptComponent", "share": "PurposeShare", "editions": "PolicyCandidate[]", "evidence": "EvidencePage[]"}
returns: "ExpenseLineDecision"
---
Bind rules = selectTripEffectiveRule(component, editions, evidence). Bind exception = checkScopedException(component, rules, evidence). Apply category limits, meal inclusions and documented business shares to the exact amount in its authorized currency. A policy changed during the trip follows the supplied effective-date rule; do not silently substitute the newest edition. Keep allowed/rejected/held null when a required amount or conversion basis is unverified. Explain each reduction or hold by the actual rule and component.
