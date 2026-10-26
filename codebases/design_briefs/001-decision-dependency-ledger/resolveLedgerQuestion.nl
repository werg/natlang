---
args: {"ledger": "DecisionLedger", "response": "LedgerOwnerResponse"}
returns: "LedgerResolutionReview"
---
Find the exact pending question before interpreting the answer. Bind binding = verifyOwnerResolution(ledger, response). For stale, unauthorized, wording-only or unresolved outcomes leave decisions unchanged and return the explicit intervening changes or missing evidence. Only an adopt outcome appends a LedgerResolution and new affected decision revisions. Invalidate consumers of the replaced revision only; keep independently justified downstream work and historical resolutions. Mark the matching question answered exactly once. A repeated response identity with identical content is a no-op; conflicting content under that identity is unresolved.
