---
args: {"component": "ReceiptComponent", "editions": "PolicyCandidate[]", "evidence": "EvidencePage[]"}
returns: "string[]"
---
Locate the policy's applicability rule for expense date, booking date, trip start or another explicitly specified event. Resolve applicable edition and category using the claimant's supplied context. Preserve an unresolved overlap or missing transition rule; newer upload time does not authorize retroactive treatment. Return edition identity, applicable rule and unresolved scope with source references.
