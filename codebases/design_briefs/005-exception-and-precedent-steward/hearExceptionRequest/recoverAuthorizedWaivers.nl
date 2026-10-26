---
args: {"history": "EvidencePage[]", "rules": "PolicyCandidate[]"}
returns: "WaiverBoundary[]"
---
Extract actual authorization boundaries from each grant: beneficiary, precise waived requirement, effective interval, expiry, surviving controls, prerequisites, signer and any explicit denial of precedent. Verify the granting role's remit under the applicable source edition. A missing expiry creates a governanceDefect; it does not extend validity indefinitely. A request, recommendation or copied decision is not a grant. Keep revoked and expired boundaries available for historical reasoning without treating them as current permission.
