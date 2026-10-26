---
args: {"clock": "ObligationClock", "deliveries": "DeliveryClaim[]", "authority": "PolicyCandidate[]", "sources": "EvidencePage[]"}
returns: "AcceptanceDecision"
---
Bind response = interpretCustomerAct(clock, deliveries, sources). Bind authorityFindings = verifyAcceptanceAuthority(response, clock, authority). If the response is merely acknowledged or signer authority is unresolved, retain that status and ask for confirmation; do not discard the message. Consider deemed acceptance only through testDeemedPrerequisites(clock, deliveries, sources). Preserve phase-specific acceptance and every still-open condition. Return an accepted interval only when both the actual contractual event and required authority are established.
