---
args: {"frontier": "PackageCandidate[]", "requirementId": "string", "offers": "QuoteOption[]", "replies": "TermReply[]", "mandate": "MandateBoundary"}
returns: "PackageCandidate[]"
---
For each carried package enumerate the finite explicitly offered allocations for this requirement, including split supply only where offered units and increments permit it. Call comparePackageConditions(candidate, offers, replies, mandate) for each extension. Reject extensions that borrow unavailable capacity, break exclusivity or ignore dependent terms. Keep the unfilled requirement as a blocker if no extension survives. Do not select a package merely because it has the lowest quoted number.
