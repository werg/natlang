---
args: {"desk": "SourcingDesk", "offers": "QuoteOption[]", "replies": "TermReply[]", "mandate": "MandateBoundary"}
returns: "PackageCandidate[]"
---
Start one empty candidate with the requirements recovered from the sourcing events. Traverse that finite requirement inventory once. For each requirement call expandBundleFrontier(frontier, requirementId, offers, replies, mandate), carrying returned allocations and rejected reasons forward. Branch only on supplied quantities, supplier capacities and approved allocation increments; if an increment is unspecified, request it rather than inventing a continuous search. After the final requirement reject incomplete packages and retain every feasible package with distinct operational tradeoffs. Rank resilience and commercial fit using supplied buyer priorities, without inventing customer preferences.
