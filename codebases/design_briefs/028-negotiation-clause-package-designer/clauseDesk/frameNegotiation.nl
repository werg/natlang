---
args: { request: NegotiationPackageRequest, records: ClauseNegotiation[] }
returns: GoalContract
---
Bind requested issues, parties, draft version and negotiation owner. Preserve each party's position and non-negotiable boundary; do not infer consent from silence, prior drafts or an internal fallback note.
