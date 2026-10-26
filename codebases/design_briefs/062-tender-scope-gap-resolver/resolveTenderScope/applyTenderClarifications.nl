---
args: { cells: TenderCoverageCell[], clarifications: TenderClarification[], revisions: TenderRevision[] }
returns: ClarificationResult
---
Bind invalidation = invalidateClarificationsAfterRevision(cells, clarifications, revisions). Recompute only cells supported by an active bidder revision and an applicable clarification issued to that bidder and requirement. Preserve stale result IDs so an earlier inclusion cannot remain active after a late bid revision removes it.
