---
args: { cells: TenderCoverageCell[], clarifications: TenderClarification[], revisions: TenderRevision[] }
returns: ClarificationResult
---
Apply a clarification only to its bidder, requirement, issue date, and current revision. When a later revision removes that inclusion or changes its exclusion, invalidate the earlier cell and all dependent clarification findings rather than carrying the old positive result forward.
