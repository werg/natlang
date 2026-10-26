---
args: { bids: BidScope[], requirements: TenderRequirement[], revisions: TenderRevision[] }
returns: ClassifiedBidCells
---
Bind pins = selectCurrentBidRevisions(bids, revisions). Bind allowances = assessAllowanceBasis(bids, requirements, pins). Form one coverage cell per bidder and requirement from the current revision only. Explicit inclusions, exclusions, allowances, and silent scope stay distinct; an allowance is never firm coverage without a quantity basis and acceptance rule.
