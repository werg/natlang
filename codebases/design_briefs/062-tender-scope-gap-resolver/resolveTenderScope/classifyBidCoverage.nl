---
args: { bids: BidScope[], requirements: TenderRequirement[], revisions: TenderRevision[] }
returns: ClassifiedBidCells
---
Classify each bidder-requirement pair from the current bid revision only. Separate explicit inclusion, exclusion, allowance, and silent scope. Preserve the allowance's quantity and price basis as uncertain; never infer firm coverage from an aggregate total.
