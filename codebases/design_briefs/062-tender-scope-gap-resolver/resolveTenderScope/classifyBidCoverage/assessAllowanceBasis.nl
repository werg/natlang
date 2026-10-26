---
args: { bids: BidScope[], requirements: TenderRequirement[], pins: BidRevisionPin[] }
returns: AllowanceAssessment[]
---
For each allowance identify the affected requirement, stated quantity basis, and whether the bidder commits to a firm amount. Keep missing quantity, escalation, or acceptance conditions explicit; never convert an allowance into fixed coverage because it appears inside the total.
