---
args: { bids: BidScope[], revisions: TenderRevision[] }
returns: BidRevisionPin[]
---
Select a revision only when its supersedes chain and receipt ordering are supported. Preserve superseded revision IDs. If two revisions claim to be current or the chain has a gap, pin no current revision and block coverage claims derived from either one.
