---
args: { interfaces: TradeInterface[], cells: TenderCoverageCell[] }
returns: TenderSeam[]
---
For each physical or functional connection, inspect both trade cells and their explicit boundary assumptions. Where neither accepts ownership, create a named acceptance task, keep owner unresolved, and block completeness until a bidder or accountable owner accepts it.
