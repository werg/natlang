---
args: { interfaces: TradeInterface[], cells: TenderCoverageCell[] }
returns: TenderSeam[]
---
Bind boundaries = compareTradeBoundaryAssumptions(interfaces, cells). For a connection neither trade accepts, bind task = createUnownedConnectionAcceptanceTask(boundaries). Do not infer ownership from trade names or adjacency. Preserve an unresolved owner and block completeness until acceptance is explicit.
