---
args: { interfaces: TradeInterface[], cells: TenderCoverageCell[] }
returns: TradeBoundaryAssessment[]
---
Compare each trade's line-level inclusion and exclusion for the named interface. Retain both assumptions and their source cells. Mark an interface unassigned only when neither party's current cell accepts the connection.
