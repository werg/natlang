---
args: {"findings": "InventoryWork[]", "identities": "StockIdentityDecision[]", "procedures": "string[]", "prior": "WarehouseJournal", "evidence": "Evidence[]"}
returns: "InventoryWork[]"
---
For each finding choose a bounded approved search, recount, inspection, relabel or transfer task. Make identity confirmation precede relabeling and custody confirmation precede transfer closure. Call verifyRecountIndependence(work, prior, evidence, procedures) before any consequential adjustment. Bind expected ledger version and explicit prerequisites. An adjustment cannot be used solely to make the displayed balance match an expectation.
