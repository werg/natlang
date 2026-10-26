---
args: {"identities": "StockIdentityDecision[]", "observations": "DeviceObservation[]", "prior": "WarehouseJournal", "evidence": "Evidence[]"}
returns: "InventoryWork[]"
---
Classify each discrepancy against established identity and movement state. Damage changes availability while physical presence remains; pending intake is not established absence; a location mismatch is not a quantity loss. Derive affected quantities exactly in compatible units and retain uncertain quantities as null. Limit holds to identified stock/commitments supported by the evidence; do not broadly quarantine unrelated inventory to hide ambiguity.
