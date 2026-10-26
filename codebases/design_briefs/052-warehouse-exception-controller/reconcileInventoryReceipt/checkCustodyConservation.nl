---
args: {"journal": "WarehouseJournal", "receipt": "InventoryReceipt"}
returns: "string[]"
---
Compare before/after bins for the exact item, lot, location and unit. Transfers conserve physical quantity across locations; damage changes availability only; approved quantity adjustments require their own actual receipt. Verify operation identity and ledger version and reject a receipt already consumed by a different operation. Use exact quantity arithmetic; return every residual difference and leave the pending operation unknown on incomplete evidence.
