---
args: {"work": "InventoryWork[]", "adjustments": "StockAdjustment[]", "evidence": "Evidence[]"}
returns: "string[]"
---
Match late receipt events to original purchase/lot/serial identity and previous shortage adjustments. If the receipt may describe stock already compensated by an adjustment, suspend any new increment and identify the exact prior operation to reconcile. Require actual reversal or matching ledger receipt before releasing the difference. Do not offset a different lot or delete damaged physical stock to balance the ledger.
