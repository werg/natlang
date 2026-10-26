---
args: {"journal": "WarehouseJournal", "receipt": "InventoryReceipt"}
returns: "InventoryReceiptReview"
---
Find the pending work/adjustment by exact operation identity and expected ledger version. Call checkCustodyConservation(journal, receipt). If movement, adjustment or quantity evidence is incomplete, preserve unknown status and residual differences. Otherwise apply the actual returned bin versions once and attach the authoritative receipts. Keep physical presence and availability separate, preserve lot/location custody, and do not label a customer order fulfilled merely because an item was found.
