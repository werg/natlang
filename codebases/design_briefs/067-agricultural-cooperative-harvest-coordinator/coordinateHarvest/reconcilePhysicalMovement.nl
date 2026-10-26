---
description: Reconcile physical harvest, storage and delivery receipts without collapsing their quantities.
args: { desk: HarvestDeskState; receipts: MovementReceipt[]; allocation: LotAllocation; totals: MovementReceiptSum }
returns: MovementReconciliation
---
Bind reconciliation = `comparePlannedAndObserved(desk, receipts, allocation, totals)`. Require a sourced transition from harvested to stored and from stored to delivered; an absent receipt is not a zero quantity. Propagate invalid groups as unresolved and flag unexplained loss, duplicate delivery, and any delivered quantity exceeding accepted demand. Only an exact, traceable customer receipt may close a commitment; return reconciliation with the exact totals and issues attached.
