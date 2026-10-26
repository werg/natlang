---
description: Reconcile physical harvest, storage and delivery receipts without collapsing their quantities.
args: { desk: HarvestDeskState; receipts: MovementReceipt[]; allocation: LotAllocation }
returns: MovementReconciliation
---
Call `sumMovementReceipts(receipts)`, retaining separate quantity totals for each lot and movement phase. Call `comparePlannedAndObserved(desk, receipts, allocation, totals)`. Require a sourced transition from harvested to stored and from stored to delivered; an absent receipt is not a zero quantity. Flag unexplained loss, duplicate delivery, and any delivered quantity exceeding accepted demand. Only an exact, traceable customer receipt may close a commitment.
