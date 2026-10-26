---
description: Compare lot-level receipt totals to allocations and customer demand.
args: { desk: HarvestDeskState; receipts: MovementReceipt[]; allocation: LotAllocation; totals: MovementReceiptSum }
returns: MovementReconciliation
---
Preserve harvested, stored and delivered totals as independent ledgers. Match each delivery receipt to an accepted commitment and lot allocation by identity and quantity. Record missing transfer receipts, unexplained differences, and customer-rejected substitutions as blockers; do not label a plan fulfilled from a harvest or storage receipt alone.
