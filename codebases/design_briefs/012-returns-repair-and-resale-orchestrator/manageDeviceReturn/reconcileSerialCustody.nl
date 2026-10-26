---
description: Reconcile distinct serial-specific custody transfers from observed receipts.
args:{scope:GoalContract;policy:PolicySelection;disposition:StageResultReturnDisposition;repair:StageResultRepairPlan;serialLedger:EvidencePage[];receipts:EvidencePage[]}
returns:StageResultCustodyEvent
---
Bind handoffs = `compareSerialHandoffs(scope, serialLedger, receipts)`. Verify exact device serial and from/to custodians at every collection, receipt, repair-transfer and return. Inventory changes require a confirmed custody event. Preserve condition change at each boundary and route disputed transit damage independently. A serial mismatch blocks transfer, repair intake and resale.
