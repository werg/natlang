---
description: Recompute the exact production batches from the current menu and reconcile the persisted ledger artifact.
args:{ledger:KitchenLedger;dishes:Dish[]}
returns:BatchReconciliation
---
Bind exact = `reconcileScaledBatches(ledger.brief.covers, dishes, ledger.exactScaledBatches)` using the bounded integer mechanism over the actual dishes selected for production after composition. It returns typed blockers for malformed/nonpositive integer quantities, invalid covers/yields and nonintegral minor-unit scaling rather than throwing. Compare expected and supplied rows by dish, ingredient and unit, including duplicate, missing, extra and unequal quantities. Return both complete row sets, every discrepancy and every arithmetic blocker; do not silently rewrite the production ledger.
