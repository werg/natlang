---
args: { ledger: IdentityCorrectionLedger, plan: IdentityCorrectionPlan, receipt: IdentityCorrectionReceipt }
returns: IdentityCorrectionReconciliation
---
1. If receipt.receiptId is already consumed, return the unchanged ledger as replayed only when every receipt field matches its stored record; a reused ID with changed content is held as an identity conflict.
2. Bind assessment = assessAppliedGraph(plan, receipt, ledger). If receipt identity, resolution, or prior graph version does not bind to the current plan and graph, preserve it as held evidence and do not advance the graph version.
3. For "applied", advance currentGraphVersion to receipt.observedGraphVersion only when assessment.safeToAdvance is true. If it is partial, preserve the exact receipt and leave the graph version unchanged while every affected consumer remains under reconciliation.
4. For "rolled-back", retain the original prior graph version and mark the correction rolled back only when every required reversal is evidenced. A partial reversal stays open and uncertain.
5. Record partial, unknown, applied, and rolled-back events without deleting canonical history or source attributions. Increment ledger.revision once per new receipt, consume receiptId once, and return the affected consumers and exact follow-up work. A merged identity correction never erases the records or evidence that justified it.
