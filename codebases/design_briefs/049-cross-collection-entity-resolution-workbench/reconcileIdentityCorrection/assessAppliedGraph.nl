---
args: { plan: IdentityCorrectionPlan, receipt: IdentityCorrectionReceipt, ledger: IdentityCorrectionLedger }
returns: IdentityApplyAssessment
---
1. Require receipt.resolutionId to match the correction plan and receipt.priorGraphVersion to equal both plan.priorGraphVersion and ledger.currentGraphVersion. A stale graph receipt cannot advance the current identity graph.
2. For status "applied", require observedGraphVersion to equal plan.proposedVersion and every changed edge, source-revocation effect, and downstream link in the plan to have a corresponding actual receipt entry. Missing any entry leaves the result partial and names every affected consumer.
3. For "rolled-back", require observedGraphVersion to equal the original prior graph version and receipt.reversedOperationIds to cover the applied operations that require reversal. A rollback without every reversal receipt remains unknown.
4. For partial or unknown status, never advance ledger.currentGraphVersion. Return the unresolved consumer set and reconciliation evidence needed; do not infer that reverse operations ran merely because the plan lists them.
