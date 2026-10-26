---
description: Reconcile authorized customer-service operations by stable operation identity.
args:{scope:GoalContract;policy:PolicySelection;remedies:RemedyBundle[];receipts:EvidencePage[]}
returns:StageResultRecoveryAction
---
Bind reconciliation = `reconcileOperationReceipts(remedies, receipts)`. Require each refund, replacement, pickup or communication to carry a stable operation ID and matching approval or preauthorized grant. Classify confirmed, failed, unknown and reservation-expired separately. An unknown external effect requires status readback before retry; do not duplicate irreversible payment or shipment. Preserve the action plan independently from its execution receipts.
