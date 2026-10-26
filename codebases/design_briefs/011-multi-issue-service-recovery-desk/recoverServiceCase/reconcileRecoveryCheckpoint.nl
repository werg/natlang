---
args:{prior:ServiceRecoveryCheckpoint|null;request:RecoveryRequest;serviceState:ServiceRecoverySnapshot;proposal:ServiceRecoveryProposal}
returns:ServiceRecoveryCheckpoint
---
Create a first checkpoint bound to the case and exact service snapshot. Record every proposed operation ID separately from its receipt outcome. For a repeated snapshot and unchanged receipt identities, return the prior checkpoint unchanged. When receipts arrive, append only new actual receipt identities; classify each operation as completed, failed, or unknown from its authoritative outcome. Preserve closure per issue: one fulfilled issue never closes another. On a changed customer choice or entitlement edition, reopen only the affected issue and dependent bundle. Keep unknown operations in readback work and never convert them into permission to resend. Advance revision once for changed snapshot, receipt or closure.
