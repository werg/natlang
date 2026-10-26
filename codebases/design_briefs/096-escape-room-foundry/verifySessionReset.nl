---
description: Reconcile an actual post-reset readback with the room's exact prop baseline and session identity.
args: { room: EscapeRoomState; checkpoint: EscapeOperationsCheckpoint; resetReadback: ResetReadback | null }
returns: EscapeResetResult
---
Bind verification = `verifyResetReadback(room, checkpoint, resetReadback)` and use that returned check without reinterpretation. Set findings to verification.findings and status to verified only when verification.verified is true. Preserve the checkpoint's room and revision, set resetVerified to that result, and retain all prior receipt IDs. Keep checkpoint.resetReceiptIds unchanged when verification failed or no readback exists. When verification passed, append the exact nonempty resetReadback.receiptId only if that ID is not already in checkpoint.resetReceiptIds; never manufacture or repair a receipt. Set result.receiptId only to the exact supplied resetReadback.receiptId when verification passed; otherwise leave it absent. Keep nextSessionId unchanged. Return the complete updated checkpoint so an interrupted session can be reconciled without treating a checklist or proposed reset as evidence.
