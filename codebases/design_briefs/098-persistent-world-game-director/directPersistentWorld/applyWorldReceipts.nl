---
description: Reconcile exact simulator outcomes and idempotent retries against one world revision.
args:{world:PersistentWorldState;actions:LegalAction[];receipts:WorldTransitionReceipt[];checkpoint:GameCheckpoint}
returns:{receipts:WorldTransitionReceipt[];pendingActionIds:string[];unknownReceiptIds:string[];committedEvents:WorldEvent[];status:'awaiting-simulator'|'reconciled'|'blocked';}
---
Match each receipt to the exact action ID and prior world revision. Apply only committed receipts and their event identifiers. A rejected action leaves state unchanged. If the same clientActionId already has a receipt, return that exact result without applying twice. An absent or ambiguous receipt is unknown and must be read back; never retry blindly or narrate an outcome. Verify ledger hash against authoritative simulator readback before marking reconciled.
