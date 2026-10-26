---
description: Propagate witnessed outcomes into character beliefs and stable world checkpoints.
args:{world:PersistentWorldState;reconciliation:{receipts:WorldTransitionReceipt[];pendingActionIds:string[];unknownReceiptIds:string[];committedEvents:WorldEvent[];status:'awaiting-simulator'|'reconciled'|'blocked';};checkpoint:GameCheckpoint}
returns:{beliefUpdates:CharacterBelief[];continuityNotes:string[];checkpoint:GameCheckpoint;}
---
Update only characters included in event witnesses or message recipients. Preserve beliefs as attributed perceptions; a rumor does not alter world truth. Do not advance replay cursor past unknown effects. Apply participant boundary changes immediately to future generated content. Migrate changed rule versions only at a recorded checkpoint and retain prior outcomes for replay.
