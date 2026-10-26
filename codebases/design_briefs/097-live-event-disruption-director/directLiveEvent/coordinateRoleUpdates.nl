---
description: Coordinate producer approval, resource reservation and role-specific acknowledgements.
args:{event:EventProductionState;proof:RunSheetProof;checkpoint:LiveCheckpoint}
returns:ChangePacket
---
Require producer approval before public announcement. Send only each role's necessary revision and request acknowledgment against the same run-sheet revision. Keep conflicting schedule versions visible and hold if a critical safety, accessibility or technical role has not acknowledged. Record reservation receipts and actual execution events separately from plans.
