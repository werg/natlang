---
description: Rebuild the run of show against exact resource intervals and decision deadlines.
args:{event:EventProductionState;commitments:PromiseLedgerItem[];options:string[];checkpoint:LiveCheckpoint}
returns:RunSheetProof
---
Recompute only segments downstream of the disruption. Verify ordered setup and teardown, exact resource availability, no double-booking, role ownership and the hard venue close. Record the last minute each option remains feasible. Preserve the current run sheet when proof fails.
