---
description: Rebuild the run of show against exact resource intervals and decision deadlines.
args:{event:EventProductionState;commitments:PromiseLedgerItem[];options:string[];checkpoint:LiveCheckpoint}
returns:RunSheetProof
---
Bind impact = `traceDisruptionDependencyImpact(event, commitments, options, checkpoint)`. Recompute only its affected and downstream segment IDs; compare every exact commitment and resource ID in that impact set. Carry its preserved and at-risk commitment IDs and unresolved dependency IDs into the proof findings. Verify ordered setup and teardown, exact resource availability, no double-booking, role ownership and the hard venue close. Use its per-option switch deadlines; do not infer additional slack. Preserve the current run sheet when proof fails.
