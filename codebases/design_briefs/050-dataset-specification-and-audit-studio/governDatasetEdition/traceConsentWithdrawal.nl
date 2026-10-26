---
args: {"events": "Evidence[]", "prior": "DatasetEditionState"}
returns: "ConsentRemoval[]"
---
Verify the withdrawing authority and exact source/purpose scope. For each affected source call traceReleasedDescendants(sourceId, prior.editions). Quarantine prospective use immediately in the returned state, retain a minimal restricted audit reference, and identify released artifacts, derivative annotations and downstream recipients requiring action. Record removal and notification as outstanding until actual receipts arrive. Unknown lineage is an explicit investigation, never a claim that all derivatives were deleted.
