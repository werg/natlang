---
description: Remove a changed premise's dependent proof cone while preserving alternative support.
args:
  state: ClosureState
  invalidatedProofIds: string[]
returns: RetractionDelta
---
1. Start a finite invalid set from the caller's explicitly changed proof IDs. Build reverse references
   from each proof premise to the proofs depending on it; source clause IDs and proof IDs remain distinct tags.
2. Keep frontier=invalidProofIds and visited empty. Consume one unvisited frontier ID, add it to
   visited, and inspect each reverse dependent. Add a dependent only when it requires that removed
   proof premise; append newly invalid IDs to the frontier once. The finite proof inventory bounds
   this traversal. A different proof for the same business key is not automatically dependent.
3. Preserve all proofs outside the invalid cone. Collect affected business keys separately so the
   caller can recompute obligation status from surviving alternative justifications and new evidence.
4. Keep exact source witnesses explaining initial changes and dependency links. A changed conclusion
   without an identified changed premise is an unresolved reconstruction defect.
5. Return invalidProofIds, survivingProofs, affectedKeys and witnesses. This is pure snapshot data;
   the previous SliceLedger remains unchanged for the comparison certificate.
