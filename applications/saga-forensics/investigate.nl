---
description: Reconstruct a distributed saga and propose an evidence-backed contingent repair.
kind: directory-reducer
args:
  request: InvestigationRequest
returns: ForensicReport
---
1. Read the immutable source snapshot by calling inventory(folder, request). Keep its exact documents and coverage.
   The folder contains evidence, including its manifest; expected answers are outside this folder.
2. If the manifest case ID differs from request.caseId, stop with a source mismatch rather than combining cases.
3. Call correlate(index, request) to resolve scoped request identities, event times and directed evidence links.
   Consume that result as the sole correlation state supplied to subsequent branches.
4. Call adjudicate(index, correlationResult) to turn request groups and observations into an effect ledger.
   An unresolved outcome remains a set of possibilities. It is not a zero-valued or absent effect.
5. Call repair(request, index, correlationResult, ledgerResult). This computes obligations before choosing actions.
   The repair is a proposal over the snapshot; do not write source files or contact an external bank or warehouse.
6. Call audit(request, index, correlationResult, ledgerResult, repairResult).
   Return the audited report, including unresolved propositions and defects even when no valid policy was found.
7. Preserve source identifiers and the manifest's output conventions. Generated action identifiers may be any
   unique strings. A source event, attempt and committed effect keep their distinct identifier roles.
8. Do not reconstruct a second answer from raw source after the children return. If their contributions conflict,
   expose the conflict through audit. Every asserted output must follow the returned typed state and witnesses.
