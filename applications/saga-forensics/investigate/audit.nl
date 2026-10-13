---
description: Replay every proposed outcome branch and challenge the forensic reconstruction.
args:
  request: InvestigationRequest
  index: EvidenceIndex
  correlation: Correlation
  ledger: EffectLedger
  repairResult: RepairPolicy
returns: ForensicReport
---
1. Check every returned witness by looking up its sourceId and path in index.documents and requiring quote to
   occur verbatim in that document. Record a defect for missing IDs, wrong paths or altered quotations.
2. Enumerate the finite supported committed/absent choices for uncertain ledger refunds. Compute each starting
   cash value from obligations.confirmedCash minus the refunds committed in that world. Deduplicate numbers
   for currentCashPossibilities without merging worlds that require different repair actions.
3. If candidate is null, retain its blocked reasons and return no branch accounting; do not manufacture a policy.
4. Otherwise visit each policy branch. Recover its starting cash from its conditions, then replay common and
   branch actions in a dependency-compatible order. Check referenced subjects and quantities before arithmetic.
5. Refund subtracts money once. Release transfers the specified unconsumed reservation units to free stock.
   Signed-custody units remain inUse, protected reservations remain reserved, and deposits remain separate.
6. Record BranchAccounting: condition pairs, startingCash, finalCash, free, inUse, reserved, writeCount and
   protectedEffectIds for whole effects preserved from compensation (deposits and nontarget reservations).
   Partially consumed reservations are represented by inUse, not whole-effect protection. Verify initialStock
   equals free + inUse + reserved in every branch.
7. Check every supported world reaches exactly one semantically matching outcome branch. Branch order and
   generated action IDs do not affect correctness; missing, contradictory or duplicated business actions do.
8. Call challengePolicy(request, index, correlation, ledger, repairResult, branchAccounting). Append its findings
   to quote, coverage and accounting defects. Preserve defended tempting alternatives for later review.
9. Return ForensicReport with the actual identity, timeline, edges, ledger, repair, cash possibilities, accounting,
   findings and inventory coverage. A defect is visible failure evidence, not something to remove from the report.
10. This audit provides a second natural-language challenge and exact arithmetic within execution. Independent
    external source/oracle verification is still required before claiming correctness or training admission.
