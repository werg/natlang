---
description: Choose only fully audited alternatives under the director's repair objective.
args:
  component: AffectedComponent
  reviews: CandidateReview[]
  rules: PreservationRule[]
returns: CandidateChoice
---
1. Partition reviews by complete audit coverage. Reject a candidate with a violated hard obligation,
   an unresolved required check, missing obligation ID or invalid exact source check. Preserve reasons.
2. If no candidate remains, return no selections, empty localDecisions and the component's unresolved conjunction.
   A candidate with fewer edits but invalid knowledge is not a tradeoff winner.
3. Capture the valid reviews, preservation rules and director's declared minimality objective. In eval
   declare local RepairTradeoff with preferredIds: string[], equivalentIds: string[], reason: string,
   witnesses: {sourceId: string; path: string; quote: string}[]. Create nl.with<RepairTradeoff> over the changing input valid candidate IDs.
4. Its instructions are: compare only audited valid alternatives; prefer the declared smaller coherent
   semantic disturbance; use edit counts only after meaning preservation; retain equally valid alternatives
   when the rules do not rank their dramatic choices; cite the objective and audited coverage.
5. Directly await the literal. Consume preferredIds as the choice candidates and equivalentIds as
   explicit alternatives. Verify every returned ID belongs to the valid review set.
6. Select one compatible complete alternative for the packet when several are genuinely tied; use
   the visible deterministic tie convention only if specified. Otherwise retain an unresolved tie.
7. Return selectedCandidateIds, rejected records, equivalentAlternatives and unresolved. Rejected
   candidates cite the exact violated obligation IDs, not a generic low-score explanation.
8. Never compare invalid repairs as if a hard rule were a preference. The local literal's contribution
   is a reasoned ordering over valid semantic alternatives, consumed by the folder reducer.
9. Include localDecisions with this site, compared candidate IDs, actual returned tradeoff premise
   and witnesses. An unresolved tie still records the returned judgment; no valid candidates records none.
