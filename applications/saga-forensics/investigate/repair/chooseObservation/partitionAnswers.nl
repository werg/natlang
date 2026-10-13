---
description: Interpret one query answer at a time and build a repair-relevant outcome partition.
args:
  queryAtoms: EvidenceAtom[]
  effect: Effect
  correlation: Correlation
  clauses: EvidenceAtom[]
returns: ObservationChoice
---
1. Read the proposed query ID, cost and coverage-through boundary from queryAtoms. Recover the effect's
   processing boundary from source clauses and normalized timeline; retain the exact supporting witnesses.
2. Enumerate only the query's explicitly promised answer categories. Do not add an answer that a real service
   might give if it lies outside this fictional snapshot's declared query contract.
3. Define a local AnswerScope type matching QueryAnswer. Create a scoped NL literal capturing the effect
   proposition, its processing boundary, this query's coverage and the applicable retry clause.
4. Directly await that literal separately for each answer description. Its instructions ask which committed or
   absent worlds remain possible after this answer, and whether absence is final for this processing attempt.
5. Consume each returned QueryAnswer into answers. A single answer compatible with both worlds is a merged
   partition cell; two different strings do not constitute a useful distinction unless their supported worlds differ.
6. Set sufficient only when every supported effect outcome belongs to a cell that determines a safe repair,
   and every absent cell establishes final absence. A coverage boundary preceding processing is insufficient.
7. For a committed answer, keep the named effect reference and amount in witnesses. A mismatch in amount,
   tenant or key leaves the query insufficient for this particular effect rather than redefining its proposition.
8. Return ObservationChoice with query ID, ledger ID, cost, coverage, processing boundary, answer partition,
   sufficient, reason and witnesses. Its partition is subsequently consumed to generate actual policy branches.
