---
description: Choose the next admissible probe by semantic discrimination and cost.
args:
  candidates: ProbeCandidate[]
  hypotheses: Hypothesis[]
returns: string | null
---
1. Identify hypotheses whose source-grounded mechanism still needs an observed discriminating result.
   A supported role can still need an interaction control before the whole repair is verified.
2. Among admissible candidates, compare which remaining mechanisms their changed premises and
   intermediate plan receipts distinguish. Prefer a coherent targeted test to an unexplained large percent gap.
3. Prefer lower stated probe cost among candidates that resolve the same semantic uncertainty. Preserve
   the visible tie convention if supplied; otherwise choose a reasoned stable candidate and record its rationale.
4. If no admissible candidate distinguishes a remaining mechanism, select one unused control that
   can settle a source-supported coincidence; exhausted controls lead to null rather than a repeated experiment.
5. Inadmissible seeds still need finite consumption. If only rejected seeds remain, choose one of their
   IDs for explicit rejection by the reducer; it cannot generate a numeric outcome.
6. Return the chosen seed ID or null. A missing hypothesis answer is honest exhaustion, not an
   opportunity to create a probe outside the authorized source universe.
7. The caller uses this ID to consume exactly one pending seed and write its returned evidence.
   Do not report a whole schedule whose later choices assume outcomes not yet observed.
