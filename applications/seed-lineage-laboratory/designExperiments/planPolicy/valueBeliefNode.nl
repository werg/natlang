---
description: Value one dependency-ready belief state from actual returned child values.
args:
  state: ValuationFrontier
returns: ValuationFrontier
---
1. Select an unvalued node whose reachable successor nodes all have actual NodeValues. Solved leaves
   are ready immediately. If no dependency-ready node exists, return an explicit unresolved terminal.
2. A solved leaf has zero expected/worst cost and its one mechanism goal class. Preserve its worlds
   for later residual-genotype reporting, then remove its ID from the unvalued frontier.
3. For each actual experiment transition, join every answer branch to its returned child NodeValue.
   A missing or unresolved child makes that candidate incomplete, with the reason retained.
4. For a complete candidate, add the experiment's cost to the exact probability-weighted child cost
   sum. Worst cost adds the same immediate cost to the maximum reachable child worst cost.
5. Call compareCandidates(candidates, state.objective, state.objectiveWitnesses) and consume its
   actual selected experiment ID. Bind selected costs from that exact CandidateValue.
6. Append NodeValue including all alternatives and the returned comparison. If no complete candidate
   exists for a nonsolved goal, keep the state unresolved instead of assigning an invented zero cost.
7. Remove the processed original closure-node ID. Return the augmented frontier, finished exactly
   when all nodes have values or the explicit unresolved dependency condition prevents progress.
