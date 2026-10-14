---
description: Build and value a finite adaptive belief-and-material policy closure.
args:
  source: BankSource
  posterior: Posterior
  predictions: PredictionIndex
  request: LaboratoryRequest
returns: PolicyResult
---
1. If posterior or necessary prediction weights remain unresolved, preserve that limitation and do
   not claim exact optimality. Construct a root BeliefNode from actual weighted worlds and stock.
2. Give the root all original experiment slot IDs and goal classes derived from the requested target.
   A state identity includes joint world content, conditional rational weights, remaining slots and stock.
3. Initialize PolicyClosure with that root, actual partitions, stock rules and the goal. Call
   expandBeliefNode.iterateOn until every reachable state is expanded or an unresolved terminal occurs.
4. The closure is finite: transitions consume an original slot, worlds come from the finite original
   universe and stock changes are source-bounded. Mark expanded nodes permanently; do not use pending
   queue length as a decreasing measure while new successor identities are being discovered.
5. Consume the returned closure. Initialize ValuationFrontier with all its node IDs unvalued, the
   source objective/tie witnesses and no values. Call valueBeliefNode.iterateOn with remainingNodeIds.length.
6. Consume the returned actual values. Bind the root's expected and worst cost from its NodeValue,
   keeping each reachable chosen branch and the alternative candidate comparisons in the certificate.
7. Return PolicyResult with the complete closure and values. Exact state interning and fraction
   arithmetic are bookkeeping; semantic feasibility, goal completion and comparisons belong to children.
