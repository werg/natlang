---
description: Expand one actual posterior-and-stock state through source experiment choices.
args:
  state: PolicyClosure
returns: PolicyClosure
---
1. Select one pending unexpanded node. If none remains, return finished. If its positive-mass worlds
   all share the requested goal class, mark solved without requiring their genotypes to agree.
2. Otherwise visit source partitions whose original slot remains. Call judgeExperimentUse(node,
   partition, state.stockRules, state.goal) and retain the actual returned feasibility/relevance decision.
3. For each feasible usable experiment, derive branch likelihoods from its actual world predictions.
   Recompute answer probabilities against this node's conditional weights; discard only exact-zero branches.
4. Construct each child's worlds by multiplying current weights by its answer likelihood and
   normalizing. Consume the selected original slot and subtract its actual material requirements from
   a fresh copy of this parent's stock. Every sibling starts from the same parent inventory.
5. Intern a child by its complete posterior/remaining-slot/stock content. Reuse an identical existing
   node or add a new unexpanded one. Preserve the actual transition, answer, probability and Use decision.
6. An uninformative experiment may remain as a dominated alternative if feasible; it cannot create
   new knowledge. An infeasible or unusable experiment gets no executable transition and keeps its reason.
7. Mark this node expanded and remove its pending identity. If no valid continuation can solve a
   nonsingleton goal, retain an unresolved terminal. Finish when no pending node remains.
