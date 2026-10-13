---
description: Own one all-outcome dynamic-programming value and its chosen action.
args:
  state: ValueFrontier
returns: ValueFrontier
---
1. If remainingStateIds is empty, return state. Otherwise take its first ID and remove it from a copy
   of that list. Retrieve the corresponding state and expansion from graph.
2. A goal state is winning with zero remaining energy/observations and null chosenActionId. A dead
   nongoal state is losing with null costs and no chosen action.
3. For each nonterminal action edge, gather the already-settled value of every outcome successor.
   A missing value is an unresolved graph-order defect; a losing outcome makes this action losing.
4. For each all-outcome winning action, compute action.cost + maximum successor remaining energy.
   Compute its worst remaining observation count by adding one only for observe and taking the maximum
   child count. These maxima range over actual possible answers, not a preferred physical map.
5. Choose minimum worst energy, then minimum worst observation count. For an exact objective tie,
   either action is a valid canonical witness; use stable actionId only to make this report reproducible.
6. Return a StateValue with the chosen action, costs, reason and the source witnesses establishing
   its legality and objective. If no winning action exists, explain the concrete obstructing branches.
7. Append this value and return the shorter frontier. Do not change prior values or create graph
   transitions here; expansion owns legality, and this stage owns the minimax consequence of those edges.
