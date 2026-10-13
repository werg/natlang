---
description: Explore the finite belief-state graph and solve its all-outcome cost recurrence.
args:
  request: MissionRequest
  survey: SurveyEvidence
  worlds: WorldSet
returns: PlanningResult
---
1. If worlds.worlds is empty or a source unknown affects legality, return a blocked policy with an empty graph
   and explicit reasons. Do not treat absence of models as an automatically successful universal plan.
2. Initialize the robot from source gear and request.start/initialEnergy, and its belief with every
   surviving world ID. A state key includes location, energy, sampleHeld, rope, rigSide and the world-ID set.
3. Derive a finite universe bound from source locations, integer energies0..initialEnergy, source gear
   configurations and nonempty subsets of world IDs. Store this as universeSize; it is a structural bound,
   not a timeout or an arbitrary allowance for search.
4. Initialize BeliefFrontier with the initial state pending. Invoke expandBeliefState.iterateOn(frontier)
   with measure universeSize - expandedStateIds.length until pendingStateIds is empty. Every step expands
   one previously unexpanded exact state key, and the source-derived universe cannot grow.
5. Consume the returned graph. Strictly positive action costs mean every retained transition leads
   to lower energy; the value computation therefore has an explicit finite dependency order.
6. Call evaluateBranches(graph, request), then closePolicy(graph, values, request). The first computes
   all-outcome winning values; the second projects only chosen policy edges without guessing a new route.
7. Return graph, values and policy. Keep rejected action evidence even when an optimal policy exists,
   because it explains why a superficially shorter route fails under another supported map.
8. Optimization compares worst-case energy, then observations, only among all-world winning policies.
   State reuse preserves the exact energy field and all outgoing outcome dependencies.
