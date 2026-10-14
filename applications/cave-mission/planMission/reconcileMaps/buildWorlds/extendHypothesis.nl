---
description: Translate one hypothesis into a directed, conditioned world model and reconcile overlapping reports.
args:
  state: WorldFrontier
returns: WorldFrontier
---
1. If pending is empty, return state. Otherwise take its first hypothesis and remove it from a copy
   of pending. Translate each local passage endpoint through that hypothesis's bindings.
2. Read each assertion's physicalExistence before constructing travel edges. Present connections
   contribute allowed directions: two-way gives both, one-way gives only its declared from-to direction.
   Retain reverse-forbidden witnesses and carried-state conditions on those actual edges. Absent physical
   connections contribute negative facts without a travel edge; unknown physical existence remains an
   explicit unresolved premise wherever it could affect this mission.
3. When global and local assertions describe the same passage, join their evidence. If direction,
   physicalExistence, allowed direction, cost or applicable scope appears inconsistent, call
   resolveContradiction with the complete witnessed claim group, hypothesis and survey. Preserve all
   negative claims during that assessment; a repeated comparison uses the same complete evidence.
4. Consume that assessment: compatible=true scope differences can coexist; compatible=false rejects this hypothesis.
   compatible=null appends a witnessed missing premise to state.unresolved and leaves the hypothesis
   unclaimed rather than treating it as disproved. A blocked source model cannot justify a successful plan.
5. Derive the sample location and correctRigSide from the sample-chamber binding and anchor rule.
   The visually similar dead-end niche does not acquire a sample or a return chimney by resemblance.
6. Construct WorldModel with unique worldId, hypothesis, directed passages, source facts and witnesses.
   Keep explicit absence clauses in facts so later action generation can distinguish absence from ignorance.
7. Append the model or RejectedWorld to the copied state. Preserve every previously constructed model.
   The new state's pending count is exactly one smaller regardless of acceptance or rejection.
8. Return the updated frontier. No robot actions or sensor observations occur during map construction.
