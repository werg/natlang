---
description: Propagate winning values in increasing-energy order over the positive-cost belief DAG.
args:
  graph: BeliefGraph
  request: MissionRequest
returns: StateValue[]
---
1. Check every retained edge's successor exists and has strictly less energy than its source by the
   stated action cost. A violated energy ordering is a graph defect, not a reason to iterate indefinitely.
2. Order all graph state IDs by increasing energy, then stable ID. This is a dependency order because
   a state's possible successors have already appeared at lower energy.
3. Initialize ValueFrontier with those remaining IDs, empty values, graph and request. Invoke
   settleValue.iterateOn(initialState) with measure remainingStateIds.length until it is empty.
4. Each step owns one StateValue derived from already-settled successor values. A policy action
   wins only if all of its possible outcomes win; selecting one good outcome is not an action choice.
5. Return the accumulated values. Losing and dominated alternatives remain explicit reasons, allowing
   an independent audit to check why an apparently cheap route or observation was not selected.
