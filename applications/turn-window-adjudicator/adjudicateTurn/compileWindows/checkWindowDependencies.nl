---
description: Reconcile timing constraints without introducing a priority between independent effects.
args:
  timing: TimingBinding[]
  precedence: TimingEdge[]
returns: DependencyCheck
---
1. Inventory the finite interpreted rule identities. Require every precedence endpoint to identify an
   actual timing row; an absent semantic target remains unresolved rather than becoming an empty node.
2. Derive reachability through the source-backed edges. Group mutually reachable rule identities into
   components. These graph operations are exact over already interpreted edges; they do not create a
   new source obligation from a similar rule name.
3. Inspect each cyclic component's actual timing meanings. Distinguish a genuine impossible before-cycle
   from source wording about independent phases or simultaneous siblings that was joined incorrectly.
4. If a cycle is supported and no source priority resolves it, retain that cyclicGroup and a precise
   unresolved finding. Do not choose one edge to discard merely because its paragraph is shorter.
5. For the acyclic remainder, produce orderedRuleIds consistent with all established edges and keep
   independentGroups explicit. Sibling effects declared simultaneous are not executed sequentially just
   because this representation gives their rule IDs a stable display order.
6. Return the actual order, independence and cycle findings. The caller uses them to constrain event
   windows; they are not a precomputed ordering of future eligible triggers that have not been judged.
