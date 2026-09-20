---
description: Semantically reconcile nodes and labelled edges.
args:
  base: State
  updates: Update[]
  policy: Text
returns: Draft
---
Interpret whether two labels refer to one entity, whether an edge addition changes an
existing relation, and whether removal of a node should also remove or preserve related
claims. Keep all edge endpoints tied to returned node IDs. Do not collapse two entities
because their labels happen to be similar. Put unresolved identity or contradictory edge
claims in alternatives and account for every update ID.
